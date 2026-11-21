"""Speech-to-text behind an interface (phase 1, step 10; design: "Verifikation"; spike S5).

`Stt.transcribe` turns synthesised audio (mono samples and their rate) into a `Transcript`: the
text and its words with start and end time in seconds. The times locate a deviation in the audio
(second hearing, resynthesis of single turns); a transcript without words only disables those.
The implementation is Whisper run locally with faster-whisper: model `large-v3`, `int8`, 4 CPU
threads, German, beam size 5, VAD filter on, no conditioning on previous text (the settings
measured in spike S5; all from the config), word timestamps on. The model is loaded on the first
call and kept for the run; it is downloaded from Hugging Face into `~/.cache/huggingface/hub`,
which the episode workflow caches.

Whisper reads arrays as 16 kHz mono, so the samples are resampled first (pure function). Any
failure to load the model or to transcribe raises `SttError`: the run fails and escalates. Tests
use fakes; nothing here downloads a model unless `transcribe` is called on a real `WhisperStt`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

# Whisper's input format: a property of the model, not a setting.
WHISPER_RATE_HZ = 16_000


class SttError(Exception):
    """The model could not be loaded or the audio could not be transcribed."""


@dataclass(frozen=True)
class Word:
    """One word as Whisper wrote it, with its time in the audio (seconds)."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class Transcript:
    text: str
    words: list[Word] = field(default_factory=list)


class Stt(Protocol):
    def transcribe(self, y: np.ndarray, sr: int) -> Transcript:
        """Text spoken in the mono samples `y` at `sr` Hz, with word times."""
        ...


def to_whisper_rate(y: np.ndarray, sr: int) -> np.ndarray:
    """Mono float32 samples at 16 kHz (polyphase resampling, exact rational ratio)."""
    if sr <= 0:
        raise SttError(f"Invalid sample rate {sr}.")
    if sr != WHISPER_RATE_HZ:
        g = np.gcd(sr, WHISPER_RATE_HZ)
        y = resample_poly(y, WHISPER_RATE_HZ // g, sr // g)
    return np.asarray(y, dtype=np.float32)


def join_segments(texts: list[str]) -> str:
    """Whisper segments to one text: each segment stripped, joined by single spaces."""
    return " ".join(t.strip() for t in texts if t.strip())


def _faster_whisper_model(name: str, compute_type: str, cpu_threads: int) -> Any:
    from faster_whisper import WhisperModel  # heavy import, only when a real model is needed

    return WhisperModel(name, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)


class WhisperStt:
    """`Stt` with faster-whisper on the CPU (spike S5)."""

    def __init__(
        self,
        model: str,
        settings: Mapping,
        load: Callable[[str, str, int], Any] = _faster_whisper_model,
    ) -> None:
        self._name = model
        self._settings = dict(settings)
        self._load = load
        self._model: Any = None

    def transcribe(self, y: np.ndarray, sr: int) -> Transcript:
        s = self._settings
        if self._model is None:
            log.info("STT: loading model %s", self._name)
            start = time.monotonic()
            try:
                self._model = self._load(self._name, s["compute_type"], s["cpu_threads"])
            except Exception as exc:  # download, disk, unknown model name
                raise SttError(f"Whisper model '{self._name}' could not be loaded: {exc}") from exc
            log.info("STT: model loaded in %.1f s", time.monotonic() - start)
        samples = to_whisper_rate(y, sr)
        audio_seconds = len(samples) / WHISPER_RATE_HZ
        log.info("STT: transcribing %.0f s of audio", audio_seconds)
        start = time.monotonic()
        try:
            segments, _ = self._model.transcribe(
                samples,
                language=s["language"],
                beam_size=s["beam_size"],
                vad_filter=s["vad_filter"],
                condition_on_previous_text=s["condition_on_previous_text"],
                word_timestamps=True,
            )
            segments = list(segments)  # the generator does the work
            words = [
                Word(w.word.strip(), float(w.start), float(w.end))
                for seg in segments
                for w in (getattr(seg, "words", None) or [])
                if w.word.strip()
            ]
            log.info(
                "STT: transcribed %.0f s of audio in %.1f s (%d words)",
                audio_seconds,
                time.monotonic() - start,
                len(words),
            )
            return Transcript(join_segments([seg.text for seg in segments]), words)
        except Exception as exc:
            raise SttError(f"Whisper transcription failed: {exc}") from exc
