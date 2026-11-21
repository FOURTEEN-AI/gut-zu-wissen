from types import SimpleNamespace

import numpy as np
import pytest

from gut_zu_wissen.run import load_config
from gut_zu_wissen.stt import (
    WHISPER_RATE_HZ,
    SttError,
    Transcript,
    WhisperStt,
    Word,
    join_segments,
    to_whisper_rate,
)

REAL = load_config()


class FakeModel:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[tuple[np.ndarray, dict]] = []

    def transcribe(self, samples, **kwargs):
        self.calls.append((samples, kwargs))
        segments = []
        for n, text in enumerate(self.texts):
            words = [
                SimpleNamespace(word=f" {w}", start=n + 0.1, end=n + 0.5) for w in text.split()
            ]
            segments.append(SimpleNamespace(text=text, words=words))
        return (s for s in segments), None


def test_resampling_to_16_khz_keeps_duration_and_frequency():
    sr = 24_000
    y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)
    out = to_whisper_rate(y, sr)
    assert out.dtype == np.float32 and len(out) == WHISPER_RATE_HZ
    spectrum = np.abs(np.fft.rfft(out))
    assert np.argmax(spectrum) == pytest.approx(440, abs=1)  # 1 s of audio: bin = Hz


def test_16_khz_input_is_only_converted_to_float32():
    y = np.linspace(-1, 1, 100)
    assert np.array_equal(to_whisper_rate(y, WHISPER_RATE_HZ), y.astype(np.float32))


def test_invalid_rate_is_an_error():
    with pytest.raises(SttError):
        to_whisper_rate(np.zeros(10), 0)


def test_segments_are_joined_with_single_spaces():
    assert join_segments([" Hallo und", " willkommen! ", "  "]) == "Hallo und willkommen!"


def test_whisper_uses_the_spike_settings_and_loads_once():
    model, loads = FakeModel([" Eins.", " Zwei."]), []

    def load(name, compute_type, threads):
        loads.append((name, compute_type, threads))
        return model

    stt = WhisperStt(REAL["models"]["stt"], REAL["stt"], load)
    assert stt.transcribe(np.zeros(24_000), 24_000) == Transcript(
        "Eins. Zwei.", [Word("Eins.", 0.1, 0.5), Word("Zwei.", 1.1, 1.5)]
    )
    stt.transcribe(np.zeros(24_000), 24_000)
    assert loads == [("large-v3", "int8", 4)]
    samples, kwargs = model.calls[0]
    assert len(samples) == WHISPER_RATE_HZ
    assert kwargs == {
        "language": "de",
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "word_timestamps": True,
        "hotwords": None,
    }


def test_a_hint_goes_to_whisper_as_hotwords():
    model = FakeModel([" Rainald."])
    stt = WhisperStt(REAL["models"]["stt"], REAL["stt"], lambda *_: model)
    stt.transcribe(np.zeros(16_000), 16_000, hint="Rainald")
    assert model.calls[0][1]["hotwords"] == "Rainald"


def test_load_failure_is_an_stt_error():
    def load(*_):
        raise OSError("no network")

    with pytest.raises(SttError, match="could not be loaded: no network"):
        WhisperStt("large-v3", REAL["stt"], load).transcribe(np.zeros(10), 16_000)


def test_transcription_failure_is_an_stt_error():
    class Broken:
        def transcribe(self, samples, **kwargs):
            raise RuntimeError("bad input")

    with pytest.raises(SttError, match="bad input"):
        WhisperStt("large-v3", REAL["stt"], lambda *_: Broken()).transcribe(np.zeros(10), 16_000)


def test_transcription_is_logged_and_the_model_load_only_once(caplog):
    stt = WhisperStt(REAL["models"]["stt"], REAL["stt"], lambda *_: FakeModel([" Eins."]))
    with caplog.at_level("INFO", logger="gut_zu_wissen.stt"):
        stt.transcribe(np.zeros(24_000), 24_000)
        stt.transcribe(np.zeros(24_000), 24_000)
    messages = [r.getMessage() for r in caplog.records]
    assert sum(m.startswith("STT: loading model") for m in messages) == 1
    assert sum(m.startswith("STT: transcribing 1 s") for m in messages) == 2
    assert sum("transcribed 1 s of audio in" in m and "(1 words)" in m for m in messages) == 2
