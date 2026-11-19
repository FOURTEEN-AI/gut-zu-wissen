from types import SimpleNamespace

import numpy as np
import pytest

from gut_zu_wissen.run import load_config
from gut_zu_wissen.stt import (
    WHISPER_RATE_HZ,
    SttError,
    WhisperStt,
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
        return (SimpleNamespace(text=t) for t in self.texts), None


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
    assert stt.transcribe(np.zeros(24_000), 24_000) == "Eins. Zwei."
    stt.transcribe(np.zeros(24_000), 24_000)
    assert loads == [("large-v3", "int8", 4)]
    samples, kwargs = model.calls[0]
    assert len(samples) == WHISPER_RATE_HZ
    assert kwargs == {
        "language": "de",
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }


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
