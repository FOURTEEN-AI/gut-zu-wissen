import io

import numpy as np
import pytest
import soundfile as sf

from gut_zu_wissen import audio
from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.episode_audio import (
    AudioStepError,
    cost_usd,
    new_record,
    produce_audio,
)
from gut_zu_wissen.run import load_config
from gut_zu_wissen.tts import Synthesis

SR = 24_000
REAL = load_config()


def config(max_chars: int = 6000) -> dict:
    return {**REAL, "quotas": {**REAL["quotas"], "tts_section_max_chars": max_chars}}


def tone(seconds: float, sr: int = SR) -> np.ndarray:
    y = 0.05 * np.sin(2 * np.pi * 220 * np.arange(round(seconds * sr)) / sr)
    return np.concatenate([np.zeros(sr // 2), y, np.zeros(sr // 2)])


def wav(y: np.ndarray, sr: int = SR) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class FakeTts:
    def __init__(self, sr: int = SR) -> None:
        self.calls: list[list] = []
        self.sr = sr

    def synthesize(self, turns):
        self.calls.append(list(turns))
        return Synthesis(wav(tone(1.0, self.sr), self.sr), 10, 20)


def fixed() -> dict:
    return {"intro": (tone(0.5), SR, "i" * 64), "outro": (tone(0.5), SR, "o" * 64)}


def turns(*texts: str) -> list[Turn]:
    return [Turn(n, "Lena" if n % 2 == 0 else "Felix", t, []) for n, t in enumerate(texts)]


def produce(script, tts, work, cfg=None):
    record = new_record(cfg or config())
    produce_audio(record, script, tts, fixed(), work, cfg or config())
    return record


def test_sections_are_synthesised_joined_and_normalised(tmp_path):
    tts = FakeTts()
    script = turns("A" * 40, "B" * 40, "C" * 40, "D" * 40)
    record = produce(script, tts, tmp_path, config(max_chars=80))
    assert [s.turn_indices for s in record.sections] == [[0, 1], [2, 3]]
    assert tts.calls[1] == [("Lena", "C" * 40), ("Felix", "D" * 40)]
    assert [s.chars for s in record.sections] == [80, 80]
    out, sr = audio.read(tmp_path / "episode.flac")
    keep, pause = REAL["audio"]["edge_silence_keep_seconds"], REAL["audio"]["join_pause_seconds"]
    # intro 0.5 s, two sections of 1 s, outro 0.5 s; edges trimmed to `keep`; three pauses
    assert len(out) / sr == pytest.approx(3 + 4 * 2 * keep + 3 * pause, abs=0.05)
    assert record.seconds == pytest.approx(len(out) / sr, abs=0.01)
    assert abs(audio.integrated_loudness(out, sr) - REAL["quotas"]["target_loudness_lufs"]) <= 0.1
    assert audio.true_peak_dbtp(out, sr) <= REAL["quotas"]["true_peak_ceiling_dbtp"]
    assert (tmp_path / "sections" / "01.flac").exists()
    assert record.fixed_parts == {"intro": "i" * 64, "outro": "o" * 64}
    assert record.usd == pytest.approx(2 * cost_usd(10, 20, REAL["prices"]))


def test_markup_in_the_final_script_stops_before_tts(tmp_path):
    tts = FakeTts()
    with pytest.raises(AudioStepError, match="Markup gate: turn 1"):
        produce(turns("Sauber.", "Mit [[Link]]."), tts, tmp_path)
    assert tts.calls == []


def test_turn_too_long_for_a_section_stops_before_tts(tmp_path):
    tts = FakeTts()
    with pytest.raises(AudioStepError, match="Sections: Turn 0"):
        produce(turns("A" * 81), tts, tmp_path, config(max_chars=80))
    assert tts.calls == []


def test_sample_rate_mismatch_is_an_error(tmp_path):
    with pytest.raises(AudioStepError, match="48000 Hz"):
        produce(turns("Text."), FakeTts(sr=48_000), tmp_path)


def test_cost_is_unknown_with_placeholder_prices():
    assert cost_usd(10, 20, {"gemini_tts_input": "PLACEHOLDER", "gemini_tts_output": 9.0}) is None
    assert cost_usd(None, 20, REAL["prices"]) is None
    assert cost_usd(1_000_000, 1_000_000, REAL["prices"]) == 9.5
