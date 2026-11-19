import io
import json

import numpy as np
import pytest
import soundfile as sf

from gut_zu_wissen import intro_outro
from gut_zu_wissen.intro_outro import (
    INTRO,
    OUTRO,
    FixedAudioError,
    fingerprint,
    load_fixed,
    synthesize_fixed,
)
from gut_zu_wissen.markup import find_markup
from gut_zu_wissen.run import load_config
from gut_zu_wissen.tts import Synthesis, voices_of

MODEL = "gemini-3.8-flash-tts"
VOICES = {"Lena": "de-de-advisor-3", "Felix": "de-de-tutor-4"}


def wav(seconds: float, sr: int = 24_000) -> bytes:
    y = 0.2 * np.sin(2 * np.pi * 200 * np.arange(round(seconds * sr)) / sr)
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class FakeTts:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def synthesize(self, turns):
        self.calls.append(tuple(turns))
        return Synthesis(wav(1.0 + len(self.calls)), 10, 25)


def test_text_is_the_design_text():
    assert INTRO[0] == ("Felix", "Hallo und willkommen bei „Gut zu wissen“!")
    assert [s for s, _ in OUTRO] == ["Lena", "Felix", "Lena"]
    assert "CC BY-SA 4.0" in OUTRO[2][1] and "Wikimedia Foundation" in OUTRO[2][1]
    assert "deutschsprachigen Wikipedia" in OUTRO[0][1]
    assert all(not find_markup(text) for _, text in INTRO + OUTRO)


def test_voices_come_from_the_config():
    assert voices_of(load_config()) == VOICES


def test_synthesis_writes_both_parts_and_metadata(tmp_path):
    tts = FakeTts()
    metadata = synthesize_fixed(tts, tmp_path, MODEL, VOICES)
    assert tts.calls == [INTRO, OUTRO]
    assert [p["name"] for p in metadata["parts"]] == ["intro", "outro"]
    assert json.loads((tmp_path / "fixed_audio.json").read_text()) == metadata
    loaded = load_fixed(tmp_path, MODEL, VOICES)
    y, sr, print_ = loaded["intro"]
    assert sr == 24_000 and len(y) == 2 * sr
    assert print_ == fingerprint(INTRO, MODEL, VOICES)
    assert metadata["parts"][1]["seconds"] == 3.0


def test_missing_audio_names_the_command(tmp_path):
    with pytest.raises(FixedAudioError, match="gut_zu_wissen.intro_outro"):
        load_fixed(tmp_path, MODEL, VOICES)


def test_changed_voice_makes_the_audio_stale(tmp_path):
    synthesize_fixed(FakeTts(), tmp_path, MODEL, VOICES)
    with pytest.raises(FixedAudioError, match="does not match"):
        load_fixed(tmp_path, MODEL, {**VOICES, "Felix": "Puck"})
    with pytest.raises(FixedAudioError, match="does not match"):
        load_fixed(tmp_path, "gemini-3.8-flash-lite-tts", VOICES)


def test_changed_text_makes_the_audio_stale(tmp_path, monkeypatch):
    synthesize_fixed(FakeTts(), tmp_path, MODEL, VOICES)
    monkeypatch.setitem(intro_outro.PARTS, "intro", (("Felix", "Servus!"),))
    with pytest.raises(FixedAudioError, match="intro audio does not match"):
        load_fixed(tmp_path, MODEL, VOICES)


def test_cli_keeps_current_audio_without_api_call(tmp_path, monkeypatch, capsys):
    synthesize_fixed(FakeTts(), tmp_path, MODEL, VOICES)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)  # would fail if it tried to synthesise
    assert intro_outro.main(["--out", str(tmp_path)]) == 0
    assert "nothing to do" in capsys.readouterr().out
