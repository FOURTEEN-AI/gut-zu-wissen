import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile

from gut_zu_wissen.mp3 import Mp3Error, encode_mp3, ffmpeg_command


def test_command_sets_bitrate_and_tags_and_drops_source_metadata():
    command = ffmpeg_command(Path("a.flac"), Path("b.mp3"), 128, {"title": "T", "artist": "A"})
    assert command[command.index("-b:a") + 1] == "128k"
    assert command[command.index("-map_metadata") + 1] == "-1"
    assert command.count("-metadata") == 2
    assert "title=T" in command
    assert command[-1] == "b.mp3"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_produces_an_mp3_of_the_same_length(tmp_path):
    rate = 24_000
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate)
    source = tmp_path / "in.flac"
    soundfile.write(source, tone, rate, format="FLAC")
    target = tmp_path / "out" / "episode.mp3"

    encode_mp3(source, target, 128, {"title": "Titel mit Ümläut"})

    assert target.read_bytes()[:3] == b"ID3"
    info = soundfile.info(target)
    assert info.format == "MP3"
    assert abs(info.duration - 2.0) < 0.1


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_encode_failure_is_reported(tmp_path):
    with pytest.raises(Mp3Error):
        encode_mp3(tmp_path / "missing.flac", tmp_path / "x.mp3", 128, {})
