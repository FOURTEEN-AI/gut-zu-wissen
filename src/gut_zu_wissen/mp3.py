"""MP3 for the feed (phase 2): the pipeline's FLAC master encoded with ffmpeg (libmp3lame).

User decision 2026-11-20: the feed and the Internet Archive item carry an MP3 encoded here, not
a derivative the Archive makes later, so the enclosure URL, size and format are known at upload.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path


class Mp3Error(Exception):
    """ffmpeg is missing or failed."""


def ffmpeg_command(
    source: Path, target: Path, bitrate_kbps: int, tags: Mapping[str, str]
) -> list[str]:
    command = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(source), "-vn"]
    command += ["-map_metadata", "-1", "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k"]
    command += ["-id3v2_version", "3"]
    for key, value in tags.items():
        command += ["-metadata", f"{key}={value}"]
    return [*command, str(target)]


def encode_mp3(source: Path, target: Path, bitrate_kbps: int, tags: Mapping[str, str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            ffmpeg_command(source, target, bitrate_kbps, tags),
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as error:
        raise Mp3Error("ffmpeg is not installed") from error
    if result.returncode != 0:
        raise Mp3Error(f"ffmpeg failed: {result.stderr.strip()[:500]}")
