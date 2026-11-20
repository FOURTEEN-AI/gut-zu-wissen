"""Publish a completed episode: `python -m gut_zu_wissen.publish <episode_id>` (phase 2).

User decision 2026-11-20: publishing is a manual step after the review, not part of the run. It
takes the episode's FLAC master from `build/audio/<episode>/` (the run's workflow artifact),
encodes the MP3, renders the cover from the article title, uploads both to one Internet Archive
item and waits until the item lists them with the right sizes. Only then the episode becomes
`published` (this also arms the D4 repeat gate) and `10_publication.json` records what the feed
needs: URLs, sizes, duration, MIME types. Only a `completed` episode can be published; a rerun
after a failed upload is safe (files are overwritten, the item is created only once).
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from gut_zu_wissen.archive import (
    AUDIO_FILE,
    PUBLICATION_FILE,
    EpisodeRecord,
    episode_dir,
    format_timestamp,
    load_episode,
    read_json,
    save_episode,
    write_json,
)
from gut_zu_wissen.cover import render_png
from gut_zu_wissen.internet_archive import (
    ArchiveHost,
    HostingError,
    HttpArchiveHost,
    download_url,
    item_identifier,
    item_metadata,
    wait_until_ready,
)
from gut_zu_wissen.mp3 import Mp3Error, encode_mp3

MASTER_FILE = "episode.flac"
AUDIO_MIME = "audio/mpeg"
COVER_MIME = "image/png"


class PublishRefused(Exception):
    """The episode may not be published (state, missing audio)."""


@dataclass(frozen=True)
class PublishedFile:
    name: str
    url: str
    bytes: int
    mime: str


@dataclass(frozen=True)
class Publication:
    """Content of 10_publication.json; field order is the file's field order."""

    identifier: str
    item_url: str
    published_at: str
    seconds: float
    mp3_bitrate_kbps: int
    audio: PublishedFile
    cover: PublishedFile


def publish_refusal(record: EpisodeRecord) -> str | None:
    """Why the episode cannot be published, or None."""
    if record.status == "published":
        return "already published"
    if record.status != "completed":
        return f"status is '{record.status}', only a completed episode can be published"
    if record.explicit is None:
        return "explicit flag missing (step 11 did not run)"
    return None


def publish(
    host: ArchiveHost,
    archive_root: Path,
    episode_id: str,
    audio_root: Path,
    work_dir: Path,
    config: Mapping,
    now: datetime,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Publication:
    record = load_episode(archive_root, episode_id)
    refusal = publish_refusal(record)
    if refusal:
        raise PublishRefused(f"{episode_id}: {refusal}")
    master = audio_root / episode_id / MASTER_FILE
    if not master.is_file():
        raise PublishRefused(f"{episode_id}: audio master {master} not found")
    seconds = read_json(episode_dir(archive_root, episode_id) / AUDIO_FILE)["seconds"]

    work_dir.mkdir(parents=True, exist_ok=True)
    audio_name, cover_name = config["audio_file"], config["cover_file"]
    audio_path, cover_path = work_dir / audio_name, work_dir / cover_name
    tags = {
        "title": f"Gut zu wissen: {record.title}",
        "artist": config["creator"],
        "album": "Gut zu wissen",
        "date": f"{now:%Y}",
    }
    encode_mp3(master, audio_path, config["mp3_bitrate_kbps"], tags)
    cover_path.write_bytes(render_png(record.title, config["cover_size_px"]))

    identifier = item_identifier(
        episode_id, config["identifier_prefix"], config["identifier_max_chars"]
    )
    # Item metadata goes with the first PUT of a new item only; a rerun finds the item.
    metadata = (
        None
        if host.item_exists(identifier)
        else item_metadata(record.title, f"{now:%Y-%m-%d}", config)
    )
    host.put_file(identifier, audio_name, audio_path, AUDIO_MIME, metadata)
    host.put_file(identifier, cover_name, cover_path, COVER_MIME, None)

    sizes = {audio_name: audio_path.stat().st_size, cover_name: cover_path.stat().st_size}
    wait_until_ready(
        host,
        identifier,
        sizes,
        config["ready_timeout_seconds"],
        config["ready_poll_seconds"],
        sleep,
        clock,
    )

    base = config["download_base"]
    published_at = format_timestamp(now)
    publication = Publication(
        identifier=identifier,
        item_url=f"{config['details_base'].rstrip('/')}/{identifier}",
        published_at=published_at,
        seconds=seconds,
        mp3_bitrate_kbps=config["mp3_bitrate_kbps"],
        audio=PublishedFile(
            audio_name, download_url(base, identifier, audio_name), sizes[audio_name], AUDIO_MIME
        ),
        cover=PublishedFile(
            cover_name, download_url(base, identifier, cover_name), sizes[cover_name], COVER_MIME
        ),
    )
    write_json(episode_dir(archive_root, episode_id) / PUBLICATION_FILE, asdict(publication))
    record.status = "published"
    record.published_at = published_at
    save_episode(archive_root, record)
    return publication


def main(argv: list[str] | None = None) -> int:
    from gut_zu_wissen.run import ARCHIVE_ROOT, AUDIO_ROOT, REPO_ROOT, load_config

    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not args[0].strip():
        print("Usage: python -m gut_zu_wissen.publish <episode_id>", file=sys.stderr)
        return 2
    access, secret = (
        os.environ.get("ARCHIVE_S3_ACCESS_KEY", ""),
        os.environ.get("ARCHIVE_S3_SECRET_KEY", ""),
    )
    if not (access and secret):
        print(
            "::error::ARCHIVE_S3_ACCESS_KEY and ARCHIVE_S3_SECRET_KEY must be set.", file=sys.stderr
        )
        return 2
    config = load_config()["hosting"]
    host = HttpArchiveHost(access, secret, config)
    try:
        result = publish(
            host,
            ARCHIVE_ROOT,
            args[0],
            AUDIO_ROOT,
            REPO_ROOT / "build" / "publish" / args[0],
            config,
            datetime.now(UTC),
        )
    except (PublishRefused, HostingError, Mp3Error) as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    print(f"Published {args[0]}: {result.item_url}")
    print(f"Audio: {result.audio.url} ({result.audio.bytes} bytes, {result.seconds:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
