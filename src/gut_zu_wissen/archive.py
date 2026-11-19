"""Archive layout: JSON files as pipeline state (design: "Archiv").

archive/episodes/<YYYY-MM-DD>_<slug>/episode.json plus one file per pipeline step.
All files are indented with stable field order.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

EPISODES_DIR = "episodes"
EPISODE_FILE = "episode.json"
ARTICLE_FILE = "01_article.json"
SEGMENTS_FILE = "02_segments.json"
FACTS_FILE = "03_facts.json"
RANKING_FILE = "04_ranking.json"
DIALOG_FILE = "05_dialog.json"
LINKED_DIR = "linked"

EpisodeStatus = Literal["running", "published", "failed"]

_TRANSLITERATION = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def slugify(title: str) -> str:
    """Lowercase, umlauts transliterated, everything except a-z and 0-9 becomes '-'."""
    text = title.lower().translate(_TRANSLITERATION)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def format_timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime(TIMESTAMP_FORMAT)


def parse_timestamp(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT).replace(tzinfo=UTC)


@dataclass
class EpisodeRecord:
    """Content of episode.json; field order is the file's field order."""

    episode_id: str
    title: str
    status: EpisodeStatus
    started_at: str
    published_at: str | None = None
    explicit: bool | None = None
    failure: str | None = None


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def episode_dir(root: Path, episode_id: str) -> Path:
    return root / EPISODES_DIR / episode_id


def create_episode(root: Path, title: str, now: datetime) -> EpisodeRecord:
    """Create a running episode folder; a same-day repeat gets a `-2`, `-3` suffix."""
    base = f"{now.astimezone(UTC):%Y-%m-%d}_{slugify(title)}"
    episode_id, n = base, 1
    while episode_dir(root, episode_id).exists():
        n += 1
        episode_id = f"{base}-{n}"
    record = EpisodeRecord(
        episode_id=episode_id, title=title, status="running", started_at=format_timestamp(now)
    )
    save_episode(root, record)
    return record


def save_episode(root: Path, record: EpisodeRecord) -> None:
    write_json(episode_dir(root, record.episode_id) / EPISODE_FILE, asdict(record))


def load_episode(root: Path, episode_id: str) -> EpisodeRecord:
    return EpisodeRecord(**read_json(episode_dir(root, episode_id) / EPISODE_FILE))


def last_published(root: Path, title: str) -> datetime | None:
    """Publication date of the latest published episode for this article (input to the D4 gate).

    Failed and running episodes do not count. `title` is the canonical article title.
    """
    dates = []
    for path in (root / EPISODES_DIR).glob(f"*/{EPISODE_FILE}"):
        record = EpisodeRecord(**read_json(path))
        if record.status == "published" and record.title == title and record.published_at:
            dates.append(parse_timestamp(record.published_at))
    return max(dates, default=None)
