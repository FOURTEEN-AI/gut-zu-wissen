"""Archive layout: JSON files as pipeline state (design: "Archiv").

archive/episodes/<YYYY-MM-DD>_<slug>/episode.json plus one file per pipeline step, and
archive/state/review.json for the sampling rate r. All files are indented with stable field order.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
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
JUDGEMENT_FILE = "06_judgement.json"
EXPLICIT_FILE = "07_explicit.json"
AUDIO_FILE = "08_audio.json"
REVIEW_FILE = "09_review.json"
PUBLICATION_FILE = "10_publication.json"
LINKED_DIR = "linked"
STATE_DIR = "state"
REVIEW_STATE_FILE = "review.json"  # sampling rate r (design: archive/state/review.json)

# running: the run has not ended; completed: every pipeline step done (phase 2 moves it on to
# published); failed: gate failure or crash.
EpisodeStatus = Literal["running", "completed", "published", "failed"]

# Pipeline steps as recorded in `failed_step`, in run order.
STEP_GATES = "gates"
STEP_SEGMENTATION = "segmentation"
STEP_EXTRACTION = "extraction"
STEP_DIALOG = "dialog"  # ranking, dialog author, judge and orchestrator
STEP_EXPLICIT = "explicit"
STEP_AUDIO = "audio"  # sections, TTS, STT verification, normalisation
STEP_REVIEW = "review"
STEPS = (
    STEP_GATES,
    STEP_SEGMENTATION,
    STEP_EXTRACTION,
    STEP_DIALOG,
    STEP_EXPLICIT,
    STEP_AUDIO,
    STEP_REVIEW,
)

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
    """Content of episode.json; field order is the file's field order.

    `finished_at` is when the run ended, successfully or not (runtime, step 13); `usage` holds the
    token counts per API client role (`usage.Usage`, cost log); `cost` the USD per role and for
    TTS with the prices in force at run time (`usage.episode_cost`). All three are missing in
    files written before they were recorded.

    `failed_step` names the pipeline step a failed run stopped in (`STEPS`; None while the run
    has not failed, and in files written before it was recorded). `resumed` lists every resume
    of the audio step (`resume.py`, user decision 2026-11-20): when it started, the workflow run,
    when the attempt before it ended and why it failed.
    """

    episode_id: str
    title: str
    status: EpisodeStatus
    started_at: str
    finished_at: str | None = None
    published_at: str | None = None
    explicit: bool | None = None
    failure: str | None = None
    usage: dict[str, dict] = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    failed_step: str | None = None
    resumed: list[dict] = field(default_factory=list)


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

    Only published episodes count; running, completed and failed ones do not. `title` is the
    canonical article title.
    """
    dates = []
    for path in (root / EPISODES_DIR).glob(f"*/{EPISODE_FILE}"):
        record = EpisodeRecord(**read_json(path))
        if record.status == "published" and record.title == title and record.published_at:
            dates.append(parse_timestamp(record.published_at))
    return max(dates, default=None)
