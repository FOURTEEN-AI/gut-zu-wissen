"""Delayed automatic publication (phase 3, user decision 2026-11-20).

A completed episode is published at a fixed time of the day (the schedule of `auto-publish.yml`,
10:00 UTC) if its run ended at least `min_age_hours` before; the owner can intervene until then. It
is published without further action, unless something stops it:

- the review issue carries the label `hold_label` (the owner's brake; remove the label to release),
- the review found an error (`outcome == "error"`): a person decides then, with `publish.yml`,
- the audio master cannot be found (`audio_run_id` missing: episodes from before this feature).

`python -m gut_zu_wissen.autopublish` prints the verdict per completed episode and writes the first
due one (oldest first) as the step outputs `episode_id` and `run_id` for the workflow, which then
runs the same steps as `publish.yml`. One episode per run; with a backlog the next one follows on
the next day (or by a manual run).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from gut_zu_wissen.archive import (
    EPISODE_FILE,
    EPISODES_DIR,
    EpisodeRecord,
    parse_timestamp,
    read_json,
)

REVIEW_FILE = "09_review.json"


class LabelReader(Protocol):
    def labels(self, number: int) -> list[str]: ...


@dataclass
class Verdict:
    episode_id: str
    due: bool
    reason: str
    run_id: str | None = None


def verdict(
    record: EpisodeRecord,
    review: Mapping | None,
    labels: list[str],
    now: datetime,
    min_age_hours: float,
    hold_label: str,
) -> Verdict:
    """Whether the episode is due for automatic publication, and why (not) (pure)."""
    eid = record.episode_id
    if record.status != "completed":
        return Verdict(eid, False, f"status is '{record.status}'")
    if not record.audio_run_id:
        return Verdict(eid, False, "no audio run recorded; publish by hand")
    if record.explicit is None or not record.finished_at:
        return Verdict(eid, False, "explicit check or end of run missing")
    if hold_label in labels:
        return Verdict(eid, False, f"held (label '{hold_label}' on the review issue)")
    if review and review.get("outcome") == "error":
        return Verdict(eid, False, "the review found an error; publish by hand")
    ready = parse_timestamp(record.finished_at) + timedelta(hours=min_age_hours)
    if now < ready:
        return Verdict(eid, False, f"too recent: ready from {ready.isoformat()}")
    return Verdict(eid, True, "due", record.audio_run_id)


def find_due(
    archive_root: Path,
    now: datetime,
    min_age_hours: float,
    hold_label: str,
    issues: LabelReader | None,
    log: Callable[[str], None] = print,
) -> Verdict | None:
    """The oldest due episode, or None; every completed episode is logged with its verdict."""
    candidates: list[tuple[str, EpisodeRecord]] = []
    for path in sorted((archive_root / EPISODES_DIR).glob(f"*/{EPISODE_FILE}")):
        record = EpisodeRecord(**read_json(path))
        if record.status == "completed":
            candidates.append((record.finished_at or "", record))
    for _, record in sorted(candidates, key=lambda c: c[0]):
        folder = archive_root / EPISODES_DIR / record.episode_id
        review = read_json(folder / REVIEW_FILE) if (folder / REVIEW_FILE).exists() else None
        number = review.get("issue") if review else None
        labels = issues.labels(number) if issues and isinstance(number, int) else []
        result = verdict(record, review, labels, now, min_age_hours, hold_label)
        log(f"{result.episode_id}: {'DUE' if result.due else 'not due'} ({result.reason})")
        if result.due:
            return result
    return None


def main() -> int:
    from gut_zu_wissen.github import HttpIssues
    from gut_zu_wissen.run import ARCHIVE_ROOT, load_config, write_step_output

    config = load_config()["publishing"]
    repository, token = os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")
    issues = HttpIssues(repository, token) if repository and token else None
    due = find_due(
        ARCHIVE_ROOT, datetime.now(UTC), config["min_age_hours"], config["hold_label"], issues
    )
    if due:
        write_step_output("episode_id", due.episode_id)
        write_step_output("run_id", due.run_id or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
