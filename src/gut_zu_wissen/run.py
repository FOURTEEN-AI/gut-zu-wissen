"""Run entry point: `python -m gut_zu_wissen.run "<article title>"`.

Phase 1 step 2: fetch the article, apply the gates, write the episode folder. Later steps are
added here and move into the orchestrator (step 8).
"""

from __future__ import annotations

import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from gut_zu_wissen.archive import (
    ARTICLE_FILE,
    create_episode,
    episode_dir,
    format_timestamp,
    last_published,
    save_episode,
    write_json,
)
from gut_zu_wissen.fetch import ArticleNotFoundError, HttpWikipediaApi, WikipediaApi, fetch_article
from gut_zu_wissen.gates import is_living_person, repeat_gate_allows

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "pipeline.toml"
ARCHIVE_ROOT = REPO_ROOT / "archive"


class GateFailure(Exception):
    """A gate stopped the run; the message is the notice for the owner."""


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def run(api: WikipediaApi, archive_root: Path, config: dict, title: str, now: datetime) -> str:
    """Run the steps implemented so far. Returns the episode ID; raises GateFailure."""
    article = fetch_article(api, title, now)
    record = create_episode(archive_root, article.title, now)
    write_json(episode_dir(archive_root, record.episode_id) / ARTICLE_FILE, article.to_dict())

    failure: str | None = None
    if is_living_person(article.categories):
        failure = (
            f"Gate living person: '{article.title}' is a person article without death category."
        )
    else:
        months = config["quotas"]["repeat_gate_months"]
        last = last_published(archive_root, article.title)
        if not repeat_gate_allows(last, now, months):
            assert last is not None
            failure = (
                f"Gate repeat (D4): '{article.title}' was published on "
                f"{format_timestamp(last)}, less than {months} months ago."
            )
    if failure:
        record.status, record.failure = "failed", failure
        save_episode(archive_root, record)
        raise GateFailure(failure)
    return record.episode_id


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or not args[0].strip():
        print('Usage: python -m gut_zu_wissen.run "<article title>"', file=sys.stderr)
        return 2
    try:
        episode_id = run(
            HttpWikipediaApi(), ARCHIVE_ROOT, load_config(), args[0].strip(), datetime.now(UTC)
        )
    except ArticleNotFoundError:
        print(f"Article not found: {args[0]}", file=sys.stderr)
        return 1
    except GateFailure as exc:
        print(f"Aborted: {exc}", file=sys.stderr)
        return 1
    print(f"Episode {episode_id}: steps up to fetch and gates done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
