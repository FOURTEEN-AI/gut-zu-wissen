"""Shownotes data of a published episode (design: "Lizenz", "Feedback").

Attribution: a revision permalink for the main article and for every linked article whose facts
the final script uses, plus a link to the version history (the list of authors). The articles
come from the facts the script cites: a fact ID starts with the revision ID of its article.
The feedback link is a `mailto:` with the episode title and ID in the subject.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode

from gut_zu_wissen.archive import ARTICLE_FILE, DIALOG_FILE, RANKING_FILE, read_json


@dataclass(frozen=True)
class Source:
    title: str
    revid: int


def revision_of(fact_id: str) -> int:
    """`272085229-0008` -> 272085229"""
    return int(fact_id.split("-", 1)[0])


def titles_by_revision(article: Mapping, ranking: Mapping) -> dict[int, str]:
    titles = {int(article["revid"]): str(article["title"])}
    for entry in ranking.get("linked_articles", []):
        titles[int(entry["revid"])] = str(entry["title"])
    for key in ("main_facts", "linked_facts", "einordnung_facts"):
        for entry in ranking.get(key, []):
            titles.setdefault(int(entry["revid"]), str(entry["article"]))
    return titles


def used_sources(
    main_revid: int, used_fact_ids: Iterable[str], titles: Mapping[int, str]
) -> list[Source]:
    """The main article first, then the linked articles used, sorted by title."""
    revids = {revision_of(fact_id) for fact_id in used_fact_ids} | {main_revid}
    unknown = revids - set(titles)
    if unknown:
        raise ValueError(f"no article known for revision(s) {sorted(unknown)}")
    linked = sorted(
        (Source(titles[r], r) for r in revids if r != main_revid), key=lambda s: s.title
    )
    return [Source(titles[main_revid], main_revid), *linked]


def episode_sources(episode_folder: Path) -> list[Source]:
    article = read_json(episode_folder / ARTICLE_FILE)
    ranking = read_json(episode_folder / RANKING_FILE)
    dialog = read_json(episode_folder / DIALOG_FILE)
    return used_sources(
        int(article["revid"]), dialog["used_fact_ids"], titles_by_revision(article, ranking)
    )


def permalink(wikipedia_index: str, source: Source) -> str:
    query = urlencode({"title": source.title.replace(" ", "_"), "oldid": source.revid}, safe="_")
    return f"{wikipedia_index}?{query}"


def history_url(wikipedia_index: str, source: Source) -> str:
    query = urlencode({"title": source.title.replace(" ", "_"), "action": "history"}, safe="_")
    return f"{wikipedia_index}?{query}"


def feedback_mailto(address: str, title: str, episode_id: str) -> str:
    subject = quote(f"Gut zu wissen: {title} ({episode_id})", safe="")
    return f"mailto:{address}?subject={subject}"
