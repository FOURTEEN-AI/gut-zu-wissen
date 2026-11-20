"""Article of the day and the daily run (phase 3; design: "Betrieb", "Fehlerbehandlung";
spike S8, docs/spikes/s8-artikel-des-tages.md).

Source: the German Wikipedia lists every day's featured article ("Artikel des Tages") on
`Wikipedia:Hauptseite/Artikel des Tages/Chronologie <year>`, one line per day,
`* DD.MM.YYYY [[Title]]`, written by a bot at 23:00 UTC for the next day. The weekday pages
(`.../Montag` and so on) carry the same title as `LEMMA` with a `DATUM`; they are the fallback.

Substitutes come from the members of `Kategorie:Wikipedia:Exzellent` and
`Kategorie:Wikipedia:Lesenswert`, in a random order seeded with the date. A candidate is checked
against the living-person gate and the repeat gate (D4) before it is chosen; a rejected candidate
costs nothing and does not count towards the limit. After a gate failure, an exhausted budget or an
unknown article the day moves on to the next substitute, at most `substitute_articles_max` times;
then the episode is dropped. A Jev abort drops the episode at once (no substitute, error table).

The daily decisions are pure functions or take their collaborators as arguments; tests use fakes.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from gut_zu_wissen.fetch import ArticleNotFoundError, WikipediaApi
from gut_zu_wissen.gates import is_living_person, repeat_gate_allows
from gut_zu_wissen.orchestrator import GateFailure, JevAbortFailure

CHRONOLOGY_PAGE = "Wikipedia:Hauptseite/Artikel des Tages/Chronologie {year}"
WEEKDAY_PAGE = "Wikipedia:Hauptseite/Artikel des Tages/{weekday}"
WEEKDAYS = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")
FEATURED_CATEGORIES = ("Kategorie:Wikipedia:Exzellent", "Kategorie:Wikipedia:Lesenswert")

_LINK = re.compile(r"\[\[([^\]|#]+)(?:\|[^\]]*)?\]\]")


class ArticleOfTheDayError(Exception):
    """No article of the day could be found for the date."""


# ---------------------------------------------------------------- pure parsing


def parse_chronology(wikitext: str, day: date) -> str | None:
    """Title of the article of `day` in the chronology page, or None."""
    prefix = f"* {day.day:02d}.{day.month:02d}.{day.year} "
    for line in wikitext.splitlines():
        if line.startswith(prefix):
            match = _LINK.search(line[len(prefix) :])
            if match:
                return match.group(1).strip()
    return None


def parse_weekday_page(wikitext: str, day: date) -> str | None:
    """`LEMMA` of a weekday page whose `DATUM` is `day`, or None."""
    datum = re.search(r"^\|DATUM\s*=\s*(\S+)", wikitext, re.MULTILINE)
    lemma = re.search(r"^\|LEMMA\s*=\s*(.+?)\s*$", wikitext, re.MULTILINE)
    if not datum or not lemma or datum.group(1) != f"{day.day:02d}.{day.month:02d}.{day.year}":
        return None
    return lemma.group(1).strip() or None


def candidate_order(titles: Sequence[str], excluded: set[str], seed: str) -> list[str]:
    """The titles without `excluded`, in a random order that is fixed by `seed`."""
    pool = sorted(set(titles) - excluded)
    random.Random(seed).shuffle(pool)
    return pool


# ---------------------------------------------------------------- Wikipedia access


def _page_text(api: WikipediaApi, title: str) -> str | None:
    data = api.query(
        action="query",
        titles=title,
        prop="revisions",
        rvprop="content",
        rvslots="main",
        formatversion="2",
        format="json",
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    return pages[0]["revisions"][0]["slots"]["main"]["content"]


def article_of_the_day(api: WikipediaApi, day: date) -> str:
    """The article of the day, from the chronology and then from the weekday page."""
    text = _page_text(api, CHRONOLOGY_PAGE.format(year=day.year))
    title = parse_chronology(text, day) if text else None
    if title:
        return title
    text = _page_text(api, WEEKDAY_PAGE.format(weekday=WEEKDAYS[day.weekday()]))
    title = parse_weekday_page(text, day) if text else None
    if title:
        return title
    raise ArticleOfTheDayError(f"No article of the day found for {day.isoformat()}.")


def featured_titles(api: WikipediaApi) -> list[str]:
    """Main-namespace members of the featured-article categories (about 7,500 titles)."""
    titles: list[str] = []
    for category in FEATURED_CATEGORIES:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmnamespace": "0",
            "cmlimit": "500",
            "format": "json",
        }
        while True:
            data = api.query(**params)
            titles += [m["title"] for m in data.get("query", {}).get("categorymembers", [])]
            more = data.get("continue", {}).get("cmcontinue")
            if not more:
                break
            params["cmcontinue"] = more
    return titles


def categories_of(api: WikipediaApi, title: str) -> list[str] | None:
    """Categories of an article, None if it does not exist."""
    data = api.query(
        action="query",
        titles=title,
        prop="categories",
        cllimit="500",
        redirects="1",
        formatversion="2",
        format="json",
    )
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return None
    return [c["title"] for c in pages[0].get("categories", [])]


# ---------------------------------------------------------------- the day


@dataclass
class DayResult:
    """What the daily run did; `episode_id` is set if an episode was produced."""

    episode_id: str | None = None
    tried: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dropped_reason: str | None = None


def candidate_rejection(
    categories: list[str] | None,
    last_published: datetime | None,
    now: datetime,
    repeat_months: int,
) -> str | None:
    """Why a substitute candidate is not allowed (living person, D4), or None."""
    if categories is None:
        return "does not exist"
    if is_living_person(categories):
        return "living person"
    if not repeat_gate_allows(last_published, now, repeat_months):
        return "episode published less than the repeat period ago"
    return None


def run_day(
    title: str,
    substitutes: Iterator[str],
    max_substitutes: int,
    run_episode: Callable[[str], str],
    check_candidate: Callable[[str], str | None],
) -> DayResult:
    """Run the article of the day; on failure the substitutes in turn.

    `run_episode(title)` returns the episode ID or raises `GateFailure` (or `JevAbortFailure`),
    or `ArticleNotFoundError`. `check_candidate(title)` returns the reason a
    substitute is not allowed, or None.
    """
    result = DayResult()
    current = title
    used = 0
    while True:
        result.tried.append(current)
        try:
            result.episode_id = run_episode(current)
            return result
        except JevAbortFailure as exc:
            result.dropped_reason = f"Jev abort with '{current}', no substitute: {exc}"
            return result
        except (GateFailure, ArticleNotFoundError) as exc:
            result.notes.append(f"'{current}' failed: {exc}")
        if used >= max_substitutes:
            result.dropped_reason = (
                f"'{title}' and {max_substitutes} substitute(s) failed; the episode is dropped."
            )
            return result
        current = _next_allowed(substitutes, check_candidate, result)
        if current is None:
            result.dropped_reason = "No allowed substitute article left; the episode is dropped."
            return result
        used += 1


def _next_allowed(
    substitutes: Iterator[str], check: Callable[[str], str | None], result: DayResult
) -> str | None:
    for candidate in substitutes:
        reason = check(candidate)
        if reason is None:
            return candidate
        result.notes.append(f"Substitute '{candidate}' rejected: {reason}")
    return None
