"""Linked intros (phase 1, step 5, design: "Korpus").

For links in the offered main facts: fetch the article (same revision rule as the main article),
apply the living-person gate, take the intro (text before the first `==` heading), segment and
extract it. Each article keeps its own revid for the shownotes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from gut_zu_wissen.extraction import ExtractionResult, extract_facts
from gut_zu_wissen.fetch import ArticleNotFoundError, WikipediaApi, fetch_article
from gut_zu_wissen.gates import is_living_person
from gut_zu_wissen.llm import StructuredLlm
from gut_zu_wissen.render import HtmlRenderer
from gut_zu_wissen.segmentation import Segment, segment_article

FIRST_HEADING = re.compile(r"(?m)^==")


@dataclass
class LinkedIntro:
    """A linked article's intro with its extraction; field order is the archive field order."""

    title: str
    revid: int
    revision_status: str
    revision_timestamp: str
    fetched_at: str
    categories: list[str]
    intro_wikitext: str
    segments: list[Segment] = field(default_factory=list)
    extraction: ExtractionResult | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SkippedLink:
    title: str
    reason: str


def intro_wikitext(wikitext: str) -> str:
    """Text before the first heading of any level-2 or deeper section."""
    match = FIRST_HEADING.search(wikitext)
    return (wikitext[: match.start()] if match else wikitext).strip()


def fetch_linked_intros(
    api: WikipediaApi,
    llm: StructuredLlm,
    renderer: HtmlRenderer,
    titles: list[str],
    main_title: str,
    config: dict,
    now: datetime,
) -> tuple[list[LinkedIntro], list[SkippedLink]]:
    intros: list[LinkedIntro] = []
    skipped: list[SkippedLink] = []
    seen = {main_title}
    for title in titles:
        try:
            article = fetch_article(api, title, now)
        except ArticleNotFoundError:
            skipped.append(SkippedLink(title, "not_found"))
            continue
        if article.title in seen:  # redirect to an article already in the corpus
            skipped.append(SkippedLink(title, "duplicate"))
            continue
        seen.add(article.title)
        if is_living_person(article.categories):
            skipped.append(SkippedLink(article.title, "living_person"))
            continue
        intro = intro_wikitext(article.wikitext)
        segments = segment_article(intro, article.title, article.revid, renderer)
        extraction = extract_facts(llm, renderer, article.title, article.revid, segments, config)
        intros.append(
            LinkedIntro(
                article.title,
                article.revid,
                article.revision_status,
                article.revision_timestamp,
                article.fetched_at,
                article.categories,
                intro,
                segments,
                extraction,
            )
        )
    return intros, skipped
