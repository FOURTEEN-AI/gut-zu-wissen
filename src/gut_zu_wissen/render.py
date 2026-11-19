"""Rendering wikitext to HTML through the Wikipedia `action=parse` API (spike S3).

Many snippets go into one call, separated by marker paragraphs, so the API load stays small.
The API sits behind the `HtmlRenderer` interface; tests use fakes.
"""

from __future__ import annotations

import re
from typing import Protocol

from gut_zu_wissen.fetch import WikipediaApi

BOUNDARY = "SEGMENTGRENZE{:05d}"
BOUNDARY_PARAGRAPH = re.compile(r"<p>\s*SEGMENTGRENZE(\d{5})\s*</p>")
BATCH_CHARS = 60_000


class HtmlRenderer(Protocol):
    def render_html(self, title: str, snippets: list[str]) -> list[str]:
        """Render each wikitext snippet in the context of the article `title`; HTML per snippet."""
        ...


def batches(snippets: list[str], max_chars: int = BATCH_CHARS) -> list[list[str]]:
    """Group snippets in order; a batch is closed once it holds `max_chars` characters."""
    out: list[list[str]] = []
    current: list[str] = []
    size = 0
    for snippet in snippets:
        current.append(snippet)
        size += len(snippet)
        if size >= max_chars:
            out.append(current)
            current, size = [], 0
    if current:
        out.append(current)
    return out


class ParseRenderer:
    """`HtmlRenderer` on `action=parse` (POST, one call per batch)."""

    def __init__(self, api: WikipediaApi, max_chars: int = BATCH_CHARS) -> None:
        self._api = api
        self._max_chars = max_chars

    def render_html(self, title: str, snippets: list[str]) -> list[str]:
        results: list[str] = []
        for batch in batches(snippets, self._max_chars):
            text = (
                "\n\n".join(f"{BOUNDARY.format(i)}\n\n{raw}" for i, raw in enumerate(batch))
                + f"\n\n{BOUNDARY.format(len(batch))}"
            )
            data = self._api.post(
                action="parse",
                title=title,
                text=text,
                contentmodel="wikitext",
                prop="text",
                disableeditsection="1",
                disablelimitreport="1",
            )
            parts = BOUNDARY_PARAGRAPH.split(data["parse"]["text"])
            # parts = [before, idx0, html0, idx1, html1, ..., idxN, after]
            by_index = {int(parts[k]): parts[k + 1] for k in range(1, len(parts) - 1, 2)}
            results += [by_index.get(i, "") for i in range(len(batch))]
        return results
