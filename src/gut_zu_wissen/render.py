"""Rendering wikitext to HTML through the Wikipedia `action=parse` API (spike S3).

Many snippets go into one call, separated by marker paragraphs, so the API load stays small.
The API sits behind the `HtmlRenderer` interface; tests use fakes.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Protocol

from bs4 import BeautifulSoup
from bs4.element import Tag

from gut_zu_wissen.fetch import WikipediaApi

BOUNDARY = "SEGMENTGRENZE{:05d}"
BOUNDARY_PARAGRAPH = re.compile(r"<p>\s*SEGMENTGRENZE(\d{5})\s*</p>")
BATCH_CHARS = 60_000
# Rendered elements that are not reading text.
DROP = (
    "sup.reference, .mw-editsection, style, script, .mw-ref, .reference, "
    ".noprint, .metadata, .error, .mw-empty-elt"
)
INVISIBLE_CHARS = "\u00ad\u200b\u200e"  # soft hyphen, zero-width space, left-to-right mark


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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for ch in INVISIBLE_CHARS:
        text = text.replace(ch, "")
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def plain_text(soup: BeautifulSoup | Tag) -> str:
    """Reading text of rendered HTML: no references, edit links, styles; `<br>` is a space."""
    for el in soup.select(DROP):
        el.decompose()
    for br in soup.select("br"):
        br.replace_with(" ")
    return normalize_text(soup.get_text(""))


def html_to_plain(html: str) -> str:
    return plain_text(BeautifulSoup(html, "html.parser"))
