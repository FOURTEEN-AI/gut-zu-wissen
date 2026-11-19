"""Segmentation of raw wikitext into sentence and infobox-row segments (phase 1, step 3).

Masking, then SoMaJo, then unmasking (spike S2, docs/spikes/s2-segmentation.md). The raw text of
a segment always keeps its markup. Infobox rows come from the marker render (spike S3, D3) and
are resolved through the `HtmlRenderer` interface; see `infobox.py`.

Segment IDs are `{revid}-{index:04d}`: sentence and infobox-row segments are numbered together in
document order from 0.
"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass
from functools import cache
from typing import Literal

from somajo import SoMaJo

from gut_zu_wissen.infobox import INFOBOX, infobox_rows, split_top, template_name
from gut_zu_wissen.render import HtmlRenderer

SegmentType = Literal["sentence", "infobox_row"]

TEMPLATE_SURROGATE = "Vorlage"
# Extension tags whose content is not prose. Block-level when alone on a line.
EXT_TAGS = (
    "math|chem|ce|code|nowiki|syntaxhighlight|source|pre|gallery|score|timeline|references"
    "|poem|imagemap|mapframe|maplink|templatestyles|section|graph|hiero|inputbox|categorytree"
)
# Extension tags that carry content inside a sentence; the surrogate keeps the sentence whole.
INLINE_EXT_TAGS = {"math", "chem", "ce", "code"}
FILE_NS = ("datei:", "bild:", "file:", "image:")
CATEGORY_NS = ("kategorie:", "category:")
EXTLINK = re.compile(r"\[(https?:)?//[^\s\]]+(\s+([^\]]*))?\]")
HEADING = re.compile(r"^(=+)(.+?)\1\s*$")
LIST_MARKER = re.compile(r"^[*#:;]+\s*")
ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
EXCLUDED_SECTION_LEVEL = 2
# Level-2 sections that hold no article prose (bibliography, navigation); subsections included.
EXCLUDED_SECTIONS = {
    "literatur",
    "weblinks",
    "einzelnachweise",
    "anmerkungen",
    "siehe auch",
    "quellen",
    "belege",
    "fußnoten",
}
# SoMaJo reads "°C." as the abbreviation "C." and "3:1." as an ordinal; masked as words.
UNIT = re.compile(r"°[CF]\b")
SCORE = re.compile(r"\d+:\d+(?!\d)")
# Hatnote templates are navigation, not content. A line starting with one is skipped as a whole,
# including text after the template on the same line (S2 decision 5). Extended as cases are found.
HATNOTES = {
    "dieser artikel",
    "hauptartikel",
    "siehe auch",
    "begriffsklärungshinweis",
    "weiterleitungshinweis",
    "andere bedeutung",
    "unterscheide",
}
# Entries of SoMaJo's abbreviation list that are ordinary words in prose (S2 decision 4).
# SoMaJo never ends a sentence after them; we split when the next token is capitalised.
FALSE_ABBREVIATIONS = {"italien.", "japan.", "mag."}
# Masks that are not prose; a line holding only these is a block, not a paragraph.
INLINE_MASK_KINDS = {"link", "extlink", "entity", "unit", "score"}


class SegmentationError(Exception):
    """An invariant of the segmentation was violated (a bug, not a property of the article)."""


@dataclass
class Segment:
    """A segment; field order is the archive field order.

    `plain` is the plain text of an infobox row (D3). Sentences are rendered in step 4, only
    for the facts that are offered, so their `plain` stays empty.
    """

    id: str
    type: SegmentType
    section: str
    raw: str
    plain: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Mask:
    raw_start: int
    raw_end: int
    kind: str  # link, file, category, extlink, template, table, ref, comment, ext, ...
    surrogate: str
    attach: str = "left"  # zero-width masks only: side of a sentence boundary
    m_start: int = 0
    m_end: int = 0

    @property
    def zero_width(self) -> bool:
        return self.surrogate == ""


@cache
def default_tokenizer() -> SoMaJo:
    return SoMaJo("de_CMC", character_offsets=True)


# ---------------------------------------------------------------- scanning


def _balanced(text: str, start: int, open_: str, close: str) -> int:
    """End index (exclusive) of the balanced construct starting at start, or -1."""
    depth, i = 0, start
    while i < len(text):
        if text.startswith(open_, i):
            depth += 1
            i += len(open_)
        elif text.startswith(close, i):
            depth -= 1
            i += len(close)
            if depth == 0:
                return i
        elif text.startswith("<!--", i):
            end = text.find("-->", i)
            i = len(text) if end < 0 else end + 3
        else:
            i += 1
    return -1


def _table_end(text: str, start: int) -> int:
    """Tables: '{|' and '|}' at line start, nested."""
    depth = 0
    for m in re.finditer(r"(?m)^[ \t]*(\{\||\|\})", text[start:]):
        depth += 1 if m.group(1) == "{|" else -1
        if depth == 0:
            return start + m.end()
    return -1


def _link_surrogate(inner: str) -> tuple[str, str]:
    target = inner.split("|", 1)[0].strip().lower()
    if target.startswith(FILE_NS):
        return "file", ""
    if target.startswith(CATEGORY_NS):
        return "category", ""
    label = inner.split("|", 1)[1] if "|" in inner else inner.split("#")[0]
    label = re.sub(r"\[\[|\]\]|\{\{.*?\}\}", "", label).replace("\n", " ").strip()
    return "link", label or "Link"


_EXT_OPEN = re.compile(rf"<({EXT_TAGS})(\s[^>]*)?>", re.IGNORECASE)
_EXT_SELF = re.compile(rf"<({EXT_TAGS})(\s[^>]*)?/>", re.IGNORECASE)
_REF_SELF = re.compile(r"<ref(\s[^>]*)?/>", re.IGNORECASE)
_REF_OPEN = re.compile(r"<ref(\s[^>]*)?>", re.IGNORECASE)
_REF_CLOSE = re.compile(r"</ref\s*>", re.IGNORECASE)
_QUOTES = re.compile(r"'{2,5}")


def scan(text: str) -> list[Mask]:
    """Scan raw wikitext once; every markup construct becomes a mask with a surrogate."""
    masks: list[Mask] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("<!--", i):
            end = text.find("-->", i)
            end = n if end < 0 else end + 3
            masks.append(Mask(i, end, "comment", ""))
            i = end
        elif text.startswith("''", i):
            m = _QUOTES.match(text, i)
            assert m is not None
            masks.append(Mask(i, m.end(), "format", "", attach="right"))
            i = m.end()
        elif c == "&" and (m := ENTITY.match(text, i)):
            masks.append(Mask(i, m.end(), "entity", html.unescape(m.group()).replace("\xa0", " ")))
            i = m.end()
        elif c == "°" and (m := UNIT.match(text, i)):
            masks.append(Mask(i, m.end(), "unit", "Grad"))
            i = m.end()
        elif c.isdigit() and (i == 0 or not text[i - 1].isdigit()) and (m := SCORE.match(text, i)):
            masks.append(Mask(i, m.end(), "score", "Zahl"))
            i = m.end()
        elif c == "<" and (m := _REF_SELF.match(text, i)):
            masks.append(Mask(i, m.end(), "ref", ""))
            i = m.end()
        elif c == "<" and (m := _REF_OPEN.match(text, i)):
            close = _REF_CLOSE.search(text, m.end())
            end = n if close is None else close.end()
            masks.append(Mask(i, end, "ref", ""))
            i = end
        elif c == "<" and (m := _EXT_SELF.match(text, i)):
            masks.append(Mask(i, m.end(), "ext", ""))
            i = m.end()
        elif c == "<" and (m := _EXT_OPEN.match(text, i)):
            tag = m.group(1)
            close = re.compile(rf"</{tag}\s*>", re.IGNORECASE).search(text, m.end())
            end = n if close is None else close.end()
            surrogate = TEMPLATE_SURROGATE if tag.lower() in INLINE_EXT_TAGS else ""
            masks.append(Mask(i, end, "ext", surrogate))
            i = end
        elif text.startswith("{|", i) and (i == 0 or text[i - 1] == "\n"):
            end = _table_end(text, i)
            end = n if end < 0 else end
            masks.append(Mask(i, end, "table", TEMPLATE_SURROGATE))
            i = end
        elif text.startswith("{{", i):
            end = _balanced(text, i, "{{", "}}")
            end = n if end < 0 else end
            masks.append(Mask(i, end, "template", TEMPLATE_SURROGATE))
            i = end
        elif text.startswith("[[", i):
            end = _balanced(text, i, "[[", "]]")
            if end < 0:
                i += 2
                continue
            kind, surrogate = _link_surrogate(text[i + 2 : end - 2])
            masks.append(Mask(i, end, kind, surrogate))
            i = end
        elif c == "[" and (m := EXTLINK.match(text, i)):
            masks.append(Mask(i, m.end(), "extlink", (m.group(3) or "Link").strip()))
            i = m.end()
        else:
            i += 1
    return masks


def strip_refs(raw: str) -> str:
    """Raw text without `<ref>` tags and comments: the text that is rendered to plain text."""
    out, pos = [], 0
    for m in scan(raw):
        if m.kind in ("ref", "comment"):
            out.append(raw[pos : m.raw_start])
            pos = m.raw_end
    out.append(raw[pos:])
    return "".join(out).strip()


def build_masked(text: str, masks: list[Mask]) -> str:
    """Replace every mask by its surrogate and record the masked offsets on the masks."""
    out, pos, m_pos = [], 0, 0
    for mask in masks:
        out.append(text[pos : mask.raw_start])
        m_pos += mask.raw_start - pos
        mask.m_start = m_pos
        out.append(mask.surrogate)
        m_pos += len(mask.surrogate)
        mask.m_end = m_pos
        pos = mask.raw_end
    out.append(text[pos:])
    return "".join(out)


class OffsetMap:
    """Maps a sentence boundary in masked text back to raw text.

    Zero-width masks sitting exactly on a boundary go to the side they attach to:
    refs and comments to the preceding sentence, bold/italic quotes to the following one.
    """

    def __init__(self, masks: list[Mask]) -> None:
        self.masks = masks

    def boundary(self, m: int, unit_end: bool = False) -> int:
        raw = m
        for mask in self.masks:
            if mask.m_start < m < mask.m_end:
                raise SegmentationError("sentence boundary inside a mask")
            before = mask.m_end < m or (mask.m_end == m and mask.m_start < m)
            if mask.zero_width and mask.m_start == m:
                before = unit_end or mask.attach == "left"
            if not before:
                break
            raw += (mask.raw_end - mask.raw_start) - len(mask.surrogate)
        return raw


# ---------------------------------------------------------------- units


@dataclass
class _Unit:
    kind: str  # para | list | block | heading
    start: int  # masked offsets
    end: int


def _units(masked: str, masks: list[Mask], text: str) -> list[_Unit]:
    """Split the masked text into paragraphs, list items, blocks and headings."""
    units: list[_Unit] = []
    para: list[int] | None = None
    pos = 0
    for line in masked.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        line_masks = [m for m in masks if start <= m.m_start and m.m_end <= end]
        marker = LIST_MARKER.match(line)
        rest = line
        for m in sorted(line_masks, key=lambda m: -m.m_start):
            rest = rest[: m.m_start - start] + rest[m.m_end - start :]
        if marker and rest.startswith(marker.group()):
            rest = rest[marker.end() :]
        block_only = rest.strip() == "" and any(m.kind not in INLINE_MASK_KINDS for m in line_masks)
        first = min(line_masks, key=lambda m: m.m_start) if line_masks else None
        if (
            first is not None
            and first.kind == "template"
            and line[: first.m_start - start].strip() == ""
            and template_name(text[first.raw_start : first.raw_end]).lower() in HATNOTES
        ):
            block_only = True
        heading = HEADING.match(line)
        if para and (not line.strip() or heading or block_only or marker):
            units.append(_Unit("para", para[0], para[1]))
            para = None
        if heading:
            units.append(_Unit("heading", start, end))
        elif block_only:
            units.append(_Unit("block", start, end))
        elif not line.strip() or line.startswith((" ", "----", "__")):
            continue
        elif marker:
            units.append(_Unit("list", start + marker.end(), end))
        elif para:
            para[1] = end
        else:
            para = [start, end]
    if para:
        units.append(_Unit("para", para[0], para[1]))
    return units


def _sentence_spans(unit_text: str, offset: int, tokenizer: SoMaJo) -> list[list[int]]:
    """Sentence spans (masked offsets) of one unit, with the false-abbreviation repair."""
    spans: list[list[int]] = []
    for tokens in tokenizer.tokenize_text([unit_text]):
        cut = 0
        for k, tok in enumerate(tokens[:-1]):
            if tok.text.lower() in FALSE_ABBREVIATIONS and tokens[k + 1].text[:1].isupper():
                part = tokens[cut : k + 1]
                spans.append(
                    [offset + part[0].character_offset[0], offset + tok.character_offset[1]]
                )
                cut = k + 1
        rest = tokens[cut:]
        spans.append([offset + rest[0].character_offset[0], offset + rest[-1].character_offset[1]])
    return spans


def _merge_inside_masks(spans: list[list[int]], masks: list[Mask]) -> list[list[int]]:
    """Never split inside a mask (e.g. the link label "J. S. Bach")."""
    merged: list[list[int]] = []
    for s, e in spans:
        if merged and any(mk.m_start < s < mk.m_end for mk in masks):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return merged


# ---------------------------------------------------------------- segmentation


@dataclass
class _Piece:
    """A segment before numbering and before the infobox render."""

    section: str
    raw: str | None = None  # sentence
    infobox: str | None = None  # raw infobox template


def split_pieces(wikitext: str, tokenizer: SoMaJo) -> list[_Piece]:
    """Sentences and infobox templates in document order. Pure; no network."""
    masks = scan(wikitext)
    masked = build_masked(wikitext, masks)
    omap = OffsetMap(masks)
    pieces: list[_Piece] = []
    section = ""
    excluded = False

    for unit in _units(masked, masks, wikitext):
        if unit.kind == "heading":
            h = HEADING.match(masked[unit.start : unit.end])
            assert h is not None
            section = h.group(2).strip()
            if len(h.group(1)) <= EXCLUDED_SECTION_LEVEL:
                excluded = section.lower() in EXCLUDED_SECTIONS
            continue
        if excluded:
            continue
        if unit.kind == "block":
            for mask in masks:
                if (
                    unit.start <= mask.m_start
                    and mask.m_end <= unit.end
                    and mask.kind == "template"
                ):
                    raw_template = wikitext[mask.raw_start : mask.raw_end]
                    if INFOBOX.match(split_top(raw_template[2:-2])[0]):
                        pieces.append(_Piece(section, infobox=raw_template))
            continue
        unit_text = masked[unit.start : unit.end].replace("\n", " ")
        spans = _merge_inside_masks(_sentence_spans(unit_text, unit.start, tokenizer), masks)
        raw_start, raw_end = omap.boundary(unit.start), omap.boundary(unit.end, unit_end=True)
        # Trailing zero-width masks (refs) at the unit end belong to the last sentence.
        bounds = [raw_start] + [omap.boundary(s) for s, _ in spans[1:]] + [raw_end]
        texts = [wikitext[bounds[j] : bounds[j + 1]] for j in range(len(spans))]
        if "".join(texts) != wikitext[raw_start:raw_end]:
            raise SegmentationError(f"round trip mismatch at {raw_start}")
        pieces += [_Piece(section, raw=t.strip()) for t in texts if t.strip()]
    return pieces


def segment_article(
    wikitext: str,
    title: str,
    revid: int,
    renderer: HtmlRenderer,
    tokenizer: SoMaJo | None = None,
) -> list[Segment]:
    """All segments of one revision, numbered in document order."""
    pieces = split_pieces(wikitext, tokenizer or default_tokenizer())
    segments: list[Segment] = []

    def add(type_: SegmentType, section: str, raw: str, plain: str | None = None) -> None:
        segments.append(Segment(f"{revid}-{len(segments):04d}", type_, section, raw, plain))

    for piece in pieces:
        if piece.infobox is not None:
            for row in infobox_rows(renderer, title, piece.infobox):
                add("infobox_row", piece.section, row.raw, row.plain)
        else:
            assert piece.raw is not None
            add("sentence", piece.section, piece.raw)
    return segments
