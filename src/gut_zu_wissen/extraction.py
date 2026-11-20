"""Fact extraction per chunk (phase 1, step 4).

Per chunk the model proposes surprising segments as fact candidates, each with the minimal
context it needs (preceding segments). Deterministic gates decide which candidates become
offered facts. Only the offered facts and their context are rendered to plain text.

A fact is exactly one segment; its ID is the segment ID (design: "Fakten").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from itertools import groupby
from math import ceil

from gut_zu_wissen.llm import StructuredLlm
from gut_zu_wissen.render import HtmlRenderer, html_to_plain
from gut_zu_wissen.segmentation import Segment, strip_refs

SENTENCE_START = re.compile(r"^[\"'„“‚‘»«(\[]*[A-ZÄÖÜ0-9]")
SENTENCE_END = re.compile(r"[.!?…][\"'“”‘’«»)\]]*$")
LAYOUT_ARTEFACTS = ("|}", "{|")  # a layout template rendered as table syntax (spike S3)

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "segment_id": {"type": "string"},
                    "context_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["segment_id", "context_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
Du wählst aus einem Abschnitt eines Wikipedia-Artikels die Kandidaten für Fakten eines \
Podcasts aus. Du bekommst den Abschnitt als Liste von Segmenten, jedes mit einer ID in eckigen \
Klammern. Ein Fakt ist genau ein Segment.

Wähle jedes Segment, das für sich genommen eine konkrete, interessante Aussage enthält, die man \
einem Laien erzählen kann. Welche davon am überraschendsten sind, entscheidet ein späterer \
Schritt; sortiere hier also nicht streng aus. Wähle nichts Triviales, keine reinen \
Aufzählungen, keine bloßen Verweise und keine Sätze, die nur Bekanntes oder Allgemeines \
wiederholen. Wenn der Abschnitt nichts davon enthält, gib eine leere Liste zurück.

Zu jedem gewählten Segment nennst du den Kontext, den man braucht, um es zu verstehen \
(zum Beispiel, worauf sich ein Pronomen oder eine Ortsangabe bezieht). Der Kontext besteht aus \
den Segmenten, die unmittelbar davor stehen, lückenlos und in der Reihenfolge des Abschnitts. \
Wähle so wenig Kontext wie möglich, höchstens {max_context} Segmente; wenn das Segment für \
sich verständlich ist, gib eine leere Liste zurück. Kontext dient nur zum Verstehen, nie als \
Beleg.

Antworte mit JSON im vorgegebenen Schema. Verwende nur IDs aus dem Abschnitt."""


@dataclass(frozen=True)
class Chunk:
    index: int
    section: str
    segments: list[Segment]


@dataclass
class Fact:
    """An offered fact; field order is the archive field order."""

    segment_id: str
    type: str
    section: str
    context_ids: list[str]
    plain: str
    context_plain: list[str]


@dataclass
class Rejected:
    segment_id: str
    reason: str


@dataclass
class ExtractionResult:
    revid: int
    chunk_count: int
    facts: list[Fact] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- pure functions


def make_chunks(segments: Sequence[Segment], max_segments: int) -> list[Chunk]:
    """One chunk per section; a section above `max_segments` is split into equal parts."""
    chunks: list[Chunk] = []
    for section, group in groupby(segments, key=lambda s: s.section):
        items = list(group)
        parts = ceil(len(items) / max_segments)
        base, extra = divmod(len(items), parts)
        start = 0
        for k in range(parts):
            size = base + (1 if k < extra else 0)
            chunks.append(Chunk(len(chunks), section, items[start : start + size]))
            start += size
    return chunks


def is_unique(segment: Segment, segments: Sequence[Segment]) -> bool:
    """Gate: the raw text occurs once in the article."""
    return sum(s.raw == segment.raw for s in segments) == 1


def context_allowed(
    fact_id: str, context_ids: Sequence[str], chunk: Chunk, max_context: int
) -> bool:
    """Gate: context is contiguous, directly precedes the fact, stays in the chunk, ≤ max."""
    ids = [s.id for s in chunk.segments]
    if fact_id not in ids or len(context_ids) > max_context:
        return False
    position = ids.index(fact_id)
    expected = ids[max(position - len(context_ids), 0) : position]
    return list(context_ids) == expected and len(expected) == len(context_ids)


def is_full_sentence(plain: str) -> bool:
    return bool(SENTENCE_START.match(plain) and SENTENCE_END.search(plain))


def plain_rejection(segment: Segment, plain: str, min_words: int) -> str | None:
    """Gate on the plain text: the reason for a rejection, or None. Spike S3 artefact gate for
    all types; word count and full sentence for sentence segments only."""
    if not plain:
        return "plain_empty"
    if any(a in plain for a in LAYOUT_ARTEFACTS):
        return "layout_artefact"
    if segment.type == "sentence":
        if len(plain.split()) < min_words:
            return "too_short"
        if not is_full_sentence(plain):
            return "not_a_sentence"
    return None


def format_chunk(chunk: Chunk) -> str:
    lines = [f"Abschnitt: {chunk.section or '(Einleitung)'}", ""]
    for s in chunk.segments:
        text = s.plain if s.type == "infobox_row" and s.plain else strip_refs(s.raw)
        lines.append(f"[{s.id}] {text.replace(chr(10), ' ')}")
    return "\n".join(lines)


# ---------------------------------------------------------------- extraction


def extract_facts(
    llm: StructuredLlm,
    renderer: HtmlRenderer,
    title: str,
    revid: int,
    segments: Sequence[Segment],
    config: dict,
) -> ExtractionResult:
    quotas = config["quotas"]
    max_context = quotas["max_context_segments"]
    chunks = make_chunks(segments, quotas["max_chunk_segments"])
    result = ExtractionResult(revid=revid, chunk_count=len(chunks))
    system = SYSTEM_PROMPT.format(max_context=max_context)
    by_id = {s.id: s for s in segments}

    accepted: list[tuple[Segment, list[str]]] = []
    for chunk in chunks:
        answer = llm.complete_json(system, format_chunk(chunk), EXTRACTION_SCHEMA)
        in_chunk = {s.id for s in chunk.segments}
        seen: set[str] = set()
        for cand in answer.get("candidates", []):
            fid = cand.get("segment_id")
            ctx = list(cand.get("context_ids", []))
            if fid in seen:
                continue
            seen.add(fid)
            if fid not in by_id or fid not in in_chunk:
                result.rejected.append(Rejected(str(fid), "unknown_segment"))
            elif not is_unique(by_id[fid], segments):
                result.rejected.append(Rejected(fid, "not_unique"))
            elif not context_allowed(fid, ctx, chunk, max_context):
                result.rejected.append(Rejected(fid, "context_invalid"))
            else:
                accepted.append((by_id[fid], ctx))

    plain = render_plain(renderer, title, accepted, by_id)
    for seg, ctx in accepted:
        reason = plain_rejection(seg, plain[seg.id], quotas["min_fact_words"])
        if reason:
            result.rejected.append(Rejected(seg.id, reason))
        else:
            result.facts.append(
                Fact(seg.id, seg.type, seg.section, ctx, plain[seg.id], [plain[c] for c in ctx])
            )
    return result


def render_plain(
    renderer: HtmlRenderer,
    title: str,
    accepted: Sequence[tuple[Segment, list[str]]],
    by_id: dict[str, Segment],
) -> dict[str, str]:
    """Plain text of every offered fact and context segment, in one batched render.

    Infobox rows already carry their plain text from step 3.
    """
    ids = list(dict.fromkeys(i for seg, ctx in accepted for i in [*ctx, seg.id]))
    plain = {i: by_id[i].plain or "" for i in ids if by_id[i].type == "infobox_row"}
    to_render = [i for i in ids if by_id[i].type == "sentence"]
    if to_render:
        html = renderer.render_html(title, [strip_refs(by_id[i].raw) for i in to_render])
        plain.update({i: html_to_plain(h) for i, h in zip(to_render, html, strict=True)})
    return plain
