"""Orientation facts ("Einordnung"; user decision 2026-11-20).

The first sentences of the main article's intro (section "", the text before the first heading)
are always offered, outside the surprise ranking and the main quota, so the dialog can say what,
when and where before the first surprise. Only the deterministic fact gates apply (uniqueness,
plain-text gates); a sentence that fails one is skipped and the next one is taken. Infobox rows
do not count. Context: the directly preceding intro sentences, at most `max_context_segments`.
Facts excluded by a conversation regeneration (D6) are skipped like a failed gate.

Selection is a pure function over the rendered candidates; rendering is one batched call through
the same `HtmlRenderer` as the extraction.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from gut_zu_wissen.extraction import Fact, is_unique, plain_rejection, render_plain
from gut_zu_wissen.render import HtmlRenderer
from gut_zu_wissen.segmentation import Segment

INTRO_SECTION = ""

Candidate = tuple[Segment, list[str]]


def intro_candidates(segments: Sequence[Segment], max_context: int) -> list[Candidate]:
    """Unique intro sentences in article order, each with the preceding intro sentences as
    context (contiguous, at most `max_context`)."""
    candidates: list[Candidate] = []
    run: list[str] = []  # contiguous intro sentences so far
    for segment in segments:
        if segment.section != INTRO_SECTION or segment.type != "sentence":
            run = []
            continue
        if is_unique(segment, segments):
            candidates.append(
                (segment, run[max(len(run) - max_context, 0) :] if max_context else [])
            )
        run.append(segment.id)
    return candidates


def select_einordnung(
    candidates: Sequence[Candidate],
    plain: Mapping[str, str],
    min_words: int,
    maximum: int,
    exclude: Collection[str] = (),
) -> list[Fact]:
    """The first `maximum` candidates that pass the plain-text gates and are not excluded."""
    facts: list[Fact] = []
    for segment, context in candidates:
        if len(facts) >= maximum:
            break
        if segment.id in exclude or plain_rejection(segment, plain[segment.id], min_words):
            continue
        facts.append(
            Fact(
                segment.id,
                segment.type,
                segment.section,
                list(context),
                plain[segment.id],
                [plain[c] for c in context],
            )
        )
    return facts


def render_candidates(
    renderer: HtmlRenderer, title: str, segments: Sequence[Segment], candidates: Sequence[Candidate]
) -> dict[str, str]:
    """Plain text of every candidate and its context, one batched render (none without
    candidates)."""
    if not candidates:
        return {}
    return render_plain(renderer, title, candidates, {s.id: s for s in segments})
