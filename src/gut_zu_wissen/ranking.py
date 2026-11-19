"""Ranking by surprise and the offer gate (phase 1, step 5).

The model orders the extracted facts by surprise only (design: "Ranking und Kontingent"). The
deterministic parts are pure functions: validating the answer, the quota, the link targets of the
offered facts and the minimum-content gate (D5). `exclude` is the hook for the conversation
regeneration (D6): facts cited by rejected statements are left out of the ranking.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import asdict, dataclass

from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.llm import StructuredLlm
from gut_zu_wissen.segmentation import Segment, strip_refs

RANKING_SCHEMA = {
    "type": "object",
    "properties": {"ranked_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["ranked_ids"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
Du bekommst Fakten aus einem Wikipedia-Artikel, jeweils mit einer ID in eckigen Klammern und \
gegebenenfalls einem Kontext, der nur zum Verstehen dient. Wähle die {quota} überraschendsten \
Fakten für einen Podcast aus und ordne sie: der überraschendste zuerst.

Das einzige Kriterium ist Überraschung: etwas, das ein Laie nicht erwartet. Wähle nicht zwei \
Fakten, die im Kern dasselbe sagen; nimm dann den überraschenderen. Gibt es weniger als {quota} \
Fakten, die du für überraschend hältst, gib entsprechend weniger zurück.

Antworte mit JSON im vorgegebenen Schema. Verwende nur IDs aus der Liste, jede höchstens einmal."""

# Namespace prefixes that are not articles; links with these prefixes are no link targets.
NON_ARTICLE_PREFIXES = {
    "datei", "bild", "file", "image", "kategorie", "category", "medium", "media",
    "spezial", "special", "wikipedia", "hilfe", "help", "portal", "vorlage", "template",
    "benutzer", "user", "mediawiki", "modul", "module", "diskussion", "talk",
    "benutzer diskussion", "wikipedia diskussion",
}  # fmt: skip
LINK = re.compile(r"\[\[([^\[\]|]+)")


@dataclass
class RankedFact:
    """A fact in the offer; field order is the archive field order."""

    rank: int
    article: str
    revid: int
    fact: Fact

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- pure functions


def format_facts(facts: Sequence[Fact]) -> str:
    lines: list[str] = []
    for f in facts:
        lines.append(f"[{f.segment_id}] {f.plain.replace(chr(10), ' ')}")
        if f.context_plain:
            lines.append("  Kontext: " + " ".join(c.replace(chr(10), " ") for c in f.context_plain))
    return "\n".join(lines)


def select_ranked(
    answer_ids: Iterable[object], candidates: Sequence[Fact], quota: int
) -> list[Fact]:
    """The model's order, cut to `quota`. Unknown IDs and repeats are dropped."""
    by_id = {f.segment_id: f for f in candidates}
    chosen: list[Fact] = []
    for fid in answer_ids:
        if isinstance(fid, str) and fid in by_id and by_id[fid] not in chosen:
            chosen.append(by_id[fid])
    return chosen[:quota]


def link_targets(raw_texts: Iterable[str], own_title: str = "") -> list[str]:
    """Article titles linked in the given raw segment texts, first occurrence order.

    Links in `<ref>` tags, files, categories and other namespaces are no targets. Anchors are
    cut off; underscores become spaces.
    """
    targets: list[str] = []
    for raw in raw_texts:
        for match in LINK.finditer(strip_refs(raw)):
            target = match.group(1).split("#", 1)[0].replace("_", " ").strip()
            if not target or target.startswith(":"):
                continue
            prefix = target.split(":", 1)[0].strip().lower() if ":" in target else ""
            if prefix in NON_ARTICLE_PREFIXES:
                continue
            if target != own_title and target not in targets:
                targets.append(target)
    return targets


def offer_gate(offered: int, minimum: int) -> str | None:
    """D5: fewer than `minimum` offered facts is a gate failure before the dialog author."""
    if offered < minimum:
        return f"Gate minimum content (D5): {offered} facts offered, at least {minimum} needed."
    return None


# ---------------------------------------------------------------- ranking


def rank_facts(
    llm: StructuredLlm,
    facts: Sequence[Fact],
    quota: int,
    exclude: Collection[str] = (),
) -> list[Fact]:
    """The `quota` most surprising facts, most surprising first; `exclude` lists IDs to omit."""
    candidates = [f for f in facts if f.segment_id not in exclude]
    if len(candidates) < 2:
        return candidates
    answer = llm.complete_json(
        SYSTEM_PROMPT.format(quota=quota), format_facts(candidates), RANKING_SCHEMA
    )
    return select_ranked(answer.get("ranked_ids", []), candidates, quota)


def ranked(facts: Sequence[Fact], article: str, revid: int) -> list[RankedFact]:
    return [RankedFact(i, article, revid, f) for i, f in enumerate(facts, start=1)]


def raw_of(facts: Iterable[Fact], segments: Sequence[Segment]) -> list[str]:
    by_id = {s.id: s.raw for s in segments}
    return [by_id[f.segment_id] for f in facts]
