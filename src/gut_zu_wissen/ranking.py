"""Ranking by surprise and the offer gate (phase 1, step 5).

The model orders the extracted facts by surprise (design: "Ranking und Kontingent"); for linked
intros it first keeps only facts directly connected to the main topic (decided 2026-11-20). The
deterministic parts are pure functions: validating the answer, the quota range, the link targets
of the offered facts and their bound, and the minimum-content gate (D5). `exclude` is the hook for
the conversation regeneration (D6): facts cited by rejected statements are left out of the ranking.
"""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
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
gegebenenfalls einem Kontext, der nur zum Verstehen dient. Wähle die überraschendsten Fakten für \
einen Podcast aus und ordne sie nach Überraschung: der überraschendste zuerst.

Ordne alle Fakten nach Überraschung: der überraschendste zuerst. Gib jeden Fakt zurück; \
weglassen musst du nichts, das Abschneiden übernimmt das Programm.

Das einzige Kriterium ist Überraschung: etwas, das ein Laie nicht erwartet. Sagen zwei Fakten im \
Kern dasselbe, steht der überraschendere weit vor dem anderen. Die Reihenfolge ist wichtig: die \
obersten Fakten muss das Gespräch verwenden.

Antworte mit JSON im vorgegebenen Schema. Verwende nur IDs aus der Liste, jede höchstens einmal."""

LINKED_SYSTEM_PROMPT = """\
Der Podcast handelt vom Wikipedia-Artikel „{main_title}“ (Hauptthema). Du bekommst Fakten aus \
den Einleitungen verlinkter Artikel, jeweils mit einer ID in eckigen Klammern, dem verlinkten \
Artikel in Klammern und gegebenenfalls einem Kontext, der nur zum Verstehen dient.

Wähle nur Fakten, die direkt mit dem Hauptthema „{main_title}“ verbunden sind: Der Fakt sagt \
etwas über das Hauptthema selbst oder über seine direkte Beziehung zum verlinkten Gegenstand. \
Beispiele: Der frühere Name eines Verschlüsselungsprotokolls spielt auf den Axolotl an; \
Regeneration bei Amphibien in einer Folge über den Axolotl. Lass Fakten weg, die nur den \
verlinkten Gegenstand für sich beschreiben, auch wenn sie überraschen: Biografie, Größe einer \
Institution, ein Ereignis ohne Bezug zum Hauptthema.

Unter den direkt verbundenen Fakten ordnest du nach Überraschung: der überraschendste zuerst. \
Wähle mindestens {minimum} Fakten, wenn es so viele direkt verbundene gibt, und höchstens \
{maximum}. Gehe über {minimum} nur mit Fakten hinaus, die wirklich überraschend sind und keine \
Dopplungen. Gibt es weniger direkt verbundene Fakten, gib entsprechend weniger zurück, auch \
keinen.

Wähle nicht zwei Fakten, die im Kern dasselbe sagen; nimm dann den überraschenderen. Antworte mit \
JSON im vorgegebenen Schema. Verwende nur IDs aus der Liste, jede höchstens einmal."""

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


def format_facts(facts: Sequence[Fact], sources: Mapping[str, str] | None = None) -> str:
    """One line per fact; `sources` maps fact IDs to their linked article, shown after the ID."""
    lines: list[str] = []
    for f in facts:
        source = f" (verlinkter Artikel: {sources[f.segment_id]})" if sources else ""
        lines.append(f"[{f.segment_id}]{source} {f.plain.replace(chr(10), ' ')}")
        if f.context_plain:
            lines.append("  Kontext: " + " ".join(c.replace(chr(10), " ") for c in f.context_plain))
    return "\n".join(lines)


def select_ranked(
    answer_ids: Iterable[object],
    candidates: Sequence[Fact],
    maximum: int,
    complete: bool = False,
) -> list[Fact]:
    """The model's order, cut to `maximum`. Unknown IDs and repeats are dropped; fewer than the
    minimum is allowed (`offer_count` makes it visible). With `complete`, candidates the model
    did not name follow in their given order, so the cut, not the model, decides what is left
    out (D8)."""
    by_id = {f.segment_id: f for f in candidates}
    chosen: list[Fact] = []
    for fid in answer_ids:
        if isinstance(fid, str) and fid in by_id and by_id[fid] not in chosen:
            chosen.append(by_id[fid])
    if complete:
        chosen += [f for f in candidates if f not in chosen]
    return chosen[:maximum]


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


def bound_targets(targets: Sequence[str], maximum: int) -> tuple[list[str], list[str]]:
    """The first `maximum` link targets (those of the most surprising facts) and the rest."""
    return list(targets[:maximum]), list(targets[maximum:])


def offer_size(extracted: int, share: float, minimum: int, maximum: int) -> int:
    """D8: size of the main offer: `share` of the extracted facts, rounded up, at least `minimum`,
    at most `maximum` (2026-11-21), never more than extracted."""
    return min(extracted, maximum, max(minimum, math.ceil(extracted * share)))


def must_use_count(extracted: int, share: float, maximum: int) -> int:
    """D8: number of top-ranked facts the dialog must use: `share` of the extracted facts, rounded
    up, at most `maximum` (2026-11-21), 0 when there are none."""
    return min(maximum, math.ceil(extracted * share)) if extracted else 0


def offer_count(offered: int, minimum: int, maximum: int) -> dict:
    """Offered facts against the range; field order is the archive field order."""
    return {"offered": offered, "min": minimum, "max": maximum, "below_min": offered < minimum}


def offer_summary(counts: dict) -> str:
    """One line for the job summary: offered facts per ranking against the range."""
    parts = []
    for name, c in counts.items():
        flag = ", below the minimum" if c["below_min"] else ""
        parts.append(f"{c['offered']} {name} facts (range {c['min']}-{c['max']}{flag})")
    return "Offer: " + ", ".join(parts)


def offer_gate(offered: int, minimum: int) -> str | None:
    """D5: fewer than `minimum` offered facts is a gate failure before the dialog author."""
    if offered < minimum:
        return f"Gate minimum content (D5): {offered} facts offered, at least {minimum} needed."
    return None


# ---------------------------------------------------------------- ranking


def rank_facts(
    llm: StructuredLlm,
    facts: Sequence[Fact],
    minimum: int,
    maximum: int,
    exclude: Collection[str] = (),
) -> list[Fact]:
    """At least `minimum` (if there are that many) and at most `maximum` of the most surprising
    facts, most surprising first; `exclude` lists IDs to omit."""
    candidates = [f for f in facts if f.segment_id not in exclude]
    if len(candidates) < 2:
        return candidates
    answer = llm.complete_json(
        SYSTEM_PROMPT.format(minimum=minimum, maximum=maximum),
        format_facts(candidates),
        RANKING_SCHEMA,
    )
    return select_ranked(answer.get("ranked_ids", []), candidates, maximum, complete=True)


def rank_linked_facts(
    llm: StructuredLlm,
    facts: Sequence[Fact],
    sources: Mapping[str, str],
    main_title: str,
    minimum: int,
    maximum: int,
    exclude: Collection[str] = (),
) -> list[Fact]:
    """Facts from linked intros that directly concern `main_title`, most surprising first.

    `sources` maps each fact ID to its linked article. Unlike `rank_facts`, a single candidate
    still goes to the model: relevance to the main topic is not given.
    """
    candidates = [f for f in facts if f.segment_id not in exclude]
    if not candidates:
        return []
    answer = llm.complete_json(
        LINKED_SYSTEM_PROMPT.format(main_title=main_title, minimum=minimum, maximum=maximum),
        format_facts(candidates, sources),
        RANKING_SCHEMA,
    )
    return select_ranked(answer.get("ranked_ids", []), candidates, maximum)


def ranked(facts: Sequence[Fact], article: str, revid: int) -> list[RankedFact]:
    return [RankedFact(i, article, revid, f) for i, f in enumerate(facts, start=1)]


def raw_of(facts: Iterable[Fact], segments: Sequence[Segment]) -> list[str]:
    by_id = {s.id: s.raw for s in segments}
    return [by_id[f.segment_id] for f in facts]
