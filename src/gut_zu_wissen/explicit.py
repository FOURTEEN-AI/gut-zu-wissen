"""Explicit check (phase 1, step 11; decision D7).

The LLM judge model reads the final script, i.e. the dialog turns accepted by the judge, and
reports drastic depictions of violence, sexuality or drugs, and spoken profanity; a factual
or passing mention does not count (user decision 2026-11-19). The flag is a pure function
of its answer: the episode is explicit unless the model says "clean" and reports no finding;
"uncertain", a finding, or an answer outside the schema all count as explicit (D7: uncertain ->
explicit). Intro and outro are fixed text outside the script and not checked.

The flag does not block the episode; it goes to `episode.json` (`explicit`) and, with the
findings, to `08_explicit.json`, and becomes the per-episode `itunes:explicit` in phase 2. A
failed call (`LlmError`: refusal, cut-off or invalid answer) gives no judgement at all; the run
escalates, as a failed LLM decision does in the STT verification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.llm import StructuredLlm


class Category(StrEnum):
    VIOLENCE = "violence"
    SEXUALITY = "sexuality"
    DRUGS = "drugs"
    PROFANITY = "profanity"


class Verdict(StrEnum):
    EXPLICIT = "explicit"
    CLEAN = "clean"
    UNCERTAIN = "uncertain"


@dataclass
class Finding:
    """One passage the model reports; field order is the archive field order."""

    turn_index: int | None
    category: str
    quote: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExplicitCheck:
    """Content of 08_explicit.json; field order is the archive field order.

    `verdict` is the model's answer as given; `explicit` is the flag derived from it.
    """

    model: str
    explicit: bool | None = None
    verdict: str | None = None
    findings: list[Finding] = field(default_factory=list)
    reason: str | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


EXPLICIT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn_index": {"type": "integer"},
                    "category": {"type": "string", "enum": [c.value for c in Category]},
                    "quote": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["turn_index", "category", "quote", "reason"],
                "additionalProperties": False,
            },
        },
        "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
        "reason": {"type": "string"},
    },
    "required": ["findings", "verdict", "reason"],
    "additionalProperties": False,
}
EXPLICIT_PROMPT = (
    "Du prüfst das Skript einer Podcast-Folge darauf, ob sie als „explicit“ gekennzeichnet "
    "werden muss. Das Skript ist ein Gespräch zwischen Lena und Felix über einen Artikel der "
    "deutschsprachigen Wikipedia; jeder Redebeitrag hat eine Nummer.\n\n"
    "Die Folge ist explicit, sobald irgendein Redebeitrag eines der folgenden Themen drastisch "
    "darstellt. Drastisch heißt: anschaulich, detailliert oder reißerisch geschildert, "
    "verherrlicht oder als Anleitung. Eine sachliche, historische, wissenschaftliche oder "
    "beiläufige Erwähnung ohne solche Schilderung zählt nicht; eine Schlacht, ein Raubtier, "
    "das Beute jagt, oder der Weinbau sind für sich allein nicht explicit.\n"
    "- „violence“: drastische Gewalt, etwa ausgemalte Verletzungen, Folter oder Tötungen im "
    "Detail, blutige Einzelheiten, verherrlichte Gewalt, Anleitung zu Selbstverletzung oder "
    "Suizid.\n"
    "- „sexuality“: drastische Sexualität, etwa ausdrücklich geschilderte sexuelle Handlungen, "
    "detailliert geschilderte sexuelle Gewalt oder deutliche sexuelle Anspielungen.\n"
    "- „drugs“: drastische Darstellung von Drogen und Rauschmitteln, etwa verherrlichter "
    "Konsum, ausgemalte Rauschzustände oder Anleitung zu Herstellung oder Konsum.\n"
    "- „profanity“: Schimpfwörter, Flüche, vulgäre oder beleidigende Ausdrücke, die im "
    "Gespräch tatsächlich ausgesprochen werden.\n\n"
    "Nenne unter „findings“ für jede Kategorie, die vorkommt, die erste Fundstelle: Nummer des "
    "Redebeitrags, Kategorie, die Stelle wörtlich und eine kurze Begründung auf Deutsch. "
    "Ergebnis unter „verdict“: „explicit“, wenn es mindestens eine Fundstelle gibt; „clean“ nur, "
    "wenn sicher keines der Themen vorkommt; „uncertain“, wenn du unsicher bist. Im Zweifel "
    "„uncertain“. Begründe das Ergebnis unter „reason“ kurz auf Deutsch. Antworte mit JSON im "
    "vorgegebenen Schema."
)


# ---------------------------------------------------------------- pure functions


def format_script(turns: Sequence[Turn]) -> str:
    """Input of the check: the final script, one numbered line per turn."""
    lines = ["Skript:"]
    lines += [f"{turn.index}. {turn.speaker}: {turn.text}" for turn in turns]
    return "\n".join(lines)


def parse_findings(answer: dict) -> list[Finding]:
    """Findings as given; an item outside the schema is kept with what it has, so it still
    counts (any finding makes the episode explicit)."""
    findings: list[Finding] = []
    for item in answer.get("findings", []) or []:
        if not isinstance(item, dict):
            findings.append(Finding(None, "", str(item), ""))
            continue
        index = item.get("turn_index")
        findings.append(
            Finding(
                index if isinstance(index, int) and not isinstance(index, bool) else None,
                str(item.get("category", "")),
                str(item.get("quote", "")),
                str(item.get("reason", "")),
            )
        )
    return findings


def is_explicit(verdict: object, findings: Sequence[Finding]) -> bool:
    """D7: explicit unless the verdict is "clean" and there is no finding. "uncertain", any
    finding, a missing or unknown verdict all count as explicit."""
    return verdict != Verdict.CLEAN.value or bool(findings)


def apply_answer(record: ExplicitCheck, answer: dict) -> None:
    """Write the model's answer and the derived flag into `record`."""
    verdict = answer.get("verdict")
    record.verdict = verdict if isinstance(verdict, str) else None
    record.findings = parse_findings(answer)
    record.reason = str(answer.get("reason", ""))
    record.explicit = is_explicit(verdict, record.findings)


# ---------------------------------------------------------------- check


def check_explicit(record: ExplicitCheck, llm: StructuredLlm, turns: Sequence[Turn]) -> None:
    """One LLM call on the whole script; fills `record` in place. `LlmError` propagates."""
    answer = llm.complete_json(EXPLICIT_PROMPT, format_script(turns), EXPLICIT_SCHEMA)
    apply_answer(record, answer)
