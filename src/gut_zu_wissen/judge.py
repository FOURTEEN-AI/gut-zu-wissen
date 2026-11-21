"""Judge (phase 1, step 7; design: "Judge").

Stage 1 is Jev: one question per statement plus the completeness question, one request per turn
(question wording from spike S1). The threshold decision is a pure function:
all p >= t -> supported; any p <= 1 - t -> not supported (the LLM judge only gives the reason);
otherwise the LLM judge decides with the full rubric. Overlength and transient Jev errors also go
to the LLM judge; abort errors propagate. The LLM judge sees the previous turn (what the turn
answers, never evidence), the turn, the cited fact sentences and their context, never the author's
statements or chain of thought; in its full verdict it also sees the Jev questions below t as a
hint (issue #15). Its rubric checks that each statement means the same thing as its fact
sentence when heard after the previous turn (dangling references, answers to a different
question; issues #8, #15) and that a question does not anticipate its answer.
Both LLM-judge prompts are fixed system prompts and are cached (spike S7); the turn and its facts
vary per call and follow them.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.jev import Jev, JevOverlengthError, JevTransientError
from gut_zu_wissen.llm import StructuredLlm
from gut_zu_wissen.ranking import RankedFact

COMPLETENESS_KEY = "vollstaendig"

REACTION = (
    "eine Reaktion: ein Gefühl im Moment (Staunen, Neugier, Belustigung), eine Nachfrage, "
    "eine Bitte um Erklärung, eine Überleitung oder ein Wiederaufgreifen des eben Gesagten "
    "ohne neue Information"
)

RUBRIC = (
    """\
Rubrik: Jede Behauptung, jede Meinung und jede Prämisse einer Frage im Redebeitrag muss aus den \
Fakt-Sätzen folgen: gleiche Zahlen, Daten und Namen, keine zusätzliche Angabe; andere \
Formulierungen und ausgeschriebene Zahlen sind in Ordnung, einfache Rechnungen aus den Fakt-Sätzen \
auch. Was nur im Kontext steht, gilt als nicht belegt. Ein Redebeitrag ohne Fakt-IDs darf keine \
Behauptung enthalten. Reaktionen gelten als belegfrei: """
    + REACTION
    + """. Nicht belegfrei \
sind neue Informationen, Wertungen mit Faktengehalt und Aussagen der sprechenden Person über \
sich selbst (Erlebnisse, Wissen, Vorlieben).

Der vorherige Redebeitrag zeigt, worauf der Redebeitrag antwortet; er ist kein Beleg. Zuhörer \
hören den Redebeitrag im Anschluss an den vorherigen, deshalb muss jede Aussage denselben \
Gegenstand meinen wie ihr Fakt-Satz. Nicht belegt ist eine Aussage auch dann, wenn
- der Fakt-Satz seinen Gegenstand oder eine beteiligte Person nur mit einem Bezugswort nennt \
(etwa „dieses Instrument“, „er“, „ihm“, „dort“, „die Bilder“) und weder der Redebeitrag noch der \
vorherige Redebeitrag sie eindeutig nennt, sodass Zuhörer die Aussage auf etwas anderes beziehen \
können, etwa auf den Gegenstand der vorherigen Frage;
- der Redebeitrag eine Frage aus dem vorherigen Redebeitrag beantwortet, der Fakt-Satz aber von \
einem anderen Gegenstand handelt als die Frage;
- der Redebeitrag Fakt-Sätze so verbindet, dass eine Beziehung entsteht, die keiner von ihnen \
nennt.
Einen Bezug aufzulösen, also den Gegenstand eines Bezugsworts so mit Namen zu nennen, wie der \
Kontext ihn bestimmt, ist erlaubt. Eine Nachfrage ist nur belegfrei, wenn sie offen fragt. Legt \
sie eine Antwort nahe, etwa mit einem Wertwort, einer Vermutung oder einer Einzelheit („War die \
Kritik vielleicht sogar ziemlich scharf?“), ist das eine Prämisse; sie muss aus den Fakt-Sätzen \
des Redebeitrags folgen."""
)

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["supported", "reason"],
    "additionalProperties": False,
}
REASON_SCHEMA = {
    "type": "object",
    "properties": {"reason": {"type": "string"}},
    "required": ["reason"],
    "additionalProperties": False,
}
VERDICT_PROMPT = (
    "Du prüfst einen Redebeitrag aus einem Podcast-Gespräch auf Faktentreue. Du bekommst den "
    "vorherigen Redebeitrag, falls es einen gibt, den Redebeitrag, die Fakt-Sätze, auf die er "
    "sich beruft, deren Kontext und gegebenenfalls die Fragen, bei denen eine automatische "
    "Vorprüfung unsicher war. Diese Fragen sind ein Hinweis, wo du genau hinsehen sollst, kein "
    "Urteil.\n\n"
    + RUBRIC
    + "\n\nUrteile über den ganzen Redebeitrag: „supported“ ist nur wahr, wenn er die Rubrik "
    "vollständig erfüllt. Begründe kurz auf Deutsch; nenne bei Ablehnung die betroffene Stelle. "
    "Antworte mit JSON im vorgegebenen Schema."
)
REASON_PROMPT = (
    "Ein Redebeitrag aus einem Podcast-Gespräch wurde als nicht belegt abgelehnt. Du bekommst den "
    "vorherigen Redebeitrag, falls es einen gibt, den Redebeitrag, die Fakt-Sätze, auf die er "
    "sich beruft, deren Kontext und die Stellen, die beanstandet wurden.\n\n"
    + RUBRIC
    + "\n\nDie Ablehnung steht fest; du urteilst nicht. Begründe auf Deutsch in ein bis zwei "
    "Sätzen, welche Stelle des Redebeitrags aus den Fakt-Sätzen nicht folgt und warum. "
    "Antworte mit JSON im vorgegebenen Schema."
)


class Decision(StrEnum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass
class TurnJudgement:
    """Judgement of one turn; field order is the archive field order."""

    turn_index: int
    accepted: bool
    decided_by: str  # "jev" or "llm_judge"
    jev_p: dict[str, float] = field(default_factory=dict)
    jev_error: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- pure functions


def decide(probabilities: Collection[float], t: float) -> Decision:
    """Symmetric threshold: all p >= t supported; any p <= 1 - t rejected; otherwise uncertain.

    A rejection wins over everything else. No questions at all is uncertain.
    """
    ps = list(probabilities)
    if not ps:
        return Decision.UNCERTAIN
    if any(p <= 1 - t for p in ps):
        return Decision.REJECTED
    if all(p >= t for p in ps):
        return Decision.SUPPORTED
    return Decision.UNCERTAIN


def cited_ids(turn: Turn) -> list[str]:
    """Fact IDs of the turn and its statements, first-use order."""
    ids: list[str] = []
    for fid in [*turn.fact_ids, *(i for s in turn.statements for i in s.fact_ids)]:
        if fid not in ids:
            ids.append(fid)
    return ids


def statement_question(text: str, fact_ids: Sequence[str]) -> dict:
    paths = " und ".join(f"`fakten.{f}`" for f in fact_ids)
    return {
        "type": "noul",
        "instructions": f"Folgt die Aussage „{text}“ aus {paths}?",
        "criteria": {
            "true": f"{paths} sagt das aus oder es folgt direkt daraus. Andere Formulierungen "
            "und ausgeschriebene Zahlen ändern daran nichts.",
            "false": f"Die Aussage enthält etwas, das {paths} nicht sagt, zum Beispiel eine "
            "abweichende Zahl, ein anderes Datum, einen anderen Namen oder eine zusätzliche "
            "Angabe. Was nur in `kontext` steht, gilt als nicht belegt.",
        },
    }


def completeness_question() -> dict:
    return {
        "type": "noul",
        "instructions": "Ist jede Sachbehauptung und jede Prämisse einer Frage in "
        "`redebeitrag.text` unter `aussagen` erfasst?",
        "criteria": {
            "true": "Alles in `redebeitrag.text`, was nicht unter `aussagen` steht, ist "
            + REACTION
            + ".",
            "false": "`redebeitrag.text` enthält etwas, das unter `aussagen` fehlt: eine neue "
            "Information, eine Wertung mit Faktengehalt oder eine Aussage der sprechenden "
            "Person über sich selbst, etwa über Erlebnisse, Wissen oder Vorlieben.",
        },
    }


def build_request(
    turn: Turn, previous: Turn | None, facts: Mapping[str, RankedFact]
) -> tuple[dict, dict[str, dict]]:
    """Jev state and questions for one turn (request design of spike S1)."""
    cited = [i for i in cited_ids(turn) if i in facts]
    state: dict = {"redebeitrag": {"sprecher": turn.speaker, "text": turn.text}}
    if previous is not None:
        state["vorheriger_redebeitrag"] = previous.text
    state["aussagen"] = [{"text": s.text, "fakten": s.fact_ids} for s in turn.statements]
    state["fakten"] = {
        i: {"artikel": facts[i].article, "satz": facts[i].fact.plain.replace("\n", " ")}
        for i in cited
    }
    context = {i: facts[i].fact.context_plain for i in cited if facts[i].fact.context_plain}
    if context:
        state["kontext"] = context
    questions = {
        f"aussage_{n}": statement_question(s.text, s.fact_ids)
        for n, s in enumerate(turn.statements)
    }
    questions[COMPLETENESS_KEY] = completeness_question()
    return state, questions


def format_judge_input(
    turn: Turn,
    facts: Mapping[str, RankedFact],
    flagged: Sequence[str] = (),
    previous: Turn | None = None,
    doubts: Sequence[str] = (),
) -> str:
    """Input of the LLM judge: the previous turn (what the turn answers; never evidence), the turn,
    cited fact sentences, context, optionally flagged spots (rejection) or Jev's doubts
    (uncertain)."""
    lines = []
    if previous is not None:
        lines += [f"Vorheriger Redebeitrag ({previous.speaker}):", previous.text, ""]
    lines += [f"Redebeitrag ({turn.speaker}):", turn.text, "", "Zitierte Fakten:"]
    cited = [i for i in cited_ids(turn) if i in facts]
    if not cited:
        lines.append("(keine)")
    for i in cited:
        fact = facts[i].fact
        lines.append(f"[{i}] {fact.plain.replace(chr(10), ' ')}")
        if fact.context_plain:
            lines.append("  Kontext: " + " ".join(c.replace("\n", " ") for c in fact.context_plain))
    if flagged:
        lines += ["", "Beanstandete Stellen:", *flagged]
    if doubts:
        lines += ["", "Unsicher in der Vorprüfung (Hinweis, kein Urteil):", *doubts]
    return "\n".join(lines)


def flagged_spots(turn: Turn, probabilities: Mapping[str, float], t: float) -> list[str]:
    spots = []
    for n, s in enumerate(turn.statements):
        if probabilities.get(f"aussage_{n}", 1.0) <= 1 - t:
            spots.append(f"Aussage nicht belegt: {s.text}")
    if probabilities.get(COMPLETENESS_KEY, 1.0) <= 1 - t:
        spots.append(
            "Der Redebeitrag enthält Sachliches, das unter den Aussagen fehlt "
            "(neue Information, Wertung mit Faktengehalt oder Aussage über sich selbst)."
        )
    return spots


def doubtful_spots(turn: Turn, probabilities: Mapping[str, float], t: float) -> list[str]:
    """Jev questions below t, for the LLM judge's full verdict (issue #15): where to look.

    A statement question is named by its fact IDs only; the author's statement text stays hidden
    from the verdict (design: no shared context with the author)."""
    spots = []
    for n, s in enumerate(turn.statements):
        p = probabilities.get(f"aussage_{n}")
        if p is not None and p < t:
            ids = ", ".join(f"[{i}]" for i in s.fact_ids)
            spots.append(f"Eine Aussage, die sich auf {ids} beruft (p = {p:.2f}).")
    p = probabilities.get(COMPLETENESS_KEY)
    if p is not None and p < t:
        spots.append(
            f"Vollständigkeit (p = {p:.2f}): ob der Redebeitrag außer den Aussagen nur "
            "Reaktionen enthält, auch in seinen Fragen."
        )
    return spots


def used_in_accepted(
    turns: Sequence[Turn], judgements: Sequence[TurnJudgement], offered: Collection[str]
) -> list[str]:
    """Offered facts cited by accepted turns, first-use order."""
    accepted = {j.turn_index for j in judgements if j.accepted}
    used: list[str] = []
    for turn in turns:
        if turn.index in accepted:
            used += [i for i in cited_ids(turn) if i in offered and i not in used]
    return used


def judged_facts_gate(used: int, minimum: int) -> str | None:
    """D5: fewer than `minimum` used facts in accepted turns is a gate failure."""
    if used < minimum:
        return (
            f"Gate minimum content (D5): {used} facts used in accepted turns, "
            f"at least {minimum} needed."
        )
    return None


# ---------------------------------------------------------------- judging


def judge_turn(
    jev: Jev,
    llm: StructuredLlm,
    turn: Turn,
    previous: Turn | None,
    facts: Mapping[str, RankedFact],
    t: float,
) -> TurnJudgement:
    """Judge one turn. Jev abort errors propagate."""
    state, questions = build_request(turn, previous, facts)
    try:
        probabilities = jev.ask(state, questions)
    except (JevOverlengthError, JevTransientError) as exc:
        verdict = llm.complete_json(
            VERDICT_PROMPT,
            format_judge_input(turn, facts, previous=previous),
            VERDICT_SCHEMA,
            cache_system=True,
        )
        return TurnJudgement(
            turn.index,
            verdict.get("supported") is True,
            "llm_judge",
            jev_error=f"{type(exc).__name__}: {exc}",
            reason=str(verdict.get("reason", "")),
        )
    decision = decide(probabilities.values(), t)
    if decision is Decision.SUPPORTED:
        return TurnJudgement(turn.index, True, "jev", probabilities)
    if decision is Decision.REJECTED:
        spots = flagged_spots(turn, probabilities, t)
        answer = llm.complete_json(
            REASON_PROMPT,
            format_judge_input(turn, facts, spots, previous),
            REASON_SCHEMA,
            cache_system=True,
        )
        return TurnJudgement(
            turn.index, False, "jev", probabilities, reason=str(answer.get("reason", ""))
        )
    doubts = doubtful_spots(turn, probabilities, t)
    verdict = llm.complete_json(
        VERDICT_PROMPT,
        format_judge_input(turn, facts, previous=previous, doubts=doubts),
        VERDICT_SCHEMA,
        cache_system=True,
    )
    return TurnJudgement(
        turn.index,
        verdict.get("supported") is True,
        "llm_judge",
        probabilities,
        reason=str(verdict.get("reason", "")),
    )


def judge_dialog(
    jev: Jev,
    llm: StructuredLlm,
    turns: Sequence[Turn],
    facts: Mapping[str, RankedFact],
    t: float,
) -> list[TurnJudgement]:
    return [
        judge_turn(jev, llm, turn, turns[n - 1] if n else None, facts, t)
        for n, turn in enumerate(turns)
    ]
