"""Orchestrator (phase 1, step 8; design: "Fehlerbehandlung", decisions D1, D5, D6).

Turn level: a rejected turn (gate problem or judge) is regenerated with its neighbours as context,
at most `turn_retries` times. After every regeneration the next turn is regenerated as well
(neighbour cascade, D1); Jev checks whether the regenerated turn still conveys the statements of
the version before. p >= t stops the cascade; anything else (including uncertain and Jev errors
other than abort) continues it. All regenerations of a cascade, retries after a rejection
included, share the budget of the turn that triggered it.

Conversation level: an exhausted turn budget, or fewer than `min_used_facts` used facts after the
author (D5), discards the dialog and writes a new one from a fresh offer (D6: both rankings rerun,
facts cited by rejected statements excluded, cumulatively), at most `conversation_retries` times.
Everything else ends in `GateFailure`, the escalation to the owner.

Deterministic parts are pure functions; the LLM and Jev calls sit behind their interfaces.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

from gut_zu_wissen.dialog import (
    Dialog,
    Turn,
    check_turns,
    format_offer,
    regenerate_turn,
    used_facts_gate,
    write_dialog,
)
from gut_zu_wissen.jev import Jev, JevAbortError, JevOverlengthError, JevTransientError
from gut_zu_wissen.judge import (
    Decision,
    TurnJudgement,
    cited_ids,
    decide,
    judge_turn,
    judged_facts_gate,
    used_in_accepted,
)
from gut_zu_wissen.llm import StructuredLlm
from gut_zu_wissen.markup import markup_problems
from gut_zu_wissen.ranking import RankedFact

CASCADE_KEY = "gleiche_aussagen"


class GateFailure(Exception):
    """A gate or the retry budgets stopped the run; the message is the notice for the owner."""


@dataclass
class Offer:
    """The facts offered to the dialog author."""

    main: list[RankedFact]
    linked: list[RankedFact]

    def by_id(self) -> dict[str, RankedFact]:
        return {r.fact.segment_id: r for r in [*self.main, *self.linked]}


@dataclass
class Regeneration:
    """One regenerated turn; `previous` is the version that was replaced."""

    conversation: int
    turn_index: int
    kind: str  # "retry" after a rejection, "cascade" for a neighbour
    reason: str | None
    previous: Turn


@dataclass
class CascadeCheck:
    conversation: int
    turn_index: int
    p: float | None
    confirmed: bool
    error: str | None = None


@dataclass
class DiscardedConversation:
    conversation: int
    reason: str
    excluded_fact_ids: list[str]
    dialog: Dialog
    judgements: list[TurnJudgement] = field(default_factory=list)


@dataclass
class Orchestration:
    """Everything the orchestrator produced so far; filled in place so a failed run can still be
    archived."""

    conversation: int = 0
    offer: Offer | None = None
    dialog: Dialog | None = None
    judgements: list[TurnJudgement] = field(default_factory=list)
    used_fact_ids: list[str] = field(default_factory=list)
    regenerations: list[Regeneration] = field(default_factory=list)
    cascade_checks: list[CascadeCheck] = field(default_factory=list)
    discarded: list[DiscardedConversation] = field(default_factory=list)

    def dialog_dict(self) -> dict:
        assert self.dialog is not None
        return {
            **self.dialog.to_dict(),
            "conversation": self.conversation,
            "regenerations": [asdict(r) for r in self.regenerations],
            "discarded_conversations": [asdict(d) for d in self.discarded],
        }

    def judgement_dict(self) -> dict:
        return {
            "judgements": [j.to_dict() for j in self.judgements],
            "used_fact_ids": self.used_fact_ids,
            "cascade_checks": [asdict(c) for c in self.cascade_checks],
        }


# ---------------------------------------------------------------- pure functions


def rejected_fact_ids(turn: Turn, judgement: TurnJudgement, t: float) -> list[str]:
    """Facts cited by the rejected statements of a turn (D6).

    A Jev rejection names the statements (p <= 1 - t). If none is named (completeness question)
    or the LLM judge rejected the turn, every fact the turn cites counts. A gate problem rejects
    no statement.
    """
    if judgement.accepted or judgement.decided_by == "gate":
        return []
    if judgement.decided_by == "jev":
        ids: list[str] = []
        for n, statement in enumerate(turn.statements):
            if judgement.jev_p.get(f"aussage_{n}", 1.0) <= 1 - t:
                ids += [i for i in statement.fact_ids if i not in ids]
        if ids:
            return ids
    return cited_ids(turn)


def cascade_request(before: Turn, after: Turn) -> tuple[dict, dict[str, dict]]:
    """Jev state and question: does the regenerated turn still convey the earlier statements?"""
    state = {
        "vorher": {"text": before.text, "aussagen": [s.text for s in before.statements]},
        "nachher": {"text": after.text},
    }
    question = {
        "type": "noul",
        "instructions": "Kommt jede Aussage aus `vorher.aussagen` in `nachher.text` vor?",
        "criteria": {
            "true": "Jede Aussage aus `vorher.aussagen` steht auch in `nachher.text`, "
            "gegebenenfalls anders formuliert.",
            "false": "Mindestens eine Aussage aus `vorher.aussagen` fehlt in `nachher.text` "
            "oder wird dort anders wiedergegeben.",
        },
    }
    return state, {CASCADE_KEY: question}


def cascade_confirmed(p: float, t: float) -> bool:
    """D1: only p >= t stops the cascade; uncertain counts as content lost."""
    return decide([p], t) is Decision.SUPPORTED


# ---------------------------------------------------------------- turn level


@dataclass
class RepairResult:
    exhausted: bool
    rejected_fact_ids: list[str]


def evaluate(
    turn: Turn,
    previous: Turn | None,
    offer: Offer,
    jev: Jev,
    judge_llm: StructuredLlm,
    t: float,
) -> TurnJudgement:
    """Gate problems reject a turn without a judge call; otherwise the judge decides.

    A turn the judge accepts then passes the markup gate (design flowchart: judge, then "kein
    Markup im Skript"); markup rejects it as a gate problem, so no fact counts as rejected (D6).
    """
    facts = offer.by_id()
    problems = check_turns([turn], facts.keys())
    if problems:
        return TurnJudgement(turn.index, False, "gate", reason=" ".join(p.reason for p in problems))
    judgement = judge_turn(jev, judge_llm, turn, previous, facts, t)
    markup = markup_problems([turn])
    if judgement.accepted and markup:
        return TurnJudgement(
            turn.index,
            False,
            "gate",
            judgement.jev_p,
            judgement.jev_error,
            " ".join(p.reason for p in markup),
        )
    return judgement


def check_cascade(jev: Jev, before: Turn, after: Turn, t: float, conversation: int) -> CascadeCheck:
    """Jev abort errors propagate; overlength and transient errors leave the cascade running."""
    state, questions = cascade_request(before, after)
    try:
        p = jev.ask(state, questions)[CASCADE_KEY]
    except (JevOverlengthError, JevTransientError) as exc:
        return CascadeCheck(conversation, after.index, None, False, f"{type(exc).__name__}: {exc}")
    return CascadeCheck(conversation, after.index, p, cascade_confirmed(p, t))


def repair_turns(
    state: Orchestration,
    turns: list[Turn],
    judgements: list[TurnJudgement],
    author: StructuredLlm,
    judge_llm: StructuredLlm,
    jev: Jev,
    t: float,
    turn_retries: int,
) -> RepairResult:
    """Regenerate rejected turns in place, with the cascade of D1. Stops when a budget is empty."""
    offer = state.offer
    assert offer is not None
    offer_text = format_offer(offer.main, offer.linked)
    rejected: list[str] = []

    def note(turn: Turn, judgement: TurnJudgement) -> None:
        rejected.extend(i for i in rejected_fact_ids(turn, judgement, t) if i not in rejected)

    for start in range(len(turns)):
        if judgements[start].accepted:  # also true when an earlier cascade fixed it
            continue
        note(turns[start], judgements[start])
        budget = turn_retries
        pos, kind, reason = start, "retry", judgements[start].reason
        baseline = turns[start]
        while True:
            if budget == 0:
                return RepairResult(True, rejected)
            budget -= 1
            old = turns[pos]
            new = regenerate_turn(author, offer_text, turns, pos, reason)
            state.regenerations.append(Regeneration(state.conversation, pos, kind, reason, old))
            turns[pos] = new
            judgement = evaluate(new, turns[pos - 1] if pos else None, offer, jev, judge_llm, t)
            judgements[pos] = judgement
            if not judgement.accepted:
                note(new, judgement)
                reason = judgement.reason
                continue
            if kind == "cascade":
                check = check_cascade(jev, baseline, new, t, state.conversation)
                state.cascade_checks.append(check)
                if check.confirmed:
                    break
            if pos + 1 >= len(turns):
                break
            pos, kind, reason = pos + 1, "cascade", None
            baseline = turns[pos]
    return RepairResult(False, rejected)


# ---------------------------------------------------------------- conversation level


def orchestrate(
    state: Orchestration,
    make_offer: Callable[[frozenset[str]], Offer],
    author: StructuredLlm,
    judge_llm: StructuredLlm,
    jev: Jev,
    config: Mapping,
) -> Orchestration:
    """Dialog, judgement, turn and conversation regeneration. Raises `GateFailure`.

    `make_offer(excluded)` reruns both rankings without the excluded facts and may itself raise
    `GateFailure` (offer gate, D5).
    """
    t = config["judge"]["t"]
    quotas = config["quotas"]
    minimum = quotas["min_used_facts"]
    excluded: set[str] = set()
    last_reason = ""
    try:
        for conversation in range(1, quotas["conversation_retries"] + 2):
            state.conversation = conversation
            state.offer = make_offer(frozenset(excluded))
            dialog = write_dialog(author, state.offer.main, state.offer.linked, minimum)
            state.dialog, state.judgements, state.used_fact_ids = dialog, [], []
            failure = used_facts_gate(len(dialog.used_fact_ids), minimum)
            judgements: list[TurnJudgement] = []
            rejected: list[str] = []
            if not failure:
                judgements = [
                    evaluate(
                        turn,
                        dialog.turns[n - 1] if n else None,
                        state.offer,
                        jev,
                        judge_llm,
                        t,
                    )
                    for n, turn in enumerate(dialog.turns)
                ]
                for turn, judgement in zip(dialog.turns, judgements, strict=True):
                    rejected += [
                        i for i in rejected_fact_ids(turn, judgement, t) if i not in rejected
                    ]
                state.judgements = judgements
                result = repair_turns(
                    state,
                    dialog.turns,
                    judgements,
                    author,
                    judge_llm,
                    jev,
                    t,
                    quotas["turn_retries"],
                )
                rejected += [i for i in result.rejected_fact_ids if i not in rejected]
                if result.exhausted:
                    failure = (
                        "Turn regeneration budget used up "
                        f"({quotas['turn_retries']}x, cascade included)."
                    )
            if failure:
                excluded.update(rejected)
                last_reason = failure
                if conversation <= quotas["conversation_retries"]:
                    state.discarded.append(
                        DiscardedConversation(
                            conversation, failure, sorted(excluded), dialog, list(judgements)
                        )
                    )
                    continue
                break
            dialog.problems = []
            facts = state.offer.by_id()
            used = used_in_accepted(dialog.turns, state.judgements, facts.keys())
            dialog.used_fact_ids = used
            state.used_fact_ids = used
            final = judged_facts_gate(len(used), minimum)
            if final:
                raise GateFailure(final)
            return state
    except JevAbortError as exc:
        raise GateFailure(f"Jev aborted the run: {exc}") from exc
    raise GateFailure(
        f"Conversation regeneration budget used up ({quotas['conversation_retries']}x): "
        f"{last_reason}"
    )


def summary_lines(state: Orchestration) -> Sequence[str]:
    """Short facts about the orchestration for the job summary."""
    return [
        f"Conversations written: {state.conversation}",
        f"Turn regenerations: {len(state.regenerations)}",
        f"Cascade checks: {len(state.cascade_checks)}",
    ]
