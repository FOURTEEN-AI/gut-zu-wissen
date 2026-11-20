"""Orchestrator (phase 1, step 8; design: "Fehlerbehandlung", decisions D1, D5, D6).

Turn level: a rejected turn (gate problem or judge) is regenerated with its neighbours as context.
After every regeneration the next turn is regenerated as well (neighbour cascade, D1); a
two-stage check decides whether the regenerated turn still conveys the statements of the version
before (user decision 2026-11-20). Jev asks first: p >= t stops the cascade, p <= 1 - t continues
it. Uncertain p, or an overlength or transient Jev error, goes to the LLM judge, which sees only
the earlier text and statements and the new text; anything but a clear "same statements" counts
as content lost and continues the cascade. Jev abort errors and failed LLM calls (`LlmError`)
propagate.

Budgets (D1, changed 2026-11-20): a chain (the rejected turn, its retries and its cascade) has a
judge budget of `turn_retries` and a separate gate budget of `gate_retries`. A retry charges the
type of the rejection it fixes (decided by "gate": gate budget; by Jev or the LLM judge: judge
budget); a cascade step charges the type of the rejection that started the chain. Each
regeneration records the budget it charged.

The repetition gate looks at the whole current dialog: a fact cited by too many turns rejects the
later turns, never the first narration, so a regenerated turn that creates a repetition with a
later turn rejects that later turn.

Conversation level (D6, changed 2026-11-20), at most `conversation_retries` times: facts cited by
rejected statements are excluded, cumulatively (pinned facts, the first prose sentence of the
intro as Einordnung #1, never). An exhausted turn budget keeps turns 0..k-1 with their judgements,
k being the turn that started the failing chain, and the author continues the dialog from turn k
to the end on the same offer without the excluded facts (no re-ranking; facts cited by the kept
turns stay). Only the new turns are judged; the repetition gate and the D5 gates look at the
whole dialog. A fresh conversation (both rankings rerun, a new dialog) is written instead when
k is 0, when the remaining offer cannot reach `min_used_facts`, or after fewer than
`min_used_facts` used facts after the author (D5). The discarded part (the tail k..end, or the
whole dialog) is archived with the path taken. Everything else ends in `GateFailure`, the
escalation to the owner.

Every judgement is archived (user decision 2026-11-19): the final ones in `judgements`, those of
discarded conversations with them, and those of turn versions a regeneration replaced in
`superseded_judgements`, each with its conversation, turn index, turn version (0: as the author
wrote it, n: after the n-th regeneration of that turn in that conversation) and the index of the
regeneration in `05_dialog.json` that replaced it.

Deterministic parts are pure functions; the LLM and Jev calls sit behind their interfaces.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace

from gut_zu_wissen.dialog import (
    Dialog,
    Turn,
    TurnProblem,
    check_turns,
    cited,
    continue_dialog,
    dialog_problems,
    format_offer,
    offer_aliases,
    regenerate_turn,
    repetition_problems,
    used_fact_ids,
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

CONTINUATION = (
    "continuation"  # conversation regeneration from the failing chain on (D6, 2026-11-20)
)
FRESH = "fresh"  # conversation regeneration with fresh rankings and a new dialog (D6)

JUDGE_BUDGET = "judge"  # D1: judge rejections, `turn_retries`
GATE_BUDGET = "gate"  # D1 (2026-11-20): gate rejections, `gate_retries`

CASCADE_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"same": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["same", "reason"],
    "additionalProperties": False,
}
CASCADE_PROMPT = (
    "Ein Redebeitrag aus einem Podcast-Gespräch wurde neu geschrieben. Du bekommst die vorherige "
    "Fassung mit ihren Aussagen und die neue Fassung. Prüfe: Vermittelt die neue Fassung dieselben "
    "Aussagen wie die vorherige?\n\n"
    "„same“ ist nur wahr, wenn jede Aussage der vorherigen Fassung auch in der neuen Fassung "
    "steht, gegebenenfalls anders formuliert, mit gleichen Zahlen, Daten und Namen. Fehlt eine "
    "Aussage, wird sie anders wiedergegeben oder bist du unsicher, ist „same“ falsch. Begründe "
    "kurz auf Deutsch; nenne bei „falsch“ die betroffene Aussage. Antworte mit JSON im "
    "vorgegebenen Schema."
)


class GateFailure(Exception):
    """A gate or the retry budgets stopped the run; the message is the notice for the owner."""


@dataclass
class Offer:
    """The facts offered to the dialog author; `einordnung` are the orientation facts, offered
    first and outside the ranking."""

    main: list[RankedFact]
    linked: list[RankedFact]
    einordnung: list[RankedFact] = field(default_factory=list)
    # never excluded by a conversation regeneration (D6): the first prose sentence of the intro
    # as Einordnung #1 (user decision 2026-11-20)
    pinned: frozenset[str] = frozenset()

    def by_id(self) -> dict[str, RankedFact]:
        return {r.fact.segment_id: r for r in [*self.einordnung, *self.main, *self.linked]}

    def aliases(self) -> dict[str, str]:
        return offer_aliases(self.main, self.linked, self.einordnung)

    def text(self) -> str:
        return format_offer(self.main, self.linked, self.einordnung)


@dataclass
class Regeneration:
    """One regenerated turn; `previous` is the version that was replaced. Field order is the
    archive field order (new fields appended)."""

    conversation: int
    turn_index: int
    kind: str  # "retry" after a rejection, "cascade" for a neighbour
    reason: str | None
    previous: Turn
    budget: str = JUDGE_BUDGET  # the budget this regeneration charged: "judge" or "gate"


@dataclass
class CascadeCheck:
    """One cascade check; field order is the archive field order (new fields appended)."""

    conversation: int
    turn_index: int
    p: float | None  # Jev; None after an overlength or transient Jev error
    confirmed: bool
    error: str | None
    decided_by: str  # "jev" or "llm_judge"
    reason: str | None  # the LLM judge's reason; None when Jev decided


@dataclass
class SupersededJudgement:
    """The judgement of a turn version that a regeneration replaced; field order is the archive
    field order."""

    conversation: int
    turn_index: int
    version: int  # 0: as written by the author; n: after the n-th regeneration of this turn
    replaced_by: int  # index of the replacing regeneration in `regenerations` (05_dialog.json)
    judgement: TurnJudgement


@dataclass
class DiscardedConversation:
    conversation: int
    reason: str
    excluded_fact_ids: list[str]
    dialog: Dialog
    judgements: list[TurnJudgement] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)  # what the author saw, for tracing
    # How the next conversation was written (user decision 2026-11-20): "continuation" from turn
    # `continued_from` (k) of this one, which kept turns 0..k-1 with their judgements, so
    # `dialog` and `judgements` hold only the discarded tail k..end; or "fresh" (D6: both
    # rankings rerun, a new dialog), with the reason why no continuation was possible.
    regeneration: str = FRESH
    continued_from: int | None = None
    fresh_reason: str | None = None


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
    superseded: list[SupersededJudgement] = field(default_factory=list)

    def dialog_dict(self) -> dict:
        """`aliases` (last) maps the identifiers the author saw for the final offer to segment
        IDs; it is for tracing only, every other field holds segment IDs."""
        assert self.dialog is not None
        offer = self.offer
        return {
            **self.dialog.to_dict(),
            "conversation": self.conversation,
            "regenerations": [asdict(r) for r in self.regenerations],
            "discarded_conversations": [asdict(d) for d in self.discarded],
            "aliases": offer.aliases() if offer else {},
        }

    def judgement_dict(self) -> dict:
        return {
            "judgements": [j.to_dict() for j in self.judgements],
            "used_fact_ids": self.used_fact_ids,
            "cascade_checks": [asdict(c) for c in self.cascade_checks],
            "conversation": self.conversation,
            "superseded_judgements": [
                {**asdict(s), "judgement": s.judgement.to_dict()} for s in self.superseded
            ],
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


def turn_version(regenerations: Sequence[Regeneration], conversation: int, index: int) -> int:
    """Version of the current turn at `index`: how often it was regenerated in `conversation`."""
    return sum(1 for r in regenerations if r.conversation == conversation and r.turn_index == index)


def cascade_decision(p: float, t: float) -> Decision:
    """D1, stage 1: p >= t confirmed; p <= 1 - t content lost; otherwise the LLM judge decides."""
    return decide([p], t)


def cascade_verdict_confirmed(verdict: Mapping) -> bool:
    """D1, stage 2: only a clear `same: true` confirms; anything else counts as content lost."""
    return verdict.get("same") is True


def format_cascade_input(before: Turn, after: Turn) -> str:
    """Input of the LLM judge for a cascade check: earlier text and statements, new text."""
    lines = ["Vorherige Fassung:", before.text, "", "Aussagen der vorherigen Fassung:"]
    lines += [f"- {s.text}" for s in before.statements] or ["(keine)"]
    lines += ["", "Neue Fassung:", after.text]
    return "\n".join(lines)


# ---------------------------------------------------------------- retry budgets (D1)


def budget_type(judgement: TurnJudgement) -> str:
    """The budget a regeneration after this rejection charges: gate problems (decided without a
    judge call) the gate budget, judge rejections the judge budget."""
    return GATE_BUDGET if judgement.decided_by == "gate" else JUDGE_BUDGET


@dataclass(frozen=True)
class ChainBudget:
    """Retry budgets of one chain: a rejected turn, its retries and its neighbour cascade (D1).

    `trigger` is the budget type of the rejection that started the chain; cascade steps charge
    it. `charge` is the budget type the next regeneration charges.
    """

    judge: int
    gate: int
    trigger: str
    charge: str


def chain_start(rejection: str, turn_retries: int, gate_retries: int) -> ChainBudget:
    """A new chain after the rejection of type `rejection`: both budgets full, the first
    regeneration (the retry) charges that rejection's type."""
    return ChainBudget(turn_retries, gate_retries, rejection, rejection)


def after_rejection(budget: ChainBudget, rejection: str) -> ChainBudget:
    """The regenerated turn was rejected again: its retry charges this rejection's type."""
    return ChainBudget(budget.judge, budget.gate, budget.trigger, rejection)


def cascade_step(budget: ChainBudget) -> ChainBudget:
    """The next regeneration is a cascade step: it charges the type of the triggering
    rejection."""
    return ChainBudget(budget.judge, budget.gate, budget.trigger, budget.trigger)


def spend(budget: ChainBudget) -> ChainBudget | None:
    """One regeneration charged to `budget.charge`; None when that budget is used up."""
    if budget.charge == GATE_BUDGET:
        return None if budget.gate == 0 else replace(budget, gate=budget.gate - 1)
    return None if budget.judge == 0 else replace(budget, judge=budget.judge - 1)


def exhausted_reason(charge: str, turn_retries: int, gate_retries: int) -> str:
    """Archived reason of a used-up turn budget."""
    if charge == GATE_BUDGET:
        return f"Turn regeneration budget used up (gate problems, {gate_retries}x)."
    return f"Turn regeneration budget used up (judge, {turn_retries}x)."


# ---------------------------------------------------------------- conversation regeneration


def continuation_offer(offer: Offer, excluded: Collection[str], prefix: Sequence[Turn]) -> Offer:
    """The offer of a continuation: the same offer without the excluded facts (D6), no
    re-ranking. Facts cited by the kept prefix and pinned facts stay, so the prefix keeps its
    identifiers and its facts count for D5; the author is told they are already told."""
    keep = {i for turn in prefix for i in cited(turn)} | offer.pinned

    def kept(facts: Sequence[RankedFact]) -> list[RankedFact]:
        return [r for r in facts if r.fact.segment_id in keep or r.fact.segment_id not in excluded]

    return Offer(kept(offer.main), kept(offer.linked), kept(offer.einordnung), offer.pinned)


def conversation_path(start: int | None, offered: int, minimum: int) -> tuple[str, str | None]:
    """How a failed conversation is regenerated (user decision 2026-11-20), and why not as a
    continuation. Continuation from turn `start` (the turn that started the failing chain), unless
    there is no failing chain (D5 after the author), it started at turn 0, or the remaining offer
    (`offered` facts) cannot reach the D5 minimum; then fresh rankings and a new dialog."""
    if start is None:
        return FRESH, "No failing turn chain to continue from."
    if start == 0:
        return FRESH, "The failing turn chain started at turn 0."
    if offered < minimum:
        return FRESH, (
            f"The remaining offer has {offered} facts, fewer than the D5 minimum of {minimum}."
        )
    return CONTINUATION, None


def discarded_tail(dialog: Dialog, start: int, offered_ids: Collection[str]) -> Dialog:
    """The part of a failed dialog that is discarded: turns `start`..end with their gate problems
    and used facts; the whole dialog for `start` 0."""
    if start == 0:
        return dialog
    tail = dialog.turns[start:]
    return Dialog(
        tail,
        [p for p in dialog.problems if p.turn_index >= start],
        used_fact_ids(tail, offered_ids),
    )


# ---------------------------------------------------------------- turn level


@dataclass
class RepairResult:
    exhausted: bool
    rejected_fact_ids: list[str]
    start: int | None = None  # exhausted: the turn that started the failing chain (k)
    budget: str | None = None  # exhausted: the budget type that was used up


def turn_gate_problems(
    turns: Sequence[Turn], index: int, offered_ids: Collection[str], max_turns_per_fact: int
) -> list[TurnProblem]:
    """Gate problems of turn `index` before the judge: the fact-ID gate on the turn alone, the
    repetition gate in the context of the whole current dialog."""
    repeated = repetition_problems(turns, offered_ids, max_turns_per_fact)
    return [
        *check_turns([turns[index]], offered_ids),
        *(p for p in repeated if p.turn_index == index),
    ]


def later_repetitions(
    turns: Sequence[Turn],
    judgements: Sequence[TurnJudgement],
    pos: int,
    offered_ids: Collection[str],
    max_turns_per_fact: int,
) -> list[TurnJudgement]:
    """After turn `pos` changed: accepted later turns that now repeat a fact, rejected as a gate
    problem (Jev values kept, as for the markup gate). Turns before `pos` cannot be affected."""
    reasons: dict[int, list[str]] = {}
    for p in repetition_problems(turns, offered_ids, max_turns_per_fact):
        if p.turn_index > pos and judgements[p.turn_index].accepted:
            reasons.setdefault(p.turn_index, []).append(p.reason)
    return [
        TurnJudgement(
            index,
            False,
            "gate",
            judgements[index].jev_p,
            judgements[index].jev_error,
            " ".join(texts),
        )
        for index, texts in reasons.items()
    ]


def evaluate(
    turns: Sequence[Turn],
    index: int,
    offer: Offer,
    jev: Jev,
    judge_llm: StructuredLlm,
    t: float,
    max_turns_per_fact: int,
) -> TurnJudgement:
    """Gate problems reject turn `index` without a judge call; otherwise the judge decides.

    A turn the judge accepts then passes the markup gate (design flowchart: judge, then "kein
    Markup im Skript"); markup rejects it as a gate problem, so no fact counts as rejected (D6).
    """
    turn = turns[index]
    previous = turns[index - 1] if index else None
    facts = offer.by_id()
    problems = turn_gate_problems(turns, index, facts.keys(), max_turns_per_fact)
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


def check_cascade(
    jev: Jev,
    judge_llm: StructuredLlm,
    before: Turn,
    after: Turn,
    t: float,
    conversation: int,
) -> CascadeCheck:
    """Two-stage check of D1. Jev abort errors and `LlmError` propagate."""
    state, questions = cascade_request(before, after)
    p: float | None = None
    error: str | None = None
    try:
        p = jev.ask(state, questions)[CASCADE_KEY]
    except (JevOverlengthError, JevTransientError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    if p is not None:
        decision = cascade_decision(p, t)
        if decision is not Decision.UNCERTAIN:
            confirmed = decision is Decision.SUPPORTED
            return CascadeCheck(conversation, after.index, p, confirmed, None, "jev", None)
    verdict = judge_llm.complete_json(
        CASCADE_PROMPT, format_cascade_input(before, after), CASCADE_VERDICT_SCHEMA
    )
    return CascadeCheck(
        conversation,
        after.index,
        p,
        cascade_verdict_confirmed(verdict),
        error,
        "llm_judge",
        str(verdict.get("reason", "")),
    )


def repair_turns(
    state: Orchestration,
    turns: list[Turn],
    judgements: list[TurnJudgement],
    author: StructuredLlm,
    judge_llm: StructuredLlm,
    jev: Jev,
    t: float,
    turn_retries: int,
    gate_retries: int,
    max_turns_per_fact: int,
) -> RepairResult:
    """Regenerate rejected turns in place, with the cascade of D1. Stops when a budget is empty.

    Every chain (a rejected turn, its retries and its cascade) has a judge budget
    (`turn_retries`) and a gate budget (`gate_retries`, user decision 2026-11-20): a retry
    charges the type of the rejection it fixes, a cascade step the type of the rejection that
    started the chain (`ChainBudget`).

    A regenerated turn can make a later, accepted turn repeat a fact; that turn is rejected as a
    gate problem right away and repaired with its own budget when the loop reaches it.
    """
    offer = state.offer
    assert offer is not None
    offered = offer.by_id().keys()
    offer_text = offer.text()
    aliases = offer.aliases()  # rebuilt for every offer
    rejected: list[str] = []

    def note(turn: Turn, judgement: TurnJudgement) -> None:
        rejected.extend(i for i in rejected_fact_ids(turn, judgement, t) if i not in rejected)

    for start in range(len(turns)):
        if judgements[start].accepted:  # also true when an earlier cascade fixed it
            continue
        note(turns[start], judgements[start])
        budget = chain_start(budget_type(judgements[start]), turn_retries, gate_retries)
        pos, kind, reason = start, "retry", judgements[start].reason
        baseline = turns[start]
        while True:
            charged = budget.charge
            remaining = spend(budget)
            if remaining is None:
                return RepairResult(True, rejected, start, charged)
            budget = remaining
            old = turns[pos]
            new = regenerate_turn(author, offer_text, aliases, turns, pos, reason)
            version = turn_version(state.regenerations, state.conversation, pos)
            state.superseded.append(
                SupersededJudgement(
                    state.conversation, pos, version, len(state.regenerations), judgements[pos]
                )
            )
            state.regenerations.append(
                Regeneration(state.conversation, pos, kind, reason, old, charged)
            )
            turns[pos] = new
            judgement = evaluate(turns, pos, offer, jev, judge_llm, t, max_turns_per_fact)
            judgements[pos] = judgement
            for later in later_repetitions(turns, judgements, pos, offered, max_turns_per_fact):
                judgements[later.turn_index] = later
            if not judgement.accepted:
                note(new, judgement)
                reason = judgement.reason
                budget = after_rejection(budget, budget_type(judgement))
                continue
            if kind == "cascade":
                check = check_cascade(jev, judge_llm, baseline, new, t, state.conversation)
                state.cascade_checks.append(check)
                if check.confirmed:
                    break
            if pos + 1 >= len(turns):
                break
            pos, kind, reason = pos + 1, "cascade", None
            budget = cascade_step(budget)
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
    `GateFailure` (offer gate, D5). A conversation regeneration continues the dialog from the
    turn that started the failing chain when it can (`conversation_path`); only the new turns are
    judged and gated, the repetition gate and the D5 gates look at the whole dialog.
    """
    t = config["judge"]["t"]
    quotas = config["quotas"]
    minimum = quotas["min_used_facts"]
    max_turns_per_fact = quotas["max_turns_per_fact"]
    excluded: set[str] = set()
    last_reason = ""
    prefix: list[Turn] = []  # kept turns of a continuation, with their judgements
    prefix_judgements: list[TurnJudgement] = []
    try:
        for conversation in range(1, quotas["conversation_retries"] + 2):
            state.conversation = conversation
            if prefix:
                offer = state.offer
                assert offer is not None
                new = continue_dialog(author, offer.text(), offer.aliases(), prefix, minimum)
                turns = [*prefix, *new]
                offered = offer.by_id().keys()
                dialog = Dialog(
                    turns,
                    dialog_problems(turns, offered, max_turns_per_fact),
                    used_fact_ids(turns, offered),
                )
            else:
                offer = state.offer = make_offer(frozenset(excluded))
                dialog = write_dialog(
                    author, offer.main, offer.linked, minimum, max_turns_per_fact, offer.einordnung
                )
            kept = len(prefix)
            state.dialog, state.judgements, state.used_fact_ids = dialog, [], []
            failure = used_facts_gate(len(dialog.used_fact_ids), minimum)
            failed_chain: int | None = None
            judgements: list[TurnJudgement] = []
            rejected: list[str] = []
            if not failure:
                judgements = [
                    *prefix_judgements,
                    *(
                        evaluate(dialog.turns, n, offer, jev, judge_llm, t, max_turns_per_fact)
                        for n in range(kept, len(dialog.turns))
                    ),
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
                    quotas["gate_retries"],
                    max_turns_per_fact,
                )
                rejected += [i for i in result.rejected_fact_ids if i not in rejected]
                if result.exhausted:
                    assert result.budget is not None
                    failure = exhausted_reason(
                        result.budget, quotas["turn_retries"], quotas["gate_retries"]
                    )
                    failed_chain = result.start
            if failure:
                excluded.update(i for i in rejected if i not in offer.pinned)
                last_reason = failure
                if conversation <= quotas["conversation_retries"]:
                    k = failed_chain or 0
                    next_offer = continuation_offer(offer, excluded, dialog.turns[:k])
                    path, fresh_reason = conversation_path(
                        failed_chain, len(next_offer.by_id()), minimum
                    )
                    cut = k if path == CONTINUATION else 0
                    state.discarded.append(
                        DiscardedConversation(
                            conversation,
                            failure,
                            sorted(excluded),
                            discarded_tail(dialog, cut, offer.by_id().keys()),
                            list(judgements[cut:]),
                            offer.aliases(),
                            path,
                            cut if path == CONTINUATION else None,
                            fresh_reason,
                        )
                    )
                    if path == CONTINUATION:
                        state.offer = next_offer
                        prefix, prefix_judgements = dialog.turns[:k], judgements[:k]
                    else:
                        prefix, prefix_judgements = [], []
                    continue
                break
            dialog.problems = []
            facts = offer.by_id()
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
    paths = [d.regeneration for d in state.discarded]
    return [
        f"Conversations written: {state.conversation} (continued: {paths.count(CONTINUATION)}, "
        f"fresh: {paths.count(FRESH)})",
        f"Turn regenerations: {len(state.regenerations)}",
        f"Cascade checks: {len(state.cascade_checks)} "
        f"(LLM judge: {sum(c.decided_by == 'llm_judge' for c in state.cascade_checks)})",
    ]
