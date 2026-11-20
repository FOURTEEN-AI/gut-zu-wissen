from dataclasses import asdict

import pytest

from gut_zu_wissen.dialog import CONTINUE_PROMPT, Dialog, Statement, Turn, repetition_reason
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.jev import JevAbortError, JevOverlengthError, JevTransientError
from gut_zu_wissen.judge import Decision, TurnJudgement
from gut_zu_wissen.llm import LlmError
from gut_zu_wissen.orchestrator import (
    CASCADE_KEY,
    CASCADE_PROMPT,
    CASCADE_VERDICT_SCHEMA,
    GATE_BUDGET,
    JUDGE_BUDGET,
    CascadeCheck,
    ChainBudget,
    GateFailure,
    Offer,
    Orchestration,
    Rejection,
    after_rejection,
    budget_type,
    cascade_decision,
    cascade_request,
    cascade_step,
    cascade_verdict_confirmed,
    chain_start,
    check_cascade,
    continuation_offer,
    continuation_rejections,
    conversation_path,
    discarded_tail,
    format_cascade_input,
    orchestrate,
    rejected_fact_ids,
    spend,
    summary_lines,
)
from gut_zu_wissen.ranking import RankedFact

T = 0.95
CONFIG = {
    "judge": {"t": T},
    "quotas": {
        "min_used_facts": 1,
        "max_turns_per_fact": 2,
        "turn_retries": 2,
        "gate_retries": 3,
        "conversation_retries": 2,
    },
}


def rf(i: int) -> RankedFact:
    sid = f"1-{i:04d}"
    return RankedFact(i + 1, "Haupt", 1, Fact(sid, "sentence", "", [], f"Fakt {i}.", []))


def turn_json(speaker: str, text: str, fact: str | None) -> dict:
    ids = [fact] if fact else []
    return {
        "speaker": speaker,
        "text": text,
        "fact_ids": ids,
        "statements": [{"text": text, "fact_ids": ids}] if fact else [],
    }


def dialog_json(*texts: str, fact: str = "1-0000", facts: list[str] | None = None) -> dict:
    """Every turn cites `fact`, or turn n cites `facts[n]`."""
    cites = facts or [fact] * len(texts)
    return {
        "turns": [
            turn_json("Lena" if n % 2 == 0 else "Felix", text, cites[n])
            for n, text in enumerate(texts)
        ]
    }


# one fact per turn, so the repetition gate stays out of the way (the last turn is not counted)
DISTINCT = ["1-0000", "1-0001", "1-0002", "1-0000"]


class FakeAuthor:
    """Dialogs and single turns come from queues; every call is recorded."""

    def __init__(self, dialogs: list[dict], turns: list[dict] | None = None) -> None:
        self.dialogs, self.turns, self.calls = list(dialogs), list(turns or []), []
        self.systems: list[str] = []

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        self.calls.append(user)
        self.systems.append(system)
        return self.dialogs.pop(0) if "turns" in schema["properties"] else self.turns.pop(0)


class FakeJev:
    """Statement and completeness questions: p is 0.01 for a turn whose text is in `bad`, else
    0.99. The cascade question answers from `cascade`, one value per call."""

    def __init__(self, bad: set[str] = frozenset(), cascade: list[float] | None = None) -> None:
        self.bad, self.cascade, self.cascade_asked = bad, list(cascade or []), 0
        self.error: Exception | None = None

    def ask(self, state: dict, questions: dict) -> dict:
        if self.error:
            raise self.error
        if CASCADE_KEY in questions:
            self.cascade_asked += 1
            return {CASCADE_KEY: self.cascade.pop(0)}
        p = 0.01 if state["redebeitrag"]["text"] in self.bad else 0.99
        return {key: p for key in questions}


class FakeJudgeLlm:
    """Rejects every turn it judges. Cascade checks answer from `cascade`, one verdict per call;
    every cascade call is recorded."""

    def __init__(self, cascade: list[dict] | None = None) -> None:
        self.cascade, self.cascade_calls = list(cascade or []), []

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        if schema is CASCADE_VERDICT_SCHEMA:
            self.cascade_calls.append((system, user))
            return self.cascade.pop(0)
        return (
            {"supported": False, "reason": "Grund."}
            if "supported" in schema["properties"]
            else {"reason": "Grund."}
        )


def offer_of(n: int):
    calls: list[frozenset[str]] = []

    def make(excluded: frozenset[str]) -> Offer:
        calls.append(excluded)
        facts = [rf(i) for i in range(n) if f"1-{i:04d}" not in excluded]
        if not facts:
            raise GateFailure("Gate minimum content (D5): 0 facts offered.")
        return Offer(facts, [])

    return make, calls


def run(author, jev, n_facts=3, config=CONFIG, judge_llm=None):
    make, calls = offer_of(n_facts)
    state = Orchestration()
    try:
        orchestrate(state, make, author, judge_llm or FakeJudgeLlm(), jev, config)
    finally:
        run.state, run.offers = state, calls
    return state


# ---------------------------------------------------------------- pure functions


def judgement(accepted=False, by="jev", p=None) -> TurnJudgement:
    return TurnJudgement(0, accepted, by, p or {})


def two_statement_turn() -> Turn:
    return Turn(0, "Lena", "T", ["a", "b"], [Statement("S0", ["a"]), Statement("S1", ["b", "c"])])


def test_rejected_facts_are_those_of_the_rejected_statements():
    j = judgement(p={"aussage_0": 0.99, "aussage_1": 0.02, "vollstaendig": 0.99})
    assert rejected_fact_ids(two_statement_turn(), j, T) == ["b", "c"]


def test_rejected_facts_fall_back_to_all_cited_facts():
    turn = two_statement_turn()
    only_completeness = judgement(p={"aussage_0": 0.99, "aussage_1": 0.99, "vollstaendig": 0.02})
    assert rejected_fact_ids(turn, only_completeness, T) == ["a", "b", "c"]
    assert rejected_fact_ids(turn, judgement(by="llm_judge"), T) == ["a", "b", "c"]


def test_accepted_and_gate_problem_turns_reject_no_facts():
    turn = two_statement_turn()
    assert rejected_fact_ids(turn, judgement(accepted=True), T) == []
    assert rejected_fact_ids(turn, judgement(by="gate"), T) == []


@pytest.mark.parametrize(
    "p,expected",
    [
        (0.95, Decision.SUPPORTED),
        (1.0, Decision.SUPPORTED),
        (0.9499, Decision.UNCERTAIN),
        (0.5, Decision.UNCERTAIN),
        (0.0501, Decision.UNCERTAIN),
        (0.05, Decision.REJECTED),
        (0.0, Decision.REJECTED),
    ],
)
def test_cascade_decision_at_t_at_one_minus_t_and_between(p, expected):
    assert cascade_decision(p, T) is expected


@pytest.mark.parametrize(
    "verdict,expected",
    [
        ({"same": True, "reason": "Gleich."}, True),
        ({"same": False, "reason": "Fehlt."}, False),
        ({"same": "true", "reason": ""}, False),  # malformed: content lost
        ({"same": None}, False),
        ({"reason": "Unklar."}, False),  # undecided: content lost
        ({}, False),
    ],
)
def test_llm_cascade_verdict_confirms_only_a_clear_yes(verdict, expected):
    assert cascade_verdict_confirmed(verdict) is expected


def test_cascade_input_has_earlier_text_and_statements_and_new_text_only():
    before = Turn(1, "Felix", "Alt.", ["a"], [Statement("Aussage alt.", ["a"])])
    after = Turn(1, "Felix", "Neu.", ["b"], [Statement("Aussage neu.", ["b"])])
    text = format_cascade_input(before, after)
    assert text == (
        "Vorherige Fassung:\nAlt.\n\nAussagen der vorherigen Fassung:\n- Aussage alt.\n\n"
        "Neue Fassung:\nNeu."
    )
    empty = format_cascade_input(Turn(1, "Felix", "Alt.", [], []), after)
    assert "(keine)" in empty


def test_cascade_prompt_is_german_and_asks_for_the_same_statements():
    assert "dieselben Aussagen" in CASCADE_PROMPT
    assert "unsicher" in CASCADE_PROMPT
    assert CASCADE_VERDICT_SCHEMA["required"] == ["same", "reason"]


BEFORE = Turn(2, "Lena", "Alt.", ["a"], [Statement("Aussage alt.", ["a"])])
AFTER = Turn(2, "Lena", "Neu.", ["a"], [Statement("Aussage neu.", ["a"])])


class CascadeJev:
    def __init__(self, p: float | None = None, error: Exception | None = None) -> None:
        self.p, self.error = p, error

    def ask(self, state: dict, questions: dict) -> dict:
        if self.error:
            raise self.error
        return {CASCADE_KEY: self.p}


@pytest.mark.parametrize("p,confirmed", [(0.99, True), (0.01, False)])
def test_cascade_check_jev_decides_clear_cases_without_llm(p, confirmed):
    llm = FakeJudgeLlm()
    check = check_cascade(CascadeJev(p), llm, BEFORE, AFTER, T, 1)
    assert check == CascadeCheck(1, 2, p, confirmed, None, "jev", None)
    assert llm.cascade_calls == []


@pytest.mark.parametrize(
    "verdict,confirmed",
    [
        ({"same": True, "reason": "Gleiche Aussagen."}, True),
        ({"same": False, "reason": "Aussage alt fehlt."}, False),
    ],
)
def test_cascade_check_uncertain_jev_goes_to_the_llm_judge(verdict, confirmed):
    llm = FakeJudgeLlm([verdict])
    check = check_cascade(CascadeJev(0.74), llm, BEFORE, AFTER, T, 1)
    assert check == CascadeCheck(1, 2, 0.74, confirmed, None, "llm_judge", verdict["reason"])
    assert llm.cascade_calls == [(CASCADE_PROMPT, format_cascade_input(BEFORE, AFTER))]


def test_cascade_check_malformed_llm_answer_counts_as_content_lost():
    check = check_cascade(CascadeJev(0.5), FakeJudgeLlm([{}]), BEFORE, AFTER, T, 1)
    assert (check.confirmed, check.decided_by, check.reason) == (False, "llm_judge", "")


@pytest.mark.parametrize("error", [JevTransientError("429"), JevOverlengthError("400")])
def test_cascade_check_jev_error_goes_to_the_llm_judge(error):
    llm = FakeJudgeLlm([{"same": True, "reason": "Gleich."}])
    check = check_cascade(CascadeJev(error=error), llm, BEFORE, AFTER, T, 3)
    assert (check.p, check.confirmed, check.decided_by) == (None, True, "llm_judge")
    assert type(error).__name__ in check.error


def test_cascade_check_jev_abort_and_llm_error_propagate():
    with pytest.raises(JevAbortError):
        check_cascade(CascadeJev(error=JevAbortError("401")), FakeJudgeLlm(), BEFORE, AFTER, T, 1)

    class Failing:
        def complete_json(self, system, user, schema, **cache):
            raise LlmError("refused")

    with pytest.raises(LlmError):
        check_cascade(CascadeJev(0.5), Failing(), BEFORE, AFTER, T, 1)


def test_cascade_check_field_order_is_stable():
    check = CascadeCheck(1, 2, 0.5, False, None, "llm_judge", "Grund.")
    assert list(asdict(check)) == [
        "conversation",
        "turn_index",
        "p",
        "confirmed",
        "error",
        "decided_by",
        "reason",
    ]


def test_cascade_request_names_old_statements_and_new_text():
    before = Turn(1, "Felix", "Alt.", ["a"], [Statement("Aussage alt.", ["a"])])
    after = Turn(1, "Felix", "Neu.", ["a"], [])
    state, questions = cascade_request(before, after)
    assert state == {
        "vorher": {"text": "Alt.", "aussagen": ["Aussage alt."]},
        "nachher": {"text": "Neu."},
    }
    assert list(questions) == [CASCADE_KEY]


# ---------------------------------------------------------------- retry budgets (D1)


def test_budget_type_is_gate_only_for_gate_rejections():
    assert budget_type(judgement(by="gate")) == GATE_BUDGET
    assert budget_type(judgement(by="jev")) == JUDGE_BUDGET
    assert budget_type(judgement(by="llm_judge")) == JUDGE_BUDGET


def spend_all(budget: ChainBudget) -> int:
    """How many regenerations the current charge type still allows."""
    n = 0
    while (after := spend(budget)) is not None:
        budget, n = after, n + 1
    return n


def test_chain_starts_full_and_its_retry_charges_the_rejection_type():
    for rejection in (GATE_BUDGET, JUDGE_BUDGET):
        budget = chain_start(rejection, 2, 3)
        assert budget == ChainBudget(2, 3, rejection, rejection)
    assert spend_all(chain_start(GATE_BUDGET, 2, 3)) == 3  # gate budget exhausted after 3
    assert spend_all(chain_start(JUDGE_BUDGET, 2, 3)) == 2  # judge budget exhausted after 2


def test_gate_then_judge_rejection_charges_each_its_own_budget():
    budget = spend(chain_start(GATE_BUDGET, 2, 3))
    assert (budget.judge, budget.gate) == (2, 2)
    budget = spend(after_rejection(budget, JUDGE_BUDGET))
    assert (budget.judge, budget.gate, budget.trigger) == (1, 2, GATE_BUDGET)


def test_judge_then_gate_rejection_charges_each_its_own_budget():
    budget = spend(chain_start(JUDGE_BUDGET, 2, 3))
    budget = spend(after_rejection(budget, GATE_BUDGET))
    assert (budget.judge, budget.gate, budget.trigger) == (1, 2, JUDGE_BUDGET)


def test_cascade_steps_charge_the_triggering_rejection_type():
    # after a gate retry the cascade runs on the gate budget, even with no judge budget at all
    budget = spend(chain_start(GATE_BUDGET, 0, 3))
    budget = cascade_step(budget)
    assert budget.charge == GATE_BUDGET
    budget = spend(budget)
    assert (budget.judge, budget.gate) == (0, 1)
    # a cascade step rejected by the judge: its retry charges the judge budget, the next cascade
    # step the trigger again
    budget = after_rejection(budget, JUDGE_BUDGET)
    assert budget.charge == JUDGE_BUDGET and spend(budget) is None
    assert cascade_step(budget).charge == GATE_BUDGET


def test_used_up_budget_of_one_type_leaves_the_other():
    budget = ChainBudget(0, 1, JUDGE_BUDGET, JUDGE_BUDGET)
    assert spend(budget) is None  # judge budget exhausted
    assert spend(after_rejection(budget, GATE_BUDGET)) == ChainBudget(0, 0, JUDGE_BUDGET, "gate")


# ---------------------------------------------------------------- turn level


def test_accepted_dialog_needs_no_regeneration():
    state = run(FakeAuthor([dialog_json("Eins.", "Zwei.")]), FakeJev())
    assert state.regenerations == [] and state.discarded == []
    assert state.used_fact_ids == ["1-0000"]


def test_rejected_turn_is_regenerated_and_cascade_stops_when_confirmed():
    author = FakeAuthor(
        [dialog_json("Eins.", "Falsch.", "Drei.", "Vier.", facts=DISTINCT)],
        [turn_json("Lena", "Besser.", "1-0001"), turn_json("Felix", "Dazu passend.", "1-0002")],
    )
    jev = FakeJev(bad={"Falsch."}, cascade=[0.99])
    state = run(author, jev)
    assert [(r.turn_index, r.kind) for r in state.regenerations] == [(1, "retry"), (2, "cascade")]
    assert [t.text for t in state.dialog.turns] == ["Eins.", "Besser.", "Dazu passend.", "Vier."]
    assert state.dialog.turns[1].speaker == "Felix"  # speaker of the replaced turn is kept
    assert [c.confirmed for c in state.cascade_checks] == [True]
    assert state.regenerations[0].previous.text == "Falsch."
    assert all(j.accepted for j in state.judgements)


def test_cascade_continues_while_not_confirmed_and_shares_the_budget():
    author = FakeAuthor(
        [
            dialog_json("Eins.", "Falsch.", "Drei.", "Vier.", facts=DISTINCT),
            dialog_json("Neu eins.", "Neu zwei.", fact="1-0000"),
        ],
        [turn_json("Felix", "B.", "1-0001"), turn_json("Lena", "C.", "1-0002")],
    )
    jev = FakeJev(bad={"Falsch."}, cascade=[0.01])  # content lost: cascade goes on, budget out
    state = run(author, jev)
    assert state.discarded[0].reason == "Turn regeneration budget used up (judge, 2x)."
    # the chain failed in the cascade of turn 1: the dialog continues from turn 1
    assert state.discarded[0].continued_from == 1
    assert [t.text for t in state.dialog.turns] == ["Eins.", "Neu eins.", "Neu zwei."]
    # trigger plus one cascade turn: the 2x judge budget of the judge rejection
    assert [(r.kind, r.budget) for r in state.regenerations] == [
        ("retry", "judge"),
        ("cascade", "judge"),
    ]


def test_uncertain_cascade_check_confirmed_by_llm_judge_stops_the_cascade():
    author = FakeAuthor(
        [dialog_json("Eins.", "Falsch.", "Drei.", "Vier.", facts=DISTINCT)],
        [turn_json("Lena", "Besser.", "1-0001"), turn_json("Felix", "Dazu passend.", "1-0002")],
    )
    llm = FakeJudgeLlm([{"same": True, "reason": "Gleiche Aussage."}])
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.74]), judge_llm=llm)
    assert state.discarded == []
    assert [(r.turn_index, r.kind) for r in state.regenerations] == [(1, "retry"), (2, "cascade")]
    assert state.cascade_checks == [
        CascadeCheck(1, 2, 0.74, True, None, "llm_judge", "Gleiche Aussage.")
    ]
    assert state.judgement_dict()["cascade_checks"][0] == {
        "conversation": 1,
        "turn_index": 2,
        "p": 0.74,
        "confirmed": True,
        "error": None,
        "decided_by": "llm_judge",
        "reason": "Gleiche Aussage.",
    }


def test_uncertain_cascade_check_rejected_by_llm_judge_continues_the_cascade():
    author = FakeAuthor(
        [
            dialog_json("Eins.", "Falsch.", "Drei.", "Vier.", facts=DISTINCT),
            dialog_json("Neu eins.", "Neu zwei.", fact="1-0002"),
        ],
        [turn_json("Felix", "B.", "1-0001"), turn_json("Lena", "C.", "1-0002")],
    )
    llm = FakeJudgeLlm([{"same": False, "reason": "Aussage fehlt."}])
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.38]), judge_llm=llm)
    assert state.discarded[0].reason.startswith("Turn regeneration budget")
    assert [(c.decided_by, c.confirmed) for c in state.cascade_checks] == [("llm_judge", False)]


def test_cascade_runs_to_the_end_of_the_dialog_without_check_after_last_turn():
    author = FakeAuthor(
        [dialog_json("Eins.", "Falsch.")], [turn_json("Felix", "Besser.", "1-0000")]
    )
    state = run(author, FakeJev(bad={"Falsch."}))
    assert [r.kind for r in state.regenerations] == ["retry"]
    assert state.cascade_checks == []


def test_jev_error_on_cascade_check_goes_to_llm_judge_and_continues_the_cascade():
    class Flaky(FakeJev):
        def ask(self, state, questions):
            if CASCADE_KEY in questions:
                raise JevTransientError("429")
            return super().ask(state, questions)

    author = FakeAuthor(
        [dialog_json("Eins.", "Falsch.", "Drei."), dialog_json("N.", fact="1-0001")],
        [turn_json("Felix", "B.", "1-0000"), turn_json("Lena", "C.", "1-0000")],
    )
    llm = FakeJudgeLlm([{"same": False, "reason": "Unklar."}])
    state = run(author, Flaky(bad={"Falsch."}), judge_llm=llm)
    check = state.cascade_checks[0]
    assert check.p is None and not check.confirmed and "JevTransientError" in check.error
    assert check.decided_by == "llm_judge" and len(llm.cascade_calls) == 1


def test_regenerated_turn_rejected_again_is_retried_within_the_budget():
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "turn_retries": 3}}
    author = FakeAuthor(
        [dialog_json("Falsch.", "Zwei.")],
        [
            turn_json("Lena", "Falsch.", "1-0000"),
            turn_json("Lena", "Gut.", "1-0000"),
            turn_json("Felix", "Passt.", "1-0000"),
        ],
    )
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.99]), config=config)
    assert [(r.turn_index, r.kind) for r in state.regenerations] == [
        (0, "retry"),
        (0, "retry"),
        (1, "cascade"),
    ]
    assert [t.text for t in state.dialog.turns] == ["Gut.", "Passt."]
    # Every judgement of a replaced turn version is kept, with conversation and version.
    assert [(s.conversation, s.turn_index, s.version, s.replaced_by) for s in state.superseded] == [
        (1, 0, 0, 0),
        (1, 0, 1, 1),
        (1, 1, 0, 2),
    ]
    assert [s.judgement.accepted for s in state.superseded] == [False, False, True]
    assert state.regenerations[1].previous.text == "Falsch."  # version 1, replaced by #1
    dialog = state.dialog_dict()
    assert list(dialog) == [
        "turns",
        "problems",
        "used_fact_ids",
        "conversation",
        "regenerations",
        "discarded_conversations",
        "aliases",
    ]
    assert dialog["aliases"] == {"F1": "1-0000", "F2": "1-0001", "F3": "1-0002"}
    data = state.judgement_dict()
    assert list(data) == [
        "judgements",
        "used_fact_ids",
        "cascade_checks",
        "conversation",
        "superseded_judgements",
    ]
    assert data["superseded_judgements"][0] == {
        "conversation": 1,
        "turn_index": 0,
        "version": 0,
        "replaced_by": 0,
        "judgement": state.superseded[0].judgement.to_dict(),
    }


def test_turn_version_counts_regenerations_of_that_turn_in_that_conversation():
    from gut_zu_wissen.orchestrator import Regeneration, turn_version

    turn = Turn(0, "Lena", "T.", [], [])
    regs = [Regeneration(1, 0, "retry", None, turn), Regeneration(2, 0, "retry", None, turn)]
    regs.append(Regeneration(1, 1, "cascade", None, turn))
    assert turn_version(regs, 1, 0) == 1 and turn_version(regs, 2, 0) == 1
    assert turn_version(regs, 1, 2) == 0


def test_superseded_judgements_of_a_discarded_conversation_keep_its_number():
    author = FakeAuthor(
        [
            dialog_json("Eins.", "Falsch.", "Drei.", "Vier.", facts=DISTINCT),
            dialog_json("Neu eins.", "Neu zwei.", fact="1-0000"),
        ],
        [turn_json("Felix", "B.", "1-0001"), turn_json("Lena", "C.", "1-0002")],
    )
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.01]))
    assert state.conversation == 2
    assert [(s.conversation, s.turn_index, s.version) for s in state.superseded] == [
        (1, 1, 0),
        (1, 2, 0),
    ]


def test_turn_with_unknown_fact_id_is_regenerated_without_a_judge_call():
    author = FakeAuthor(
        [dialog_json("Eins.", "Zwei.", "Drei.")],
        [turn_json("Felix", "Ok.", "1-0000"), turn_json("Lena", "Drei neu.", "1-0000")],
    )
    author.dialogs[0]["turns"][1]["fact_ids"] = ["9-9999"]
    state = run(author, FakeJev(cascade=[0.99]))
    assert state.regenerations[0].reason.startswith("Unknown fact IDs")
    assert state.judgements[1].decided_by == "jev"


def test_repeated_fact_is_regenerated_as_a_gate_problem_without_a_judge_call():
    author = FakeAuthor(
        [dialog_json("Eins.", "Zwei.", "Drei.", "Vier.")],  # all cite 1-0000
        [turn_json("Lena", "Drei neu.", "1-0001"), turn_json("Felix", "Vier neu.", "1-0002")],
    )
    state = run(author, FakeJev(cascade=[0.99]))
    first = state.regenerations[0]
    assert (first.turn_index, first.kind) == (2, "retry")
    assert first.reason == repetition_reason("1-0000", [0, 1])
    assert "Der Fakt F1 wurde schon in Redebeitrag 0 und 1 erzählt." in author.calls[1]
    assert "1-0000" not in author.calls[1]
    old = state.superseded[0]
    assert (old.turn_index, old.judgement.decided_by, old.judgement.jev_p) == (2, "gate", {})
    assert list(state.judgement_dict()["superseded_judgements"][0]["judgement"]) == [
        "turn_index",
        "accepted",
        "decided_by",
        "jev_p",
        "jev_error",
        "reason",
    ]
    assert all(j.accepted for j in state.judgements)


def test_regenerated_turn_that_creates_a_repetition_rejects_the_later_turn():
    # 1-0000 is cited in turns 0 and 3. Regenerated turn 1 cites it too, so turn 3 (the later
    # one) repeats it, not turn 1; turn 3 is repaired with its own budget.
    author = FakeAuthor(
        [
            dialog_json(
                "Eins.",
                "Falsch.",
                "Drei.",
                "Vier.",
                "Fünf.",
                "Sechs.",
                facts=["1-0000", "1-0001", "1-0002", "1-0000", "1-0003", "1-0000"],
            )
        ],
        [
            turn_json("Felix", "Besser.", "1-0000"),
            turn_json("Lena", "Drei neu.", "1-0002"),
            turn_json("Felix", "Vier neu.", "1-0003"),
            turn_json("Lena", "Fünf neu.", "1-0003"),
        ],
    )
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.99, 0.99]), n_facts=4)
    assert [(r.turn_index, r.kind) for r in state.regenerations] == [
        (1, "retry"),
        (2, "cascade"),
        (3, "retry"),
        (4, "cascade"),
    ]
    assert state.regenerations[2].reason == repetition_reason("1-0000", [0, 1])
    later = state.superseded[2].judgement
    assert (later.turn_index, later.accepted, later.decided_by) == (3, False, "gate")
    assert later.jev_p  # the earlier judge values are kept
    assert [t.text for t in state.dialog.turns][1:5] == [
        "Besser.",
        "Drei neu.",
        "Vier neu.",
        "Fünf neu.",
    ]
    assert all(j.accepted for j in state.judgements)


def test_accepted_turn_with_markup_is_regenerated_as_a_gate_problem():
    author = FakeAuthor(
        [dialog_json("Eins.", "In [[Paris]].", "Drei.")],
        [turn_json("Felix", "In Paris.", "1-0000"), turn_json("Lena", "Drei neu.", "1-0000")],
    )
    state = run(author, FakeJev(cascade=[0.99]))
    first = state.regenerations[0]
    assert (first.turn_index, first.kind) == (1, "retry")
    assert first.reason.startswith("Markup im Text: Wiki-Link")
    assert state.dialog.turns[1].text == "In Paris."
    assert all(j.accepted for j in state.judgements)


def test_markup_rejection_excludes_no_facts():
    # Markup is no fact problem: the fresh offer after a used-up budget excludes nothing (D6).
    markup = turn_json("Lena", "In [[Paris]].", "1-0000")
    author = FakeAuthor([dialog_json("In [[Paris]]."), dialog_json("Sauber.")], [markup] * 3)
    state = run(author, FakeJev())
    assert state.discarded[0].excluded_fact_ids == []
    assert run.offers == [frozenset(), frozenset()]


def test_jev_abort_is_a_gate_failure():
    jev = FakeJev()
    jev.error = JevAbortError("401")
    with pytest.raises(GateFailure, match="Jev aborted"):
        run(FakeAuthor([dialog_json("Eins.")]), jev)


# ---------------------------------------------------------------- conversation level


def test_conversation_regeneration_excludes_rejected_facts_cumulatively():
    # Every regeneration fails again, so each conversation ends with an exhausted budget.
    bad_turn = turn_json("Lena", "Falsch.", "1-0000")
    author = FakeAuthor(
        [dialog_json("Falsch.", fact="1-0000"), dialog_json("Falsch.", fact="1-0001")],
        [bad_turn, bad_turn, turn_json("Lena", "Falsch.", "1-0001")] * 2,
    )
    jev = FakeJev(bad={"Falsch."})
    with pytest.raises(GateFailure):
        run(author, jev, n_facts=2)
    # The third offer is empty and fails the offer gate (D5).
    assert run.offers == [frozenset(), frozenset({"1-0000"}), frozenset({"1-0000", "1-0001"})]


def test_conversation_regeneration_budget_ends_in_gate_failure():
    dialogs = [dialog_json("Falsch.", fact=f"1-000{i}") for i in range(3)]
    turns = [turn_json("Lena", "Falsch.", f"1-000{i}") for i in range(3) for _ in range(2)]
    with pytest.raises(GateFailure, match="Conversation regeneration budget used up"):
        run(FakeAuthor(dialogs, turns), FakeJev(bad={"Falsch."}), n_facts=5)
    assert len(run.offers) == 3  # first run plus two regenerations
    assert run.offers[2] == frozenset({"1-0000", "1-0001"})


def test_too_few_used_facts_after_author_triggers_conversation_regeneration():
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "min_used_facts": 2}}
    author = FakeAuthor([dialog_json("Eins."), dialog_json("Eins.", fact="1-0000")])
    author.dialogs[1]["turns"].append(turn_json("Felix", "Zwei.", "1-0001"))
    state = run(author, FakeJev(), config=config)
    assert "D5" in state.discarded[0].reason
    assert run.offers == [frozenset(), frozenset()]  # nothing was rejected, nothing excluded


def two_fact_turns(first: str) -> dict:
    """Turn 0 cites both facts, turn 1 is a reaction without facts."""
    both = ["1-0000", "1-0001"]
    turn0 = {
        "speaker": "Lena",
        "text": first,
        "fact_ids": both,
        "statements": [{"text": "S", "fact_ids": both}],
    }
    return {"turns": [turn0, turn_json("Felix", "Echt?", None)]}


def test_enough_used_facts_after_judging_pass_the_final_gate():
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "min_used_facts": 2}}
    state = run(FakeAuthor([two_fact_turns("Eins.")]), FakeJev(), config=config)
    assert state.used_fact_ids == ["1-0000", "1-0001"]


def test_too_few_used_facts_after_judging_is_a_gate_failure():
    # The repaired turn is a reaction without facts, so the accepted turns use none (D5).
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "min_used_facts": 2}}
    author = FakeAuthor(
        [two_fact_turns("Falsch.")],
        [turn_json("Lena", "Okay.", None), turn_json("Felix", "Echt?", None)],
    )
    with pytest.raises(GateFailure, match="used in accepted turns"):
        run(author, FakeJev(bad={"Falsch."}, cascade=[0.99]), config=config)


def test_tunguska_like_wrong_id_is_rejected_and_a_corrected_regeneration_accepted():
    # The author cites an alias outside the offer and a real-looking segment ID that was never
    # offered (archive 2026-11-20_tunguska-ereignis-2); the regeneration uses a valid alias.
    author = FakeAuthor(
        [dialog_json("Eins.", "Zwei.", "Drei.", fact="F1")],
        [turn_json("Felix", "Zwei neu.", "f2"), turn_json("Lena", "Drei neu.", "[F1]")],
    )
    author.dialogs[0]["turns"][1]["fact_ids"] = ["F53"]
    author.dialogs[0]["turns"][1]["statements"][0]["fact_ids"] = ["1-0099"]
    state = run(author, FakeJev(cascade=[0.99]))
    first = state.regenerations[0]
    assert (first.turn_index, first.kind) == (1, "retry")
    assert first.reason == (
        "Unknown fact IDs: 1-0099, F53 (cited by the author; no offered fact has this identifier)."
    )
    retry_prompt, cascade_prompt = author.calls[1], author.calls[2]
    assert "Die Kennungen 1-0099, F53 gibt es im Angebot nicht." in retry_prompt
    assert "(F1–F3)" in retry_prompt and "Unknown fact IDs" not in retry_prompt
    assert "Zwei neu.\nKennungen: F2" in cascade_prompt  # the corrected neighbour, as aliases
    assert not any("1-000" in call for call in author.calls)  # the author never sees segment IDs
    assert [t.fact_ids for t in state.dialog.turns] == [["1-0000"], ["1-0001"], ["1-0000"]]
    assert all(j.accepted for j in state.judgements)
    assert state.used_fact_ids == ["1-0000", "1-0001"]


def test_discarded_conversation_keeps_the_aliases_of_its_own_offer():
    # D6: the fresh offer lacks the excluded fact, so its aliases are rebuilt.
    bad = turn_json("Lena", "Falsch.", "F1")
    author = FakeAuthor(
        [dialog_json("Falsch.", fact="F1"), dialog_json("Gut.", fact="F1")], [bad, bad]
    )
    state = run(author, FakeJev(bad={"Falsch."}), n_facts=2)
    (discarded,) = state.discarded
    assert discarded.excluded_fact_ids == ["1-0000"]
    assert asdict(discarded)["aliases"] == {"F1": "1-0000", "F2": "1-0001"}
    assert list(asdict(discarded))[-5:] == [
        "aliases",
        "regeneration",
        "continued_from",
        "fresh_reason",
        "continuation_reason",
    ]
    assert discarded.continuation_reason is None
    assert (discarded.regeneration, discarded.continued_from) == ("fresh", None)
    assert discarded.fresh_reason == "The failing turn chain started at turn 0."
    assert state.dialog_dict()["aliases"] == {"F1": "1-0001"}
    assert state.dialog.turns[0].fact_ids == ["1-0001"]  # F1 of the new offer


def unknown_id_turn(speaker: str, text: str) -> dict:
    return turn_json(speaker, text, "9-9999")


def test_gate_then_judge_rejection_in_one_chain():
    # turn 1: unknown ID (gate) -> retry rejected by Jev (judge) -> retry accepted -> cascade
    author = FakeAuthor(
        [dialog_json("Eins.", "Zwei.", "Drei.", "Vier.", facts=DISTINCT)],
        [
            turn_json("Felix", "Falsch.", "1-0001"),
            turn_json("Felix", "Gut.", "1-0001"),
            turn_json("Lena", "Drei neu.", "1-0002"),
        ],
    )
    author.dialogs[0]["turns"][1]["fact_ids"] = ["9-9999"]
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.99]))
    assert [(r.turn_index, r.kind, r.budget) for r in state.regenerations] == [
        (1, "retry", "gate"),
        (1, "retry", "judge"),
        (2, "cascade", "gate"),  # the chain was triggered by a gate problem
    ]
    assert state.discarded == [] and all(j.accepted for j in state.judgements)


def test_judge_then_gate_rejection_in_one_chain():
    author = FakeAuthor(
        [dialog_json("Eins.", "Falsch.", "Drei.", "Vier.", facts=DISTINCT)],
        [
            unknown_id_turn("Felix", "Zwei."),
            turn_json("Felix", "Gut.", "1-0001"),
            turn_json("Lena", "Drei neu.", "1-0002"),
        ],
    )
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.99]))
    assert [(r.turn_index, r.kind, r.budget) for r in state.regenerations] == [
        (1, "retry", "judge"),
        (1, "retry", "gate"),
        (2, "cascade", "judge"),
    ]
    assert state.discarded == []


def test_cascade_after_a_gate_retry_does_not_touch_the_judge_budget():
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "turn_retries": 0}}
    author = FakeAuthor(
        [dialog_json("Eins.", "Zwei.", "Drei.", "Vier.", facts=DISTINCT)],
        [
            turn_json("Felix", "Zwei neu.", "1-0001"),
            turn_json("Lena", "Drei neu.", "1-0002"),
            turn_json("Felix", "Vier neu.", "1-0000"),
        ],
    )
    author.dialogs[0]["turns"][1]["fact_ids"] = ["9-9999"]
    state = run(author, FakeJev(cascade=[0.01, 0.99]), config=config)  # content lost: goes on
    assert [(r.turn_index, r.kind, r.budget) for r in state.regenerations] == [
        (1, "retry", "gate"),
        (2, "cascade", "gate"),
        (3, "cascade", "gate"),
    ]
    assert state.discarded == [] and all(j.accepted for j in state.judgements)


def test_gate_budget_exhausted_after_gate_retries():
    author = FakeAuthor(
        [dialog_json("Eins.", "Zwei.", facts=["1-0000", "1-0001"])],
        [unknown_id_turn("Felix", f"Neu {n}.") for n in range(3)],
    )
    author.dialogs[0]["turns"][1]["fact_ids"] = ["9-9999"]
    with pytest.raises(GateFailure, match=r"gate problems, 3x"):
        run(
            author,
            FakeJev(),
            config={**CONFIG, "quotas": {**CONFIG["quotas"], "conversation_retries": 0}},
        )
    assert [r.budget for r in run.state.regenerations] == ["gate"] * 3


def test_judge_budget_exhausted_after_turn_retries():
    author = FakeAuthor(
        [dialog_json("Eins.", "Falsch.", facts=["1-0000", "1-0001"])],
        [turn_json("Felix", "Falsch.", "1-0001")] * 2,
    )
    with pytest.raises(GateFailure, match=r"judge, 2x"):
        run(
            author,
            FakeJev(bad={"Falsch."}),
            config={**CONFIG, "quotas": {**CONFIG["quotas"], "conversation_retries": 0}},
        )
    assert [r.budget for r in run.state.regenerations] == ["judge"] * 2


def test_regeneration_records_its_budget_as_the_last_field():
    author = FakeAuthor([dialog_json("Eins.", "Falsch.")], [turn_json("Felix", "Gut.", "1-0000")])
    state = run(author, FakeJev(bad={"Falsch."}))
    (regeneration,) = state.dialog_dict()["regenerations"]
    assert list(regeneration) == [
        "conversation",
        "turn_index",
        "kind",
        "reason",
        "previous",
        "budget",
    ]
    assert regeneration["budget"] == "judge"


# ---------------------------------------------------------------- continuation (D6, 2026-11-20)


class CountingJev(FakeJev):
    """Records the text of every judged turn."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.judged: list[str] = []

    def ask(self, state: dict, questions: dict) -> dict:
        if CASCADE_KEY not in questions:
            self.judged.append(state["redebeitrag"]["text"])
        return super().ask(state, questions)


def test_conversation_path_continues_unless_k_is_zero_or_the_offer_is_too_small():
    assert conversation_path(3, 10, 10) == ("continuation", None)
    assert conversation_path(0, 10, 10) == ("fresh", "The failing turn chain started at turn 0.")
    assert conversation_path(None, 10, 10) == ("fresh", "No failing turn chain to continue from.")
    path, reason = conversation_path(3, 9, 10)
    assert path == "fresh" and "fewer than the D5 minimum of 10" in reason


def test_continuation_offer_drops_excluded_facts_but_keeps_prefix_and_pinned_ones():
    offer = Offer([rf(1), rf(2), rf(3)], [rf(4)], [rf(0)], frozenset({"1-0000"}))
    prefix = [Turn(0, "Lena", "T.", ["1-0001"], [Statement("S.", ["1-0001"])])]
    excluded = {"1-0000", "1-0001", "1-0002", "1-0004"}
    kept = continuation_offer(offer, excluded, prefix)
    assert [r.fact.segment_id for r in kept.einordnung] == ["1-0000"]  # pinned
    assert [r.fact.segment_id for r in kept.main] == ["1-0001", "1-0003"]  # prefix cites 1-0001
    assert kept.linked == [] and kept.pinned == offer.pinned


def test_discarded_tail_holds_turns_from_k_with_their_problems():
    turns = [Turn(n, "Lena", f"T{n}.", [f"1-000{n}"]) for n in range(4)]
    from gut_zu_wissen.dialog import TurnProblem

    dialog = Dialog(turns, [TurnProblem(1, "a"), TurnProblem(3, "b")], [])
    tail = discarded_tail(dialog, 2, {"1-0002", "1-0003"})
    assert [t.index for t in tail.turns] == [2, 3]
    assert [p.turn_index for p in tail.problems] == [3]
    assert tail.used_fact_ids == ["1-0002", "1-0003"]
    assert discarded_tail(dialog, 0, set()) is dialog


def failing_at_two() -> FakeAuthor:
    """Turn 2 is rejected by the judge three times: judge budget used up, k = 2."""
    return FakeAuthor(
        [
            dialog_json("Eins.", "Zwei.", "Falsch.", "Vier.", facts=DISTINCT),
            dialog_json("Weiter.", "Schluss.", facts=["1-0003", "1-0000"]),
        ],
        [turn_json("Lena", "Falsch.", "1-0002")] * 2,
    )


def test_exhausted_turn_budget_continues_from_the_failing_turn():
    author = failing_at_two()
    jev = CountingJev(bad={"Falsch."})
    state = run(author, jev, n_facts=5)
    assert [(t.index, t.text) for t in state.dialog.turns] == [
        (0, "Eins."),
        (1, "Zwei."),
        (2, "Weiter."),
        (3, "Schluss."),
    ]
    assert run.offers == [frozenset()]  # no re-ranking
    assert jev.judged[-2:] == ["Weiter.", "Schluss."]  # only the new turns are judged
    assert jev.judged.count("Eins.") == 1 and jev.judged.count("Zwei.") == 1
    assert all(j.accepted for j in state.judgements) and state.conversation == 2
    assert [j.turn_index for j in state.judgements] == [0, 1, 2, 3]
    # the continuation author saw the prefix as aliases and the excluded fact was dropped
    assert author.systems[-1] == CONTINUE_PROMPT.format(min_used=1)
    user = author.calls[-1]
    assert "[F3] Fakt 2." not in user and "Fakt 2." not in user
    assert "Redebeitrag 1 (Felix):\nZwei.\nKennungen: F2" in user
    assert "- F1: Redebeitrag 0\n- F2: Redebeitrag 1" in user
    assert "ab Redebeitrag 2 bis zum Ende" in user and "1-000" not in user
    assert state.dialog_dict()["aliases"] == {
        "F1": "1-0000",
        "F2": "1-0001",
        "F3": "1-0003",
        "F4": "1-0004",
    }
    (discarded,) = state.discarded
    assert (discarded.regeneration, discarded.continued_from, discarded.fresh_reason) == (
        "continuation",
        2,
        None,
    )
    assert [t.index for t in discarded.dialog.turns] == [2, 3]
    assert [j.turn_index for j in discarded.judgements] == [2, 3]
    assert discarded.excluded_fact_ids == ["1-0002"]
    assert discarded.aliases["F3"] == "1-0002"  # the offer the failed conversation saw
    assert summary_lines(state)[0] == "Conversations written: 2 (continued: 1, fresh: 0)"


def test_failure_inside_a_cascade_continues_from_the_triggering_turn():
    # turn 1 is rejected, its retry accepted, the cascade loses content at turn 2 and runs out
    # of budget at turn 3: k is 1, the triggering turn
    author = FakeAuthor(
        [
            dialog_json("Eins.", "Falsch.", "Drei.", "Vier.", facts=DISTINCT),
            dialog_json("Neu.", "Ende.", facts=["1-0002", "1-0000"]),
        ],
        [turn_json("Felix", "B.", "1-0001"), turn_json("Lena", "C.", "1-0002")],
    )
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.01]), n_facts=3)
    assert state.discarded[0].continued_from == 1
    assert [t.text for t in state.discarded[0].dialog.turns] == ["B.", "C.", "Vier."]
    assert [t.text for t in state.dialog.turns] == ["Eins.", "Neu.", "Ende."]


def test_repetition_gate_of_the_continuation_sees_the_prefix():
    author = FakeAuthor(
        [
            dialog_json(
                "Eins.", "Zwei.", "Falsch.", "Vier.", facts=["1-0000", "1-0000", "1-0002", "1-0000"]
            ),
            dialog_json("Weiter.", "Schluss.", facts=["1-0000", "1-0000"]),
        ],
        [turn_json("Lena", "Falsch.", "1-0002")] * 2
        + [turn_json("Lena", "Anders.", "1-0003"), turn_json("Felix", "Ende.", "1-0000")],
    )
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.99]), n_facts=5)
    second = [r for r in state.regenerations if r.conversation == 2]
    assert [(r.turn_index, r.kind, r.budget) for r in second] == [
        (2, "retry", "gate"),
        (3, "cascade", "gate"),
    ]
    assert second[0].reason == repetition_reason("1-0000", [0, 1])
    assert [t.text for t in state.dialog.turns] == ["Eins.", "Zwei.", "Anders.", "Ende."]


def test_d5_after_a_continuation_counts_the_whole_dialog():
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "min_used_facts": 2}}
    author = FakeAuthor(
        [
            dialog_json("Eins.", "Falsch.", facts=["1-0000", "1-0001"]),
            dialog_json("Neu.", facts=["1-0002"]),  # one new fact, two with the prefix
        ],
        [turn_json("Felix", "Falsch.", "1-0001")] * 2,
    )
    state = run(author, FakeJev(bad={"Falsch."}), n_facts=3, config=config)
    assert state.discarded[0].regeneration == "continuation"
    assert state.used_fact_ids == ["1-0000", "1-0002"]


def test_too_small_remaining_offer_falls_back_to_a_fresh_conversation():
    config = {**CONFIG, "quotas": {**CONFIG["quotas"], "min_used_facts": 2}}
    author = FakeAuthor(
        [
            dialog_json("Eins.", "Falsch.", "Drei.", facts=["1-0000", "1-0001", "1-0002"]),
            dialog_json("Neu.", fact="1-0000"),  # D5 fails: one used fact
            dialog_json("Neu.", fact="1-0000"),
        ],
        [turn_json("Felix", "Falsch.", "1-0002")] * 2,
    )
    # 1-0001 and 1-0002 are excluded: only 1-0000 (cited by the prefix) remains, one fact < 2
    with pytest.raises(GateFailure):
        run(author, FakeJev(bad={"Falsch."}), n_facts=3, config=config)
    first = run.state.discarded[0]
    assert (first.regeneration, first.continued_from) == ("fresh", None)
    assert "fewer than the D5 minimum of 2" in first.fresh_reason
    assert [t.text for t in first.dialog.turns] == ["Eins.", "Falsch.", "Drei."]  # whole dialog
    assert run.offers[1] == frozenset({"1-0001", "1-0002"})  # fresh rankings


def test_pinned_einordnung_fact_is_never_excluded():
    pinned = frozenset({"1-0000"})
    offers: list[frozenset[str]] = []

    def make(excluded: frozenset[str]) -> Offer:
        offers.append(excluded)
        return Offer([rf(1)], [], [rf(0)], pinned)

    author = FakeAuthor(
        [dialog_json("Falsch."), dialog_json("Gut.")],
        [turn_json("Lena", "Falsch.", "1-0000")] * 2,
    )
    state = Orchestration()
    orchestrate(state, make, author, FakeJudgeLlm(), FakeJev(bad={"Falsch."}), CONFIG)
    assert offers == [frozenset(), frozenset()]
    assert state.discarded[0].excluded_fact_ids == []


class ReasonJudgeLlm(FakeJudgeLlm):
    """Rejects with a reason that names segment IDs, as a judge's reason can."""

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        answer = super().complete_json(system, user, schema)
        if schema is not CASCADE_VERDICT_SCHEMA:
            answer["reason"] = "1-0002 sagt das nicht; 1-0001 schon."
        return answer


def test_continuation_author_gets_the_rejection_reason_with_aliases():
    author = failing_at_two()
    state = run(author, CountingJev(bad={"Falsch."}), n_facts=5, judge_llm=ReasonJudgeLlm())
    user = author.calls[-1]
    assert (
        "Grund, warum der bisherige Redebeitrag 2 verworfen wurde:\n"
        "[nicht mehr im Angebot] sagt das nicht; F2 schon." in user
    )
    # the same reason three times: shown once
    assert "Davor" not in user and "Vermeide diesen Fehler in der Fortsetzung." in user
    assert "1-000" not in user
    (discarded,) = state.discarded
    assert discarded.continuation_reason == "1-0002 sagt das nicht; 1-0001 schon."
    assert list(asdict(discarded))[-1] == "continuation_reason"


def test_continuation_rejections_are_in_the_author_form_for_the_new_offer():
    offered = {"1-0000", "1-0001", "1-0002"}
    aliases = {"F1": "1-0000", "F2": "1-0001"}  # 1-0002 excluded
    unknown = Turn(3, "Lena", "T", ["9-9999"])
    chain = [
        Rejection(Turn(2, "Lena", "T", ["1-0000"]), repetition_reason("1-0000", [0, 1])),
        Rejection(
            unknown,
            "Unknown fact IDs: 9-9999 (cited by the author; no offered fact has this identifier).",
        ),
    ]
    (first, second) = continuation_rejections(chain, offered, aliases)
    assert first == (2, repetition_reason("F1", [0, 1]))
    assert second[0] == 3
    assert second[1].startswith("Die Kennung 9-9999 gibt es im Angebot nicht.")


def test_felix_premises_on_the_pinned_fact_pass_the_gate_in_the_orchestrator():
    # all turns cite the pinned Einordnung fact 1-0000; Felix's turns 1 and 3 are exempt, Lena's
    # 0 and 2 are two turns: no repetition, no regeneration
    def make(excluded: frozenset[str]) -> Offer:
        return Offer([rf(1)], [], [rf(0)], frozenset({"1-0000"}))

    author = FakeAuthor([dialog_json("Eins.", "Zwei.", "Drei.", "Vier.", "Fünf.", fact="F1")])
    state = Orchestration()
    orchestrate(state, make, author, FakeJudgeLlm(), FakeJev(), CONFIG)
    assert state.regenerations == [] and state.discarded == []
    assert all(j.accepted for j in state.judgements)


def test_lena_retelling_the_pinned_fact_is_still_rejected_in_the_orchestrator():
    def make(excluded: frozenset[str]) -> Offer:
        return Offer([rf(1)], [], [rf(0)], frozenset({"1-0000"}))

    author = FakeAuthor(
        [dialog_json("Eins.", "Zwei.", "Drei.", "Vier.", "Fünf.", "Sechs.", fact="F1")],
        [turn_json("Lena", "Fünf neu.", "1-0001"), turn_json("Felix", "Sechs neu.", "1-0000")],
    )
    state = Orchestration()
    orchestrate(state, make, author, FakeJudgeLlm(), FakeJev(cascade=[0.99]), CONFIG)
    (first, _) = state.regenerations
    assert (first.turn_index, first.kind, first.budget) == (4, "retry", "gate")
    assert first.reason == repetition_reason("1-0000", [0, 2])
    # the regenerating author saw the told listing with the mark
    assert "(Felix darf ihn erneut als Prämisse einer Frage verwenden)" in author.calls[1]
