import pytest

from gut_zu_wissen.dialog import Statement, Turn
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.jev import JevAbortError, JevTransientError
from gut_zu_wissen.judge import TurnJudgement
from gut_zu_wissen.orchestrator import (
    CASCADE_KEY,
    GateFailure,
    Offer,
    Orchestration,
    cascade_confirmed,
    cascade_request,
    orchestrate,
    rejected_fact_ids,
)
from gut_zu_wissen.ranking import RankedFact

T = 0.95
CONFIG = {
    "judge": {"t": T},
    "quotas": {"min_used_facts": 1, "turn_retries": 2, "conversation_retries": 2},
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


def dialog_json(*texts: str, fact: str = "1-0000") -> dict:
    return {
        "turns": [
            turn_json("Lena" if n % 2 == 0 else "Felix", text, fact) for n, text in enumerate(texts)
        ]
    }


class FakeAuthor:
    """Dialogs and single turns come from queues; every call is recorded."""

    def __init__(self, dialogs: list[dict], turns: list[dict] | None = None) -> None:
        self.dialogs, self.turns, self.calls = list(dialogs), list(turns or []), []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append(user)
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
    def complete_json(self, system: str, user: str, schema: dict) -> dict:
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


def run(author, jev, n_facts=3, config=CONFIG):
    make, calls = offer_of(n_facts)
    state = Orchestration()
    try:
        orchestrate(state, make, author, FakeJudgeLlm(), jev, config)
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


@pytest.mark.parametrize("p,expected", [(0.95, True), (1.0, True), (0.9499, False), (0.5, False)])
def test_cascade_stops_only_at_p_at_least_t(p, expected):
    assert cascade_confirmed(p, T) is expected


def test_cascade_request_names_old_statements_and_new_text():
    before = Turn(1, "Felix", "Alt.", ["a"], [Statement("Aussage alt.", ["a"])])
    after = Turn(1, "Felix", "Neu.", ["a"], [])
    state, questions = cascade_request(before, after)
    assert state == {
        "vorher": {"text": "Alt.", "aussagen": ["Aussage alt."]},
        "nachher": {"text": "Neu."},
    }
    assert list(questions) == [CASCADE_KEY]


# ---------------------------------------------------------------- turn level


def test_accepted_dialog_needs_no_regeneration():
    state = run(FakeAuthor([dialog_json("Eins.", "Zwei.")]), FakeJev())
    assert state.regenerations == [] and state.discarded == []
    assert state.used_fact_ids == ["1-0000"]


def test_rejected_turn_is_regenerated_and_cascade_stops_when_confirmed():
    author = FakeAuthor(
        [dialog_json("Eins.", "Falsch.", "Drei.", "Vier.")],
        [turn_json("Lena", "Besser.", "1-0000"), turn_json("Felix", "Dazu passend.", "1-0000")],
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
            dialog_json("Eins.", "Falsch.", "Drei.", "Vier."),
            dialog_json("Neu eins.", "Neu zwei.", fact="1-0001"),
        ],
        [turn_json("Felix", "B.", "1-0000"), turn_json("Lena", "C.", "1-0000")],
    )
    jev = FakeJev(bad={"Falsch."}, cascade=[0.5])  # uncertain: cascade goes on, budget is out
    state = run(author, jev)
    assert state.discarded[0].reason.startswith("Turn regeneration budget")
    assert [t.text for t in state.dialog.turns] == ["Neu eins.", "Neu zwei."]
    assert len(state.regenerations) == 2  # trigger plus one cascade turn: the 2x budget


def test_cascade_runs_to_the_end_of_the_dialog_without_check_after_last_turn():
    author = FakeAuthor(
        [dialog_json("Eins.", "Falsch.")], [turn_json("Felix", "Besser.", "1-0000")]
    )
    state = run(author, FakeJev(bad={"Falsch."}))
    assert [r.kind for r in state.regenerations] == ["retry"]
    assert state.cascade_checks == []


def test_jev_error_on_cascade_check_continues_the_cascade():
    class Flaky(FakeJev):
        def ask(self, state, questions):
            if CASCADE_KEY in questions:
                raise JevTransientError("429")
            return super().ask(state, questions)

    author = FakeAuthor(
        [dialog_json("Eins.", "Falsch.", "Drei."), dialog_json("N.", fact="1-0001")],
        [turn_json("Felix", "B.", "1-0000"), turn_json("Lena", "C.", "1-0000")],
    )
    state = run(author, Flaky(bad={"Falsch."}))
    check = state.cascade_checks[0]
    assert check.p is None and not check.confirmed and "JevTransientError" in check.error


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
            dialog_json("Eins.", "Falsch.", "Drei.", "Vier."),
            dialog_json("Neu eins.", "Neu zwei.", fact="1-0001"),
        ],
        [turn_json("Felix", "B.", "1-0000"), turn_json("Lena", "C.", "1-0000")],
    )
    state = run(author, FakeJev(bad={"Falsch."}, cascade=[0.5]))
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


def test_accepted_turn_with_markup_is_regenerated_as_a_gate_problem():
    author = FakeAuthor(
        [dialog_json("Eins.", "In [[Paris]].", "Drei.")],
        [turn_json("Felix", "In Paris.", "1-0000"), turn_json("Lena", "Drei neu.", "1-0000")],
    )
    state = run(author, FakeJev(cascade=[0.99]))
    first = state.regenerations[0]
    assert (first.turn_index, first.kind) == (1, "retry")
    assert first.reason.startswith("Markup in text: wiki link")
    assert state.dialog.turns[1].text == "In Paris."
    assert all(j.accepted for j in state.judgements)


def test_markup_rejection_excludes_no_facts():
    # Markup is no fact problem: the fresh offer after a used-up budget excludes nothing (D6).
    markup = turn_json("Lena", "In [[Paris]].", "1-0000")
    author = FakeAuthor([dialog_json("In [[Paris]]."), dialog_json("Sauber.")], [markup, markup])
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
    turns = [turn_json("Lena", "Falsch.", "1-0000")] * 6
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
