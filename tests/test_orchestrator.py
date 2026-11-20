from dataclasses import asdict

import pytest

from gut_zu_wissen.dialog import Statement, Turn, repetition_reason
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.jev import JevAbortError, JevOverlengthError, JevTransientError
from gut_zu_wissen.judge import Decision, TurnJudgement
from gut_zu_wissen.llm import LlmError
from gut_zu_wissen.orchestrator import (
    CASCADE_KEY,
    CASCADE_PROMPT,
    CASCADE_VERDICT_SCHEMA,
    CascadeCheck,
    GateFailure,
    Offer,
    Orchestration,
    cascade_decision,
    cascade_request,
    cascade_verdict_confirmed,
    check_cascade,
    format_cascade_input,
    orchestrate,
    rejected_fact_ids,
)
from gut_zu_wissen.ranking import RankedFact

T = 0.95
CONFIG = {
    "judge": {"t": T},
    "quotas": {
        "min_used_facts": 1,
        "max_turns_per_fact": 2,
        "turn_retries": 2,
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
    """Rejects every turn it judges. Cascade checks answer from `cascade`, one verdict per call;
    every cascade call is recorded."""

    def __init__(self, cascade: list[dict] | None = None) -> None:
        self.cascade, self.cascade_calls = list(cascade or []), []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
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
        def complete_json(self, system, user, schema):
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
            dialog_json("Eins.", "Falsch.", "Drei.", "Vier."),
            dialog_json("Neu eins.", "Neu zwei.", fact="1-0001"),
        ],
        [turn_json("Felix", "B.", "1-0000"), turn_json("Lena", "C.", "1-0000")],
    )
    jev = FakeJev(bad={"Falsch."}, cascade=[0.01])  # content lost: cascade goes on, budget out
    state = run(author, jev)
    assert state.discarded[0].reason.startswith("Turn regeneration budget")
    assert [t.text for t in state.dialog.turns] == ["Neu eins.", "Neu zwei."]
    assert len(state.regenerations) == 2  # trigger plus one cascade turn: the 2x budget


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
            dialog_json("Eins.", "Falsch.", "Drei.", "Vier."),
            dialog_json("Neu eins.", "Neu zwei.", fact="1-0001"),
        ],
        [turn_json("Felix", "B.", "1-0000"), turn_json("Lena", "C.", "1-0000")],
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
    assert list(asdict(discarded))[-1] == "aliases"
    assert state.dialog_dict()["aliases"] == {"F1": "1-0001"}
    assert state.dialog.turns[0].fact_ids == ["1-0001"]  # F1 of the new offer
