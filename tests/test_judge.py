import pytest

from gut_zu_wissen.dialog import Statement, Turn
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.jev import JevAbortError, JevOverlengthError, JevTransientError
from gut_zu_wissen.judge import (
    COMPLETENESS_KEY,
    Decision,
    TurnJudgement,
    build_request,
    decide,
    format_judge_input,
    judge_dialog,
    judge_turn,
    judged_facts_gate,
    used_in_accepted,
)
from gut_zu_wissen.ranking import RankedFact

T = 0.95
FACTS = {
    "1-0": RankedFact(
        1, "Rotfuchs", 1, Fact("1-0", "sentence", "", [], "Der Fuchs wiegt acht Kilo.", ["Davor."])
    ),
    "1-1": RankedFact(2, "Rotfuchs", 1, Fact("1-1", "sentence", "", [], "Er lebt in Höhlen.", [])),
}


def turn(i=0, text="Der Fuchs wiegt acht Kilo.", ids=("1-0",), statements=None) -> Turn:
    if statements is None:
        statements = [Statement("Der Fuchs wiegt acht Kilo.", list(ids))]
    return Turn(i, "Lena", text, list(ids), statements)


class FakeJev:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result, self.error, self.calls = result, error, []

    def ask(self, state, questions):
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        return self.result


class FakeLlm:
    def __init__(self, supported=True) -> None:
        self.supported, self.calls = supported, []

    def complete_json(self, system, user, schema):
        self.calls.append((system, user, schema))
        if "supported" in schema["properties"]:
            return {"supported": self.supported, "reason": "Begründung."}
        return {"reason": "Ablehnungsgrund."}


# ---- threshold decision at t, 1 - t and between


@pytest.mark.parametrize(
    ("ps", "expected"),
    [
        ([0.95, 0.99], Decision.SUPPORTED),  # exactly t
        ([0.9499, 0.99], Decision.UNCERTAIN),
        ([0.05, 0.99], Decision.REJECTED),  # exactly 1 - t
        ([0.0501, 0.99], Decision.UNCERTAIN),
        ([0.5], Decision.UNCERTAIN),
        ([0.02, 0.5], Decision.REJECTED),  # rejection wins over uncertainty
        ([], Decision.UNCERTAIN),
    ],
)
def test_decide(ps, expected):
    assert decide(ps, T) is expected


# ---- question builder


def test_request_has_one_question_per_statement_plus_completeness():
    t = turn(statements=[Statement("A", ["1-0"]), Statement("B", ["1-0", "1-1"])])
    state, questions = build_request(t, None, FACTS)
    assert list(questions) == ["aussage_0", "aussage_1", COMPLETENESS_KEY]
    assert "`fakten.1-0` und `fakten.1-1`" in questions["aussage_1"]["instructions"]
    assert state["fakten"]["1-0"] == {"artikel": "Rotfuchs", "satz": "Der Fuchs wiegt acht Kilo."}
    assert state["kontext"] == {"1-0": ["Davor."]}  # facts without context have no entry
    assert "vorheriger_redebeitrag" not in state


def test_request_includes_previous_turn_and_pure_reaction_has_only_completeness():
    reaction = Turn(1, "Felix", "Echt?", [], [])
    state, questions = build_request(reaction, turn(), FACTS)
    assert state["vorheriger_redebeitrag"] == "Der Fuchs wiegt acht Kilo."
    assert state["aussagen"] == [] and state["fakten"] == {} and "kontext" not in state
    assert list(questions) == [COMPLETENESS_KEY]


# ---- judging paths


def test_all_p_at_least_t_accepts_without_llm():
    jev, llm = FakeJev({"aussage_0": 0.98, COMPLETENESS_KEY: 0.95}), FakeLlm()
    j = judge_turn(jev, llm, turn(), None, FACTS, T)
    assert (j.accepted, j.decided_by, j.reason) == (True, "jev", None)
    assert llm.calls == []


def test_rejection_asks_llm_only_for_the_reason_and_always_rejects():
    jev, llm = FakeJev({"aussage_0": 0.03, COMPLETENESS_KEY: 0.99}), FakeLlm(supported=True)
    j = judge_turn(jev, llm, turn(), None, FACTS, T)
    assert (j.accepted, j.decided_by, j.reason) == (False, "jev", "Ablehnungsgrund.")
    _, user, schema = llm.calls[0]
    assert "supported" not in schema["properties"]
    assert "Aussage nicht belegt: Der Fuchs wiegt acht Kilo." in user


@pytest.mark.parametrize("supported", [True, False])
def test_uncertain_goes_to_llm_verdict(supported):
    jev, llm = FakeJev({"aussage_0": 0.98, COMPLETENESS_KEY: 0.6}), FakeLlm(supported)
    j = judge_turn(jev, llm, turn(), None, FACTS, T)
    assert (j.accepted, j.decided_by) == (supported, "llm_judge")
    assert j.jev_p[COMPLETENESS_KEY] == 0.6


@pytest.mark.parametrize("error", [JevOverlengthError("x"), JevTransientError("y")])
def test_overlength_and_transient_errors_go_to_llm_verdict(error):
    j = judge_turn(FakeJev(error=error), FakeLlm(True), turn(), None, FACTS, T)
    assert (j.accepted, j.decided_by, j.jev_p) == (True, "llm_judge", {})
    assert type(error).__name__ in j.jev_error


def test_abort_error_propagates():
    with pytest.raises(JevAbortError):
        judge_turn(FakeJev(error=JevAbortError("401")), FakeLlm(), turn(), None, FACTS, T)


def test_llm_judge_sees_only_turn_facts_and_context_not_statements():
    llm = FakeLlm()
    t = turn(statements=[Statement("GEHEIME-AUSSAGE", ["1-0"])])
    text = format_judge_input(t, FACTS)
    assert "GEHEIME-AUSSAGE" not in text
    assert "[1-0] Der Fuchs wiegt acht Kilo.\n  Kontext: Davor." in text
    assert "1-1" not in text and llm.calls == []


def test_judge_dialog_passes_previous_turn():
    jev = FakeJev({COMPLETENESS_KEY: 0.99, "aussage_0": 0.99})
    turns = [turn(0), turn(1)]
    judgements = judge_dialog(jev, FakeLlm(), turns, FACTS, T)
    assert len(judgements) == 2
    assert "vorheriger_redebeitrag" not in jev.calls[0][0]
    assert jev.calls[1][0]["vorheriger_redebeitrag"] == turns[0].text


# ---- gate after judging


def test_used_facts_count_only_accepted_turns():
    turns = [turn(0, ids=("1-0",)), turn(1, ids=("1-1",))]
    judgements = [TurnJudgement(0, True, "jev"), TurnJudgement(1, False, "jev")]
    assert used_in_accepted(turns, judgements, FACTS) == ["1-0"]


def test_judged_facts_gate_boundary():
    assert judged_facts_gate(10, 10) is None
    assert "D5" in judged_facts_gate(9, 10)
