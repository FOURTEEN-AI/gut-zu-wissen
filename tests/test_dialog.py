from gut_zu_wissen.dialog import (
    DIALOG_SCHEMA,
    Statement,
    Turn,
    check_turns,
    format_offer,
    parse_dialog,
    used_fact_ids,
    used_facts_gate,
    write_dialog,
)
from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.ranking import RankedFact

OFFER = {"1-0000", "1-0001", "2-0000"}


def rf(sid: str, article: str = "Haupt", context: list[str] | None = None) -> RankedFact:
    return RankedFact(
        1, article, int(sid[0]), Fact(sid, "sentence", "", [], f"Fakt {sid}.", context or [])
    )


def turn(i: int, speaker="Lena", text="Hallo.", ids=(), statements=()) -> Turn:
    return Turn(i, speaker, text, list(ids), [Statement(t, list(f)) for t, f in statements])


def test_format_offer_marks_linked_facts_and_context():
    out = format_offer([rf("1-0000", context=["Davor."])], [rf("2-0000", "Nebenartikel")])
    assert out == (
        "[1-0000] Fakt 1-0000.\n  Kontext: Davor.\n"
        "[2-0000] (verlinkter Artikel: Nebenartikel) Fakt 2-0000."
    )


def test_parse_keeps_order_and_tolerates_malformed_entries():
    answer = {
        "turns": [
            {
                "speaker": "Lena",
                "text": "A",
                "fact_ids": ["1-0000", 3],
                "statements": [{"text": "S", "fact_ids": ["1-0000"]}, "kaputt"],
            },
            "kaputt",
            {"speaker": "Felix", "text": "B"},
        ]
    }
    turns = parse_dialog(answer)
    assert [t.index for t in turns] == [0, 1]
    assert turns[0].fact_ids == ["1-0000"] and turns[0].statements == [Statement("S", ["1-0000"])]
    assert turns[1].fact_ids == [] and turns[1].statements == []


def test_valid_turns_pass_the_gate():
    turns = [turn(0, ids=["1-0000"], statements=[("S", ["1-0000"])]), turn(1, "Felix", "Echt?")]
    assert check_turns(turns, OFFER) == []


def test_gate_reports_unknown_ids_in_turn_and_statement():
    turns = [turn(0, ids=["9-9"], statements=[("S", ["8-8"])])]
    (problem,) = check_turns(turns, OFFER)
    assert problem.turn_index == 0 and "8-8" in problem.reason and "9-9" in problem.reason


def test_gate_reports_statement_without_ids_unknown_speaker_and_empty_text():
    turns = [turn(0, "Hans", " ", statements=[("S", [])])]
    reasons = [p.reason for p in check_turns(turns, OFFER)]
    assert len(reasons) == 3


def test_used_facts_counts_turn_and_statement_ids_once_ignoring_unknown():
    turns = [
        turn(0, ids=["1-0000", "9-9"], statements=[("S", ["1-0001"])]),
        turn(1, ids=["1-0000", "2-0000"]),
    ]
    assert used_fact_ids(turns, OFFER) == ["1-0000", "1-0001", "2-0000"]


def test_used_facts_gate_at_the_boundary():
    assert used_facts_gate(10, 10) is None
    assert "D5" in used_facts_gate(9, 10)


class FakeLlm:
    def __init__(self, answer: dict) -> None:
        self.answer, self.calls = answer, []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append((system, user, schema))
        return self.answer


def test_write_dialog_sends_offer_and_records_problems_and_use():
    answer = {
        "turns": [
            {
                "speaker": "Lena",
                "text": "A",
                "fact_ids": ["1-0000"],
                "statements": [{"text": "S", "fact_ids": ["1-0000"]}],
            },
            {"speaker": "Felix", "text": "B", "fact_ids": ["7-7"], "statements": []},
        ]
    }
    llm = FakeLlm(answer)
    dialog = write_dialog(llm, [rf("1-0000")], [rf("2-0000", "N")], 10)
    system, user, schema = llm.calls[0]
    assert "mindestens 10" in system and "[2-0000]" in user and schema is DIALOG_SCHEMA
    assert dialog.used_fact_ids == ["1-0000"]
    assert [p.turn_index for p in dialog.problems] == [1]
