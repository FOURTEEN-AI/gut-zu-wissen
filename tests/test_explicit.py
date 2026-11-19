import pytest

from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.explicit import (
    EXPLICIT_PROMPT,
    EXPLICIT_SCHEMA,
    Category,
    ExplicitCheck,
    Finding,
    apply_answer,
    check_explicit,
    format_script,
    is_explicit,
    parse_findings,
)
from gut_zu_wissen.llm import LlmError

TURNS = [
    Turn(0, "Lena", "Der Rotfuchs jagt vor allem Mäuse.", ["5-0000"]),
    Turn(1, "Felix", "Wirklich? Erzähl mehr!", []),
]
FINDING = {"turn_index": 0, "category": "violence", "quote": "jagt", "reason": "Jagd."}


class FakeLlm:
    def __init__(self, answer: dict | None = None, error: Exception | None = None) -> None:
        self.answer, self.error = answer, error
        self.calls: list[tuple[str, str, dict]] = []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append((system, user, schema))
        if self.error:
            raise self.error
        assert self.answer is not None
        return self.answer


def test_script_lists_every_turn_with_index_and_speaker():
    assert format_script(TURNS) == (
        "Skript:\n0. Lena: Der Rotfuchs jagt vor allem Mäuse.\n1. Felix: Wirklich? Erzähl mehr!"
    )


def test_clean_without_findings_is_not_explicit():
    assert is_explicit("clean", []) is False


@pytest.mark.parametrize("verdict", ["explicit", "uncertain", "", None, "sauber", 0])
def test_any_other_verdict_is_explicit(verdict):
    assert is_explicit(verdict, []) is True


def test_a_finding_makes_a_clean_verdict_explicit():
    assert is_explicit("clean", [Finding(0, "drugs", "Wein", "Alkohol.")]) is True


def test_findings_outside_the_schema_still_count():
    findings = parse_findings({"findings": [{"category": "profanity"}, "Fluch", FINDING]})
    assert findings == [
        Finding(None, "profanity", "", ""),
        Finding(None, "", "Fluch", ""),
        Finding(0, "violence", "jagt", "Jagd."),
    ]
    assert parse_findings({}) == [] and parse_findings({"findings": None}) == []
    assert parse_findings({"findings": [{"turn_index": True}]})[0].turn_index is None


@pytest.mark.parametrize(
    ("answer", "explicit"),
    [
        ({"findings": [], "verdict": "clean", "reason": "Nichts."}, False),
        ({"findings": [], "verdict": "uncertain", "reason": "Unklar."}, True),
        ({"findings": [FINDING], "verdict": "explicit", "reason": "Gewalt."}, True),
        ({"findings": [FINDING], "verdict": "clean", "reason": "Widerspruch."}, True),
        ({}, True),  # no verdict at all
    ],
)
def test_answer_is_written_into_the_record(answer, explicit):
    record = ExplicitCheck("model")
    apply_answer(record, answer)
    assert record.explicit is explicit
    assert record.verdict == answer.get("verdict")
    assert record.reason == answer.get("reason", "")


def test_one_call_on_the_whole_script():
    llm = FakeLlm({"findings": [], "verdict": "clean", "reason": "Nichts davon."})
    record = ExplicitCheck("claude-sonnet-5-5")
    check_explicit(record, llm, TURNS)
    ((system, user, schema),) = llm.calls
    assert system == EXPLICIT_PROMPT and schema == EXPLICIT_SCHEMA
    assert user == format_script(TURNS)
    assert record.to_dict() == {
        "model": "claude-sonnet-5-5",
        "explicit": False,
        "verdict": "clean",
        "findings": [],
        "reason": "Nichts davon.",
        "error": None,
    }


def test_failed_call_propagates_and_leaves_the_record_undecided():
    record = ExplicitCheck("model")
    with pytest.raises(LlmError):
        check_explicit(record, FakeLlm(error=LlmError("refused")), TURNS)
    assert record.explicit is None


def test_schema_offers_the_four_d7_categories_and_three_verdicts():
    item = EXPLICIT_SCHEMA["properties"]["findings"]["items"]
    assert item["properties"]["category"]["enum"] == [c.value for c in Category]
    assert {c.value for c in Category} == {"violence", "sexuality", "drugs", "profanity"}
    assert EXPLICIT_SCHEMA["properties"]["verdict"]["enum"] == ["explicit", "clean", "uncertain"]
    assert "Im Zweifel" in EXPLICIT_PROMPT
