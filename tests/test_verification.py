import pytest

from gut_zu_wissen.verification import (
    DECISION_PROMPT,
    DECISION_SCHEMA,
    NO_ANSWER,
    Deviation,
    RetryAction,
    apply_decisions,
    cardinal,
    changes_statement,
    decade,
    decide_deviations,
    diff,
    format_decision_input,
    ordinal,
    retry_action,
    tokenize,
)


class FakeLlm:
    def __init__(self, answer: dict) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str, dict]] = []

    def complete_json(self, system: str, user: str, schema: dict, **cache) -> dict:
        self.calls.append((system, user, schema))
        return self.answer


def deviations(n: int) -> list[Deviation]:
    return [Deviation(i, f"soll{i}", f"ist{i}", "vor", "nach") for i in range(n)]


def test_decision_input_lists_script_transcript_and_numbered_deviations():
    found = [Deviation(0, "neun", "", "Es waren", "Leute"), Deviation(1, "", "äh", "", "")]
    text = format_decision_input([("Lena", "Es waren neun Leute.")], "Es waren Leute äh.", found)
    assert text == (
        "Skript des Abschnitts:\nLena: Es waren neun Leute.\n\n"
        "Transkript der Spracherkennung:\nEs waren Leute äh.\n\n"
        "Abweichungen:\n"
        "0. Skript: „Es waren [neun] Leute“ – gehört: „(nichts)“\n"
        "1. Skript: „[(nichts)]“ – gehört: „äh“"
    )


def test_decisions_are_applied_per_index():
    found = deviations(2)
    apply_decisions(
        found,
        {
            "decisions": [
                {"index": 1, "changes_statement": True, "reason": "Andere Zahl."},
                {"index": 0, "changes_statement": False, "reason": "Schreibweise."},
                {"index": 0, "changes_statement": True, "reason": "Zweite Antwort."},
                {"index": 7, "changes_statement": True, "reason": "Unbekannt."},
            ]
        },
    )
    assert [(d.changes_statement, d.reason) for d in found] == [
        (False, "Schreibweise."),
        (True, "Andere Zahl."),
    ]
    assert changes_statement(found)
    assert not changes_statement(found[:1])


def test_undecided_deviation_counts_as_a_change():
    found = deviations(2)
    apply_decisions(found, {"decisions": [{"index": 0, "changes_statement": False, "reason": ""}]})
    assert (found[1].changes_statement, found[1].reason) == (True, NO_ANSWER)
    assert changes_statement(found)
    assert changes_statement(deviations(1))  # not decided at all


def test_one_llm_call_per_section_and_none_without_deviations():
    llm = FakeLlm({"decisions": [{"index": 0, "changes_statement": False, "reason": "ok"}]})
    decide_deviations(llm, [("Felix", "Text.")], "Text.", [])
    assert llm.calls == []
    found = deviations(1)
    decide_deviations(llm, [("Felix", "Text.")], "Text.", found)
    assert len(llm.calls) == 1
    system, user, schema = llm.calls[0]
    assert system == DECISION_PROMPT and schema == DECISION_SCHEMA
    assert "0. Skript: „vor [soll0] nach“ – gehört: „ist0“" in user
    assert found[0].changes_statement is False


def norms(text: str) -> list[str]:
    return [t.norm for t in tokenize(text)]


@pytest.mark.parametrize(
    ("word", "value"),
    [
        ("null", 0),
        ("eins", 1),
        ("ein", 1),
        ("eine", 1),
        ("zwo", 2),
        ("sechzehn", 16),
        ("siebzehn", 17),
        ("einundzwanzig", 21),
        ("dreissig", 30),  # casefolded "dreißig"
        ("neunundneunzig", 99),
        ("hundert", 100),
        ("einhundert", 100),
        ("hunderteins", 101),
        ("dreihundertvierundzwanzig", 324),
        ("achtzehnhundertneunundachtzig", 1889),
        ("neunzehnhundert", 1900),
        ("tausend", 1000),
        ("zweitausendzwanzig", 2020),
        ("zehntausend", 10_000),
        ("hunderttausend", 100_000),
        ("neunhundertneunundneunzigtausendneunhundertneunundneunzig", 999_999),
    ],
)
def test_cardinals(word, value):
    assert cardinal(word) == value


@pytest.mark.parametrize(
    "word", ["einfach", "tausende", "hunderte", "zweiundzwanzigig", "undzwanzig", "nullhundert"]
)
def test_non_numbers_are_not_cardinals(word):
    assert cardinal(word) is None


@pytest.mark.parametrize(
    ("word", "value"),
    [
        ("ersten", 1),
        ("zweite", 2),
        ("dritten", 3),
        ("vierter", 4),
        ("siebte", 7),
        ("siebente", 7),
        ("achten", 8),
        ("elftes", 11),
        ("neunzehnten", 19),
        ("zwanzigsten", 20),
        ("einunddreissigsten", 31),
        ("hundertste", 100),
        ("hunderterste", 101),
        ("hundertachte", 108),
        ("tausendsten", 1000),
    ],
)
def test_ordinals(word, value):
    assert ordinal(word) == value


@pytest.mark.parametrize("word", ["beste", "gerste", "weihnachten", "weste", "elfen", "e"])
def test_non_numbers_are_not_ordinals(word):
    assert ordinal(word) is None


def test_decades():
    assert decade("zwanziger") == 20
    assert decade("neunzehnhundertneunziger") == 1990
    assert decade("dreier") is None
    assert decade("lehrer") is None


def test_case_punctuation_quotes_and_hyphens_are_ignored():
    assert norms("„Gut zu wissen“ – der Wikipedia-Artikel!") == norms(
        "Gut zu Wissen, der Wikipedia Artikel."
    )


def test_apostrophes_inside_words_are_removed():
    assert norms("Wie geht’s?") == norms("wie gehts")


def test_sharp_s_and_ss_are_equal():
    assert norms("Straße") == norms("Strasse")


def test_spelled_numbers_equal_numerals():
    script = (
        "Im Jahr achtzehnhundertneunundachtzig war er dreihundertvierundzwanzig Meter hoch, "
        "zehntausend Menschen kamen am dritten Tag, in den zwanziger Jahren zwei Komma fünf "
        "Prozent."
    )
    heard = (
        "Im Jahr 1889 war er 324 Meter hoch, 10.000 Menschen kamen am 3. Tag, "
        "in den 20er Jahren 2,5 %."
    )
    assert norms(script) == norms(heard)


@pytest.mark.parametrize(
    ("numeral", "norm"),
    [
        ("10.000", "10000"),
        ("10 000", "10000"),
        ("1.000.000", "1000000"),
        ("2.5", "2,5"),
        ("1.000,5", "1000,5"),
        ("007", "7"),
        ("1990er", "1990er"),
    ],
)
def test_numeral_forms(numeral, norm):
    assert norms(numeral) == [norm]


def test_four_digit_numbers_are_not_joined():
    assert norms("1889 345") == ["1889", "345"]


def test_tokens_keep_their_written_form():
    tokens = tokenize("Zwei Komma fünf Meter")
    assert [(t.norm, t.raw) for t in tokens] == [("2,5", "Zwei Komma fünf"), ("meter", "Meter")]


def test_identical_text_has_no_deviation():
    text = "Der Turm ist dreihundert Meter hoch. Echt?"
    assert diff(text, "der Turm ist 300 Meter hoch, echt", 3) == []


def test_deviations_carry_words_and_context():
    script = "Der Turm wurde achtzehnhundertneunundachtzig in Paris eröffnet und war sehr hoch."
    heard = "Der Turm wurde 1898 in Paris eröffnet und war hoch. Danke."
    found = diff(script, heard, 2)
    assert [(d.index, d.expected, d.heard) for d in found] == [
        (0, "achtzehnhundertneunundachtzig", "1898"),
        (1, "sehr", ""),
        (2, "", "Danke"),
    ]
    assert (found[0].context_before, found[0].context_after) == ("Turm wurde", "in Paris")
    assert (found[1].context_before, found[1].context_after) == ("und war", "hoch")
    assert (found[2].context_before, found[2].context_after) == ("sehr hoch", "")
    assert found[0].changes_statement is None


def test_retry_action_is_the_audio_budget():
    assert retry_action(0, False, 2) is RetryAction.ACCEPT
    assert retry_action(2, False, 2) is RetryAction.ACCEPT
    assert retry_action(0, True, 2) is RetryAction.RESYNTHESIZE
    assert retry_action(1, True, 2) is RetryAction.RESYNTHESIZE
    assert retry_action(2, True, 2) is RetryAction.ESCALATE
    assert retry_action(0, True, 0) is RetryAction.ESCALATE


def test_decision_prompt_treats_near_homophones_of_names_as_stt_errors():
    # User decision 2026-11-20: "Indiana" heard as "Indianer" is a recognition error, not a change.
    assert "„Indianer“ statt „Indiana“" in DECISION_PROMPT
    assert "Das gilt nur bei fast gleichem Klang" in DECISION_PROMPT
    assert "Im Zweifel gilt die Aussage als verändert." in DECISION_PROMPT
