import pytest

from gut_zu_wissen.verification import (
    RetryAction,
    cardinal,
    decade,
    diff,
    ordinal,
    retry_action,
    tokenize,
)


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
