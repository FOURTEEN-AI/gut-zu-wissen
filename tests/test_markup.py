import pytest

from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.markup import find_markup, markup_problems


@pytest.mark.parametrize(
    "text",
    [
        "Der Turm steht in [[Paris]].",
        "Siehe [https://example.org Seite].",
        "Er wurde {{Lebensdaten|1832|1923}} alt.",
        "Ein Tabellenrest {| class=wikitable.",
        "Ziel|Text",
        "Das ist '''fett''' gedruckt.",
        "== Geschichte ==",
        "Höhe<ref>Quelle</ref> dreihundert Meter.",
        "Zeile<br />Umbruch.",
        "Kommentar <!-- versteckt --> im Text.",
        "Dreihundert&nbsp;Meter.",
        "Ein Zeichen &#8211; im Text.",
        "__NOTOC__ Text.",
        "Signiert ~~~~",
        "* Punkt einer Liste",
        "Erste Zeile.\n# Nummerierte Zeile",
        ": eingerückt",
        "----",
        "Der Turm ist dreihundert Meter hoch.[1]",
    ],
)
def test_markup_is_found(text):
    assert find_markup(text)


@pytest.mark.parametrize(
    "text",
    [
        "Hallo und willkommen bei „Gut zu wissen“!",
        "Ein wichtiger Hinweis: Diese Folge ist mit KI entstanden – von der Auswahl der Fakten.",
        "Die Sprache C# entstand später, und C++ auch.",
        "Er sagte: „Das ist’s.“ Und dann? Drei mal vier ist zwölf; so einfach.",
        "Die Temperatur liegt unter null, also kälter als drei Grad (gemessen im Winter).",
        "Das Verhältnis ist eins zu zwei, also a < b und b > c.",
        "Rock & Roll war damals neu.",
        "Ein Gedankenstrich - mitten im Satz.",
        "Der Name lautet [sic] so.",
    ],
)
def test_plain_spoken_text_passes(text):
    assert find_markup(text) == []


def test_finding_names_rule_and_match():
    assert find_markup("In [[Paris]].") == ["Wiki-Link '[['"]


def test_markup_problems_name_the_turn():
    turns = [Turn(0, "Lena", "Sauber.", []), Turn(1, "Felix", "Mit [[Link]].", [])]
    problems = markup_problems(turns)
    assert [p.turn_index for p in problems] == [1]
    assert problems[0].reason.startswith("Markup im Text: Wiki-Link")
