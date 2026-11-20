from gut_zu_wissen.pronunciation import canonicalize, respell

ENTRIES = [
    {
        "written": "Andikythira",
        "spoken": "Andikütira",
        "heard": ["Antikythera", "Antikytheros"],
    }
]


def test_respell_replaces_the_written_form_only() -> None:
    assert respell("Die Insel Andikythira liegt.", ENTRIES) == "Die Insel Andikütira liegt."


def test_respell_matches_whole_words_only() -> None:
    assert respell("Andikythiras Hafen", ENTRIES) == "Andikythiras Hafen"


def test_respell_without_entries_changes_nothing() -> None:
    assert respell("Hallo Welt", []) == "Hallo Welt"


def test_canonicalize_maps_heard_variants_to_the_written_form() -> None:
    assert canonicalize("vor Antikytheros und Antikythera", "vor Andikythira", ENTRIES) == (
        "vor Andikythira und Andikythira"
    )


def test_canonicalize_keeps_variants_that_the_script_contains() -> None:
    script = "Der Antikythera-Mechanismus von Andikythira"
    heard = "Der Antikythera-Mechanismus von Antikytheros"
    assert canonicalize(heard, script, ENTRIES) == ("Der Antikythera-Mechanismus von Andikythira")
