import pytest

from gut_zu_wissen.aliases import (
    ALIAS_PREFIX,
    alias_range,
    build_aliases,
    normalize,
    replace_segment_ids,
    resolve,
    to_aliases,
    unknown_alias_problem,
)

IDS = ["270155664-0071", "270155664-0041", "99-0003"]
ALIASES = build_aliases(IDS)


def test_aliases_follow_the_offer_order_and_start_at_one():
    assert ALIASES == {"F1": "270155664-0071", "F2": "270155664-0041", "F3": "99-0003"}
    assert all(a.startswith(ALIAS_PREFIX) for a in ALIASES)


def test_a_repeated_segment_id_keeps_its_first_alias():
    assert build_aliases(["a", "b", "a", "c"]) == {"F1": "a", "F2": "b", "F3": "c"}


def test_an_empty_offer_has_no_aliases():
    assert build_aliases([]) == {}


@pytest.mark.parametrize("token", ["F2", "f2", "[F2]", " F2 ", "[ f2 ]", "F02"])
def test_alias_spellings_resolve_to_the_segment_id(token):
    assert resolve(token, ALIASES) == "270155664-0041"


def test_an_offered_segment_id_written_out_is_accepted():
    assert resolve(" [270155664-0071] ", ALIASES) == "270155664-0071"


@pytest.mark.parametrize(
    ("token", "kept"),
    [
        ("F53", "F53"),
        ("[f53]", "F53"),
        ("F0", "F0"),
        ("270155664-0001", "270155664-0001"),
        ("x", "x"),
    ],
)
def test_unknown_tokens_are_kept_as_written_never_guessed(token, kept):
    assert resolve(token, ALIASES) == kept
    assert resolve(token, ALIASES) not in ALIASES.values()


def test_normalize_only_canonicalises_alias_shaped_tokens():
    assert normalize(" [f7] ") == "F7" and normalize("F7a") == "F7a"


def test_forward_mapping_keeps_order_and_drops_ids_without_alias():
    assert to_aliases(["99-0003", "9-9", "270155664-0071"], ALIASES) == ["F3", "F1"]


def test_forward_and_back_mapping_round_trip():
    assert [resolve(a, ALIASES) for a in to_aliases(IDS, ALIASES)] == IDS


def test_alias_range():
    assert alias_range(build_aliases(str(i) for i in range(29))) == "F1–F29"
    assert alias_range(build_aliases(["a"])) == "F1"


def test_unknown_alias_problem_text_is_german_and_actionable():
    aliases = build_aliases(str(i) for i in range(29))
    assert unknown_alias_problem(["F53"], aliases) == (
        "Die Kennung F53 gibt es im Angebot nicht. Verwende nur Kennungen aus dem Angebot "
        "(F1–F29); was du mit keiner angebotenen Kennung belegen kannst, lässt du weg."
    )
    assert unknown_alias_problem(["F53", "F60"], aliases).startswith(
        "Die Kennungen F53, F60 gibt es im Angebot nicht."
    )


def test_offered_segment_ids_in_a_text_become_aliases():
    text = "Fakt 270155664-0041 sagt das nicht, 1-1 ist unbekannt."
    assert replace_segment_ids(text, ALIASES) == "Fakt F2 sagt das nicht, 1-1 ist unbekannt."
