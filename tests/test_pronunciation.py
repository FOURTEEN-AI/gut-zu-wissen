import json

from gut_zu_wissen.pronunciation import canonicalize, load_episode_entries, respell, with_additions

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


def test_episode_entries_are_empty_without_a_file(tmp_path) -> None:
    from gut_zu_wissen.pronunciation import load_episode_entries

    assert load_episode_entries(tmp_path) == []


def test_episode_entries_are_read_from_the_folder(tmp_path) -> None:
    from gut_zu_wissen.pronunciation import load_episode_entries

    (tmp_path / "pronunciation.json").write_text(
        '[{"written": "Hublot", "spoken": "Ublo", "heard": ["Hublo"]}]', encoding="utf-8"
    )
    assert load_episode_entries(tmp_path) == [
        {"written": "Hublot", "spoken": "Ublo", "heard": ["Hublo"]}
    ]


import pytest  # noqa: E402


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"written": "x"}',
        '[{"written": "x"}]',
        '[{"written": "x", "spoken": "y", "extra": 1}]',
        '[{"written": "x", "spoken": "", "heard": []}]',
        '[{"written": "x", "spoken": "y", "heard": "z"}]',
    ],
)
def test_invalid_episode_file_is_refused(tmp_path, content: str) -> None:
    from gut_zu_wissen.pronunciation import PronunciationError, load_episode_entries

    (tmp_path / "pronunciation.json").write_text(content, encoding="utf-8")
    with pytest.raises(PronunciationError):
        load_episode_entries(tmp_path)


def test_merge_puts_episode_entries_first_and_replaces_the_same_word() -> None:
    from gut_zu_wissen.pronunciation import merge

    config = [{"written": "A", "spoken": "a1"}, {"written": "B", "spoken": "b"}]
    episode = [{"written": "A", "spoken": "a2"}, {"written": "C", "spoken": "c"}]
    assert merge(config, episode) == [
        {"written": "A", "spoken": "a2"},
        {"written": "C", "spoken": "c"},
        {"written": "B", "spoken": "b"},
    ]


def test_episode_entries_may_carry_a_note(tmp_path) -> None:
    entry = {"written": "Rainald", "spoken": "Rainald", "heard": ["Reinhard"], "note": "x"}
    (tmp_path / "pronunciation.json").write_text(json.dumps([entry]), encoding="utf-8")
    assert load_episode_entries(tmp_path) == [entry]


def test_additions_extend_an_existing_entry_or_are_appended() -> None:
    entries = [{"written": "Rainald", "spoken": "Rainald", "heard": ["Reinhard"]}]
    additions = [
        {"written": "Rainald", "spoken": "Rainald", "heard": ["Reinald"], "note": "n"},
        {"written": "Patrizier", "spoken": "Patrizier", "heard": ["Patricia"], "note": "n"},
    ]
    out = with_additions(entries, additions)
    assert out[0]["heard"] == ["Reinhard", "Reinald"]
    assert out[1]["written"] == "Patrizier"
    assert entries[0]["heard"] == ["Reinhard"]  # the input stays unchanged
