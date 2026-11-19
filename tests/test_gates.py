from datetime import UTC, datetime

import pytest

from gut_zu_wissen.gates import is_living_person, repeat_gate_allows


def cats(*names: str) -> list[str]:
    return [f"Kategorie:{n}" for n in names]


@pytest.mark.parametrize("gender", ["Mann", "Frau", "Divers", "Geschlecht unbekannt"])
def test_person_without_death_category_is_living(gender):
    assert is_living_person(cats("Geboren 1990", gender))


@pytest.mark.parametrize(
    "death",
    ["Gestorben 1750", "Gestorben im 20. Jahrhundert", "Gestorben im 5. Jahrhundert v. Chr."],
)
def test_person_with_death_category_is_dead(death):
    assert not is_living_person(cats("Mann", death))


def test_non_person_article_passes():
    assert not is_living_person(cats("Haarwild", "Hunde"))


def test_no_categories_passes():
    assert not is_living_person([])


def test_unclear_case_counts_as_living():
    assert is_living_person(cats("Frau", "Politikerin"))


def utc(y, m, d):
    return datetime(y, m, d, tzinfo=UTC)


def test_repeat_without_earlier_episode_allowed():
    assert repeat_gate_allows(None, utc(2026, 11, 19), 6)


def test_repeat_before_boundary_blocked():
    assert not repeat_gate_allows(utc(2026, 5, 20), utc(2026, 11, 19), 6)


def test_repeat_at_boundary_allowed():
    assert repeat_gate_allows(utc(2026, 5, 19), utc(2026, 11, 19), 6)


def test_repeat_after_boundary_allowed():
    assert repeat_gate_allows(utc(2026, 1, 1), utc(2026, 11, 19), 6)


def test_repeat_month_end_clamped():
    assert repeat_gate_allows(utc(2026, 8, 31), utc(2027, 2, 28), 6)
    assert not repeat_gate_allows(utc(2026, 8, 31), utc(2027, 2, 27), 6)


def test_repeat_year_rollover():
    assert not repeat_gate_allows(utc(2026, 11, 19), utc(2027, 5, 18), 6)
    assert repeat_gate_allows(utc(2026, 11, 19), utc(2027, 5, 19), 6)
