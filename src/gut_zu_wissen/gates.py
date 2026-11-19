"""Deterministic gates for article selection (pure functions)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

CATEGORY_PREFIX = "Kategorie:"
PERSON_CATEGORIES = frozenset({"Mann", "Frau", "Divers", "Geschlecht unbekannt"})
DEAD_CATEGORY_PREFIX = "Gestorben"


def _names(categories: Iterable[str]) -> set[str]:
    return {c.removeprefix(CATEGORY_PREFIX).strip() for c in categories}


def is_living_person(categories: Iterable[str]) -> bool:
    """Living-person gate (design): person article without any "Gestorben" category.

    Person article: category Mann, Frau, Divers or Geschlecht unbekannt. Any category starting
    with "Gestorben" (including fuzzy dates) counts as dead. Unclear cases count as living,
    so a person article without a death category is blocked.
    """
    names = _names(categories)
    if not names & PERSON_CATEGORIES:
        return False
    return not any(n.startswith(DEAD_CATEGORY_PREFIX) for n in names)


def _add_months(moment: datetime, months: int) -> datetime:
    total = moment.year * 12 + (moment.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    # Clamp the day to the last day of the target month.
    for day in (moment.day, 30, 29, 28):
        try:
            return moment.replace(year=year, month=month, day=day)
        except ValueError:
            continue
    raise AssertionError("unreachable")


def repeat_gate_allows(last_published: datetime | None, now: datetime, months: int) -> bool:
    """Repeat gate (D4): a new episode is allowed `months` after the last publication date.

    No earlier episode: allowed. Exactly at the boundary: allowed.
    """
    if last_published is None:
        return True
    return now >= _add_months(last_published, months)
