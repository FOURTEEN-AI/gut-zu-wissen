from datetime import UTC, date, datetime

import pytest

from gut_zu_wissen.daily import (
    ArticleOfTheDayError,
    article_of_the_day,
    candidate_order,
    candidate_rejection,
    featured_titles,
    parse_chronology,
    parse_weekday_page,
    run_day,
)
from gut_zu_wissen.fetch import ArticleNotFoundError
from gut_zu_wissen.orchestrator import GateFailure, JevAbortFailure

DAY = date(2026, 11, 20)
CHRONOLOGY = """{{Navigationsleiste}}
=== November 2026 ===
* 20.11.2026 [[Volksmudschahedin]]
* 19.11.2026 [[Diplodia sapinea]]
* 14.11.2026 [[Totenhütte von Benzingerode]] (erneut)
* 12.11.2026 [[Völkerschau der „Amazonen von Dahomey“ 1888–1903|Völkerschau]]
"""
WEEKDAY = """<onlyinclude>{{Wikipedia:Hauptseite/Artikel des Tages|SPRACHE=
|DATUM            = 20.11.2026
|LEMMA            = Volksmudschahedin
|BILD             = File:x.jpg
"""


def test_chronology_gives_the_title_of_the_day():
    assert parse_chronology(CHRONOLOGY, DAY) == "Volksmudschahedin"
    assert parse_chronology(CHRONOLOGY, date(2026, 11, 14)) == "Totenhütte von Benzingerode"


def test_chronology_link_with_a_label_gives_the_target():
    assert parse_chronology(CHRONOLOGY, date(2026, 11, 12)).startswith("Völkerschau der")


def test_chronology_without_the_day_gives_none():
    assert parse_chronology(CHRONOLOGY, date(2026, 11, 21)) is None


def test_weekday_page_counts_only_for_its_own_date():
    assert parse_weekday_page(WEEKDAY, DAY) == "Volksmudschahedin"
    assert parse_weekday_page(WEEKDAY, date(2026, 11, 27)) is None
    assert parse_weekday_page("nothing", DAY) is None


def test_candidate_order_is_fixed_by_the_seed_and_leaves_out_the_excluded():
    titles = [f"A{n}" for n in range(30)]
    first = candidate_order(titles, {"A3"}, "2026-11-20")
    assert first == candidate_order(titles, {"A3"}, "2026-11-20")
    assert first != candidate_order(titles, {"A3"}, "2026-11-21")
    assert "A3" not in first and len(first) == 29


class PagesApi:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.titles: list[str] = []

    def query(self, **params):
        title = params["titles"]
        self.titles.append(title)
        if title not in self.pages:
            return {"query": {"pages": [{"title": title, "missing": True}]}}
        content = self.pages[title]
        return {"query": {"pages": [{"revisions": [{"slots": {"main": {"content": content}}}]}]}}


def test_article_of_the_day_reads_the_chronology_first():
    api = PagesApi({"Wikipedia:Hauptseite/Artikel des Tages/Chronologie 2026": CHRONOLOGY})
    assert article_of_the_day(api, DAY) == "Volksmudschahedin"
    assert len(api.titles) == 1


def test_article_of_the_day_falls_back_to_the_weekday_page():
    api = PagesApi({"Wikipedia:Hauptseite/Artikel des Tages/Freitag": WEEKDAY})
    assert article_of_the_day(api, DAY) == "Volksmudschahedin"


def test_article_of_the_day_missing_everywhere_is_an_error():
    with pytest.raises(ArticleOfTheDayError):
        article_of_the_day(PagesApi({}), DAY)


class CategoryApi:
    def query(self, **params):
        pages = {
            ("Kategorie:Wikipedia:Exzellent", None): (["A", "B"], "next"),
            ("Kategorie:Wikipedia:Exzellent", "next"): (["C"], None),
            ("Kategorie:Wikipedia:Lesenswert", None): (["D"], None),
        }
        titles, more = pages[(params["cmtitle"], params.get("cmcontinue"))]
        data = {"query": {"categorymembers": [{"title": t} for t in titles]}}
        if more:
            data["continue"] = {"cmcontinue": more}
        return data


def test_featured_titles_follow_the_continuation_in_both_categories():
    assert featured_titles(CategoryApi()) == ["A", "B", "C", "D"]


NOW = datetime(2026, 11, 20, 3, 17, tzinfo=UTC)


@pytest.mark.parametrize(
    ("categories", "last", "reason"),
    [
        (None, None, "does not exist"),
        (["Kategorie:Mann"], None, "living person"),
        (["Kategorie:Mann", "Kategorie:Gestorben 1990"], None, None),
        (["Kategorie:Fluss"], datetime(2026, 10, 1, tzinfo=UTC), "episode published"),
        (["Kategorie:Fluss"], datetime(2025, 1, 1, tzinfo=UTC), None),
        (["Kategorie:Fluss"], None, None),
    ],
)
def test_candidate_rejection(categories, last, reason):
    got = candidate_rejection(categories, last, NOW, 6)
    assert (got is None) if reason is None else (reason in got)


def make_run(outcomes: dict):
    calls = []

    def run_episode(title):
        calls.append(title)
        outcome = outcomes[title]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return run_episode, calls


def test_article_of_the_day_succeeds_without_substitute():
    run_episode, calls = make_run({"AdT": "episode-1"})
    result = run_day("AdT", iter(["S1"]), 2, run_episode, lambda t: None)
    assert (result.episode_id, calls) == ("episode-1", ["AdT"])


def test_gate_failure_moves_on_to_the_substitute():
    run_episode, calls = make_run({"AdT": GateFailure("too few facts"), "S1": "episode-2"})
    result = run_day("AdT", iter(["S1", "S2"]), 2, run_episode, lambda t: None)
    assert (result.episode_id, calls) == ("episode-2", ["AdT", "S1"])
    assert "too few facts" in result.notes[0]


def test_unknown_article_moves_on_to_the_substitute():
    run_episode, _ = make_run({"AdT": ArticleNotFoundError("x"), "S1": "episode-2"})
    assert run_day("AdT", iter(["S1"]), 2, run_episode, lambda t: None).episode_id == "episode-2"


def test_rejected_candidates_do_not_count_towards_the_limit():
    run_episode, calls = make_run({"AdT": GateFailure("x"), "S3": "episode-3"})
    check = lambda t: "living person" if t in ("S1", "S2") else None  # noqa: E731
    result = run_day("AdT", iter(["S1", "S2", "S3"]), 1, run_episode, check)
    assert (result.episode_id, calls) == ("episode-3", ["AdT", "S3"])
    assert sum("rejected" in n for n in result.notes) == 2


def test_at_most_two_substitutes_then_the_episode_is_dropped():
    failing = {t: GateFailure(f"{t} failed") for t in ("AdT", "S1", "S2", "S3")}
    run_episode, calls = make_run(failing)
    result = run_day("AdT", iter(["S1", "S2", "S3"]), 2, run_episode, lambda t: None)
    assert result.episode_id is None and calls == ["AdT", "S1", "S2"]
    assert "dropped" in result.dropped_reason


def test_jev_abort_drops_the_episode_without_substitute():
    run_episode, calls = make_run({"AdT": JevAbortFailure("Jev aborted the run: x")})
    result = run_day("AdT", iter(["S1"]), 2, run_episode, lambda t: None)
    assert result.episode_id is None and calls == ["AdT"]
    assert "Jev abort" in result.dropped_reason


def test_no_allowed_substitute_drops_the_episode():
    run_episode, _ = make_run({"AdT": GateFailure("x")})
    result = run_day("AdT", iter(["S1"]), 2, run_episode, lambda t: "living person")
    assert result.episode_id is None and "No allowed substitute" in result.dropped_reason
