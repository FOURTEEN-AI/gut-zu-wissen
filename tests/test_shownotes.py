from pathlib import Path

import pytest

from gut_zu_wissen.archive import EPISODE_FILE, read_json
from gut_zu_wissen.shownotes import (
    Source,
    episode_sources,
    feedback_mailto,
    history_url,
    permalink,
    revision_of,
    titles_by_revision,
    used_sources,
)

INDEX = "https://de.wikipedia.org/w/index.php"
ARCHIVE = Path(__file__).parents[1] / "archive" / "episodes"


def test_revision_of_a_fact_id():
    assert revision_of("272085229-0008") == 272085229


def test_titles_come_from_main_article_linked_articles_and_ranked_facts():
    ranking = {
        "linked_articles": [{"title": "Quantenmechanik", "revid": 2}],
        "linked_facts": [{"article": "Tabun", "revid": 3}],
        "main_facts": [{"article": "Helgoland", "revid": 1}],
    }
    assert titles_by_revision({"title": "Helgoland", "revid": 1}, ranking) == {
        1: "Helgoland",
        2: "Quantenmechanik",
        3: "Tabun",
    }


def test_main_article_comes_first_linked_ones_are_sorted_and_unused_ones_left_out():
    titles = {1: "Helgoland", 2: "Zebra", 3: "Ampel", 4: "Unbenutzt"}
    sources = used_sources(1, ["2-0001", "3-0002", "2-0003"], titles)
    assert sources == [Source("Helgoland", 1), Source("Ampel", 3), Source("Zebra", 2)]


def test_main_article_is_listed_even_without_a_used_fact():
    assert used_sources(1, [], {1: "A"}) == [Source("A", 1)]


def test_unknown_revision_is_an_error_not_a_silent_omission():
    with pytest.raises(ValueError, match="9"):
        used_sources(1, ["9-0001"], {1: "A"})


def test_permalink_and_history_link():
    source = Source("Tunguska-Ereignis", 270155664)
    assert permalink(INDEX, source) == f"{INDEX}?title=Tunguska-Ereignis&oldid=270155664"
    assert history_url(INDEX, source) == f"{INDEX}?title=Tunguska-Ereignis&action=history"
    assert permalink(INDEX, Source("Schlacht bei Bouvines", 1)).count("title=Schlacht_bei_Bouvines")


def test_feedback_mailto_carries_title_and_episode_id():
    url = feedback_mailto("a@b.de", "Bernstein", "2026-11-22_bernstein")
    assert (
        url == "mailto:a@b.de?subject=Gut%20zu%20wissen%3A%20Bernstein%20%282026-11-22_bernstein%29"
    )


@pytest.mark.parametrize(
    "folder",
    sorted(
        p.name
        for p in ARCHIVE.iterdir()
        if read_json(p / EPISODE_FILE)["status"] in ("completed", "published")
    ),
)
def test_every_completed_archived_episode_resolves_all_its_sources(folder):
    sources = episode_sources(ARCHIVE / folder)
    assert sources[0].title == read_json(ARCHIVE / folder / EPISODE_FILE)["title"]
    assert len({s.revid for s in sources}) == len(sources)
