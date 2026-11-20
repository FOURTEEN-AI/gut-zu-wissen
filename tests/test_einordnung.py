import json
from pathlib import Path

from gut_zu_wissen.einordnung import (
    first_prose_sentence,
    intro_candidates,
    pinned_ids,
    render_candidates,
    select_einordnung,
)
from gut_zu_wissen.run import load_config
from gut_zu_wissen.segmentation import Segment

BERNSTEIN = Path(__file__).parent / "fixtures" / "wikipedia" / "bernstein_intro.json"


def seg(i: int, raw: str, section: str = "", type_: str = "sentence") -> Segment:
    return Segment(f"1-{i:04d}", type_, section, raw)


# A synthetic article: infobox rows, the intro, then a section.
SEGMENTS = [
    seg(0, "Infobox | Ort = Sibirien", type_="infobox_row"),
    seg(1, "Das Ereignis war eine große Explosion im Jahr 1908."),
    seg(2, "Kurz."),  # fails the word gate
    seg(3, "Gleicher Satz kommt im Artikel zweimal vor."),  # not unique
    seg(4, "Sie fand in der Nähe eines Flusses in Sibirien statt."),
    seg(5, "Die Explosion ereignete sich am Morgen gegen sieben Uhr."),
    seg(6, "Bäume wurden auf einer großen Fläche umgeknickt."),
    seg(7, "Ein weiterer Satz der Einleitung ist hier zu lesen."),
    seg(8, "Gleicher Satz kommt im Artikel zweimal vor.", "Forschung"),
    seg(9, "Ein Satz im Abschnitt Forschung mit genug Wörtern.", "Forschung"),
]
PLAIN = {s.id: s.raw for s in SEGMENTS}


def ids(facts) -> list[str]:
    return [f.segment_id for f in facts]


def test_candidates_are_unique_intro_sentences_in_article_order():
    candidates = intro_candidates(SEGMENTS, 3)
    assert [s.id for s, _ in candidates] == [
        "1-0001",
        "1-0002",
        "1-0004",
        "1-0005",
        "1-0006",
        "1-0007",
    ]


def test_context_is_the_preceding_intro_sentences_at_most_max():
    context = {s.id: c for s, c in intro_candidates(SEGMENTS, 3)}
    assert context["1-0001"] == []  # the infobox row before it is no context
    assert context["1-0002"] == ["1-0001"]
    assert context["1-0004"][:2] == ["1-0001", "1-0002"]
    assert {s.id: c for s, c in intro_candidates(SEGMENTS[:4], 3)}["1-0002"] == ["1-0001"]
    two_before = {s.id: c for s, c in intro_candidates(SEGMENTS[:5], 3)}["1-0003"]
    assert two_before == ["1-0001", "1-0002"]  # fewer than the maximum precede it
    assert context["1-0004"] == ["1-0001", "1-0002", "1-0003"]  # a non-unique one still counts
    assert context["1-0007"] == ["1-0004", "1-0005", "1-0006"]
    assert all(c == [] for _, c in intro_candidates(SEGMENTS, 0))


def test_selection_skips_failed_gates_takes_the_next_and_stops_at_the_maximum():
    facts = select_einordnung(intro_candidates(SEGMENTS, 3), PLAIN, 4, 4)
    assert ids(facts) == ["1-0001", "1-0004", "1-0005", "1-0006"]
    first = facts[1]
    assert (first.type, first.section, first.plain) == ("sentence", "", PLAIN["1-0004"])
    assert first.context_plain == [PLAIN["1-0001"], PLAIN["1-0002"], PLAIN["1-0003"]]
    assert ids(select_einordnung(intro_candidates(SEGMENTS, 3), PLAIN, 4, 2)) == [
        "1-0001",
        "1-0004",
    ]


def test_excluded_facts_are_skipped_like_a_failed_gate():
    facts = select_einordnung(intro_candidates(SEGMENTS, 3), PLAIN, 4, 4, {"1-0004"})
    assert ids(facts) == ["1-0001", "1-0005", "1-0006", "1-0007"]


def test_no_intro_no_facts_and_no_render():
    body = [seg(0, "Ein Satz im Abschnitt mit genug Wörtern.", "A")]

    class NoRenderer:
        def render_html(self, title, snippets):
            raise AssertionError("no render without candidates")

    assert intro_candidates(body, 3) == []
    assert render_candidates(NoRenderer(), "T", body, []) == {}


def test_render_is_one_batch_without_refs():
    segments = [
        seg(0, "Erster Satz der Einleitung.<ref>x</ref>"),
        seg(1, "Zweiter Satz der Einleitung."),
    ]
    calls = []

    class EchoRenderer:
        def render_html(self, title, snippets):
            calls.append(snippets)
            return [f"<p>{s}</p>" for s in snippets]

    plain = render_candidates(EchoRenderer(), "T", segments, intro_candidates(segments, 3))
    assert len(calls) == 1
    assert plain == {
        "1-0000": "Erster Satz der Einleitung.",
        "1-0001": "Zweiter Satz der Einleitung.",
    }


def test_first_prose_sentence_skips_infobox_rows_and_other_sections():
    assert first_prose_sentence(SEGMENTS) == "1-0001"
    assert first_prose_sentence([seg(0, "Ein Satz.", "A")]) is None


def test_first_prose_sentence_is_number_one_even_when_excluded():
    candidates = intro_candidates(SEGMENTS, 3)
    facts = select_einordnung(candidates, PLAIN, 4, 2, {"1-0001", "1-0004"}, "1-0001")
    assert ids(facts) == ["1-0001", "1-0005"]  # D6 still applies to the others
    assert pinned_ids(facts, "1-0001") == frozenset({"1-0001"})


def test_a_first_prose_sentence_that_fails_the_gates_is_not_pinned():
    segments = [seg(0, "Kurz."), *SEGMENTS[1:]]
    candidates = intro_candidates(segments, 3)
    facts = select_einordnung(candidates, {s.id: s.raw for s in segments}, 4, 2, (), "1-0000")
    assert ids(facts) == ["1-0001", "1-0004"]
    assert pinned_ids(facts, "1-0000") == frozenset()
    assert pinned_ids([], None) == frozenset()


def test_real_config_offers_two_einordnung_facts():
    assert load_config()["quotas"]["einordnung_facts_max"] == 2


def bernstein() -> tuple[list[Segment], dict]:
    data = json.loads(BERNSTEIN.read_text())
    return [Segment(**s) for s in data["segments"]], data


def test_bernstein_definition_stays_einordnung_one_after_the_d6_exclusion():
    # Regression (archive 2026-11-20_bernstein): the intro starts with 17 infobox rows; the
    # definition sentence 0017 was Einordnung #1 in conversation 1, but the rejected closing
    # summary cited it, D6 excluded it, and conversations 2 and 3 got 0018-0021 instead.
    segments, data = bernstein()
    quotas = load_config()["quotas"]
    first = first_prose_sentence(segments)
    assert first == "271638927-0017"
    candidates = intro_candidates(segments, quotas["max_context_segments"])
    excluded = data["excluded_after_conversation_1"]
    assert first in excluded
    args = (candidates, data["plain"], quotas["min_fact_words"], quotas["einordnung_facts_max"])
    assert ids(select_einordnung(*args, excluded)) == ["271638927-0018", "271638927-0019"]
    facts = select_einordnung(*args, excluded, first)
    assert ids(facts) == ["271638927-0017", "271638927-0018"]
    assert facts[0].plain.startswith("Bernstein bezeichnet einen seit Jahrtausenden bekannten")
    assert facts[0].context_ids == []  # the infobox rows before it are no context
    assert pinned_ids(facts, first) == frozenset({first})
    assert ids(select_einordnung(*args, (), first)) == ["271638927-0017", "271638927-0018"]
