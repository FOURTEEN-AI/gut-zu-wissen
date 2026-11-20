import pytest

from gut_zu_wissen.extraction import Fact
from gut_zu_wissen.linked import intro_wikitext
from gut_zu_wissen.ranking import (
    LINKED_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    bound_targets,
    format_facts,
    link_targets,
    offer_count,
    offer_gate,
    offer_summary,
    rank_facts,
    rank_linked_facts,
    ranked,
    select_ranked,
)


def fact(i: int, context: list[str] | None = None) -> Fact:
    return Fact(f"1-{i:04d}", "sentence", "", [], f"Fakt {i}.", context or [])


class FakeLlm:
    def __init__(self, ids: list) -> None:
        self.ids, self.calls = ids, []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append((system, user))
        return {"ranked_ids": self.ids}


def test_select_keeps_model_order_and_cuts_to_maximum():
    facts = [fact(i) for i in range(4)]
    chosen = select_ranked(["1-0003", "1-0001", "1-0002"], facts, 2)
    assert [f.segment_id for f in chosen] == ["1-0003", "1-0001"]


def test_select_drops_unknown_and_repeated_ids():
    facts = [fact(0), fact(1)]
    chosen = select_ranked(["9-9999", "1-0001", "1-0001", 7, "1-0000"], facts, 5)
    assert [f.segment_id for f in chosen] == ["1-0001", "1-0000"]


def test_model_may_return_fewer_than_minimum():
    assert [f.segment_id for f in select_ranked(["1-0001"], [fact(0), fact(1)], 5)] == ["1-0001"]


def test_rank_facts_sends_plain_text_and_context_and_range():
    llm = FakeLlm(["1-0001", "1-0000"])
    out = rank_facts(llm, [fact(0), fact(1, ["Davor."])], 2, 3)
    assert [f.segment_id for f in out] == ["1-0001", "1-0000"]
    system, user = llm.calls[0]
    assert "mindestens 2 Fakten, wenn es so viele gibt, und höchstens 3" in system
    assert user == "[1-0000] Fakt 0.\n[1-0001] Fakt 1.\n  Kontext: Davor."


def test_exclude_removes_facts_before_ranking():
    llm = FakeLlm(["1-0000", "1-0002"])
    out = rank_facts(llm, [fact(0), fact(1), fact(2)], 2, 5, exclude={"1-0001"})
    assert [f.segment_id for f in out] == ["1-0000", "1-0002"]
    assert "1-0001" not in llm.calls[0][1]


def test_excluded_fact_the_model_names_anyway_is_dropped():
    out = rank_facts(FakeLlm(["1-0001", "1-0000"]), [fact(0), fact(1)], 2, 5, exclude={"1-0001"})
    assert [f.segment_id for f in out] == ["1-0000"]


@pytest.mark.parametrize("count", [0, 1])
def test_no_model_call_below_two_candidates(count):
    llm = FakeLlm([])
    assert len(rank_facts(llm, [fact(i) for i in range(count)], 2, 5)) == count
    assert llm.calls == []


def test_rank_facts_cuts_answer_to_maximum():
    llm = FakeLlm([f"1-{i:04d}" for i in range(5)])
    out = rank_facts(llm, [fact(i) for i in range(5)], 2, 3)
    assert [f.segment_id for f in out] == ["1-0000", "1-0001", "1-0002"]


def test_prompt_states_range_order_and_rule_beyond_minimum():
    system = SYSTEM_PROMPT.format(minimum=20, maximum=40)
    assert "mindestens 20 Fakten, wenn es so viele gibt, und höchstens 40" in system
    assert "der überraschendste zuerst" in system
    assert "über 20 nur mit Fakten hinaus, die wirklich überraschend sind und keine Dopplungen" in (
        system.replace("\n", " ")
    )


def test_linked_prompt_names_main_topic_and_relevance_rule():
    system = " ".join(
        LINKED_SYSTEM_PROMPT.format(main_title="Axolotl", minimum=6, maximum=12).split()
    )
    assert "vom Wikipedia-Artikel „Axolotl“ (Hauptthema)" in system
    assert "Wähle nur Fakten, die direkt mit dem Hauptthema „Axolotl“ verbunden sind" in system
    assert "über das Hauptthema selbst oder über seine direkte Beziehung zum verlinkten" in system
    assert "Lass Fakten weg, die nur den verlinkten Gegenstand für sich beschreiben" in system
    assert "Biografie, Größe einer Institution, ein Ereignis ohne Bezug zum Hauptthema" in system
    assert "Unter den direkt verbundenen Fakten ordnest du nach Überraschung" in system
    assert (
        "mindestens 6 Fakten, wenn es so viele direkt verbundene gibt, und höchstens 12" in system
    )


def test_rank_linked_facts_passes_main_title_and_sources():
    llm = FakeLlm(["1-0001"])
    sources = {"1-0000": "Hauptmann von Köpenick", "1-0001": "Signal-Protokoll"}
    out = rank_linked_facts(llm, [fact(0), fact(1)], sources, "Axolotl", 6, 12)
    assert [f.segment_id for f in out] == ["1-0001"]
    system, user = llm.calls[0]
    assert "„Axolotl“" in system and "mindestens 6" in system
    assert user == (
        "[1-0000] (verlinkter Artikel: Hauptmann von Köpenick) Fakt 0.\n"
        "[1-0001] (verlinkter Artikel: Signal-Protokoll) Fakt 1."
    )


def test_rank_linked_facts_checks_a_single_candidate_and_skips_none():
    llm = FakeLlm([])
    assert rank_linked_facts(llm, [fact(0)], {"1-0000": "Wald"}, "Fuchs", 6, 12) == []
    assert len(llm.calls) == 1
    llm = FakeLlm([])
    assert rank_linked_facts(llm, [fact(0)], {}, "Fuchs", 6, 12, exclude={"1-0000"}) == []
    assert llm.calls == []


def test_rank_linked_facts_excludes_and_cuts_to_maximum():
    llm = FakeLlm(["1-0002", "1-0001", "1-0000"])
    sources = {f"1-{i:04d}": "Wald" for i in range(3)}
    facts = [fact(i) for i in range(3)]
    out = rank_linked_facts(llm, facts, sources, "Fuchs", 1, 1, exclude={"1-0002"})
    assert [f.segment_id for f in out] == ["1-0001"]
    assert "1-0002" not in llm.calls[0][1]


def test_format_facts_single_line_per_fact():
    assert format_facts([fact(0)]) == "[1-0000] Fakt 0."


def test_ranked_numbers_from_one():
    r = ranked([fact(5), fact(6)], "Fuchs", 7)
    assert [(x.rank, x.article, x.revid) for x in r] == [(1, "Fuchs", 7), (2, "Fuchs", 7)]
    assert r[0].to_dict()["fact"]["segment_id"] == "1-0005"


def test_link_targets():
    raw = [
        "Der [[Rotfuchs|Fuchs]] jagt im [[Wald#Aufbau|Wald]] und im [[Wald]].",
        "[[Datei:Fuchs.jpg|mini]] [[Kategorie:Tiere]] [[:Kategorie:Tiere]] [[Wikipedia:Hilfe]]",
        "Satz mit [[Star Wars: Episode IV]] und [[Eigener Titel]]<ref>[[Nur im Beleg]]</ref>.",
        "Unterstrich [[Mount_Everest]].",
    ]
    assert link_targets(raw, "Eigener Titel") == [
        "Rotfuchs",
        "Wald",
        "Star Wars: Episode IV",
        "Mount Everest",
    ]


def test_bound_targets_keeps_first_and_lists_rest():
    assert bound_targets(["A", "B", "C"], 2) == (["A", "B"], ["C"])
    assert bound_targets(["A"], 2) == (["A"], [])


def test_offer_count_flags_below_minimum():
    assert offer_count(20, 20, 40) == {"offered": 20, "min": 20, "max": 40, "below_min": False}
    assert offer_count(5, 6, 12)["below_min"] is True
    assert list(offer_count(1, 2, 3)) == ["offered", "min", "max", "below_min"]


def test_offer_summary():
    counts = {"main": offer_count(25, 20, 40), "linked": offer_count(4, 6, 12)}
    assert offer_summary(counts) == (
        "Offer: 25 main facts (range 20-40), 4 linked facts (range 6-12, below the minimum)"
    )


def test_offer_gate():
    assert offer_gate(10, 10) is None
    assert offer_gate(25, 10) is None
    message = offer_gate(9, 10)
    assert message and "9 facts offered" in message and "D5" in message


def test_intro_is_text_before_first_heading():
    text = "Einleitung.\n\nNoch ein Satz.\n\n== Abschnitt ==\nText.\n=== Unter ===\nMehr."
    assert intro_wikitext(text) == "Einleitung.\n\nNoch ein Satz."
    assert intro_wikitext("Nur Einleitung.") == "Nur Einleitung."
