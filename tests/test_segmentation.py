import json
import re
from pathlib import Path

import pytest

from gut_zu_wissen.segmentation import (
    Segment,
    default_tokenizer,
    scan,
    segment_article,
    split_pieces,
)

FIXTURES = Path(__file__).parent / "fixtures" / "wikipedia" / "spikes"
# Sentence segments per article of the S2 corpus (docs/spikes/s2-segmentation.md).
CORPUS_SENTENCES = {
    "Eiffelturm": 760,
    "FC Bayern München": 1222,
    "Johann Sebastian Bach": 589,
    "Mount Everest": 554,
    "Python (Programmiersprache)": 210,
    "Rotfuchs": 367,
    "Satz des Pythagoras": 254,
    "Schlacht bei Hastings": 104,
    "Wasser": 345,
    "Zugspitze": 459,
}


class NoRender:
    def render_html(self, title: str, snippets: list[str]) -> list[str]:
        raise AssertionError("no infobox expected")


def sentences(wikitext: str) -> list[str]:
    return [p.raw for p in split_pieces(wikitext, default_tokenizer()) if p.raw is not None]


def test_two_sentences_in_one_paragraph():
    assert sentences("Der Fuchs ist ein Raubtier. Er lebt in Wäldern.") == [
        "Der Fuchs ist ein Raubtier.",
        "Er lebt in Wäldern.",
    ]


def test_soft_line_break_joins_a_paragraph_and_blank_line_separates():
    assert sentences("Erster Satz\nzweite Zeile.\n\nNeuer Absatz.") == [
        "Erster Satz\nzweite Zeile.",
        "Neuer Absatz.",
    ]


def test_ref_attaches_to_the_preceding_sentence():
    text = "Der Turm ist hoch.<ref>Beleg.</ref> Er steht in Paris.<ref name=a />"
    assert sentences(text) == [
        "Der Turm ist hoch.<ref>Beleg.</ref>",
        "Er steht in Paris.<ref name=a />",
    ]


def test_bold_quotes_attach_to_the_following_sentence():
    assert sentences("Es regnet. '''Wasser''' ist nass.") == [
        "Es regnet.",
        "'''Wasser''' ist nass.",
    ]


def test_no_split_inside_a_link_label():
    text = "Die Werke von [[Johann Sebastian Bach|J. S. Bach]] sind berühmt. Er starb 1750."
    assert sentences(text) == [
        "Die Werke von [[Johann Sebastian Bach|J. S. Bach]] sind berühmt.",
        "Er starb 1750.",
    ]


def test_template_inside_a_sentence_does_not_end_it():
    text = "Der Berg ist {{Metadaten Höhe|2962|m}}. hoch. Das ist viel."
    assert len(sentences(text)) >= 2
    assert all(s.count("{{") == s.count("}}") for s in sentences(text))


def test_false_abbreviation_is_split_before_a_capital():
    assert sentences("Er reiste nach Italien. Danach fuhr er nach Hause.") == [
        "Er reiste nach Italien.",
        "Danach fuhr er nach Hause.",
    ]


def test_unit_and_score_do_not_swallow_the_full_stop():
    assert sentences("Es hatte 20 °C. Das Spiel endete 3:1. Dann ging man heim.") == [
        "Es hatte 20 °C.",
        "Das Spiel endete 3:1.",
        "Dann ging man heim.",
    ]


def test_list_items_are_own_segments():
    assert sentences("* Erstes Element.\n* Zweites Element.") == [
        "Erstes Element.",
        "Zweites Element.",
    ]


def test_excluded_sections_and_their_subsections_are_skipped():
    text = (
        "== Leben ==\nEin Satz im Text.\n"
        "== Weblinks ==\nEin Satz in den Weblinks.\n=== Unter ===\nNoch einer.\n"
        "== Rezeption ==\nWieder ein Satz."
    )
    assert sentences(text) == ["Ein Satz im Text.", "Wieder ein Satz."]


def test_section_is_recorded_per_segment():
    pieces = split_pieces(
        "Einleitung hier.\n== Leben ==\nText.\n=== Jugend ===\nMehr.", default_tokenizer()
    )
    assert [(p.section, p.raw) for p in pieces] == [
        ("", "Einleitung hier."),
        ("Leben", "Text."),
        ("Jugend", "Mehr."),
    ]


def test_hatnote_line_is_skipped_including_trailing_text():
    text = "{{Hauptartikel|Wasser}} mit allen Eigenschaften.\nEin echter Satz."
    assert sentences(text) == ["Ein echter Satz."]


def test_files_categories_tables_and_comments_are_not_prose():
    text = (
        "[[Datei:Bild.jpg|miniatur|Ein Bild]]\n"
        "{| class=wikitable\n|-\n| Zelle. Zwei.\n|}\n"
        "<!-- Kommentar. -->\n[[Kategorie:Tiere]]\n"
        "Nur dieser Satz zählt."
    )
    assert sentences(text) == ["Nur dieser Satz zählt."]


def test_scan_is_deterministic_and_covers_nested_templates():
    text = "a {{x|{{y|1}}|2}} b [[A|{{z}}]] c"
    masks = scan(text)
    assert [m.kind for m in masks] == ["template", "link"]
    assert text[masks[0].raw_start : masks[0].raw_end] == "{{x|{{y|1}}|2}}"


def test_segment_ids_and_archive_field_order():
    segs = segment_article("Erster Satz hier. Zweiter Satz hier.", "T", 271590031, NoRender())
    assert [s.id for s in segs] == ["271590031-0000", "271590031-0001"]
    assert list(segs[0].to_dict()) == ["id", "type", "section", "raw", "plain"]
    assert segs[0] == Segment("271590031-0000", "sentence", "", "Erster Satz hier.", None)


def test_segments_are_deterministic():
    text = "Ein Satz.<ref>x</ref> Noch ein Satz. '''Fett''' dann Ende."
    assert segment_article(text, "T", 1, NoRender()) == segment_article(text, "T", 1, NoRender())


@pytest.mark.parametrize("title", sorted(CORPUS_SENTENCES))
def test_corpus_round_trip_and_counts(title: str):
    article = json.loads((FIXTURES / f"{title.replace(' ', '_')}.json").read_text())
    pieces = split_pieces(article["wikitext"], default_tokenizer())  # raises on a round-trip break
    raws = [p.raw for p in pieces if p.raw is not None]
    assert len(raws) == CORPUS_SENTENCES[title]
    for raw in raws:
        assert raw.count("[[") == raw.count("]]")
        assert raw.count("{{") == raw.count("}}")
        self_closing = len(re.findall(r"<ref[^>]*/>", raw))
        assert raw.count("<ref") - self_closing == raw.count("</ref")
