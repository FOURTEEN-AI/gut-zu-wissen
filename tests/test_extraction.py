import pytest

from gut_zu_wissen.extraction import (
    Chunk,
    context_allowed,
    extract_facts,
    format_chunk,
    is_full_sentence,
    is_unique,
    make_chunks,
    plain_rejection,
)
from gut_zu_wissen.segmentation import Segment

CONFIG = {"quotas": {"max_context_segments": 3, "max_chunk_segments": 3, "min_fact_words": 4}}


def seg(i: int, raw: str, section: str = "A", type_: str = "sentence", plain=None) -> Segment:
    return Segment(f"1-{i:04d}", type_, section, raw, plain)


def chunk_of(n: int) -> Chunk:
    return Chunk(0, "A", [seg(i, f"Satz {i}.") for i in range(n)])


def test_chunks_follow_sections():
    segs = [seg(0, "a.", "X"), seg(1, "b.", "X"), seg(2, "c.", "Y")]
    chunks = make_chunks(segs, 40)
    assert [(c.section, [s.id for s in c.segments]) for c in chunks] == [
        ("X", ["1-0000", "1-0001"]),
        ("Y", ["1-0002"]),
    ]
    assert [c.index for c in chunks] == [0, 1]


def test_large_section_is_split_into_equal_parts():
    segs = [seg(i, f"s{i}.") for i in range(7)]
    assert [len(c.segments) for c in make_chunks(segs, 3)] == [3, 2, 2]
    assert [len(c.segments) for c in make_chunks([seg(i, "x.") for i in range(6)], 3)] == [3, 3]


def test_no_segments_no_chunks():
    assert make_chunks([], 40) == []


def test_uniqueness():
    segs = [seg(0, "Gleich."), seg(1, "Gleich."), seg(2, "Anders.")]
    assert not is_unique(segs[0], segs)
    assert is_unique(segs[2], segs)


@pytest.mark.parametrize(
    ("fact", "context", "ok"),
    [
        ("1-0003", [], True),
        ("1-0003", ["1-0002"], True),
        ("1-0003", ["1-0001", "1-0002"], True),
        ("1-0003", ["1-0000", "1-0001", "1-0002"], True),
        ("1-0003", ["1-0001"], False),  # gap
        ("1-0003", ["1-0002", "1-0001"], False),  # wrong order
        ("1-0003", ["1-0003"], False),  # the fact itself
        ("1-0003", ["1-0002", "1-0004"], False),  # successor
        ("1-0000", ["1-0000"], False),
        ("1-0009", [], False),  # fact not in the chunk
    ],
)
def test_context_gate(fact, context, ok):
    assert context_allowed(fact, context, chunk_of(5), 3) is ok


def test_context_gate_at_most_max_segments_and_in_the_same_chunk():
    chunk = chunk_of(6)
    assert not context_allowed("1-0005", ["1-0001", "1-0002", "1-0003", "1-0004"], chunk, 3)
    tail = Chunk(1, "A", chunk.segments[3:])
    assert not context_allowed("1-0004", ["1-0002", "1-0003"], tail, 3)  # 1-0002 is in no chunk
    assert context_allowed("1-0004", ["1-0003"], tail, 3)


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("Der Turm ist hoch.", True),
        ("„Hallo“, sagte er.", True),
        ("Er sagte: „Es regnet.“", True),
        ("Im Jahr 1887 begann der Bau!", True),
        ("1887 begann der Bau.", True),
        ("der Turm ist hoch.", False),
        ("Der Turm ist hoch", False),
        ("Der Turm ist hoch:", False),
        ("Filme von", False),
    ],
)
def test_full_sentence_heuristic(text, ok):
    assert is_full_sentence(text) is ok


def test_plain_gate_reasons():
    s = seg(0, "x")
    row = seg(1, "x", type_="infobox_row")
    assert plain_rejection(s, "", 4) == "plain_empty"
    assert plain_rejection(s, "Das ist |} Unsinn und lang.", 4) == "layout_artefact"
    assert plain_rejection(s, "Zu kurz hier.", 4) == "too_short"
    assert plain_rejection(s, "Das hier ist lang genug", 4) == "not_a_sentence"
    assert plain_rejection(s, "Das hier ist lang genug.", 4) is None
    # Infobox rows: no word count, no sentence rule
    assert plain_rejection(row, "Infobox X – Ort: Y", 8) is None
    assert plain_rejection(row, "", 8) == "plain_empty"


def test_format_chunk_shows_ids_without_refs_and_infobox_plain():
    chunk = Chunk(
        0,
        "",
        [
            seg(0, "Ein Satz.<ref>Beleg</ref>", ""),
            seg(1, "Infobox X | A = b", "", "infobox_row", "Infobox X – A: b"),
        ],
    )
    assert format_chunk(chunk) == (
        "Abschnitt: (Einleitung)\n\n[1-0000] Ein Satz.\n[1-0001] Infobox X – A: b"
    )


class FakeLlm:
    def __init__(self, answers: list[dict]) -> None:
        self.answers, self.calls = list(answers), []

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        self.calls.append((system, user, schema))
        return self.answers.pop(0)


class EchoRenderer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def render_html(self, title: str, snippets: list[str]) -> list[str]:
        self.calls.append(snippets)
        return [f"<p>{s}</p>" for s in snippets]


def cand(i: int, *ctx: int) -> dict:
    return {"segment_id": f"1-{i:04d}", "context_ids": [f"1-{c:04d}" for c in ctx]}


def test_extract_applies_gates_and_renders_once():
    segs = [
        seg(0, "Er ist ein Fuchs aus dem Wald."),
        seg(1, "Der Fuchs wiegt zehn Kilo und mehr.<ref>x</ref>"),
        seg(2, "Doppelt vorhanden und lang genug."),
        seg(3, "Doppelt vorhanden und lang genug."),
        seg(4, "Kurz."),
        seg(5, "Infobox X | A = b", type_="infobox_row", plain="Infobox X – A: b"),
        seg(6, "Hier kommt ein Satz in einem Abschnitt.", "B"),
    ]
    llm = FakeLlm(
        [
            {
                "candidates": [
                    cand(1, 0),  # accepted with context
                    cand(1),  # duplicate proposal, ignored
                    cand(2),  # not unique
                    cand(4),  # too short
                    cand(5, 4),  # accepted infobox row; context is a short sentence
                    {"segment_id": "9-9999", "context_ids": []},  # unknown
                    cand(6),  # other chunk
                ]
            },
            {"candidates": [cand(6, 5)]},  # context from another chunk
        ]
    )
    renderer = EchoRenderer()
    config = {"quotas": {**CONFIG["quotas"], "max_chunk_segments": 6}}
    result = extract_facts(llm, renderer, "T", 1, segs, config)
    assert result.chunk_count == 2
    assert [(f.segment_id, f.context_ids) for f in result.facts] == [
        ("1-0001", ["1-0000"]),
        ("1-0005", ["1-0004"]),
    ]
    assert result.facts[0].plain == "Der Fuchs wiegt zehn Kilo und mehr."  # ref removed
    assert result.facts[0].context_plain == ["Er ist ein Fuchs aus dem Wald."]
    assert result.facts[1].plain == "Infobox X – A: b"
    assert [(r.segment_id, r.reason) for r in result.rejected] == [
        ("1-0002", "not_unique"),
        ("9-9999", "unknown_segment"),
        ("1-0006", "unknown_segment"),
        ("1-0006", "context_invalid"),
        ("1-0004", "too_short"),  # plain-text gates run after the render
    ]
    assert len(renderer.calls) == 1  # one batched render for all offered facts and context


def test_empty_answer_and_no_render_call():
    renderer = EchoRenderer()
    result = extract_facts(
        FakeLlm([{"candidates": []}]), renderer, "T", 1, [seg(0, "Ein Satz.")], CONFIG
    )
    assert result.facts == [] and renderer.calls == []


def test_prompt_and_schema_are_passed():
    llm = FakeLlm([{"candidates": []}])
    extract_facts(llm, EchoRenderer(), "T", 1, [seg(0, "Ein Satz.")], CONFIG)
    system, user, schema = llm.calls[0]
    assert "höchstens 3 Segmente" in system
    assert "[1-0000] Ein Satz." in user
    assert schema["required"] == ["candidates"]


def test_result_dict_field_order():
    result = extract_facts(FakeLlm([{"candidates": []}]), EchoRenderer(), "T", 7, [], CONFIG)
    assert list(result.to_dict()) == ["revid", "chunk_count", "facts", "rejected"]
