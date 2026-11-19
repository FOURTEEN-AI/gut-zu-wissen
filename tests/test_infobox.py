import re

from gut_zu_wissen.infobox import infobox_rows, split_top, template_name
from gut_zu_wissen.render import ParseRenderer, batches
from gut_zu_wissen.segmentation import segment_article

TEMPLATE = """{{Infobox Berg
| NAME = Zugspitze
| ORT = Wetterstein
| HÖHE = 2962
| BAUZEIT = 1887
| TEILVON = Teil von: Alpen
| LABEL = 2025/26
| WERT = 1. Platz
| FARBE = FF0000
}}"""


def tr(*cells: str) -> str:
    return "<tr>" + "".join(cells) + "</tr>"


def render_template(raw: str) -> str:
    """A tiny stand-in for the MediaWiki rendering of the template above."""
    values = {}
    for part in split_top(raw[2:-2])[1:]:
        key, _, value = part.partition("=")
        values[key.strip()] = value.strip()
    height = values.get("HÖHE", "")
    rows = [
        tr(f'<th colspan="2">{values.get("NAME", "")}</th>'),
        tr('<td colspan="2">Basisdaten</td>'),
        tr("<td>Ort:</td>", f"<td>{values.get('ORT', '')}</td>"),
        # The height only shows when it is a number: the template calculates with it.
        tr("<td>Höhe:</td>", f"<td>{height + ' m' if height.isdigit() else ''}</td>"),
        tr('<td colspan="2">Geschichte</td>'),
        tr("<td>Bauzeit:</td>", f"<td>{values.get('BAUZEIT', '')}</td>"),
        tr(f"<td>{values.get('LABEL', '')}</td>", f"<td>{values.get('WERT', '')}</td>"),
        tr(f'<td colspan="2">{values.get("TEILVON", "")}</td>'),
        tr(f'<td colspan="2"><span style="display:none">{values.get("FARBE", "")}</span></td>'),
    ]
    return f'<table class="infobox">{"".join(rows)}</table>'


class FakeRenderer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def render_html(self, title: str, snippets: list[str]) -> list[str]:
        self.calls.append(snippets)
        return [render_template(s) for s in snippets]


def test_template_helpers():
    assert split_top("A|b={{x|1}}|[[l|k]]") == ["A", "b={{x|1}}", "[[l|k]]"]
    assert template_name("{{Infobox_Berg\n| a=1}}") == "Infobox Berg"


def test_rows_in_rendered_order_with_header_and_plain_text():
    rows = infobox_rows(FakeRenderer(), "Zugspitze", TEMPLATE)
    assert [r.plain for r in rows] == [
        "Infobox Berg – Basisdaten – Ort: Wetterstein",
        "Infobox Berg – Basisdaten – Höhe: 2962 m",
        "Infobox Berg – Geschichte – Bauzeit: 1887",
        "Infobox Berg – Geschichte – 2025/26: 1. Platz",
        "Infobox Berg – Geschichte – Teil von: Alpen",
    ]


def test_raw_is_the_parameter_lines_of_the_row_in_template_order():
    rows = infobox_rows(FakeRenderer(), "Zugspitze", TEMPLATE)
    assert rows[0].raw == "Infobox Berg | ORT = Wetterstein"
    assert rows[3].raw == "Infobox Berg | LABEL = 2025/26 | WERT = 1. Platz"


def test_hidden_and_title_rows_are_no_segments():
    plains = " ".join(r.plain for r in infobox_rows(FakeRenderer(), "Zugspitze", TEMPLATE))
    assert "FF0000" not in plains
    assert "Zugspitze" not in plains


def test_calculated_parameter_is_found_with_the_digit_marker():
    renderer = FakeRenderer()
    infobox_rows(renderer, "Zugspitze", TEMPLATE)
    # baseline, one marked copy per parameter, then the digit-marker retry for the parameters
    # without a word-marker hit: HÖHE (calculated) and FARBE (never shown)
    assert [len(c) for c in renderer.calls] == [1, 8, 2]
    assert "7000000" in renderer.calls[2][0]


def test_header_is_omitted_when_the_row_has_none():
    raw = "{{Infobox Berg\n| ORT = Wetterstein\n}}"

    class Flat(FakeRenderer):
        def render_html(self, title: str, snippets: list[str]) -> list[str]:
            self.calls.append(snippets)
            out = []
            for s in snippets:
                value = re.search(r"ORT\s*=\s*(\S+)", s)
                row = tr("<td>Ort:</td>", f"<td>{value.group(1)}</td>")
                out.append(f'<table class="infobox">{row}</table>')
            return out

    assert [r.plain for r in infobox_rows(Flat(), "X", raw)] == ["Infobox Berg – Ort: Wetterstein"]


def test_template_without_parameters_makes_no_api_call():
    renderer = FakeRenderer()
    assert infobox_rows(renderer, "X", "{{Infobox Berg}}") == []
    assert renderer.calls == []


def test_infobox_rows_are_numbered_in_document_position():
    text = f"Vor dem Kasten steht ein Satz.\n{TEMPLATE}\nNach dem Kasten steht noch einer."
    segs = segment_article(text, "Zugspitze", 99, FakeRenderer())
    assert [s.type for s in segs] == ["sentence"] + ["infobox_row"] * 5 + ["sentence"]
    assert [s.id for s in segs] == [f"99-{i:04d}" for i in range(7)]
    assert segs[1].plain == "Infobox Berg – Basisdaten – Ort: Wetterstein"


def test_personendaten_is_not_an_infobox():
    text = "{{Personendaten\n| NAME = Bach\n}}\nEin Satz steht hier."
    segs = segment_article(text, "T", 1, FakeRenderer())
    assert [s.type for s in segs] == ["sentence"]


class FakePostApi:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query(self, **params: str) -> dict:
        raise AssertionError("parse must use POST")

    def post(self, **params: str) -> dict:
        self.calls.append(params)
        paragraphs = [p for p in params["text"].split("\n\n")]
        html = "".join(
            f"<p>{p}</p>" if p.startswith("SEGMENTGRENZE") else f"<div>{p}</div>"
            for p in paragraphs
        )
        return {"parse": {"text": html}}


def test_parse_renderer_splits_the_batch_at_boundary_paragraphs():
    api = FakePostApi()
    out = ParseRenderer(api).render_html("Titel", ["eins", "zwei", "drei"])
    assert out == ["<div>eins</div>", "<div>zwei</div>", "<div>drei</div>"]
    assert len(api.calls) == 1
    assert api.calls[0]["action"] == "parse" and api.calls[0]["title"] == "Titel"


def test_parse_renderer_makes_one_call_per_batch():
    api = FakePostApi()
    out = ParseRenderer(api, max_chars=5).render_html("T", ["aaaaaa", "bb", "cccccc", "d"])
    assert out == ["<div>aaaaaa</div>", "<div>bb</div>", "<div>cccccc</div>", "<div>d</div>"]
    assert len(api.calls) == 3


def test_batches_keep_order_and_close_at_the_limit():
    assert batches(["aaa", "bb", "c", "dddd"], max_chars=4) == [["aaa", "bb"], ["c", "dddd"]]
    assert batches([], max_chars=4) == []
