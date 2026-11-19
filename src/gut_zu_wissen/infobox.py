"""Infobox rows by marker render (spike S3, D3).

An infobox segment is a *rendered labelled row*, not a parameter. Each parameter value is
replaced by a unique marker in a copy of the template; the rendered row that shows the marker is
the parameter's row. Label and value come from the baseline render of the unmodified template.
Plain text: „Infobox type – section header – label: value“, header omitted when there is none.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup
from bs4.element import Tag

from gut_zu_wissen.render import HtmlRenderer, plain_text

INFOBOX = re.compile(r"^\s*(Infobox[ _].*|Taxobox)\s*$", re.IGNORECASE)
MARKER = "Qzmarkerq{}"
NUMERIC_MARKER_BASE = 7_000_000
NUMERIC_MARKER_STEP = 7919
# Present in the HTML but not visible: the geo microformat and hidden spans.
HIDDEN = (
    ".geo, .geo-nondefault, .geo-multi-punct, [style*='display:none'], [style*='display: none']"
)
# Rows fed by a parameter whose name contains one of these are not section headers.
CAPTION_PARAM = re.compile(r"bild|beschreibung|karte|logo|formel|image|caption", re.IGNORECASE)
HEADER_MAX_WORDS = 8
INVISIBLE_CHARS = "­​‎"  # soft hyphen, zero-width space, left-to-right mark
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


@dataclass(frozen=True)
class InfoboxRow:
    """One segment: the raw text is the parameter lines of the row in template order."""

    raw: str
    plain: str


# ---------------------------------------------------------------- template text


def split_top(text: str) -> list[str]:
    """Split template inner text on top-level '|'."""
    parts, depth, cur, i = [], 0, [], 0
    while i < len(text):
        two = text[i : i + 2]
        if two in ("{{", "[["):
            depth += 1
            cur.append(two)
            i += 2
        elif two in ("}}", "]]"):
            depth -= 1
            cur.append(two)
            i += 2
        elif text[i] == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
            i += 1
        else:
            cur.append(text[i])
            i += 1
    parts.append("".join(cur))
    return parts


def template_name(raw_template: str) -> str:
    return " ".join(split_top(raw_template[2:-2])[0].replace("_", " ").split())


@dataclass(frozen=True)
class _Param:
    index: int  # position in the template's parts
    key: str
    value: str
    line: str  # "key = value" as written, stripped


def _params(raw_template: str) -> list[_Param]:
    """Named parameters with a non-empty value (comments removed), in template order."""
    params = []
    for k, part in enumerate(split_top(raw_template[2:-2])[1:], start=1):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key.strip() and COMMENT.sub("", value).strip():
            params.append(_Param(k, key.strip(), value.strip(), part.strip()))
    return params


def _letters(n: int) -> str:
    out = ""
    for _ in range(4):
        out = "abcdefghijklmnopqrstuvwxyz"[n % 26] + out
        n //= 26
    return out


def _marked_copy(raw_template: str, param: _Param, marker: str) -> str:
    parts = split_top(raw_template[2:-2])
    parts[param.index] = f"{param.key}= {marker}\n"
    return "{{" + "|".join(parts) + "}}"


def _numeric_forms(marker: str) -> list[str]:
    n = int(marker)
    return [marker, *(f"{n:,}".replace(",", sep) for sep in (".", " ", " "))]


# ---------------------------------------------------------------- rendered HTML


def _visible(el: Tag) -> str:
    """Visible plain text of an element, without mutating it."""
    soup = BeautifulSoup(str(el), "html.parser")
    for hidden in soup.select(HIDDEN):
        hidden.decompose()
    return plain_text(soup)


def _table_rows(html: str) -> list[Tag]:
    return list(BeautifulSoup(html, "html.parser").select("table.infobox tr, table.taxobox tr"))


def _cells(tr: Tag) -> list[str]:
    return [_visible(c) for c in tr.find_all(["td", "th"], recursive=False)]


# ---------------------------------------------------------------- mapping


@dataclass
class _Mapped:
    row: int
    param: _Param
    cell: int


def _row_cells(html: str) -> list[list[str]]:
    return [_cells(tr) for tr in _table_rows(html)]


def infobox_rows(renderer: HtmlRenderer, title: str, raw_template: str) -> list[InfoboxRow]:
    """The segments of one infobox: one per rendered labelled row, in row order."""
    name = template_name(raw_template)
    params = _params(raw_template)
    if not params:
        return []
    base_html, *_ = renderer.render_html(title, [raw_template])
    base_tr = _table_rows(base_html)
    real = [_cells(tr) for tr in base_tr]

    hits: dict[int, list[tuple[_Param, int]]] = {}  # row index -> [(param, cell index)]

    def place(param: _Param, rows: list[list[str]], found: list[tuple[int, int]]) -> None:
        if not found or len(rows) != len(real):
            return  # not shown, or the marker changes the row structure
        i, j = found[-1]  # innermost: nested tables come later in document order
        hits.setdefault(i, []).append((param, j))

    markers = [MARKER.format(_letters(n)) for n in range(len(params))]
    word_html = renderer.render_html(
        title, [_marked_copy(raw_template, p, m) for p, m in zip(params, markers, strict=True)]
    )
    retry: list[_Param] = []
    for param, marker, html in zip(params, markers, word_html, strict=True):
        rows = _row_cells(html)
        found = [(i, j) for i, r in enumerate(rows) for j, c in enumerate(r) if marker in c]
        if not found:
            retry.append(param)
        else:
            place(param, rows, found)

    # Parameters the template feeds into a calculation (height, prominence) need a digit marker.
    if retry:
        num_markers = [
            str(NUMERIC_MARKER_BASE + NUMERIC_MARKER_STEP * n) for n in range(len(retry))
        ]
        num_html = renderer.render_html(
            title,
            [_marked_copy(raw_template, p, m) for p, m in zip(retry, num_markers, strict=True)],
        )
        for param, marker, html in zip(retry, num_markers, num_html, strict=True):
            forms = _numeric_forms(marker)
            rows = _row_cells(html)
            found = [
                (i, j)
                for i, r in enumerate(rows)
                for j, c in enumerate(r)
                if any(f in c for f in forms)
            ]
            place(param, rows, found)

    headers = _header_rows(real, hits, base_tr)
    out: list[InfoboxRow] = []
    for i, feeders in sorted(hits.items()):
        built = _row_text(real, hits, headers, i, feeders)
        if built is None:
            continue
        label, value, header = built
        used = sorted({p.index: p for p, _ in feeders}.values(), key=lambda p: p.index)
        raw = " | ".join([name, *(p.line for p in used)])
        out.append(InfoboxRow(raw, _plain_text(name, header, label, value)))
    return out


def _row_text(
    real: list[list[str]],
    hits: dict[int, list[tuple[_Param, int]]],
    headers: set[int],
    i: int,
    feeders: list[tuple[_Param, int]],
) -> tuple[str, str, str] | None:
    """(label, value, section header) of rendered row i, or None when it is not a segment."""
    row = real[i]
    header = _nearest_header(real, headers, i)
    if len(row) == 1:
        label, sep, value = row[0].partition(":")
        if sep and label.strip() and value.strip():
            # Single-cell row with its own label: split at the first colon (D3).
            return label.strip(), value.strip(), header
        return None  # title, image caption, map caption
    label_from_param = any(j == 0 for _, j in feeders)
    under_header = i > 0 and len(real[i - 1]) == 1
    if label_from_param and not under_header:
        # The label cell is filled from a parameter (e.g. "Aktionäre", "2025/26").
        return row[0], " / ".join(c for c in row[1:] if c), header
    if label_from_param:
        # Comparison row without a label cell: label from the preceding single-cell row,
        # column names from the first label-less row of the table (the parties row).
        label = next((real[k][0] for k in range(i - 1, -1, -1) if len(real[k]) == 1), "")
        columns = next(
            (real[k] for k in sorted(hits) if len(real[k]) > 1 and any(j == 0 for _, j in hits[k])),
            row,
        )
        if columns is row:
            return label, " / ".join(row), ""
        value = "; ".join(f"{c}: {v}" for c, v in zip(columns, row, strict=False) if v)
        return label, value, ""
    return row[0], " / ".join(c for c in row[1:] if c), header


def _header_rows(real: list[list[str]], hits: dict, rows: list[Tag]) -> set[int]:
    """Indices of section-header rows.

    A header is a single-cell row that is a <th>, or a <td colspan=2> (nested tables use other
    spans), with non-empty text, no colon, at most HEADER_MAX_WORDS words. Not headers: the title
    (first non-empty single-cell row), rows fed by a parameter whose name marks an image, caption,
    map or logo, and the label row of a two-sided comparison.
    """
    title = next((k for k, r in enumerate(real) if len(r) == 1 and r[0]), -1)
    comparison = {
        i - 1
        for i, feeders in hits.items()
        if len(real[i]) > 1 and any(j == 0 for _, j in feeders) and i > 0 and len(real[i - 1]) == 1
    }
    found = set()
    for k, tr in enumerate(rows):
        cs = tr.find_all(["td", "th"], recursive=False)
        r = real[k]
        if len(cs) != 1 or not r[0] or k == title or k in comparison or ":" in r[0]:
            continue
        if not (cs[0].name == "th" or cs[0].get("colspan") == "2") or cs[0].find("table"):
            continue
        if any(CAPTION_PARAM.search(p.key) for p, _ in hits.get(k, [])):
            continue
        if len(r[0].split()) <= HEADER_MAX_WORDS:
            found.add(k)
    return found


def _nearest_header(real: list[list[str]], headers: set[int], i: int) -> str:
    return next((real[k][0] for k in range(i - 1, -1, -1) if k in headers), "")


def _plain_text(name: str, header: str, label: str, value: str) -> str:
    """D3 plain text: „Infobox type – section header – label: value“, header only if present."""
    return f"{' – '.join([name, *([header] if header else [])])} – {label.rstrip(':')}: {value}"
