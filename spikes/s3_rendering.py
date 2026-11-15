"""Spike S3: rendering segments to plain text.

Questions: does `action=parse` of a single segment (in the article's context) give the same
plain text as the segment has in the fully rendered article? How do infobox parameters map to
the row labels a reader sees (D3)?

Method: render every segment of the spike articles on its own (many segments per API call,
separated by marker paragraphs, so the API load stays small), render each full article once,
and compare normalized plain text. Infobox rows: render each parameter value, find the rendered
infobox row whose value cell contains it; that row's label cell is the label.

Usage: uv run --group spikes python spikes/s3_rendering.py [--dump DIR]
Rendered HTML is cached in tests/fixtures/wikipedia/spikes/s3_cache/ (CC BY-SA).
"""

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from s2_segmentation import FIXTURES, scan, segment
from somajo import SoMaJo

API = "https://de.wikipedia.org/w/api.php"
USER_AGENT = "GutZuWissen/0.1 (spike; gut-zu-wissen@fourteen-it.de)"
CACHE = FIXTURES / "s3_cache"
MARK = "SEGMENTGRENZE{:05d}"
BATCH_CHARS = 60_000
# Rendered elements that are not reading text.
DROP = (
    "sup.reference, .mw-editsection, style, script, .mw-ref, .reference, "
    ".noprint, .metadata, .error, .mw-empty-elt"
)


def _cached(key: str, fetch) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".html")
    if path.exists():
        return path.read_text()
    html = fetch()
    path.write_text(html)
    time.sleep(0.5)
    return html


def parse_text(client: httpx.Client, title: str, wikitext: str) -> str:
    def fetch() -> str:
        r = client.post(
            API,
            data={
                "action": "parse",
                "format": "json",
                "formatversion": "2",
                "title": title,
                "text": wikitext,
                "contentmodel": "wikitext",
                "prop": "text",
                "disableeditsection": "1",
                "disablelimitreport": "1",
                "maxlag": "5",
            },
        )
        r.raise_for_status()
        return r.json()["parse"]["text"]

    return _cached(f"text|{title}|{wikitext}", fetch)


def parse_revision(client: httpx.Client, revid: int) -> str:
    def fetch() -> str:
        r = client.get(
            API,
            params={
                "action": "parse",
                "format": "json",
                "formatversion": "2",
                "oldid": str(revid),
                "prop": "text",
                "disableeditsection": "1",
                "disablelimitreport": "1",
                "maxlag": "5",
            },
        )
        r.raise_for_status()
        return r.json()["parse"]["text"]

    return _cached(f"rev|{revid}", fetch)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("­", "").replace("​", "").replace("‎", "")
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def plain(html_fragment) -> str:
    soup = (
        html_fragment
        if hasattr(html_fragment, "select")
        else BeautifulSoup(html_fragment, "html.parser")
    )
    for el in soup.select(DROP):
        el.decompose()
    for br in soup.select("br"):
        br.replace_with(" ")
    return norm(soup.get_text(""))


def strip_refs(raw: str) -> str:
    """Remove <ref> and comments using the S2 scanner (same rules as segmentation)."""
    out, pos = [], 0
    for m in scan(raw):
        if m.kind in ("ref", "comment"):
            out.append(raw[pos : m.raw_start])
            pos = m.raw_end
    out.append(raw[pos:])
    return "".join(out).strip()


def render_batched(client: httpx.Client, title: str, raws: list[str]) -> list[str]:
    """Render each wikitext snippet as its own paragraph; return plain text per snippet."""
    results: list[str] = []
    start = 0
    while start < len(raws):
        batch, size = [], 0
        while start + len(batch) < len(raws) and (size < BATCH_CHARS or not batch):
            batch.append(raws[start + len(batch)])
            size += len(batch[-1])
        text = (
            "\n\n".join(f"{MARK.format(i)}\n\n{raw}" for i, raw in enumerate(batch))
            + f"\n\n{MARK.format(len(batch))}"
        )
        html = parse_text(client, title, text)
        parts = re.split(r"<p>\s*SEGMENTGRENZE(\d{5})\s*</p>", html)
        # parts = [before, idx0, html0, idx1, html1, ..., idxN, after]
        by_index = {int(parts[k]): parts[k + 1] for k in range(1, len(parts) - 1, 2)}
        for i in range(len(batch)):
            results.append(plain(by_index.get(i, "")))
        start += len(batch)
    return results


def article_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.select("table"):
        table.decompose()
    return plain(soup)


def infobox_tables(html: str) -> list:
    soup = BeautifulSoup(html, "html.parser")
    return soup.select("table.infobox, table.taxobox")


def infobox_rows_rendered(table) -> list[tuple[str, str]]:
    rows = []
    for tr in table.select("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        if len(cells) == 2:
            rows.append((plain(cells[0]), plain(cells[1])))
    return rows


def classify(seg_plain: str, full: str) -> str:
    if not seg_plain:
        return "empty"
    if seg_plain in full:
        return "identical"
    # Same text up to spacing/punctuation inside the sentence?
    squash = re.sub(r"[\s\W_]+", "", seg_plain)
    if squash and squash in re.sub(r"[\s\W_]+", "", full):
        return "spacing_only"
    return "different"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path)
    args = parser.parse_args()
    tokenizer = SoMaJo("de_CMC", character_offsets=True)
    totals: Counter = Counter()
    report: dict = {}
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120) as client:
        for path in sorted(FIXTURES.glob("*.json")):
            if path.name.startswith("s1_"):
                continue
            article = json.loads(path.read_text())
            segs, _ = segment(article, tokenizer)
            title, revid = article["title"], article["revid"]
            full_html = parse_revision(client, revid)
            full = article_text(full_html)
            sentences = [s for s in segs if s.type == "sentence"]
            rendered = render_batched(client, title, [strip_refs(s.raw) for s in sentences])
            counts: Counter = Counter()
            diffs = []
            for s, text in zip(sentences, rendered, strict=True):
                kind = classify(text, full)
                counts[kind] += 1
                if kind != "identical":
                    diffs.append({"id": s.id, "kind": kind, "raw": s.raw, "plain": text})
            # Infobox rows
            ibox = [s for s in segs if s.type == "infobox_row"]
            mapping = Counter()
            ibox_out = []
            if ibox:
                tables = infobox_tables(full_html)
                rows = [r for t in tables for r in infobox_rows_rendered(t)]
                values = [strip_refs(s.raw.split("=", 1)[1]) for s in ibox]
                rendered_values = render_batched(client, title, values)
                for s, value in zip(ibox, rendered_values, strict=True):
                    param = s.raw.split("|", 1)[1].split("=", 1)[0].strip()
                    if not value:
                        hit, labels = "value_renders_empty", []
                    else:
                        labels = [lab for lab, val in rows if value and value in val]
                        exact = [lab for lab, val in rows if val == value]
                        if len(exact) == 1:
                            hit, labels = "exact", exact
                        elif len(labels) == 1:
                            hit = "contained"
                        elif labels:
                            hit = "ambiguous"
                        else:
                            hit = "not_shown"
                    mapping[hit] += 1
                    ibox_out.append(
                        {
                            "id": s.id,
                            "param": param,
                            "value": value,
                            "result": hit,
                            "labels": labels,
                        }
                    )
            totals.update(counts)
            totals.update({f"ibox_{k}": v for k, v in mapping.items()})
            print(
                f"{title[:30]:30} "
                + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                + (
                    "  | infobox " + " ".join(f"{k}={v}" for k, v in sorted(mapping.items()))
                    if mapping
                    else ""
                )
            )
            report[title] = {"diffs": diffs, "infobox": ibox_out}
    print("TOTAL", dict(totals))
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        (args.dump / "s3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- marker method for infoboxes

MARKER = "Qzmarkerq{}"


def _letters(n: int) -> str:
    out = ""
    for _ in range(4):
        out = "abcdefghijklmnopqrstuvwxyz"[n % 26] + out
        n //= 26
    return out


def infobox_templates(wikitext: str) -> list[tuple[str, str]]:
    """(name, raw) of every Infobox */Taxobox template in the article (D3)."""
    from s2_segmentation import INFOBOX, _split_top, _template_name

    out = []
    for m in scan(wikitext):
        if m.kind == "template":
            raw = wikitext[m.raw_start : m.raw_end]
            name = _template_name(raw)
            if INFOBOX.match(_split_top(raw[2:-2])[0]):
                out.append((name, raw))
    return out


def marked_copies(raw: str, numeric: bool = False) -> list[tuple[str, str, str, str]]:
    """(param, original value, template copy with that param's value replaced by a marker).

    numeric: use a digit marker, for parameters the template feeds into a calculation.
    """
    from s2_segmentation import _split_top

    parts = _split_top(raw[2:-2])
    copies = []
    for k, part in enumerate(parts[1:], start=1):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if not re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL).strip():
            continue
        n = len(copies)
        marker = str(7_000_000 + 7919 * n) if numeric else MARKER.format(_letters(n))
        new = parts.copy()
        new[k] = f"{key}= {marker}\n"
        copies.append((key.strip(), value.strip(), "{{" + "|".join(new) + "}}", marker))
    return copies


def row_cells(tr) -> list[str]:
    return [plain(c) for c in tr.find_all(["td", "th"], recursive=False)]


HIDDEN = (
    ".geo, .geo-nondefault, .geo-multi-punct, [style*='display:none'], [style*='display: none']"
)


def visible(el) -> str:
    """Visible plain text of an element, without mutating it."""
    soup = BeautifulSoup(str(el), "html.parser")
    for h in soup.select(HIDDEN):
        h.decompose()
    return plain(soup)


def table_rows(soup) -> list:
    """Rows of all infobox tables, innermost tables included once, in document order."""
    seen, rows = set(), []
    for tr in soup.select("table.infobox tr, table.taxobox tr"):
        if id(tr) not in seen:
            seen.add(id(tr))
            rows.append(tr)
    return rows


def cells(tr) -> list[str]:
    return [visible(c) for c in tr.find_all(["td", "th"], recursive=False)]


def _numeric_forms(marker: str) -> list[str]:
    n = int(marker)
    return [
        marker,
        f"{n:,}".replace(",", "."),
        f"{n:,}".replace(",", " "),
        f"{n:,}".replace(",", "\u202f"),
    ]


def map_infobox(client: httpx.Client, title: str, raw: str, full_html: str) -> list[dict]:
    """One entry per rendered infobox row: label, rendered value, feeding parameters.

    Baseline is the unmodified template rendered the same way as the marked copies
    (action=parse of the template text in the article's context); S3 shows this matches
    the in-article rendering for sentences. full_html is kept for comparison only.
    """
    base_html = render_batched_html(client, title, [raw])[0]
    real = [cells(tr) for tr in table_rows(BeautifulSoup(base_html, "html.parser"))]
    hits: dict[int, list[tuple[str, int]]] = {}  # row index -> [(param, cell index)]
    unmapped = []
    word = marked_copies(raw)
    num = {c[0]: c for c in marked_copies(raw, numeric=True)}
    word_html = render_batched_html(client, title, [c[2] for c in word])
    retry = []
    for (param, _value, _, marker), html in zip(word, word_html, strict=True):
        rows = [cells(tr) for tr in table_rows(BeautifulSoup(html, "html.parser"))]
        found = [(i, j) for i, r in enumerate(rows) for j, c in enumerate(r) if marker in c]
        if not found:
            retry.append(param)
        elif len(rows) != len(real):
            unmapped.append((param, "layout_changes"))
        else:
            i, j = found[-1]  # innermost (nested tables come later in document order)
            hits.setdefault(i, []).append((param, j))
    retry_html = render_batched_html(client, title, [num[p][2] for p in retry]) if retry else []
    for param, html in zip(retry, retry_html, strict=True):
        forms = _numeric_forms(num[param][3])
        rows = [cells(tr) for tr in table_rows(BeautifulSoup(html, "html.parser"))]
        found = [
            (i, j)
            for i, r in enumerate(rows)
            for j, c in enumerate(r)
            if any(f in c for f in forms)
        ]
        if not found:
            unmapped.append((param, "not_shown"))
        elif len(rows) != len(real):
            unmapped.append((param, "layout_changes"))
        else:
            i, j = found[-1]
            hits.setdefault(i, []).append((param + " (numeric)", j))
    out = []
    for i, feeders in sorted(hits.items()):
        row = real[i]
        params = [p for p, _ in feeders]
        if len(row) == 1:
            out.append({"result": "no_label_single_cell", "params": params, "text": row[0]})
            continue
        label_from_param = any(j == 0 for _, j in feeders)
        under_header = i > 0 and len(real[i - 1]) == 1
        if label_from_param and not under_header:
            # The label cell is filled from a parameter (e.g. "Aktionäre", "2025/26").
            out.append(
                {
                    "result": "mapped_param_label",
                    "params": params,
                    "label": row[0],
                    "value": " / ".join(c for c in row[1:] if c),
                }
            )
            continue
        if label_from_param:
            # Comparison row without a label cell: label from the preceding header row,
            # column names from the first label-less row of the table (the parties row).
            header = next((real[k][0] for k in range(i - 1, -1, -1) if len(real[k]) == 1), "")
            columns = next(
                (
                    real[k]
                    for k in sorted(hits)
                    if len(real[k]) > 1 and any(j == 0 for _, j in hits[k])
                ),
                row,
            )
            if columns is row:
                out.append(
                    {
                        "result": "column_names",
                        "params": params,
                        "label": header,
                        "value": " / ".join(row),
                    }
                )
            else:
                value = "; ".join(f"{c}: {v}" for c, v in zip(columns, row, strict=False) if v)
                out.append(
                    {
                        "result": "mapped_comparison",
                        "params": params,
                        "label": header,
                        "value": value,
                    }
                )
            continue
        out.append(
            {
                "result": "mapped",
                "params": params,
                "label": row[0],
                "value": " / ".join(c for c in row[1:] if c),
            }
        )
    out += [{"result": r, "params": [p]} for p, r in unmapped]
    return out


def render_batched_html(client: httpx.Client, title: str, raws: list[str]) -> list[str]:
    results: list[str] = []
    start = 0
    while start < len(raws):
        batch, size = [], 0
        while start + len(batch) < len(raws) and (size < BATCH_CHARS or not batch):
            batch.append(raws[start + len(batch)])
            size += len(batch[-1])
        text = (
            "\n\n".join(f"{MARK.format(i)}\n\n{raw}" for i, raw in enumerate(batch))
            + f"\n\n{MARK.format(len(batch))}"
        )
        html = parse_text(client, title, text)
        parts = re.split(r"<p>\s*SEGMENTGRENZE(\d{5})\s*</p>", html)
        by_index = {int(parts[k]): parts[k + 1] for k in range(1, len(parts) - 1, 2)}
        results += [by_index.get(i, "") for i in range(len(batch))]
        start += len(batch)
    return results


def main_infobox() -> None:
    totals: Counter = Counter()
    report = {}
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120) as client:
        for path in sorted(FIXTURES.glob("*.json")):
            if path.name.startswith("s1_"):
                continue
            article = json.loads(path.read_text())
            boxes = infobox_templates(article["wikitext"])
            if not boxes:
                continue
            full_html = parse_revision(client, article["revid"])
            rows = []
            for name, raw in boxes:
                rows += [
                    {"template": name, **r}
                    for r in map_infobox(client, article["title"], raw, full_html)
                ]
            c = Counter(r["result"] for r in rows)
            c["params_total"] = sum(len(r["params"]) for r in rows)
            totals.update(c)
            report[article["title"]] = rows
            print(
                f"{article['title'][:30]:30} " + " ".join(f"{k}={v}" for k, v in sorted(c.items()))
            )
    print("TOTAL", dict(totals))
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / "s3_infobox.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "infobox":
        main_infobox()
    else:
        main()
