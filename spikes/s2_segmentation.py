"""Spike S2: masking + SoMaJo + unmasking on raw wikitext.

Question: is sentence segmentation robust on diverse articles (nested templates, tables, lists)?
Output: masking rules, segment ID scheme. Not production code; step 3 rebuilds this with tests.

Usage: uv run --group spikes python spikes/s2_segmentation.py [--dump DIR]
"""

import argparse
import html
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from somajo import SoMaJo

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "wikipedia" / "spikes"

TEMPLATE_SURROGATE = "Vorlage"
# Extension tags whose content is not prose. Block-level when alone on a line.
EXT_TAGS = (
    "math|chem|ce|code|nowiki|syntaxhighlight|source|pre|gallery|score|timeline|references"
    "|poem|imagemap|mapframe|maplink|templatestyles|section|graph|hiero|inputbox|categorytree"
)
FILE_NS = ("datei:", "bild:", "file:", "image:")
CATEGORY_NS = ("kategorie:", "category:")
INFOBOX = re.compile(r"^\s*(Infobox[ _].*|Taxobox)\s*$", re.IGNORECASE)
EXTLINK = re.compile(r"\[(https?:)?//[^\s\]]+(\s+([^\]]*))?\]")
HEADING = re.compile(r"^(=+)(.+?)\1\s*$")
LIST_MARKER = re.compile(r"^[*#:;]+\s*")
# Level-2 sections that hold no article prose (bibliography, navigation).
EXCLUDED_SECTIONS = {
    "literatur",
    "weblinks",
    "einzelnachweise",
    "anmerkungen",
    "siehe auch",
    "quellen",
    "belege",
    "fußnoten",
}
ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]+);")
# Tokens SoMaJo misreads as abbreviation/ordinal before a full stop: "°C.", "3:1."
UNIT = re.compile(r"°[CF]\b")
SCORE = re.compile(r"\d+:\d+(?!\d)")
# Entries in SoMaJo's abbreviation list (matched case-insensitively) that are ordinary
# words in article prose; SoMaJo never ends a sentence after them. Found in the spike corpus.
# Hatnote templates: navigation, not content. A line starting with one is skipped as a whole,
# including text after the template on the same line (S2 decision 5).
HATNOTES = {
    "dieser artikel",
    "hauptartikel",
    "siehe auch",
    "begriffsklärungshinweis",
    "weiterleitungshinweis",
    "andere bedeutung",
    "unterscheide",
}
FALSE_ABBREVIATIONS = {"italien.", "japan.", "mag."}
SENTENCE_END = re.compile(r"[.!?:;…][\"'“”„‚‘’»«)\]]*$")


@dataclass
class Mask:
    raw_start: int
    raw_end: int
    kind: str  # link, file, category, extlink, template, table, ref, comment, ext
    surrogate: str
    attach: str = "left"  # zero-width masks only: side of a sentence boundary
    m_start: int = 0
    m_end: int = 0

    @property
    def zero_width(self) -> bool:
        return self.surrogate == ""


@dataclass
class Segment:
    id: str
    type: str  # sentence | infobox_row
    section: str
    raw: str
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- scanning


def _balanced(text: str, start: int, open_: str, close: str) -> int:
    """End index (exclusive) of the balanced construct starting at start, or -1."""
    depth, i = 0, start
    while i < len(text):
        if text.startswith(open_, i):
            depth += 1
            i += len(open_)
        elif text.startswith(close, i):
            depth -= 1
            i += len(close)
            if depth == 0:
                return i
        elif text.startswith("<!--", i):
            end = text.find("-->", i)
            i = len(text) if end < 0 else end + 3
        else:
            i += 1
    return -1


def _table_end(text: str, start: int) -> int:
    """Tables: '{|' and '|}' at line start, nested."""
    depth = 0
    for m in re.finditer(r"(?m)^[ \t]*(\{\||\|\})", text[start:]):
        depth += 1 if m.group(1) == "{|" else -1
        if depth == 0:
            return start + m.end()
    return -1


def _link_surrogate(inner: str) -> tuple[str, str]:
    target = inner.split("|", 1)[0].strip().lower()
    if target.startswith(FILE_NS):
        return "file", ""
    if target.startswith(CATEGORY_NS):
        return "category", ""
    label = inner.split("|", 1)[1] if "|" in inner else inner.split("#")[0]
    label = re.sub(r"\[\[|\]\]|\{\{.*?\}\}", "", label).replace("\n", " ").strip()
    return "link", label or "Link"


def scan(text: str) -> list[Mask]:
    masks: list[Mask] = []
    i, n = 0, len(text)
    ext_open = re.compile(rf"<({EXT_TAGS})(\s[^>]*)?>", re.IGNORECASE)
    ext_self = re.compile(rf"<({EXT_TAGS})(\s[^>]*)?/>", re.IGNORECASE)
    ref_self = re.compile(r"<ref(\s[^>]*)?/>", re.IGNORECASE)
    ref_open = re.compile(r"<ref(\s[^>]*)?>", re.IGNORECASE)
    while i < n:
        c = text[i]
        if text.startswith("<!--", i):
            end = text.find("-->", i)
            end = n if end < 0 else end + 3
            masks.append(Mask(i, end, "comment", ""))
            i = end
        elif text.startswith("''", i):
            m = re.compile(r"'{2,5}").match(text, i)
            masks.append(Mask(i, m.end(), "format", "", attach="right"))
            i = m.end()
        elif c == "&" and (m := ENTITY.match(text, i)):
            masks.append(Mask(i, m.end(), "entity", html.unescape(m.group()).replace("\xa0", " ")))
            i = m.end()
        elif c == "°" and (m := UNIT.match(text, i)):
            masks.append(Mask(i, m.end(), "unit", "Grad"))
            i = m.end()
        elif c.isdigit() and (i == 0 or not text[i - 1].isdigit()) and (m := SCORE.match(text, i)):
            masks.append(Mask(i, m.end(), "score", "Zahl"))
            i = m.end()
        elif c == "<" and (m := ref_self.match(text, i)):
            masks.append(Mask(i, m.end(), "ref", ""))
            i = m.end()
        elif c == "<" and (m := ref_open.match(text, i)):
            close = re.compile(r"</ref\s*>", re.IGNORECASE).search(text, m.end())
            end = n if close is None else close.end()
            masks.append(Mask(i, end, "ref", ""))
            i = end
        elif c == "<" and (m := ext_self.match(text, i)):
            masks.append(Mask(i, m.end(), "ext", ""))
            i = m.end()
        elif c == "<" and (m := ext_open.match(text, i)):
            tag = m.group(1)
            close = re.compile(rf"</{tag}\s*>", re.IGNORECASE).search(text, m.end())
            end = n if close is None else close.end()
            # Inline math/chem carries content; keep a word so the sentence stays whole.
            surrogate = TEMPLATE_SURROGATE if tag.lower() in ("math", "chem", "ce", "code") else ""
            masks.append(Mask(i, end, "ext", surrogate))
            i = end
        elif text.startswith("{|", i) and (i == 0 or text[i - 1] == "\n"):
            end = _table_end(text, i)
            end = n if end < 0 else end
            masks.append(Mask(i, end, "table", TEMPLATE_SURROGATE))
            i = end
        elif text.startswith("{{", i):
            end = _balanced(text, i, "{{", "}}")
            end = n if end < 0 else end
            masks.append(Mask(i, end, "template", TEMPLATE_SURROGATE))
            i = end
        elif text.startswith("[[", i):
            end = _balanced(text, i, "[[", "]]")
            if end < 0:
                i += 2
                continue
            kind, surrogate = _link_surrogate(text[i + 2 : end - 2])
            masks.append(Mask(i, end, kind, surrogate))
            i = end
        elif c == "[" and (m := EXTLINK.match(text, i)):
            masks.append(Mask(i, m.end(), "extlink", (m.group(3) or "Link").strip()))
            i = m.end()
        else:
            i += 1
    return masks


def build_masked(text: str, masks: list[Mask]) -> str:
    out, pos, m_pos = [], 0, 0
    for mask in masks:
        out.append(text[pos : mask.raw_start])
        m_pos += mask.raw_start - pos
        mask.m_start = m_pos
        out.append(mask.surrogate)
        m_pos += len(mask.surrogate)
        mask.m_end = m_pos
        pos = mask.raw_end
    out.append(text[pos:])
    return "".join(out)


class OffsetMap:
    """Maps a sentence boundary in masked text back to raw text.

    Zero-width masks sitting exactly on a boundary go to the side they attach to:
    refs and comments to the preceding sentence, bold/italic quotes to the following one.
    """

    def __init__(self, masks: list[Mask]):
        self.masks = masks

    def boundary(self, m: int, unit_end: bool = False) -> int:
        raw = m
        for mask in self.masks:
            if mask.m_start < m < mask.m_end:
                raise ValueError("boundary inside mask")
            before = mask.m_end < m or (mask.m_end == m and mask.m_start < m)
            if mask.zero_width and mask.m_start == m:
                before = unit_end or mask.attach == "left"
            if not before:
                break
            raw += (mask.raw_end - mask.raw_start) - len(mask.surrogate)
        return raw


# ---------------------------------------------------------------- segmentation


def _split_top(text: str) -> list[str]:
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


def _template_name(raw_template: str) -> str:
    return " ".join(_split_top(raw_template[2:-2])[0].replace("_", " ").split())


def infobox_rows(raw_template: str) -> tuple[str, list[str]]:
    parts = _split_top(raw_template[2:-2])
    name = parts[0].strip()
    rows = []
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        value = re.sub(r"<!--.*?-->", "", value, flags=re.DOTALL).strip()
        if key.strip() and value:
            rows.append(part.strip())
    return name, rows


def flags_for(raw: str) -> list[str]:
    visible = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>|<!--.*?-->", "", raw, flags=re.DOTALL)
    visible = re.sub(r"\{\{[^{}]*\}\}", "", visible).strip()
    words = visible.split()
    flags = []
    if not SENTENCE_END.search(visible):
        flags.append("no_end_punct")
    if visible[:1].islower():
        flags.append("lower_start")
    if len(words) < 4:
        flags.append("short")
    if len(words) > 60:
        flags.append("long")
    for o, c in (("[[", "]]"), ("{{", "}}"), ("<ref", "</ref")):
        if raw.count(o) != raw.count(c) and not (o == "<ref" and "/>" in raw):
            flags.append("unbalanced")
    return flags


def segment(article: dict, tokenizer: SoMaJo) -> tuple[list[Segment], list[str]]:
    text, revid = article["wikitext"], article["revid"]
    masks = scan(text)
    masked = build_masked(text, masks)
    omap = OffsetMap(masks)
    problems: list[str] = []
    segments: list[Segment] = []
    section = ""

    # Units: (kind, masked_start, masked_end)
    units: list[tuple[str, int, int]] = []
    para: list[int] | None = None
    pos = 0
    for line in masked.split("\n"):
        start, end = pos, pos + len(line)
        pos = end + 1
        line_masks = [m for m in masks if start <= m.m_start and m.m_end <= end]
        marker = LIST_MARKER.match(line)
        rest = line
        for m in sorted(line_masks, key=lambda m: -m.m_start):
            rest = rest[: m.m_start - start] + rest[m.m_end - start :]
        if marker:
            rest = rest[marker.end() :] if rest.startswith(marker.group()) else rest
        block_only = rest.strip() == "" and any(
            m.kind not in ("link", "extlink", "entity", "unit", "score") for m in line_masks
        )
        first = min(line_masks, key=lambda m: m.m_start) if line_masks else None
        if (
            first is not None
            and first.kind == "template"
            and line[: first.m_start - start].strip() == ""
            and _template_name(text[first.raw_start : first.raw_end]).lower() in HATNOTES
        ):
            block_only = True
        heading = HEADING.match(line)
        if para and (not line.strip() or heading or block_only or LIST_MARKER.match(line)):
            units.append(("para", para[0], para[1]))
            para = None
        if heading:
            units.append(("heading", start, end))
        elif block_only:
            units.append(("block", start, end))
        elif not line.strip() or line.startswith((" ", "----", "__")):
            continue
        elif marker:
            units.append(("list", start + marker.end(), end))
        elif para:
            para[1] = end
        else:
            para = [start, end]
    if para:
        units.append(("para", para[0], para[1]))

    index = 0
    excluded = False

    def add(kind: str, raw: str) -> None:
        nonlocal index
        seg = Segment(f"{revid}-{index:04d}", kind, section, raw)
        if kind == "sentence":
            seg.flags = flags_for(raw)
        segments.append(seg)
        index += 1

    for kind, m_start, m_end in units:
        if kind == "heading":
            h = HEADING.match(masked[m_start:m_end])
            section = h.group(2).strip()
            if len(h.group(1)) <= 2:
                excluded = section.lower() in EXCLUDED_SECTIONS
            continue
        if excluded:
            continue
        if kind == "block":
            for mask in masks:
                if m_start <= mask.m_start and mask.m_end <= m_end and mask.kind == "template":
                    raw_t = text[mask.raw_start : mask.raw_end]
                    name = _split_top(raw_t[2:-2])[0]
                    if INFOBOX.match(name):
                        for row in infobox_rows(raw_t)[1]:
                            add("infobox_row", f"{name.strip()} | {row}")
            continue
        unit_text = masked[m_start:m_end].replace("\n", " ")
        sentences = []
        for tokens in tokenizer.tokenize_text([unit_text]):
            cut = 0
            for k, tok in enumerate(tokens[:-1]):
                if tok.text.lower() in FALSE_ABBREVIATIONS and tokens[k + 1].text[:1].isupper():
                    part = tokens[cut : k + 1]
                    sentences.append(
                        [m_start + part[0].character_offset[0], m_start + tok.character_offset[1]]
                    )
                    cut = k + 1
                    problems.append("repaired_false_abbreviation")
            rest = tokens[cut:]
            s, e = rest[0].character_offset[0], rest[-1].character_offset[1]
            sentences.append([m_start + s, m_start + e])
        # Never split inside a mask (e.g. link label "J. S. Bach").
        merged: list[list[int]] = []
        for s, e in sentences:
            inside = any(mk.m_start < s < mk.m_end for mk in masks) if merged else False
            if inside:
                merged[-1][1] = e
                problems.append("merged_split_inside_mask")
            else:
                merged.append([s, e])
        raw_unit_start, raw_unit_end = omap.boundary(m_start), omap.boundary(m_end, unit_end=True)
        if text[raw_unit_end : raw_unit_end + 1] not in ("", "\n"):
            problems.append(f"unit_end_short at {raw_unit_end}")
        # Trailing zero-width masks (refs) at the unit end belong to the last sentence.
        raw_pieces = []
        for j, (s, _e) in enumerate(merged):
            rs = omap.boundary(s) if j else raw_unit_start
            re_ = omap.boundary(merged[j + 1][0]) if j + 1 < len(merged) else raw_unit_end
            raw_pieces.append(text[rs:re_])
        # Invariant: pieces reproduce the raw unit exactly.
        if "".join(raw_pieces) != text[raw_unit_start:raw_unit_end]:
            problems.append(f"roundtrip_mismatch at {raw_unit_start}")
        for piece in raw_pieces:
            stripped = piece.strip()
            if stripped:
                add("sentence", stripped)
    return segments, problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, help="write segments per article as JSON here")
    args = parser.parse_args()
    tokenizer = SoMaJo("de_CMC", character_offsets=True)
    totals: Counter = Counter()
    print(f"{'article':32} {'sent':>5} {'ibox':>5} " + " ".join(f"{f:>12}" for f in FLAG_NAMES))
    for path in sorted(FIXTURES.glob("*.json")):
        article = json.loads(path.read_text())
        segs, problems = segment(article, tokenizer)
        again, _ = segment(article, tokenizer)
        if [s.raw for s in segs] != [s.raw for s in again]:
            problems.append("nondeterministic")
        counts = Counter(f for s in segs for f in s.flags)
        sent = sum(s.type == "sentence" for s in segs)
        ibox = sum(s.type == "infobox_row" for s in segs)
        totals.update(counts)
        totals["sentence"] += sent
        totals["infobox_row"] += ibox
        print(
            f"{article['title'][:32]:32} {sent:>5} {ibox:>5} "
            + " ".join(f"{counts[f]:>12}" for f in FLAG_NAMES)
        )
        for p, c in Counter(problems).items():
            print(f"    problem: {p} ×{c}", file=sys.stderr)
        if args.dump:
            args.dump.mkdir(parents=True, exist_ok=True)
            out = args.dump / path.name
            data = [s.__dict__ for s in segs]
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(
        f"{'TOTAL':32} {totals['sentence']:>5} {totals['infobox_row']:>5} "
        + " ".join(f"{totals[f]:>12}" for f in FLAG_NAMES)
    )


FLAG_NAMES = ("no_end_punct", "lower_start", "short", "long", "unbalanced")

if __name__ == "__main__":
    main()
