"""Episode cover: fits the article title into the template's title box and renders a PNG.

Fitting rule (D7): wrap at word boundaries, no line limit, then shrink the font
size until the block fits the box. No ellipsis, no minimum size.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = REPO_ROOT / "assets" / "cover" / "cover-template.svg"
DEFAULT_FONT = REPO_ROOT / "assets" / "cover" / "fonts" / "DejaVuSans-Bold.ttf"

# (text, font size) -> advance width in SVG units, letter spacing included
Measure = Callable[[str, int], float]
# font size -> (ascent, descent) in SVG units, both positive
Metrics = Callable[[int], tuple[float, float]]


@dataclass(frozen=True)
class TitleBox:
    x: float
    y: float
    width: float
    height: float
    font_max: int
    line_height: float
    letter_spacing_em: float


@dataclass(frozen=True)
class Line:
    text: str
    x: float
    baseline: float


@dataclass(frozen=True)
class Layout:
    font_size: int
    lines: tuple[Line, ...]


class TitleDoesNotFit(ValueError):
    pass


# --- pure functions --------------------------------------------------------


def wrap(words: list[str], size: int, max_width: float, measure: Measure) -> list[str]:
    """Greedy wrap at word boundaries. A word wider than max_width gets its own line."""
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}" if current else word
        if current and measure(candidate, size) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def block_height(line_count: int, size: int, box: TitleBox, metrics: Metrics) -> float:
    ascent, descent = metrics(size)
    return ascent + (line_count - 1) * size * box.line_height + descent


def fit(title: str, box: TitleBox, measure: Measure, metrics: Metrics) -> Layout:
    """Largest font size <= box.font_max at which the wrapped title fits the box.

    Block is left-aligned and anchored to the bottom of the box.
    """
    words = title.split()
    if not words:
        raise TitleDoesNotFit("empty title")
    for size in range(box.font_max, 0, -1):
        lines = wrap(words, size, box.width, measure)
        fits_width = all(measure(line, size) <= box.width for line in lines)
        if fits_width and block_height(len(lines), size, box, metrics) <= box.height:
            _, descent = metrics(size)
            last_baseline = box.y + box.height - descent
            step = size * box.line_height
            return Layout(
                font_size=size,
                lines=tuple(
                    Line(text, box.x, last_baseline - (len(lines) - 1 - i) * step)
                    for i, text in enumerate(lines)
                ),
            )
    raise TitleDoesNotFit(title)


# --- template --------------------------------------------------------------


def _find(root: ET.Element, element_id: str) -> ET.Element:
    element = root.find(f".//*[@id='{element_id}']")
    if element is None:
        raise ValueError(f"template has no element with id '{element_id}'")
    return element


def read_title_box(root: ET.Element) -> TitleBox:
    rect = _find(root, "title-box")
    text = _find(root, "title")
    return TitleBox(
        x=float(rect.get("x")),
        y=float(rect.get("y")),
        width=float(rect.get("width")),
        height=float(rect.get("height")),
        font_max=int(rect.get("data-font-max")),
        line_height=float(rect.get("data-line-height")),
        letter_spacing_em=float(text.get("letter-spacing", "0")) / float(text.get("font-size")),
    )


def font_measure(font_path: Path, letter_spacing_em: float) -> tuple[Measure, Metrics]:
    from PIL import ImageFont

    cache: dict[int, ImageFont.FreeTypeFont] = {}

    def font(size: int) -> ImageFont.FreeTypeFont:
        if size not in cache:
            cache[size] = ImageFont.truetype(str(font_path), size)
        return cache[size]

    def measure(text: str, size: int) -> float:
        return font(size).getlength(text) + letter_spacing_em * size * (len(text) - 1)

    def metrics(size: int) -> tuple[float, float]:
        ascent, descent = font(size).getmetrics()
        return float(ascent), float(descent)

    return measure, metrics


def fill_title(root: ET.Element, layout: Layout, box: TitleBox) -> None:
    text = _find(root, "title")
    for child in list(text):
        text.remove(child)
    text.text = None
    text.set("font-size", str(layout.font_size))
    text.set("letter-spacing", f"{box.letter_spacing_em * layout.font_size:.2f}")
    for line in layout.lines:
        tspan = ET.SubElement(text, f"{{{SVG_NS}}}tspan", x=f"{line.x:g}", y=f"{line.baseline:.1f}")
        tspan.text = line.text


# --- rendering -------------------------------------------------------------

_fontconfig_dir: Path | None = None


def register_font_dir(font_path: Path) -> None:
    """Make cairo (via fontconfig) see only the directory of the bundled font.

    Must run before cairo draws its first text in this process. The system fonts are
    deliberately hidden so that rendering never falls back to a different font than the
    one used for measuring.
    """
    global _fontconfig_dir
    font_dir = font_path.resolve().parent
    if _fontconfig_dir == font_dir:
        return
    work = Path(tempfile.mkdtemp(prefix="gzw-fontconfig-"))
    conf = work / "fonts.conf"
    conf.write_text(
        '<?xml version="1.0"?>\n<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n<fontconfig>\n'
        f"  <dir>{font_dir}</dir>\n  <cachedir>{work / 'cache'}</cachedir>\n</fontconfig>\n",
        encoding="utf-8",
    )
    os.environ["FONTCONFIG_FILE"] = str(conf)
    _fontconfig_dir = font_dir


def render_svg(
    title: str, template: Path = DEFAULT_TEMPLATE, font_path: Path = DEFAULT_FONT
) -> tuple[bytes, Layout]:
    root = ET.parse(template).getroot()
    box = read_title_box(root)
    measure, metrics = font_measure(font_path, box.letter_spacing_em)
    layout = fit(title, box, measure, metrics)
    fill_title(root, layout, box)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), layout


def render_png(
    title: str,
    size_px: int,
    template: Path = DEFAULT_TEMPLATE,
    font_path: Path = DEFAULT_FONT,
) -> bytes:
    register_font_dir(font_path)
    import cairosvg

    svg, _ = render_svg(title, template, font_path)
    return cairosvg.svg2png(bytestring=svg, output_width=size_px, output_height=size_px)
