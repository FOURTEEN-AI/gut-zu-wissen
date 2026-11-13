import io
import xml.etree.ElementTree as ET

import pytest

from gut_zu_wissen.cover import (
    DEFAULT_FONT,
    DEFAULT_TEMPLATE,
    TitleBox,
    TitleDoesNotFit,
    block_height,
    fit,
    font_measure,
    read_title_box,
    render_png,
    render_svg,
    wrap,
)

TITLES = [
    "Schlacht bei Bouvines",
    "Liste der Baudenkmäler in Hamburg-Wilhelmsburg",
    "Donaudampfschifffahrtsgesellschaftskapitän",
    "Ei",
]

# --- pure tests with a fake monospace font ---------------------------------

BOX = TitleBox(x=0, y=0, width=1000, height=500, font_max=100, line_height=1.0, letter_spacing_em=0)


def mono(text: str, size: int) -> float:
    return len(text) * size * 0.5


def mono_metrics(size: int) -> tuple[float, float]:
    return size * 0.8, size * 0.2


def test_wrap_breaks_only_at_word_boundaries():
    words = ["aaaa", "bbbb", "cccc"]
    assert wrap(words, 100, 400, mono) == ["aaaa", "bbbb", "cccc"]
    assert wrap(words, 100, 450, mono) == ["aaaa bbbb", "cccc"]
    assert wrap(words, 100, 1000, mono) == ["aaaa bbbb cccc"]


def test_wrap_keeps_overlong_word_on_its_own_line():
    assert wrap(["x" * 50, "y"], 100, 1000, mono) == ["x" * 50, "y"]


def test_short_title_keeps_start_size():
    assert fit("Ei", BOX, mono, mono_metrics).font_size == BOX.font_max


def test_overlong_word_shrinks_until_it_fits():
    layout = fit("x" * 40, BOX, mono, mono_metrics)
    assert layout.font_size == 50  # 40 * 50 * 0.5 = 1000
    assert len(layout.lines) == 1


def test_size_is_the_largest_that_fits():
    title = "aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj"
    box = TitleBox(
        x=0, y=0, width=1000, height=200, font_max=100, line_height=1.0, letter_spacing_em=0
    )
    layout = fit(title, box, mono, mono_metrics)
    assert layout.font_size < box.font_max
    bigger = layout.font_size + 1
    lines = wrap(title.split(), bigger, box.width, mono)
    too_wide = any(mono(line, bigger) > box.width for line in lines)
    too_high = block_height(len(lines), bigger, box, mono_metrics) > box.height
    assert too_wide or too_high


def test_block_is_bottom_anchored_and_left_aligned():
    layout = fit("aaaa bbbb", TitleBox(0, 0, 400, 500, 100, 1.0, 0), mono, mono_metrics)
    assert [line.x for line in layout.lines] == [0, 0]
    assert layout.lines[-1].baseline == 500 - mono_metrics(layout.font_size)[1]


def test_empty_title_is_rejected():
    with pytest.raises(TitleDoesNotFit):
        fit("   ", BOX, mono, mono_metrics)


# --- tests with the real template and font ---------------------------------

needs_font = pytest.mark.skipif(not DEFAULT_FONT.exists(), reason="fonts-dejavu-core not installed")


@needs_font
@pytest.mark.parametrize("title", TITLES)
def test_real_layout_fits_box_and_keeps_all_words(title):
    root = ET.parse(DEFAULT_TEMPLATE).getroot()
    box = read_title_box(root)
    measure, metrics = font_measure(DEFAULT_FONT, box.letter_spacing_em)
    layout = fit(title, box, measure, metrics)

    assert " ".join(line.text for line in layout.lines) == " ".join(title.split())
    assert all(measure(line.text, layout.font_size) <= box.width for line in layout.lines)
    assert block_height(len(layout.lines), layout.font_size, box, metrics) <= box.height


@needs_font
def test_svg_contains_one_tspan_per_line():
    svg, layout = render_svg(TITLES[1])
    root = ET.fromstring(svg)
    title = root.find(".//*[@id='title']")
    assert [t.text for t in title] == [line.text for line in layout.lines]
    assert title.get("font-size") == str(layout.font_size)


def _png(svg: bytes, size_px: int):
    import cairosvg
    from PIL import Image

    png = cairosvg.svg2png(bytestring=svg, output_width=size_px, output_height=size_px)
    return Image.open(io.BytesIO(png)).convert("RGB")


@needs_font
@pytest.mark.parametrize("title", TITLES)
def test_rendered_title_stays_inside_box(title):
    pytest.importorskip("cairosvg")
    from PIL import Image, ImageChops

    size_px = 3000  # SVG units map 1:1 to pixels at this size
    root = ET.parse(DEFAULT_TEMPLATE).getroot()
    box = read_title_box(root)

    with_title = Image.open(io.BytesIO(render_png(title, size_px))).convert("RGB")
    assert with_title.size == (size_px, size_px)

    title_element = root.find(".//*[@id='title']")
    for child in list(title_element):
        title_element.remove(child)
    without_title = _png(ET.tostring(root), size_px)

    left, top, right, bottom = ImageChops.difference(with_title, without_title).getbbox()
    tolerance = 4  # antialiasing and glyph overhang
    assert left >= box.x - tolerance
    assert right <= box.x + box.width + tolerance
    assert top >= box.y - tolerance
    assert bottom <= box.y + box.height + tolerance
