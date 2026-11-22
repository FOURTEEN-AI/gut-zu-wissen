import json
import re
import shutil
import tomllib
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import pytest

from gut_zu_wissen.archive import ARTICLE_FILE, DIALOG_FILE, EPISODE_FILE, RANKING_FILE
from gut_zu_wissen.site import (
    STATUS_PAGE,
    StatusInput,
    build_site,
    german_date,
    load_published,
    read_live,
    review_labels,
)

REPO = Path(__file__).parents[1]
FULL_CONFIG = tomllib.loads((REPO / "config" / "pipeline.toml").read_text("utf-8"))
CONFIG = FULL_CONFIG["site"]
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
GIVEN = StatusInput(FULL_CONFIG["status"], FULL_CONFIG["publishing"], NOW)
SOURCE = REPO / "archive" / "episodes" / "2026-11-20_helgoland"
PUBLICATION = {
    "identifier": "gut-zu-wissen_2026-11-20_helgoland",
    "item_url": "https://archive.org/details/x",
    "published_at": "2026-11-22T09:00:00Z",
    "seconds": 905.4,
    "mp3_bitrate_kbps": 128,
    "audio": {
        "name": "episode.mp3",
        "url": "https://archive.org/download/x/episode.mp3",
        "bytes": 14000000,
        "mime": "audio/mpeg",
    },
    "cover": {
        "name": "cover.png",
        "url": "https://archive.org/download/x/cover.png",
        "bytes": 1,
        "mime": "image/png",
    },
}


def make_archive(tmp_path: Path, explicit: bool = False) -> Path:
    """A published copy of the Helgoland episode plus a failed one that must not appear."""
    episodes = tmp_path / "archive" / "episodes"
    published = episodes / "2026-11-20_helgoland"
    published.mkdir(parents=True)
    for name in (ARTICLE_FILE, RANKING_FILE, DIALOG_FILE):
        shutil.copy(SOURCE / name, published / name)
    record = json.loads((SOURCE / EPISODE_FILE).read_text("utf-8"))
    record |= {"status": "published", "published_at": "2026-11-22T09:00:00Z", "explicit": explicit}
    (published / EPISODE_FILE).write_text(json.dumps(record), encoding="utf-8")
    (published / "10_publication.json").write_text(json.dumps(PUBLICATION), encoding="utf-8")
    failed = episodes / "2026-11-21_bernstein"
    failed.mkdir()
    record |= {"status": "failed", "title": "Bernstein", "episode_id": "2026-11-21_bernstein"}
    (failed / EPISODE_FILE).write_text(json.dumps(record), encoding="utf-8")
    return tmp_path / "archive"


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags: list[tuple[str, dict]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def tags_of(path: Path) -> list[tuple[str, dict]]:
    parser = Tags()
    parser.feed(path.read_text("utf-8"))
    return parser.tags


@pytest.fixture
def built(tmp_path):
    archive = make_archive(tmp_path)
    out = tmp_path / "out"
    build_site(archive, REPO, out, CONFIG, GIVEN)
    return out


def test_only_published_episodes_appear(built):
    assert (built / "episodes" / "2026-11-20_helgoland.html").exists()
    assert not (built / "episodes" / "2026-11-21_bernstein.html").exists()
    index = (built / "index.html").read_text("utf-8")
    assert "Helgoland" in index and "Bernstein" not in index


def test_a_published_episode_without_publication_file_is_an_error(tmp_path):
    archive = make_archive(tmp_path)
    (archive / "episodes" / "2026-11-20_helgoland" / "10_publication.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_published(archive)


def test_every_page_links_impressum_and_datenschutz(built):
    pages = list(built.rglob("*.html"))
    assert len(pages) >= 5
    for page in pages:
        hrefs = [a.get("href", "") for tag, a in tags_of(page) if tag == "a"]
        assert any(h.endswith("impressum.html") for h in hrefs), page
        assert any(h.endswith("datenschutz.html") for h in hrefs), page


def test_pages_load_nothing_from_elsewhere(built):
    for page in built.rglob("*.html"):
        for tag, attrs in tags_of(page):
            assert tag not in ("iframe", "video", "source", "embed", "object"), page
            if tag == "script":  # inline only (the status page's refresh)
                assert "src" not in attrs, page
            if tag == "img":  # only the site's own files
                assert not attrs["src"].startswith(("http:", "https:", "//")), page
            if tag == "link":
                assert attrs.get("rel") == "alternate", page  # no stylesheet, font or preload
            if tag == "audio":
                assert attrs.get("preload") == "none", page
        # url(#id) is an in-page SVG reference; anything else would fetch a resource.
        assert not re.search(r"url\((?!#)|@import|@font-face", page.read_text("utf-8")), page


def test_episode_page_has_sources_history_and_feedback_link(built):
    text = (built / "episodes" / "2026-11-20_helgoland.html").read_text("utf-8")
    assert "title=Helgoland&amp;oldid=272085229" in text
    assert "title=Helgoland&amp;action=history" in text
    assert (
        'href="mailto:gut-zu-wissen@fourteen-it.de?subject=Gut%20zu%20wissen%3A%20Helgoland' in text
    )
    assert "2026-11-20_helgoland%29" in text
    assert "Folge vom 22. November 2026" in text
    assert "nicht für alle Hörer" not in text


def test_explicit_episode_carries_a_notice(tmp_path):
    archive = make_archive(tmp_path, explicit=True)
    build_site(archive, REPO, tmp_path / "out", CONFIG, GIVEN)
    assert "nicht für alle Hörer" in (
        tmp_path / "out/episodes/2026-11-20_helgoland.html"
    ).read_text("utf-8")


def test_legal_pages_keep_line_breaks_and_headings(built):
    text = (built / "impressum.html").read_text("utf-8")
    assert "<h1>Impressum</h1>" in text
    assert "Salomon-Petri-Ring 27<br" in text


def test_the_written_datenschutz_has_no_open_placeholder(built):
    assert "⚠" not in (built / "datenschutz.html").read_text("utf-8")


def test_feed_is_part_of_the_site_and_points_at_the_hosted_audio(built):
    channel = ET.parse(built / "feed.xml").getroot().find("channel")
    assert channel.find("item/enclosure").get("url") == PUBLICATION["audio"]["url"]
    assert channel.find("item/guid").text == "2026-11-20_helgoland"
    assert channel.find("item/link").text.endswith("/episodes/2026-11-20_helgoland.html")
    assert (built / "show-art.png").read_bytes()[:4] == b"\x89PNG"
    image = channel.find("item/{http://www.itunes.com/dtds/podcast-1.0.dtd}image").get("href")
    assert image.startswith(CONFIG["base_url"]) and image.endswith(
        "/covers/2026-11-20_helgoland.png"
    )


def test_cover_is_rendered_into_the_site_at_the_configured_size(built):
    from PIL import Image

    cover = built / "covers" / "2026-11-20_helgoland.png"
    assert Image.open(cover).size == (CONFIG["cover_size_px"],) * 2
    page = (built / "episodes" / "2026-11-20_helgoland.html").read_text("utf-8")
    assert 'src="../covers/2026-11-20_helgoland.png"' in page


def test_explainer_no_longer_says_episodes_are_unpublished():
    text = (REPO / "site" / "so-funktionierts.html").read_text("utf-8")
    assert "Probezeit" not in text and "noch nicht öffentlich" not in text


def test_german_date():
    assert german_date(datetime(2026, 3, 1)) == "1. März 2026"


# ---------------------------------------------------------------- status page


def test_status_page_name_matches_the_config():
    assert FULL_CONFIG["status"]["page"] == STATUS_PAGE


def test_status_page_is_built_and_linked_from_every_page(built):
    text = (built / STATUS_PAGE).read_text("utf-8")
    assert "<h1>Was gerade entsteht</h1>" in text
    assert "Im Moment entsteht keine neue Folge" in text
    assert 'href="episodes/2026-11-20_helgoland.html"' in text
    assert "„Bernstein“" in text  # failed within the last days
    for page in built.rglob("*.html"):
        if page.name == "so-funktionierts.html":  # copied as is
            continue
        hrefs = [a.get("href", "") for tag, a in tags_of(page) if tag == "a"]
        assert any(h.endswith(STATUS_PAGE) for h in hrefs), page


def test_status_page_refreshes_itself_from_its_own_address(built):
    text = (built / STATUS_PAGE).read_text("utf-8")
    assert 'fetch(location.pathname, {cache: "no-store"})' in text
    seconds = FULL_CONFIG["status"]["refresh_seconds"]
    assert f'<noscript><meta http-equiv="refresh" content="{seconds}"></noscript>' in text
    assert f"}}, {seconds * 1000});" in text


def test_status_page_shows_the_live_step(tmp_path):
    archive = make_archive(tmp_path)
    live = {
        "episode_id": "2026-11-22_koelner-dom",
        "title": "Kölner Dom",
        "step": "dialog",
        "started_at": "2026-11-22T11:00:00Z",
        "step_started_at": "2026-11-22T11:20:00Z",
    }
    path = tmp_path / "live.json"
    path.write_text(json.dumps(live), encoding="utf-8")
    given = StatusInput(FULL_CONFIG["status"], FULL_CONFIG["publishing"], NOW, read_live(path))
    build_site(archive, REPO, tmp_path / "out", CONFIG, given)
    text = (tmp_path / "out" / STATUS_PAGE).read_text("utf-8")
    assert "Folge über „Kölner Dom“" in text
    assert "Schritt 4 von 7" in text
    assert "Angefangen um 12:00 Uhr, dieser Schritt seit 12:20 Uhr." in text


def test_an_unreadable_live_file_is_ignored(tmp_path):
    path = tmp_path / "live.json"
    path.write_text("{", encoding="utf-8")
    assert read_live(path) is None
    path.write_text("{}", encoding="utf-8")
    assert read_live(path) is None
    assert read_live(tmp_path / "missing.json") is None
    assert read_live(None) is None


def test_review_labels_are_read_for_completed_episodes_only(tmp_path):
    archive = make_archive(tmp_path)
    folder = archive / "episodes" / "2026-11-21_bernstein"
    record = json.loads((folder / EPISODE_FILE).read_text("utf-8")) | {"status": "completed"}
    (folder / EPISODE_FILE).write_text(json.dumps(record), encoding="utf-8")
    (folder / "09_review.json").write_text(json.dumps({"issue": 7}), encoding="utf-8")
    asked = []

    def read(number):
        asked.append(number)
        return ["hold"]

    assert review_labels(archive, read) == {7: ["hold"]}
    assert asked == [7]
    assert review_labels(archive, None) == {}

    def broken(number):
        raise RuntimeError("no access")

    assert review_labels(archive, broken) == {}
