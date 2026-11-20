import xml.etree.ElementTree as ET
from datetime import UTC, datetime

from gut_zu_wissen.feed import ITUNES, FeedEpisode, build_feed, duration_text

CHANNEL = {
    "title": "Gut zu wissen",
    "base_url": "https://example.org/gzw",
    "description": "Beschreibung",
    "language": "de",
    "copyright": "CC BY-SA 4.0",
    "author": "FOURTEEN IT",
    "owner_email": "owner@example.org",
    "category": "Education",
    "feed_explicit": False,
}


def episode(episode_id, day, explicit=False, title="Ei & Co"):
    return FeedEpisode(
        episode_id=episode_id,
        title=title,
        description="Text",
        page_url=f"https://example.org/gzw/episodes/{episode_id}.html",
        audio_url=f"https://archive.org/download/x_{episode_id}/episode.mp3",
        audio_bytes=14_000_000,
        audio_mime="audio/mpeg",
        seconds=905.4,
        cover_url=f"https://archive.org/download/x_{episode_id}/cover.png",
        published_at=datetime(2026, 11, day, 9, 0, tzinfo=UTC),
        explicit=explicit,
    )


def parse(episodes):
    xml = build_feed(CHANNEL, episodes, "https://example.org/gzw/feed.xml", "https://e/art.png")
    return ET.fromstring(xml)


def it(tag):
    return f"{{{ITUNES}}}{tag}"


def test_duration_text():
    assert duration_text(905.4) == "0:15:05"
    assert duration_text(3725) == "1:02:05"


def test_channel_tags():
    channel = parse([]).find("channel")
    assert channel.find("title").text == "Gut zu wissen"
    assert channel.find("language").text == "de"
    assert channel.find(it("explicit")).text == "false"
    assert channel.find(it("category")).get("text") == "Education"
    assert channel.find(it("owner")).find(it("email")).text == "owner@example.org"
    assert channel.find(it("image")).get("href") == "https://e/art.png"


def test_item_has_stable_guid_enclosure_and_episode_tags():
    item = parse([episode("2026-11-22_ei", 22, explicit=True)]).find("channel/item")
    guid = item.find("guid")
    assert (guid.text, guid.get("isPermaLink")) == ("2026-11-22_ei", "false")
    enclosure = item.find("enclosure")
    assert enclosure.get("length") == "14000000"
    assert enclosure.get("type") == "audio/mpeg"
    assert item.find(it("duration")).text == "0:15:05"
    assert item.find(it("explicit")).text == "true"
    assert item.find(it("image")).get("href").endswith("/cover.png")
    assert item.find("pubDate").text == "Sun, 22 Nov 2026 09:00:00 GMT"
    assert item.find("title").text == "Ei & Co"  # escaped on the wire, plain after parsing


def test_newest_episode_first_and_guids_unique():
    items = parse([episode("a", 20), episode("c", 23), episode("b", 22)]).findall("channel/item")
    assert [i.find("guid").text for i in items] == ["c", "b", "a"]


def test_output_is_byte_identical_for_the_same_input():
    episodes = [episode("a", 20), episode("b", 22)]
    args = (CHANNEL, episodes, "https://f/feed.xml", "https://e/art.png")
    assert build_feed(*args) == build_feed(*args)
