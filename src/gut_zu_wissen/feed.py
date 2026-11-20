"""RSS feed with iTunes tags (phase 2; decision D7).

Pure function of the channel configuration and the published episodes. The GUID is the episode
ID (stable, not a link: `isPermaLink="false"`). Explicit is `false` on the channel and per
episode from the explicit check. Newest episode first.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from email.utils import format_datetime

ITUNES = "http://www.itunes.com/dtds/podcast-1.0.dtd"
ATOM = "http://www.w3.org/2005/Atom"
ET.register_namespace("itunes", ITUNES)
ET.register_namespace("atom", ATOM)


@dataclass(frozen=True)
class FeedEpisode:
    episode_id: str
    title: str
    description: str
    page_url: str
    audio_url: str
    audio_bytes: int
    audio_mime: str
    seconds: float
    cover_url: str
    published_at: datetime
    explicit: bool


def _bool(value: bool) -> str:
    return "true" if value else "false"


def duration_text(seconds: float) -> str:
    total = round(seconds)
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}"


def _add(parent: ET.Element, tag: str, content: str | None = None, **attrs: str) -> ET.Element:
    element = ET.SubElement(parent, tag, attrs)
    if content is not None:
        element.text = content
    return element


def build_feed(
    channel: Mapping, episodes: Sequence[FeedEpisode], feed_url: str, show_art_url: str
) -> bytes:
    root = ET.Element("rss", {"version": "2.0"})
    node = _add(root, "channel")
    _add(node, "title", channel["title"])
    _add(node, "link", channel["base_url"])
    _add(node, "description", channel.get("description_html", channel["description"]))
    _add(node, "language", channel["language"])
    _add(node, "copyright", channel["copyright"])
    _add(node, f"{{{ATOM}}}link", href=feed_url, rel="self", type="application/rss+xml")
    _add(node, f"{{{ITUNES}}}author", channel["author"])
    _add(node, f"{{{ITUNES}}}summary", channel["description"])
    _add(node, f"{{{ITUNES}}}explicit", _bool(channel["feed_explicit"]))
    _add(node, f"{{{ITUNES}}}type", "episodic")
    _add(node, f"{{{ITUNES}}}image", href=show_art_url)
    _add(node, f"{{{ITUNES}}}category", **{"text": channel["category"]})
    owner = _add(node, f"{{{ITUNES}}}owner")
    _add(owner, f"{{{ITUNES}}}name", channel["author"])
    _add(owner, f"{{{ITUNES}}}email", channel["owner_email"])
    image = _add(node, "image")
    _add(image, "url", show_art_url)
    _add(image, "title", channel["title"])
    _add(image, "link", channel["base_url"])

    for episode in sorted(episodes, key=lambda e: e.published_at, reverse=True):
        item = _add(node, "item")
        _add(item, "title", episode.title)
        _add(item, "link", episode.page_url)
        _add(item, "description", episode.description)
        _add(item, "guid", episode.episode_id, isPermaLink="false")
        _add(item, "pubDate", format_datetime(episode.published_at, usegmt=True))
        _add(
            item,
            "enclosure",
            url=episode.audio_url,
            length=str(episode.audio_bytes),
            type=episode.audio_mime,
        )
        _add(item, f"{{{ITUNES}}}duration", duration_text(episode.seconds))
        _add(item, f"{{{ITUNES}}}explicit", _bool(episode.explicit))
        _add(item, f"{{{ITUNES}}}image", href=episode.cover_url)
        _add(item, f"{{{ITUNES}}}episodeType", "full")
    ET.indent(root)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
