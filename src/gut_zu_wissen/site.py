"""Static GitHub Pages site and RSS feed: `python -m gut_zu_wissen.site [output dir]` (phase 2).

Built from the archive (published episodes), `site/*.md` (legal pages) and
`site/so-funktionierts.html` (copied as is). Requirements from the data protection statement:
no cookies, no tracking, no external resources (no scripts, fonts, images or stylesheets from
elsewhere; the cover is rendered from the title at build time and served from the site), the
audio player loads only on play (`preload="none"`), Impressum and Datenschutz linked on every
page. Everything is a pure function of the archive and the config, except `build_site`, which
writes the files.
"""

from __future__ import annotations

import html
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import markdown

from gut_zu_wissen.archive import (
    EPISODE_FILE,
    EPISODES_DIR,
    PUBLICATION_FILE,
    EpisodeRecord,
    parse_timestamp,
    read_json,
)
from gut_zu_wissen.cover import render_png
from gut_zu_wissen.feed import FeedEpisode, build_feed
from gut_zu_wissen.shownotes import (
    Source,
    episode_sources,
    feedback_mailto,
    history_url,
    permalink,
)

MONTHS = (
    "Januar Februar März April Mai Juni Juli August September Oktober November Dezember"
).split()
LICENCE_URL = "https://creativecommons.org/licenses/by-sa/4.0/deed.de"

CSS = """
:root { --bg:#fffdf8; --text:#1b1b1b; --muted:#464646; --line:#c9c2b4; --card:#f4efe4; --accent:#1d4f7a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#17181a; --text:#f0ede6; --muted:#cfcac0; --line:#55524c; --card:#23252a; --accent:#9cc6ec; }
}
* { box-sizing: border-box; }
html { font-size: 125%; -webkit-text-size-adjust: 100%; }
body { margin:0; background:var(--bg); color:var(--text); line-height:1.65; overflow-wrap:break-word;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main, header.site, footer.site { max-width: 42rem; margin: 0 auto; padding: 0 1rem; }
header.site { padding-top: 1.5rem; }
header.site a { font-weight: 700; text-decoration: none; }
h1 { font-size: 2rem; line-height: 1.2; margin: 0.5rem 0 1rem; }
h2 { font-size: 1.4rem; line-height: 1.25; margin: 2.5rem 0 0.75rem; }
p, ul { margin: 0 0 1rem; }
li { margin-bottom: 0.4rem; }
a { color: var(--accent); text-underline-offset: 0.15em; }
a:focus-visible, audio:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; }
audio { width: 100%; margin: 0.5rem 0 1rem; }
img { max-width: 100%; height: auto; border-radius: 0.3rem; }
.meta { color: var(--muted); }
.box { background: var(--card); border-left: 6px solid var(--accent); padding: 1rem 1.1rem;
  margin: 1.5rem 0; border-radius: 0.3rem; }
.box > :last-child { margin-bottom: 0; }
footer.site { margin-top: 3rem; padding-bottom: 2rem; border-top: 2px solid var(--line); color: var(--muted); }
footer.site ul { list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 0.4rem 1.2rem; margin: 1rem 0 0.5rem; }
"""


@dataclass(frozen=True)
class PublishedEpisode:
    record: EpisodeRecord
    publication: dict
    sources: list[Source]

    @property
    def published_at(self) -> datetime:
        return parse_timestamp(self.record.published_at or "")


def german_date(moment: datetime) -> str:
    return f"{moment.day}. {MONTHS[moment.month - 1]} {moment.year}"


def load_published(archive_root: Path) -> list[PublishedEpisode]:
    """Published episodes, newest first. A published episode without 10_publication.json is an
    error (the archive is inconsistent), not something to skip silently."""
    episodes = []
    for folder in sorted((archive_root / EPISODES_DIR).iterdir()):
        record = EpisodeRecord(**read_json(folder / EPISODE_FILE))
        if record.status != "published":
            continue
        publication = read_json(folder / PUBLICATION_FILE)
        episodes.append(PublishedEpisode(record, publication, episode_sources(folder)))
    return sorted(episodes, key=lambda e: e.published_at, reverse=True)


# --- pages -----------------------------------------------------------------


def page(title: str, body: str, root: str, config: dict, description: str = "") -> str:
    """One page. `root` is the relative path to the site root ('' or '../')."""
    meta = (
        f'<meta name="description" content="{html.escape(description)}">\n' if description else ""
    )
    return f"""<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
{meta}<link rel="alternate" type="application/rss+xml" title="{html.escape(config["title"])}" href="{root}feed.xml">
<style>{CSS}</style>
</head>
<body>
<header class="site"><a href="{root}index.html">{html.escape(config["title"])}</a></header>
<main>
{body}
</main>
<footer class="site">
  <nav aria-label="Rechtliches">
    <ul>
      <li><a href="{root}impressum.html">Impressum</a></li>
      <li><a href="{root}datenschutz.html">Datenschutzerklärung</a></li>
      <li><a href="{root}so-funktionierts.html">So entsteht eine Folge</a></li>
      <li><a href="{root}feed.xml">RSS-Feed</a></li>
    </ul>
  </nav>
  <p>Ein Projekt der {html.escape(config["author"])}. Inhalte aus der deutschsprachigen Wikipedia, Lizenz <a href="{LICENCE_URL}">CC BY-SA 4.0</a>. Unabhängig von der Wikimedia Foundation.</p>
</footer>
</body>
</html>
"""


def episode_page_name(episode: PublishedEpisode) -> str:
    return f"episodes/{episode.record.episode_id}.html"


def cover_name(episode: PublishedEpisode) -> str:
    """The cover is rendered at build time and served from the site itself (not from the
    Internet Archive): no external resource on the page, and a direct, stable feed image URL."""
    return f"covers/{episode.record.episode_id}.png"


def render_episode(episode: PublishedEpisode, config: dict) -> str:
    record, audio = episode.record, episode.publication["audio"]
    index = config["wikipedia_index"]
    sources = "\n".join(
        f'  <li><a href="{html.escape(permalink(index, s))}">„{html.escape(s.title)}“</a>'
        f' (<a href="{html.escape(history_url(index, s))}">Versionsgeschichte und Autorinnen und Autoren</a>)</li>'
        for s in episode.sources
    )
    explicit = (
        "<p><strong>Hinweis:</strong> Diese Folge enthält Inhalte, die nicht für alle Hörer geeignet sind.</p>\n"
        if record.explicit
        else ""
    )
    mailto = feedback_mailto(config["feedback_address"], record.title, record.episode_id)
    body = f"""<h1>{html.escape(record.title)}</h1>
<p class="meta">Folge vom {german_date(episode.published_at)}</p>
<img src="../{cover_name(episode)}" alt="Cover der Folge „{html.escape(record.title)}“" width="400" height="400">
<audio controls preload="none" src="{html.escape(audio["url"])}">Ihr Browser kann die Audiodatei nicht abspielen. <a href="{html.escape(audio["url"])}">Folge herunterladen</a>.</audio>
{explicit}<p>Diese Folge ist mit KI entstanden, von der Auswahl der Fakten über das Gespräch bis zu den Stimmen. KI kann Fehler machen, und die Wikipedia-Inhalte selbst sind nicht unabhängig geprüft. <a href="../so-funktionierts.html">So entsteht eine Folge.</a></p>
<h2>Quellen</h2>
<p>Alle Inhalte stammen aus diesen Fassungen der deutschsprachigen Wikipedia (Lizenz <a href="{LICENCE_URL}">CC BY-SA 4.0</a>):</p>
<ul>
{sources}
</ul>
<h2>Rückmeldung</h2>
<p>Fehler gefunden oder etwas zu sagen? <a href="{html.escape(mailto)}">Schreiben Sie uns</a> (die Folgen-ID steht schon im Betreff).</p>"""
    return page(
        f"{record.title} – {config['title']}", body, "../", config, f"Folge über {record.title}"
    )


def render_index(episodes: list[PublishedEpisode], config: dict) -> str:
    items = "\n".join(
        f'  <li><a href="{episode_page_name(e)}">{html.escape(e.record.title)}</a>'
        f' <span class="meta">({german_date(e.published_at)})</span></li>'
        for e in episodes
    )
    listing = f"<ul>\n{items}\n</ul>" if items else "<p>Die erste Folge erscheint in Kürze.</p>"
    body = f"""<h1>{html.escape(config["title"])}</h1>
<p>{html.escape(config["description"])}</p>
<div class="box"><p>Den Podcast abonnieren: <a href="feed.xml">RSS-Feed</a> in der Podcast-App Ihrer Wahl einfügen.</p></div>
<h2>Folgen</h2>
{listing}"""
    return page(config["title"], body, "", config, config["description"])


def render_legal(title: str, markdown_text: str, config: dict) -> str:
    body = markdown.markdown(markdown_text, extensions=["nl2br"])
    return page(f"{title} – {config['title']}", body, "", config)


def feed_episodes(episodes: list[PublishedEpisode], config: dict) -> list[FeedEpisode]:
    base = config["base_url"].rstrip("/")
    result = []
    for e in episodes:
        publication = e.publication
        page_url = f"{base}/{episode_page_name(e)}"
        result.append(
            FeedEpisode(
                episode_id=e.record.episode_id,
                title=e.record.title,
                description=config["episode_description"].format(
                    title=html.escape(e.record.title),
                    page_url=page_url,
                    howto_url=f"{base}/so-funktionierts.html",
                    feedback_url=feedback_mailto(
                        config["feedback_address"], e.record.title, e.record.episode_id
                    ),
                ),
                page_url=page_url,
                audio_url=publication["audio"]["url"],
                audio_bytes=publication["audio"]["bytes"],
                audio_mime=publication["audio"]["mime"],
                seconds=publication["seconds"],
                cover_url=f"{base}/{cover_name(e)}",
                published_at=e.published_at,
                explicit=bool(e.record.explicit),
            )
        )
    return result


# --- build -----------------------------------------------------------------


def build_site(archive_root: Path, repo_root: Path, out_dir: Path, config: dict) -> list[Path]:
    """Write the site; returns the files written. An existing output folder is replaced."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    episodes = load_published(archive_root)
    base = config["base_url"].rstrip("/")
    files: dict[str, str | bytes] = {
        "index.html": render_index(episodes, config),
        "feed.xml": build_feed(
            {
                **config,
                "description_html": config["description_html"].format(
                    description=html.escape(config["description"]),
                    base_url=base,
                    feedback_address=config["feedback_address"],
                ),
            },
            feed_episodes(episodes, config),
            f"{base}/feed.xml",
            f"{base}/{config['show_art_file']}",
        ),
        "impressum.html": render_legal(
            "Impressum", (repo_root / "site" / "impressum.md").read_text("utf-8"), config
        ),
        "datenschutz.html": render_legal(
            "Datenschutzerklärung",
            (repo_root / "site" / "datenschutz.md").read_text("utf-8"),
            config,
        ),
        "so-funktionierts.html": (repo_root / "site" / "so-funktionierts.html").read_text("utf-8"),
        config["show_art_file"]: (repo_root / config["show_art_source"]).read_bytes(),
    }
    for e in episodes:
        files[episode_page_name(e)] = render_episode(e, config)
        files[cover_name(e)] = render_png(e.record.title, config["cover_size_px"])
    written = []
    for name, content in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    from gut_zu_wissen.run import ARCHIVE_ROOT, REPO_ROOT, load_config

    args = sys.argv[1:] if argv is None else argv
    out_dir = Path(args[0]) if args else REPO_ROOT / "build" / "site"
    written = build_site(ARCHIVE_ROOT, REPO_ROOT, out_dir, load_config()["site"])
    print(f"Site with {len(written)} files written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
