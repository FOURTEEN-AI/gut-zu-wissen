"""Status page of the Pages site: what is being made right now (user decision 2026-11-22).

A plain-language overview for listeners without technical knowledge: the episode being made and
its step, finished episodes waiting to appear (with the expected time of the automatic
publication), the latest published episodes and the runs of the last days that did not work out.

Sources:

- the archive (episodes that ended: completed, published, failed; `09_review.json`),
- the live file `live.json` on the branch `status`, written by the running pipeline at every step
  (`progress.py`); the archive itself is committed only when a run ends,
- the labels of the review issues (the owner's `hold` label, `autopublish.py`).

Everything here is a pure function of these inputs and the build time; `site.py` writes the page.
The page reloads its own content every `refresh_seconds` with an inline script (same origin, no
external resource); without JavaScript a `<noscript>` refresh does the same. Times are absolute
(a relative "for 12 minutes" would go stale between two builds).
"""

from __future__ import annotations

import html
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from gut_zu_wissen.archive import STEPS, EpisodeRecord, parse_timestamp
from gut_zu_wissen.autopublish import verdict

WEEKDAYS = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")
MONTHS = (
    "Januar Februar März April Mai Juni Juli August September Oktober November Dezember"
).split()

# One sentence per pipeline step (`archive.STEPS`), in run order.
STEP_TEXTS = {
    "gates": "Den Wikipedia-Artikel holen und prüfen, ob er sich für eine Folge eignet",
    "segmentation": "Den Artikel in einzelne Sätze zerlegen",
    "extraction": "Die überraschendsten Fakten aus dem Artikel heraussuchen",
    "dialog": "Lena und Felix bekommen ihr Gespräch geschrieben. Jede Aussage wird mit dem "
    "Artikel verglichen",
    "explicit": "Prüfen, ob die Folge für alle Ohren geeignet ist",
    "audio": "Die Stimmen aufnehmen und zur Kontrolle noch einmal anhören",
    "review": "Stichproben auswählen, die ein Mensch nachprüft",
}

# Plain reasons of a failed run, by the start of the archived (English) failure text.
FAILURE_TEXTS = (
    (
        "Gate living person",
        "Der Artikel handelt von einem lebenden Menschen. Über lebende Menschen macht der "
        "Podcast keine Folgen.",
    ),
    ("Gate repeat", "Über diesen Artikel gab es vor Kurzem schon eine Folge."),
    ("Gate minimum content", "Der Artikel hatte nicht genug Stoff für eine ganze Folge."),
    ("Gate top facts", "Die wichtigsten Fakten des Artikels passten nicht ins Gespräch."),
    (
        "Conversation regeneration budget",
        "Es gelang kein Gespräch, in dem jede Aussage nachweislich stimmt. Lieber keine Folge "
        "als eine mit Fehlern.",
    ),
)
FAILURE_DEFAULT = "Eine technische Störung hat die Arbeit unterbrochen."


# ---------------------------------------------------------------- data


@dataclass(frozen=True)
class Live:
    """Content of `live.json`: the step a running pipeline has reached."""

    episode_id: str
    title: str
    step: str
    started_at: str
    step_started_at: str

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "title": self.title,
            "step": self.step,
            "started_at": self.started_at,
            "step_started_at": self.step_started_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> Live:
        return cls(**{k: str(data[k]) for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Episode:
    """An episode of the archive with what the status page needs of it."""

    record: EpisodeRecord
    review: Mapping | None = None
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Running:
    title: str
    step_index: int  # 0-based index into STEPS
    started_at: datetime
    step_started_at: datetime


@dataclass(frozen=True)
class Waiting:
    title: str
    finished_at: datetime
    publish_at: datetime | None  # None: not published automatically
    note: str  # "hold", "error", "manual", "review" or ""


@dataclass(frozen=True)
class Done:
    episode_id: str
    title: str
    published_at: datetime


@dataclass(frozen=True)
class Failed:
    title: str
    finished_at: datetime
    reason: str


@dataclass(frozen=True)
class Overview:
    built_at: datetime
    running: Running | None
    waiting: list[Waiting]
    published: list[Done]
    failed: list[Failed]


# ---------------------------------------------------------------- pure decisions


def running_from(
    live: Live | None, episodes: Sequence[Episode], now: datetime, max_age_hours: float
) -> Running | None:
    """The running episode, or None. The live file is stale once its run ended in the archive
    (the episode's `finished_at` is not before the last step change; an audio resume updates the
    live file after the earlier end), or after `max_age_hours` (a run killed before its archive
    commit); an unknown step is ignored."""
    if live is None or live.step not in STEPS:
        return None
    changed = parse_timestamp(live.step_started_at)
    if now - changed > timedelta(hours=max_age_hours):
        return None
    for e in episodes:
        r = e.record
        if r.episode_id == live.episode_id and r.status != "running" and r.finished_at:
            if parse_timestamp(r.finished_at) >= changed:
                return None
    return Running(live.title, STEPS.index(live.step), parse_timestamp(live.started_at), changed)


def next_slot(moment: datetime, daily_time_utc: str) -> datetime:
    """The first daily publication time (`HH:MM` UTC) at or after `moment`."""
    hour, minute = (int(part) for part in daily_time_utc.split(":"))
    slot = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return slot if slot >= moment else slot + timedelta(days=1)


def waiting_list(
    episodes: Sequence[Episode],
    now: datetime,
    min_age_hours: float,
    hold_label: str,
    daily_time_utc: str,
) -> list[Waiting]:
    """Completed episodes, oldest first, with the expected automatic publication.

    Mirrors `autopublish.find_due`: one episode per daily run, the oldest due one first; an
    episode that is held, has a review error or no audio run is skipped and gets no time.
    """
    completed = sorted(
        (e for e in episodes if e.record.status == "completed" and e.record.finished_at),
        key=lambda e: e.record.finished_at or "",
    )
    result = []
    free = next_slot(now, daily_time_utc)
    for e in completed:
        finished = parse_timestamp(e.record.finished_at or "")
        ready = finished + timedelta(hours=min_age_hours)
        check = verdict(e.record, e.review, list(e.labels), ready, min_age_hours, hold_label)
        publish_at = None
        if check.due:
            publish_at = next_slot(max(ready, free), daily_time_utc)
            free = publish_at + timedelta(minutes=1)
            note = "review" if e.review and e.review.get("status") != "answered" else ""
        elif hold_label in e.labels:
            note = "hold"
        elif e.review and e.review.get("outcome") == "error":
            note = "error"
        else:
            note = "manual"
        result.append(Waiting(e.record.title, finished, publish_at, note))
    return result


def failure_text(failure: str | None) -> str:
    for prefix, text in FAILURE_TEXTS:
        if (failure or "").startswith(prefix):
            return text
    return FAILURE_DEFAULT


def overview(
    episodes: Sequence[Episode],
    live: Live | None,
    now: datetime,
    status_config: Mapping,
    publishing_config: Mapping,
) -> Overview:
    published = sorted(
        (
            Done(e.record.episode_id, e.record.title, parse_timestamp(e.record.published_at))
            for e in episodes
            if e.record.status == "published" and e.record.published_at
        ),
        key=lambda d: d.published_at,
        reverse=True,
    )[: status_config["recent_published"]]
    since = now - timedelta(days=status_config["failed_days"])
    failed = sorted(
        (
            Failed(
                e.record.title,
                parse_timestamp(e.record.finished_at),
                failure_text(e.record.failure),
            )
            for e in episodes
            if e.record.status == "failed"
            and e.record.finished_at
            and parse_timestamp(e.record.finished_at) >= since
        ),
        key=lambda f: f.finished_at,
        reverse=True,
    )
    return Overview(
        built_at=now,
        running=running_from(live, episodes, now, status_config["live_max_age_hours"]),
        waiting=waiting_list(
            episodes,
            now,
            publishing_config["min_age_hours"],
            publishing_config["hold_label"],
            publishing_config["daily_time_utc"],
        ),
        published=published,
        failed=failed,
    )


# ---------------------------------------------------------------- rendering


def local_time(moment: datetime, zone: str) -> str:
    """'Sonntag, 22. November, 11:00 Uhr' in the given time zone."""
    t = moment.astimezone(ZoneInfo(zone))
    return f"{WEEKDAYS[t.weekday()]}, {t.day}. {MONTHS[t.month - 1]}, {t:%H:%M} Uhr"


def clock(moment: datetime, zone: str) -> str:
    return f"{moment.astimezone(ZoneInfo(zone)):%H:%M} Uhr"


def _title(title: str) -> str:
    return f"„{html.escape(title)}“"


def render_running(running: Running | None, zone: str) -> str:
    if running is None:
        return (
            '<div class="box"><p>Im Moment entsteht keine neue Folge. '
            "Jeden Morgen beginnt automatisch die nächste.</p></div>"
        )
    total = len(STEPS)
    number = running.step_index + 1
    items = []
    for i, step in enumerate(STEPS):
        if i < running.step_index:
            mark, cls, label = "✓", "done", "erledigt"
        elif i == running.step_index:
            mark, cls, label = "▶", "now", "gerade dabei"
        else:
            mark, cls, label = "○", "todo", "kommt noch"
        items.append(
            f'  <li class="{cls}"><span class="mark" aria-hidden="true">{mark}</span> '
            f'{html.escape(STEP_TEXTS[step])}<span class="sr"> ({label})</span></li>'
        )
    steps = "\n".join(items)
    percent = round(100 * running.step_index / total)
    return f"""<div class="box now">
<p class="big">Folge über {_title(running.title)}</p>
<p>Schritt {number} von {total}: <strong>{html.escape(STEP_TEXTS[STEPS[running.step_index]])}.</strong></p>
<div class="bar" role="img" aria-label="{number - 1} von {total} Schritten erledigt"><div style="width:{percent}%"></div></div>
<p class="meta">Angefangen um {clock(running.started_at, zone)}, dieser Schritt seit {clock(running.step_started_at, zone)}.</p>
</div>
<ol class="steps">
{steps}
</ol>"""


WAITING_NOTES = {
    "hold": "Angehalten. Die Folge erscheint erst, wenn sie von Hand freigegeben wird.",
    "error": "Bei der Prüfung ist ein Fehler aufgefallen. Die Folge erscheint erst, wenn sie "
    "korrigiert oder von Hand freigegeben wird.",
    "manual": "Die Folge erscheint, sobald sie von Hand freigegeben wird.",
    "review": "Ein Mensch hört gerade Stichproben nach. Fällt dabei ein Fehler auf, wird die "
    "Folge angehalten.",
}


def render_waiting(waiting: Sequence[Waiting], zone: str) -> str:
    if not waiting:
        return "<p>Gerade wartet keine fertige Folge.</p>"
    items = []
    for w in waiting:
        when = (
            f"<p><strong>Erscheint am {local_time(w.publish_at, zone)}.</strong></p>"
            if w.publish_at
            else ""
        )
        note = f"<p>{html.escape(WAITING_NOTES[w.note])}</p>" if w.note else ""
        items.append(
            f'<div class="box wait"><p class="big">Folge über {_title(w.title)}</p>'
            f'<p class="meta">Fertig seit {local_time(w.finished_at, zone)}.</p>{when}{note}</div>'
        )
    return "\n".join(items)


def render_published(published: Sequence[Done], zone: str, page_name: Callable[[str], str]) -> str:
    if not published:
        return "<p>Noch keine Folge erschienen.</p>"
    items = "\n".join(
        f'  <li><a href="{page_name(d.episode_id)}">{html.escape(d.title)}</a>'
        f' <span class="meta">({local_time(d.published_at, zone)})</span></li>'
        for d in published
    )
    return f"<ul>\n{items}\n</ul>"


def render_failed(failed: Sequence[Failed], zone: str, days: int) -> str:
    if not failed:
        return f"<p>In den letzten {days} Tagen ist nichts schiefgegangen.</p>"
    return "\n".join(
        f'<div class="box fail"><p class="big">{_title(f.title)}</p>'
        f'<p class="meta">{local_time(f.finished_at, zone)}</p>'
        f"<p>{html.escape(f.reason)}</p></div>"
        for f in failed
    )


STATUS_CSS = """
:root { --warn:#7a4f00; --no:#9b2222; }
@media (prefers-color-scheme: dark) { :root { --warn:#f0c870; --no:#f2a39b; } }
.big { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.4rem; }
.box.wait { border-left-color: var(--warn); }
.box.fail { border-left-color: var(--no); }
.bar { height: 1.2rem; background: var(--bg); border: 2px solid var(--line); border-radius: 0.6rem; overflow: hidden; margin: 0.5rem 0 1rem; }
.bar > div { height: 100%; background: var(--accent); }
ol.steps { list-style: none; padding: 0; }
ol.steps li { padding: 0.3rem 0; }
ol.steps .mark { display: inline-block; width: 1.6rem; font-weight: 700; }
ol.steps li.done { color: var(--muted); }
ol.steps li.now { font-weight: 700; }
ol.steps li.todo { color: var(--muted); }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
"""

# Reloads the page's own <main> from the same address without the browser cache; nothing is
# loaded from elsewhere. A failed fetch keeps the current content.
REFRESH_SCRIPT = """<script>
setInterval(function () {
  fetch(location.pathname, {cache: "no-store"}).then(function (r) {
    return r.ok ? r.text() : null;
  }).then(function (text) {
    if (!text) return;
    var fresh = new DOMParser().parseFromString(text, "text/html").querySelector("main");
    if (fresh) document.querySelector("main").innerHTML = fresh.innerHTML;
  }).catch(function () {});
}, %d);
</script>"""


def render_body(o: Overview, status_config: Mapping, page_name: Callable[[str], str]) -> str:
    zone = status_config["timezone"]
    days = status_config["failed_days"]
    return f"""<h1>Was gerade entsteht</h1>
<p>Hier sehen Sie, woran der Podcast gerade arbeitet. Die Seite aktualisiert sich von selbst.</p>
<p class="meta">Stand: {local_time(o.built_at, zone)}</p>
<h2>Gerade in Arbeit</h2>
{render_running(o.running, zone)}
<h2>Fertig und bald zu hören</h2>
{render_waiting(o.waiting, zone)}
<h2>Zuletzt erschienen</h2>
{render_published(o.published, zone, page_name)}
<h2>Hat nicht geklappt</h2>
{render_failed(o.failed, zone, days)}
<p><a href="so-funktionierts.html">So entsteht eine Folge, Schritt für Schritt erklärt.</a></p>"""


def head_extra(status_config: Mapping) -> str:
    seconds = int(status_config["refresh_seconds"])
    return (
        f"<style>{STATUS_CSS}</style>\n"
        f'<noscript><meta http-equiv="refresh" content="{seconds}"></noscript>\n'
        f"{REFRESH_SCRIPT % (seconds * 1000)}\n"
    )
