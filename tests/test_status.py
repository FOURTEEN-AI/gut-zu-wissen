"""Status page "Was gerade entsteht": pure decisions and rendering."""

import re
from datetime import UTC, datetime

from gut_zu_wissen.archive import STEPS, EpisodeRecord
from gut_zu_wissen.run import REPO_ROOT, load_config
from gut_zu_wissen.status import (
    FAILURE_DEFAULT,
    STEP_TEXTS,
    Episode,
    Live,
    failure_text,
    local_time,
    next_slot,
    overview,
    render_body,
    running_from,
    waiting_list,
)

CONFIG = load_config()
STATUS = CONFIG["status"]
PUBLISHING = CONFIG["publishing"]
NOW = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)
HOLD = PUBLISHING["hold_label"]


def record(eid="2026-11-22_a", status="completed", finished="2026-11-22T05:00:00Z", **kw):
    fields = dict(
        episode_id=eid,
        title=kw.pop("title", eid),
        status=status,
        started_at="2026-11-22T04:00:00Z",
        finished_at=finished,
        explicit=False,
        audio_run_id="123",
    )
    return EpisodeRecord(**(fields | kw))


def live(step="dialog", changed="2026-11-22T11:30:00Z", eid="2026-11-22_b"):
    return Live(eid, "Kölner Dom", step, "2026-11-22T11:00:00Z", changed)


# ---------------------------------------------------------------- running


def test_every_step_has_a_text():
    assert set(STEP_TEXTS) == set(STEPS)


def test_live_file_shows_its_step():
    running = running_from(live(), [], NOW, 5)
    assert running.title == "Kölner Dom" and running.step_index == STEPS.index("dialog")


def test_no_live_file_or_unknown_step_shows_nothing():
    assert running_from(None, [], NOW, 5) is None
    assert running_from(live(step="something"), [], NOW, 5) is None


def test_old_live_file_is_stale():
    assert running_from(live(changed="2026-11-22T06:59:00Z"), [], NOW, 5) is None
    assert running_from(live(changed="2026-11-22T07:00:00Z"), [], NOW, 5) is not None


def test_live_file_is_stale_once_the_run_ended_in_the_archive():
    ended = Episode(record("2026-11-22_b", "completed", "2026-11-22T11:30:00Z"))
    assert running_from(live(), [ended], NOW, 5) is None


def test_audio_resume_after_a_failed_run_is_shown():
    failed = Episode(record("2026-11-22_b", "failed", "2026-11-22T09:00:00Z"))
    assert running_from(live(step="audio"), [failed], NOW, 5) is not None


# ---------------------------------------------------------------- waiting


def test_next_slot():
    assert next_slot(datetime(2026, 11, 22, 9, 0, tzinfo=UTC), "10:00") == datetime(
        2026, 11, 22, 10, 0, tzinfo=UTC
    )
    assert next_slot(datetime(2026, 11, 22, 10, 0, tzinfo=UTC), "10:00") == datetime(
        2026, 11, 22, 10, 0, tzinfo=UTC
    )
    assert next_slot(datetime(2026, 11, 22, 10, 1, tzinfo=UTC), "10:00") == datetime(
        2026, 11, 23, 10, 0, tzinfo=UTC
    )


def test_backlog_is_published_one_per_day_oldest_first():
    episodes = [
        Episode(record("b", finished="2026-11-21T23:00:00Z")),
        Episode(record("a", finished="2026-11-21T05:00:00Z")),
        Episode(record("c", finished="2026-11-22T11:00:00Z")),
    ]
    result = waiting_list(episodes, NOW, 2, HOLD, "10:00")
    assert [w.title for w in result] == ["a", "b", "c"]
    days = [w.publish_at.day for w in result]
    assert days == [23, 24, 25]


def test_a_recent_episode_waits_for_its_minimum_age():
    morning = datetime(2026, 11, 22, 9, 0, tzinfo=UTC)
    episodes = [Episode(record(finished="2026-11-22T08:30:00Z"))]
    (w,) = waiting_list(episodes, morning, 2, HOLD, "10:00")
    assert w.publish_at == datetime(2026, 11, 23, 10, 0, tzinfo=UTC)


def test_held_error_and_manual_episodes_get_no_time_and_take_no_slot():
    episodes = [
        Episode(record("held", finished="2026-11-21T01:00:00Z"), labels=(HOLD,)),
        Episode(
            record("error", finished="2026-11-21T02:00:00Z"),
            review={"status": "answered", "outcome": "error"},
        ),
        Episode(record("manual", finished="2026-11-21T03:00:00Z", audio_run_id=None)),
        Episode(
            record("ok", finished="2026-11-21T04:00:00Z"),
            review={"status": "pending", "outcome": None},
        ),
    ]
    result = {w.title: w for w in waiting_list(episodes, NOW, 2, HOLD, "10:00")}
    assert [result[k].note for k in ("held", "error", "manual", "ok")] == [
        "hold",
        "error",
        "manual",
        "review",
    ]
    assert result["held"].publish_at is None and result["manual"].publish_at is None
    assert result["ok"].publish_at == datetime(2026, 11, 23, 10, 0, tzinfo=UTC)


def test_only_completed_episodes_wait():
    episodes = [Episode(record(status="published")), Episode(record(status="failed"))]
    assert waiting_list(episodes, NOW, 2, HOLD, "10:00") == []


# ---------------------------------------------------------------- failures and overview


def test_failure_texts():
    assert "lebenden Menschen" in failure_text("Gate living person: 'X' is a person article")
    assert "vor Kurzem" in failure_text("Gate repeat (D4): 'X' was published on")
    assert "nicht genug Stoff" in failure_text("Gate minimum content (D5): 3 facts offered")
    assert "nachweislich stimmt" in failure_text(
        "Conversation regeneration budget used up (2x): Turn regeneration budget used up."
    )
    assert failure_text("Crash: LlmError: cut off") == FAILURE_DEFAULT
    assert failure_text(None) == FAILURE_DEFAULT


def test_overview_limits_published_and_failed():
    published = [
        Episode(record(f"p{i}", "published", published_at=f"2026-11-{10 + i}T10:00:00Z"))
        for i in range(8)
    ]
    failed = [
        Episode(record("old", "failed", "2026-11-14T10:00:00Z")),
        Episode(record("new", "failed", "2026-11-21T10:00:00Z", failure="Crash: x")),
    ]
    o = overview(published + failed, None, NOW, STATUS, PUBLISHING)
    assert [d.title for d in o.published] == [f"p{i}" for i in range(7, 7 - 5, -1)]
    assert [f.title for f in o.failed] == ["new"]


def test_local_time_is_german_time():
    assert local_time(NOW, "Europe/Berlin") == "Sonntag, 22. November, 13:00 Uhr"
    summer = datetime(2026, 7, 1, 8, 5, tzinfo=UTC)
    assert local_time(summer, "Europe/Berlin") == "Mittwoch, 1. Juli, 10:05 Uhr"


def test_rendered_body_marks_done_current_and_coming_steps():
    episodes = [Episode(record("w", title="Tenochtitlan", finished="2026-11-21T22:00:00Z"))]
    o = overview(episodes, live(), NOW, STATUS, PUBLISHING)
    body = render_body(o, STATUS, lambda eid: f"episodes/{eid}.html")
    assert body.count('class="done"') == STEPS.index("dialog")
    assert body.count('class="now"') == 1
    assert body.count('class="todo"') == len(STEPS) - STEPS.index("dialog") - 1
    assert "Folge über „Tenochtitlan“" in body
    assert "Erscheint am Montag, 23. November, 11:00 Uhr." in body
    assert "In den letzten 7 Tagen ist nichts schiefgegangen." in body


def test_titles_are_escaped():
    escaped = Live("x", "<b>", "gates", "2026-11-22T11:00:00Z", "2026-11-22T11:00:00Z")
    body = render_body(overview([], escaped, NOW, STATUS, PUBLISHING), STATUS, lambda eid: eid)
    assert "<b>" not in body and "&lt;b&gt;" in body


def test_daily_time_matches_the_auto_publish_schedule():
    text = (REPO_ROOT / ".github" / "workflows" / "auto-publish.yml").read_text("utf-8")
    minute, hour = re.search(r'cron: "(\d+) (\d+) \* \* \*"', text).groups()
    assert PUBLISHING["daily_time_utc"] == f"{int(hour):02d}:{int(minute):02d}"
