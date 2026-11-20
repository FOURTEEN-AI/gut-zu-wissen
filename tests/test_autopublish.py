import json
from datetime import UTC, datetime, timedelta

from gut_zu_wissen.archive import EpisodeRecord
from gut_zu_wissen.autopublish import find_due, verdict

NOW = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
FINISHED = "2026-11-21T05:30:00Z"


def record(**changes) -> EpisodeRecord:
    values = {
        "episode_id": "2026-11-21_x",
        "title": "X",
        "status": "completed",
        "started_at": "2026-11-21T03:17:00Z",
        "finished_at": FINISHED,
        "explicit": False,
        "audio_run_id": "123",
    }
    return EpisodeRecord(**{**values, **changes})


def check(rec=None, review=None, labels=(), now=NOW, hours=6.0):
    return verdict(rec or record(), review, list(labels), now, hours, "hold")


def test_episode_is_due_when_old_enough():
    result = check()
    assert result.due and result.run_id == "123"


def test_episode_that_ended_too_recently_waits():
    result = check(now=datetime(2026, 11, 21, 11, 29, tzinfo=UTC))
    assert not result.due and "too recent: ready from 2026-11-21T11:30:00" in result.reason


def test_hold_label_stops_the_publication_and_removing_it_releases():
    assert "held" in check(labels=["hold"]).reason
    assert check(labels=["other"]).due


def test_review_error_stops_the_publication():
    result = check(review={"outcome": "error"})
    assert not result.due and "review found an error" in result.reason
    assert check(review={"outcome": "clean"}).due
    assert check(review={"outcome": None}).due


def test_episode_without_audio_run_or_not_completed_is_never_due():
    assert not check(record(audio_run_id=None)).due
    for status in ("failed", "running", "published"):
        assert not check(record(status=status)).due


def test_missing_explicit_check_blocks():
    assert not check(record(explicit=None)).due


class Issues:
    def __init__(self, labels):
        self._labels = labels
        self.asked = []

    def labels(self, number):
        self.asked.append(number)
        return self._labels.get(number, [])


def write(root, rec, review=None):
    folder = root / "episodes" / rec.episode_id
    folder.mkdir(parents=True)
    (folder / "episode.json").write_text(json.dumps(rec.__dict__), encoding="utf-8")
    if review is not None:
        (folder / "09_review.json").write_text(json.dumps(review), encoding="utf-8")


def test_find_due_takes_the_oldest_due_episode_and_asks_the_issue(tmp_path):
    old = record(episode_id="2026-11-20_a", finished_at="2026-11-20T05:00:00Z", audio_run_id="1")
    new = record(episode_id="2026-11-21_b", finished_at="2026-11-21T05:00:00Z", audio_run_id="2")
    write(tmp_path, old, {"issue": 10})
    write(tmp_path, new, {"issue": 11})
    issues = Issues({10: ["hold"]})
    logged = []
    due = find_due(tmp_path, NOW, 6, "hold", issues, logged.append)
    assert due.episode_id == "2026-11-21_b" and due.run_id == "2"
    assert issues.asked == [10, 11]
    assert any("2026-11-20_a: not due (held" in line for line in logged)


def test_find_due_without_a_due_episode_is_none(tmp_path):
    write(tmp_path, record(finished_at=(NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")))
    assert find_due(tmp_path, NOW, 6, "hold", None, lambda _: None) is None
