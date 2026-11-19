import json
from datetime import UTC, datetime

import pytest

from gut_zu_wissen.archive import (
    EpisodeRecord,
    create_episode,
    episode_dir,
    last_published,
    load_episode,
    save_episode,
    slugify,
)

NOW = datetime(2026, 11, 19, 10, 30, tzinfo=UTC)


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("Rotfuchs", "rotfuchs"),
        ("Python (Programmiersprache)", "python-programmiersprache"),
        ("Größe über Äpfel", "groesse-ueber-aepfel"),
        ("FC Bayern München", "fc-bayern-muenchen"),
        ("Café – Théâtre", "cafe-theatre"),
        ("AC/DC", "ac-dc"),
    ],
)
def test_slugify(title, slug):
    assert slugify(title) == slug


def test_create_episode_writes_stable_indented_json(tmp_path):
    record = create_episode(tmp_path, "Rotfuchs", NOW)
    assert record.episode_id == "2026-11-19_rotfuchs"
    text = (episode_dir(tmp_path, record.episode_id) / "episode.json").read_text()
    assert text.endswith("}\n")
    data = json.loads(text)
    assert list(data) == [
        "episode_id",
        "title",
        "status",
        "started_at",
        "finished_at",
        "published_at",
        "explicit",
        "failure",
        "usage",
        "cost",
    ]
    assert text.splitlines()[1].startswith('  "episode_id"')
    assert data["status"] == "running"
    assert data["started_at"] == "2026-11-19T10:30:00Z"


def test_same_day_retry_gets_suffix(tmp_path):
    first = create_episode(tmp_path, "Rotfuchs", NOW)
    second = create_episode(tmp_path, "Rotfuchs", NOW)
    third = create_episode(tmp_path, "Rotfuchs", NOW)
    assert [first.episode_id, second.episode_id, third.episode_id] == [
        "2026-11-19_rotfuchs",
        "2026-11-19_rotfuchs-2",
        "2026-11-19_rotfuchs-3",
    ]


def test_save_and_load_roundtrip(tmp_path):
    record = create_episode(tmp_path, "Rotfuchs", NOW)
    record.status, record.published_at, record.explicit = "published", "2026-11-20T03:00:00Z", False
    save_episode(tmp_path, record)
    assert load_episode(tmp_path, record.episode_id) == record


def _episode(tmp_path, title, day, status, published_at):
    record = create_episode(tmp_path, title, datetime(2026, 1, day, tzinfo=UTC))
    record.status, record.published_at = status, published_at
    save_episode(tmp_path, record)


def test_last_published_without_archive(tmp_path):
    assert last_published(tmp_path, "Rotfuchs") is None


def test_last_published_ignores_failed_running_completed_and_other_titles(tmp_path):
    _episode(tmp_path, "Rotfuchs", 1, "failed", None)
    _episode(tmp_path, "Rotfuchs", 2, "running", None)
    _episode(tmp_path, "Rotfuchs", 4, "completed", None)
    _episode(tmp_path, "Rotfuchs", 5, "completed", "2026-01-05T12:00:00Z")  # not published
    _episode(tmp_path, "Wasser", 3, "published", "2026-01-03T12:00:00Z")
    assert last_published(tmp_path, "Rotfuchs") is None


def test_last_published_takes_latest(tmp_path):
    _episode(tmp_path, "Rotfuchs", 1, "published", "2026-01-01T12:00:00Z")
    _episode(tmp_path, "Rotfuchs", 9, "published", "2026-01-09T12:00:00Z")
    _episode(tmp_path, "Rotfuchs", 5, "failed", None)
    assert last_published(tmp_path, "Rotfuchs") == datetime(2026, 1, 9, 12, tzinfo=UTC)


def test_record_type_matches_fields():
    assert EpisodeRecord("a", "b", "running", "c").published_at is None


def test_episode_file_without_finished_at_and_usage_still_loads(tmp_path):
    record = create_episode(tmp_path, "Rotfuchs", NOW)
    path = episode_dir(tmp_path, record.episode_id) / "episode.json"
    data = json.loads(path.read_text())
    del data["finished_at"], data["usage"]
    path.write_text(json.dumps(data))
    loaded = load_episode(tmp_path, record.episode_id)
    assert loaded.finished_at is None and loaded.usage == {}
