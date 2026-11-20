import json
import shutil
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile

from gut_zu_wissen.archive import (
    AUDIO_FILE,
    PUBLICATION_FILE,
    EpisodeRecord,
    create_episode,
    episode_dir,
    last_published,
    load_episode,
    save_episode,
    write_json,
)
from gut_zu_wissen.internet_archive import HostingError
from gut_zu_wissen.publish import PublishRefused, publish, publish_refusal

CONFIG = tomllib.loads((Path(__file__).parents[1] / "config" / "pipeline.toml").read_text("utf-8"))[
    "hosting"
]
NOW = datetime(2026, 11, 22, 9, 0, tzinfo=UTC)
TITLE = "Schlacht bei Bouvines"

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


class FakeHost:
    """Records uploads; lists the files with their sizes after `listed_after` file-list calls."""

    def __init__(self, exists=False, listed_after=0):
        self.exists = exists
        self.uploads: list[tuple[str, str, bytes, bool]] = []
        self.listed_after = listed_after
        self.sizes: dict[str, int] = {}

    def item_exists(self, identifier):
        return self.exists

    def put_file(self, identifier, name, path, content_type, metadata):
        self.uploads.append((identifier, name, path.read_bytes(), metadata is not None))
        self.sizes[name] = len(path.read_bytes())

    def file_sizes(self, identifier):
        if self.listed_after > 0:
            self.listed_after -= 1
            return {}
        return dict(self.sizes)


@pytest.fixture
def setup(tmp_path):
    archive, audio = tmp_path / "archive", tmp_path / "audio"
    record = create_episode(archive, TITLE, datetime(2026, 11, 21, tzinfo=UTC))
    record.status, record.explicit = "completed", False
    save_episode(archive, record)
    write_json(episode_dir(archive, record.episode_id) / AUDIO_FILE, {"seconds": 2.0})
    rate = 24_000
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate)
    (audio / record.episode_id).mkdir(parents=True)
    soundfile.write(audio / record.episode_id / "episode.flac", tone, rate, format="FLAC")
    return archive, audio, record.episode_id, tmp_path / "work"


def run(setup, host, **kwargs):
    archive, audio, episode_id, work = setup
    return publish(
        host, archive, episode_id, audio, work, CONFIG, NOW, sleep=lambda s: None, **kwargs
    )


def test_publish_uploads_mp3_then_cover_and_marks_the_episode_published(setup):
    host = FakeHost()
    publication = run(setup, host)
    archive, _, episode_id, _ = setup

    assert [(name, created) for _, name, _, created in host.uploads] == [
        ("episode.mp3", True),  # the first PUT creates the item with its metadata
        ("cover.png", False),
    ]
    assert host.uploads[0][2][:3] == b"ID3"
    assert host.uploads[1][2][:8] == b"\x89PNG\r\n\x1a\n"
    assert publication.identifier == f"gut-zu-wissen_{episode_id}"
    assert publication.audio.url == (
        f"https://archive.org/download/gut-zu-wissen_{episode_id}/episode.mp3"
    )
    assert publication.audio.bytes == len(host.uploads[0][2])
    assert publication.seconds == 2.0

    record = load_episode(archive, episode_id)
    assert record.status == "published"
    assert record.published_at == "2026-11-22T09:00:00Z"
    assert last_published(archive, TITLE) == NOW  # arms the D4 repeat gate

    stored = json.loads((episode_dir(archive, episode_id) / PUBLICATION_FILE).read_text("utf-8"))
    assert list(stored)[:3] == ["identifier", "item_url", "published_at"]
    assert stored["audio"]["mime"] == "audio/mpeg"


def test_rerun_on_an_existing_item_sends_no_metadata(setup):
    host = FakeHost(exists=True)
    run(setup, host)
    assert [created for *_, created in host.uploads] == [False, False]


def test_waits_until_the_item_lists_the_files(setup):
    host = FakeHost(listed_after=2)
    run(setup, host)
    assert host.listed_after == 0
    assert load_episode(setup[0], setup[2]).status == "published"


def test_files_that_never_show_up_leave_the_episode_completed(setup):
    host = FakeHost(listed_after=10**6)
    with pytest.raises(HostingError):
        publish(
            host,
            setup[0],
            setup[2],
            setup[1],
            setup[3],
            CONFIG | {"ready_timeout_seconds": 0},
            NOW,
            sleep=lambda s: None,
        )
    assert load_episode(setup[0], setup[2]).status == "completed"
    assert not (episode_dir(setup[0], setup[2]) / PUBLICATION_FILE).exists()


@pytest.mark.parametrize("status", ["running", "failed", "published"])
def test_only_a_completed_episode_is_published(setup, status):
    record = load_episode(setup[0], setup[2])
    record.status = status
    save_episode(setup[0], record)
    host = FakeHost()
    with pytest.raises(PublishRefused):
        run(setup, host)
    assert host.uploads == []


def test_missing_master_is_refused(setup):
    (setup[1] / setup[2] / "episode.flac").unlink()
    with pytest.raises(PublishRefused, match="not found"):
        run(setup, FakeHost())


def test_missing_explicit_flag_is_refused():
    record = EpisodeRecord("id", "T", "completed", "2026-11-21T00:00:00Z")
    assert "explicit" in publish_refusal(record)
    record.explicit = True
    assert publish_refusal(record) is None
