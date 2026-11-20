import tomllib
from pathlib import Path

import httpx
import pytest

from gut_zu_wissen.internet_archive import (
    HostingError,
    HttpArchiveHost,
    download_url,
    files_ready,
    item_identifier,
    item_metadata,
    metadata_headers,
    wait_until_ready,
)

CONFIG = tomllib.loads((Path(__file__).parents[1] / "config" / "pipeline.toml").read_text("utf-8"))[
    "hosting"
] | {"s3_endpoint": "https://s3.example", "metadata_endpoint": "https://meta.example"}


# --- pure functions ---------------------------------------------------------


def test_identifier_is_prefix_and_episode_id():
    assert item_identifier("2026-11-22_schlacht-bei-bouvines", "gut-zu-wissen", 100) == (
        "gut-zu-wissen_2026-11-22_schlacht-bei-bouvines"
    )


def test_long_identifier_is_cut_with_a_hash_and_stays_unique():
    first = item_identifier("2026-11-22_" + "a" * 120 + "-eins", "gut-zu-wissen", 100)
    second = item_identifier("2026-11-22_" + "a" * 120 + "-zwei", "gut-zu-wissen", 100)
    assert len(first) <= 100 and len(second) <= 100
    assert first != second
    assert all(c.isalnum() or c in "_.-" for c in first)


def test_metadata_carries_licence_collection_and_title():
    pairs = dict(item_metadata("Ei", "2026-11-22", CONFIG))
    assert pairs["mediatype"] == "audio"
    assert pairs["collection"] == "opensource_audio"
    assert pairs["licenseurl"].endswith("/by-sa/4.0/")
    assert pairs["title"] == "Gut zu wissen: Ei"
    assert "„Ei“" in pairs["description"]


def test_repeated_keys_are_numbered_and_non_ascii_is_encoded():
    headers = metadata_headers([("title", "Über"), ("subject", "a"), ("subject", "b")])
    assert headers == {
        "x-archive-meta-title": "uri(%C3%9Cber)",
        "x-archive-meta01-subject": "a",
        "x-archive-meta02-subject": "b",
    }


def test_download_url_quotes_the_file_name():
    assert download_url("https://a/download/", "id", "x y.mp3") == "https://a/download/id/x%20y.mp3"


def test_files_ready_needs_every_name_and_size():
    assert files_ready({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 2})
    assert not files_ready({"a": 1}, {"a": 1, "b": 2})
    assert not files_ready({"a": 9}, {"a": 1})


# --- wait_until_ready -------------------------------------------------------


class FakeHost:
    def __init__(self, listings):
        self.listings = list(listings)

    def file_sizes(self, identifier):
        return self.listings.pop(0) if len(self.listings) > 1 else self.listings[0]


def test_wait_polls_until_files_are_listed():
    host = FakeHost([{}, {"a": 1}, {"a": 1, "b": 2}])
    waits: list[float] = []
    wait_until_ready(host, "id", {"a": 1, "b": 2}, 100, 5, sleep=waits.append, clock=lambda: 0)
    assert waits == [5, 5]


def test_wait_gives_up_at_the_deadline():
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    with pytest.raises(HostingError):
        wait_until_ready(FakeHost([{}]), "id", {"a": 1}, 20, 5, sleep=sleep, clock=lambda: now[0])


# --- HTTP -------------------------------------------------------------------


def host(handler, sleeps=None):
    sent: list[httpx.Request] = []

    def record(request):
        sent.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record))
    sleeps = [] if sleeps is None else sleeps
    return HttpArchiveHost("ACCESS-123", "SECRET-456", CONFIG, client, sleeps.append), sent


def test_put_creates_item_with_metadata_and_keys(tmp_path):
    api, sent = host(lambda r: httpx.Response(200))
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"abc")

    api.put_file("id", "episode.mp3", audio, "audio/mpeg", [("title", "T"), ("language", "ger")])

    request = sent[0]
    assert request.method == "PUT"
    assert str(request.url) == "https://s3.example/id/episode.mp3"
    assert request.content == b"abc"
    assert request.headers["authorization"] == "LOW ACCESS-123:SECRET-456"
    assert request.headers["x-amz-auto-make-bucket"] == "1"
    assert request.headers["x-archive-meta-title"] == "T"
    assert request.headers["x-archive-queue-derive"] == "0"
    assert request.headers["x-archive-size-hint"] == "3"
    assert request.headers["content-type"] == "audio/mpeg"


def test_put_to_an_existing_item_sends_no_metadata(tmp_path):
    api, sent = host(lambda r: httpx.Response(200))
    cover = tmp_path / "cover.png"
    cover.write_bytes(b"png")
    api.put_file("id", "cover.png", cover, "image/png", None)
    assert not any(name.startswith("x-archive-meta") for name in sent[0].headers)
    assert "x-amz-auto-make-bucket" not in sent[0].headers


def test_item_exists_reads_the_metadata_answer():
    assert host(lambda r: httpx.Response(200, json={}))[0].item_exists("id") is False
    assert host(lambda r: httpx.Response(200, json={"created": 1}))[0].item_exists("id") is True


def test_file_sizes_skips_entries_without_a_size():
    files = [{"name": "a.mp3", "size": "10"}, {"name": "a_meta.xml"}]
    api, _ = host(lambda r: httpx.Response(200, json={"files": files}))
    assert api.file_sizes("id") == {"a.mp3": 10}


def test_transient_errors_are_retried_with_backoff(tmp_path):
    answers = [httpx.Response(503), httpx.Response(429), httpx.Response(200)]
    sleeps: list[float] = []
    api, sent = host(lambda r: answers.pop(0), sleeps)
    f = tmp_path / "x"
    f.write_bytes(b"1")
    api.put_file("id", "x", f, "text/plain", None)
    assert len(sent) == 3
    assert sleeps == [2.0, 4.0]


def test_retries_are_limited_and_the_error_hides_the_keys(tmp_path):
    api, sent = host(lambda r: httpx.Response(503, text="slow down"))
    f = tmp_path / "x"
    f.write_bytes(b"1")
    with pytest.raises(HostingError) as error:
        api.put_file("id", "x", f, "text/plain", None)
    assert len(sent) == CONFIG["max_retries"] + 1
    assert "ACCESS-123" not in str(error.value)
    assert "SECRET-456" not in str(error.value)


def test_client_errors_are_not_retried(tmp_path):
    api, sent = host(lambda r: httpx.Response(403, text="denied"))
    f = tmp_path / "x"
    f.write_bytes(b"1")
    with pytest.raises(HostingError, match="403"):
        api.put_file("id", "x", f, "text/plain", None)
    assert len(sent) == 1


def test_a_timeout_is_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={})

    api, _ = host(handler)
    assert api.item_exists("id") is False
    assert len(calls) == 2
