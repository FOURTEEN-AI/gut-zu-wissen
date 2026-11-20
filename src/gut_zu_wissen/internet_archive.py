"""Internet Archive behind an interface (phase 2: audio hosting; design: "Anbieter").

`HttpArchiveHost` talks to the Archive's S3-like API (upload) and its metadata API (does the item
exist, which files does it list). Tests use fakes and `httpx.MockTransport`; CI never calls the
live service. The keys come from the environment and appear in no log line or error message.

One item per episode. The first PUT to a new item creates it and carries the item metadata
(`x-archive-meta*` headers are only honoured at creation); later PUTs add files. Derivative jobs
are switched off (`x-archive-queue-derive: 0`): the MP3 is encoded by the pipeline.
"""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import httpx

Metadata = Sequence[tuple[str, str]]
_UNSAFE = re.compile(r"[^a-z0-9_.-]+")


class HostingError(Exception):
    """The host answered with an error, or a file did not show up."""


# --- pure functions --------------------------------------------------------


def item_identifier(episode_id: str, prefix: str, max_chars: int) -> str:
    """`<prefix>_<episode id>`; too long ones are cut and get a short hash of the full name."""
    name = _UNSAFE.sub("-", f"{prefix}_{episode_id}".lower()).strip("-.")
    if len(name) <= max_chars:
        return name
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    return f"{name[: max_chars - 9].rstrip('-._')}-{digest}"


def item_metadata(title: str, published: str, config: Mapping) -> list[tuple[str, str]]:
    """Item metadata in header order; `published` is a date (YYYY-MM-DD)."""
    pairs = [
        ("mediatype", config["mediatype"]),
        ("collection", config["collection"]),
        ("title", f"Gut zu wissen: {title}"),
        ("creator", config["creator"]),
        ("description", config["description"].format(title=title)),
        ("language", config["language"]),
        ("licenseurl", config["licence_url"]),
        ("date", published),
    ]
    return pairs + [("subject", subject) for subject in config["subjects"]]


def _header_value(value: str) -> str:
    """Non-ASCII values travel as `uri(<percent-encoded>)`; plain ones unchanged."""
    return value if value.isascii() and value.isprintable() else f"uri({quote(value, safe='')})"


def metadata_headers(pairs: Metadata) -> dict[str, str]:
    """`x-archive-meta-<key>`; a key that repeats becomes `x-archive-meta01-<key>`, `02`, ..."""
    counts: dict[str, int] = {}
    for key, _ in pairs:
        counts[key] = counts.get(key, 0) + 1
    seen: dict[str, int] = {}
    headers: dict[str, str] = {}
    for key, value in pairs:
        seen[key] = seen.get(key, 0) + 1
        name = (
            f"x-archive-meta-{key}" if counts[key] == 1 else f"x-archive-meta{seen[key]:02d}-{key}"
        )
        headers[name] = _header_value(value)
    return headers


def download_url(download_base: str, identifier: str, name: str) -> str:
    return f"{download_base.rstrip('/')}/{identifier}/{quote(name)}"


def files_ready(listed: Mapping[str, int], expected: Mapping[str, int]) -> bool:
    return all(listed.get(name) == size for name, size in expected.items())


# --- interface -------------------------------------------------------------


class ArchiveHost(Protocol):
    def item_exists(self, identifier: str) -> bool: ...

    def put_file(
        self,
        identifier: str,
        name: str,
        path: Path,
        content_type: str,
        metadata: Metadata | None,
    ) -> None:
        """Upload a file. `metadata` is given when the item does not exist yet and creates it."""
        ...

    def file_sizes(self, identifier: str) -> dict[str, int]:
        """Name -> size in bytes of the files the item lists."""
        ...


def wait_until_ready(
    host: ArchiveHost,
    identifier: str,
    expected: Mapping[str, int],
    timeout_seconds: float,
    poll_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Poll the item's file list until every expected file is there with its size."""
    deadline = clock() + timeout_seconds
    while True:
        if files_ready(host.file_sizes(identifier), expected):
            return
        if clock() >= deadline:
            raise HostingError(f"files of {identifier} not listed after {timeout_seconds:g} s")
        sleep(poll_seconds)


# --- HTTP ------------------------------------------------------------------


class HttpArchiveHost:
    def __init__(
        self,
        access_key: str,
        secret_key: str,
        config: Mapping,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._s3 = config["s3_endpoint"].rstrip("/")
        self._metadata = config["metadata_endpoint"].rstrip("/")
        self._retries = config["max_retries"]
        self._backoff = config["backoff_seconds"]
        self._client = client or httpx.Client(timeout=config["timeout_seconds"])
        self._sleep = sleep
        self._auth = {"authorization": f"LOW {access_key}:{secret_key}"}

    def item_exists(self, identifier: str) -> bool:
        return bool(self._get_metadata(identifier))

    def file_sizes(self, identifier: str) -> dict[str, int]:
        files = self._get_metadata(identifier).get("files", [])
        return {f["name"]: int(f["size"]) for f in files if str(f.get("size", "")).isdigit()}

    def put_file(
        self,
        identifier: str,
        name: str,
        path: Path,
        content_type: str,
        metadata: Metadata | None,
    ) -> None:
        data = path.read_bytes()
        headers = {
            **self._auth,
            "content-type": content_type,
            "x-archive-queue-derive": "0",
            "x-archive-size-hint": str(len(data)),
        }
        if metadata is not None:
            headers["x-amz-auto-make-bucket"] = "1"
            headers.update(metadata_headers(metadata))
        self._send("PUT", f"{self._s3}/{identifier}/{quote(name)}", content=data, headers=headers)

    def _get_metadata(self, identifier: str) -> dict:
        response = self._send("GET", f"{self._metadata}/{identifier}")
        try:
            data = response.json()
        except ValueError as error:
            raise HostingError("metadata answer is not JSON") from error
        return data if isinstance(data, dict) else {}

    def _send(self, method: str, url: str, **kwargs) -> httpx.Response:
        for attempt in range(self._retries + 1):
            try:
                response = self._client.request(method, url, **kwargs)
            except httpx.TransportError as error:  # timeout, connection reset
                if attempt == self._retries:
                    raise HostingError(f"{method} {url}: {type(error).__name__}") from error
            else:
                status = response.status_code
                if status < 400:
                    return response
                transient = status == 429 or status >= 500
                if not transient or attempt == self._retries:
                    raise HostingError(f"{method} {url} answered {status}: {response.text[:300]}")
            self._sleep(self._backoff * 2**attempt)
        raise AssertionError("unreachable")
