"""Fetch an article from the German Wikipedia (phase 1, step 2).

Revision rule (D2): latest sighted revision; latest revision if none exists. The status is
recorded. The Wikipedia API sits behind the `WikipediaApi` interface; tests use fakes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal, Protocol

import httpx

API_URL = "https://de.wikipedia.org/w/api.php"
USER_AGENT = (
    "GutZuWissen/0.1 (https://github.com/FOURTEEN-AI/gut-zu-wissen; gut-zu-wissen@fourteen-it.de)"
)
MAX_LAG_SECONDS = "5"
MIN_REQUEST_INTERVAL_SECONDS = 1.0

RevisionStatus = Literal["sighted", "latest"]


class ArticleNotFoundError(Exception):
    """The title does not exist on the Wikipedia."""


class WikipediaApi(Protocol):
    def query(self, **params: str) -> dict: ...

    def post(self, **params: str) -> dict:
        """Same as `query`, but as POST: for requests with long text (`action=parse`)."""
        ...


class HttpWikipediaApi:
    """MediaWiki API client: descriptive User-Agent, maxlag, one request per interval."""

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        min_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30)
        self._min_interval = min_interval
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request: float | None = None

    def query(self, **params: str) -> dict:
        return self._request("GET", params)

    def post(self, **params: str) -> dict:
        return self._request("POST", params)

    def _request(self, method: str, params: dict[str, str]) -> dict:
        if self._last_request is not None:
            wait = self._min_interval - (self._monotonic() - self._last_request)
            if wait > 0:
                self._sleep(wait)
        full = {"format": "json", "formatversion": "2", "maxlag": MAX_LAG_SECONDS, **params}
        if method == "POST":
            response = self._client.post(API_URL, data=full)
        else:
            response = self._client.get(API_URL, params=full)
        self._last_request = self._monotonic()
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"Wikipedia API error: {data['error']}")
        return data


@dataclass(frozen=True)
class Article:
    """Fetched article; field order is the archive field order."""

    title: str
    revid: int
    revision_status: RevisionStatus
    revision_timestamp: str
    fetched_at: str
    categories: list[str]
    wikitext: str

    def to_dict(self) -> dict:
        return asdict(self)


def select_revision(page: Mapping) -> tuple[int, RevisionStatus]:
    """D2: stable (sighted) revision if present, otherwise the latest revision."""
    flagged = page.get("flagged")
    if flagged and "stable_revid" in flagged:
        return int(flagged["stable_revid"]), "sighted"
    return int(page["lastrevid"]), "latest"


def fetch_article(api: WikipediaApi, title: str, now: datetime | None = None) -> Article:
    info = api.query(action="query", prop="info|flagged", titles=title, redirects="1")
    page = info["query"]["pages"][0]
    if page.get("missing"):
        raise ArticleNotFoundError(title)
    revid, status = select_revision(page)
    data = api.query(
        action="query",
        prop="revisions|categories",
        revids=str(revid),
        rvprop="ids|timestamp|content",
        rvslots="main",
        cllimit="max",
    )
    rpage = data["query"]["pages"][0]
    rev = rpage["revisions"][0]
    fetched = (now or datetime.now(UTC)).astimezone(UTC)
    return Article(
        title=rpage["title"],
        revid=rev["revid"],
        revision_status=status,
        revision_timestamp=rev["timestamp"],
        fetched_at=fetched.strftime("%Y-%m-%dT%H:%M:%SZ"),
        categories=[c["title"] for c in rpage.get("categories", [])],
        wikitext=rev["slots"]["main"]["content"],
    )
