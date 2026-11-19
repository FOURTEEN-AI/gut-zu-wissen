import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from gut_zu_wissen.fetch import (
    ArticleNotFoundError,
    HttpWikipediaApi,
    fetch_article,
    select_revision,
)

FIXTURES = Path(__file__).parent / "fixtures" / "wikipedia" / "spikes"
NOW = datetime(2026, 11, 19, 10, 30, tzinfo=UTC)


class FakeApi:
    def __init__(self, info: dict, revisions: dict) -> None:
        self.info, self.revisions, self.calls = info, revisions, []

    def query(self, **params: str) -> dict:
        self.calls.append(params)
        return self.info if params["prop"] == "info|flagged" else self.revisions


def make_api(page_info: dict, revid: int = 42) -> FakeApi:
    return FakeApi(
        {"query": {"pages": [{"title": "Rotfuchs", **page_info}]}},
        {
            "query": {
                "pages": [
                    {
                        "title": "Rotfuchs",
                        "categories": [{"title": "Kategorie:Hunde"}],
                        "revisions": [
                            {
                                "revid": revid,
                                "timestamp": "2026-10-24T13:21:55Z",
                                "slots": {"main": {"content": "Text"}},
                            }
                        ],
                    }
                ]
            }
        },
    )


def test_select_sighted():
    assert select_revision({"lastrevid": 9, "flagged": {"stable_revid": 7}}) == (7, "sighted")


def test_select_fallback_to_latest():
    assert select_revision({"lastrevid": 9}) == (9, "latest")
    assert select_revision({"lastrevid": 9, "flagged": {}}) == (9, "latest")


def test_fetch_uses_sighted_revision():
    api = make_api({"lastrevid": 9, "flagged": {"stable_revid": 7}}, revid=7)
    article = fetch_article(api, "Rotfuchs", NOW)
    assert api.calls[1]["revids"] == "7"
    assert article.revision_status == "sighted"
    assert article.revid == 7
    assert article.fetched_at == "2026-11-19T10:30:00Z"
    assert article.categories == ["Kategorie:Hunde"]
    assert article.wikitext == "Text"


def test_fetch_falls_back_to_latest():
    api = make_api({"lastrevid": 9}, revid=9)
    article = fetch_article(api, "Rotfuchs", NOW)
    assert api.calls[1]["revids"] == "9"
    assert article.revision_status == "latest"


def test_fetch_missing_article():
    api = make_api({"missing": True})
    with pytest.raises(ArticleNotFoundError):
        fetch_article(api, "Rotfuchs", NOW)


def test_archive_dict_field_order_matches_fixture():
    fixture = json.loads((FIXTURES / "Rotfuchs.json").read_text())
    article = fetch_article(make_api({"lastrevid": 9}), "Rotfuchs", NOW)
    assert list(article.to_dict()) == list(fixture)


def test_http_client_sets_user_agent_params_and_rate_limit():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"query": {}})

    client = httpx.Client(
        transport=httpx.MockTransport(handler), headers={"User-Agent": "GutZuWissen/test (x@y.z)"}
    )
    sleeps: list[float] = []
    clock = iter([0.0, 0.2, 0.2, 5.0])  # after req 1; before req 2; after req 2
    api = HttpWikipediaApi(
        client, min_interval=1.0, sleep=sleeps.append, monotonic=lambda: next(clock)
    )
    api.query(action="query")
    api.query(action="query")
    assert seen[0].headers["User-Agent"].startswith("GutZuWissen/")
    assert seen[0].url.params["maxlag"] == "5"
    assert seen[0].url.params["formatversion"] == "2"
    assert sleeps == [pytest.approx(0.8)]


def test_http_client_raises_on_api_error():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"error": {"code": "maxlag"}})
        )
    )
    with pytest.raises(RuntimeError):
        HttpWikipediaApi(client, sleep=lambda s: None).query(action="query")
