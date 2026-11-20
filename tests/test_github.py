import json

import httpx
import pytest

from gut_zu_wissen.github import Comment, GitHubError, HttpIssues

COMMENT = {
    "id": 11,
    "user": {"login": "ajFourteen"},
    "author_association": "MEMBER",
    "body": "3: belegt",
    "created_at": "2026-11-19T10:30:00Z",
    "updated_at": "2026-11-19T10:31:00Z",
    "html_url": "https://github.com/o/r/issues/7#issuecomment-11",
}


def issues(handler) -> tuple[HttpIssues, list[httpx.Request]]:
    sent: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return handler(request)

    client = httpx.Client(transport=httpx.MockTransport(record))
    return HttpIssues("o/r", "secret-token", "https://api.example", client), sent


def test_create_issue_sends_title_body_and_token():
    api, sent = issues(lambda r: httpx.Response(201, json={"number": 7, "html_url": "https://x/7"}))
    assert api.create_issue("Review: A", "Text") == (7, "https://x/7")
    request = sent[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.example/repos/o/r/issues"
    assert json.loads(request.content) == {"title": "Review: A", "body": "Text"}
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"


def test_create_comment():
    api, sent = issues(lambda r: httpx.Response(201, json={"html_url": "https://x/c"}))
    assert api.create_comment(7, "Hallo") == "https://x/c"
    assert str(sent[0].url) == "https://api.example/repos/o/r/issues/7/comments"
    assert json.loads(sent[0].content) == {"body": "Hallo"}


def test_close_issue_as_completed():
    api, sent = issues(lambda r: httpx.Response(200, json={"number": 7, "state": "closed"}))
    api.close_issue(7)
    assert sent[0].method == "PATCH"
    assert str(sent[0].url) == "https://api.example/repos/o/r/issues/7"
    assert json.loads(sent[0].content) == {"state": "closed", "state_reason": "completed"}


def test_list_comments_follows_pagination():
    second = "https://api.example/repos/o/r/issues/7/comments?per_page=100&page=2"

    def handler(request: httpx.Request) -> httpx.Response:
        if "page=2" in str(request.url):
            return httpx.Response(200, json=[{**COMMENT, "id": 12, "body": None}])
        return httpx.Response(200, json=[COMMENT], headers={"Link": f'<{second}>; rel="next"'})

    api, sent = issues(handler)
    comments = api.list_comments(7)
    assert len(sent) == 2
    assert comments[0] == Comment(
        11,
        "ajFourteen",
        "MEMBER",
        "3: belegt",
        "2026-11-19T10:30:00Z",
        "2026-11-19T10:31:00Z",
        "https://github.com/o/r/issues/7#issuecomment-11",
    )
    assert comments[1].id == 12 and comments[1].body == ""


def test_error_status_raises_without_the_token():
    api, _ = issues(lambda r: httpx.Response(403, json={"message": "Resource not accessible"}))
    with pytest.raises(GitHubError, match="403") as info:
        api.create_issue("T", "B")
    assert "secret-token" not in str(info.value)


def test_labels_are_the_names_of_the_issue_labels():
    api, sent = issues(
        lambda request: httpx.Response(200, json={"labels": [{"name": "hold"}, {"name": "x"}]})
    )
    assert api.labels(7) == ["hold", "x"]
    assert sent[0].method == "GET" and str(sent[0].url) == "https://api.example/repos/o/r/issues/7"
