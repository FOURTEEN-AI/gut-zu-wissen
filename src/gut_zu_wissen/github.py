"""GitHub issues behind an interface (review loop, phase 1 step 12).

`HttpIssues` talks to the REST API with the workflow's `GITHUB_TOKEN`; tests use fakes and
`httpx.MockTransport`, CI never calls the live API. Only what the review loop needs: open an
issue, comment on it, list its comments, close it once its review is answered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import httpx

API_VERSION = "2022-11-28"
USER_AGENT = "GutZuWissen/0.1 (https://github.com/FOURTEEN-AI/gut-zu-wissen)"
PAGE_SIZE = 100  # GitHub's maximum per page
TIMEOUT_SECONDS = 30
CLOSED_STATE_REASON = "completed"  # GitHub's reason for an issue closed as done


class GitHubError(Exception):
    """The GitHub API answered with an error."""


@dataclass(frozen=True)
class Comment:
    id: int
    login: str
    association: str
    body: str
    created_at: str
    updated_at: str
    url: str


class Issues(Protocol):
    def create_issue(self, title: str, body: str) -> tuple[int, str]:
        """Open an issue; returns its number and web URL."""
        ...

    def create_comment(self, number: int, body: str) -> str:
        """Comment on an issue; returns the comment's web URL."""
        ...

    def list_comments(self, number: int) -> list[Comment]:
        """Every comment of an issue, oldest first."""
        ...

    def close_issue(self, number: int) -> None:
        """Close an issue as completed; comments on it stay possible."""
        ...


def parse_comment(item: dict) -> Comment:
    user = item.get("user") or {}
    return Comment(
        id=int(item["id"]),
        login=str(user.get("login", "")),
        association=str(item.get("author_association", "")),
        body=str(item.get("body") or ""),
        created_at=str(item.get("created_at", "")),
        updated_at=str(item.get("updated_at", "")),
        url=str(item.get("html_url", "")),
    )


class HttpIssues:
    def __init__(
        self,
        repository: str,
        token: str,
        api_url: str = "https://api.github.com",
        client: httpx.Client | None = None,
    ) -> None:
        self._base = f"{api_url.rstrip('/')}/repos/{repository}/issues"
        self._client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": USER_AGENT,
        }

    def create_issue(self, title: str, body: str) -> tuple[int, str]:
        data = self._send("POST", self._base, json={"title": title, "body": body}).json()
        return int(data["number"]), str(data["html_url"])

    def create_comment(self, number: int, body: str) -> str:
        url = f"{self._base}/{number}/comments"
        return str(self._send("POST", url, json={"body": body}).json()["html_url"])

    def list_comments(self, number: int) -> list[Comment]:
        comments: list[Comment] = []
        url: str | None = f"{self._base}/{number}/comments?per_page={PAGE_SIZE}"
        while url:
            response = self._send("GET", url)
            comments += [parse_comment(item) for item in response.json()]
            url = response.links.get("next", {}).get("url")
        return comments

    def close_issue(self, number: int) -> None:
        body = {"state": "closed", "state_reason": CLOSED_STATE_REASON}
        self._send("PATCH", f"{self._base}/{number}", json=body)

    def _send(self, method: str, url: str, json: dict | None = None) -> httpx.Response:
        response = self._client.request(method, url, json=json, headers=self._headers)
        if response.status_code >= 400:
            raise GitHubError(f"GitHub answered {response.status_code}: {response.text[:500]}")
        return response
