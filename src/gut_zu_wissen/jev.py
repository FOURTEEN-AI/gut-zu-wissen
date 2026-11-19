"""Jev client (TypeSafe) behind an interface, with the error table of spike S1.

`Jev.ask` sends one state and several Noul questions and returns p per question key. Reaction to
errors (design: "API-Fehler"):

- overlength (400, `max_tokens_exceeded`) -> `JevOverlengthError`: the LLM judge decides;
- transient (429, 500, 529, timeout) -> retried with backoff, then `JevTransientError`: LLM judge;
- configuration and invalid request (401, 402, 403, 404, other 400, 413, 415, 422)
  -> `JevAbortError`: abort and escalate.

The API key comes from the environment (`TYPESAFE_API_KEY`). The token counts of every successful
answer (`usage` in the response) are added to a `Usage` counter when given (cost log); error
answers are not counted.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol

import httpx

from gut_zu_wissen.usage import Usage

API_URL = "https://api.typesafe.ai/v1/systemone"
TRANSIENT_STATUS = {429, 500, 502, 503, 504, 529}


class JevError(Exception):
    """Base class of all Jev failures."""


class JevOverlengthError(JevError):
    """State plus question exceed the token limits; the LLM judge takes over."""


class JevTransientError(JevError):
    """Retries are used up; the LLM judge takes over."""


class JevAbortError(JevError):
    """Configuration or request error; the run aborts."""


class Jev(Protocol):
    def ask(self, state: dict, questions: dict[str, dict]) -> dict[str, float]:
        """p per question key (Noul questions only)."""
        ...


def error_type(body: object) -> str | None:
    """`detail.error_type` of an error body, if present."""
    detail = body.get("detail") if isinstance(body, dict) else None
    value = detail.get("error_type") if isinstance(detail, dict) else None
    return value if isinstance(value, str) else None


def classify(status: int, body: object) -> type[JevError] | None:
    """Error class for a response, `None` for success."""
    if status == 200:
        return None
    if status == 400 and error_type(body) == "max_tokens_exceeded":
        return JevOverlengthError
    if status in TRANSIENT_STATUS:
        return JevTransientError
    return JevAbortError


class HttpJev:
    def __init__(
        self,
        model: str,
        max_retries: int,
        backoff_seconds: float,
        timeout_seconds: float,
        api_key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        usage: Usage | None = None,
    ) -> None:
        self._model = model
        self._usage = usage
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._sleep = sleep
        self._client = client or httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout_seconds
        )

    def ask(self, state: dict, questions: dict[str, dict]) -> dict[str, float]:
        payload = {"model": self._model, "state": state, "questions": questions}
        for attempt in range(self._max_retries + 1):
            retry_after: float | None = None
            try:
                response = self._client.post(API_URL, json=payload)
                status, body = response.status_code, _json(response)
                retry_after = _retry_after(response)
            except httpx.TransportError:  # timeout, connection reset
                status, body = 599, None
            error = classify(status, body) if status != 599 else JevTransientError
            if error is None:
                if self._usage is not None:
                    self._usage.add(*_tokens(body))
                return _probabilities(body, questions)
            if error is JevTransientError and attempt < self._max_retries:
                # The `retry-after` header, when present, is the server's own wish.
                self._sleep(retry_after if retry_after is not None else self._backoff * 2**attempt)
                continue
            raise error(f"Jev answered {status}: {body}")
        raise AssertionError("unreachable")


def _retry_after(response: httpx.Response) -> float | None:
    try:
        value = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return value if value >= 0 else None


def _json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return None


def _tokens(body: object) -> tuple[int, int]:
    """`usage.input_tokens` and `usage.output_tokens` of a 200 answer (OpenAPI: both required);
    a missing count is 0, so a malformed answer still counts as a call."""
    usage = body.get("usage") if isinstance(body, dict) else None
    usage = usage if isinstance(usage, dict) else {}

    def count(key: str) -> int:
        value = usage.get(key)
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    return count("input_tokens"), count("output_tokens")


def _probabilities(body: object, questions: dict[str, dict]) -> dict[str, float]:
    answers = body.get("answers") if isinstance(body, dict) else None
    if not isinstance(answers, dict):
        raise JevAbortError("Jev answer has no 'answers' object.")
    result: dict[str, float] = {}
    for key in questions:
        p = answers.get(key, {}).get("noul") if isinstance(answers.get(key), dict) else None
        if not isinstance(p, int | float):
            raise JevAbortError(f"Jev answer has no probability for question '{key}'.")
        result[key] = float(p)
    return result
