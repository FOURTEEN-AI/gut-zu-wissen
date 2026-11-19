import httpx
import pytest

from gut_zu_wissen.jev import (
    HttpJev,
    JevAbortError,
    JevOverlengthError,
    JevTransientError,
    classify,
)

Q = {"a": {"type": "noul"}, "b": {"type": "noul"}}
OK = {"answers": {"a": {"type": "noul", "noul": 0.97}, "b": {"type": "noul", "noul": 0.2}}}


def jev(responses: list, sleeps: list | None = None, retries: int = 2) -> tuple[HttpJev, list]:
    sent: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return httpx.Response(item[0], json=item[1])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep = sleeps.append if sleeps is not None else (lambda s: None)
    return HttpJev("jev-1.14.0", retries, 2.0, 10, "key", client, sleep), sent


def test_success_returns_p_per_question_and_sends_pinned_model():
    client, sent = jev([(200, OK)])
    assert client.ask({"x": 1}, Q) == {"a": 0.97, "b": 0.2}
    assert b'"model":"jev-1.14.0"' in sent[0].content


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (400, {"detail": {"error_type": "max_tokens_exceeded"}}, JevOverlengthError),
        (429, {}, JevTransientError),
        (500, {}, JevTransientError),
        (529, {}, JevTransientError),
        (400, {"detail": {"error_type": "api_usage_error"}}, JevAbortError),
        (400, {"detail": "Noul question must have criteria"}, JevAbortError),
        (401, {}, JevAbortError),
        (402, {}, JevAbortError),
        (403, {}, JevAbortError),
        (404, {}, JevAbortError),
        (413, {}, JevAbortError),
        (415, {}, JevAbortError),
        (422, {}, JevAbortError),
    ],
)
def test_error_table(status, body, expected):
    assert classify(status, body) is expected


def test_success_is_no_error():
    assert classify(200, OK) is None


def test_overlength_is_not_retried():
    client, sent = jev([(400, {"detail": {"error_type": "max_tokens_exceeded"}})])
    with pytest.raises(JevOverlengthError):
        client.ask({}, Q)
    assert len(sent) == 1


def test_transient_errors_are_retried_with_backoff_then_succeed():
    sleeps: list = []
    client, sent = jev([(429, {}), (529, {}), (200, OK)], sleeps)
    assert client.ask({}, Q)["a"] == 0.97
    assert len(sent) == 3 and sleeps == [2.0, 4.0]


def test_transient_errors_exhaust_the_retry_budget():
    client, sent = jev([(500, {})] * 3)
    with pytest.raises(JevTransientError):
        client.ask({}, Q)
    assert len(sent) == 3


def test_timeout_counts_as_transient():
    client, _ = jev([httpx.ReadTimeout("slow")] * 3)
    with pytest.raises(JevTransientError):
        client.ask({}, Q)


def test_abort_errors_are_not_retried():
    client, sent = jev([(401, {"detail": {"error_type": "authentication_error"}})])
    with pytest.raises(JevAbortError):
        client.ask({}, Q)
    assert len(sent) == 1


def test_missing_probability_aborts():
    client, _ = jev([(200, {"answers": {"a": {"type": "noul", "noul": 0.9}}})])
    with pytest.raises(JevAbortError):
        client.ask({}, Q)
