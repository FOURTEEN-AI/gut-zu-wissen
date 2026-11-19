import json
from pathlib import Path

import httpx
import pytest

from gut_zu_wissen import audio
from gut_zu_wissen.run import load_config
from gut_zu_wissen.tts import (
    API_URL,
    GeminiTts,
    SectionError,
    TtsError,
    build_request,
    parse_response,
    retry_seconds,
    split_sections,
)

FIXTURE = Path(__file__).parent / "fixtures" / "gemini" / "interaction_completed.json"
VOICES = {"Lena": "de-de-advisor-3", "Felix": "de-de-tutor-4"}


def completed() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- sections


def sizes(lengths, sections):
    return [sum(lengths[i] for i in s) for s in sections]


def test_short_script_is_one_section():
    assert split_sections([100, 200, 300], 6000) == [[0, 1, 2]]


def test_typical_episode_gives_three_balanced_sections():
    # About 16,000 characters in turns of 100 to 600 characters (spike S4: 3 sections).
    lengths = [100 + (n * 137) % 500 for n in range(48)]
    assert 15_500 < sum(lengths) < 16_500
    sections = split_sections(lengths, 6000)
    assert len(sections) == 3
    assert [i for s in sections for i in s] == list(range(len(lengths)))  # order, no gaps
    assert max(sizes(lengths, sections)) <= 6000
    assert max(sizes(lengths, sections)) - min(sizes(lengths, sections)) <= 600


def test_longest_section_is_as_short_as_possible():
    # Greedy filling would give [5000, 1000] + ...; the balanced split is 3000 / 3000.
    lengths = [1000] * 6
    assert sizes(lengths, split_sections(lengths, 5000)) == [3000, 3000]


def test_more_sections_when_turn_boundaries_force_them():
    # Total 9000 fits two sections by count, but no cut at a turn boundary gives two <= 5000.
    lengths = [3000, 3000, 3000]
    assert split_sections(lengths, 5000) == [[0], [1], [2]]


def test_limit_is_inclusive():
    assert split_sections([3000, 3000], 6000) == [[0, 1]]


def test_turn_longer_than_a_section_is_an_error():
    with pytest.raises(SectionError, match="Turn 1 has 6001 characters"):
        split_sections([10, 6001], 6000)


def test_empty_script_is_an_error():
    with pytest.raises(SectionError):
        split_sections([], 6000)


def test_real_config_section_size():
    assert load_config()["quotas"]["tts_section_max_chars"] == 6000


# ---------------------------------------------------------------- request and response


def test_request_annotates_each_turn_with_its_speaker():
    body = build_request("gemini-3.8-flash-tts", VOICES, [("Felix", "Hallo!"), ("Lena", "Hi.")])
    assert body["model"] == "gemini-3.8-flash-tts"
    content = body["input"][0]["content"]
    assert [c["text"] for c in content] == ["Hallo!", "Hi."]
    assert [c["annotations"][0]["speaker"] for c in content] == ["Felix", "Lena"]
    assert body["generation_config"]["speech_config"]["speakers"] == [
        {"speaker": "Lena", "voice": "de-de-advisor-3"},
        {"speaker": "Felix", "voice": "de-de-tutor-4"},
    ]
    assert body["response_format"] == {"type": "audio"}


def test_recorded_response_gives_audio_and_usage():
    result = parse_response(completed())
    assert (result.input_tokens, result.output_tokens) == (3181, 18034)
    y, sr = audio.decode(result.audio)
    assert sr == 24_000 and len(y) == 24_000


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.update(status="failed"),
        lambda b: b.update(steps=[]),
        lambda b: b["steps"][0]["content"][0].update(data="###"),
    ],
)
def test_response_without_usable_audio_is_an_error(change):
    body = completed()
    change(body)
    with pytest.raises(TtsError):
        parse_response(body)


def test_missing_usage_is_recorded_as_unknown():
    body = completed()
    del body["usage"]
    assert parse_response(body).input_tokens is None


def test_retry_time_is_read_from_a_429_body():
    text = '{"error": {"code": "too_many_requests", "message": "Quota. Please retry in 25s."}}'
    assert retry_seconds(text) == 25.0
    assert retry_seconds("Please retry in 1.5s") == 1.5
    assert retry_seconds("no hint") is None


# ---------------------------------------------------------------- client


def client(responses: list, seen: list | None = None) -> tuple[GeminiTts, list[float]]:
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    http = httpx.Client(transport=httpx.MockTransport(handler), headers={"x-goog-api-key": "k"})
    tts = GeminiTts("gemini-3.8-flash-tts", VOICES, 3, 2.0, 10, "k", http, slept.append)
    return tts, slept


def test_client_posts_the_interaction():
    seen: list[httpx.Request] = []
    tts, _ = client([httpx.Response(200, json=completed())], seen)
    result = tts.synthesize([("Lena", "Hallo.")])
    assert result.output_tokens == 18034
    assert str(seen[0].url) == API_URL
    assert json.loads(seen[0].content)["input"][0]["content"][0]["text"] == "Hallo."


def test_429_waits_the_named_time_then_succeeds():
    busy = httpx.Response(429, json={"error": {"message": "Please retry in 25s."}})
    tts, slept = client([busy, httpx.Response(200, json=completed())])
    tts.synthesize([("Lena", "Hallo.")])
    assert slept == [25.0]


def test_transient_errors_back_off_then_give_up():
    timeout = httpx.ReadTimeout("slow")
    tts, slept = client([httpx.Response(503, text="busy"), timeout, httpx.Response(500), timeout])
    with pytest.raises(TtsError, match="599"):
        tts.synthesize([("Lena", "Hallo.")])
    assert slept == [2.0, 4.0, 8.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_request_and_configuration_errors_are_not_retried(status):
    tts, slept = client([httpx.Response(status, text="nope")])
    with pytest.raises(TtsError, match=str(status)):
        tts.synthesize([("Lena", "Hallo.")])
    assert slept == []
