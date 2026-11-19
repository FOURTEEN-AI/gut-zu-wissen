"""Dialog TTS (phase 1, step 9; design: "Audio", "Synthese", "Ratenlimit"; spike S4).

Sections: the script is split at turn boundaries into the fewest sections of at most
`tts_section_max_chars` characters, balanced so the longest section is as short as possible.
Sections exist for cheap resynthesis of one part (step 10); the API does not enforce its
documented limits. Splitting is a pure function.

`Tts.synthesize` turns one section (speaker and text per turn) into audio with Gemini's
two-speaker dialog. The HTTP client reads the wait time from a `429` ("Please retry in 25s") and
waits; other transient errors (5xx, timeout) are retried with backoff. When the retries are used
up, or on any other error, it raises `TtsError`: the run fails and escalates (the design's error
table covers Jev only; there is no fallback TTS). The API key comes from the environment
(`GEMINI_API_KEY`).
"""

from __future__ import annotations

import base64
import binascii
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import httpx

API_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
TRANSIENT_STATUS = {429, 500, 502, 503, 504, 529}
_RETRY_IN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


class TtsError(Exception):
    """Synthesis failed and retrying the same request will not help (or retries are used up)."""


class SectionError(Exception):
    """The script cannot be split into sections within the size limit."""


@dataclass
class Synthesis:
    """Audio of one request (WAV bytes as delivered) and the token usage for the cost log."""

    audio: bytes
    input_tokens: int | None
    output_tokens: int | None


class Tts(Protocol):
    def synthesize(self, turns: Sequence[tuple[str, str]]) -> Synthesis:
        """Audio for the turns, each (speaker, text), in one request."""
        ...


# ---------------------------------------------------------------- pure functions


def split_sections(lengths: Sequence[int], max_chars: int) -> list[list[int]]:
    """Turn indices per section: fewest sections of at most `max_chars`, longest one minimal.

    `lengths` are the characters per turn. Raises `SectionError` for an empty script or a turn
    longer than `max_chars`, which no split at turn boundaries can fit.
    """
    if not lengths:
        raise SectionError("The script has no turns.")
    too_long = [n for n, length in enumerate(lengths) if length > max_chars]
    if too_long:
        raise SectionError(
            f"Turn {too_long[0]} has {lengths[too_long[0]]} characters; "
            f"a section holds at most {max_chars}."
        )
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    total, n = prefix[-1], len(lengths)
    for count in range(max(1, -(-total // max_chars)), n + 1):
        cuts = _balanced_cuts(prefix, count)
        bounds = [0, *cuts, n]
        sizes = [prefix[b] - prefix[a] for a, b in zip(bounds, bounds[1:], strict=False)]
        if max(sizes) <= max_chars:
            return [list(range(a, b)) for a, b in zip(bounds, bounds[1:], strict=False)]
    raise AssertionError("unreachable: one turn per section always fits")


def _balanced_cuts(prefix: Sequence[int], count: int) -> list[int]:
    """Cut positions splitting the turns into `count` non-empty parts with the smallest
    possible largest part (dynamic programme on prefix sums; earliest cut wins ties)."""
    n = len(prefix) - 1
    inf = float("inf")
    # best[j][i]: smallest largest part when the first i turns form j parts.
    best = [[inf] * (n + 1) for _ in range(count + 1)]
    cut = [[0] * (n + 1) for _ in range(count + 1)]
    best[0][0] = 0
    for j in range(1, count + 1):
        for i in range(j, n + 1):
            for k in range(j - 1, i):
                value = max(best[j - 1][k], prefix[i] - prefix[k])
                if value < best[j][i]:
                    best[j][i], cut[j][i] = value, k
    cuts, i = [], n
    for j in range(count, 1, -1):
        i = cut[j][i]
        cuts.append(i)
    return cuts[::-1]


def voices_of(config: Mapping) -> dict[str, str]:
    """Speaker name to voice, from the `[models]` table (persona document, spike S4)."""
    models = config["models"]
    return {"Lena": models["tts_voice_lena"], "Felix": models["tts_voice_felix"]}


def build_request(model: str, voices: Mapping[str, str], turns: Sequence[tuple[str, str]]) -> dict:
    """Two-speaker interaction (spike S4): one text item per turn, annotated with its speaker."""
    return {
        "model": model,
        "input": [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [{"type": "speech_metadata", "speaker": speaker}],
                    }
                    for speaker, text in turns
                ],
            }
        ],
        "response_format": {"type": "audio"},
        "generation_config": {
            "speech_config": {
                "speakers": [{"speaker": name, "voice": voice} for name, voice in voices.items()]
            }
        },
    }


def retry_seconds(text: str) -> float | None:
    """Wait time from a `429` body ("... Please retry in 25s ...")."""
    match = _RETRY_IN.search(text)
    return float(match.group(1)) if match else None


def parse_response(body: object) -> Synthesis:
    """Audio and usage from a successful interaction; `TtsError` if there is no audio."""
    if not isinstance(body, dict):
        raise TtsError("TTS answer is not a JSON object.")
    status = body.get("status")
    if status is not None and status != "completed":
        raise TtsError(f"TTS interaction ended with status '{status}'.")
    audio = [
        c
        for step in body.get("steps", []) or []
        if isinstance(step, dict)
        for c in step.get("content", []) or []
        if isinstance(c, dict) and c.get("type") == "audio"
    ]
    if len(audio) != 1 or not isinstance(audio[0].get("data"), str):
        raise TtsError(f"TTS answer has {len(audio)} audio parts, expected one.")
    try:
        data = base64.b64decode(audio[0]["data"], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise TtsError(f"TTS audio is not valid base64: {exc}") from exc
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return Synthesis(
        data, _int(usage.get("total_input_tokens")), _int(usage.get("total_output_tokens"))
    )


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# ---------------------------------------------------------------- client


class GeminiTts:
    def __init__(
        self,
        model: str,
        voices: Mapping[str, str],
        max_retries: int,
        backoff_seconds: float,
        timeout_seconds: float,
        api_key: str,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._model = model
        self._voices = dict(voices)
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._sleep = sleep
        self._client = client or httpx.Client(
            headers={"x-goog-api-key": api_key}, timeout=timeout_seconds
        )

    def synthesize(self, turns: Sequence[tuple[str, str]]) -> Synthesis:
        payload = build_request(self._model, self._voices, turns)
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(API_URL, json=payload)
            except httpx.TransportError as exc:  # timeout, connection reset
                status, text, wait = 599, f"{type(exc).__name__}: {exc}", None
            else:
                status, text = response.status_code, response.text
                if status == 200:
                    return parse_response(_json(response))
                wait = retry_seconds(text) if status == 429 else None
            if (status in TRANSIENT_STATUS or status == 599) and attempt < self._max_retries:
                # The wait the server names in a 429 is its own wish.
                self._sleep(wait if wait is not None else self._backoff * 2**attempt)
                continue
            raise TtsError(f"TTS answered {status}: {text[:500]}")
        raise AssertionError("unreachable")


def _json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return None
