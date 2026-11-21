"""LLM access behind an interface (CLAUDE.md: LLM and API calls behind interfaces).

`StructuredLlm` returns a JSON object that follows a JSON schema. Tests use fakes; CI never
calls the live API. The API key comes from the environment (`ANTHROPIC_API_KEY`).

Prompt caching (user decision 2026-11-20, spike S7): a caller names the stable prefix of its
request. `cache_system` marks the system prompt (the same for many calls: extraction, LLM
judge); `cache_user_prefix` marks the start of the user message (the offer of a turn
regeneration). The marked part ends in a `cache_control` breakpoint (5-minute TTL); everything
after it varies per call. Caching is a prefix match over schema, system prompt and messages, so
a caller only marks what really repeats; a marker on a prefix that is never sent again only pays
the write price.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Protocol

import anthropic

from gut_zu_wissen.usage import Usage

log = logging.getLogger(__name__)


class LlmError(Exception):
    """The model call failed in a way a retry of the same request will not fix."""


class StructuredLlm(Protocol):
    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        cache_system: bool = False,
        cache_user_prefix: str = "",
    ) -> dict:
        """One request; the answer is a JSON object that follows `schema`.

        `cache_system`: the system prompt is a stable prefix. `cache_user_prefix`: `user` starts
        with this stable prefix. Neither changes what the model reads."""
        ...


CACHE_CONTROL = {"type": "ephemeral"}  # 5-minute TTL (spike S7: calls follow within minutes)


def request_params(
    model: str,
    max_tokens: int,
    system: str,
    user: str,
    schema: dict,
    *,
    cache_system: bool = False,
    cache_user_prefix: str = "",
) -> dict:
    """The Messages API parameters of one `complete_json` call (pure; also the request builder
    of the batch measurement). Without cache markers system and user stay plain strings; with
    them they become text blocks whose concatenation is the same text, the marked block carrying
    `cache_control`."""
    if cache_user_prefix and not user.startswith(cache_user_prefix):
        raise ValueError("cache_user_prefix must be the start of the user message")
    system_param: str | list[dict] = system
    if cache_system:
        system_param = [{"type": "text", "text": system, "cache_control": CACHE_CONTROL}]
    content: str | list[dict] = user
    if cache_user_prefix:
        content = [{"type": "text", "text": cache_user_prefix, "cache_control": CACHE_CONTROL}]
        rest = user[len(cache_user_prefix) :]
        if rest:
            content.append({"type": "text", "text": rest})
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_param,
        "messages": [{"role": "user", "content": content}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }


class AnthropicLlm:
    """`StructuredLlm` on the Claude Messages API with structured output.

    No forced `tool_choice` and no sampling parameters: Claude Sonnet 5.5 rejects both, and
    Claude Haiku 5.5 (extraction) rejects non-default sampling parameters. No `thinking` and no
    `effort`: each model runs its default (adaptive thinking on both; effort `high` on Sonnet 5.5,
    `medium` on Haiku 5.5, as measured in spike S7). The answer is read from the text block by
    type, so leading thinking blocks (Haiku 5.5 thinks by default) are skipped.
    The token counts of every response, a refused or cut-off one included (both are billed), are
    added to `usage` when given (cost log), cache writes and reads included.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        client: anthropic.Anthropic | None = None,
        usage: Usage | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic()
        self._usage = usage

    def complete_json(
        self,
        system: str,
        user: str,
        schema: dict,
        *,
        cache_system: bool = False,
        cache_user_prefix: str = "",
    ) -> dict:
        params = request_params(
            self._model,
            self._max_tokens,
            system,
            user,
            schema,
            cache_system=cache_system,
            cache_user_prefix=cache_user_prefix,
        )
        # Streamed, so that large output ceilings (dialog author) stay under the SDK's HTTP
        # timeout; the final message carries the same fields as a non-streamed response.
        start = time.monotonic()
        with self._client.messages.stream(**params) as stream:
            response = stream.get_final_message()
        counts = getattr(response, "usage", None)
        if self._usage is not None and counts is not None:
            self._usage.add(
                counts.input_tokens,
                counts.output_tokens,
                getattr(counts, "cache_creation_input_tokens", None) or 0,
                getattr(counts, "cache_read_input_tokens", None) or 0,
            )
        log.debug(
            "LLM %s: %.1f s, %s in, %s out, stop %s",
            self._model,
            time.monotonic() - start,
            getattr(counts, "input_tokens", "?"),
            getattr(counts, "output_tokens", "?"),
            response.stop_reason,
        )
        if response.stop_reason == "refusal":
            raise LlmError(f"The model declined the request: {response.stop_details}")
        if response.stop_reason == "max_tokens":
            raise LlmError("The answer was cut off at max_tokens.")
        text = next((b.text for b in response.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LlmError(f"The answer is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LlmError("The answer is not a JSON object.")
        return data
