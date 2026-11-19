"""LLM access behind an interface (CLAUDE.md: LLM and API calls behind interfaces).

`StructuredLlm` returns a JSON object that follows a JSON schema. Tests use fakes; CI never
calls the live API. The API key comes from the environment (`ANTHROPIC_API_KEY`).
"""

from __future__ import annotations

import json
from typing import Protocol

import anthropic


class LlmError(Exception):
    """The model call failed in a way a retry of the same request will not fix."""


class StructuredLlm(Protocol):
    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """One request; the answer is a JSON object that follows `schema`."""
        ...


class AnthropicLlm:
    """`StructuredLlm` on the Claude Messages API with structured output.

    No forced `tool_choice` and no sampling parameters: Claude Sonnet 5.5 rejects both.
    """

    def __init__(
        self, model: str, max_tokens: int, client: anthropic.Anthropic | None = None
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic()

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
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
