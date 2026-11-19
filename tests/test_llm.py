from types import SimpleNamespace

import pytest

from gut_zu_wissen.llm import AnthropicLlm, LlmError
from gut_zu_wissen.usage import Usage


class FakeClient:
    def __init__(self, text: str = "{}", stop_reason: str = "end_turn") -> None:
        self.calls: list[dict] = []
        response = SimpleNamespace(
            stop_reason=stop_reason,
            stop_details=None,
            content=[SimpleNamespace(type="text", text=text)],
        )
        self.messages = SimpleNamespace(create=lambda **kw: (self.calls.append(kw), response)[1])


def test_request_uses_structured_output_without_forced_tool_or_sampling():
    client = FakeClient('{"candidates": []}')
    out = AnthropicLlm("claude-sonnet-5-5", 100, client).complete_json("sys", "usr", {"type": "x"})
    assert out == {"candidates": []}
    call = client.calls[0]
    assert call["model"] == "claude-sonnet-5-5" and call["max_tokens"] == 100
    assert call["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "x"}}}
    assert call["system"] == "sys" and call["messages"] == [{"role": "user", "content": "usr"}]
    assert not {"tool_choice", "tools", "temperature", "top_p", "top_k"} & set(call)


@pytest.mark.parametrize(
    ("text", "stop"),
    [("{}", "refusal"), ("{}", "max_tokens"), ("kein json", "end_turn"), ("[1]", "end_turn")],
)
def test_failures_raise_llm_error(text, stop):
    with pytest.raises(LlmError):
        AnthropicLlm("m", 10, FakeClient(text, stop)).complete_json("s", "u", {})


def with_usage(client: FakeClient, input_tokens: int, output_tokens: int) -> FakeClient:
    response = client.messages.create()
    client.calls.clear()
    response.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return client


def test_token_counts_are_added_to_the_usage_counter_also_for_a_refusal():
    usage = Usage("claude-sonnet-5-5")
    AnthropicLlm("m", 10, with_usage(FakeClient("{}"), 120, 30), usage).complete_json("s", "u", {})
    refusal = with_usage(FakeClient("{}", "refusal"), 80, 5)
    with pytest.raises(LlmError):
        AnthropicLlm("m", 10, refusal, usage).complete_json("s", "u", {})
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (2, 200, 35)


def test_without_usage_counter_a_response_without_usage_is_fine():
    assert AnthropicLlm("m", 10, FakeClient("{}")).complete_json("s", "u", {}) == {}
