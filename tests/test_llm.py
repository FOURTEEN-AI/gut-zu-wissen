from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from gut_zu_wissen.llm import CACHE_CONTROL, AnthropicLlm, LlmError, request_params
from gut_zu_wissen.usage import Usage


class FakeClient:
    def __init__(self, text: str = "{}", stop_reason: str = "end_turn") -> None:
        self.calls: list[dict] = []
        response = SimpleNamespace(
            stop_reason=stop_reason,
            stop_details=None,
            content=[SimpleNamespace(type="text", text=text)],
        )
        self.response = response
        self.messages = SimpleNamespace(stream=self._stream)

    @contextmanager
    def _stream(self, **kw):
        self.calls.append(kw)
        yield SimpleNamespace(get_final_message=lambda: self.response)


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
    client.response.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
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


def test_without_cache_markers_system_and_user_stay_plain_strings():
    params = request_params("m", 10, "sys", "usr", {})
    assert params["system"] == "sys"
    assert params["messages"] == [{"role": "user", "content": "usr"}]
    assert "cache_control" not in str(params)


def test_cache_system_puts_a_breakpoint_on_the_system_prompt():
    params = request_params("m", 10, "sys", "usr", {}, cache_system=True)
    assert params["system"] == [{"type": "text", "text": "sys", "cache_control": CACHE_CONTROL}]
    assert params["messages"][0]["content"] == "usr"
    assert CACHE_CONTROL == {"type": "ephemeral"}  # 5-minute TTL


def test_cache_user_prefix_splits_the_user_message_at_the_breakpoint():
    params = request_params(
        "m", 10, "sys", "Angebot: A\n\nRest", {}, cache_user_prefix="Angebot: A\n\n"
    )
    assert params["system"] == "sys"
    content = params["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": "Angebot: A\n\n", "cache_control": CACHE_CONTROL},
        {"type": "text", "text": "Rest"},
    ]
    assert "".join(b["text"] for b in content) == "Angebot: A\n\nRest"  # same text


def test_cache_user_prefix_must_start_the_user_message():
    with pytest.raises(ValueError):
        request_params("m", 10, "s", "user", {}, cache_user_prefix="other")


def test_the_client_sends_the_cache_markers():
    client = FakeClient("{}")
    AnthropicLlm("m", 10, client).complete_json("s", "pu", {}, cache_system=True,
                                                cache_user_prefix="p")  # fmt: skip
    call = client.calls[0]
    assert call["system"][0]["cache_control"] == CACHE_CONTROL
    assert call["messages"][0]["content"][0] == {
        "type": "text", "text": "p", "cache_control": CACHE_CONTROL
    }  # fmt: skip


def test_cache_tokens_are_added_to_the_usage_counter():
    usage = Usage("m")
    client = FakeClient("{}")
    client.response.usage = SimpleNamespace(
        input_tokens=50, output_tokens=7, cache_creation_input_tokens=900,
        cache_read_input_tokens=None,
    )  # fmt: skip
    AnthropicLlm("m", 10, client, usage).complete_json("s", "u", {}, cache_system=True)
    assert (usage.input_tokens, usage.cache_creation_input_tokens) == (50, 900)
    assert usage.cache_read_input_tokens == 0  # None in the response counts as 0
