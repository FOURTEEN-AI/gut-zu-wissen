from gut_zu_wissen.usage import (
    audio_tts_cost_usd,
    episode_cost,
    merge_usage,
    token_cost,
    tts_cost_usd,
)

PRICES = {
    "claude_sonnet_input": 2.0,
    "claude_sonnet_output": 10.0,
    "jev": 0.042,
    "gemini_tts_input": 0.5,
    "gemini_tts_output": 9.0,
    "unrelated": 1.0,
}
USAGE = {
    "author": {"model": "a", "calls": 3, "input_tokens": 1_000_000, "output_tokens": 100_000},
    "llm_judge": {"model": "a", "calls": 1, "input_tokens": 200_000, "output_tokens": 0},
    "jev": {"model": "j", "calls": 9, "input_tokens": 500_000, "output_tokens": 9_000},
}


def test_token_cost_per_million_tokens_and_unknown_without_a_price():
    assert token_cost(USAGE["author"], PRICES, ("claude_sonnet_input", "claude_sonnet_output")) == 3
    assert token_cost(USAGE["jev"], PRICES, ("jev", None)) == 0.021  # output tokens free
    assert token_cost(USAGE["jev"], {"jev": "PLACEHOLDER"}, ("jev", None)) is None


def test_tts_cost_from_sections():
    assert tts_cost_usd([]) == 0
    assert tts_cost_usd([{"usd": 0.05}, {"usd": 0.04}]) == 0.09
    assert tts_cost_usd([{"usd": 0.05}, {"usd": None}]) is None


def test_episode_cost_records_usd_and_the_prices_used():
    cost = episode_cost(USAGE, 0.09, PRICES)
    assert cost == {
        "prices": {
            "claude_sonnet_input": 2.0,
            "claude_sonnet_output": 10.0,
            "jev": 0.042,
            "gemini_tts_input": 0.5,
            "gemini_tts_output": 9.0,
        },
        "usd": {"author": 3.0, "llm_judge": 0.4, "jev": 0.021, "tts": 0.09},
    }


def test_episode_cost_invents_nothing():
    prices = {**PRICES, "claude_sonnet_output": "PLACEHOLDER"}
    usage = {**USAGE, "other": {"model": "x", "calls": 1, "input_tokens": 1, "output_tokens": 1}}
    cost = episode_cost(usage, None, prices)
    assert cost["prices"]["claude_sonnet_output"] == "PLACEHOLDER"
    assert cost["usd"] == {
        "author": None,
        "llm_judge": None,
        "jev": 0.021,
        "other": None,  # no price key for this role
        "tts": None,
    }


def test_merge_usage_sums_the_counts_per_role_and_keeps_the_order():
    added = {
        "llm_judge": {"model": "a", "calls": 2, "input_tokens": 10, "output_tokens": 5},
        "new": {"model": "n", "calls": 1, "input_tokens": 1, "output_tokens": 1},
    }
    merged = merge_usage(USAGE, added)
    assert list(merged) == ["author", "llm_judge", "jev", "new"]
    assert merged["llm_judge"] == {
        "model": "a",
        "calls": 3,
        "input_tokens": 200_010,
        "output_tokens": 5,
    }
    assert merged["author"] == USAGE["author"] and merged["new"] == added["new"]
    assert USAGE["llm_judge"]["calls"] == 1  # inputs unchanged


def test_merge_usage_names_both_models_when_the_model_changed():
    added = {"jev": {"model": "k", "calls": 1, "input_tokens": 0, "output_tokens": 0}}
    once = merge_usage(USAGE, added)
    assert once["jev"]["model"] == "j, k"
    assert merge_usage(once, added)["jev"]["model"] == "j, k"


def test_merge_usage_with_nothing_added_is_the_stored_usage():
    assert merge_usage(USAGE, {}) == USAGE
    assert merge_usage({}, USAGE) == USAGE


def test_audio_tts_cost_covers_the_previous_attempts():
    audio = {
        "sections": [{"usd": 0.1}, {"usd": 0.2}],
        "previous_attempts": [{"sections": [{"usd": 0.05}]}, {"sections": []}],
    }
    assert audio_tts_cost_usd(audio) == 0.35
    assert audio_tts_cost_usd({"sections": [{"usd": 0.1}]}) == 0.1  # no resume
    unknown = {**audio, "previous_attempts": [{"sections": [{"usd": None}]}]}
    assert audio_tts_cost_usd(unknown) is None
