from gut_zu_wissen.usage import episode_cost, token_cost, tts_cost_usd

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
