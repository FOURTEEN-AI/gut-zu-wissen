from dataclasses import asdict

from gut_zu_wissen.usage import (
    PRICE_KEYS,
    SONNET,
    PriceKeys,
    Usage,
    audio_tts_cost_usd,
    cache_saving,
    episode_cost,
    merge_usage,
    missing_prices,
    token_cost,
    tts_cost_usd,
)

JEV = PriceKeys("jev", None)

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
    assert token_cost(USAGE["author"], PRICES, SONNET) == 3  # no cache tokens: no cache price
    assert token_cost(USAGE["jev"], PRICES, JEV) == 0.021  # output tokens free
    assert token_cost(USAGE["jev"], {"jev": "PLACEHOLDER"}, JEV) is None


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
            "claude_sonnet_cache_write": None,  # not in these prices; no cache tokens either
            "claude_sonnet_cache_read": None,
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


CACHE_PRICES = {**PRICES, "claude_sonnet_cache_write": 2.5, "claude_sonnet_cache_read": 0.1}
CACHED = {
    "model": "a",
    "calls": 4,
    "input_tokens": 100_000,
    "output_tokens": 10_000,
    "cache_creation_input_tokens": 200_000,
    "cache_read_input_tokens": 1_000_000,
}


def test_usage_records_cache_tokens_after_the_older_fields():
    usage = Usage("m")
    usage.add(10, 5)
    usage.add(20, 6, 300, 0)
    usage.add(1, 1, 0, 300)
    assert list(asdict(usage)) == [
        "model",
        "calls",
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ]
    assert asdict(usage) == {
        "model": "m",
        "calls": 3,
        "input_tokens": 31,
        "output_tokens": 12,
        "cache_creation_input_tokens": 300,
        "cache_read_input_tokens": 300,
    }


def test_token_cost_prices_cache_writes_and_reads():
    # 0.1M * 2 + 0.01M * 10 + 0.2M * 2.5 + 1M * 0.1 = 0.2 + 0.1 + 0.5 + 0.1
    assert token_cost(CACHED, CACHE_PRICES, SONNET) == 0.9


def test_token_cost_needs_a_cache_price_only_for_cache_tokens():
    assert token_cost(CACHED, PRICES, SONNET) is None
    assert missing_prices(CACHED, PRICES, SONNET) == [
        "claude_sonnet_cache_write",
        "claude_sonnet_cache_read",
    ]
    reads_only = {**CACHED, "cache_creation_input_tokens": 0}
    assert missing_prices(reads_only, PRICES, SONNET) == ["claude_sonnet_cache_read"]
    assert missing_prices(USAGE["author"], PRICES, SONNET) == []
    # a client that caches nothing cannot have cache tokens
    assert token_cost({**USAGE["jev"], "cache_read_input_tokens": 1}, PRICES, JEV) is None


def test_cache_saving_against_uncached_input():
    # writes cost 0.5 more per M (0.2M: -0.1), reads 1.9 less per M (1M: +1.9)
    assert cache_saving(CACHED, CACHE_PRICES, SONNET) == 1.8
    assert cache_saving(USAGE["author"], PRICES, SONNET) == 0.0  # older folder: no cache
    assert cache_saving(CACHED, PRICES, SONNET) is None
    written_only = {**CACHED, "cache_read_input_tokens": 0}
    assert cache_saving(written_only, CACHE_PRICES, SONNET) == -0.1


def test_episode_cost_with_cache_tokens_adds_the_cache_prices_last():
    cost = episode_cost({"author": CACHED, "jev": USAGE["jev"]}, 0.1, CACHE_PRICES)
    assert list(cost["prices"]) == [
        "claude_sonnet_input",
        "claude_sonnet_output",
        "jev",
        "gemini_tts_input",
        "gemini_tts_output",
        "claude_sonnet_cache_write",
        "claude_sonnet_cache_read",
    ]
    assert cost["usd"] == {"author": 0.9, "jev": 0.021, "tts": 0.1}


def test_merge_usage_adds_cache_counts_onto_an_older_entry():
    old = {"author": USAGE["author"]}
    merged = merge_usage(old, {"author": CACHED})
    assert list(merged["author"]) == [*USAGE["author"], "cache_creation_input_tokens",
                                      "cache_read_input_tokens"]  # fmt: skip
    assert merged["author"]["cache_read_input_tokens"] == 1_000_000
    assert merged["author"]["input_tokens"] == 1_100_000


def test_every_claude_role_has_cache_price_keys():
    for role in ("author", "llm_judge"):
        assert PRICE_KEYS[role].cache_write and PRICE_KEYS[role].cache_read
