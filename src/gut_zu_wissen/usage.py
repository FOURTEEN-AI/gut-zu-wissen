"""Token usage per API client, for the per-episode cost log (design: "Kosten"; step 13).

The Claude and Jev clients add the token counts of every successful response to a `Usage`
counter; the run writes the counters to `episode.json` (`usage`). TTS usage and cost are recorded
per synthesis in `08_audio.json`. At the end of the run `episode.json` also gets `cost`: the USD
per client role and for TTS, computed with the config prices in force at run time, and those
prices (user decision 2026-11-19). A price that is not a number (`PLACEHOLDER`) or a synthesis
without a cost gives `None` for that part; nothing is estimated. Whisper runs locally and costs
nothing per call.

Prompt caching (user decision 2026-11-20, spike S7): Claude reports cached prompt tokens apart from
`input_tokens`, which counts only the uncached rest. `Usage` therefore also records the tokens
written to the cache (`cache_creation_input_tokens`, billed at the cache write price) and read
from it (`cache_read_input_tokens`, billed at the cache read price). Folders from before have no
cache fields; they count as 0. A cache price is only needed when its token count is not 0.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PriceKeys:
    """Keys in `[prices]` for the token counts of one client role. None: the provider does not
    charge for these tokens (Jev output tokens, spike S1) or the client caches nothing (Jev)."""

    input: str
    output: str | None
    cache_write: str | None = None
    cache_read: str | None = None

    def token_keys(self) -> tuple[str | None, ...]:
        """In the order of the token fields of `Usage` (`TOKEN_FIELDS`)."""
        return (self.input, self.output, self.cache_write, self.cache_read)


SONNET = PriceKeys(
    "claude_sonnet_input",
    "claude_sonnet_output",
    "claude_sonnet_cache_write",
    "claude_sonnet_cache_read",
)
HAIKU = PriceKeys(
    "claude_haiku_input",
    "claude_haiku_output",
    "claude_haiku_cache_write",
    "claude_haiku_cache_read",
)
# Price keys per client role in `episode.json` `usage`. "extraction" runs on Claude Haiku 5.5
# (user decision 2026-11-20, spike S7); ranking stays with the dialog author ("author", Sonnet).
# Folders from before have no "extraction" role: their extraction is counted under "author".
PRICE_KEYS: dict[str, PriceKeys] = {
    "author": SONNET,
    "llm_judge": SONNET,
    "jev": PriceKeys("jev", None),
    "extraction": HAIKU,
}
TTS_PART = "tts"
TTS_PRICE_KEYS = ("gemini_tts_input", "gemini_tts_output")
TOKENS_PER_PRICE_UNIT = 1_000_000  # prices are per million tokens
USD_DIGITS = 4  # as in 08_audio.json


@dataclass
class Usage:
    """Summed over all calls of one client; field order is the archive field order."""

    model: str
    calls: int = 0
    input_tokens: int = 0  # uncached input only
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cache_creation_input_tokens += cache_creation_input_tokens
        self.cache_read_input_tokens += cache_read_input_tokens


def usage_dict(usage: Mapping[str, Usage] | None) -> dict[str, dict]:
    """The counters as archive data, keyed by client role."""
    return {role: asdict(u) for role, u in (usage or {}).items()}


# Token fields of `Usage`, in archive order; folders from before caching lack the last two.
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)
COUNT_FIELDS = ("calls", *TOKEN_FIELDS)
MODEL_SEPARATOR = ", "


def merge_usage(stored: Mapping[str, Mapping], added: Mapping[str, Mapping]) -> dict[str, dict]:
    """`usage` of an episode after a resume: the counts of `added` summed onto `stored`, per
    role; stored roles keep their order, new roles follow, and so do count fields the stored
    entry lacks (cache tokens on a folder from before caching). When the model of a role changed
    in between, both are named (stored first), so the counts stay attributable."""
    merged = {role: dict(counts) for role, counts in stored.items()}
    for role, counts in added.items():
        if role not in merged:
            merged[role] = dict(counts)
            continue
        entry = merged[role]
        models = str(entry["model"]).split(MODEL_SEPARATOR)
        if counts["model"] not in models:
            entry["model"] = MODEL_SEPARATOR.join([*models, counts["model"]])
        for name in COUNT_FIELDS:  # a field neither has (older folders) is not added
            if name in entry or name in counts:
                entry[name] = entry.get(name, 0) + counts.get(name, 0)
    return merged


# ---------------------------------------------------------------- cost (pure)


def is_price(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def token_counts(usage: Mapping) -> tuple[int, ...]:
    """The token counts of one role in `TOKEN_FIELDS` order; a missing field (older folders)
    is 0."""
    return tuple(int(usage.get(name) or 0) for name in TOKEN_FIELDS)


def token_cost(usage: Mapping, prices: Mapping, keys: PriceKeys) -> float | None:
    """USD for the token counts of one role; None when a price that is needed is not a number.
    Input and output always need their price (an output key of None is free); a cache price is
    needed only when its token count is not 0."""
    usd = 0.0
    for n, (count, key) in enumerate(zip(token_counts(usage), keys.token_keys(), strict=True)):
        is_cache = n >= 2
        if key is None:  # output: free; cache: the client caches nothing
            if is_cache and count:
                return None
            continue
        price = prices.get(key)
        if not is_price(price):
            if count or not is_cache:
                return None
            continue
        usd += count * price
    return round(usd / TOKENS_PER_PRICE_UNIT, USD_DIGITS)


def missing_prices(usage: Mapping, prices: Mapping, keys: PriceKeys) -> list[str]:
    """The price keys `token_cost` needs for these counts that are not a number."""
    needed = [
        key
        for n, (count, key) in enumerate(zip(token_counts(usage), keys.token_keys(), strict=True))
        if key and (n < 2 or count)
    ]
    return [k for k in needed if not is_price(prices.get(k))]


def cache_saving(usage: Mapping, prices: Mapping, keys: PriceKeys) -> float | None:
    """USD the prompt cache saved one role: what the cached tokens would have cost as uncached
    input, minus what their writes and reads cost. Negative when writes were never read back.
    None when a needed price is not a number; 0 without cache tokens."""
    _, _, written, read = token_counts(usage)
    if not written and not read:
        return 0.0
    price_in = prices.get(keys.input)
    price_write = prices.get(keys.cache_write) if written else 0  # no key: None, unknown
    price_read = prices.get(keys.cache_read) if read else 0
    if not all(is_price(p) for p in (price_in, price_write, price_read)):
        return None
    usd = written * (price_in - price_write) + read * (price_in - price_read)
    return round(usd / TOKENS_PER_PRICE_UNIT, USD_DIGITS)


def tts_cost_usd(sections: Sequence[Mapping]) -> float | None:
    """TTS cost from the sections of `08_audio.json`; None when a section has no cost. No
    section (no synthesis): 0."""
    costs = [s.get("usd") for s in sections]
    if any(c is None for c in costs):
        return None
    return round(sum(costs), USD_DIGITS)


def audio_tts_cost_usd(audio: Mapping) -> float | None:
    """TTS cost of `08_audio.json` over every attempt of the audio step: its sections and those
    of the earlier attempts a resume kept (`previous_attempts`); None when one is unknown."""
    records = [audio, *audio.get("previous_attempts", [])]
    costs = [tts_cost_usd(r.get("sections", [])) for r in records]
    if any(c is None for c in costs):
        return None
    return round(sum(costs), USD_DIGITS)


def episode_cost(usage: Mapping[str, Mapping], tts_usd: float | None, prices: Mapping) -> dict:
    """`cost` of `episode.json`: the prices used (as in the config at run time, also when not a
    number) and the USD per client role and for TTS. A role without a price key gets None.

    Price order: input and output per role, TTS, then the cache prices per role (added with
    caching, so the older keys keep their order)."""
    role_keys = [PRICE_KEYS[role] for role in usage if role in PRICE_KEYS]
    keys = [k for p in role_keys for k in (p.input, p.output) if k]
    keys += list(TTS_PRICE_KEYS)
    keys += [k for p in role_keys for k in (p.cache_write, p.cache_read) if k]
    used = {k: prices.get(k) for k in dict.fromkeys(keys)}
    usd: dict[str, float | None] = {}
    for role, counts in usage.items():
        keys_of_role = PRICE_KEYS.get(role)
        usd[role] = token_cost(counts, prices, keys_of_role) if keys_of_role else None
    usd[TTS_PART] = tts_usd
    return {"prices": used, "usd": usd}
