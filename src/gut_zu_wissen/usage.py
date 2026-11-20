"""Token usage per API client, for the per-episode cost log (design: "Kosten"; step 13).

The Claude and Jev clients add the token counts of every successful response to a `Usage`
counter; the run writes the counters to `episode.json` (`usage`). TTS usage and cost are recorded
per synthesis in `08_audio.json`. At the end of the run `episode.json` also gets `cost`: the USD
per client role and for TTS, computed with the config prices in force at run time, and those
prices (user decision 2026-11-19). A price that is not a number (`PLACEHOLDER`) or a synthesis
without a cost gives `None` for that part; nothing is estimated. Whisper runs locally and costs
nothing per call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

# Price keys in `[prices]` per client role in `episode.json` `usage` (input, output). None: the
# provider does not charge for these tokens (Jev output tokens, spike S1).
PRICE_KEYS: dict[str, tuple[str, str | None]] = {
    "author": ("claude_sonnet_input", "claude_sonnet_output"),
    "llm_judge": ("claude_sonnet_input", "claude_sonnet_output"),
    "jev": ("jev", None),
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
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens


def usage_dict(usage: Mapping[str, Usage] | None) -> dict[str, dict]:
    """The counters as archive data, keyed by client role."""
    return {role: asdict(u) for role, u in (usage or {}).items()}


COUNT_FIELDS = ("calls", "input_tokens", "output_tokens")
MODEL_SEPARATOR = ", "


def merge_usage(stored: Mapping[str, Mapping], added: Mapping[str, Mapping]) -> dict[str, dict]:
    """`usage` of an episode after a resume: the counts of `added` summed onto `stored`, per
    role; stored roles keep their order, new roles follow. When the model of a role changed in
    between, both are named (stored first), so the counts stay attributable."""
    merged = {role: dict(counts) for role, counts in stored.items()}
    for role, counts in added.items():
        if role not in merged:
            merged[role] = dict(counts)
            continue
        entry = merged[role]
        models = str(entry["model"]).split(MODEL_SEPARATOR)
        if counts["model"] not in models:
            entry["model"] = MODEL_SEPARATOR.join([*models, counts["model"]])
        for name in COUNT_FIELDS:
            entry[name] = entry.get(name, 0) + counts.get(name, 0)
    return merged


# ---------------------------------------------------------------- cost (pure)


def is_price(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def token_cost(usage: Mapping, prices: Mapping, keys: tuple[str, str | None]) -> float | None:
    """USD for the token counts of one role; None when a price is not a number."""
    price_in = prices.get(keys[0])
    price_out = prices.get(keys[1]) if keys[1] else 0
    if not is_price(price_in) or not is_price(price_out):
        return None
    usd = (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out) / (
        TOKENS_PER_PRICE_UNIT
    )
    return round(usd, USD_DIGITS)


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
    number) and the USD per client role and for TTS. A role without a price key gets None."""
    keys = [k for role in usage for k in PRICE_KEYS.get(role, ()) if k] + list(TTS_PRICE_KEYS)
    used = {k: prices.get(k) for k in dict.fromkeys(keys)}
    usd: dict[str, float | None] = {}
    for role, counts in usage.items():
        role_keys = PRICE_KEYS.get(role)
        usd[role] = token_cost(counts, prices, role_keys) if role_keys else None
    usd[TTS_PART] = tts_usd
    return {"prices": used, "usd": usd}
