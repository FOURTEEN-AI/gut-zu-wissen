"""Token usage per API client, for the per-episode cost log (design: "Kosten"; step 13).

The Claude and Jev clients add the token counts of every successful response to a `Usage`
counter; the run writes the counters to `episode.json` (`usage`), so the cost can be computed
from the archive with the prices in the config. TTS usage and cost are already recorded per
synthesis in `07_audio.json`. Whisper runs locally and costs nothing per call.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass


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
