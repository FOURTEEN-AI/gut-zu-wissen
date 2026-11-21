"""Resynthesis of single turns (decided 2026-11-21; design: "Fehlerbehandlung", "Audio").

A deviation that still changes the statement after the second hearing lies in one turn (or at the
border of two). Instead of the whole section, only that turn is synthesised again and put into the
section's audio in place of the old one. A shorter synthesis has fewer chances of new errors.

Pure functions:

- `turn_offsets`: where each turn starts in the normalised script words of a stretch;
- `affected_turns`: the turns that hold the deviations that count;
- `boundary_cuts`: per border between two turns, the sample where the audio can be cut (in the
  quietest frame between the last word of one turn and the first word of the next), or `None`
  where the word times do not allow a cut;
- `blocks`: the runs of affected turns, widened to the next border that can be cut.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gut_zu_wissen.audio import quietest_sample
from gut_zu_wissen.verification import Deviation, Span, tokenize


def turn_offsets(texts: Sequence[str]) -> list[int] | None:
    """Index of each turn's first word in the normalised words of the joined texts, plus the
    total; `None` if normalising the turns one by one gives other words than the joined text (a
    number split across two turns)."""
    offsets, words = [0], []
    for text in texts:
        tokens = [t.norm for t in tokenize(text)]
        words += tokens
        offsets.append(offsets[-1] + len(tokens))
    if words != [t.norm for t in tokenize(" ".join(texts))]:
        return None
    return offsets


def _turn_of(word: int, offsets: Sequence[int]) -> int:
    """Turn holding normalised word `word` (clamped to the first and last turn)."""
    turns = len(offsets) - 1
    for t in range(turns):
        if word < offsets[t + 1]:
            return t
    return turns - 1


def affected_turns(deviations: Sequence[Deviation], offsets: Sequence[int]) -> set[int]:
    """Turns holding the given deviations. A deviation spanning a border affects both turns; an
    insertion at a border (Whisper heard extra words between two turns) affects both too."""
    affected: set[int] = set()
    for d in deviations:
        if d.script_start == d.script_end:
            words = {max(0, d.script_start - 1), d.script_start}
        else:
            words = {d.script_start, d.script_end - 1}
        turns = sorted(_turn_of(w, offsets) for w in words)
        affected.update(range(turns[0], turns[-1] + 1))
    return affected


def boundary_cuts(
    times: Sequence[Span | None],
    offsets: Sequence[int],
    y: np.ndarray,
    sr: int,
    frame_seconds: float,
) -> list[int | None]:
    """Sample to cut at for each border between turn k-1 and turn k (k = 1 … turns-1).

    A cut needs the time of the last word before and the first word after the border, and a gap
    between them; it lies in the quietest frame of that gap.
    """
    cuts: list[int | None] = []
    for k in range(1, len(offsets) - 1):
        border = offsets[k]
        if border == 0 or border >= len(times):
            cuts.append(None)
            continue
        before, after = times[border - 1], times[border]
        if before is None or after is None or before[1] >= after[0]:
            cuts.append(None)
            continue
        cuts.append(quietest_sample(y, sr, before[1], after[0], frame_seconds))
    return cuts


def blocks(affected: set[int], turns: int, cuttable: Sequence[bool]) -> list[tuple[int, int]]:
    """Runs [first, end) of turns to synthesise again.

    `cuttable[k-1]` says whether the border before turn k can be cut. Each run of affected turns
    is widened until both its borders can be cut or it reaches the edge; overlapping or touching
    runs are merged.
    """
    runs: list[tuple[int, int]] = []
    for t in sorted(affected):
        first, end = t, t + 1
        while first > 0 and not cuttable[first - 1]:
            first -= 1
        while end < turns and not cuttable[end - 1]:
            end += 1
        if runs and first <= runs[-1][1]:
            runs[-1] = (runs[-1][0], max(runs[-1][1], end))
        else:
            runs.append((first, end))
    return runs
