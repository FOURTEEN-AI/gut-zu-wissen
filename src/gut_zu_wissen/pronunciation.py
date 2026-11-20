"""Pronunciation lexicon for names the TTS voices get wrong (decided 2026-11-20).

Two pure functions around the audio step, both driven by `[[pronunciation]]` entries in the
config (`written`, `spoken`, `heard`):

- `respell` replaces the written form by a German respelling in the text sent to the TTS only.
  The script, the archive and the verification keep the written form.
- `canonicalize` replaces the listed variants in the Whisper transcript by the written form before
  the diff. Whisper knows famous names under their common spelling and writes that, not what was
  said; without this, the same word is flagged as a changed statement in every attempt.

Both match whole words only and are case-sensitive. A variant that the script itself contains
is never rewritten.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence


def _whole_word(word: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(word)}(?!\w)")


def respell(text: str, entries: Sequence[Mapping]) -> str:
    """`text` with every `written` form replaced by its `spoken` respelling."""
    for entry in entries:
        text = _whole_word(entry["written"]).sub(entry["spoken"], text)
    return text


def canonicalize(transcript: str, script: str, entries: Sequence[Mapping]) -> str:
    """`transcript` with every listed `heard` variant replaced by the `written` form, except
    variants that are words of the `script` itself (they are legitimate there)."""
    for entry in entries:
        for variant in entry.get("heard", []):
            if _whole_word(variant).search(script):
                continue
            transcript = _whole_word(variant).sub(entry["written"], transcript)
    return transcript
