"""Pronunciation lexicon for names the TTS voices get wrong (decided 2026-11-20).

Two pure functions around the audio step, both driven by `[[pronunciation]]` entries in the
config (`written`, `spoken`, `heard`):

- `respell` replaces the written form by a German respelling in the text sent to the TTS only.
  The script, the archive and the verification keep the written form.
- `canonicalize` replaces the listed variants in the Whisper transcript by the written form before
  the diff. Whisper knows famous names under their common spelling and writes that, not what was
  said; without this, the same word is flagged as a changed statement in every attempt.

A run can add entries of its own in `pronunciation.json` in the episode folder (a list of the same
entries, written by hand); they come first and replace a config entry with the same `written`.

Both match whole words only and are case-sensitive. A variant that the script itself contains
is never rewritten.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path


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


FILE = "pronunciation.json"
_FIELDS = {"written", "spoken", "heard"}


class PronunciationError(Exception):
    """The episode's lexicon file is not a list of valid entries."""


def load_episode_entries(folder: Path) -> list[dict]:
    """Entries of `<folder>/pronunciation.json`; none if the file does not exist."""
    path = folder / FILE
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise PronunciationError(f"{FILE}: not valid JSON: {exc}") from exc
    if not isinstance(entries, list):
        raise PronunciationError(f"{FILE}: expected a list of entries.")
    for n, entry in enumerate(entries):
        ok = isinstance(entry, dict) and {"written", "spoken"} <= entry.keys()
        if not ok or not entry.keys() <= _FIELDS:
            raise PronunciationError(
                f"{FILE}: entry {n} needs `written` and `spoken`, optionally `heard`."
            )
        if not all(isinstance(entry[k], str) and entry[k] for k in ("written", "spoken")):
            raise PronunciationError(f"{FILE}: entry {n}: `written` and `spoken` must be text.")
        heard = entry.get("heard", [])
        if not isinstance(heard, list) or not all(isinstance(h, str) and h for h in heard):
            raise PronunciationError(f"{FILE}: entry {n}: `heard` must be a list of text.")
    return entries


def merge(config_entries: Sequence[Mapping], episode_entries: Sequence[Mapping]) -> list[dict]:
    """Episode entries first; a config entry with the same `written` form is replaced."""
    own = {e["written"] for e in episode_entries}
    return [dict(e) for e in episode_entries] + [
        dict(e) for e in config_entries if e["written"] not in own
    ]
