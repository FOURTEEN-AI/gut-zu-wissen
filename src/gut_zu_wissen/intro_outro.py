"""Fixed intro and outro (phase 1, step 9; design: "Begrüßung und Abspann").

Fixed text outside the dialog author and the persona rules, synthesised once with the episode
voices and committed under `assets/audio/`; every episode reuses the files. They are not part of
the STT loop. `fixed_audio.json` records model, voices and a fingerprint of text, model and
voices per part, so the pipeline refuses audio that no longer matches the text or the voices.

Synthesise once (needs `GEMINI_API_KEY`; existing matching files are kept unless `--force`):

    uv run python -m gut_zu_wissen.intro_outro
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from gut_zu_wissen import audio
from gut_zu_wissen.archive import read_json, write_json
from gut_zu_wissen.tts import GeminiTts, Tts, voices_of

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets" / "audio"
METADATA_FILE = "fixed_audio.json"

# Text of the design document, section "Begrüßung und Abspann". Quotation marks are the
# standalone form („…“) of the nested quotes there (‚…‘).
INTRO: tuple[tuple[str, str], ...] = (
    ("Felix", "Hallo und willkommen bei „Gut zu wissen“!"),
    (
        "Lena",
        "In jeder Folge nehmen wir uns einen Wikipedia-Artikel vor und präsentieren euch die "
        "interessantesten Fakten.",
    ),
)
OUTRO: tuple[tuple[str, str], ...] = (
    (
        "Lena",
        "Das war „Gut zu wissen“. Alles, was wir erzählt haben, stammt aus der "
        "deutschsprachigen Wikipedia. Die Artikel und ihre Autorinnen und Autoren findet ihr in "
        "den Shownotes.",
    ),
    (
        "Felix",
        "Ein wichtiger Hinweis: Diese Folge ist mit KI entstanden – von der Auswahl der Fakten "
        "über das Gespräch bis zu unseren Stimmen. KI kann Fehler machen. Und die "
        "Wikipedia-Inhalte selbst sind nicht unabhängig geprüft. Wenn ihr euch auf etwas "
        "verlassen wollt, prüft es bitte nach.",
    ),
    (
        "Lena",
        "„Gut zu wissen“ steht unter der Lizenz CC BY-SA 4.0 und ist ein unabhängiges Projekt, "
        "nicht verbunden mit der Wikimedia Foundation. Bis zum nächsten Mal!",
    ),
)
PARTS: dict[str, tuple[tuple[str, str], ...]] = {"intro": INTRO, "outro": OUTRO}


class FixedAudioError(Exception):
    """Intro or outro audio is missing or does not match the current text and voices."""


def fingerprint(turns: Sequence[tuple[str, str]], model: str, voices: Mapping[str, str]) -> str:
    canonical = json.dumps(
        {"model": model, "voices": dict(sorted(voices.items())), "turns": [list(t) for t in turns]},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_current(folder: Path, model: str, voices: Mapping[str, str]) -> bool:
    try:
        load_fixed(folder, model, voices)
    except FixedAudioError:
        return False
    return True


def synthesize_fixed(tts: Tts, folder: Path, model: str, voices: Mapping[str, str]) -> dict:
    """Synthesise intro and outro, write `<part>.flac` and the metadata file; returns it."""
    parts = []
    for name, turns in PARTS.items():
        result = tts.synthesize(turns)
        y, sr = audio.decode(result.audio)
        audio.write(folder / f"{name}.flac", y, sr)
        parts.append(
            {
                "name": name,
                "file": f"{name}.flac",
                "fingerprint": fingerprint(turns, model, voices),
                "sample_rate": sr,
                "seconds": round(audio.duration(y, sr), 2),
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            }
        )
    metadata = {"model": model, "voices": dict(voices), "parts": parts}
    write_json(folder / METADATA_FILE, metadata)
    return metadata


def load_fixed(
    folder: Path, model: str, voices: Mapping[str, str]
) -> dict[str, tuple[np.ndarray, int, str]]:
    """Samples, sample rate and fingerprint per part. Raises `FixedAudioError` with a notice."""
    hint = "Synthesise them once with `uv run python -m gut_zu_wissen.intro_outro` and commit."
    path = folder / METADATA_FILE
    if not path.exists():
        raise FixedAudioError(f"Intro and outro audio missing ({path}). {hint}")
    entries = {p["name"]: p for p in read_json(path).get("parts", [])}
    loaded: dict[str, tuple[np.ndarray, int, str]] = {}
    for name, turns in PARTS.items():
        expected = fingerprint(turns, model, voices)
        entry = entries.get(name)
        if entry is None or entry.get("fingerprint") != expected:
            raise FixedAudioError(
                f"The {name} audio does not match the current text, model or voices. {hint}"
            )
        file = folder / entry["file"]
        if not file.exists():
            raise FixedAudioError(f"The {name} audio file is missing ({file}). {hint}")
        y, sr = audio.read(file)
        loaded[name] = (y, sr, expected)
    return loaded


def main(argv: list[str] | None = None) -> int:
    from gut_zu_wissen.run import load_config  # run imports this module

    parser = argparse.ArgumentParser(description="Synthesise the fixed intro and outro once.")
    parser.add_argument("--force", action="store_true", help="synthesise even if current")
    parser.add_argument("--out", type=Path, default=ASSETS_DIR, help="target folder")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    config = load_config()
    model, voices = config["models"]["tts"], voices_of(config)
    if not args.force and is_current(args.out, model, voices):
        print(f"Intro and outro in {args.out} match text, model and voices; nothing to do.")
        return 0
    tts_config = config["tts"]
    tts = GeminiTts(
        model,
        voices,
        tts_config["max_retries"],
        tts_config["backoff_seconds"],
        tts_config["timeout_seconds"],
        os.environ["GEMINI_API_KEY"],
    )
    metadata = synthesize_fixed(tts, args.out, model, voices)
    for part in metadata["parts"]:
        print(f"{part['name']}: {part['seconds']} s -> {args.out / part['file']}")
    print("Listen to both files, then commit them with fixed_audio.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
