"""Audio step of a run (phase 1, step 9; design: "Audio").

Markup gate on the final script, sections at turn boundaries, one TTS request per section,
then intro + sections + outro joined (edge silence trimmed, a pause between parts) and
normalised to the target loudness with the true-peak limiter.

Audio files go to a work folder outside the archive (`build/audio/<episode>/`, not committed;
design: audio only in the published form at the host). The archive gets `07_audio.json`: per
section the turns, characters, duration and cost, plus the loudness report and the fingerprints
of the intro and outro used. `AudioRecord` is filled in place, so a failed step can still be
archived with what it produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gut_zu_wissen import audio
from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.markup import markup_problems
from gut_zu_wissen.tts import SectionError, Tts, TtsError, split_sections, voices_of

SECTIONS_DIR = "sections"
EPISODE_AUDIO = "episode.flac"


class AudioStepError(Exception):
    """The audio step cannot finish; the message is the notice for the owner."""


@dataclass
class SectionRecord:
    """One synthesised section; field order is the archive field order."""

    index: int
    turn_indices: list[int]
    chars: int
    file: str
    seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None


@dataclass
class AudioRecord:
    """Content of 07_audio.json; field order is the archive field order."""

    model: str
    voices: dict[str, str]
    section_max_chars: int
    sections: list[SectionRecord] = field(default_factory=list)
    fixed_parts: dict[str, str] = field(default_factory=dict)  # part -> fingerprint
    joins: dict[str, float] = field(default_factory=dict)
    loudness: audio.LoudnessReport | None = None
    file: str | None = None
    sample_rate: int | None = None
    seconds: float | None = None
    usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def cost_usd(input_tokens: int | None, output_tokens: int | None, prices: Mapping) -> float | None:
    """USD for one request (prices per million tokens); `None` if anything is unknown."""
    price_in, price_out = prices.get("gemini_tts_input"), prices.get("gemini_tts_output")
    numbers = (input_tokens, output_tokens, price_in, price_out)
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in numbers):
        return None
    return round((input_tokens * price_in + output_tokens * price_out) / 1e6, 4)


def produce_audio(
    record: AudioRecord,
    turns: Sequence[Turn],
    tts: Tts,
    fixed: Mapping[str, tuple[np.ndarray, int, str]],
    work_dir: Path,
    config: Mapping,
) -> None:
    """Fill `record` and write the section and episode audio to `work_dir`.

    `fixed` is the loaded intro and outro (`intro_outro.load_fixed`). Raises `AudioStepError`.
    """
    quotas, joins = config["quotas"], config["audio"]
    problems = markup_problems(turns)
    if problems:  # the orchestrator rejects such turns, so this is a safety net
        raise AudioStepError(
            "Markup gate: " + " ".join(f"turn {p.turn_index}: {p.reason}" for p in problems)
        )
    try:
        sections = split_sections([len(t.text) for t in turns], record.section_max_chars)
    except SectionError as exc:
        raise AudioStepError(f"Sections: {exc}") from exc
    for n, indices in enumerate(sections):
        record.sections.append(
            SectionRecord(
                n,
                [turns[i].index for i in indices],
                sum(len(turns[i].text) for i in indices),
                f"{SECTIONS_DIR}/{n:02d}.flac",
            )
        )
    intro_y, rate, intro_print = fixed["intro"]
    outro_y, outro_rate, outro_print = fixed["outro"]
    if outro_rate != rate:
        raise AudioStepError(f"Intro and outro have different sample rates ({rate}, {outro_rate}).")
    record.fixed_parts = {"intro": intro_print, "outro": outro_print}
    parts: list[np.ndarray] = []
    for section, indices in zip(record.sections, sections, strict=True):
        try:
            result = tts.synthesize([(turns[i].speaker, turns[i].text) for i in indices])
            y, sr = audio.decode(result.audio)
        except (TtsError, audio.AudioError) as exc:
            raise AudioStepError(f"TTS section {section.index}: {exc}") from exc
        section.input_tokens, section.output_tokens = result.input_tokens, result.output_tokens
        section.usd = cost_usd(result.input_tokens, result.output_tokens, config["prices"])
        section.seconds = round(audio.duration(y, sr), 2)
        if sr != rate:
            raise AudioStepError(
                f"TTS section {section.index} has {sr} Hz; intro and outro have {rate} Hz."
            )
        audio.write(work_dir / section.file, y, sr)
        parts.append(y)
    record.joins = {
        "edge_silence_floor_db": joins["edge_silence_floor_db"],
        "edge_silence_keep_seconds": joins["edge_silence_keep_seconds"],
        "join_pause_seconds": joins["join_pause_seconds"],
    }
    try:
        trimmed = [
            audio.trim_edges(
                y,
                rate,
                joins["edge_silence_floor_db"],
                joins["edge_silence_frame_seconds"],
                joins["edge_silence_keep_seconds"],
            )
            for y in [intro_y, *parts, outro_y]
        ]
        joined = audio.join(trimmed, rate, joins["join_pause_seconds"])
        out, record.loudness = audio.normalize(
            joined,
            rate,
            quotas["target_loudness_lufs"],
            quotas["true_peak_ceiling_dbtp"],
            joins["limiter_attack_db_per_second"],
            joins["limiter_release_db_per_second"],
            joins["loudness_tolerance_lu"],
            joins["loudness_max_iterations"],
        )
    except audio.AudioError as exc:
        raise AudioStepError(f"Audio: {exc}") from exc
    audio.write(work_dir / EPISODE_AUDIO, out, rate)
    record.file, record.sample_rate = EPISODE_AUDIO, rate
    record.seconds = round(audio.duration(out, rate), 2)
    costs = [s.usd for s in record.sections]
    record.usd = round(sum(costs), 4) if all(c is not None for c in costs) else None


def new_record(config: Mapping) -> AudioRecord:
    return AudioRecord(
        config["models"]["tts"], voices_of(config), config["quotas"]["tts_section_max_chars"]
    )
