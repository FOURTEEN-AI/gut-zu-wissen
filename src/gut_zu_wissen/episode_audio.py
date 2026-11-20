"""Audio step of a run (phase 1, steps 9 and 10; design: "Audio", "Fehlerbehandlung").

Markup gate on the final script, sections at turn boundaries, one TTS request per section, STT
verification of every section, then intro + sections + outro joined (edge silence trimmed, a
pause between parts) and normalised to the target loudness with the true-peak limiter.

Verification (step 10): Whisper transcribes the raw section audio; the transcript is diffed
against the section's script after normalisation; the LLM judge model decides per deviation
whether it changes the statement. If one does, the section is synthesised again, at most
`audio_section_retries` times (audio level of the error table); after that the run escalates.
Intro and outro are fixed audio and not verified (design: "nicht Teil des STT-Loops").

Audio files go to a work folder outside the archive (`build/audio/<episode>/`, not committed;
design: audio only in the published form at the host); a resynthesis overwrites the section
file. The archive gets `08_audio.json`: per section the turns, characters, duration of the
accepted audio, tokens and cost summed over all syntheses, and per synthesis attempt its
transcript, deviations and decisions (design: transcripts and diff decisions of discarded
syntheses are archived), plus the loudness report and the fingerprints of the intro and outro.
`AudioRecord` is filled in place, so a failed step can still be archived with what it produced.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gut_zu_wissen import audio
from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.llm import LlmError, StructuredLlm
from gut_zu_wissen.markup import markup_problems
from gut_zu_wissen.stt import Stt, SttError
from gut_zu_wissen.tts import SectionError, Tts, TtsError, split_sections, voices_of
from gut_zu_wissen.usage import TTS_PRICE_KEYS
from gut_zu_wissen.verification import (
    Deviation,
    RetryAction,
    changes_statement,
    decide_deviations,
    diff,
    retry_action,
)

SECTIONS_DIR = "sections"
EPISODE_AUDIO = "episode.flac"


class AudioStepError(Exception):
    """The audio step cannot finish; the message is the notice for the owner."""


@dataclass
class AttemptRecord:
    """One synthesis of a section and its verification; field order is the archive field order."""

    attempt: int  # 0: first synthesis, then one per resynthesis
    seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None
    transcript: str | None = None
    deviations: list[Deviation] = field(default_factory=list)
    changes_statement: bool | None = None
    action: str | None = None  # RetryAction value


@dataclass
class SectionRecord:
    """One section; field order is the archive field order.

    `seconds` is the accepted audio; tokens and cost are summed over all attempts.
    """

    index: int
    turn_indices: list[int]
    chars: int
    file: str
    seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    usd: float | None = None
    verified: bool = False
    attempts: list[AttemptRecord] = field(default_factory=list)


@dataclass
class AudioRecord:
    """Content of 08_audio.json; field order is the archive field order."""

    model: str
    voices: dict[str, str]
    section_max_chars: int
    stt_model: str
    section_retries: int
    sections: list[SectionRecord] = field(default_factory=list)
    fixed_parts: dict[str, str] = field(default_factory=dict)  # part -> fingerprint
    joins: dict[str, float] = field(default_factory=dict)
    loudness: audio.LoudnessReport | None = None
    file: str | None = None
    sample_rate: int | None = None
    seconds: float | None = None
    usd: float | None = None
    # Earlier attempts of the audio step, when it was resumed (`resume.py`): their records as
    # archived, oldest first, so nothing paid for is lost.
    previous_attempts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def cost_usd(input_tokens: int | None, output_tokens: int | None, prices: Mapping) -> float | None:
    """USD for one request (prices per million tokens); `None` if anything is unknown."""
    price_in, price_out = (prices.get(k) for k in TTS_PRICE_KEYS)
    numbers = (input_tokens, output_tokens, price_in, price_out)
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in numbers):
        return None
    return round((input_tokens * price_in + output_tokens * price_out) / 1e6, 4)


def _sum(values: Sequence[int | float | None], digits: int | None = None) -> int | float | None:
    """Sum over all attempts; `None` if any value is unknown."""
    known = [v for v in values if v is not None]
    if len(known) != len(values):
        return None
    total = sum(known)
    return round(total, digits) if digits is not None else total


def produce_audio(
    record: AudioRecord,
    turns: Sequence[Turn],
    tts: Tts,
    stt: Stt,
    decision_llm: StructuredLlm,
    fixed: Mapping[str, tuple[np.ndarray, int, str]],
    work_dir: Path,
    config: Mapping,
) -> None:
    """Fill `record` and write the section and episode audio to `work_dir`.

    `fixed` is the loaded intro and outro (`intro_outro.load_fixed`); `decision_llm` decides
    whether a deviation changes the statement. Raises `AudioStepError`.
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
        parts.append(
            _verified_section(
                section,
                [(turns[i].speaker, turns[i].text) for i in indices],
                tts,
                stt,
                decision_llm,
                rate,
                work_dir,
                config,
            )
        )
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


def _verified_section(
    section: SectionRecord,
    section_turns: Sequence[tuple[str, str]],
    tts: Tts,
    stt: Stt,
    decision_llm: StructuredLlm,
    rate: int,
    work_dir: Path,
    config: Mapping,
) -> np.ndarray:
    """Synthesise and verify one section, resynthesising within the budget; the accepted audio."""
    budget = config["quotas"]["audio_section_retries"]
    script = " ".join(text for _, text in section_turns)
    while True:
        attempt = AttemptRecord(len(section.attempts))
        section.attempts.append(attempt)
        try:
            result = tts.synthesize(section_turns)
            y, sr = audio.decode(result.audio)
        except (TtsError, audio.AudioError) as exc:
            raise AudioStepError(f"TTS section {section.index}: {exc}") from exc
        attempt.input_tokens, attempt.output_tokens = result.input_tokens, result.output_tokens
        attempt.usd = cost_usd(result.input_tokens, result.output_tokens, config["prices"])
        attempt.seconds = round(audio.duration(y, sr), 2)
        section.input_tokens = _sum([a.input_tokens for a in section.attempts])
        section.output_tokens = _sum([a.output_tokens for a in section.attempts])
        section.usd = _sum([a.usd for a in section.attempts], 4)
        if sr != rate:
            raise AudioStepError(
                f"TTS section {section.index} has {sr} Hz; intro and outro have {rate} Hz."
            )
        audio.write(work_dir / section.file, y, sr)
        try:
            attempt.transcript = stt.transcribe(y, sr)
            attempt.deviations = diff(script, attempt.transcript, config["stt"]["context_words"])
            decide_deviations(decision_llm, section_turns, attempt.transcript, attempt.deviations)
        except (SttError, LlmError) as exc:
            raise AudioStepError(f"STT verification, section {section.index}: {exc}") from exc
        attempt.changes_statement = changes_statement(attempt.deviations)
        action = retry_action(attempt.attempt, attempt.changes_statement, budget)
        attempt.action = action.value
        if action is RetryAction.ACCEPT:
            section.verified, section.seconds = True, attempt.seconds
            return y
        if action is RetryAction.ESCALATE:
            changed = [d for d in attempt.deviations if d.changes_statement is not False]
            details = "; ".join(
                f"'{d.expected}' heard as '{d.heard}' ({d.reason})" for d in changed
            )
            raise AudioStepError(
                f"STT verification: section {section.index} still changes the statement after "
                f"{budget} resyntheses: {details}"
            )


def new_record(config: Mapping) -> AudioRecord:
    return AudioRecord(
        config["models"]["tts"],
        voices_of(config),
        config["quotas"]["tts_section_max_chars"],
        config["models"]["stt"],
        config["quotas"]["audio_section_retries"],
    )
