"""Audio step of a run (phase 1, steps 9 and 10; design: "Audio", "Fehlerbehandlung").

Markup gate on the final script, sections at turn boundaries, one TTS request per section, STT
verification of every section, then intro + sections + outro joined (edge silence trimmed, a
pause between parts) and normalised to the target loudness with the true-peak limiter.

Verification (step 10): Whisper transcribes the raw section audio; the transcript is diffed
against the section's script after normalisation; the LLM judge model decides per deviation
whether it changes the statement. Before that, each deviation is heard a second time: Whisper's
word times locate it, the stretch around it is cut out and transcribed again; if that hearing
matches the script there, the deviation does not count and does not go to the LLM (decided
2026-11-21). If one still counts, the section
is not synthesised again as a whole: only the turns holding the remaining deviations are
(decided 2026-11-21, `splice.py`), and their new audio replaces the old between the quietest
points at the turn borders. Each turn is synthesised again at most `audio_section_retries` times
(audio level of the error table); after that the run escalates. Without word times, or where no
border can be cut, the affected stretch grows up to the whole section.
Intro and outro are fixed audio and not verified (design: "nicht Teil des STT-Loops").

Audio files go to a work folder outside the archive (`build/audio/<episode>/`, not committed;
design: audio only in the published form at the host); the section file holds the accepted
audio. The archive gets `08_audio.json`: per section the turns, characters, duration of the
accepted audio, tokens and cost summed over all syntheses, and per synthesis attempt its
turns, transcript, deviations and decisions (design: transcripts and diff decisions of discarded
syntheses are archived), plus the loudness report and the fingerprints of the intro and outro.
`AudioRecord` is filled in place, so a failed step can still be archived with what it produced.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from gut_zu_wissen import audio
from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.llm import LlmError, StructuredLlm
from gut_zu_wissen.markup import markup_problems
from gut_zu_wissen.pronunciation import canonicalize, respell
from gut_zu_wissen.splice import affected_turns, blocks, boundary_cuts, turn_offsets
from gut_zu_wissen.stt import Stt, SttError
from gut_zu_wissen.tts import SectionError, Tts, TtsError, split_sections, voices_of
from gut_zu_wissen.usage import TTS_PRICE_KEYS
from gut_zu_wissen.verification import (
    Deviation,
    RetryAction,
    Span,
    changes_statement,
    counts,
    decide_deviations,
    diff_tokens,
    heard_again_matches,
    hearing_window,
    retry_action,
    script_times,
    token_times,
    tokenize,
)

SECTIONS_DIR = "sections"
EPISODE_AUDIO = "episode.flac"

log = logging.getLogger(__name__)


class AudioStepError(Exception):
    """The audio step cannot finish; the message is the notice for the owner."""


@dataclass
class AttemptRecord:
    """One synthesis of a section and its verification; field order is the archive field order."""

    attempt: int  # 0: first synthesis, then one per resynthesis
    turn_indices: list[int] = field(default_factory=list)  # turns synthesised in this attempt
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
    pronunciation: list[dict] = field(default_factory=list)  # lexicon in force (config + episode)
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
        "turn_pause_seconds": joins["turn_pause_seconds"],
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


@dataclass
class _Piece:
    """Turns [first, end) of a section (positions in the section) and their accepted audio
    (`None`: still to be synthesised)."""

    first: int
    end: int
    y: np.ndarray | None = None


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
    """Synthesise and verify one section; the accepted audio.

    The section starts as one piece. When a piece's audio changes the statement, only the
    affected turns are synthesised again (`splice.blocks`); the rest of its audio is kept, cut at
    the quietest point between two turns. A turn is synthesised at most budget + 1 times; then the
    run escalates. Accepted pieces are joined with their edge silence trimmed and a pause between
    them.
    """
    budget = config["quotas"]["audio_section_retries"]
    lexicon = config.get("pronunciation", [])
    joins = config["audio"]
    syntheses = [0] * len(section_turns)
    pieces = [_Piece(0, len(section_turns))]
    n = 0
    while n < len(pieces):
        piece = pieces[n]
        if piece.y is not None:
            n += 1
            continue
        piece_turns = section_turns[piece.first : piece.end]
        attempt = AttemptRecord(
            len(section.attempts),
            [section.turn_indices[t] for t in range(piece.first, piece.end)],
        )
        section.attempts.append(attempt)
        log.info(
            "Section %d: synthesis attempt %d (turns %s)",
            section.index,
            attempt.attempt,
            f"{attempt.turn_indices[0]}-{attempt.turn_indices[-1]}",
        )
        y = _synthesize(section, attempt, piece_turns, tts, lexicon, rate, config["prices"])
        for t in range(piece.first, piece.end):
            syntheses[t] += 1
        try:
            attempt.transcript, attempt.deviations, times = verify(
                y, rate, piece_turns, stt, decision_llm, lexicon, config["stt"]
            )
        except (SttError, LlmError) as exc:
            raise AudioStepError(f"STT verification, section {section.index}: {exc}") from exc
        attempt.changes_statement = changes_statement(attempt.deviations)
        counting = [d for d in attempt.deviations if counts(d)]
        offsets = turn_offsets([text for _, text in piece_turns])
        if offsets is None or times is None:
            affected = set(range(len(piece_turns)))
        else:
            affected = affected_turns(counting, offsets)
        used = max((syntheses[piece.first + t] - 1 for t in affected), default=0)
        action = retry_action(used, attempt.changes_statement, budget)
        attempt.action = action.value
        log.info(
            "Section %d: STT %d deviations, %d change the statement -> %s",
            section.index,
            len(attempt.deviations),
            len(counting),
            action.value,
        )
        if action is RetryAction.ESCALATE:
            details = "; ".join(
                f"'{d.expected}' heard as '{d.heard}' ({d.reason})" for d in counting
            )
            raise AudioStepError(
                f"STT verification: section {section.index} still changes the statement after "
                f"{budget} resyntheses: {details}"
            )
        if action is RetryAction.ACCEPT:
            piece.y = y
            n += 1
            continue
        if offsets is None or times is None:
            cuts: list[int | None] = [None] * (len(piece_turns) - 1)
        else:
            cuts = boundary_cuts(times, offsets, y, rate, joins["edge_silence_frame_seconds"])
        pieces[n : n + 1] = _split(piece, y, cuts, blocks(affected, len(piece_turns), cuts))
    if len(pieces) == 1:
        out = pieces[0].y
    else:
        try:
            trimmed = [
                audio.trim_edges(
                    p.y,
                    rate,
                    joins["edge_silence_floor_db"],
                    joins["edge_silence_frame_seconds"],
                    joins["edge_silence_keep_seconds"],
                )
                for p in pieces
            ]
        except audio.AudioError as exc:
            raise AudioStepError(f"Section {section.index}: {exc}") from exc
        out = audio.join(trimmed, rate, joins["turn_pause_seconds"])
    audio.write(work_dir / section.file, out, rate)
    section.verified, section.seconds = True, round(audio.duration(out, rate), 2)
    return out


def _split(
    piece: _Piece,
    y: np.ndarray,
    cuts: Sequence[int | None],
    runs: Sequence[tuple[int, int]],
) -> list[_Piece]:
    """The piece split into kept audio and the runs to synthesise again (positions in the
    section). `cuts[k-1]` is the sample at the border before turn k of the piece."""
    out: list[_Piece] = []
    turn, sample = 0, 0
    for first, end in runs:
        if first > turn:
            cut = cuts[first - 1]
            out.append(_Piece(piece.first + turn, piece.first + first, y[sample:cut]))
        out.append(_Piece(piece.first + first, piece.first + end))
        turn = end
        sample = cuts[end - 1] if end < piece.end - piece.first else len(y)
    if turn < piece.end - piece.first:
        out.append(_Piece(piece.first + turn, piece.end, y[sample:]))
    return out


def _synthesize(
    section: SectionRecord,
    attempt: AttemptRecord,
    piece_turns: Sequence[tuple[str, str]],
    tts: Tts,
    lexicon: Sequence[Mapping],
    rate: int,
    prices: Mapping,
) -> np.ndarray:
    """One TTS request for the turns; usage goes into the attempt and the section."""
    spoken = [(speaker, respell(text, lexicon)) for speaker, text in piece_turns]
    try:
        result = tts.synthesize(spoken)
        y, sr = audio.decode(result.audio)
    except (TtsError, audio.AudioError) as exc:
        raise AudioStepError(f"TTS section {section.index}: {exc}") from exc
    attempt.input_tokens, attempt.output_tokens = result.input_tokens, result.output_tokens
    attempt.usd = cost_usd(result.input_tokens, result.output_tokens, prices)
    attempt.seconds = round(audio.duration(y, sr), 2)
    section.input_tokens = _sum([a.input_tokens for a in section.attempts])
    section.output_tokens = _sum([a.output_tokens for a in section.attempts])
    section.usd = _sum([a.usd for a in section.attempts], 4)
    if sr != rate:
        raise AudioStepError(
            f"TTS section {section.index} has {sr} Hz; intro and outro have {rate} Hz."
        )
    return y


def verify(
    y: np.ndarray,
    sr: int,
    section_turns: Sequence[tuple[str, str]],
    stt: Stt,
    decision_llm: StructuredLlm,
    lexicon: Sequence[Mapping],
    settings: Mapping,
) -> tuple[str, list[Deviation], list[Span | None] | None]:
    """Transcript of the audio, its deviations from the turns' script and the time of each
    normalised script word (`None` without word times). Raises `SttError`, `LlmError`.

    Every deviation is first heard a second time; only those the second hearing does not clear
    go to the LLM (one call), which decides whether they change the statement.
    """
    script = " ".join(text for _, text in section_turns)
    transcript = stt.transcribe(y, sr)
    expected = tokenize(script)
    heard = tokenize(canonicalize(transcript.text, script, lexicon))
    deviations = diff_tokens(expected, heard, settings["context_words"])
    times = None
    if transcript.words:
        times = script_times(expected, heard, token_times(heard, transcript.words))
        duration = audio.duration(y, sr)
        for d in deviations:
            first, last, t0, t1 = hearing_window(
                times,
                d.script_start,
                d.script_end,
                settings["second_hearing_context_words"],
                duration,
                settings["second_hearing_margin_seconds"],
            )
            again = stt.transcribe(y[round(t0 * sr) : round(t1 * sr)], sr)
            d.heard_again = again.text
            d.heard_again_matches = heard_again_matches(
                expected[first:last], tokenize(canonicalize(again.text, script, lexicon)), first, d
            )
    open_ = [d for d in deviations if d.heard_again_matches is not True]
    decide_deviations(decision_llm, section_turns, transcript.text, open_)
    return transcript.text, deviations, times


def new_record(config: Mapping) -> AudioRecord:
    return AudioRecord(
        config["models"]["tts"],
        voices_of(config),
        config["quotas"]["tts_section_max_chars"],
        config["models"]["stt"],
        config["quotas"]["audio_section_retries"],
    )
