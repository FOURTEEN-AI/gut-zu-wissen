import io

import numpy as np
import pytest
import soundfile as sf

from gut_zu_wissen import audio
from gut_zu_wissen.dialog import Turn
from gut_zu_wissen.episode_audio import (
    AudioStepError,
    cost_usd,
    new_record,
    produce_audio,
)
from gut_zu_wissen.llm import LlmError
from gut_zu_wissen.run import load_config
from gut_zu_wissen.stt import SttError, Transcript, Word
from gut_zu_wissen.tts import Synthesis

SR = 24_000
REAL = load_config()


def config(max_chars: int = 6000, retries: int = 2) -> dict:
    quotas = {
        **REAL["quotas"],
        "tts_section_max_chars": max_chars,
        "audio_section_retries": retries,
    }
    return {**REAL, "quotas": quotas}


def tone(seconds: float, sr: int = SR) -> np.ndarray:
    y = 0.05 * np.sin(2 * np.pi * 220 * np.arange(round(seconds * sr)) / sr)
    return np.concatenate([np.zeros(sr // 2), y, np.zeros(sr // 2)])


def wav(y: np.ndarray, sr: int = SR) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, y, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class FakeTts:
    def __init__(self, sr: int = SR) -> None:
        self.calls: list[list] = []
        self.sr = sr

    def synthesize(self, turns):
        self.calls.append(list(turns))
        return Synthesis(wav(tone(1.0, self.sr), self.sr), 10, 20)


class FakeStt:
    """Returns the queued transcripts in order; when the queue is empty, `default`."""

    def __init__(self, *transcripts: str, default: str | None = None) -> None:
        self.queue, self.default = list(transcripts), default
        self.calls: list[int] = []

    def transcribe(self, y, sr):
        self.calls.append(sr)
        if self.queue:
            return Transcript(self.queue.pop(0))
        if self.default is None:
            raise AssertionError("no transcript queued")
        return Transcript(self.default)


class EchoStt:
    """Hears exactly what the TTS was asked to say."""

    def __init__(self, tts: FakeTts) -> None:
        self.tts = tts

    def transcribe(self, y, sr):
        return Transcript(" ".join(text for _, text in self.tts.calls[-1]))


class TimedStt:
    """Hears the queued texts in order, with word times spread evenly over the audio."""

    def __init__(self, *texts: str) -> None:
        self.queue = list(texts)
        self.calls: list[float] = []  # seconds of audio per call

    def transcribe(self, y, sr, hint=None):
        seconds = len(y) / sr
        self.calls.append(seconds)
        self.hints = [*getattr(self, "hints", []), hint]
        text = self.queue.pop(0)
        step = seconds / max(1, len(text.split()))
        timed = [Word(w, n * step, (n + 0.8) * step) for n, w in enumerate(text.split())]
        return Transcript(text, timed)


class FakeLlm:
    """Decides every deviation with `changes`; or raises `error`."""

    def __init__(self, changes: bool = True, error: Exception | None = None) -> None:
        self.changes, self.error = changes, error
        self.calls: list[str] = []

    def complete_json(self, system, user, schema, **cache):
        self.calls.append(user)
        if self.error:
            raise self.error
        numbers = [
            int(line.split(".", 1)[0])
            for line in user.splitlines()
            if line[:1].isdigit() and ". Skript:" in line
        ]
        return {
            "decisions": [
                {"index": n, "changes_statement": self.changes, "reason": "Begründung."}
                for n in numbers
            ]
        }


def fixed() -> dict:
    return {"intro": (tone(0.5), SR, "i" * 64), "outro": (tone(0.5), SR, "o" * 64)}


def turns(*texts: str) -> list[Turn]:
    return [Turn(n, "Lena" if n % 2 == 0 else "Felix", t, []) for n, t in enumerate(texts)]


def produce(script, tts, work, cfg=None, stt=None, llm=None, words=frozenset()):
    record = new_record(cfg or config())
    stt = stt or EchoStt(tts)
    produce_audio(record, script, tts, stt, llm or FakeLlm(), fixed(), work, cfg or config(), words)
    return record


def test_sections_are_synthesised_joined_and_normalised(tmp_path):
    tts = FakeTts()
    script = turns("A" * 40, "B" * 40, "C" * 40, "D" * 40)
    record = produce(script, tts, tmp_path, config(max_chars=80))
    assert [s.turn_indices for s in record.sections] == [[0, 1], [2, 3]]
    assert tts.calls[1] == [("Lena", "C" * 40), ("Felix", "D" * 40)]
    assert [s.chars for s in record.sections] == [80, 80]
    out, sr = audio.read(tmp_path / "episode.flac")
    keep, pause = REAL["audio"]["edge_silence_keep_seconds"], REAL["audio"]["join_pause_seconds"]
    # intro 0.5 s, two sections of 1 s, outro 0.5 s; edges trimmed to `keep`; three pauses
    assert len(out) / sr == pytest.approx(3 + 4 * 2 * keep + 3 * pause, abs=0.05)
    assert record.seconds == pytest.approx(len(out) / sr, abs=0.01)
    assert abs(audio.integrated_loudness(out, sr) - REAL["quotas"]["target_loudness_lufs"]) <= 0.1
    assert audio.true_peak_dbtp(out, sr) <= REAL["quotas"]["true_peak_ceiling_dbtp"]
    assert (tmp_path / "sections" / "01.flac").exists()
    assert record.fixed_parts == {"intro": "i" * 64, "outro": "o" * 64}
    assert record.usd == pytest.approx(2 * cost_usd(10, 20, REAL["prices"]))


def test_markup_in_the_final_script_stops_before_tts(tmp_path):
    tts = FakeTts()
    with pytest.raises(AudioStepError, match="Markup gate: turn 1"):
        produce(turns("Sauber.", "Mit [[Link]]."), tts, tmp_path)
    assert tts.calls == []


def test_turn_too_long_for_a_section_stops_before_tts(tmp_path):
    tts = FakeTts()
    with pytest.raises(AudioStepError, match="Sections: Turn 0"):
        produce(turns("A" * 81), tts, tmp_path, config(max_chars=80))
    assert tts.calls == []


def test_sample_rate_mismatch_is_an_error(tmp_path):
    with pytest.raises(AudioStepError, match="48000 Hz"):
        produce(turns("Text."), FakeTts(sr=48_000), tmp_path)


def test_cost_is_unknown_with_placeholder_prices():
    assert cost_usd(10, 20, {"gemini_tts_input": "PLACEHOLDER", "gemini_tts_output": 9.0}) is None
    assert cost_usd(None, 20, REAL["prices"]) is None
    assert cost_usd(1_000_000, 1_000_000, REAL["prices"]) == 9.5


def test_matching_transcript_accepts_the_section_without_an_llm_call(tmp_path):
    tts, llm = FakeTts(), FakeLlm()
    stt = FakeStt("Es waren 324 Meter, echt!")
    record = produce(turns("Es waren dreihundertvierundzwanzig Meter.", "Echt?"), tts, tmp_path,
                     stt=stt, llm=llm)  # fmt: skip
    assert llm.calls == [] and len(tts.calls) == 1 and stt.calls == [SR]
    section = record.sections[0]
    assert section.verified and len(section.attempts) == 1
    attempt = section.attempts[0]
    assert (attempt.transcript, attempt.deviations) == ("Es waren 324 Meter, echt!", [])
    assert (attempt.changes_statement, attempt.action) == (False, "accept")


def test_harmless_deviation_is_accepted(tmp_path):
    tts, llm = FakeTts(), FakeLlm(changes=False)
    record = produce(turns("Der Eiffelturm steht in Paris."), tts, tmp_path,
                     stt=FakeStt("Der Eifelturm steht in Paris."), llm=llm)  # fmt: skip
    assert len(tts.calls) == 1 and len(llm.calls) == 1
    assert "Skript des Abschnitts:\nLena: Der Eiffelturm steht in Paris." in llm.calls[0]
    deviation = record.sections[0].attempts[0].deviations[0]
    assert (deviation.expected, deviation.heard) == ("Eiffelturm", "Eifelturm")
    assert deviation.changes_statement is False and record.sections[0].verified


def test_changed_statement_resynthesises_only_the_affected_section(tmp_path):
    tts = FakeTts()
    script = turns("A" * 40, "B" * 40, "C" * 40, "D" * 40)
    a, c = f"{'A' * 40} {'B' * 40}", f"{'C' * 40} {'D' * 40}"
    stt = FakeStt(a, "falsch", c)
    record = produce(script, tts, tmp_path, config(max_chars=80), stt=stt)
    assert [len(s.attempts) for s in record.sections] == [1, 2]
    assert tts.calls[1] == tts.calls[2] == [("Lena", "C" * 40), ("Felix", "D" * 40)]
    first, second = record.sections[1].attempts
    assert (first.transcript, first.changes_statement, first.action) == (
        "falsch",
        True,
        "resynthesize",
    )
    assert first.deviations[0].reason == "Begründung."  # discarded synthesis stays archived
    assert (second.attempt, second.action) == (1, "accept")
    assert record.sections[1].verified
    # cost and tokens of both syntheses count; duration is that of the accepted audio
    assert record.sections[1].input_tokens == 20 and record.sections[1].output_tokens == 40
    assert record.sections[1].usd == pytest.approx(2 * cost_usd(10, 20, REAL["prices"]))
    assert record.usd == pytest.approx(3 * cost_usd(10, 20, REAL["prices"]))
    assert record.sections[1].seconds == second.seconds
    assert (tmp_path / "episode.flac").exists()


def test_budget_used_up_escalates(tmp_path):
    tts = FakeTts()
    with pytest.raises(AudioStepError, match=r"section 0 still changes the statement after 2 "
                       r"resyntheses: 'neun' heard as 'zehn' \(Begründung\.\)"):  # fmt: skip
        record = new_record(config())
        produce_audio(record, turns("Es waren neun."), tts, FakeStt(default="Es waren zehn."),
                      FakeLlm(), fixed(), tmp_path, config())  # fmt: skip
    assert len(tts.calls) == 3
    assert [a.action for a in record.sections[0].attempts] == [
        "resynthesize", "resynthesize", "escalate"
    ]  # fmt: skip
    assert not record.sections[0].verified and record.file is None


def test_budget_comes_from_the_config(tmp_path):
    tts = FakeTts()
    with pytest.raises(AudioStepError, match="after 0 resyntheses"):
        produce(turns("Eins."), tts, tmp_path, config(retries=0), stt=FakeStt(default="Zwei."))
    assert len(tts.calls) == 1


def test_stt_failure_is_an_audio_step_error(tmp_path):
    class Broken:
        def transcribe(self, y, sr):
            raise SttError("model missing")

    with pytest.raises(AudioStepError, match="STT verification, section 0: model missing"):
        produce(turns("Text."), FakeTts(), tmp_path, stt=Broken())


def test_llm_failure_is_an_audio_step_error(tmp_path):
    with pytest.raises(AudioStepError, match="STT verification, section 0: cut off"):
        produce(turns("Text."), FakeTts(), tmp_path, stt=FakeStt("Anders."),
                llm=FakeLlm(error=LlmError("cut off")))  # fmt: skip


SCRIPT = "Was hing dort eigentlich an den Wänden der Galerie?"


def test_second_hearing_that_matches_clears_the_deviation(tmp_path):
    tts, llm = FakeTts(), FakeLlm()
    stt = TimedStt("Was hindert eigentlich an den Wänden der Galerie?", "Was hing dort eigentlich")
    record = produce(turns(SCRIPT), tts, tmp_path, stt=stt, llm=llm)
    assert len(tts.calls) == 1 and len(stt.calls) == 2
    assert stt.calls[1] < stt.calls[0]  # only a stretch of the audio is heard again
    assert llm.calls == []  # a cleared deviation does not go to the LLM
    (attempt,) = record.sections[0].attempts
    (d,) = attempt.deviations
    assert (d.expected, d.heard, d.changes_statement) == ("hing dort", "hindert", None)
    assert (d.heard_again, d.heard_again_matches) == ("Was hing dort eigentlich", True)
    assert (attempt.changes_statement, attempt.action) == (False, "accept")


def test_second_hearing_that_hears_the_same_leads_to_resynthesis(tmp_path):
    tts = FakeTts()
    stt = TimedStt(
        "Was hindert eigentlich an den Wänden der Galerie?",
        "Was hindert eigentlich",
        SCRIPT,
    )
    record = produce(turns(SCRIPT), tts, tmp_path, stt=stt)
    first, second = record.sections[0].attempts
    assert first.deviations[0].heard_again_matches is False and first.action == "resynthesize"
    assert second.action == "accept" and len(tts.calls) == 2


def test_only_deviations_the_second_hearing_confirms_go_to_the_llm(tmp_path):
    script = "Was hing dort eigentlich an den Wänden der Galerie in New York?"
    stt = TimedStt(
        "Was hindert eigentlich an den Wänden der Gallerie in New York?",
        "Was hing dort eigentlich an",  # first deviation: cleared
        "Wänden der Gallerie in New",  # second deviation: heard the same
    )
    llm = FakeLlm(changes=False)
    record = produce(turns(script), FakeTts(), tmp_path, stt=stt, llm=llm)
    first, second = record.sections[0].attempts[0].deviations
    assert (first.heard_again_matches, first.changes_statement) == (True, None)
    assert (second.heard_again_matches, second.changes_statement) == (False, False)
    (call,) = llm.calls
    assert "1. Skript:" in call and "0. Skript:" not in call
    assert record.sections[0].attempts[0].action == "accept"


def test_only_the_affected_turn_is_synthesised_again(tmp_path):
    tts = FakeTts()
    script = turns("Es waren neun Leute.", "Wirklich neun?", "Ja, genau neun.")
    stt = TimedStt(
        "Es waren neun Leute. Wirklich zehn? Ja, genau neun.",
        "Leute. Wirklich zehn? Ja, genau",  # second hearing: the same
        "Wirklich neun?",
    )
    record = produce(script, tts, tmp_path, stt=stt)
    section = record.sections[0]
    assert tts.calls[1] == [("Felix", "Wirklich neun?")]
    first, second = section.attempts
    assert (first.turn_indices, first.action) == ([0, 1, 2], "resynthesize")
    assert (second.turn_indices, second.action) == ([1], "accept")
    assert section.verified and section.input_tokens == 20
    out, sr = audio.read(tmp_path / "sections" / "00.flac")
    assert section.seconds == pytest.approx(len(out) / sr, abs=0.01)
    # turn 0 and turn 2 come from the first synthesis, turn 1 from the second (2 s with edges):
    # longer than one synthesis, shorter than two
    assert 2.0 < section.seconds < 4.0


def test_turn_budget_escalates(tmp_path):
    tts = FakeTts()
    script = turns("Es waren neun Leute.", "Wirklich neun?")
    stt = TimedStt(
        "Es waren neun Leute. Wirklich zehn?",
        "Leute. Wirklich zehn?",
        "Wirklich zehn?",
        "Wirklich zehn?",
        "Wirklich zehn?",
        "Wirklich zehn?",
    )
    with pytest.raises(AudioStepError, match="after 2 resyntheses: 'neun' heard as 'zehn'"):
        produce(script, tts, tmp_path, stt=stt)
    assert tts.calls[1:] == [[("Felix", "Wirklich neun?")]] * 2


def test_without_word_times_the_whole_section_is_synthesised_again(tmp_path):
    tts = FakeTts()
    script = turns("Es waren neun Leute.", "Wirklich neun?")
    stt = FakeStt("Es waren neun Leute. Wirklich zehn?", "Es waren neun Leute. Wirklich neun?")
    record = produce(script, tts, tmp_path, stt=stt)
    assert tts.calls[1] == tts.calls[0]
    assert [a.turn_indices for a in record.sections[0].attempts] == [[0, 1], [0, 1]]


# ---------------------------------------------------------------- third hearing with a hint

DASSEL = "Der Erzbischof Rainald von Dassel kam nach Köln."
DASSEL_WORDS = frozenset({"erzbischof", "rainald", "von", "dassel"})
MISHEARD = "Der Erzbischof Reinhard von Dassel kam nach Köln."


def test_hint_hearing_that_matches_clears_the_deviation_and_adds_the_word(tmp_path):
    tts = FakeTts()
    stt = TimedStt(MISHEARD, "Erzbischof Reinhard von Dassel", "Erzbischof Rainald von Dassel")
    record = produce(turns(DASSEL), tts, tmp_path, stt=stt, words=DASSEL_WORDS)
    (attempt,) = record.sections[0].attempts
    (d,) = attempt.deviations
    assert stt.hints == [None, None, "Rainald"]
    assert (d.heard_with_hint, d.hint_matches) == ("Erzbischof Rainald von Dassel", True)
    assert attempt.action == "accept" and len(tts.calls) == 1
    (entry,) = record.lexicon_additions
    assert (entry["written"], entry["spoken"], entry["heard"]) == (
        "Rainald",
        "Rainald",
        ["Reinhard"],
    )


def test_hint_hearing_that_still_hears_the_other_word_leads_to_resynthesis(tmp_path):
    tts = FakeTts()
    stt = TimedStt(
        MISHEARD,
        "Erzbischof Reinhard von Dassel",
        "Erzbischof Reinhard von Dassel",  # with the hint: still the other word
        DASSEL,
    )
    record = produce(turns(DASSEL), tts, tmp_path, stt=stt, words=DASSEL_WORDS)
    first, second = record.sections[0].attempts
    assert first.deviations[0].hint_matches is False and first.action == "resynthesize"
    assert second.action == "accept" and record.lexicon_additions == []


def test_no_hint_hearing_for_a_word_outside_the_fact_sentences(tmp_path):
    stt = TimedStt(MISHEARD, "Erzbischof Reinhard von Dassel", DASSEL)
    record = produce(turns(DASSEL), FakeTts(), tmp_path, stt=stt, words=frozenset({"dassel"}))
    first, _ = record.sections[0].attempts
    assert first.deviations[0].heard_with_hint is None and None in stt.hints
    assert "Rainald" not in stt.hints


def test_no_hint_hearing_once_the_episode_has_its_maximum_of_words(tmp_path):
    cfg = config()
    cfg = {**cfg, "stt": {**cfg["stt"], "hint_max_entries": 0}}
    stt = TimedStt(MISHEARD, "Erzbischof Reinhard von Dassel", DASSEL)
    record = produce(turns(DASSEL), FakeTts(), tmp_path, cfg, stt=stt, words=DASSEL_WORDS)
    assert "Rainald" not in stt.hints and record.lexicon_additions == []


def test_a_word_added_in_one_section_counts_in_the_next(tmp_path):
    cfg = config(max_chars=len(DASSEL))
    stt = TimedStt(
        MISHEARD, "Erzbischof Reinhard von Dassel", "Erzbischof Rainald von Dassel",
        MISHEARD,  # second section: the lexicon now turns Reinhard into Rainald
    )  # fmt: skip
    record = produce(turns(DASSEL, DASSEL), FakeTts(), tmp_path, cfg, stt=stt, words=DASSEL_WORDS)
    assert [len(s.attempts) for s in record.sections] == [1, 1]
    assert record.sections[1].attempts[0].deviations == []
