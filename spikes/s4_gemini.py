"""Spike S4: Gemini 3.8 Flash TTS for a two-speaker German dialog.

Questions: which two voices (Lena, Felix)? Which model (flash vs flash-lite)? How long may a
section be (output limit 16,384 tokens per call)? Are the seams between sections audible?

I cannot listen, so every claim rests on measurements: Whisper word error rate (intelligibility),
Praat pitch (speaker separation, drift), pyloudnorm loudness (seams), audio-token cost.

Usage (GEMINI_API_KEY in the environment, WAVs go to --out, results JSON to RESULTS):
  uv run --group spikes python spikes/s4_gemini.py voices   --out DIR
  uv run --group spikes python spikes/s4_gemini.py models   --out DIR
  uv run --group spikes python spikes/s4_gemini.py limit    --out DIR
  uv run --group spikes python spikes/s4_gemini.py episode  --out DIR
"""

import argparse
import io
import json
import os
import re
import time
from pathlib import Path

import httpx
import numpy as np
import parselmouth
import pyloudnorm
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "wikipedia" / "spikes"
RESULTS = FIXTURES / "s4_results"
SCRIPT = FIXTURES / "s4_script.json"
URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
FLASH, LITE = "gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"
SR = 24_000
# Output price, USD per million audio tokens through 2026-12-31 (config/pipeline.toml).
PRICE_OUT = {FLASH: 9.00, LITE: 6.00}
PRICE_IN = 0.50

# Voice-test text: German, no numerals, a mix of long and short sentences and a question.
SAMPLE_TEXT = (
    "Ein Anwohner hatte Angst, der Turm könnte einstürzen und sein Haus zerstören. Das Gericht "
    "erlaubte den Weiterbau nur mit einer Auflage. Der Ingenieur musste bei möglichen Schäden "
    "haften. Und was passierte, als die ersten Besucher kamen? Zunächst gab es nur Ärger, aber "
    "dann wurde der Turm zum beliebtesten Ausflugsziel der ganzen Stadt."
)


# ---------------------------------------------------------------- API


def api_key() -> str:
    return os.environ["GEMINI_API_KEY"]


def synth(
    model: str,
    speakers: list[tuple[str, str]],
    turns: list[tuple[str, str]],
    style: dict[str, str] | None = None,
) -> dict:
    """One interaction. speakers = [(name, voice)]; one speaker uses the single-voice form."""
    content = []
    for who, text in turns:
        ann = {"type": "speech_metadata", "speaker": who}
        if style and who in style:
            ann["style"] = style[who]
        content.append({"type": "text", "text": text, "annotations": [ann]})
    if len(speakers) == 1:
        speech = [{"voice": speakers[0][1]}]
        for c in content:
            c["annotations"] = [a for a in c["annotations"] if a.get("style")] or []
            if not c["annotations"]:
                del c["annotations"]
    else:
        speech = {"speakers": [{"speaker": n, "voice": v} for n, v in speakers]}
    body = {
        "model": model,
        "input": [{"type": "user_input", "content": content}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": speech},
    }
    headers = {"x-goog-api-key": api_key(), "Content-Type": "application/json"}
    for attempt in range(5):
        started = time.perf_counter()
        r = httpx.post(URL, headers=headers, json=body, timeout=900)
        secs = time.perf_counter() - started
        if r.status_code in (429, 500, 503, 529) and attempt < 4:
            # 429 body: "... Please retry in 25s ..." (limit: 10 requests per minute, Tier 1).
            m = re.search(r"retry in (\d+(?:\.\d+)?)s", r.text)
            time.sleep(float(m.group(1)) + 2 if m else 2 ** (attempt + 1))
            continue
        break
    out = {"status": r.status_code, "seconds": round(secs, 2), "request": body}
    try:
        data = r.json()
    except ValueError:
        out["error"] = r.text[:500]
        return out
    if r.status_code != 200:
        out["error"] = data
        return out
    out["usage"] = data.get("usage")
    out["finish"] = data.get("status")
    import base64

    audio = [
        c
        for step in data.get("steps", [])
        for c in step.get("content", [])
        if c.get("type") == "audio"
    ]
    out["audio_parts"] = len(audio)
    if audio:
        out["mime"] = audio[0].get("mime_type")
        out["wav"] = base64.b64decode(audio[0]["data"])
    return out


def to_array(wav: bytes) -> np.ndarray:
    y, sr = sf.read(io.BytesIO(wav), dtype="float32")
    assert sr == SR, sr
    return y if y.ndim == 1 else y.mean(axis=1)


def cost(model: str, usage: dict) -> float:
    return (
        usage["total_input_tokens"] * PRICE_IN + usage["total_output_tokens"] * PRICE_OUT[model]
    ) / 1e6


# ---------------------------------------------------------------- analysis


def loudness(y: np.ndarray) -> float:
    return float(pyloudnorm.Meter(SR).integrated_loudness(y))


def edges_silence(y: np.ndarray, floor_db: float = -45.0) -> tuple[float, float]:
    """Leading and trailing silence in seconds (10 ms RMS frames below floor_db)."""
    hop = SR // 100
    n = len(y) // hop
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms) > floor_db)[0]
    if len(loud) == 0:
        return len(y) / SR, 0.0
    return loud[0] * 0.01, (n - 1 - loud[-1]) * 0.01


def pitch_track(y: np.ndarray) -> np.ndarray:
    """Voiced F0 values in Hz (Praat autocorrelation, 10 ms step)."""
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    f0 = snd.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=450).selected_array["frequency"]
    return f0[f0 > 0]


def semitone_spread(f0: np.ndarray) -> float:
    """Standard deviation of F0 in semitones around its median (liveliness of intonation)."""
    return float(np.std(12 * np.log2(f0 / np.median(f0)))) if len(f0) > 10 else 0.0


# Word error rate with numerals removed on both sides (Whisper writes digits, the script
# spells numbers as spoken; S5 handles numeral normalisation properly).
NUM_STEMS = (
    "null",
    "ein",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
    "zehn",
    "elf",
    "zwölf",
    "zig",
    "ßig",
    "hundert",
    "tausend",
    "million",
)
NUM_PART = re.compile(
    r"^(?:null|ein|eins|eine|einen|einem|einer|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|"
    r"zwölf|zwan|dreiß|vier|fünf|sech|sieb|acht|neun|zig|ßig|hundert|tausend|million|millionen|und|"
    r"ten|ste|sten|ster|te|ig|zehn|ß)+$"
)


def words(text: str) -> list[str]:
    text = text.lower().replace("ß", "ss")
    toks = re.findall(r"[a-zäöüé]+|\d+", text)
    out = []
    for t in toks:
        if t.isdigit():
            continue
        if len(t) >= 4 and any(s in t for s in NUM_STEMS) and NUM_PART.match(t.replace("ss", "ß")):
            continue
        out.append(t)
    return out


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev = cur
    return d[-1] / max(len(r), 1)


_whisper = None


def transcribe(y: np.ndarray, size: str = "small") -> str:
    global _whisper
    from faster_whisper import WhisperModel

    if _whisper is None:
        _whisper = WhisperModel(size, device="cpu", compute_type="int8")
    # Plain defaults hallucinate repeated text after the speech ends (seen in this spike):
    # VAD and no conditioning on previous text avoid most of it.
    segs, _ = _whisper.transcribe(
        y.astype(np.float32),
        language="de",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return " ".join(s.text.strip() for s in segs)


def save(out: Path | None, name: str, y: np.ndarray) -> None:
    if out:
        out.mkdir(parents=True, exist_ok=True)
        sf.write(out / name, y, SR, subtype="PCM_16")


def write_results(name: str, data: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


# ---------------------------------------------------------------- experiments


def load_script() -> dict:
    return json.loads(SCRIPT.read_text())


def candidate_voices() -> dict[str, list[str]]:
    """Persona profiles (docs/persona.md): Lena female, medium pitch, calm, clear;
    Felix male, more energy, clearly distinguishable from Lena."""
    voices = json.loads((RESULTS / "voices_de_library.json").read_text())
    by_id = {v["id"]: v for v in voices}
    lena = [
        "de-de-advisor-3",
        "de-de-tutor-2",
        "de-de-techagent-1",
        "de-de-podcaster-2",
        "de-de-storyteller-1",
        "de-de-tutor-8",
        "de-de-concierge-4",
        "de-de-csagent-7",
    ]
    felix = [
        "de-de-podcaster-3",
        "de-de-podcaster-1",
        "de-de-csagent-11",
        "de-de-tutor-4",
        "de-de-assistant-10",
        "de-de-concierge-9",
        "de-de-techagent-3",
        "de-de-assistant-3",
    ]
    named_f = ["Kore", "Leda", "Aoede", "Despina", "Achernar", "Sulafat"]
    named_m = ["Puck", "Charon", "Fenrir", "Orus", "Iapetus", "Achird", "Sadachbia"]
    for v in lena + felix:
        assert v in by_id, v
    return {"lena": lena + named_f, "felix": felix + named_m}


def exp_voices(out: Path | None) -> None:
    from concurrent.futures import ThreadPoolExecutor

    path = RESULTS / "voices.json"
    done = (
        {r["voice"]: r for r in json.loads(path.read_text())["rows"] if "error" not in r}
        if path.exists()
        else {}
    )
    jobs = [
        (role, voice)
        for role, ids in candidate_voices().items()
        for voice in ids
        if voice not in done
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda j: synth(LITE, [("Sprecher", j[1])], [("Sprecher", SAMPLE_TEXT)]), jobs)
        )
    rows = list(done.values())
    for (role, voice), r in zip(jobs, results, strict=True):
        if r["status"] != 200 or "wav" not in r:
            rows.append({"role": role, "voice": voice, "error": r.get("error")})
            print(role, voice, "ERROR", str(r.get("error"))[:100])
            continue
        y = to_array(r["wav"])
        f0 = pitch_track(y)
        started = time.perf_counter()
        hyp = transcribe(y)
        stt_s = time.perf_counter() - started
        lead, trail = edges_silence(y)
        row = {
            "role": role,
            "voice": voice,
            "seconds_audio": round(len(y) / SR, 2),
            "words_per_second": round(len(SAMPLE_TEXT.split()) / (len(y) / SR), 2),
            "f0_median_hz": round(float(np.median(f0)), 1),
            "f0_spread_semitones": round(semitone_spread(f0), 2),
            "loudness_lufs": round(loudness(y), 1),
            "wer": round(wer(SAMPLE_TEXT, hyp), 3),
            "lead_trail_silence_s": [round(lead, 2), round(trail, 2)],
            "hyp": hyp,
        }
        rows.append(row)
        save(out, f"voice_{voice}.wav", y)
        print(
            f"{role:5} {voice:22} f0={row['f0_median_hz']:6} sp={row['f0_spread_semitones']:4} "
            f"wps={row['words_per_second']:4} wer={row['wer']:.3f} lufs={row['loudness_lufs']} "
            f"stt={stt_s:.1f}s",
            flush=True,
        )
    write_results("voices.json", {"sample_text": SAMPLE_TEXT, "model": LITE, "rows": rows})


def dialog_turns(script: dict, repeat: int = 1) -> list[tuple[str, str]]:
    return [tuple(t) for _ in range(repeat) for t in script["turns"]]


def speaker_masks(y: np.ndarray, split_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Voiced F0 split into 'lower voice' and 'higher voice' around split_hz."""
    f0 = pitch_track(y)
    return f0[f0 < split_hz], f0[f0 >= split_hz]


STYLE = {"Lena": "calm, clear", "Felix": "curious, energetic"}


def exp_models(out: Path | None, lena: str, felix: str) -> None:
    """Flash vs Flash-Lite, each twice, plus Flash-Lite with speech_metadata style."""
    script = load_script()
    turns = dialog_turns(script)
    text = " ".join(t for _, t in turns)
    solo = {r["voice"]: r for r in json.loads((RESULTS / "voices.json").read_text())["rows"]}
    split_hz = float(np.sqrt(solo[lena]["f0_median_hz"] * solo[felix]["f0_median_hz"]))
    rows = []
    for model, style in ((FLASH, None), (LITE, None), (LITE, STYLE)):
        for rep in range(2):
            r = synth(model, [("Lena", lena), ("Felix", felix)], turns, style)
            label = f"{model}{'+style' if style else ''}"
            if r["status"] != 200 or "wav" not in r:
                rows.append({"variant": label, "rep": rep, "error": r.get("error")})
                print(label, rep, "ERROR", str(r.get("error"))[:150], flush=True)
                continue
            y = to_array(r["wav"])
            hyp = transcribe(y)
            st = speaker_stats(y, split_hz)
            row = {
                "variant": label,
                "rep": rep,
                "seconds_audio": round(len(y) / SR, 1),
                "request_seconds": r["seconds"],
                "output_tokens": r["usage"]["total_output_tokens"],
                "tokens_per_audio_second": round(
                    r["usage"]["total_output_tokens"] / (len(y) / SR), 2
                ),
                "usd": round(cost(model, r["usage"]), 5),
                "wer": round(wer(text, hyp), 3),
                "loudness_lufs": round(loudness(y), 1),
                "chars_per_audio_second": round(len(text) / (len(y) / SR), 2),
                "level_difference_db": round(st["lena"]["level_db"] - st["felix"]["level_db"], 1),
                "lena_spread": st["lena"]["f0_spread_semitones"],
                "felix_spread": st["felix"]["f0_spread_semitones"],
                "f0_separation_semitones": semitones(
                    st["felix"]["f0_median_hz"], st["lena"]["f0_median_hz"]
                ),
            }
            rows.append(row)
            save(out, f"models_{label}_{rep}.wav", y)
            print(json.dumps(row), flush=True)
    write_results("models.json", {"lena": lena, "felix": felix, "style": STYLE, "rows": rows})


def split_sections(turns: list[tuple[str, str]], max_chars: int) -> list[list[tuple[str, str]]]:
    """Balanced split at turn boundaries into the fewest sections of at most max_chars each."""
    total = sum(len(t) for _, t in turns)
    n = max(1, -(-total // max_chars))
    while True:
        target = total / n
        sections, current, size = [], [], 0
        for turn in turns:
            if current and size + len(turn[1]) / 2 > target and len(sections) < n - 1:
                sections.append(current)
                current, size = [], 0
            current.append(turn)
            size += len(turn[1])
        sections.append(current)
        if all(sum(len(t) for _, t in sec) <= max_chars for sec in sections) or n > 12:
            return sections
        n += 1


def window_stats(y: np.ndarray, split_hz: float, win: float = 30.0) -> list[dict]:
    """Per window of `win` seconds: integrated loudness and median F0 of the lower and
    higher voice (split at split_hz)."""
    out = []
    step = int(win * SR)
    for i in range(0, len(y) - step // 2, step):
        w = y[i : i + step]
        if len(w) < step // 2:
            break
        low, high = speaker_masks(w, split_hz)
        out.append(
            {
                "t": round(i / SR),
                "lufs": round(loudness(w), 1),
                "f0_low": round(float(np.median(low)), 1) if len(low) > 20 else None,
                "f0_high": round(float(np.median(high)), 1) if len(high) > 20 else None,
            }
        )
    return out


def semitones(a: float | None, b: float | None) -> float | None:
    return None if not a or not b else round(12 * np.log2(b / a), 2)


def episode_turns(script: dict, target_chars: int) -> list[tuple[str, str]]:
    base = dialog_turns(script)
    turns, size = [], 0
    while size < target_chars:
        for t in base:
            turns.append(t)
            size += len(t[1])
    return turns


def statistics_mean(xs: list[float]) -> float:
    return float(np.mean(xs))


def exp_limit(out: Path | None, lena: str, felix: str, model: str) -> None:
    """How does the API behave at and beyond the output limit?"""
    script = load_script()
    rows = []
    cases = (("about_11_min", 11_500), ("about_14_min", 15_000), ("input_over_8192_tokens", 36_000))
    for label, chars in cases:
        turns = episode_turns(script, chars)
        text = " ".join(t for _, t in turns)
        r = synth(model, [("Lena", lena), ("Felix", felix)], turns)
        row = {
            "case": label,
            "chars": len(text),
            "status": r["status"],
            "request_seconds": r["seconds"],
        }
        if r["status"] == 200 and "wav" in r:
            y = to_array(r["wav"])
            hyp = transcribe(y)
            row.update(
                usage=r["usage"],
                finish=r.get("finish"),
                seconds_audio=round(len(y) / SR, 1),
                chars_per_audio_second=round(len(text) / (len(y) / SR), 2),
                wer=round(wer(text, hyp), 3),
                last_words_ref=" ".join(text.split()[-8:]),
                last_words_hyp=" ".join(hyp.split()[-8:]),
            )
            save(out, f"limit_{label}.wav", y)
        else:
            row["error"] = r.get("error")
        rows.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "usage"}, ensure_ascii=False)[:700],
            flush=True,
        )
    write_results("limit.json", {"model": model, "rows": rows})


def exp_episode(out: Path | None, lena: str, felix: str, model: str, max_chars: int) -> None:
    """Full-length episode in sections; measure the seams."""
    script = load_script()
    turns = episode_turns(script, 15_000)
    text = " ".join(t for _, t in turns)
    solo = {r["voice"]: r for r in json.loads((RESULTS / "voices.json").read_text())["rows"]}
    split_hz = float(np.sqrt(solo[lena]["f0_median_hz"] * solo[felix]["f0_median_hz"]))
    sections = split_sections(turns, max_chars)
    print(
        f"{len(turns)} turns, {len(text)} chars -> {len(sections)} sections "
        f"{[sum(len(t) for _, t in sec) for sec in sections]}, split at {split_hz:.0f} Hz",
        flush=True,
    )
    audio, rows = [], []
    for k, sec in enumerate(sections):
        r = synth(model, [("Lena", lena), ("Felix", felix)], sec)
        if r["status"] != 200 or "wav" not in r:
            raise SystemExit(f"section {k} failed: {r.get('error')}")
        y = to_array(r["wav"])
        audio.append(y)
        save(out, f"episode_{len(sections)}sec_{k}.wav", y)
        st = speaker_stats(y, split_hz)
        lead, trail = edges_silence(y)
        sec_text = " ".join(t for _, t in sec)
        rows.append(
            {
                "section": k,
                "chars": len(sec_text),
                "seconds_audio": round(len(y) / SR, 1),
                "output_tokens": r["usage"]["total_output_tokens"],
                "request_seconds": r["seconds"],
                "usd": round(cost(model, r["usage"]), 4),
                "lufs": round(loudness(y), 1),
                "f0_lena_hz": st["lena"]["f0_median_hz"],
                "f0_felix_hz": st["felix"]["f0_median_hz"],
                "level_difference_db": round(st["lena"]["level_db"] - st["felix"]["level_db"], 1),
                "chars_per_second": round(len(sec_text) / (len(y) / SR), 2),
                "lead_trail_silence_s": [round(lead, 2), round(trail, 2)],
                "windows": window_stats(y, split_hz),
            }
        )
        print({k_: v for k_, v in rows[-1].items() if k_ != "windows"}, flush=True)
    seams = []
    for k in range(len(rows) - 1):
        a, b = rows[k], rows[k + 1]
        wa, wb = a["windows"][-2:], b["windows"][:2]
        seams.append(
            {
                "between": [k, k + 1],
                "lufs_step": round(b["lufs"] - a["lufs"], 1),
                "lena_semitones": semitones(a["f0_lena_hz"], b["f0_lena_hz"]),
                "felix_semitones": semitones(a["f0_felix_hz"], b["f0_felix_hz"]),
                "rate_change_percent": round(
                    100 * (b["chars_per_second"] / a["chars_per_second"] - 1), 1
                ),
                "lufs_step_edge_windows": round(
                    statistics_mean([w["lufs"] for w in wb])
                    - statistics_mean([w["lufs"] for w in wa]),
                    1,
                ),
            }
        )
    # Variation inside sections, for scale: spread of 30 s window loudness and F0.
    ref_lena, ref_felix = rows[0]["f0_lena_hz"], rows[0]["f0_felix_hz"]
    inner = {
        "window_lufs_std": round(float(np.std([w["lufs"] for r in rows for w in r["windows"]])), 2),
        "window_lena_f0_std_semitones": round(
            float(
                np.std(
                    [
                        12 * np.log2(w["f0_high"] / ref_lena)
                        for r in rows
                        for w in r["windows"]
                        if w["f0_high"]
                    ]
                )
            ),
            2,
        ),
        "window_felix_f0_std_semitones": round(
            float(
                np.std(
                    [
                        12 * np.log2(w["f0_low"] / ref_felix)
                        for r in rows
                        for w in r["windows"]
                        if w["f0_low"]
                    ]
                )
            ),
            2,
        ),
    }
    # Mitigation: normalise every section to the target loudness, join with a pause.
    meter = pyloudnorm.Meter(SR)
    pause = np.zeros(int(0.5 * SR), dtype=np.float32)
    normed = [
        pyloudnorm.normalize.loudness(y, meter.integrated_loudness(y), -16.0).astype(np.float32)
        for y in audio
    ]
    joined = normed[0]
    for y in normed[1:]:
        joined = np.concatenate([joined, pause, y])
    joined = np.clip(joined, -1.0, 1.0)
    save(out, f"episode_{len(sections)}sec_joined.wav", joined)
    hyp = transcribe(joined)
    summary = {
        "model": model,
        "sections": len(sections),
        "max_chars": max_chars,
        "total_seconds": round(len(joined) / SR, 1),
        "total_usd": round(sum(r["usd"] for r in rows), 4),
        "joined_lufs": round(loudness(joined), 1),
        "wer_joined": round(wer(text, hyp), 3),
        "inner_variation": inner,
        "seams": seams,
        "rows": rows,
    }
    write_results(f"episode_{len(sections)}sections.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
    if out and len(sections) > 1:
        cut = len(normed[0])
        save(out, f"seam_excerpt_{len(sections)}sec.wav", joined[cut - 12 * SR : cut + 13 * SR])


def speaker_stats(y: np.ndarray, split_hz: float) -> dict:
    """Per speaker (voiced frames split by pitch band): median F0, F0 spread, level in dB."""
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=70, pitch_ceiling=450)
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    half = int(0.0125 * SR)
    out = {}
    for name, sel in (("felix", (f0 > 0) & (f0 < split_hz)), ("lena", f0 >= split_hz)):
        rms = [
            float(np.sqrt(np.mean(y[max(int(t * SR) - half, 0) : int(t * SR) + half] ** 2) + 1e-12))
            for t in times[sel]
        ]
        out[name] = {
            "f0_median_hz": round(float(np.median(f0[sel])), 1),
            "f0_spread_semitones": round(semitone_spread(f0[sel]), 2),
            "level_db": round(20 * np.log10(np.mean(rms)), 1),
            "voiced_seconds": round(float(sel.sum() * 0.01), 1),
        }
    return out


PAIRS = {
    "A_advisor3_tutor4": ("de-de-advisor-3", "de-de-tutor-4"),
    "B_Sulafat_Puck": ("Sulafat", "Puck"),
    "C_tutor2_concierge9": ("de-de-tutor-2", "de-de-concierge-9"),
    "D_Despina_assistant10": ("Despina", "de-de-assistant-10"),
}


def exp_pairs(out: Path | None) -> None:
    script = load_script()
    turns = dialog_turns(script)
    text = " ".join(t for _, t in turns)
    solo = {r["voice"]: r for r in json.loads((RESULTS / "voices.json").read_text())["rows"]}
    rows = []
    for name, (lena, felix) in PAIRS.items():
        split_hz = float(np.sqrt(solo[lena]["f0_median_hz"] * solo[felix]["f0_median_hz"]))
        r = synth(LITE, [("Lena", lena), ("Felix", felix)], turns)
        if r["status"] != 200 or "wav" not in r:
            rows.append({"pair": name, "error": r.get("error")})
            print(name, "ERROR", str(r.get("error"))[:150])
            continue
        y = to_array(r["wav"])
        hyp = transcribe(y)
        st = speaker_stats(y, split_hz)
        row = {
            "pair": name,
            "lena": lena,
            "felix": felix,
            "seconds_audio": round(len(y) / SR, 1),
            "wer": round(wer(text, hyp), 3),
            "loudness_lufs": round(loudness(y), 1),
            "f0_separation_semitones": semitones(
                st["felix"]["f0_median_hz"], st["lena"]["f0_median_hz"]
            ),
            "level_difference_db": round(st["lena"]["level_db"] - st["felix"]["level_db"], 1),
            "speakers": st,
            "chars_per_second": round(len(text) / (len(y) / SR), 2),
            "usd": round(cost(LITE, r["usage"]), 4),
        }
        rows.append(row)
        save(out, f"pair_{name}.wav", y)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    write_results("pairs.json", {"model": LITE, "rows": rows})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["voices", "pairs", "models", "limit", "episode"])
    p.add_argument("--out", type=Path)
    p.add_argument("--lena", default="")
    p.add_argument("--felix", default="")
    p.add_argument("--model", default=LITE)
    p.add_argument("--max-chars", type=int, default=8000)
    a = p.parse_args()
    if a.experiment == "voices":
        exp_voices(a.out)
    elif a.experiment == "pairs":
        exp_pairs(a.out)
    elif a.experiment == "models":
        exp_models(a.out, a.lena, a.felix)
    elif a.experiment == "limit":
        exp_limit(a.out, a.lena, a.felix, a.model)
    else:
        exp_episode(a.out, a.lena, a.felix, a.model, a.max_chars)


if __name__ == "__main__":
    main()
