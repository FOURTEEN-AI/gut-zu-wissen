"""Spike S5: Whisper runtime on a CPU runner.

Question (plan, Phase 0): how long does Whisper take for 15 minutes of audio on a GitHub Actions
CPU runner, for faster-whisper with int8 versus larger models? Output: model size, cache setup.

Usage:
  # Fresh test audio: the S4 test dialog repeated to ~15 min, synthesized in sections.
  uv run --group spikes python spikes/s5_whisper.py synth OUT_DIR
  uv run --group spikes python spikes/s5_whisper.py run AUDIO.wav --models small,medium,large-v3 \
      --reference REFERENCE.txt --out results.json

Prints, per model: load time, transcription time, real-time factor (RTF = seconds spent per
second of audio), peak resident memory, and word error rate against the reference script
(numerals removed on both sides, as in S4).
"""

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from s4_gemini import FLASH, SR, episode_turns, load_script, split_sections, synth, to_array, wer
from scipy.signal import resample_poly

# About 17 characters per audio second in S4: 15,300 characters give about 15 minutes.
EPISODE_CHARS = 15_300
SECTION_MAX_CHARS = 8_000
VOICES = (("Lena", "de-de-advisor-3"), ("Felix", "de-de-tutor-4"))


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


WHISPER_SR = 16_000


def load_16k(audio: Path) -> np.ndarray:
    """Whisper reads arrays as 16 kHz mono. Decode with soundfile, not PyAV: faster-whisper 1.2
    passes an argument that PyAV 19 no longer accepts."""
    y, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != WHISPER_SR:
        g = np.gcd(sr, WHISPER_SR)
        y = resample_poly(y, WHISPER_SR // g, sr // g).astype(np.float32)
    return y


def run(
    audio: Path, model_name: str, compute_type: str, threads: int, reference: str | None
) -> dict:
    from faster_whisper import WhisperModel

    started = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type=compute_type, cpu_threads=threads)
    load_s = time.perf_counter() - started
    y = load_16k(audio)
    started = time.perf_counter()
    segs, _ = model.transcribe(
        y,
        language="de",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segs)  # the generator does the work here
    stt_s = time.perf_counter() - started
    seconds = len(y) / WHISPER_SR
    row = {
        "model": model_name,
        "compute_type": compute_type,
        "cpu_threads": threads,
        "audio_seconds": round(seconds, 1),
        "load_seconds": round(load_s, 1),
        "transcribe_seconds": round(stt_s, 1),
        "rtf": round(stt_s / seconds, 3),
        "peak_rss_mb": round(peak_mb()),
        "wer": round(wer(reference, text), 3) if reference else None,
    }
    print(json.dumps(row), flush=True)
    return row


def make_audio(out: Path) -> None:
    turns = episode_turns(load_script(), EPISODE_CHARS)
    parts = []
    for k, sec in enumerate(split_sections(turns, SECTION_MAX_CHARS)):
        r = synth(FLASH, list(VOICES), sec)
        if r["status"] != 200 or "wav" not in r:
            raise SystemExit(f"section {k} failed: {r.get('error')}")
        parts += [to_array(r["wav"]), np.zeros(SR // 2, dtype=np.float32)]
        print(f"section {k}: {len(parts[-2]) / SR:.1f} s, finish={r.get('finish')}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    y = np.concatenate(parts)
    sf.write(out / "episode.wav", y, SR, subtype="PCM_16")
    (out / "reference.txt").write_text(" ".join(t for _, t in turns) + "\n")
    print(f"{len(y) / SR / 60:.1f} min -> {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("synth").add_argument("out", type=Path)
    r = sub.add_parser("run")
    r.add_argument("audio", type=Path)
    r.add_argument("--models", default="small,medium,large-v3")
    r.add_argument("--compute-type", default="int8")
    # GitHub-hosted ubuntu-latest runners for public repos have 4 vCPUs.
    r.add_argument("--cpu-threads", type=int, default=4)
    r.add_argument("--reference", type=Path)
    r.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.cmd == "synth":
        make_audio(a.out)
        return
    ref = a.reference.read_text() if a.reference else None
    env = {
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "machine": platform.machine(),
    }
    print(json.dumps(env))
    rows = [run(a.audio, m, a.compute_type, a.cpu_threads, ref) for m in a.models.split(",")]
    if a.out:
        a.out.write_text(json.dumps({"env": env, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
