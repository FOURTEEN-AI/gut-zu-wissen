import json
import math
import re
import shutil
import subprocess

import numpy as np
import pytest
from scipy.signal import lfilter, resample, resample_poly

from gut_zu_wissen import audio
from gut_zu_wissen.run import load_config

SR = 24_000
CONFIG = load_config()
TARGET = CONFIG["quotas"]["target_loudness_lufs"]
CEILING = CONFIG["quotas"]["true_peak_ceiling_dbtp"]
A = CONFIG["audio"]


def speech_like(seconds: float, seed: int = 1) -> np.ndarray:
    """Shaped noise in syllables and pauses, soft-clipped at full scale, at -20 LUFS with true
    peaks between about -2 and -1 dBTP, like the raw TTS output measured in spike S4 (-20 LUFS,
    peaks near -1 dBFS). More samples need limiting than in the spike (0.16 % against 0.03 %)."""
    rng = np.random.default_rng(seed)
    n = round(seconds * SR)
    t = np.arange(n) / SR
    noise = lfilter([1.0], [1.0, -0.9], rng.standard_normal(n))
    syllables = np.clip(np.sin(2 * np.pi * 4.1 * t), 0, None) ** 2
    phrases = (np.sin(2 * np.pi * 0.23 * t) > -0.6).astype(float)
    y = noise * syllables * phrases
    y *= 10 ** ((-20 - audio.integrated_loudness(y, SR)) / 20)
    y = np.tanh(y)
    return y * 10 ** ((-20 - audio.integrated_loudness(y, SR)) / 20)


def normalize(y: np.ndarray) -> tuple[np.ndarray, audio.LoudnessReport]:
    return audio.normalize(
        y,
        SR,
        TARGET,
        CEILING,
        A["limiter_attack_db_per_second"],
        A["limiter_release_db_per_second"],
        A["loudness_tolerance_lu"],
        A["loudness_max_iterations"],
    )


# ---------------------------------------------------------------- measuring


def test_true_peak_finds_the_peak_between_samples():
    # A quarter-rate sine shifted by 45 degrees: every sample is at 0.707, the wave reaches 1.
    # Faded in and out, so switching it on and off adds no overshoot.
    n = np.arange(SR)
    fade = np.hanning(SR // 5)
    envelope = np.concatenate([fade[: SR // 10], np.ones(SR - SR // 5), fade[SR // 10 :]])
    y = np.sin(np.pi / 2 * n + np.pi / 4) * envelope
    assert 20 * math.log10(np.abs(y).max()) == pytest.approx(-3.01, abs=0.01)
    assert audio.true_peak_dbtp(y, SR) == pytest.approx(0.0, abs=0.02)


def test_true_peak_of_blocks_matches_the_whole_signal():
    y = speech_like(10)
    padded = np.pad(y, 4096)
    fft = np.abs(resample(padded, len(padded) * 8)).max()
    poly = np.abs(resample_poly(padded, 8, 1)).max()
    whole = 20 * math.log10(max(fft, poly))
    assert audio.true_peak_dbtp(y, SR) == pytest.approx(whole, abs=0.01)


def test_silence_has_no_loudness():
    with pytest.raises(audio.AudioError):
        audio.integrated_loudness(np.zeros(SR * 2), SR)


# ---------------------------------------------------------------- normalising


def test_raw_tts_level_needs_the_limiter():
    y = speech_like(20)
    assert audio.integrated_loudness(y, SR) == pytest.approx(-20, abs=0.01)
    plain = y * 10 ** ((TARGET - audio.integrated_loudness(y, SR)) / 20)
    assert -2 < audio.true_peak_dbtp(y, SR) < -1
    assert audio.true_peak_dbtp(plain, SR) > CEILING + 2  # plain gain would overshoot


def test_output_meets_loudness_target_and_peak_ceiling(tmp_path):
    out, report = normalize(speech_like(30))
    path = tmp_path / "episode.flac"
    audio.write(path, out, SR)
    written, sr = audio.read(path)
    assert sr == SR
    assert np.array_equal(written, out)  # measured signal is the file's content
    assert abs(audio.integrated_loudness(written, SR) - TARGET) <= A["loudness_tolerance_lu"]
    assert audio.true_peak_dbtp(written, SR) <= CEILING
    assert report.output_true_peak_dbtp <= CEILING
    assert report.limited_seconds > 0
    assert report.input_lufs == pytest.approx(-20, abs=0.01)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_output_passes_an_independent_meter(tmp_path):
    """ffmpeg's EBU R128 meter (BS.1770 reference filter) as a second opinion."""
    out, _ = normalize(speech_like(30, seed=2))
    path = tmp_path / "episode.flac"
    audio.write(path, out, SR)
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path)]
        + ["-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    measured = json.loads(re.findall(r"\{[^{}]*\}", result.stderr)[-1])
    assert abs(float(measured["input_i"]) - TARGET) <= 0.2  # ffmpeg reports 2 decimals
    assert float(measured["input_tp"]) <= CEILING


def test_quiet_input_without_peaks_is_only_amplified():
    t = np.arange(SR * 10) / SR
    y = 0.01 * np.sin(2 * np.pi * 440 * t)
    out, report = normalize(y)
    assert report.limited_seconds == 0 and report.iterations == 1
    assert abs(audio.integrated_loudness(out, SR) - TARGET) <= A["loudness_tolerance_lu"]


def test_limiter_gain_never_exceeds_the_allowed_gain():
    rng = np.random.default_rng(3)
    envelope = np.abs(rng.standard_normal(5000))
    change = audio.limiter_gain_db(audio.level_db(envelope), -6.0, 0.5, 0.05)
    allowed = np.minimum(0.0, -6.0 - 20 * np.log10(envelope))
    assert np.all(change <= allowed + 1e-12)
    assert np.all(np.diff(change) <= 0.05 + 1e-9)  # release rate
    assert np.all(np.diff(change) >= -0.5 - 1e-9)  # attack rate


def test_limiter_leaves_signal_below_the_ceiling_untouched():
    change = audio.limiter_gain_db(np.full(100, -20.0), -1.5, 1.0, 0.1)
    assert np.all(change == 0)


def test_unreachable_target_raises():
    with pytest.raises(audio.AudioError, match="did not reach"):
        audio.normalize(speech_like(5), SR, TARGET, CEILING, 1200.0, 60.0, 0.0, 1)


# ---------------------------------------------------------------- edges and joins


def tone(seconds: float) -> np.ndarray:
    return 0.5 * np.sin(2 * np.pi * 300 * np.arange(round(seconds * SR)) / SR)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * SR))


def edges(y: np.ndarray) -> tuple[float, float]:
    return audio.edge_silence(y, SR, A["edge_silence_floor_db"], A["edge_silence_frame_seconds"])


def trim(y: np.ndarray) -> np.ndarray:
    return audio.trim_edges(
        y,
        SR,
        A["edge_silence_floor_db"],
        A["edge_silence_frame_seconds"],
        A["edge_silence_keep_seconds"],
    )


def test_edge_silence_is_measured():
    lead, trail = edges(np.concatenate([silence(0.25), tone(1), silence(0.8)]))
    assert lead == pytest.approx(0.25, abs=0.011) and trail == pytest.approx(0.8, abs=0.011)


def test_long_edges_are_trimmed_to_the_kept_silence():
    keep = A["edge_silence_keep_seconds"]
    y = trim(np.concatenate([silence(0.25), tone(1), silence(0.8)]))
    lead, trail = edges(y)
    assert lead == pytest.approx(keep, abs=0.011) and trail == pytest.approx(keep, abs=0.011)
    assert len(y) / SR == pytest.approx(1 + 2 * keep, abs=0.02)


def test_short_edges_stay():
    y = np.concatenate([silence(0.05), tone(1), silence(0.05)])
    assert np.array_equal(trim(y), y)


def test_silent_part_is_an_error():
    with pytest.raises(audio.AudioError):
        trim(silence(1))


def test_join_puts_the_pause_between_parts_only():
    pause = A["join_pause_seconds"]
    joined = audio.join([tone(1), tone(2), tone(0.5)], SR, pause)
    assert len(joined) == round(3.5 * SR) + 2 * round(pause * SR)
    assert np.all(joined[SR : SR + round(pause * SR)] == 0)


def test_decode_rejects_garbage():
    with pytest.raises(audio.AudioError):
        audio.decode(b"not audio")
