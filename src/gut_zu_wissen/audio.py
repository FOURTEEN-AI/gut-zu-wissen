"""Audio processing (phase 1, step 9; design: "Audio", "Lautheit").

Pure functions on mono float arrays: reading and writing, trimming silence at the edges of a
part, joining parts with a pause, measuring integrated loudness (ITU-R BS.1770 via pyloudnorm)
and true peak, and normalising to a target loudness with a true-peak limiter.

True peak follows BS.1770 annex 2 (oversampling to at least 192 kHz). Interpolation filters
disagree on content close to the Nyquist frequency, so the envelope takes the higher of two:
band-limited (FFT) interpolation and a windowed-sinc polyphase filter. A meter with either kind of
filter then reads the output at or below the ceiling.

The limiter works offline with look-ahead: the true-peak envelope gives the gain each sample may
have; the gain curve falls towards a peak at the attack rate and
recovers after it at the release rate, never above the allowed gain. Loudness is measured again
after limiting, and gain and ceiling are corrected until both targets hold.
Output is quantised to 16 bit before it is measured, so the measurement is that of the file.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
from scipy.signal import resample, resample_poly

# BS.1770-4 annex 2: true peak by oversampling to at least 192 kHz (4x at 48 kHz).
TRUE_PEAK_RATE_HZ = 192_000
# Input samples per oversampling block, and the overlap at each block edge; with this overlap
# the blockwise result is within 0.01 dB of oversampling the whole signal at once.
_BLOCK = 1 << 16
_MARGIN = 1 << 12
# 16-bit PCM as written and read by soundfile: value = int / 2**15.
_PCM16_SCALE = 32768
SUBTYPE = "PCM_16"
FORMAT = "FLAC"


class AudioError(Exception):
    """The audio cannot be processed (wrong format, silence, targets unreachable)."""


@dataclass
class LoudnessReport:
    """Result of the normalisation; field order is the archive field order."""

    input_lufs: float
    input_true_peak_dbtp: float
    gain_db: float
    output_lufs: float
    output_true_peak_dbtp: float
    limited_seconds: float
    iterations: int


# ---------------------------------------------------------------- reading and writing


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """Mono float64 samples and sample rate of a WAV or FLAC file."""
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise AudioError(f"Unreadable audio: {exc}") from exc
    return y.mean(axis=1), int(sr)


def read(path: Path) -> tuple[np.ndarray, int]:
    return decode(path.read_bytes())


def quantize(y: np.ndarray) -> np.ndarray:
    """Round to the 16-bit grid, exactly as a write and a read through soundfile would."""
    return np.clip(np.round(y * _PCM16_SCALE), -_PCM16_SCALE, _PCM16_SCALE - 1) / _PCM16_SCALE


def write(path: Path, y: np.ndarray, sr: int) -> None:
    """16-bit FLAC (lossless)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, quantize(y), sr, format=FORMAT, subtype=SUBTYPE)


def duration(y: np.ndarray, sr: int) -> float:
    return len(y) / sr


# ---------------------------------------------------------------- edges and joins


def edge_silence(
    y: np.ndarray, sr: int, floor_db: float, frame_seconds: float
) -> tuple[float, float]:
    """Leading and trailing silence in seconds: frames whose RMS is below `floor_db` dBFS."""
    hop = max(1, round(frame_seconds * sr))
    n = len(y) // hop
    if n == 0:
        return duration(y, sr), 0.0
    rms = np.sqrt((y[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-20)
    loud = np.nonzero(20 * np.log10(rms) > floor_db)[0]
    if len(loud) == 0:
        return duration(y, sr), 0.0
    return loud[0] * hop / sr, (len(y) - (loud[-1] + 1) * hop) / sr


def trim_edges(
    y: np.ndarray, sr: int, floor_db: float, frame_seconds: float, keep_seconds: float
) -> np.ndarray:
    """Shorten the silence at both edges to at most `keep_seconds`; speech is never cut."""
    lead, trail = edge_silence(y, sr, floor_db, frame_seconds)
    if lead >= duration(y, sr):
        raise AudioError("The audio part is silent.")
    start = max(0, round((lead - keep_seconds) * sr))
    end = len(y) - max(0, round((trail - keep_seconds) * sr))
    return y[start:end]


def join(parts: Sequence[np.ndarray], sr: int, pause_seconds: float) -> np.ndarray:
    """Concatenate the parts with a pause of digital silence between them."""
    pause = np.zeros(round(pause_seconds * sr))
    pieces: list[np.ndarray] = []
    for n, part in enumerate(parts):
        if n:
            pieces.append(pause)
        pieces.append(part)
    return np.concatenate(pieces) if pieces else np.zeros(0)


# ---------------------------------------------------------------- measuring


def integrated_loudness(y: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS (BS.1770, gated)."""
    value = float(pyloudnorm.Meter(sr).integrated_loudness(y))
    if not math.isfinite(value):
        raise AudioError("The audio is silent; loudness is undefined.")
    return value


def oversampling(sr: int) -> int:
    return max(1, math.ceil(TRUE_PEAK_RATE_HZ / sr))


def peak_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Per input sample n, the largest absolute value of the oversampled signal in [n, n+1),
    the higher of FFT and polyphase interpolation.

    Oversampling runs per block; each block overlaps its neighbours by `_MARGIN` samples on both
    sides, and the signal is padded with silence, so block edges do not show in the result.
    """
    up = oversampling(sr)
    padded = np.pad(y, _MARGIN)
    envelope = np.empty(len(y))
    for start in range(0, len(y), _BLOCK):
        end = min(len(y), start + _BLOCK)
        segment = padded[start : end + 2 * _MARGIN]
        core = slice(_MARGIN * up, (_MARGIN + end - start) * up)
        fft = np.abs(resample(segment, len(segment) * up)[core])
        poly = np.abs(resample_poly(segment, up, 1)[core])
        envelope[start:end] = np.maximum(fft, poly).reshape(end - start, up).max(axis=1)
    return np.maximum(envelope, np.abs(y))


def to_db(value: float) -> float:
    return 20 * math.log10(value) if value > 0 else -math.inf


def true_peak_dbtp(y: np.ndarray, sr: int) -> float:
    return to_db(float(peak_envelope(y, sr).max(initial=0.0)))


# ---------------------------------------------------------------- normalising


def level_db(envelope: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return 20 * np.log10(envelope)


def limiter_gain_db(
    level: np.ndarray, headroom_db: float, attack: float, release: float
) -> np.ndarray:
    """Per-sample gain change in dB (<= 0) that keeps `level` (dB) below `headroom_db`.

    The curve is the largest one that stays at or below the gain each sample allows and changes
    by at most `attack` dB per sample while falling towards a peak (look-ahead) and `release` dB
    per sample while recovering after it. Both bounds are cumulative minima:
    g[n] = min over m <= n of (a[m] + (n - m) * release), and the same backwards for the attack.
    """
    allowed = np.minimum(0.0, headroom_db - level)
    n = np.arange(len(allowed), dtype=np.float64)
    after = n * release + np.minimum.accumulate(allowed - n * release)
    reverse = allowed[::-1]
    before = (n * attack + np.minimum.accumulate(reverse - n * attack))[::-1]
    return np.minimum(after, before)


def next_gain(
    gain_db: float, lufs: float, target_lufs: float, previous: tuple[float, float] | None
) -> float:
    """Secant step towards the target. Limiting makes loudness grow slower than gain, so the
    slope from the last two tries (kept between 0.1 and 1) gives a better step than 1:1."""
    slope = 1.0
    if previous is not None and gain_db != previous[0]:
        slope = min(1.0, max(0.1, (lufs - previous[1]) / (gain_db - previous[0])))
    return gain_db + (target_lufs - lufs) / slope


def normalize(
    y: np.ndarray,
    sr: int,
    target_lufs: float,
    ceiling_dbtp: float,
    attack_db_per_second: float,
    release_db_per_second: float,
    tolerance_lu: float,
    max_iterations: int,
) -> tuple[np.ndarray, LoudnessReport]:
    """Loudness `target_lufs` (within `tolerance_lu`) with true peak at most `ceiling_dbtp`.

    The result is on the 16-bit grid. Raises `AudioError` if the targets are not met within
    `max_iterations`.
    """
    input_lufs = integrated_loudness(y, sr)
    envelope = peak_envelope(y, sr)
    input_peak = to_db(float(envelope.max(initial=0.0)))
    level = level_db(envelope)
    attack, release = attack_db_per_second / sr, release_db_per_second / sr
    gain_db, ceiling_db = target_lufs - input_lufs, ceiling_dbtp
    previous: tuple[float, float] | None = None
    for iteration in range(1, max_iterations + 1):
        change = limiter_gain_db(level, ceiling_db - gain_db, attack, release)
        out = quantize(y * 10 ** ((gain_db + change) / 20))
        lufs, peak = integrated_loudness(out, sr), true_peak_dbtp(out, sr)
        loud_ok, peak_ok = abs(lufs - target_lufs) <= tolerance_lu, peak <= ceiling_dbtp
        if loud_ok and peak_ok:
            report = LoudnessReport(
                round(input_lufs, 2),
                round(input_peak, 2),
                round(gain_db, 2),
                round(lufs, 2),
                round(peak, 2),
                round(float(np.count_nonzero(change < 0)) / sr, 2),
                iteration,
            )
            return out, report
        if not peak_ok:  # rounding to 16 bit can lift a peak by up to one step
            step_db = to_db(1 + 1 / (_PCM16_SCALE * 10 ** (ceiling_dbtp / 20)))
            ceiling_db -= peak - ceiling_dbtp + step_db
        if not loud_ok:
            gain_db, previous = next_gain(gain_db, lufs, target_lufs, previous), (gain_db, lufs)
    raise AudioError(
        f"Loudness normalisation did not reach {target_lufs} LUFS / {ceiling_dbtp} dBTP "
        f"in {max_iterations} iterations (last: {lufs:.2f} LUFS, {peak:.2f} dBTP)."
    )
