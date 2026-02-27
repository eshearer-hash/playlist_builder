#!/usr/bin/env python3
"""
Bulk Audio Processor — local Spotify-style audio feature & analysis extraction.

Decodes .m4a (or any ffmpeg-supported format) to PCM via ffmpeg, then computes:
  • Audio Features  (danceability, energy, speechiness, acousticness, …)
  • Audio Analysis   (track summary, bars, beats, sections, segments, tatums)

Dependencies: numpy, scipy, ffmpeg (CLI).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import struct
import sys
import uuid
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
from tqdm import tqdm

import numpy as np
from numpy.typing import NDArray
from scipy import signal, ndimage

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
ANALYSIS_SR = 22050          # Spotify uses 22 050 Hz mono for analysis
HOP_LENGTH = 512
FRAME_LENGTH = 2048
N_FFT = 2048
N_MELS = 128
N_CHROMA = 12
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ──────────────────────────────────────────────────────────────────────────────
# Audio I/O via ffmpeg
# ──────────────────────────────────────────────────────────────────────────────
def decode_audio(path: Path, sr: int = ANALYSIS_SR) -> tuple[NDArray[np.float32], int]:
    """Decode any audio file to mono float32 PCM using ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(sr),
        "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode()}")
    samples = np.frombuffer(proc.stdout, dtype=np.float32)
    return samples, sr


# ──────────────────────────────────────────────────────────────────────────────
# Low-level DSP helpers
# ──────────────────────────────────────────────────────────────────────────────
def _stft(y: NDArray, n_fft: int = N_FFT, hop: int = HOP_LENGTH) -> NDArray:
    """Short-time Fourier transform (complex)."""
    win = signal.windows.hann(n_fft, sym=False).astype(np.float32)
    # Pad signal
    pad_len = n_fft // 2
    y_padded = np.pad(y, (pad_len, pad_len), mode="reflect")
    n_frames = 1 + (len(y_padded) - n_fft) // hop
    shape = (n_fft, n_frames)
    strides = (y_padded.strides[0], y_padded.strides[0] * hop)
    frames = np.lib.stride_tricks.as_strided(y_padded, shape=shape, strides=strides)
    return np.fft.rfft(win[:, None] * frames, n=n_fft, axis=0)


def _power_spec(S: NDArray) -> NDArray:
    return np.abs(S) ** 2


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS) -> NDArray:
    """Create a Mel filterbank matrix."""
    f_min, f_max = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    fft_freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = freqs[i], freqs[i + 1], freqs[i + 2]
        up = (fft_freqs - lo) / max(mid - lo, 1e-10)
        down = (hi - fft_freqs) / max(hi - mid, 1e-10)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


def _mel_spectrogram(y: NDArray, sr: int) -> NDArray:
    S = _power_spec(_stft(y))
    fb = _mel_filterbank(sr, N_FFT, N_MELS)
    mel = fb @ S
    return mel


def _log_mel(y: NDArray, sr: int) -> NDArray:
    return np.log1p(_mel_spectrogram(y, sr))


def _chroma(y: NDArray, sr: int) -> NDArray:
    """Compute chromagram (12 × frames)."""
    S = np.abs(_stft(y))
    n_fft_bins = S.shape[0]
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    chroma_fb = np.zeros((N_CHROMA, n_fft_bins), dtype=np.float32)
    for i in range(n_fft_bins):
        if freqs[i] < 20:
            continue
        pitch = 12.0 * np.log2(freqs[i] / 440.0) + 69.0
        chroma_bin = int(round(pitch)) % 12
        chroma_fb[chroma_bin, i] += 1.0
    C = chroma_fb @ S
    norms = np.maximum(C.max(axis=0, keepdims=True), 1e-10)
    return C / norms


def _rms(y: NDArray, frame_length: int = FRAME_LENGTH, hop: int = HOP_LENGTH) -> NDArray:
    """Root-mean-square energy per frame."""
    pad = frame_length // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(yp) - frame_length) // hop
    shape = (frame_length, n_frames)
    strides = (yp.strides[0], yp.strides[0] * hop)
    frames = np.lib.stride_tricks.as_strided(yp, shape=shape, strides=strides)
    return np.sqrt(np.mean(frames ** 2, axis=0))


def _spectral_centroid(y: NDArray, sr: int) -> NDArray:
    S = np.abs(_stft(y))
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    return np.sum(freqs[:, None] * S, axis=0) / np.maximum(np.sum(S, axis=0), 1e-10)


def _spectral_rolloff(y: NDArray, sr: int, pct: float = 0.85) -> NDArray:
    S = np.abs(_stft(y))
    cum = np.cumsum(S, axis=0)
    thresh = pct * cum[-1:]
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    idx = np.argmax(cum >= thresh, axis=0)
    return freqs[idx]


def _spectral_flatness(y: NDArray) -> NDArray:
    S = np.abs(_stft(y)) + 1e-10
    geo = np.exp(np.mean(np.log(S), axis=0))
    arith = np.mean(S, axis=0)
    return geo / arith


def _zcr(y: NDArray, frame_length: int = FRAME_LENGTH, hop: int = HOP_LENGTH) -> NDArray:
    pad = frame_length // 2
    yp = np.pad(y, (pad, pad), mode="reflect")
    n_frames = 1 + (len(yp) - frame_length) // hop
    out = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = yp[start : start + frame_length]
        out[i] = np.mean(np.abs(np.diff(np.sign(frame))) > 0)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Onset / beat / tempo detection
# ──────────────────────────────────────────────────────────────────────────────
def _onset_strength(y: NDArray, sr: int) -> NDArray:
    """Spectral flux onset envelope."""
    mel = _mel_spectrogram(y, sr)
    log_mel = np.log1p(mel)
    diff = np.diff(log_mel, axis=1)
    diff = np.maximum(0, diff)
    return np.mean(diff, axis=0)


def _estimate_tempo(onset_env: NDArray, sr: int) -> float:
    """Autocorrelation-based tempo estimation."""
    hop_sec = HOP_LENGTH / sr
    # Autocorrelate
    n = len(onset_env)
    # BPM range 40–220
    min_lag = int(60.0 / 220.0 / hop_sec)
    max_lag = int(60.0 / 40.0 / hop_sec)
    max_lag = min(max_lag, n - 1)
    if min_lag >= max_lag:
        return 120.0
    oe = onset_env - np.mean(onset_env)
    corr = np.correlate(oe, oe, mode="full")
    corr = corr[n - 1:]  # positive lags
    corr_slice = corr[min_lag:max_lag + 1]
    if len(corr_slice) == 0:
        return 120.0
    # Weight toward common tempos
    lags = np.arange(min_lag, max_lag + 1)
    bpms = 60.0 / (lags * hop_sec)
    # Gaussian weighting around 120 BPM
    weight = np.exp(-0.5 * ((bpms - 120.0) / 40.0) ** 2)
    weighted = corr_slice * weight
    best = np.argmax(weighted)
    return float(bpms[best])


def _detect_beats(onset_env: NDArray, sr: int, tempo: float) -> NDArray[np.float64]:
    """Simple beat tracking via peak-picking on onset envelope."""
    hop_sec = HOP_LENGTH / sr
    period_frames = int(round(60.0 / tempo / hop_sec))
    if period_frames < 1:
        period_frames = 1

    # Smooth onset envelope
    kernel_size = max(3, period_frames // 4)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smooth = ndimage.uniform_filter1d(onset_env.astype(np.float64), kernel_size)

    # Adaptive threshold
    threshold = ndimage.uniform_filter1d(smooth, period_frames * 2) + 0.01

    # Peak pick with minimum distance
    peaks = []
    last = -period_frames
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1]:
            if smooth[i] > threshold[i] and (i - last) >= period_frames * 0.7:
                peaks.append(i)
                last = i
    return np.array(peaks, dtype=np.float64) * hop_sec


def _detect_onsets(onset_env: NDArray, sr: int) -> NDArray[np.float64]:
    """Detect onset times from onset strength envelope."""
    hop_sec = HOP_LENGTH / sr
    threshold = np.mean(onset_env) + 0.5 * np.std(onset_env)
    peaks = []
    min_gap = int(0.03 / hop_sec)  # 30ms minimum gap
    last = -min_gap
    for i in range(1, len(onset_env) - 1):
        if onset_env[i] > onset_env[i - 1] and onset_env[i] > onset_env[i + 1]:
            if onset_env[i] > threshold and (i - last) >= min_gap:
                peaks.append(i)
                last = i
    return np.array(peaks, dtype=np.float64) * hop_sec


# ──────────────────────────────────────────────────────────────────────────────
# Key & mode detection
# ──────────────────────────────────────────────────────────────────────────────
# Krumhansl-Kessler profiles
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _detect_key(y: NDArray, sr: int) -> tuple[int, int, float]:
    """
    Detect key (0-11) and mode (0=minor, 1=major) using chroma correlation
    with Krumhansl-Kessler profiles. Returns (key, mode, confidence).
    """
    C = _chroma(y, sr)
    chroma_mean = np.mean(C, axis=1)
    chroma_mean /= np.maximum(np.linalg.norm(chroma_mean), 1e-10)

    best_corr = -2.0
    best_key = 0
    best_mode = 1
    for shift in range(12):
        for mode, profile in enumerate([_MINOR_PROFILE, _MAJOR_PROFILE]):
            rolled = np.roll(profile, shift)
            rolled = rolled / np.linalg.norm(rolled)
            corr = float(np.dot(chroma_mean, rolled))
            if corr > best_corr:
                best_corr = corr
                best_key = shift
                best_mode = mode
    confidence = max(0.0, min(1.0, (best_corr + 1.0) / 2.0))
    return best_key, best_mode, confidence


# ──────────────────────────────────────────────────────────────────────────────
# Time signature estimation
# ──────────────────────────────────────────────────────────────────────────────
def _estimate_time_signature(beats: NDArray, tempo: float) -> tuple[int, float]:
    """Estimate time signature (3, 4, 5, 6, 7) from beat periodicity."""
    if len(beats) < 4:
        return 4, 0.5
    ibis = np.diff(beats)
    if len(ibis) < 2:
        return 4, 0.5
    med_ibi = np.median(ibis)
    if med_ibi <= 0:
        return 4, 0.5

    best_sig = 4
    best_score = 0.0
    for ts in [3, 4, 5, 6, 7]:
        bar_dur = med_ibi * ts
        # Check how well beats align to bar boundaries
        phases = (beats % bar_dur) / bar_dur
        # Circular variance (lower = more periodic)
        c = np.mean(np.cos(2 * np.pi * phases))
        s = np.mean(np.sin(2 * np.pi * phases))
        r = np.sqrt(c ** 2 + s ** 2)
        # Weight toward 4/4
        weight = 1.0 if ts == 4 else (0.9 if ts == 3 else 0.7)
        score = r * weight
        if score > best_score:
            best_score = score
            best_sig = ts
    confidence = min(1.0, best_score)
    return best_sig, confidence


# ──────────────────────────────────────────────────────────────────────────────
# Fade detection
# ──────────────────────────────────────────────────────────────────────────────
def _detect_fades(rms_env: NDArray, sr: int) -> tuple[float, float]:
    """Detect end of fade-in and start of fade-out in seconds."""
    hop_sec = HOP_LENGTH / sr
    duration = len(rms_env) * hop_sec
    if len(rms_env) < 10:
        return 0.0, duration

    overall_mean = np.mean(rms_env)
    threshold = overall_mean * 0.15

    # Fade-in: first frame where RMS exceeds threshold
    fade_in = 0.0
    for i in range(len(rms_env)):
        if rms_env[i] > threshold:
            fade_in = i * hop_sec
            break

    # Fade-out: last frame where RMS exceeds threshold
    fade_out = duration
    for i in range(len(rms_env) - 1, -1, -1):
        if rms_env[i] > threshold:
            fade_out = i * hop_sec
            break

    return fade_in, fade_out


# ──────────────────────────────────────────────────────────────────────────────
# High-level feature heuristics (approximations of Spotify's ML models)
# ──────────────────────────────────────────────────────────────────────────────
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _compute_loudness_db(y: NDArray) -> float:
    """Integrated loudness approximation (dBFS)."""
    rms_val = np.sqrt(np.mean(y ** 2))
    if rms_val < 1e-10:
        return -60.0
    return float(20.0 * np.log10(rms_val))


def _compute_energy(rms_env: NDArray, sc: NDArray) -> float:
    """Energy: combination of RMS loudness and spectral content."""
    rms_mean = float(np.mean(rms_env))
    sc_norm = float(np.mean(sc)) / (ANALYSIS_SR / 2)
    # Energy correlates with loudness and brightness
    e = 0.6 * min(rms_mean / 0.15, 1.0) + 0.4 * min(sc_norm / 0.3, 1.0)
    return _clamp(e)


def _compute_danceability(tempo: float, onset_env: NDArray, beats: NDArray) -> float:
    """Danceability: regularity of beats + tempo in dance range."""
    # Tempo score: peak around 100-130 BPM
    tempo_score = np.exp(-0.5 * ((tempo - 115) / 30) ** 2)

    # Beat regularity
    if len(beats) > 2:
        ibis = np.diff(beats)
        cv = float(np.std(ibis) / max(np.mean(ibis), 1e-6))
        regularity = max(0.0, 1.0 - cv)
    else:
        regularity = 0.3

    # Onset strength consistency
    oe_std = float(np.std(onset_env))
    oe_mean = float(np.mean(onset_env))
    pulse_clarity = min(oe_std / max(oe_mean, 1e-6), 1.0)

    d = 0.4 * tempo_score + 0.35 * regularity + 0.25 * pulse_clarity
    return _clamp(d)


def _compute_speechiness(zcr: NDArray, sc: NDArray, sf: NDArray) -> float:
    """Speechiness: ZCR and spectral flatness patterns typical of speech."""
    zcr_mean = float(np.mean(zcr))
    sf_mean = float(np.mean(sf))
    # Speech: moderate ZCR, moderate flatness
    zcr_score = np.exp(-0.5 * ((zcr_mean - 0.08) / 0.04) ** 2)
    sf_score = min(sf_mean / 0.3, 1.0)
    s = 0.5 * zcr_score + 0.5 * sf_score
    return _clamp(s * 0.6)  # Most music has low speechiness


def _compute_acousticness(sf: NDArray, sc: NDArray, rms_env: NDArray) -> float:
    """Acousticness: high spectral flatness + lower brightness → acoustic."""
    sf_mean = float(np.mean(sf))
    sc_mean = float(np.mean(sc)) / (ANALYSIS_SR / 2)
    # Acoustic music: less compressed, more dynamic range
    rms_std = float(np.std(rms_env))
    rms_mean = float(np.mean(rms_env))
    dynamics = rms_std / max(rms_mean, 1e-6)

    # Lower centroid + higher dynamics + moderate flatness = acoustic
    a = 0.4 * (1.0 - min(sc_mean / 0.25, 1.0)) + 0.3 * min(dynamics / 0.5, 1.0) + 0.3 * sf_mean
    return _clamp(a)


def _compute_instrumentalness(zcr: NDArray, sf: NDArray, rms_env: NDArray) -> float:
    """Instrumentalness: low ZCR variance (no vocals) + spectral patterns."""
    zcr_cv = float(np.std(zcr) / max(np.mean(zcr), 1e-6))
    sf_mean = float(np.mean(sf))
    # Vocals create high ZCR variance
    vocal_absence = max(0.0, 1.0 - zcr_cv / 1.5)
    i = 0.6 * vocal_absence + 0.4 * (1.0 - sf_mean)
    return _clamp(i * 0.5)  # Most tracks with vocals get low values


def _compute_liveness(rms_env: NDArray, onset_env: NDArray) -> float:
    """Liveness: dynamic range and onset patterns suggest live performance."""
    rms_std = float(np.std(rms_env))
    rms_mean = float(np.mean(rms_env))
    dynamics = rms_std / max(rms_mean, 1e-6)
    # Live tracks tend to have more variance
    l = min(dynamics / 0.6, 1.0)
    return _clamp(l * 0.4)  # Most studio tracks score low


def _compute_valence(key: int, mode: int, tempo: float, sc: NDArray, energy: float) -> float:
    """Valence (happiness): major mode + higher tempo + brighter spectrum."""
    mode_score = 0.6 if mode == 1 else 0.35
    tempo_score = _clamp((tempo - 60) / 120)
    brightness = float(np.mean(sc)) / (ANALYSIS_SR / 2)
    v = 0.3 * mode_score + 0.25 * tempo_score + 0.25 * min(brightness / 0.2, 1.0) + 0.2 * energy
    return _clamp(v)


# ──────────────────────────────────────────────────────────────────────────────
# Section detection (simple energy-based segmentation)
# ──────────────────────────────────────────────────────────────────────────────
def _detect_sections(
    y: NDArray, sr: int, rms_env: NDArray, onset_env: NDArray,
    tempo: float, key: int, mode: int
) -> list[dict]:
    """Detect structural sections via energy changes."""
    hop_sec = HOP_LENGTH / sr
    duration = len(y) / sr

    # Compute a smoothed feature curve
    window = max(int(4.0 / hop_sec), 1)  # ~4 second smoothing
    if len(rms_env) < window * 2:
        return [_make_section(0, duration, rms_env, onset_env, sr, tempo, key, mode)]

    smooth_rms = ndimage.uniform_filter1d(rms_env.astype(np.float64), window)

    # Detect significant changes
    diff = np.abs(np.diff(smooth_rms))
    threshold = np.mean(diff) + 1.5 * np.std(diff)

    boundaries = [0.0]
    min_section = 4.0  # Minimum section length in seconds
    for i in range(1, len(diff)):
        t = i * hop_sec
        if diff[i] > threshold and (t - boundaries[-1]) > min_section:
            boundaries.append(t)
    boundaries.append(duration)

    sections = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        dur = boundaries[i + 1] - start
        # Get local features for this section
        s_frame = int(start / hop_sec)
        e_frame = int((start + dur) / hop_sec)
        e_frame = min(e_frame, len(rms_env))
        sections.append(_make_section(start, dur, rms_env[s_frame:e_frame],
                                      onset_env[min(s_frame, len(onset_env)-1):min(e_frame, len(onset_env))],
                                      sr, tempo, key, mode))
    return sections


def _make_section(start, duration, rms_slice, onset_slice, sr, tempo, key, mode):
    loudness = -60.0
    if len(rms_slice) > 0:
        rms_val = float(np.sqrt(np.mean(rms_slice ** 2)))
        if rms_val > 1e-10:
            loudness = float(20.0 * np.log10(rms_val))
    return {
        "start": round(start, 5),
        "duration": round(duration, 5),
        "confidence": round(min(1.0, float(np.std(rms_slice)) / max(float(np.mean(rms_slice)), 1e-6) + 0.3), 3) if len(rms_slice) > 0 else 0.5,
        "loudness": round(loudness, 3),
        "tempo": round(tempo, 3),
        "tempo_confidence": 0.7,
        "key": key,
        "key_confidence": 0.5,
        "mode": mode,
        "mode_confidence": 0.5,
        "time_signature": 4,
        "time_signature_confidence": 0.9,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Segment detection (short timbral segments like Spotify)
# ──────────────────────────────────────────────────────────────────────────────
def _detect_segments(y: NDArray, sr: int, onset_times: NDArray, max_segments: int = 800) -> list[dict]:
    """Build segment list from onsets, with pitch + timbre vectors."""
    hop_sec = HOP_LENGTH / sr
    duration = len(y) / sr

    if len(onset_times) == 0:
        onset_times = np.array([0.0])

    # Ensure first onset is at 0 and we have an end boundary
    if onset_times[0] > 0.05:
        onset_times = np.concatenate([[0.0], onset_times])
    ends = np.concatenate([onset_times[1:], [duration]])

    # Subsample if too many
    if len(onset_times) > max_segments:
        idx = np.linspace(0, len(onset_times) - 1, max_segments, dtype=int)
        onset_times = onset_times[idx]
        ends = np.concatenate([onset_times[1:], [duration]])

    # Precompute chroma & mel for timbre
    C = _chroma(y, sr)       # 12 × n_frames
    mel = _mel_spectrogram(y, sr)
    log_mel = np.log1p(mel)   # n_mels × n_frames

    segments = []
    for i in range(len(onset_times)):
        start = float(onset_times[i])
        dur = float(ends[i] - start)
        if dur <= 0:
            continue

        s_frame = int(start / hop_sec)
        e_frame = max(s_frame + 1, int(ends[i] / hop_sec))
        e_frame = min(e_frame, C.shape[1], log_mel.shape[1])
        s_frame = min(s_frame, e_frame - 1)

        # Pitches: mean chroma in segment (12 values)
        chroma_seg = C[:, s_frame:e_frame]
        pitches = np.mean(chroma_seg, axis=1).tolist() if chroma_seg.shape[1] > 0 else [0.0] * 12

        # Timbre: first 12 mel-frequency cepstral-like coefficients
        mel_seg = log_mel[:, s_frame:e_frame]
        if mel_seg.shape[1] > 0:
            mel_mean = np.mean(mel_seg, axis=1)
            # Simple DCT-like reduction to 12 dims
            timbre = mel_mean[:12].tolist() if len(mel_mean) >= 12 else (mel_mean.tolist() + [0.0] * (12 - len(mel_mean)))
        else:
            timbre = [0.0] * 12

        # Loudness within segment
        s_sample = int(start * sr)
        e_sample = min(int(ends[i] * sr), len(y))
        seg_y = y[s_sample:e_sample]
        rms_val = float(np.sqrt(np.mean(seg_y ** 2))) if len(seg_y) > 0 else 1e-10
        loud_db = 20.0 * np.log10(max(rms_val, 1e-10))

        # Peak loudness
        if len(seg_y) > 0:
            frame_size = min(512, len(seg_y))
            n_sub = max(1, len(seg_y) // frame_size)
            sub_rms = [np.sqrt(np.mean(seg_y[j*frame_size:(j+1)*frame_size] ** 2))
                       for j in range(n_sub)]
            peak_rms = float(max(sub_rms))
            loud_max = 20.0 * np.log10(max(peak_rms, 1e-10))
            peak_idx = sub_rms.index(peak_rms)
            loud_max_time = peak_idx * frame_size / sr
        else:
            loud_max = loud_db
            loud_max_time = 0.0

        segments.append({
            "start": round(start, 5),
            "duration": round(dur, 5),
            "confidence": round(min(1.0, rms_val * 5), 3),
            "loudness_start": round(loud_db, 3),
            "loudness_max": round(loud_max, 3),
            "loudness_max_time": round(loud_max_time, 5),
            "loudness_end": 0,
            "pitches": [round(p, 3) for p in pitches],
            "timbre": [round(t, 3) for t in timbre],
        })

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Build bars and tatums from beats
# ──────────────────────────────────────────────────────────────────────────────
def _beats_to_bars(beats: NDArray, time_sig: int) -> list[dict]:
    bars = []
    for i in range(0, len(beats) - time_sig + 1, time_sig):
        start = float(beats[i])
        end = float(beats[min(i + time_sig, len(beats) - 1)])
        bars.append({
            "start": round(start, 5),
            "duration": round(end - start, 5),
            "confidence": 0.8,
        })
    return bars


def _beats_to_tatums(beats: NDArray, subdivisions: int = 2) -> list[dict]:
    tatums = []
    for i in range(len(beats) - 1):
        dt = (beats[i + 1] - beats[i]) / subdivisions
        for s in range(subdivisions):
            tatums.append({
                "start": round(float(beats[i] + s * dt), 5),
                "duration": round(float(dt), 5),
                "confidence": 0.7,
            })
    return tatums


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis function (single track)
# ──────────────────────────────────────────────────────────────────────────────
def analyze_track(path: Path) -> dict[str, Any]:
    """
    Analyze a single audio file and return a Spotify-style
    audio_features + audio_analysis dict.
    """
    path = Path(path)
    y, sr = decode_audio(path, ANALYSIS_SR)
    duration = len(y) / sr
    num_samples = len(y)
    track_id = path.stem

    # ── Core DSP features ────────────────────────────────────────────────
    rms_env = _rms(y)
    onset_env = _onset_strength(y, sr)
    sc = _spectral_centroid(y, sr)
    sr_env = _spectral_rolloff(y, sr)
    sf = _spectral_flatness(y)
    zcr = _zcr(y)

    # ── Tempo, beats, key ────────────────────────────────────────────────
    tempo = _estimate_tempo(onset_env, sr)
    beats = _detect_beats(onset_env, sr, tempo)
    key, mode, key_conf = _detect_key(y, sr)
    time_sig, ts_conf = _estimate_time_signature(beats, tempo)
    onset_times = _detect_onsets(onset_env, sr)
    fade_in, fade_out = _detect_fades(rms_env, sr)
    loudness = _compute_loudness_db(y)

    # ── High-level features ──────────────────────────────────────────────
    energy = _compute_energy(rms_env, sc)
    danceability = _compute_danceability(tempo, onset_env, beats)
    speechiness = _compute_speechiness(zcr, sc, sf)
    acousticness = _compute_acousticness(sf, sc, rms_env)
    instrumentalness = _compute_instrumentalness(zcr, sf, rms_env)
    liveness = _compute_liveness(rms_env, onset_env)
    valence = _compute_valence(key, mode, tempo, sc, energy)

    # ── Structural analysis ──────────────────────────────────────────────
    sections = _detect_sections(y, sr, rms_env, onset_env, tempo, key, mode)
    segments = _detect_segments(y, sr, onset_times, max_segments=600)
    bars = _beats_to_bars(beats, time_sig)
    tatums = _beats_to_tatums(beats)
    beat_list = [{"start": round(float(b), 5),
                  "duration": round(float(beats[i+1] - b), 5) if i + 1 < len(beats) else 0.5,
                  "confidence": 0.85}
                 for i, b in enumerate(beats)]

    # ── MD5 ──────────────────────────────────────────────────────────────
    md5 = hashlib.md5(y.tobytes()).hexdigest()

    # ── Build output ─────────────────────────────────────────────────────
    result = {
        # Audio features (Spotify-style)
        "acousticness": round(acousticness, 5),
        "analysis_url": f"local://audio-analysis/{track_id}",
        "danceability": round(danceability, 3),
        "duration_ms": int(duration * 1000),
        "energy": round(energy, 3),
        "id": track_id,
        "instrumentalness": round(instrumentalness, 5),
        "key": key,
        "liveness": round(liveness, 4),
        "loudness": round(loudness, 3),
        "mode": mode,
        "speechiness": round(speechiness, 4),
        "tempo": round(tempo, 3),
        "time_signature": time_sig,
        "track_href": f"local://tracks/{track_id}",
        "type": "audio_features",
        "uri": f"local:track:{track_id}",
        "valence": round(valence, 3),
        # Audio analysis (Spotify-style)
        "track": {
            "num_samples": num_samples,
            "duration": round(duration, 5),
            "sample_md5": md5,
            "offset_seconds": 0,
            "window_seconds": 0,
            "analysis_sample_rate": ANALYSIS_SR,
            "analysis_channels": 1,
            "end_of_fade_in": round(fade_in, 5),
            "start_of_fade_out": round(fade_out, 5),
            "loudness": round(loudness, 3),
            "tempo": round(tempo, 3),
            "tempo_confidence": round(min(1.0, 0.5 + 0.3 * (1.0 - abs(tempo - 120) / 60)), 3),
            "time_signature": time_sig,
            "time_signature_confidence": round(ts_conf, 3),
            "key": key,
            "key_confidence": round(key_conf, 3),
            "mode": mode,
            "mode_confidence": round(key_conf * 0.9, 3),
            "codestring": "",
            "code_version": 3.15,
            "echoprintstring": "",
            "echoprint_version": 4.15,
            "synchstring": "",
            "synch_version": 1,
            "rhythmstring": "",
            "rhythm_version": 1,
        },
        "bars": bars,
        "beats": beat_list,
        "sections": sections,
        "segments": segments,
        "tatums": tatums,
    }
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Bulk processor (parallel)
# ──────────────────────────────────────────────────────────────────────────────
def process_bulk(
    paths: list[Path],
    max_workers: int | None = None,
) -> list[dict[str, Any]]:
    """
    Analyze multiple audio files in parallel.

    Parameters
    ----------
    paths : list of Path
        Audio file paths (any format ffmpeg supports).
    max_workers : int, optional
        Number of parallel processes (defaults to CPU count).

    Returns
    -------
    list of dict
        Spotify-style audio features + analysis for each track, in input order.
    """
    import os
    if max_workers is None:
        max_workers = min(len(paths), os.cpu_count() or 4)

    results: dict[int, dict] = {}

    pbar = tqdm(total=len(paths), desc="Processing tracks", unit="track")
    if max_workers <= 1 or len(paths) == 1:
        # Sequential for single file or single worker
        for i, p in enumerate(paths):
            pbar.update(1)
            results[i] = analyze_track(p)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(analyze_track, p): i for i, p in enumerate(paths)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"ERROR on {paths[idx].name}: {e}", file=sys.stderr)
                    results[idx] = {"error": str(e), "file": str(paths[idx])}
    pbar.close()
    return [results[i] for i in range(len(paths))]