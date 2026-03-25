"""Pure audio helper functions — no PyQt6, no sounddevice imports.

All functions are deterministic and fully testable in CI without hardware.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np


# ── Text normalisation ─────────────────────────────────────────────────────────

def normalize_text(text: str) -> str:
    """NFD decompose, strip diacritics, lowercase, collapse whitespace."""
    nfd = unicodedata.normalize("NFD", text)
    ascii_only = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return re.sub(r"\s+", " ", ascii_only.lower()).strip()


def wake_word_in_text(text: str, wake_word: str) -> bool:
    """Return True if *wake_word* appears as a whole word in *text*.

    Both sides are accent-normalised before comparison so that accented
    transcriptions ("vóx") still match the configured wake word ("vox").
    """
    norm_text  = normalize_text(text)
    norm_wake  = normalize_text(wake_word)
    pattern    = rf"\b{re.escape(norm_wake)}\b"
    return bool(re.search(pattern, norm_text))


def extract_post_wake(text: str, wake_word: str) -> str:
    """Return the portion of *text* that follows the first occurrence of *wake_word*.

    Both inputs are normalised; the return value is the normalised tail.
    Returns ``""`` when the wake word is not found or nothing follows it.
    """
    norm_text = normalize_text(text)
    norm_wake = normalize_text(wake_word)
    pattern   = rf"\b{re.escape(norm_wake)}\b"
    match     = re.search(pattern, norm_text)
    if match:
        return norm_text[match.end():].strip()
    return ""


# ── RMS / level helpers ────────────────────────────────────────────────────────

def compute_rms(audio: np.ndarray) -> float:
    """Return the RMS amplitude of *audio* (float32 mono array, –1.0…1.0).

    Returns 0.0 for empty or all-zero arrays.
    """
    if audio is None or audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def normalize_level(rms: float, reference: float = 0.15) -> float:
    """Map *rms* to a 0.0–1.0 display level using *reference* as the ceiling."""
    if reference <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, rms / reference)))


# ── Energy / VAD pre-screening ─────────────────────────────────────────────────

def has_sufficient_energy(
    audio: np.ndarray,
    noise_floor: float,
    speech_margin: float = 3.5,
) -> bool:
    """Return True when the audio RMS is *speech_margin*× above *noise_floor*.

    Use this to skip Whisper calls on silent or near-silent chunks.
    A *noise_floor* of 0.0 falls back to a small absolute threshold (0.002).
    """
    rms       = compute_rms(audio)
    threshold = max(noise_floor * speech_margin, 0.002)
    return rms >= threshold


# ── Noise floor tracking ───────────────────────────────────────────────────────

def update_noise_floor(current: float, new_rms: float, alpha: float = 0.97) -> float:
    """Exponential moving average of the noise floor.

    *alpha* close to 1 gives slow adaptation (persistent floor);
    lower values react faster to environment changes.
    """
    return alpha * current + (1.0 - alpha) * new_rms


def estimate_noise_floor(audio: np.ndarray, percentile: float = 20.0) -> float:
    """Estimate ambient noise floor from *audio* using the *percentile*-th frame RMS.

    Audio is split into 512-sample frames; the RMS of each frame is computed.
    The *percentile*-th lowest value (default 20th) is returned as the floor.
    Returns 0.0 for empty input.
    """
    if audio is None or audio.size == 0:
        return 0.0
    frame_size = 512
    frames = [
        audio[i : i + frame_size]
        for i in range(0, len(audio) - frame_size + 1, frame_size)
    ]
    if not frames:
        return compute_rms(audio)
    rms_values = np.array([compute_rms(f) for f in frames])
    return float(np.percentile(rms_values, percentile))


def estimate_speech_rms(audio: np.ndarray, noise_floor: float) -> float:
    """Return mean RMS of frames likely to contain speech (above 2× noise floor).

    If no frames qualify, returns the overall audio RMS.
    """
    if audio is None or audio.size == 0:
        return 0.0
    frame_size = 512
    threshold  = noise_floor * 2.0
    frames = [
        audio[i : i + frame_size]
        for i in range(0, len(audio) - frame_size + 1, frame_size)
    ]
    if not frames:
        return compute_rms(audio)
    speech_rms = [compute_rms(f) for f in frames if compute_rms(f) > threshold]
    return float(np.mean(speech_rms)) if speech_rms else compute_rms(audio)


# ── Calibration helpers ────────────────────────────────────────────────────────

def suggest_silence_threshold(noise_floor: float, margin: float = 2.5) -> float:
    """Recommend a silence_threshold value based on measured noise floor.

    Clamps the result to the range [0.002, 0.15].
    """
    suggested = noise_floor * margin
    return float(min(0.15, max(0.002, suggested)))


def signal_quality_label(noise_floor: float, speech_rms: float) -> tuple[str, str]:
    """Return a (label, explanation) tuple describing signal quality.

    Labels: ``"good"``, ``"fair"``, ``"poor"``, ``"no_signal"``.
    """
    if speech_rms <= 0.001:
        return "no_signal", "No speech detected — check microphone connection."

    snr = speech_rms / noise_floor if noise_floor > 0.0 else float("inf")

    if snr >= 6.0 and noise_floor < 0.01:
        return "good", "Clean signal with low background noise."
    if snr >= 3.5:
        return "fair", "Acceptable signal; some background noise present."
    if snr >= 2.0:
        return "poor", "High background noise — consider moving to a quieter space."
    return "poor", "Very noisy environment — voice may not be recognised reliably."


# ── Clipping detection ─────────────────────────────────────────────────────────

def compute_clipping_fraction(audio: np.ndarray, threshold: float = 0.98) -> float:
    """Return the fraction of samples whose absolute value exceeds *threshold*.

    Values near 1.0 indicate the microphone input is saturated / clipping.
    Returns 0.0 for empty input.
    """
    if audio is None or audio.size == 0:
        return 0.0
    clipped = np.sum(np.abs(audio) >= threshold)
    return float(clipped) / float(audio.size)
