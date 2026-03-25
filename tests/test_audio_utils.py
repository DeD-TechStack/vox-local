"""Tests for src/audio_utils.py — pure helper functions.

No hardware, sounddevice, or Qt dependencies.
"""
import math

import numpy as np
import pytest

from audio_utils import (
    normalize_text,
    wake_word_in_text,
    extract_post_wake,
    compute_rms,
    normalize_level,
    has_sufficient_energy,
    update_noise_floor,
    estimate_noise_floor,
    estimate_speech_rms,
    suggest_silence_threshold,
    signal_quality_label,
    compute_clipping_fraction,
)


# ── normalize_text ─────────────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lowercases(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strips_diacritics(self):
        assert normalize_text("vóx") == "vox"
        assert normalize_text("café") == "cafe"
        assert normalize_text("naïve") == "naive"

    def test_collapses_whitespace(self):
        assert normalize_text("  open   spotify  ") == "open spotify"

    def test_empty_string(self):
        assert normalize_text("") == ""

    def test_already_normalized(self):
        assert normalize_text("vox open spotify") == "vox open spotify"

    def test_mixed_accents_and_case(self):
        assert normalize_text("Vóx Açaí") == "vox acai"


# ── wake_word_in_text ──────────────────────────────────────────────────────────

class TestWakeWordInText:
    def test_basic_match(self):
        assert wake_word_in_text("vox open spotify", "vox") is True

    def test_case_insensitive(self):
        assert wake_word_in_text("VOX open discord", "vox") is True
        assert wake_word_in_text("Vox play music", "vox") is True

    def test_accent_normalised(self):
        assert wake_word_in_text("vóx open browser", "vox") is True
        assert wake_word_in_text("vox open browser", "vóx") is True

    def test_word_boundary_no_false_positive(self):
        assert wake_word_in_text("voxel texture", "vox") is False
        assert wake_word_in_text("convox", "vox") is False

    def test_wake_word_at_start(self):
        assert wake_word_in_text("vox what time is it", "vox") is True

    def test_wake_word_in_middle(self):
        assert wake_word_in_text("hey vox turn off lights", "vox") is True

    def test_wake_word_at_end(self):
        assert wake_word_in_text("hey there vox", "vox") is True

    def test_wake_word_not_present(self):
        assert wake_word_in_text("open spotify please", "vox") is False

    def test_empty_text(self):
        assert wake_word_in_text("", "vox") is False

    def test_custom_wake_word(self):
        assert wake_word_in_text("jarvis set volume 50", "jarvis") is True
        assert wake_word_in_text("jarvisapp do something", "jarvis") is False

    def test_single_word_text_matching(self):
        assert wake_word_in_text("vox", "vox") is True


# ── extract_post_wake ──────────────────────────────────────────────────────────

class TestExtractPostWake:
    def test_basic_extraction(self):
        assert extract_post_wake("vox open spotify", "vox") == "open spotify"

    def test_only_wake_word(self):
        assert extract_post_wake("vox", "vox") == ""

    def test_wake_word_not_present(self):
        assert extract_post_wake("open spotify", "vox") == ""

    def test_case_insensitive(self):
        assert extract_post_wake("VOX play music", "vox") == "play music"

    def test_accent_normalised(self):
        result = extract_post_wake("vóx open browser", "vox")
        assert result == "open browser"

    def test_word_boundary_respected(self):
        assert extract_post_wake("voxel texture open", "vox") == ""

    def test_empty_text(self):
        assert extract_post_wake("", "vox") == ""

    def test_wake_at_end(self):
        assert extract_post_wake("hey there vox", "vox") == ""

    def test_first_occurrence_used(self):
        result = extract_post_wake("vox vox open spotify", "vox")
        assert "open spotify" in result

    def test_whitespace_stripped(self):
        assert extract_post_wake("vox   open spotify   ", "vox") == "open spotify"


# ── compute_rms ───────────────────────────────────────────────────────────────

class TestComputeRms:
    def test_constant_signal(self):
        audio = np.full(1000, 0.5, dtype=np.float32)
        assert math.isclose(compute_rms(audio), 0.5, rel_tol=1e-5)

    def test_silence(self):
        audio = np.zeros(1000, dtype=np.float32)
        assert compute_rms(audio) == 0.0

    def test_sine_wave_rms(self):
        t = np.linspace(0, 2 * np.pi, 16000, endpoint=False)
        audio = np.sin(t).astype(np.float32)
        # RMS of a pure sine is amplitude / sqrt(2)
        assert math.isclose(compute_rms(audio), 1.0 / math.sqrt(2), rel_tol=1e-3)

    def test_empty_array(self):
        assert compute_rms(np.array([], dtype=np.float32)) == 0.0

    def test_none_input(self):
        assert compute_rms(None) == 0.0

    def test_returns_float(self):
        assert isinstance(compute_rms(np.ones(10, dtype=np.float32)), float)


# ── normalize_level ────────────────────────────────────────────────────────────

class TestNormalizeLevel:
    def test_zero_rms(self):
        assert normalize_level(0.0) == 0.0

    def test_at_reference(self):
        assert normalize_level(0.15, reference=0.15) == 1.0

    def test_above_reference_clamped(self):
        assert normalize_level(0.30, reference=0.15) == 1.0

    def test_half_reference(self):
        assert math.isclose(normalize_level(0.075, reference=0.15), 0.5, rel_tol=1e-5)

    def test_custom_reference(self):
        assert math.isclose(normalize_level(0.10, reference=0.20), 0.5, rel_tol=1e-5)

    def test_zero_reference_returns_zero(self):
        assert normalize_level(0.1, reference=0.0) == 0.0

    def test_negative_rms_clamped_to_zero(self):
        assert normalize_level(-0.05) == 0.0


# ── has_sufficient_energy ──────────────────────────────────────────────────────

class TestHasSufficientEnergy:
    def _audio(self, rms: float, size: int = 4096) -> np.ndarray:
        return np.full(size, rms, dtype=np.float32)

    def test_above_threshold(self):
        audio = self._audio(0.05)
        assert has_sufficient_energy(audio, noise_floor=0.01, speech_margin=3.5) is True

    def test_below_threshold(self):
        audio = self._audio(0.01)
        assert has_sufficient_energy(audio, noise_floor=0.01, speech_margin=3.5) is False

    def test_zero_noise_floor_uses_absolute_minimum(self):
        # With noise_floor=0, threshold falls back to 0.002
        audio = self._audio(0.003)
        assert has_sufficient_energy(audio, noise_floor=0.0) is True
        silent = self._audio(0.001)
        assert has_sufficient_energy(silent, noise_floor=0.0) is False

    def test_exactly_at_threshold_is_sufficient(self):
        noise = 0.01
        audio = self._audio(noise * 3.5)
        assert has_sufficient_energy(audio, noise_floor=noise, speech_margin=3.5) is True

    def test_empty_audio_is_insufficient(self):
        audio = np.array([], dtype=np.float32)
        assert has_sufficient_energy(audio, noise_floor=0.001) is False


# ── update_noise_floor ─────────────────────────────────────────────────────────

class TestUpdateNoiseFloor:
    def test_converges_toward_new_rms(self):
        current = 0.0
        for _ in range(200):
            current = update_noise_floor(current, 0.01, alpha=0.97)
        assert math.isclose(current, 0.01, rel_tol=0.01)

    def test_slow_decay_with_high_alpha(self):
        floor = update_noise_floor(0.05, 0.0, alpha=0.99)
        assert floor > 0.04  # barely moves

    def test_fast_decay_with_low_alpha(self):
        floor = update_noise_floor(0.05, 0.0, alpha=0.5)
        assert math.isclose(floor, 0.025, rel_tol=1e-5)

    def test_returns_float(self):
        assert isinstance(update_noise_floor(0.01, 0.02), float)


# ── estimate_noise_floor ───────────────────────────────────────────────────────

class TestEstimateNoiseFloor:
    def test_constant_audio(self):
        audio = np.full(16000, 0.01, dtype=np.float32)
        floor = estimate_noise_floor(audio)
        assert math.isclose(floor, 0.01, rel_tol=0.01)

    def test_loud_spikes_dont_dominate(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[:512] = 0.8  # one loud frame at the start
        floor = estimate_noise_floor(audio)
        assert floor < 0.1  # noise floor should be near silence

    def test_empty_returns_zero(self):
        assert estimate_noise_floor(np.array([], dtype=np.float32)) == 0.0

    def test_none_returns_zero(self):
        assert estimate_noise_floor(None) == 0.0

    def test_returns_float(self):
        audio = np.ones(1024, dtype=np.float32) * 0.05
        assert isinstance(estimate_noise_floor(audio), float)


# ── estimate_speech_rms ────────────────────────────────────────────────────────

class TestEstimateSpeechRms:
    def test_above_noise_frames_selected(self):
        audio = np.zeros(4096, dtype=np.float32)
        # frames 512-1023 contain "speech"
        audio[512:1024] = 0.2
        rms = estimate_speech_rms(audio, noise_floor=0.01)
        assert rms > 0.05  # higher than the silent average

    def test_all_silent_falls_back_to_overall_rms(self):
        audio = np.full(4096, 0.001, dtype=np.float32)
        rms = estimate_speech_rms(audio, noise_floor=0.05)
        assert math.isclose(rms, compute_rms(audio), rel_tol=0.05)

    def test_empty_returns_zero(self):
        assert estimate_speech_rms(np.array([], dtype=np.float32), 0.01) == 0.0

    def test_none_returns_zero(self):
        assert estimate_speech_rms(None, 0.01) == 0.0


# ── suggest_silence_threshold ─────────────────────────────────────────────────

class TestSuggestSilenceThreshold:
    def test_basic_suggestion(self):
        result = suggest_silence_threshold(0.01, margin=2.5)
        assert math.isclose(result, 0.025, rel_tol=1e-5)

    def test_clamped_to_minimum(self):
        assert suggest_silence_threshold(0.0001) >= 0.002

    def test_clamped_to_maximum(self):
        assert suggest_silence_threshold(0.5) <= 0.15

    def test_returns_float(self):
        assert isinstance(suggest_silence_threshold(0.01), float)

    def test_zero_noise_floor(self):
        result = suggest_silence_threshold(0.0)
        assert result == 0.002  # minimum clamp


# ── signal_quality_label ───────────────────────────────────────────────────────

class TestSignalQualityLabel:
    def test_good_signal(self):
        label, _ = signal_quality_label(noise_floor=0.003, speech_rms=0.08)
        assert label == "good"

    def test_fair_signal(self):
        label, _ = signal_quality_label(noise_floor=0.015, speech_rms=0.07)
        assert label == "fair"

    def test_poor_signal_high_noise(self):
        label, _ = signal_quality_label(noise_floor=0.05, speech_rms=0.08)
        assert label == "poor"

    def test_no_signal(self):
        label, _ = signal_quality_label(noise_floor=0.005, speech_rms=0.0)
        assert label == "no_signal"

    def test_returns_tuple_of_two_strings(self):
        label, explanation = signal_quality_label(0.005, 0.05)
        assert isinstance(label, str)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_zero_noise_floor_with_speech(self):
        label, _ = signal_quality_label(noise_floor=0.0, speech_rms=0.05)
        # SNR is infinite when noise_floor=0 — should be "good"
        assert label == "good"


# ── compute_clipping_fraction ─────────────────────────────────────────────────

class TestComputeClippingFraction:
    def test_no_clipping(self):
        audio = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)
        assert compute_clipping_fraction(audio) == 0.0

    def test_fully_clipped(self):
        audio = np.ones(1000, dtype=np.float32)
        assert compute_clipping_fraction(audio) == 1.0

    def test_half_clipped(self):
        audio = np.zeros(1000, dtype=np.float32)
        audio[:500] = 1.0
        frac = compute_clipping_fraction(audio)
        assert math.isclose(frac, 0.5, rel_tol=1e-5)

    def test_negative_clipping(self):
        audio = np.full(100, -1.0, dtype=np.float32)
        assert compute_clipping_fraction(audio) == 1.0

    def test_custom_threshold(self):
        audio = np.full(100, 0.9, dtype=np.float32)
        assert compute_clipping_fraction(audio, threshold=0.85) == 1.0
        assert compute_clipping_fraction(audio, threshold=0.95) == 0.0

    def test_empty_returns_zero(self):
        assert compute_clipping_fraction(np.array([], dtype=np.float32)) == 0.0

    def test_none_returns_zero(self):
        assert compute_clipping_fraction(None) == 0.0
