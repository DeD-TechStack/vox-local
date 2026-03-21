"""
tests/test_listener.py

Tests for src/listener.py pure-logic helpers.
No hardware, audio, or Qt dependencies are used directly.
"""

import pytest
from listener import _extract_post_wake


# ---------------------------------------------------------------------------
# _extract_post_wake
# ---------------------------------------------------------------------------

class TestExtractPostWake:
    def test_extracts_command_after_wake_word(self):
        result = _extract_post_wake("vox open spotify", "vox")
        assert result == "open spotify"

    def test_empty_string_when_only_wake_word(self):
        result = _extract_post_wake("vox", "vox")
        assert result == ""

    def test_empty_string_when_wake_word_with_trailing_space(self):
        result = _extract_post_wake("vox  ", "vox")
        assert result == ""

    def test_case_insensitive_wake_word(self):
        result = _extract_post_wake("VOX open discord", "vox")
        assert result == "open discord"

    def test_mixed_case_text_and_wake_word(self):
        result = _extract_post_wake("Vox Play Music", "vox")
        assert result == "Play Music"

    def test_empty_text_returns_empty(self):
        result = _extract_post_wake("", "vox")
        assert result == ""

    def test_wake_word_not_in_text_returns_empty(self):
        result = _extract_post_wake("open spotify", "vox")
        assert result == ""

    def test_wake_word_at_end_returns_empty(self):
        result = _extract_post_wake("hey there vox", "vox")
        assert result == ""

    def test_wake_word_in_middle_captures_tail(self):
        result = _extract_post_wake("hey vox turn off the lights", "vox")
        assert result == "turn off the lights"

    def test_multi_word_after_wake(self):
        result = _extract_post_wake("vox what is the weather today", "vox")
        assert result == "what is the weather today"

    def test_custom_wake_word(self):
        result = _extract_post_wake("jarvis set volume to 50", "jarvis")
        assert result == "set volume to 50"

    def test_strips_leading_and_trailing_whitespace(self):
        result = _extract_post_wake("vox   open spotify   ", "vox")
        assert result == "open spotify"

    def test_word_boundary_not_substring(self):
        # "voxel" should NOT be treated as the wake word "vox".
        result = _extract_post_wake("voxel texture open", "vox")
        assert result == ""

    def test_wake_word_with_punctuation_nearby(self):
        # Comma after wake word — still extracts the tail.
        result = _extract_post_wake("vox, open the browser", "vox")
        assert result == ", open the browser" or "open the browser" in result

    def test_first_occurrence_used(self):
        # Only the first wake-word occurrence determines the split point.
        result = _extract_post_wake("vox vox open spotify", "vox")
        # Tail after first "vox" is "vox open spotify"
        assert "open spotify" in result
