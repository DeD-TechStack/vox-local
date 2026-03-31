"""
tests/test_speaker.py

Tests for src/speaker.py.
Audio I/O (sounddevice, wave, subprocess) is mocked.
The critical invariant tested: when the primary output device fails and
playback falls back to the system default, the sample rate used must
match the actual sample rate of the audio array (which may have been
resampled from the original WAV framerate).
"""

import io
import os
import struct
import tempfile
import wave
import yaml
import pytest
from unittest.mock import MagicMock, patch, call

from utils.config import Config
from speaker import Speaker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path, extra: dict = None) -> Config:
    data = {}
    if extra:
        data.update(extra)
    p = tmp_path / "settings.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return Config(path=str(p))


def _write_wav(path: str, framerate: int = 22050, n_frames: int = 1000):
    """Write a minimal mono 16-bit WAV file at the given framerate."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(framerate)
        wf.writeframes(b"\x00\x01" * n_frames)  # 1000 frames of silence
    return path


# ---------------------------------------------------------------------------
# Fallback sample-rate correctness
# ---------------------------------------------------------------------------

class TestPlaybackFallbackRate:
    """
    When the configured output device fails, sd.play is retried on the default
    device.  The second sd.play call must use target_rate — the actual sample
    rate of the audio array — not the original WAV framerate.

    Bug scenario:
      WAV framerate  = 22050 Hz
      device rate    = 44100 Hz  → audio is resampled to 44100 Hz
      device fails   → fallback must play at 44100 Hz (not 22050 Hz)
    """

    def test_fallback_uses_target_rate_after_resampling(self, tmp_path):
        cfg = _make_config(tmp_path, {"output_device": 5})
        spk = Speaker(cfg)

        wav_path = _write_wav(str(tmp_path / "test.wav"), framerate=22050, n_frames=500)

        fake_device_info = {"default_samplerate": 44100.0}

        play_calls = []

        def fake_play(audio, samplerate, device):
            play_calls.append({"rate": samplerate, "device": device, "len": len(audio)})
            if device == 5:
                raise OSError("device unavailable")

        with patch("sounddevice.query_devices", return_value=fake_device_info), \
             patch("sounddevice.play", side_effect=fake_play), \
             patch("sounddevice.wait"):
            spk._play_wav(wav_path, output_device=5)

        assert len(play_calls) == 2, "Expected primary attempt + one fallback"

        primary  = play_calls[0]
        fallback = play_calls[1]

        # Primary should use the device rate (44100) after resampling.
        assert primary["rate"] == 44100
        assert primary["device"] == 5

        # Fallback must also use 44100 (the resampled audio's rate).
        # Using 22050 here would play back at the wrong pitch/speed.
        assert fallback["rate"] == 44100, (
            f"Fallback used rate {fallback['rate']} but audio was resampled to 44100; "
            "this would produce wrong-speed playback."
        )
        assert fallback["device"] is None

    def test_fallback_uses_framerate_when_no_resampling_needed(self, tmp_path):
        """
        If the device rate already matches the WAV framerate, no resampling
        occurs and the fallback must also use that same rate.
        """
        cfg = _make_config(tmp_path, {"output_device": 5})
        spk = Speaker(cfg)

        wav_path = _write_wav(str(tmp_path / "test.wav"), framerate=44100, n_frames=500)

        fake_device_info = {"default_samplerate": 44100.0}
        play_calls = []

        def fake_play(audio, samplerate, device):
            play_calls.append({"rate": samplerate, "device": device})
            if device == 5:
                raise OSError("device unavailable")

        with patch("sounddevice.query_devices", return_value=fake_device_info), \
             patch("sounddevice.play", side_effect=fake_play), \
             patch("sounddevice.wait"):
            spk._play_wav(wav_path, output_device=5)

        assert play_calls[0]["rate"] == 44100
        assert play_calls[1]["rate"] == 44100

    def test_no_fallback_when_primary_succeeds(self, tmp_path):
        cfg = _make_config(tmp_path, {"output_device": 5})
        spk = Speaker(cfg)

        wav_path = _write_wav(str(tmp_path / "test.wav"), framerate=22050, n_frames=100)

        fake_device_info = {"default_samplerate": 44100.0}
        play_calls = []

        with patch("sounddevice.query_devices", return_value=fake_device_info), \
             patch("sounddevice.play", side_effect=lambda audio, samplerate, device: play_calls.append(device)), \
             patch("sounddevice.wait"):
            spk._play_wav(wav_path, output_device=5)

        assert len(play_calls) == 1
        assert play_calls[0] == 5

    def test_no_resampling_when_output_device_is_none(self, tmp_path):
        cfg = _make_config(tmp_path)
        spk = Speaker(cfg)

        wav_path = _write_wav(str(tmp_path / "test.wav"), framerate=22050, n_frames=100)

        play_calls = []
        with patch("sounddevice.play", side_effect=lambda audio, samplerate, device: play_calls.append(samplerate)), \
             patch("sounddevice.wait"):
            spk._play_wav(wav_path, output_device=None)

        assert len(play_calls) == 1
        assert play_calls[0] == 22050


# ---------------------------------------------------------------------------
# Piper error logging
# ---------------------------------------------------------------------------

class TestPiperErrorLogging:
    def test_stderr_is_logged_on_nonzero_exit(self, tmp_path):
        cfg = _make_config(tmp_path, {"tts_enabled": True})
        spk = Speaker(cfg)
        spk.piper_path  = "piper"
        spk.voice_model = "model.onnx"

        proc_mock = MagicMock()
        proc_mock.returncode = 1
        proc_mock.stderr = b"model not found\n"

        with patch("subprocess.run", return_value=proc_mock), \
             patch("os.path.exists", return_value=True), \
             patch("speaker.log") as mock_log:
            spk._speak_blocking("hello")
            error_calls = [str(c) for c in mock_log.error.call_args_list]
            assert any("model not found" in c for c in error_calls), (
                "Piper stderr was not logged on non-zero exit"
            )

    def test_empty_stderr_still_logs_exit_code(self, tmp_path):
        cfg = _make_config(tmp_path, {"tts_enabled": True})
        spk = Speaker(cfg)
        spk.piper_path  = "piper"
        spk.voice_model = "model.onnx"

        proc_mock = MagicMock()
        proc_mock.returncode = 1
        proc_mock.stderr = b""

        with patch("subprocess.run", return_value=proc_mock), \
             patch("os.path.exists", return_value=True), \
             patch("speaker.log") as mock_log:
            spk._speak_blocking("hello")
            error_calls = [str(c) for c in mock_log.error.call_args_list]
            assert any("exit" in c.lower() or "code" in c.lower() for c in error_calls)


# ---------------------------------------------------------------------------
# TTS truthfulness: speaking callbacks
# ---------------------------------------------------------------------------

class TestSpeakingCallbacks:
    """on_speak_start must only fire when Piper succeeds and playback is starting.
    on_speak_end must always fire to ensure the app returns to idle.
    """

    def test_on_speak_start_does_not_fire_on_piper_failure(self, tmp_path):
        """When Piper exits non-zero, on_speak_start must NOT be called.
        The app must never enter 'speaking' state for a failed synthesis."""
        cfg = _make_config(tmp_path, {"tts_enabled": True})
        spk = Speaker(cfg)

        start_fired = []
        end_fired   = []
        spk.set_speaking_callbacks(
            on_start=lambda: start_fired.append(True),
            on_end=lambda: end_fired.append(True),
        )

        proc_mock = MagicMock()
        proc_mock.returncode = 1
        proc_mock.stderr = b"piper: model not found"

        with patch("subprocess.run", return_value=proc_mock):
            spk._speak_blocking("hello")

        assert len(start_fired) == 0, "on_speak_start must not fire when Piper fails"
        assert len(end_fired)   == 1, "on_speak_end must always fire for idle recovery"

    def test_on_speak_start_fires_only_after_piper_succeeds(self, tmp_path):
        """When Piper succeeds, on_speak_start fires once and on_speak_end fires once."""
        cfg = _make_config(tmp_path, {"tts_enabled": True})
        spk = Speaker(cfg)

        start_fired = []
        end_fired   = []
        spk.set_speaking_callbacks(
            on_start=lambda: start_fired.append(True),
            on_end=lambda: end_fired.append(True),
        )

        # Write a real WAV that Piper "produces"
        wav_src = _write_wav(str(tmp_path / "piper_out.wav"), framerate=22050, n_frames=100)

        proc_mock = MagicMock()
        proc_mock.returncode = 0
        proc_mock.stderr = b""

        def fake_run(cmd, **kwargs):
            import shutil
            out_file = cmd[cmd.index("--output_file") + 1]
            shutil.copy(wav_src, out_file)
            return proc_mock

        with patch("subprocess.run", side_effect=fake_run), \
             patch("sounddevice.play"), \
             patch("sounddevice.wait"):
            spk._speak_blocking("hello")

        assert len(start_fired) == 1, "on_speak_start must fire exactly once on success"
        assert len(end_fired)   == 1, "on_speak_end must fire exactly once on success"

    def test_on_speak_end_fires_when_piper_not_found(self, tmp_path):
        """FileNotFoundError (Piper binary missing) must not leave app stuck in a
        non-idle state — on_speak_end must fire for idle recovery."""
        cfg = _make_config(tmp_path, {"tts_enabled": True})
        spk = Speaker(cfg)

        start_fired = []
        end_fired   = []
        spk.set_speaking_callbacks(
            on_start=lambda: start_fired.append(True),
            on_end=lambda: end_fired.append(True),
        )

        with patch("subprocess.run", side_effect=FileNotFoundError("piper not found")):
            spk._speak_blocking("hello")

        assert len(start_fired) == 0, "on_speak_start must not fire when Piper is missing"
        assert len(end_fired)   == 1, "on_speak_end must fire for idle recovery"
