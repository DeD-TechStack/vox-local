"""
tests/test_executor.py

Tests for src/executor.py.
No hardware, GUI, or audio dependencies are imported.
"""

import os
import sys
import yaml
import pytest
from unittest.mock import patch, MagicMock

from utils.config import Config
from executor import Executor


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


# ---------------------------------------------------------------------------
# Allowlist enforcement
# ---------------------------------------------------------------------------

class TestAllowlist:
    def test_disallowed_action_returns_error(self, tmp_path):
        cfg = _make_config(tmp_path, {"allowed_actions": ["show_time"]})
        ex = Executor(cfg)
        result = ex.run("open_app", {"name": "notepad"})
        assert "não está permitida" in result.lower() or "not permitted" in result.lower() or "permitida" in result

    def test_allowed_action_does_not_return_permission_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex.run("show_time", {})
        assert ":" in result  # e.g. "São 14:35."

    def test_unimplemented_action_not_in_map(self, tmp_path):
        cfg = _make_config(tmp_path, {
            "allowed_actions": ["show_time", "nonexistent_action"]
        })
        ex = Executor(cfg)
        # Force it into allowed but not in _action_map
        ex.allowed_actions.add("nonexistent_action")
        result = ex.run("nonexistent_action", {})
        assert "implementada" in result or "not implemented" in result.lower()


# ---------------------------------------------------------------------------
# set_volume
# ---------------------------------------------------------------------------

class TestSetVolume:
    def test_clamps_to_100(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        # Patch out pycaw on Windows or skip OS-specific branch
        with patch.object(ex, "_set_volume", wraps=ex._set_volume):
            # We test the clamp logic directly via the internal method
            # by patching the Windows-only import block
            if os.name == "nt":
                with patch("executor.POINTER", MagicMock(), create=True), \
                     patch("executor.cast", MagicMock(), create=True):
                    try:
                        from unittest.mock import patch as p2
                        with p2.object(ex, "_set_volume") as mock_sv:
                            mock_sv.return_value = "Volume em 100%."
                            r = ex.run("set_volume", {"level": 150})
                    except Exception:
                        pass
            # Direct call — clamping happens before OS branch
            # Stub the OS-specific part
            with patch("os.name", "posix"), \
                 patch("subprocess.run", return_value=MagicMock(returncode=0)):
                result = ex._set_volume(150)
        assert "100%" in result

    def test_clamps_to_0(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        with patch("os.name", "posix"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = ex._set_volume(-50)
        assert "0%" in result

    def test_string_level_coerced(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        with patch("os.name", "posix"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)):
            result = ex._set_volume("42")
        assert "42%" in result

    def test_non_numeric_level_returns_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex._set_volume("loud")
        assert "inválido" in result.lower() or "invalid" in result.lower()

    def test_none_params_handled(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex.run("show_time", None)
        assert ":" in result  # should still work


# ---------------------------------------------------------------------------
# show_time
# ---------------------------------------------------------------------------

class TestShowTime:
    def test_show_time_returns_time_string(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex._show_time()
        assert ":" in result

    def test_show_time_contains_hours_and_minutes(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        result = ex._show_time()
        # "São HH:MM." — extract HH:MM portion
        import re
        match = re.search(r"\d{1,2}:\d{2}", result)
        assert match is not None


# ---------------------------------------------------------------------------
# show_battery
# ---------------------------------------------------------------------------

class TestShowBattery:
    def test_show_battery_returns_string(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        mock_battery = MagicMock()
        mock_battery.percent = 75
        mock_battery.power_plugged = False
        with patch("psutil.sensors_battery", return_value=mock_battery):
            result = ex._show_battery()
        assert isinstance(result, str)
        assert "75" in result

    def test_show_battery_no_sensor(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        with patch("psutil.sensors_battery", return_value=None):
            result = ex._show_battery()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_show_battery_psutil_error(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        with patch("psutil.sensors_battery", side_effect=Exception("no sensor")):
            result = ex._show_battery()
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# close_app return code handling
# ---------------------------------------------------------------------------

class TestCloseApp:
    def test_close_app_not_found_returns_message(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        mock_result = MagicMock()
        mock_result.returncode = 128
        with patch("subprocess.run", return_value=mock_result), \
             patch("os.name", "nt"):
            result = ex._close_app("nonexistent")
        assert "encontrado" in result or "not found" in result.lower()

    def test_close_app_success_returns_message(self, tmp_path):
        cfg = _make_config(tmp_path)
        ex = Executor(cfg)
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result), \
             patch("os.name", "nt"):
            result = ex._close_app("notepad")
        assert "notepad" in result.lower()
