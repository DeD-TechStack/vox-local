"""
tests/test_config.py

Tests for utils.config.Config.
"""

import os
import yaml
import pytest

from utils.config import Config, DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_wake_word(self):
        c = Config.__new__(Config)
        c._data = dict(DEFAULT_CONFIG)
        assert c.get("wake_word") == "vox"

    def test_default_activation_mode(self):
        c = Config.__new__(Config)
        c._data = dict(DEFAULT_CONFIG)
        assert c.get("activation_mode") == "wake_word"

    def test_default_push_to_talk_key(self):
        c = Config.__new__(Config)
        c._data = dict(DEFAULT_CONFIG)
        assert c.get("push_to_talk_key") == "ctrl+shift"

    def test_default_mic_device_is_none(self):
        c = Config.__new__(Config)
        c._data = dict(DEFAULT_CONFIG)
        assert c.get("mic_device") is None

    def test_default_output_device_is_none(self):
        c = Config.__new__(Config)
        c._data = dict(DEFAULT_CONFIG)
        assert c.get("output_device") is None

    def test_default_max_history(self):
        c = Config.__new__(Config)
        c._data = dict(DEFAULT_CONFIG)
        assert c.get("max_history") == 20

    def test_default_tts_enabled(self):
        c = Config.__new__(Config)
        c._data = dict(DEFAULT_CONFIG)
        assert c.get("tts_enabled") is True

    def test_missing_key_returns_fallback(self):
        c = Config.__new__(Config)
        c._data = {}
        assert c.get("nonexistent", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# YAML loading overrides defaults
# ---------------------------------------------------------------------------

class TestYamlLoading:
    def test_yaml_overrides_defaults(self, tmp_config_path):
        with open(tmp_config_path, "w") as f:
            yaml.dump({"wake_word": "jarvis", "ollama_model": "llama3:8b"}, f)
        c = Config(path=tmp_config_path)
        assert c.get("wake_word") == "jarvis"
        assert c.get("ollama_model") == "llama3:8b"

    def test_yaml_preserves_unset_defaults(self, tmp_config_path):
        with open(tmp_config_path, "w") as f:
            yaml.dump({"wake_word": "hey"}, f)
        c = Config(path=tmp_config_path)
        # Default should still be present
        assert c.get("tts_enabled") is True

    def test_nonexistent_yaml_creates_default_file(self, tmp_path):
        path = str(tmp_path / "new_dir" / "settings.yaml")
        c = Config(path=path)
        assert os.path.exists(path)
        assert c.get("wake_word") == "vox"


# ---------------------------------------------------------------------------
# Malformed YAML falls back to defaults gracefully
# ---------------------------------------------------------------------------

class TestMalformedYaml:
    def test_malformed_yaml_falls_back(self, tmp_config_path):
        with open(tmp_config_path, "w") as f:
            f.write(":::invalid yaml:::\n  bad: [unclosed")
        # Should not raise; should use defaults
        c = Config(path=tmp_config_path)
        assert c.get("wake_word") == "vox"

    def test_empty_yaml_uses_defaults(self, tmp_config_path):
        with open(tmp_config_path, "w") as f:
            f.write("")
        c = Config(path=tmp_config_path)
        assert c.get("wake_word") == "vox"


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_save_and_reload(self, tmp_config_path):
        with open(tmp_config_path, "w") as f:
            yaml.dump({"wake_word": "vox"}, f)
        c = Config(path=tmp_config_path)
        c.set("ollama_model", "mistral:7b")
        c.save()

        c2 = Config(path=tmp_config_path)
        assert c2.get("ollama_model") == "mistral:7b"
        assert c2.get("wake_word") == "vox"

    def test_set_and_get_in_memory(self, tmp_config_path):
        with open(tmp_config_path, "w") as f:
            yaml.dump({}, f)
        c = Config(path=tmp_config_path)
        c.set("language", "pt")
        assert c.get("language") == "pt"
