import yaml
import os
from typing import Any


DEFAULT_CONFIG = {
    "hotkey": "alt",
    "language": "en",
    "whisper_model": "base",
    "whisper_device": "cpu",
    "whisper_compute_type": "int8",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen2.5:14b",
    "tts_enabled": True,
    "piper_path": "piper",
    "voice_model": "en_US-ryan-high.onnx",
    "mic_device": None,
    "output_device": None,
    "search_dirs": [
        "~/Documents",
        "~/Downloads",
        "~/Desktop",
    ],
    "app_aliases": {
        # URI schemes — launch the installed app directly via Windows registry
        "discord": "discord://",
        "spotify": "spotify:",
        # Executables in PATH or App Paths registry
        "chrome": "chrome",
        "firefox": "firefox",
        "vscode": "code",
        "vs code": "code",
        "notepad": "notepad",
        "calculator": "calc",
        "explorer": "explorer",
        "paint": "mspaint",
        "steam": "steam://open/main",
    },
    "allowed_actions": [
        "open_app",
        "close_app",
        "set_volume",
        "mute_volume",
        "play_pause_media",
        "next_track",
        "prev_track",
        "search_file",
        "open_url",
        "type_text",
        "take_screenshot",
        "show_time",
        "show_battery",
    ],
}


class Config:
    def __init__(self, path: str = None):
        if path is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            path = os.path.join(base, "config", "settings.yaml")

        self._path = path
        self._data = dict(DEFAULT_CONFIG)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                self._data.update(loaded)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(DEFAULT_CONFIG, f, allow_unicode=True, default_flow_style=False)
            print(f"[Config] Created default config at {path}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value

    def save(self):
        """Persist current in-memory values back to the YAML file."""
        if os.path.exists(self._path):
            with open(self._path, "r", encoding="utf-8") as f:
                on_disk = yaml.safe_load(f) or {}
        else:
            on_disk = {}
        on_disk.update(self._data)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(on_disk, f, allow_unicode=True, default_flow_style=False)
