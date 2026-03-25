"""Central runtime and configuration state hub for VOX.

All UI components subscribe to signals here instead of maintaining
their own redundant state.  Three distinct domains:
  - Runtime state (status, transcript, response, mic level, …)
  - Diagnostics   (structured warning/error log)
  - History       (session interaction log)
"""
from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import QObject, pyqtSignal

from utils.config import Config


class AppState(QObject):
    # ── Runtime ───────────────────────────────────────────────────────────────
    status_changed            = pyqtSignal(str)    # idle/listening/transcribing/generating/speaking/error
    ollama_ok_changed         = pyqtSignal(bool)
    transcript_changed        = pyqtSignal(str)
    response_changed          = pyqtSignal(str)
    last_action_changed       = pyqtSignal(str)
    mic_level_changed         = pyqtSignal(float)  # 0.0–1.0 normalised RMS
    language_mode_changed     = pyqtSignal(str)    # configured mode  (auto/pt/en)
    detected_language_changed = pyqtSignal(str)    # transient detected language
    speaking_started          = pyqtSignal()       # emitted from Speaker thread when TTS begins
    speaking_ended            = pyqtSignal()       # emitted from Speaker thread when TTS finishes

    # ── Diagnostics ───────────────────────────────────────────────────────────
    diagnostic_added  = pyqtSignal(dict)  # {level, message, timestamp}
    diagnostics_cleared = pyqtSignal()

    # ── History ───────────────────────────────────────────────────────────────
    history_entry_added = pyqtSignal(dict)  # {timestamp, transcript, response, action}
    history_cleared     = pyqtSignal()

    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self._config      = config
        self._status      = "idle"
        self._ollama_ok   = False
        self._transcript  = ""
        self._response    = ""
        self._last_action = ""
        self._diagnostics: list[dict] = []
        self._history:     list[dict] = []

    # ── Config ────────────────────────────────────────────────────────────────

    @property
    def config(self) -> Config:
        return self._config

    # ── Runtime state ─────────────────────────────────────────────────────────

    @property
    def status(self) -> str:
        return self._status

    @property
    def ollama_ok(self) -> bool:
        return self._ollama_ok

    @property
    def transcript(self) -> str:
        return self._transcript

    @property
    def response(self) -> str:
        return self._response

    @property
    def last_action(self) -> str:
        return self._last_action

    def set_status(self, status: str) -> None:
        self._status = status
        self.status_changed.emit(status)

    def set_ollama_ok(self, ok: bool) -> None:
        changed = ok != self._ollama_ok
        self._ollama_ok = ok
        if changed:
            self.ollama_ok_changed.emit(ok)
            if not ok:
                self.add_diagnostic("warning", "Ollama is not reachable — LLM unavailable.")

    def set_transcript(self, text: str) -> None:
        self._transcript = text
        self.transcript_changed.emit(text)

    def set_response(self, text: str) -> None:
        self._response = text
        self.response_changed.emit(text)

    def set_last_action(self, action: str) -> None:
        self._last_action = action
        self.last_action_changed.emit(action)

    def set_mic_level(self, level: float) -> None:
        self.mic_level_changed.emit(level)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def diagnostics(self) -> list[dict]:
        return list(self._diagnostics)

    def add_diagnostic(self, level: str, message: str) -> None:
        """Add a structured diagnostic entry (level: info/warning/error)."""
        entry = {
            "level":     level,
            "message":   message,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
        self._diagnostics.append(entry)
        if len(self._diagnostics) > 500:
            self._diagnostics = self._diagnostics[-500:]
        self.diagnostic_added.emit(entry)

    def clear_diagnostics(self) -> None:
        self._diagnostics.clear()
        self.diagnostics_cleared.emit()

    # ── History ───────────────────────────────────────────────────────────────

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def add_history_entry(self, transcript: str, response: str, action: str = "") -> None:
        entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "transcript": transcript,
            "response":   response,
            "action":     action,
        }
        self._history.append(entry)
        if len(self._history) > 200:
            self._history = self._history[-200:]
        self.history_entry_added.emit(entry)

    def clear_history(self) -> None:
        self._history.clear()
        self.history_cleared.emit()
