"""
src/ui/settings_dialog.py

Expanded settings dialog — styled consistently with the dark-purple VOX aesthetic.
Covers activation mode, wake word, push-to-talk key, language, Ollama model,
TTS toggle, and voice model path.
"""

import os

from PyQt6.QtWidgets import (
    QDialog, QFrame, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QCheckBox, QPushButton,
    QFileDialog,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication

from utils.config import Config


# ─── Stylesheet constants ─────────────────────────────────────────────────────

_SS_BASE = "background: transparent; border: none;"

_SS_CONTAINER = """
QFrame#container {
    background-color: rgba(13, 11, 18, 245);
    border-radius: 14px;
    border: 1px solid rgba(168, 85, 247, 90);
}
"""

_SS_LABEL = f"color: rgba(255,255,255,0.5); font-size: 11px; {_SS_BASE}"

_SS_INPUT = """
QLineEdit {
    background-color: rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.85);
    border: 1px solid rgba(168,85,247,50);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
}
QLineEdit:focus { border-color: #a855f7; }
QLineEdit:disabled {
    background-color: rgba(255,255,255,0.02);
    color: rgba(255,255,255,0.2);
    border-color: rgba(168,85,247,20);
}
"""

_SS_COMBO = """
QComboBox {
    background-color: rgba(255,255,255,0.05);
    color: rgba(255,255,255,0.85);
    border: 1px solid rgba(168,85,247,50);
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
}
QComboBox:focus { border-color: #a855f7; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: rgb(20, 17, 28);
    color: rgba(255,255,255,0.85);
    selection-background-color: rgba(168,85,247,60);
    border: 1px solid rgba(168,85,247,80);
    outline: none;
}
"""

_SS_CHECKBOX = """
QCheckBox {
    color: rgba(255,255,255,0.75);
    font-size: 12px;
    background: transparent;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid rgba(168,85,247,80);
    border-radius: 3px;
    background: rgba(255,255,255,0.04);
}
QCheckBox::indicator:checked {
    background: #a855f7;
    border-color: #a855f7;
}
"""

_SS_BTN_CANCEL = """
QPushButton {
    background: transparent;
    color: rgba(255,255,255,0.5);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 6px;
    padding: 5px 16px;
    font-size: 12px;
}
QPushButton:hover { border-color: rgba(255,255,255,0.3); color: rgba(255,255,255,0.8); }
"""

_SS_BTN_SAVE = """
QPushButton {
    background-color: #a855f7;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 5px 20px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton:hover { background-color: #9333ea; }
QPushButton:pressed { background-color: #7e22ce; }
"""

_SS_BTN_BROWSE = """
QPushButton {
    background-color: rgba(168,85,247,15);
    color: rgba(168,85,247,200);
    border: 1px solid rgba(168,85,247,60);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 11px;
}
QPushButton:hover { background-color: rgba(168,85,247,30); }
"""


def _label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(_SS_LABEL)
    return lbl


class SettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self._config = config
        self._setup_window()
        self._setup_ui()
        self._populate()

    # ── Window setup ──────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(460, 520)
        screen = QGuiApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - 460) // 2,
            (screen.height() - 520) // 2,
        )

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        self._container = QFrame(self)
        self._container.setObjectName("container")
        self._container.setFixedSize(460, 520)
        self._container.setStyleSheet(_SS_CONTAINER)

        root = QVBoxLayout(self._container)
        root.setContentsMargins(24, 18, 24, 18)
        root.setSpacing(0)

        # Title row
        title_row = QHBoxLayout()
        title = QLabel("SETTINGS")
        title.setStyleSheet(
            f"color: rgba(255,255,255,0.9); font-size: 11px; "
            f"letter-spacing: 4px; font-weight: 600; {_SS_BASE}"
        )
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(22, 22)
        close_btn.setStyleSheet(
            "QPushButton { background: transparent; color: rgba(255,255,255,0.3); "
            "border: none; font-size: 13px; } "
            "QPushButton:hover { color: rgba(255,255,255,0.8); }"
        )
        close_btn.clicked.connect(self.reject)
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(close_btn)
        root.addLayout(title_row)

        root.addSpacing(10)
        root.addWidget(_divider())
        root.addSpacing(14)

        # ── Activation mode ───────────────────────────────────────────────────
        root.addWidget(_label("Activation Mode"))
        root.addSpacing(5)
        self._mode_combo = QComboBox()
        self._mode_combo.setStyleSheet(_SS_COMBO)
        self._mode_combo.addItem("Wake Word", userData="wake_word")
        self._mode_combo.addItem("Push to Talk", userData="push_to_talk")
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        root.addWidget(self._mode_combo)

        root.addSpacing(12)

        # ── Push-to-talk key ─────────────────────────────────────────────────
        self._ptt_label = _label("Push-to-Talk Key")
        root.addWidget(self._ptt_label)
        root.addSpacing(5)
        self._ptt_input = QLineEdit()
        self._ptt_input.setStyleSheet(_SS_INPUT)
        self._ptt_input.setPlaceholderText("e.g. ctrl+shift")
        root.addWidget(self._ptt_input)

        root.addSpacing(12)

        # ── Wake word ────────────────────────────────────────────────────────
        self._ww_label = _label("Wake Word")
        root.addWidget(self._ww_label)
        root.addSpacing(5)
        self._ww_input = QLineEdit()
        self._ww_input.setStyleSheet(_SS_INPUT)
        self._ww_input.setPlaceholderText("e.g. vox")
        root.addWidget(self._ww_input)

        root.addSpacing(12)

        # ── Language ─────────────────────────────────────────────────────────
        root.addWidget(_label("Language"))
        root.addSpacing(5)
        self._lang_combo = QComboBox()
        self._lang_combo.setStyleSheet(_SS_COMBO)
        self._lang_combo.addItem("Auto-detect", userData="auto")
        self._lang_combo.addItem("Portuguese (PT)", userData="pt")
        self._lang_combo.addItem("English (EN)", userData="en")
        root.addWidget(self._lang_combo)

        root.addSpacing(12)

        # ── Ollama model ─────────────────────────────────────────────────────
        root.addWidget(_label("Ollama Model"))
        root.addSpacing(5)
        self._model_input = QLineEdit()
        self._model_input.setStyleSheet(_SS_INPUT)
        self._model_input.setPlaceholderText("e.g. qwen2.5:7b")
        root.addWidget(self._model_input)

        root.addSpacing(12)

        # ── TTS enabled ───────────────────────────────────────────────────────
        self._tts_check = QCheckBox("Enable voice responses (TTS)")
        self._tts_check.setStyleSheet(_SS_CHECKBOX)
        root.addWidget(self._tts_check)

        root.addSpacing(12)

        # ── Voice model path ─────────────────────────────────────────────────
        root.addWidget(_label("Voice Model Path  (.onnx)"))
        root.addSpacing(5)
        voice_row = QHBoxLayout()
        voice_row.setSpacing(6)
        self._voice_input = QLineEdit()
        self._voice_input.setStyleSheet(_SS_INPUT)
        self._voice_input.setPlaceholderText("voices/en_US-ryan-high.onnx")
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(_SS_BTN_BROWSE)
        browse_btn.setFixedHeight(30)
        browse_btn.clicked.connect(self._browse_voice)
        voice_row.addWidget(self._voice_input)
        voice_row.addWidget(browse_btn)
        root.addLayout(voice_row)

        root.addStretch()

        root.addWidget(_divider())
        root.addSpacing(12)

        # ── Bottom row ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet(f"font-size: 11px; {_SS_BASE}")
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(_SS_BTN_CANCEL)
        btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Save")
        btn_save.setStyleSheet(_SS_BTN_SAVE)
        btn_save.clicked.connect(self._on_save)

        btn_row.addWidget(self._status)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addSpacing(8)
        btn_row.addWidget(btn_save)
        root.addLayout(btn_row)

    # ── Population ────────────────────────────────────────────────────────────

    def _populate(self):
        mode = self._config.get("activation_mode", "wake_word")
        for i in range(self._mode_combo.count()):
            if self._mode_combo.itemData(i) == mode:
                self._mode_combo.setCurrentIndex(i)
                break

        self._ptt_input.setText(self._config.get("push_to_talk_key", "ctrl+shift"))
        self._ww_input.setText(self._config.get("wake_word", "vox"))

        lang = self._config.get("language", "auto")
        for i in range(self._lang_combo.count()):
            if self._lang_combo.itemData(i) == lang:
                self._lang_combo.setCurrentIndex(i)
                break

        self._model_input.setText(self._config.get("ollama_model", "qwen2.5:14b"))
        self._tts_check.setChecked(bool(self._config.get("tts_enabled", True)))
        self._voice_input.setText(self._config.get("voice_model", "voices/en_US-ryan-high.onnx"))

        self._on_mode_changed()  # set initial enabled state

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_mode_changed(self):
        is_ptt = self._mode_combo.currentData() == "push_to_talk"
        self._ptt_input.setEnabled(is_ptt)
        self._ptt_label.setStyleSheet(
            _SS_LABEL if is_ptt
            else f"color: rgba(255,255,255,0.25); font-size: 11px; {_SS_BASE}"
        )
        is_ww = not is_ptt
        self._ww_input.setEnabled(is_ww)
        self._ww_label.setStyleSheet(
            _SS_LABEL if is_ww
            else f"color: rgba(255,255,255,0.25); font-size: 11px; {_SS_BASE}"
        )

    def _browse_voice(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Voice Model",
            os.path.expanduser("~"),
            "ONNX Model (*.onnx)",
        )
        if path:
            self._voice_input.setText(path)

    def _on_save(self):
        self._config.set("activation_mode",  self._mode_combo.currentData())
        self._config.set("push_to_talk_key", self._ptt_input.text().strip() or "ctrl+shift")
        self._config.set("wake_word",        self._ww_input.text().strip() or "vox")
        self._config.set("language",         self._lang_combo.currentData())
        self._config.set("ollama_model",     self._model_input.text().strip())
        self._config.set("tts_enabled",      self._tts_check.isChecked())
        self._config.set("voice_model",      self._voice_input.text().strip())
        try:
            self._config.save()
            self._status.setText("Saved.")
            self._status.setStyleSheet(f"color: #1d9e75; font-size: 11px; {_SS_BASE}")
            QTimer.singleShot(700, self.accept)
        except Exception as e:
            self._status.setText(f"Error: {e}")
            self._status.setStyleSheet(f"color: #ef4444; font-size: 11px; {_SS_BASE}")

    # ── Drag ─────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if hasattr(self, "_drag_pos"):
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()

    def paintEvent(self, event):
        pass


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _divider() -> QFrame:
    d = QFrame()
    d.setFrameShape(QFrame.Shape.HLine)
    d.setFixedHeight(1)
    d.setStyleSheet("background: rgba(168,85,247,30); border: none;")
    return d
