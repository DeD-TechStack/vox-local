"""VOX Control Center — main application window.

Multi-tab interface covering:
  Dashboard · Audio · Activation · Assistant · Actions · Aliases · Dirs · History · Diagnostics

Usage:
    cc = ControlCenter(config, app_state, speaker)
    cc.restart_listener_requested.connect(vox_app.restart_listener)
    cc.rerun_validation_requested.connect(vox_app.run_validation)
    cc.show()
"""
from __future__ import annotations

import os
import threading
from typing import Callable

import numpy as np
import sounddevice as sd

from audio_utils import (
    compute_rms,
    estimate_noise_floor,
    estimate_speech_rms,
    suggest_silence_threshold,
    signal_quality_label,
    compute_clipping_fraction,
)
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPushButton, QComboBox, QLineEdit, QCheckBox,
    QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QGridLayout,
    QFileDialog, QSizePolicy, QTextEdit, QRadioButton, QButtonGroup,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QIcon

from utils.config import Config
from ui.mic_meter import MicLevelBar


# ── Colour tokens ──────────────────────────────────────────────────────────────
_BG      = "#0e0c14"
_PANEL   = "#16131f"
_BORDER  = "#2a2338"
_ACCENT  = "#a855f7"
_TEXT    = "#e2e0ea"
_MUTED   = "#6b6680"
_SUCCESS = "#1d9e75"
_WARNING = "#f59e0b"
_ERROR   = "#ef4444"

# All known executor actions with description and risk label
_ALL_ACTIONS = [
    ("open_app",         "Open an application by name or alias",           "low"),
    ("close_app",        "Close a running application by name",            "low"),
    ("set_volume",       "Set system volume (0–100)",                      "low"),
    ("mute_volume",      "Toggle system audio mute",                       "low"),
    ("play_pause_media", "Play or pause media playback",                   "low"),
    ("next_track",       "Skip to the next media track",                   "low"),
    ("prev_track",       "Go back to the previous media track",            "low"),
    ("search_file",      "Search for files in the configured directories", "low"),
    ("open_url",         "Open a URL in the default browser",              "medium"),
    ("type_text",        "Simulate keyboard typing of arbitrary text",     "medium"),
    ("take_screenshot",  "Capture a screenshot and save to Desktop",       "low"),
    ("show_time",        "Report the current time",                        "low"),
    ("show_battery",     "Report battery status and percentage",           "low"),
]

_CC_STYLE = f"""
QMainWindow, QWidget {{ background: {_BG}; color: {_TEXT}; font-size: 13px; }}
QTabWidget::pane {{
    border: 1px solid {_BORDER}; background: {_PANEL};
    border-top-right-radius: 6px; border-bottom-left-radius: 6px;
    border-bottom-right-radius: 6px;
}}
QTabBar::tab {{
    background: {_BG}; color: {_MUTED}; padding: 8px 18px;
    border: none; border-bottom: 2px solid transparent; font-size: 12px;
}}
QTabBar::tab:selected {{ color: {_ACCENT}; border-bottom: 2px solid {_ACCENT}; }}
QTabBar::tab:hover {{ color: {_TEXT}; }}
QPushButton {{
    background: rgba(168,85,247,0.13); border: 1px solid rgba(168,85,247,0.28);
    border-radius: 5px; color: {_TEXT}; padding: 5px 14px; font-size: 12px;
}}
QPushButton:hover {{ background: rgba(168,85,247,0.22); }}
QPushButton:pressed {{ background: rgba(168,85,247,0.34); }}
QPushButton#danger {{
    background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.28);
}}
QPushButton#danger:hover {{ background: rgba(239,68,68,0.22); }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {{
    background: rgba(255,255,255,0.04); border: 1px solid {_BORDER};
    border-radius: 4px; color: {_TEXT}; padding: 4px 8px; font-size: 12px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: rgba(168,85,247,0.5);
}}
QComboBox::drop-down {{ border: none; }}
QCheckBox {{ color: {_TEXT}; font-size: 12px; spacing: 8px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px;
    border: 1px solid {_BORDER}; border-radius: 3px;
    background: rgba(255,255,255,0.04);
}}
QCheckBox::indicator:checked {{ background: {_ACCENT}; border-color: {_ACCENT}; }}
QRadioButton {{ color: {_TEXT}; font-size: 12px; spacing: 8px; }}
QRadioButton::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {_BORDER}; border-radius: 7px;
    background: rgba(255,255,255,0.04);
}}
QRadioButton::indicator:checked {{ background: {_ACCENT}; border-color: {_ACCENT}; }}
QGroupBox {{
    border: 1px solid {_BORDER}; border-radius: 5px;
    margin-top: 14px; padding: 10px 10px 6px 10px;
    color: {_MUTED}; font-size: 11px; letter-spacing: 0.8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    padding: 0 6px; left: 10px;
}}
QListWidget, QTableWidget {{
    background: rgba(255,255,255,0.03); border: 1px solid {_BORDER};
    border-radius: 4px; color: {_TEXT}; font-size: 12px;
    outline: none;
}}
QListWidget::item:selected, QTableWidget::item:selected {{
    background: rgba(168,85,247,0.18); color: {_TEXT};
}}
QTableWidget {{ gridline-color: {_BORDER}; }}
QHeaderView::section {{
    background: rgba(255,255,255,0.04); color: {_MUTED};
    border: none; border-right: 1px solid {_BORDER};
    padding: 5px 8px; font-size: 11px; letter-spacing: 0.5px;
}}
QScrollBar:vertical {{
    background: transparent; width: 6px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {_BORDER}; border-radius: 3px; min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"background: {_BORDER}; border: none; max-height: 1px;")
    return line


def _section(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setStyleSheet(f"color: {_MUTED}; font-size: 10px; letter-spacing: 1px;")
    return lbl


def _note(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setWordWrap(True)
    lbl.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
    return lbl


class _SettingsPanel(QWidget):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self._config = config

    def _show_saved(self, lbl: QLabel, ok: bool = True, msg: str = "") -> None:
        lbl.setText(msg or ("Saved." if ok else "Error."))
        lbl.setStyleSheet(f"color: {_SUCCESS if ok else _ERROR}; font-size: 11px;")
        QTimer.singleShot(3000, lambda: lbl.setText(""))

    def _save_row(self) -> tuple[QHBoxLayout, QLabel, QPushButton]:
        row = QHBoxLayout()
        status = QLabel("")
        btn = QPushButton("Save")
        row.addStretch()
        row.addWidget(status)
        row.addWidget(btn)
        return row, status, btn


# ── Tab: Dashboard ─────────────────────────────────────────────────────────────

class DashboardTab(QWidget):
    def __init__(self, app_state, config: Config, parent=None):
        super().__init__(parent)
        self._state  = app_state
        self._config = config
        self._build()
        self._connect()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # Status group
        grp_status = QGroupBox("Runtime")
        g1 = QGridLayout(grp_status)
        g1.setColumnMinimumWidth(0, 140)
        g1.setHorizontalSpacing(20)
        g1.setVerticalSpacing(6)

        self._lbl_status = QLabel("–")
        self._lbl_ollama = QLabel("–")
        self._lbl_piper  = QLabel("–")
        self._lbl_voice  = QLabel("–")

        for row, (label, widget) in enumerate([
            ("State",       self._lbl_status),
            ("Ollama",      self._lbl_ollama),
            ("Piper TTS",   self._lbl_piper),
            ("Voice Model", self._lbl_voice),
        ]):
            g1.addWidget(_section(label), row, 0)
            g1.addWidget(widget, row, 1)

        root.addWidget(grp_status)

        # Config group
        grp_cfg = QGroupBox("Active Configuration")
        g2 = QGridLayout(grp_cfg)
        g2.setColumnMinimumWidth(0, 140)
        g2.setHorizontalSpacing(20)
        g2.setVerticalSpacing(6)
        g2.setColumnStretch(1, 1)

        self._lbl_model    = QLabel("–")
        self._lbl_language = QLabel("–")
        self._lbl_mode     = QLabel("–")
        self._lbl_mic      = QLabel("–")
        self._lbl_out      = QLabel("–")

        for row, (label, widget) in enumerate([
            ("Model",       self._lbl_model),
            ("Language",    self._lbl_language),
            ("Activation",  self._lbl_mode),
            ("Microphone",  self._lbl_mic),
            ("Output",      self._lbl_out),
        ]):
            g2.addWidget(_section(label), row, 0)
            g2.addWidget(widget, row, 1)

        root.addWidget(grp_cfg)

        # Session group
        grp_sess = QGroupBox("Last Interaction")
        g3 = QGridLayout(grp_sess)
        g3.setColumnMinimumWidth(0, 140)
        g3.setHorizontalSpacing(20)
        g3.setVerticalSpacing(6)
        g3.setColumnStretch(1, 1)

        self._lbl_cmd    = QLabel("–")
        self._lbl_resp   = QLabel("–")
        self._lbl_action = QLabel("–")
        for w in (self._lbl_cmd, self._lbl_resp, self._lbl_action):
            w.setWordWrap(True)

        for row, (label, widget) in enumerate([
            ("Command",  self._lbl_cmd),
            ("Response", self._lbl_resp),
            ("Action",   self._lbl_action),
        ]):
            g3.addWidget(_section(label), row, 0, Qt.AlignmentFlag.AlignTop)
            g3.addWidget(widget, row, 1)

        root.addWidget(grp_sess)
        root.addStretch()

    def _connect(self) -> None:
        s = self._state
        s.status_changed.connect(self._on_status)
        s.ollama_ok_changed.connect(self._on_ollama)
        s.transcript_changed.connect(self._lbl_cmd.setText)
        s.response_changed.connect(self._lbl_resp.setText)
        s.last_action_changed.connect(self._lbl_action.setText)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        c = self._config
        project_root = _project_root()
        piper_raw = c.get("piper_path", "piper/piper/piper.exe")
        piper_ok  = os.path.exists(_resolve(project_root, piper_raw))
        voice_raw = c.get("voice_model", "")
        voice_ok  = os.path.exists(_resolve(project_root, voice_raw))

        self._lbl_piper.setText("Found" if piper_ok else "Not found")
        self._lbl_piper.setStyleSheet(f"color: {_SUCCESS if piper_ok else _ERROR};")
        self._lbl_voice.setText("Found" if voice_ok else "Not found")
        self._lbl_voice.setStyleSheet(f"color: {_SUCCESS if voice_ok else _ERROR};")

        self._lbl_model.setText(c.get("ollama_model", "–"))
        self._lbl_language.setText(c.get("language", "auto").upper())

        mode = c.get("activation_mode", "wake_word")
        if mode == "push_to_talk":
            key = c.get("push_to_talk_key", "ctrl+shift")
            self._lbl_mode.setText(f'Push-to-Talk  ({key.upper()})')
        else:
            ww = c.get("wake_word", "vox")
            self._lbl_mode.setText(f'Wake Word  ("{ww}")')

        try:
            devices   = sd.query_devices()
            mic_idx   = c.get("mic_device",    None)
            out_idx   = c.get("output_device", None)
            def_in, def_out = sd.default.device
            mic_name = devices[mic_idx]["name"] if mic_idx is not None and mic_idx < len(devices) \
                       else f'{devices[def_in]["name"]} (default)'
            out_name = devices[out_idx]["name"] if out_idx is not None and out_idx < len(devices) \
                       else f'{devices[def_out]["name"]} (default)'
        except Exception:
            mic_name = out_name = "Unknown"

        self._lbl_mic.setText(mic_name)
        self._lbl_out.setText(out_name)

        # Reflect current ollama status
        self._on_ollama(self._state.ollama_ok)

    @pyqtSlot(str)
    def _on_status(self, status: str) -> None:
        _map = {
            "idle":         ("Idle",         _MUTED),
            "monitoring":   ("Monitoring",   "#1D9E75"),
            "listening":    ("Listening",     "#2A6FF5"),
            "transcribing": ("Transcribing",  _WARNING),
            "generating":   ("Generating",    _ACCENT),
            "responding":   ("Responding",    _ACCENT),
            "speaking":     ("Speaking",      _SUCCESS),
            "error":        ("Error",         _ERROR),
        }
        text, color = _map.get(status, (status.capitalize(), _TEXT))
        self._lbl_status.setText(text)
        self._lbl_status.setStyleSheet(f"color: {color};")

    @pyqtSlot(bool)
    def _on_ollama(self, ok: bool) -> None:
        self._lbl_ollama.setText("Connected" if ok else "Not reachable")
        self._lbl_ollama.setStyleSheet(f"color: {_SUCCESS if ok else _ERROR};")


# ── Tab: Audio ─────────────────────────────────────────────────────────────────

class AudioTab(_SettingsPanel):
    _mic_test_done  = pyqtSignal(float, float)   # peak, normalised level
    _mic_test_error = pyqtSignal(str)
    _calib_done     = pyqtSignal(dict)            # calibration result dict
    _calib_error    = pyqtSignal(str)
    _stt_done       = pyqtSignal(str, str, str)   # transcript, quality_label, explanation
    _stt_error      = pyqtSignal(str)

    def __init__(self, config: Config, app_state, speaker,
                 restart_cb: Callable | None = None,
                 stt_cb: Callable | None = None,
                 parent=None):
        super().__init__(config, parent)
        self._state      = app_state
        self._speaker    = speaker
        self._restart_cb = restart_cb
        self._stt_cb     = stt_cb   # Callable[[np.ndarray], str] | None
        self._build()
        self._connect()

    def _build(self) -> None:
        # Scrollable root so content is accessible on smaller screens
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        root = QVBoxLayout(inner)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # ── Device selection ───────────────────────────────────────────────────
        grp_dev = QGroupBox("Audio Devices")
        g = QGridLayout(grp_dev)
        g.setColumnMinimumWidth(0, 160)
        g.setHorizontalSpacing(16)
        g.setVerticalSpacing(10)
        g.setColumnStretch(1, 1)

        g.addWidget(_section("Input (Microphone)"), 0, 0)
        self._combo_in = QComboBox()
        g.addWidget(self._combo_in, 0, 1)

        g.addWidget(_section("Output (Speakers)"), 1, 0)
        self._combo_out = QComboBox()
        g.addWidget(self._combo_out, 1, 1)

        root.addWidget(grp_dev)
        root.addWidget(_note("Changing microphone restarts the listener immediately. "
                             "Output changes take effect on next TTS call."))

        # ── Level meter ────────────────────────────────────────────────────────
        grp_lvl = QGroupBox("Microphone Level")
        lv = QHBoxLayout(grp_lvl)
        lv.setSpacing(12)
        lv.addWidget(_section("Live Input"))
        self._mic_bar = MicLevelBar()
        lv.addWidget(self._mic_bar, 1)
        root.addWidget(grp_lvl)

        # ── Signal health card ─────────────────────────────────────────────────
        grp_health = QGroupBox("Signal Health")
        hv = QGridLayout(grp_health)
        hv.setColumnMinimumWidth(0, 160)
        hv.setHorizontalSpacing(16)
        hv.setVerticalSpacing(6)
        hv.setColumnStretch(1, 1)

        self._lbl_noise    = QLabel("–")
        self._lbl_snr      = QLabel("–")
        self._lbl_clip     = QLabel("–")
        self._lbl_quality  = QLabel("–")

        for row, (label, widget) in enumerate([
            ("Noise Floor (RMS)", self._lbl_noise),
            ("SNR",               self._lbl_snr),
            ("Clipping",          self._lbl_clip),
            ("Quality",           self._lbl_quality),
        ]):
            hv.addWidget(_section(label), row, 0)
            hv.addWidget(widget, row, 1)

        root.addWidget(grp_health)

        # ── Calibration ────────────────────────────────────────────────────────
        grp_calib = QGroupBox("Microphone Calibration")
        cv = QVBoxLayout(grp_calib)
        cv.setSpacing(8)

        self._calib_status = QLabel(
            "Run calibration to measure your noise floor and get a recommended silence threshold."
        )
        self._calib_status.setWordWrap(True)
        self._calib_status.setStyleSheet(f"color: {_MUTED}; font-size: 11px;")
        cv.addWidget(self._calib_status)

        calib_btns = QHBoxLayout()
        self._btn_calib = QPushButton("Calibrate  (3 s silence + 3 s speech)")
        calib_btns.addWidget(self._btn_calib)
        calib_btns.addStretch()
        cv.addLayout(calib_btns)

        self._calib_result = QLabel("")
        self._calib_result.setWordWrap(True)
        self._calib_result.setStyleSheet(f"font-size: 11px;")
        cv.addWidget(self._calib_result)

        self._btn_apply_thresh = QPushButton("Apply Suggested Threshold")
        self._btn_apply_thresh.setVisible(False)
        cv.addWidget(self._btn_apply_thresh)

        root.addWidget(grp_calib)

        # ── Device testing ─────────────────────────────────────────────────────
        grp_test = QGroupBox("Device Testing")
        tv = QVBoxLayout(grp_test)
        test_row = QHBoxLayout()
        self._btn_mic = QPushButton("Test Microphone  (3 s)")
        self._btn_tts = QPushButton("Test TTS")
        test_row.addWidget(self._btn_mic)
        test_row.addWidget(self._btn_tts)
        test_row.addStretch()
        tv.addLayout(test_row)
        self._test_lbl = QLabel("")
        self._test_lbl.setStyleSheet(f"font-size: 11px;")
        tv.addWidget(self._test_lbl)
        root.addWidget(grp_test)

        # ── STT test ───────────────────────────────────────────────────────────
        grp_stt = QGroupBox("Speech Recognition Test")
        sv = QVBoxLayout(grp_stt)
        sv.setSpacing(8)

        sv.addWidget(_note(
            "Records 5 s and runs Whisper on the audio to verify speech-to-text quality. "
            "VOX must be idle (not listening) for this test."
        ))
        stt_btns = QHBoxLayout()
        self._btn_stt = QPushButton("Run STT Test  (5 s)")
        self._btn_stt.setEnabled(self._stt_cb is not None)
        stt_btns.addWidget(self._btn_stt)
        stt_btns.addStretch()
        sv.addLayout(stt_btns)

        self._stt_result = QLabel("")
        self._stt_result.setWordWrap(True)
        self._stt_result.setStyleSheet(f"font-size: 11px;")
        sv.addWidget(self._stt_result)

        root.addWidget(grp_stt)
        root.addStretch()

        save_row, self._save_lbl, btn_save = self._save_row()
        root.addLayout(save_row)

        # Wire up
        self._populate_devices()
        btn_save.clicked.connect(self._save)
        self._btn_mic.clicked.connect(self._test_mic)
        self._btn_tts.clicked.connect(self._test_tts)
        self._btn_calib.clicked.connect(self._run_calibration)
        self._btn_stt.clicked.connect(self._run_stt_test)
        self._btn_apply_thresh.clicked.connect(self._apply_suggested_threshold)
        self._mic_test_done.connect(self._on_mic_done)
        self._mic_test_error.connect(self._on_mic_error)
        self._calib_done.connect(self._on_calib_done)
        self._calib_error.connect(self._on_calib_error)
        self._stt_done.connect(self._on_stt_done)
        self._stt_error.connect(self._on_stt_error)

        self._suggested_threshold: float | None = None

    def _connect(self) -> None:
        self._state.mic_level_changed.connect(self._mic_bar.set_level)

    def _populate_devices(self) -> None:
        self._combo_in.clear()
        self._combo_out.clear()
        self._combo_in.addItem("System Default", None)
        self._combo_out.addItem("System Default", None)
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                try:
                    host = f" [{sd.query_hostapis(d['hostapi'])['name']}]"
                except Exception:
                    host = ""
                label = f"[{i}] {d['name']}{host}"
                if d["max_input_channels"] > 0:
                    self._combo_in.addItem(label, i)
                if d["max_output_channels"] > 0:
                    self._combo_out.addItem(label, i)
        except Exception:
            pass
        self._select_combo(self._combo_in,  self._config.get("mic_device",    None))
        self._select_combo(self._combo_out, self._config.get("output_device", None))

    def _select_combo(self, combo: QComboBox, value) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._populate_devices()

    def _save(self) -> None:
        old_mic = self._config.get("mic_device", None)
        new_mic = self._combo_in.currentData()
        new_out = self._combo_out.currentData()
        self._config.set("mic_device",    new_mic)
        self._config.set("output_device", new_out)
        self._config.save()
        if self._speaker:
            self._speaker.reload_config()
        msg = "Saved."
        if new_mic != old_mic and self._restart_cb:
            self._restart_cb()
            msg = "Saved. Listener restarting…"
            self._state.add_diagnostic("info", "Microphone device changed — listener restarted.")
        self._show_saved(self._save_lbl, True, msg)

    # ── Mic test ───────────────────────────────────────────────────────────────

    def _test_mic(self) -> None:
        self._test_lbl.setText("Recording for 3 s…")
        self._test_lbl.setStyleSheet(f"color: {_WARNING}; font-size: 11px;")
        mic_idx = self._combo_in.currentData()

        def _record() -> None:
            try:
                data = sd.rec(int(16000 * 3), samplerate=16000, channels=1,
                              dtype="float32", device=mic_idx)
                sd.wait()
                audio = data.flatten()
                peak  = float(np.max(np.abs(audio)))
                level = min(1.0, peak / 0.30)
                self._mic_test_done.emit(peak, level)
            except Exception as exc:
                self._mic_test_error.emit(str(exc))

        threading.Thread(target=_record, daemon=True).start()

    @pyqtSlot(float, float)
    def _on_mic_done(self, peak: float, level: float) -> None:
        pct = int(level * 100)
        if level < 0.05:
            msg, c = f"Peak {peak:.3f} ({pct}%) — very quiet, check mic", _ERROR
        elif level > 0.95:
            msg, c = f"Peak {peak:.3f} ({pct}%) — may clip, lower gain", _WARNING
        else:
            msg, c = f"Peak {peak:.3f} ({pct}%) — OK", _SUCCESS
        self._test_lbl.setText(msg)
        self._test_lbl.setStyleSheet(f"color: {c}; font-size: 11px;")

    @pyqtSlot(str)
    def _on_mic_error(self, err: str) -> None:
        self._test_lbl.setText(f"Error: {err}")
        self._test_lbl.setStyleSheet(f"color: {_ERROR}; font-size: 11px;")

    def _test_tts(self) -> None:
        if self._speaker:
            self._speaker.speak("VOX is ready.")
            self._test_lbl.setText("TTS test sent.")
            self._test_lbl.setStyleSheet(f"color: {_SUCCESS}; font-size: 11px;")
        else:
            self._test_lbl.setText("Speaker unavailable.")
            self._test_lbl.setStyleSheet(f"color: {_ERROR}; font-size: 11px;")

    # ── Calibration ────────────────────────────────────────────────────────────

    def _run_calibration(self) -> None:
        self._btn_calib.setEnabled(False)
        self._btn_apply_thresh.setVisible(False)
        self._suggested_threshold = None
        self._calib_result.setText("")
        self._calib_status.setText("Phase 1/2 — Stay silent for 3 seconds…")
        self._calib_status.setStyleSheet(f"color: {_WARNING}; font-size: 11px;")
        mic_idx = self._combo_in.currentData()

        def _run() -> None:
            try:
                SR = 16000
                # Phase 1: silence
                silence_data = sd.rec(SR * 3, samplerate=SR, channels=1,
                                      dtype="float32", device=mic_idx)
                sd.wait()
                silence_audio = silence_data.flatten()
                noise_floor   = estimate_noise_floor(silence_audio)

                # Phase 2: speech
                self._calib_status.setText("Phase 2/2 — Speak normally for 3 seconds…")
                speech_data = sd.rec(SR * 3, samplerate=SR, channels=1,
                                     dtype="float32", device=mic_idx)
                sd.wait()
                speech_audio  = speech_data.flatten()
                speech_rms    = estimate_speech_rms(speech_audio, noise_floor)
                clip_frac     = compute_clipping_fraction(speech_audio)
                suggested     = suggest_silence_threshold(noise_floor)
                quality, expl = signal_quality_label(noise_floor, speech_rms)
                snr           = speech_rms / noise_floor if noise_floor > 0 else float("inf")

                self._calib_done.emit({
                    "noise_floor": noise_floor,
                    "speech_rms":  speech_rms,
                    "snr":         snr,
                    "clip_frac":   clip_frac,
                    "suggested":   suggested,
                    "quality":     quality,
                    "explanation": expl,
                })
            except Exception as exc:
                self._calib_error.emit(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    @pyqtSlot(dict)
    def _on_calib_done(self, result: dict) -> None:
        self._btn_calib.setEnabled(True)
        self._calib_status.setText("Calibration complete.")
        self._calib_status.setStyleSheet(f"color: {_SUCCESS}; font-size: 11px;")

        nf  = result["noise_floor"]
        snr = result["snr"]
        cf  = result["clip_frac"]
        sug = result["suggested"]
        ql  = result["quality"]
        ex  = result["explanation"]

        q_color = {
            "good": _SUCCESS, "fair": _WARNING, "poor": _ERROR, "no_signal": _ERROR
        }.get(ql, _MUTED)

        lines = [
            f"Noise floor: {nf:.4f} RMS",
            f"SNR: {snr:.1f}×" if snr != float('inf') else "SNR: ∞ (perfect silence)",
            f"Clipping: {cf*100:.1f}%",
            f"Suggested silence threshold: {sug:.4f}",
            f"Quality: {ql.upper()} — {ex}",
        ]
        self._calib_result.setText("\n".join(lines))
        self._calib_result.setStyleSheet(f"color: {q_color}; font-size: 11px;")

        self._suggested_threshold = sug
        self._btn_apply_thresh.setVisible(True)

        # Update health card
        self._lbl_noise.setText(f"{nf:.4f}")
        snr_text = f"{snr:.1f}×" if snr != float('inf') else "∞"
        self._lbl_snr.setText(snr_text)
        self._lbl_clip.setText(f"{cf*100:.1f}%")
        self._lbl_clip.setStyleSheet(f"color: {_ERROR if cf > 0.01 else _SUCCESS};")
        self._lbl_quality.setText(ql.upper())
        self._lbl_quality.setStyleSheet(f"color: {q_color};")

        self._state.add_diagnostic(
            "info",
            f"Calibration: noise_floor={nf:.4f}, SNR={snr_text}, "
            f"quality={ql}, suggested_threshold={sug:.4f}",
        )

    @pyqtSlot(str)
    def _on_calib_error(self, err: str) -> None:
        self._btn_calib.setEnabled(True)
        self._calib_status.setText(f"Calibration error: {err}")
        self._calib_status.setStyleSheet(f"color: {_ERROR}; font-size: 11px;")

    def _apply_suggested_threshold(self) -> None:
        if self._suggested_threshold is not None:
            self._config.set("silence_threshold", self._suggested_threshold)
            self._config.save()
            self._calib_result.setText(
                self._calib_result.text()
                + f"\n→ Applied silence threshold: {self._suggested_threshold:.4f}"
            )
            self._btn_apply_thresh.setEnabled(False)
            self._state.add_diagnostic(
                "info",
                f"Silence threshold set to {self._suggested_threshold:.4f} from calibration.",
            )

    # ── STT test ───────────────────────────────────────────────────────────────

    def _run_stt_test(self) -> None:
        if self._stt_cb is None:
            return
        self._btn_stt.setEnabled(False)
        self._stt_result.setText("Recording for 5 s — speak now…")
        self._stt_result.setStyleSheet(f"color: {_WARNING}; font-size: 11px;")
        mic_idx = self._combo_in.currentData()
        stt_cb  = self._stt_cb

        def _run() -> None:
            try:
                SR   = 16000
                data = sd.rec(SR * 5, samplerate=SR, channels=1,
                              dtype="float32", device=mic_idx)
                sd.wait()
                audio       = data.flatten()
                transcript  = stt_cb(audio)
                nf          = estimate_noise_floor(audio)
                speech_rms  = estimate_speech_rms(audio, nf)
                quality, ex = signal_quality_label(nf, speech_rms)
                self._stt_done.emit(transcript or "(empty)", quality, ex)
            except Exception as exc:
                self._stt_error.emit(str(exc))

        threading.Thread(target=_run, daemon=True).start()

    @pyqtSlot(str, str, str)
    def _on_stt_done(self, transcript: str, quality: str, explanation: str) -> None:
        self._btn_stt.setEnabled(True)
        q_color = {
            "good": _SUCCESS, "fair": _WARNING, "poor": _ERROR, "no_signal": _ERROR
        }.get(quality, _MUTED)
        self._stt_result.setText(
            f'Transcript: "{transcript}"\n'
            f"Quality: {quality.upper()} — {explanation}"
        )
        self._stt_result.setStyleSheet(f"color: {q_color}; font-size: 11px;")

    @pyqtSlot(str)
    def _on_stt_error(self, err: str) -> None:
        self._btn_stt.setEnabled(True)
        self._stt_result.setText(f"STT test error: {err}")
        self._stt_result.setStyleSheet(f"color: {_ERROR}; font-size: 11px;")


# ── Tab: Activation ────────────────────────────────────────────────────────────

class ActivationTab(_SettingsPanel):
    def __init__(self, config: Config, parent=None):
        super().__init__(config, parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # Mode
        grp_mode = QGroupBox("Activation Mode")
        mv = QVBoxLayout(grp_mode)
        self._rb_wake = QRadioButton("Wake Word  —  always listening, activates on spoken word")
        self._rb_ptt  = QRadioButton("Push-to-Talk  —  manual hotkey to start recording")
        self._mode_grp = QButtonGroup(self)
        self._mode_grp.addButton(self._rb_wake, 0)
        self._mode_grp.addButton(self._rb_ptt,  1)
        mv.addWidget(self._rb_wake)
        mv.addWidget(self._rb_ptt)
        root.addWidget(grp_mode)

        # Wake word params
        grp_ww = QGroupBox("Wake Word")
        wv = QGridLayout(grp_ww)
        wv.setColumnMinimumWidth(0, 160)
        wv.setHorizontalSpacing(16)
        wv.setVerticalSpacing(10)
        wv.setColumnStretch(1, 1)

        wv.addWidget(_section("Wake Word"), 0, 0)
        self._edit_ww = QLineEdit()
        self._edit_ww.setPlaceholderText("vox")
        wv.addWidget(self._edit_ww, 0, 1)

        wv.addWidget(_section("Chunk Duration (s)"), 1, 0)
        self._spin_chunk = QDoubleSpinBox()
        self._spin_chunk.setRange(0.5, 5.0)
        self._spin_chunk.setSingleStep(0.5)
        self._spin_chunk.setDecimals(1)
        wv.addWidget(self._spin_chunk, 1, 1)

        root.addWidget(grp_ww)

        # PTT
        grp_ptt = QGroupBox("Push-to-Talk")
        pv = QGridLayout(grp_ptt)
        pv.setColumnMinimumWidth(0, 160)
        pv.setHorizontalSpacing(16)
        pv.setVerticalSpacing(10)
        pv.setColumnStretch(1, 1)

        pv.addWidget(_section("Key Combination"), 0, 0)
        self._edit_ptt = QLineEdit()
        self._edit_ptt.setPlaceholderText("ctrl+shift")
        pv.addWidget(self._edit_ptt, 0, 1)
        pv.addWidget(_note("Use + to combine keys, e.g. ctrl+shift or alt+z"), 1, 0, 1, 2)

        root.addWidget(grp_ptt)

        # Capture params
        grp_cap = QGroupBox("Capture Parameters")
        cv = QGridLayout(grp_cap)
        cv.setColumnMinimumWidth(0, 200)
        cv.setHorizontalSpacing(16)
        cv.setVerticalSpacing(10)
        cv.setColumnStretch(1, 1)

        cv.addWidget(_section("Silence Threshold (RMS)"), 0, 0)
        self._spin_sil_thresh = QDoubleSpinBox()
        self._spin_sil_thresh.setRange(0.001, 0.2)
        self._spin_sil_thresh.setSingleStep(0.005)
        self._spin_sil_thresh.setDecimals(3)
        cv.addWidget(self._spin_sil_thresh, 0, 1)

        cv.addWidget(_section("Silence Duration (s)"), 1, 0)
        self._spin_sil_dur = QDoubleSpinBox()
        self._spin_sil_dur.setRange(0.3, 5.0)
        self._spin_sil_dur.setSingleStep(0.1)
        self._spin_sil_dur.setDecimals(1)
        cv.addWidget(self._spin_sil_dur, 1, 1)

        cv.addWidget(_note("Silence threshold: RMS below this value counts as silence. "
                           "Silence duration: how many consecutive silent seconds end a command."),
                     2, 0, 1, 2)

        root.addWidget(grp_cap)
        root.addStretch()

        save_row, self._save_lbl, btn_save = self._save_row()
        root.addLayout(save_row)

        btn_save.clicked.connect(self._save)
        self._load()

    def _load(self) -> None:
        c = self._config
        mode = c.get("activation_mode", "wake_word")
        self._rb_wake.setChecked(mode != "push_to_talk")
        self._rb_ptt.setChecked(mode == "push_to_talk")
        self._edit_ww.setText(str(c.get("wake_word", "vox")))
        self._spin_chunk.setValue(float(c.get("chunk_duration", 2.0)))
        self._edit_ptt.setText(str(c.get("push_to_talk_key", "ctrl+shift")))
        self._spin_sil_thresh.setValue(float(c.get("silence_threshold", 0.01)))
        self._spin_sil_dur.setValue(float(c.get("silence_duration", 1.5)))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load()

    def _save(self) -> None:
        c = self._config
        mode = "push_to_talk" if self._rb_ptt.isChecked() else "wake_word"
        c.set("activation_mode",   mode)
        c.set("wake_word",         self._edit_ww.text().strip() or "vox")
        c.set("chunk_duration",    round(self._spin_chunk.value(), 1))
        c.set("push_to_talk_key",  self._edit_ptt.text().strip() or "ctrl+shift")
        c.set("silence_threshold", round(self._spin_sil_thresh.value(), 3))
        c.set("silence_duration",  round(self._spin_sil_dur.value(), 1))
        c.save()
        self._show_saved(self._save_lbl)


# ── Tab: Assistant ─────────────────────────────────────────────────────────────

class AssistantTab(_SettingsPanel):
    def __init__(self, config: Config, parent=None):
        super().__init__(config, parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        grp_llm = QGroupBox("Language Model (Ollama)")
        gv = QGridLayout(grp_llm)
        gv.setColumnMinimumWidth(0, 160)
        gv.setHorizontalSpacing(16)
        gv.setVerticalSpacing(10)
        gv.setColumnStretch(1, 1)

        gv.addWidget(_section("Ollama URL"), 0, 0)
        self._edit_url = QLineEdit()
        gv.addWidget(self._edit_url, 0, 1)

        gv.addWidget(_section("Model"), 1, 0)
        self._edit_model = QLineEdit()
        self._edit_model.setPlaceholderText("qwen2.5:14b")
        gv.addWidget(self._edit_model, 1, 1)

        gv.addWidget(_section("History Size (turns)"), 2, 0)
        self._spin_hist = QSpinBox()
        self._spin_hist.setRange(1, 100)
        gv.addWidget(self._spin_hist, 2, 1)

        root.addWidget(grp_llm)

        grp_stt = QGroupBox("Speech Recognition (Whisper)")
        sv = QGridLayout(grp_stt)
        sv.setColumnMinimumWidth(0, 160)
        sv.setHorizontalSpacing(16)
        sv.setVerticalSpacing(10)
        sv.setColumnStretch(1, 1)

        sv.addWidget(_section("Language"), 0, 0)
        self._combo_lang = QComboBox()
        for val, label in [("auto", "Auto-detect"), ("pt", "Portuguese (pt)"), ("en", "English (en)")]:
            self._combo_lang.addItem(label, val)
        sv.addWidget(self._combo_lang, 0, 1)

        root.addWidget(grp_stt)

        grp_tts = QGroupBox("Text-to-Speech (Piper)")
        tv = QGridLayout(grp_tts)
        tv.setColumnMinimumWidth(0, 160)
        tv.setHorizontalSpacing(16)
        tv.setVerticalSpacing(10)
        tv.setColumnStretch(1, 1)

        tv.addWidget(_section("Enable TTS"), 0, 0)
        self._chk_tts = QCheckBox("")
        tv.addWidget(self._chk_tts, 0, 1)

        tv.addWidget(_section("Voice Model (.onnx)"), 1, 0)
        voice_row = QHBoxLayout()
        self._edit_voice = QLineEdit()
        self._btn_browse = QPushButton("Browse…")
        self._btn_browse.setFixedWidth(80)
        voice_row.addWidget(self._edit_voice)
        voice_row.addWidget(self._btn_browse)
        tv.addLayout(voice_row, 1, 1)

        root.addWidget(grp_tts)
        root.addStretch()

        save_row, self._save_lbl, btn_save = self._save_row()
        root.addLayout(save_row)

        btn_save.clicked.connect(self._save)
        self._btn_browse.clicked.connect(self._browse_voice)
        self._load()

    def _load(self) -> None:
        c = self._config
        self._edit_url.setText(c.get("ollama_url", "http://localhost:11434"))
        self._edit_model.setText(c.get("ollama_model", "qwen2.5:14b"))
        self._spin_hist.setValue(int(c.get("max_history", 20)))
        lang = c.get("language", "auto")
        for i in range(self._combo_lang.count()):
            if self._combo_lang.itemData(i) == lang:
                self._combo_lang.setCurrentIndex(i)
                break
        self._chk_tts.setChecked(bool(c.get("tts_enabled", True)))
        self._edit_voice.setText(c.get("voice_model", ""))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load()

    def _browse_voice(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Voice Model", "", "ONNX files (*.onnx);;All files (*)"
        )
        if path:
            self._edit_voice.setText(path)

    def _save(self) -> None:
        c = self._config
        c.set("ollama_url",   self._edit_url.text().strip())
        c.set("ollama_model", self._edit_model.text().strip())
        c.set("max_history",  self._spin_hist.value())
        c.set("language",     self._combo_lang.currentData())
        c.set("tts_enabled",  self._chk_tts.isChecked())
        c.set("voice_model",  self._edit_voice.text().strip())
        c.save()
        self._show_saved(self._save_lbl)


# ── Tab: Actions & Permissions ─────────────────────────────────────────────────

class ActionsTab(_SettingsPanel):
    def __init__(self, config: Config, parent=None):
        super().__init__(config, parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)

        root.addWidget(_section("Enabled Actions"))
        root.addWidget(_note(
            "Only enabled actions can be executed by the assistant. "
            "Disabling an action prevents the LLM from triggering it, "
            "even if it appears in a response."
        ))

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["", "Action", "Description"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._table.setColumnWidth(0, 30)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setShowGrid(False)
        root.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        btn_all  = QPushButton("Enable All")
        btn_none = QPushButton("Disable All")
        btn_def  = QPushButton("Restore Defaults")
        btn_def.setObjectName("danger")
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        btn_row.addWidget(btn_def)
        btn_row.addStretch()
        root.addLayout(btn_row)

        save_row, self._save_lbl, btn_save = self._save_row()
        root.addLayout(save_row)

        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none.clicked.connect(lambda: self._set_all(False))
        btn_def.clicked.connect(self._restore_defaults)
        btn_save.clicked.connect(self._save)
        self._load()

    def _load(self) -> None:
        allowed = set(self._config.get("allowed_actions", []))
        self._table.setRowCount(0)
        for action, desc, _risk in _ALL_ACTIONS:
            row = self._table.rowCount()
            self._table.insertRow(row)
            chk = QCheckBox()
            chk.setChecked(action in allowed)
            chk.setStyleSheet("margin-left: 6px;")
            self._table.setCellWidget(row, 0, chk)
            self._table.setItem(row, 1, QTableWidgetItem(action))
            self._table.setItem(row, 2, QTableWidgetItem(desc))
            self._table.setRowHeight(row, 28)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load()

    def _set_all(self, state: bool) -> None:
        for row in range(self._table.rowCount()):
            chk = self._table.cellWidget(row, 0)
            if chk:
                chk.setChecked(state)

    def _restore_defaults(self) -> None:
        from utils.config import DEFAULT_CONFIG
        defaults = set(DEFAULT_CONFIG.get("allowed_actions", []))
        for row in range(self._table.rowCount()):
            action = self._table.item(row, 1)
            chk    = self._table.cellWidget(row, 0)
            if action and chk:
                chk.setChecked(action.text() in defaults)

    def _save(self) -> None:
        enabled = []
        for row in range(self._table.rowCount()):
            chk    = self._table.cellWidget(row, 0)
            action = self._table.item(row, 1)
            if chk and action and chk.isChecked():
                enabled.append(action.text())
        self._config.set("allowed_actions", enabled)
        self._config.save()
        self._show_saved(self._save_lbl)


# ── Tab: Apps & Aliases ────────────────────────────────────────────────────────

class AliasesTab(_SettingsPanel):
    def __init__(self, config: Config, parent=None):
        super().__init__(config, parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)

        root.addWidget(_section("App Aliases"))
        root.addWidget(_note(
            "Maps spoken names to executable commands or URI schemes. "
            'E.g. "discord" → discord://   or   "vscode" → code'
        ))

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Spoken Name", "Target (command or URI)"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        root.addWidget(self._table, 1)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add")
        self._btn_del = QPushButton("Remove Selected")
        self._btn_del.setObjectName("danger")
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_del)
        btn_row.addStretch()
        root.addLayout(btn_row)

        save_row, self._save_lbl, btn_save = self._save_row()
        root.addLayout(save_row)

        self._btn_add.clicked.connect(self._add_row)
        self._btn_del.clicked.connect(self._del_row)
        btn_save.clicked.connect(self._save)
        self._load()

    def _load(self) -> None:
        aliases = self._config.get("app_aliases", {})
        self._table.setRowCount(0)
        for name, target in sorted(aliases.items()):
            self._add_row(name, str(target))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load()

    def _add_row(self, name: str = "", target: str = "") -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, 0, QTableWidgetItem(name))
        self._table.setItem(row, 1, QTableWidgetItem(target))
        self._table.setRowHeight(row, 26)

    def _del_row(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for r in rows:
            self._table.removeRow(r)

    def _save(self) -> None:
        aliases: dict[str, str] = {}
        for row in range(self._table.rowCount()):
            n = self._table.item(row, 0)
            t = self._table.item(row, 1)
            if n and t and n.text().strip() and t.text().strip():
                aliases[n.text().strip()] = t.text().strip()
        self._config.set("app_aliases", aliases)
        self._config.save()
        self._show_saved(self._save_lbl)


# ── Tab: Search Directories ────────────────────────────────────────────────────

class DirsTab(_SettingsPanel):
    def __init__(self, config: Config, parent=None):
        super().__init__(config, parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)

        root.addWidget(_section("Search Directories"))
        root.addWidget(_note(
            "The assistant searches these directories when you ask it to find a file. "
            "~ is expanded to your home directory."
        ))

        self._lst = QListWidget()
        root.addWidget(self._lst, 1)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Add Directory…")
        self._btn_del = QPushButton("Remove Selected")
        self._btn_del.setObjectName("danger")
        self._btn_def = QPushButton("Restore Defaults")
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_del)
        btn_row.addWidget(self._btn_def)
        btn_row.addStretch()
        root.addLayout(btn_row)

        save_row, self._save_lbl, btn_save = self._save_row()
        root.addLayout(save_row)

        self._btn_add.clicked.connect(self._add_dir)
        self._btn_del.clicked.connect(self._del_dir)
        self._btn_def.clicked.connect(self._restore_defaults)
        btn_save.clicked.connect(self._save)
        self._load()

    def _load(self) -> None:
        self._lst.clear()
        for d in self._config.get("search_dirs", []):
            self._lst.addItem(str(d))

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._load()

    def _add_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            self._lst.addItem(path)

    def _del_dir(self) -> None:
        for item in self._lst.selectedItems():
            self._lst.takeItem(self._lst.row(item))

    def _restore_defaults(self) -> None:
        self._lst.clear()
        from utils.config import DEFAULT_CONFIG
        for d in DEFAULT_CONFIG.get("search_dirs", []):
            self._lst.addItem(d)

    def _save(self) -> None:
        dirs = [self._lst.item(i).text() for i in range(self._lst.count())]
        self._config.set("search_dirs", dirs)
        self._config.save()
        self._show_saved(self._save_lbl)


# ── Tab: History ───────────────────────────────────────────────────────────────

class HistoryTab(QWidget):
    def __init__(self, app_state, parent=None):
        super().__init__(parent)
        self._state = app_state
        self._build()
        self._connect()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)

        root.addWidget(_section("Session History"))

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_mono_font())
        root.addWidget(self._log, 1)

        btn_row = QHBoxLayout()
        btn_clear = QPushButton("Clear History")
        btn_clear.setObjectName("danger")
        btn_row.addStretch()
        btn_row.addWidget(btn_clear)
        root.addLayout(btn_row)

        btn_clear.clicked.connect(self._state.clear_history)

    def _connect(self) -> None:
        self._state.history_entry_added.connect(self._on_entry)
        self._state.history_cleared.connect(self._log.clear)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._reload()

    def _reload(self) -> None:
        self._log.clear()
        for entry in self._state.history:
            self._append(entry)

    @pyqtSlot(dict)
    def _on_entry(self, entry: dict) -> None:
        self._append(entry)

    def _append(self, entry: dict) -> None:
        ts = entry.get("timestamp", "")
        tx = entry.get("transcript", "")
        rx = entry.get("response",  "")
        ax = entry.get("action",    "")

        parts = [f'<span style="color:{_MUTED};">[{ts}]</span>']
        if tx:
            parts.append(f'<span style="color:#e2e0ea;">YOU: {_esc(tx)}</span>')
        if ax:
            parts.append(f'<span style="color:{_SUCCESS};">ACTION: {_esc(ax)}</span>')
        elif rx:
            parts.append(f'<span style="color:{_ACCENT};">VOX: {_esc(rx)}</span>')

        self._log.append("  ".join(parts))
        self._log.append("")
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())


# ── Tab: Diagnostics ──────────────────────────────────────────────────────────

class DiagnosticsTab(QWidget):
    def __init__(self, app_state, validate_cb: Callable | None = None, parent=None):
        super().__init__(parent)
        self._state       = app_state
        self._validate_cb = validate_cb
        self._build()
        self._connect()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(10)

        root.addWidget(_section("Diagnostics Log"))

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(_mono_font())
        root.addWidget(self._log, 1)

        btn_row = QHBoxLayout()
        self._btn_validate = QPushButton("Re-run Validation")
        btn_clear = QPushButton("Clear Log")
        btn_clear.setObjectName("danger")
        btn_row.addWidget(self._btn_validate)
        btn_row.addStretch()
        btn_row.addWidget(btn_clear)
        root.addLayout(btn_row)

        self._btn_validate.clicked.connect(self._rerun_validation)
        btn_clear.clicked.connect(self._state.clear_diagnostics)

    def _connect(self) -> None:
        self._state.diagnostic_added.connect(self._on_entry)
        self._state.diagnostics_cleared.connect(self._log.clear)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._reload()

    def _reload(self) -> None:
        self._log.clear()
        for entry in self._state.diagnostics:
            self._append(entry)

    @pyqtSlot(dict)
    def _on_entry(self, entry: dict) -> None:
        self._append(entry)

    def _append(self, entry: dict) -> None:
        lvl = entry.get("level", "info")
        ts  = entry.get("timestamp", "")
        msg = entry.get("message", "")
        color_map = {
            "info":    _MUTED,
            "warning": _WARNING,
            "error":   _ERROR,
        }
        c = color_map.get(lvl, _TEXT)
        icon = {"info": "ℹ", "warning": "⚠", "error": "✖"}.get(lvl, "·")
        self._log.append(
            f'<span style="color:{_MUTED};">[{ts}]</span> '
            f'<span style="color:{c};">{icon} {_esc(msg)}</span>'
        )
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _rerun_validation(self) -> None:
        if self._validate_cb:
            self._validate_cb()
        else:
            self._state.add_diagnostic("info", "Validation callback not connected.")


# ── ControlCenter ──────────────────────────────────────────────────────────────

class ControlCenter(QMainWindow):
    restart_listener_requested = pyqtSignal()
    rerun_validation_requested = pyqtSignal()

    def __init__(self, config: Config, app_state, speaker,
                 stt_cb: Callable | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VOX Control Center")
        self.setMinimumSize(720, 560)
        self.resize(860, 620)
        self.setStyleSheet(_CC_STYLE)

        icon_path = os.path.join(_project_root(), "assets", "icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setTabPosition(QTabWidget.TabPosition.North)

        tabs.addTab(DashboardTab(app_state, config),                              "Dashboard")
        tabs.addTab(AudioTab(config, app_state, speaker,
                             restart_cb=self.restart_listener_requested.emit,
                             stt_cb=stt_cb),                                      "Audio")
        tabs.addTab(ActivationTab(config),                                         "Activation")
        tabs.addTab(AssistantTab(config),                                          "Assistant")
        tabs.addTab(ActionsTab(config),                                            "Actions")
        tabs.addTab(AliasesTab(config),                                            "Aliases")
        tabs.addTab(DirsTab(config),                                               "Directories")
        tabs.addTab(HistoryTab(app_state),                                         "History")
        tabs.addTab(DiagnosticsTab(app_state,
                                    validate_cb=self.rerun_validation_requested.emit), "Diagnostics")

        self.setCentralWidget(tabs)


# ── Utilities ──────────────────────────────────────────────────────────────────

def _project_root() -> str:
    # src/ui/control_center.py → src/ui → src → project root
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(project_root: str, raw: str) -> str:
    if os.path.isabs(raw):
        return raw
    return os.path.normpath(os.path.join(project_root, raw))


def _esc(text: str) -> str:
    """HTML-escape text for safe QTextEdit insertion."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mono_font():
    from PyQt6.QtGui import QFont
    f = QFont("Consolas")
    f.setPointSize(11)
    if not f.exactMatch():
        f.setFamily("Monospace")
    return f
