from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QGuiApplication

from ui.mic_meter import MicLevelWaveform


# ─── Stylesheet constants ──────────────────────────────────────────────────────

_BG          = "background: transparent; border: none;"
_MUTED_LABEL = f"color: #444444; font-size: 10px; letter-spacing: 1px; {_BG}"

_CONTAINER_SS = """
QFrame#container {
    background-color: rgba(13, 11, 18, 225);
    border-radius: 14px;
    border: 1px solid rgba(168, 85, 247, 80);
}
"""

_BADGE_SS = (
    "background: rgba(168,85,247,30);"
    "border: 1px solid rgba(168,85,247,100);"
    "border-radius: 4px;"
    "color: rgba(168,85,247,200);"
    "font-size: 9px;"
    "letter-spacing: 1px;"
    "padding: 1px 5px;"
)

W, H = 400, 265


# ─── Clickable label ──────────────────────────────────────────────────────────

class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


# ─── Overlay ──────────────────────────────────────────────────────────────────

class OverlayWindow(QWidget):
    language_clicked = pyqtSignal()

    # Human-readable status labels and their colours
    _STATUS_MAP = {
        "idle":         ("idle",          "#444444"),
        "monitoring":   ("monitoring",    "#1D9E75"),
        "listening":    ("listening",     "#2A6FF5"),
        "transcribing": ("transcribing",  "#F59E0B"),
        "generating":   ("generating",    "#A855F7"),
        "responding":   ("responding",    "#A855F7"),
        "speaking":     ("speaking",      "#A855F7"),
        "error":        ("error",         "#EF4444"),
        "cancelled":    ("cancelled",     "#EF4444"),
        "done":         ("done",          "#1D9E75"),
    }

    def __init__(self):
        super().__init__()
        self._state          = "idle"
        self._pulse_on       = True
        self._pulse_color    = "#444444"
        self._pulse_dim      = "rgba(68,68,68,0.2)"
        self._footer_default = "Say VOX to activate"
        self._lang_mode_text = "AUTO"

        self._setup_window()
        self._setup_ui()

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._auto_hide)

        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._tick_pulse)

    # ── Window setup ──────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(W, H)
        screen = QGuiApplication.primaryScreen().geometry()
        self.move(screen.width() - W - 20, screen.height() - H - 44)

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        self._container = QFrame(self)
        self._container.setObjectName("container")
        self._container.setFixedSize(W, H)
        self._container.setStyleSheet(_CONTAINER_SS)

        root = QVBoxLayout(self._container)
        root.setContentsMargins(18, 14, 18, 12)
        root.setSpacing(0)

        # ── Header row ───────────────────────────────────────
        header = QHBoxLayout()
        header.setSpacing(0)

        self._title = QLabel("VOX")
        self._title.setStyleSheet(
            f"color: rgba(255,255,255,0.9); font-size: 11px; "
            f"letter-spacing: 3px; font-weight: 500; {_BG}"
        )

        # Ollama status dot
        self._ollama_dot = QLabel("●")
        self._ollama_dot.setStyleSheet(f"color: #333333; font-size: 7px; {_BG}")
        self._ollama_dot.setToolTip("Checking Ollama…")

        self._dot = QLabel("●")
        self._dot.setStyleSheet(f"color: #444444; font-size: 8px; {_BG}")

        self._status_lbl = QLabel("idle")
        self._status_lbl.setStyleSheet(f"color: #444444; font-size: 11px; {_BG}")

        status_row = QHBoxLayout()
        status_row.setSpacing(5)
        status_row.addWidget(self._ollama_dot)
        status_row.addSpacing(3)
        status_row.addWidget(self._dot)
        status_row.addWidget(self._status_lbl)

        header.addWidget(self._title)
        header.addStretch()
        header.addLayout(status_row)
        root.addLayout(header)

        root.addSpacing(8)

        # ── Divider ──────────────────────────────────────────
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setFixedHeight(1)
        div.setStyleSheet("background: rgba(168,85,247,40); border: none;")
        root.addWidget(div)

        root.addSpacing(8)

        # ── Waveform (real mic level) ────────────────────────
        wave_row = QHBoxLayout()
        wave_row.setSpacing(0)
        self._waveform = MicLevelWaveform()
        wave_row.addStretch()
        wave_row.addWidget(self._waveform)
        wave_row.addStretch()
        root.addLayout(wave_row)

        root.addSpacing(10)

        # ── Transcript section ───────────────────────────────
        self._lbl_you = QLabel("YOU")
        self._lbl_you.setStyleSheet(_MUTED_LABEL)
        root.addWidget(self._lbl_you)

        root.addSpacing(2)

        self._transcript = QLabel("–")
        self._transcript.setWordWrap(True)
        self._transcript.setMaximumHeight(44)
        self._transcript.setStyleSheet(
            f"color: rgba(255,255,255,0.5); font-size: 13px; {_BG}"
        )
        root.addWidget(self._transcript)

        root.addSpacing(8)

        # ── Divider (subtle) ─────────────────────────────────
        div2 = QFrame()
        div2.setFrameShape(QFrame.Shape.HLine)
        div2.setFixedHeight(1)
        div2.setStyleSheet("background: rgba(255,255,255,8); border: none;")
        root.addWidget(div2)

        root.addSpacing(8)

        # ── Response section ─────────────────────────────────
        self._lbl_vox = QLabel("VOX")
        self._lbl_vox.setStyleSheet(_MUTED_LABEL)
        root.addWidget(self._lbl_vox)

        root.addSpacing(2)

        self._response = QLabel("")
        self._response.setWordWrap(True)
        self._response.setMaximumHeight(55)
        self._response.setStyleSheet(f"color: #A855F7; font-size: 13px; {_BG}")
        root.addWidget(self._response)

        root.addStretch()

        # ── Footer ───────────────────────────────────────────
        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(0)

        self._footer = QLabel(self._footer_default)
        self._footer.setStyleSheet(f"color: #2a2a2a; font-size: 10px; {_BG}")

        self._lang_badge = ClickableLabel("AUTO")
        self._lang_badge.setStyleSheet(_BADGE_SS)
        self._lang_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self._lang_badge.setToolTip("Click to cycle language: AUTO → PT → EN")
        self._lang_badge.clicked.connect(self.language_clicked)

        footer_row.addWidget(self._footer)
        footer_row.addStretch()
        footer_row.addWidget(self._lang_badge)
        root.addLayout(footer_row)

    # ── Dot & pulse ───────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str, pulse: bool = False,
                    dim: str | None = None):
        self._pulse_color = color
        self._pulse_dim   = dim if dim else _darken(color)
        self._pulse_on    = True

        if pulse:
            self._pulse_timer.start(550)
        else:
            self._pulse_timer.stop()

        self._apply_dot(color)
        self._status_lbl.setText(text)
        self._status_lbl.setStyleSheet(f"color: {color}; font-size: 11px; {_BG}")

    def _tick_pulse(self):
        self._pulse_on = not self._pulse_on
        self._apply_dot(self._pulse_color if self._pulse_on else self._pulse_dim)

    def _apply_dot(self, color: str):
        self._dot.setStyleSheet(f"color: {color}; font-size: 8px; {_BG}")

    # ── Mic level ─────────────────────────────────────────────────────────────

    @pyqtSlot(float)
    def set_mic_level(self, level: float):
        """Feed real RMS level (0.0–1.0) to the waveform widget."""
        self._waveform.set_level(level)

    # ── State slots ───────────────────────────────────────────────────────────

    @pyqtSlot(str)
    def set_footer_mode(self, mode: str):
        if mode == "push_to_talk":
            self._footer_default = "Press Ctrl+Shift to activate"
        else:
            self._footer_default = "Say VOX to activate"
        self._footer.setText(self._footer_default)

    @pyqtSlot(str)
    def set_footer_mode_with_key(self, mode: str, key: str):
        if mode == "push_to_talk":
            self._footer_default = f"Press {key.upper()} to activate"
        else:
            self._footer_default = "Say VOX to activate"
        self._footer.setText(self._footer_default)

    @pyqtSlot(str)
    def show_info_notice(self, message: str, duration_ms: int = 4000):
        self._footer.setText(message)
        self._footer.setVisible(True)
        self._footer.setStyleSheet(f"color: #F59E0B; font-size: 10px; {_BG}")
        QTimer.singleShot(duration_ms, self._restore_footer)

    def _restore_footer(self):
        self._footer.setText(self._footer_default)
        self._footer.setStyleSheet(f"color: #2a2a2a; font-size: 10px; {_BG}")

    @pyqtSlot()
    def set_cancelled(self):
        self._state = "idle"
        self._set_status("cancelled", "#EF4444")
        self._waveform.set_active(False)
        self._response.setText("")
        self._footer.setVisible(False)
        self._idle_timer.start(1500)

    @pyqtSlot()
    def set_idle(self):
        self._state = "idle"
        self._set_status("idle", "#444444")
        self._waveform.set_active(False)
        self._footer.setText(self._footer_default)
        self._footer.setStyleSheet(f"color: #2a2a2a; font-size: 10px; {_BG}")
        self._footer.setVisible(True)
        self._idle_timer.start(5000)

    @pyqtSlot()
    def set_listening(self):
        self._state = "listening"
        self._set_status("listening", "#2A6FF5", pulse=True, dim="rgba(42,111,245,0.25)")
        self._transcript.setText("…")
        self._transcript.setStyleSheet(
            f"color: rgba(255,255,255,0.45); font-size: 13px; {_BG}"
        )
        self._response.setText("")
        self._response.setStyleSheet(f"color: #A855F7; font-size: 13px; {_BG}")
        self._waveform.set_active(True)
        self._footer.setVisible(False)
        self._idle_timer.stop()
        self.show()

    @pyqtSlot()
    def set_processing(self):
        """Listening stopped — transcribing audio."""
        self._state = "transcribing"
        self._set_status("transcribing…", "#F59E0B")
        self._waveform.set_active(False)
        self._response.setText("")
        self._response.setStyleSheet(f"color: #A855F7; font-size: 13px; {_BG}")
        self._footer.setVisible(False)

    @pyqtSlot()
    def set_generating(self):
        self._state = "generating"
        self._set_status("generating", "#A855F7")

    @pyqtSlot()
    def set_speaking(self):
        self._state = "speaking"
        self._set_status("speaking", "#A855F7", pulse=True, dim="rgba(168,85,247,0.25)")

    @pyqtSlot(str)
    def set_transcript(self, text: str):
        self._transcript.setText(text)
        self._transcript.setStyleSheet(
            f"color: rgba(255,255,255,0.85); font-size: 13px; {_BG}"
        )

    @pyqtSlot(str)
    def append_token(self, token: str):
        if self._state != "responding":
            self._state = "responding"
            self._set_status("responding", "#A855F7")
            self._footer.setVisible(False)
        self._response.setText(self._response.text() + token)

    @pyqtSlot(str)
    def set_response(self, text: str):
        self._state = "responding"
        self._response.setText(text)
        self._response.setStyleSheet(f"color: #A855F7; font-size: 13px; {_BG}")
        self._set_status("responding", "#A855F7")
        self._footer.setVisible(False)

    @pyqtSlot(str)
    def set_action(self, text: str):
        self._state = "responding"
        self._response.setText(text)
        self._response.setStyleSheet(f"color: #1D9E75; font-size: 13px; {_BG}")
        self._set_status("done", "#1D9E75")
        self._footer.setVisible(False)

    @pyqtSlot(bool)
    def set_ollama_ok(self, ok: bool):
        color   = "#1D9E75" if ok else "#EF4444"
        tooltip = "Ollama connected" if ok else "Ollama disconnected"
        self._ollama_dot.setStyleSheet(f"color: {color}; font-size: 7px; {_BG}")
        self._ollama_dot.setToolTip(tooltip)

    @pyqtSlot(str)
    def set_language_mode(self, mode: str):
        labels = {"auto": "AUTO", "pt": "PT", "en": "EN"}
        text = labels.get(mode, mode.upper()[:4])
        self._lang_mode_text = text
        self._lang_badge.setText(text)

    @pyqtSlot(str)
    def show_detected_language(self, lang: str):
        labels = {"pt": "PT", "en": "EN", "portuguese": "PT", "english": "EN"}
        display = labels.get(lang.lower(), lang.upper()[:2])
        self._lang_badge.setText(display)
        QTimer.singleShot(3000, self._restore_language_badge)

    def _restore_language_badge(self):
        self._lang_badge.setText(self._lang_mode_text)

    def _auto_hide(self):
        self._transcript.setText("–")
        self._transcript.setStyleSheet(
            f"color: rgba(255,255,255,0.2); font-size: 13px; {_BG}"
        )
        self._response.setText("")
        self.hide()

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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _darken(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},0.2)"
    return "rgba(80,80,80,0.2)"
