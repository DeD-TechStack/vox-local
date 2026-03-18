import math
import random

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, QTimer, QRectF, pyqtSlot
from PyQt6.QtGui import QColor, QPainter, QGuiApplication


# ─── Waveform ─────────────────────────────────────────────────────────────────

class WaveformWidget(QWidget):
    BAR_COUNT = 8
    BAR_W = 3
    BAR_GAP = 5
    BAR_MIN = 3
    BAR_MAX = 22

    def __init__(self, parent=None):
        super().__init__(parent)
        total_w = self.BAR_COUNT * self.BAR_W + (self.BAR_COUNT - 1) * self.BAR_GAP
        self.setFixedSize(total_w, self.BAR_MAX + 6)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._bars = [self.BAR_MIN] * self.BAR_COUNT
        self._targets = [self.BAR_MIN] * self.BAR_COUNT
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def set_active(self, active: bool):
        self._active = active
        if active:
            self._pulse_mode = False
            self._timer.start(80)
        else:
            if not getattr(self, "_pulse_mode", False):
                self._timer.stop()
                self._bars = [self.BAR_MIN] * self.BAR_COUNT
                self.update()

    def set_pulse(self, active: bool):
        """Slow gentle pulse for thinking state (bars move minimally)."""
        self._pulse_mode = active
        if active:
            self._timer.start(150)
        else:
            if not self._active:
                self._timer.stop()
                self._bars = [self.BAR_MIN] * self.BAR_COUNT
                self.update()

    def _tick(self):
        if getattr(self, "_pulse_mode", False) and not getattr(self, "_active", False):
            # Gentle pulse: low amplitude, slow movement
            for i in range(self.BAR_COUNT):
                lo, hi = self.BAR_MIN, self.BAR_MIN + 6
                self._targets[i] = random.randint(lo, hi)
                self._bars[i] = int(self._bars[i] * 0.6 + self._targets[i] * 0.4)
        else:
            for i in range(self.BAR_COUNT):
                self._targets[i] = random.randint(self.BAR_MIN, self.BAR_MAX)
                self._bars[i] = int(self._bars[i] * 0.4 + self._targets[i] * 0.6)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        cy = self.height() / 2
        for i, h in enumerate(self._bars):
            x = i * (self.BAR_W + self.BAR_GAP)
            y = cy - h / 2
            # Fade bars at edges
            edge = 1.0 - abs(i - (self.BAR_COUNT - 1) / 2) / ((self.BAR_COUNT - 1) / 2)
            alpha = int(80 + 140 * edge)
            p.setBrush(QColor(168, 85, 247, alpha))
            p.drawRoundedRect(QRectF(x, y, self.BAR_W, h), 1.5, 1.5)
        p.end()


# ─── Overlay ──────────────────────────────────────────────────────────────────

_SS_LABEL_BASE = "background: transparent; border: none;"
_SS_STATUS_IDLE = f"color: #3d2d54; font-size: 11px; {_SS_LABEL_BASE}"
_SS_STATUS_ACTIVE = "color: {color}; font-size: 11px; " + _SS_LABEL_BASE

_STYLE_CONTAINER = """
QFrame#container {
    background-color: rgba(13, 11, 18, 220);
    border-radius: 14px;
    border: 1px solid rgba(168, 85, 247, 80);
}
"""


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_window()
        self._setup_ui()
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._auto_hide)

    # ── Window setup ──────────────────────────────────────────────────────────

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(360, 190)

        screen = QGuiApplication.primaryScreen().geometry()
        self.move(screen.width() - 380, screen.height() - 220)

    # ── UI construction ───────────────────────────────────────────────────────

    def _setup_ui(self):
        self._container = QFrame(self)
        self._container.setObjectName("container")
        self._container.setFixedSize(360, 190)
        self._container.setStyleSheet(_STYLE_CONTAINER)

        root = QVBoxLayout(self._container)
        root.setContentsMargins(18, 14, 18, 12)
        root.setSpacing(0)

        # ── Header row
        header = QHBoxLayout()
        header.setSpacing(0)

        self._title = QLabel("VOX")
        self._title.setStyleSheet(
            f"color: rgba(255,255,255,0.9); font-size: 11px; "
            f"letter-spacing: 4px; font-weight: 600; {_SS_LABEL_BASE}"
        )

        self._status = QLabel("● idle")
        self._status.setStyleSheet(_SS_STATUS_IDLE)

        header.addWidget(self._title)
        header.addStretch()
        header.addWidget(self._status)
        root.addLayout(header)

        root.addSpacing(10)

        # ── Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: rgba(168,85,247,30); border: none;")
        root.addWidget(divider)

        root.addSpacing(10)

        # ── Waveform (centered, always present, flat when inactive)
        wave_row = QHBoxLayout()
        wave_row.setSpacing(0)
        self._waveform = WaveformWidget()
        wave_row.addStretch()
        wave_row.addWidget(self._waveform)
        wave_row.addStretch()
        root.addLayout(wave_row)

        root.addSpacing(10)

        # ── Transcript
        self._transcript = QLabel("Hold ALT to speak…")
        self._transcript.setWordWrap(True)
        self._transcript.setFixedHeight(32)
        self._transcript.setStyleSheet(
            f"color: rgba(255,255,255,0.45); font-size: 12px; {_SS_LABEL_BASE}"
        )
        root.addWidget(self._transcript)

        root.addSpacing(4)

        # ── Response
        self._response = QLabel("")
        self._response.setWordWrap(True)
        self._response.setFixedHeight(32)
        self._response.setStyleSheet(
            f"color: #a855f7; font-size: 12px; {_SS_LABEL_BASE}"
        )
        root.addWidget(self._response)

        root.addStretch()

        # ── Bottom hint
        self._hint = QLabel("Hold  ALT  to speak")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setStyleSheet(
            f"color: rgba(255,255,255,0.12); font-size: 10px; letter-spacing: 1px; {_SS_LABEL_BASE}"
        )
        root.addWidget(self._hint)

    # ── State slots ───────────────────────────────────────────────────────────

    @pyqtSlot()
    def set_listening(self):
        self._status.setText("● listening")
        self._status.setStyleSheet(_SS_STATUS_ACTIVE.format(color="#2a6ff5"))
        self._transcript.setText("…")
        self._transcript.setStyleSheet(
            f"color: rgba(255,255,255,0.55); font-size: 12px; {_SS_LABEL_BASE}"
        )
        self._response.setText("")
        self._waveform.set_active(True)
        self.show()
        self._idle_timer.stop()

    @pyqtSlot()
    def set_processing(self):
        """User released ALT — waiting for Ollama to start generating."""
        self._status.setText("● thinking…")
        self._status.setStyleSheet(_SS_STATUS_ACTIVE.format(color="#f59e0b"))
        self._waveform.set_active(False)
        self._waveform.set_pulse(True)
        self._response.setText("")
        self._response.setStyleSheet(
            f"color: #a855f7; font-size: 12px; {_SS_LABEL_BASE}"
        )

    @pyqtSlot()
    def set_generating(self):
        """First token received — model is now streaming."""
        self._status.setText("● generating")
        self._status.setStyleSheet(_SS_STATUS_ACTIVE.format(color="#a855f7"))
        self._waveform.set_pulse(False)

    @pyqtSlot(str)
    def set_transcript(self, text: str):
        self._transcript.setText(text)
        self._transcript.setStyleSheet(
            f"color: rgba(255,255,255,0.75); font-size: 12px; {_SS_LABEL_BASE}"
        )

    @pyqtSlot(str)
    def append_token(self, token: str):
        self._response.setText(self._response.text() + token)
        self._status.setText("● responding")
        self._status.setStyleSheet(_SS_STATUS_ACTIVE.format(color="#a855f7"))

    @pyqtSlot(str, bool)
    def set_response(self, text: str, is_action: bool = False):
        color = "#1d9e75" if is_action else "#a855f7"
        self._response.setText(text)
        self._response.setStyleSheet(
            f"color: {color}; font-size: 12px; {_SS_LABEL_BASE}"
        )
        self._status.setText("● responding")
        self._status.setStyleSheet(_SS_STATUS_ACTIVE.format(color=color))

    @pyqtSlot()
    def set_idle(self):
        self._status.setText("● idle")
        self._status.setStyleSheet(_SS_STATUS_IDLE)
        self._waveform.set_active(False)
        self._waveform.set_pulse(False)
        self._idle_timer.start(5000)

    def _auto_hide(self):
        self._transcript.setText("Hold ALT to speak…")
        self._transcript.setStyleSheet(
            f"color: rgba(255,255,255,0.45); font-size: 12px; {_SS_LABEL_BASE}"
        )
        self._response.setText("")

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
