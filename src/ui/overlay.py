from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPainter, QPainterPath


class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self._setup_window()
        self._setup_ui()
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.timeout.connect(self._auto_hide)

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(340, 160)

        from PyQt6.QtGui import QGuiApplication
        screen = QGuiApplication.primaryScreen().geometry()
        self.move(screen.width() - 360, screen.height() - 200)

    def _setup_ui(self):
        self._container = QFrame(self)
        self._container.setFixedSize(340, 160)
        self._container.setStyleSheet("""
            QFrame {
                background-color: rgba(13, 13, 13, 220);
                border-radius: 14px;
                border: 1px solid rgba(42, 111, 245, 80);
            }
        """)

        layout = QVBoxLayout(self._container)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)

        # Header
        header = QHBoxLayout()
        self._title = QLabel("VOX")
        self._title.setStyleSheet("color: rgba(255,255,255,0.9); font-size: 11px; letter-spacing: 3px; font-weight: 500; background: transparent; border: none;")

        self._status = QLabel("● idle")
        self._status.setStyleSheet("color: #555; font-size: 11px; background: transparent; border: none;")

        header.addWidget(self._title)
        header.addStretch()
        header.addWidget(self._status)
        layout.addLayout(header)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background: rgba(255,255,255,0.06); border: none; max-height: 1px;")
        layout.addWidget(line)

        # Transcript
        self._transcript = QLabel("Segure ALT para falar...")
        self._transcript.setWordWrap(True)
        self._transcript.setStyleSheet("color: rgba(255,255,255,0.7); font-size: 13px; background: transparent; border: none;")
        layout.addWidget(self._transcript)

        # Response
        self._response = QLabel("")
        self._response.setWordWrap(True)
        self._response.setStyleSheet("color: #1d9e75; font-size: 12px; background: transparent; border: none;")
        layout.addWidget(self._response)

        layout.addStretch()

    def paintEvent(self, event):
        pass

    @pyqtSlot()
    def set_listening(self):
        self._status.setText("● ouvindo")
        self._status.setStyleSheet("color: #2a6ff5; font-size: 11px; background: transparent; border: none;")
        self._transcript.setText("...")
        self._response.setText("")
        self.show()
        self._idle_timer.stop()

    @pyqtSlot()
    def set_processing(self):
        self._status.setText("● processando")
        self._status.setStyleSheet("color: #f59e0b; font-size: 11px; background: transparent; border: none;")

    @pyqtSlot(str)
    def set_transcript(self, text: str):
        self._transcript.setText(text)

    @pyqtSlot(str)
    def set_response(self, text: str):
        self._response.setText(text)
        self._status.setText("● respondendo")
        self._status.setStyleSheet("color: #1d9e75; font-size: 11px; background: transparent; border: none;")

    @pyqtSlot()
    def set_idle(self):
        self._status.setText("● idle")
        self._status.setStyleSheet("color: #555; font-size: 11px; background: transparent; border: none;")
        self._idle_timer.start(4000)

    def _auto_hide(self):
        self._transcript.setText("Segure ALT para falar...")
        self._response.setText("")

    def mousePressEvent(self, event):
        self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if hasattr(self, "_drag_pos"):
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()
