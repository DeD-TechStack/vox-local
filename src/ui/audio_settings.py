import re

import sounddevice as sd
from PyQt6.QtWidgets import (
    QDialog, QFrame, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QGuiApplication

from utils.config import Config


_SS_BASE = "background: transparent; border: none;"
_SS_CONTAINER = """
QFrame#container {
    background-color: rgba(13, 11, 18, 245);
    border-radius: 14px;
    border: 1px solid rgba(168, 85, 247, 90);
}
"""
_SS_LABEL = f"color: rgba(255,255,255,0.5); font-size: 11px; {_SS_BASE}"
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


def _get_devices(kind: str) -> list[dict]:
    """
    Returns deduplicated device list preferring WASAPI > DirectSound > MME.
    Each entry: {"index": int | None, "label": str}
    First entry is always "System Default" (index=None).
    """
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    hostapis = sd.query_hostapis()
    all_devices = sd.query_devices()

    PRIORITY = {"Windows WASAPI": 0, "Windows DirectSound": 1, "MME": 2}

    candidates = []
    for i, dev in enumerate(all_devices):
        if dev[channel_key] == 0:
            continue
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        if "WDM" in hostapi_name or "KS" in hostapi_name:
            continue
        priority = PRIORITY.get(hostapi_name, 99)
        candidates.append({
            "index": i,
            "name": dev["name"],
            "hostapi": hostapi_name,
            "priority": priority,
        })

    # Deduplicate: keep best-priority entry per normalized name
    def _norm(name: str) -> str:
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name)
        return name.strip().lower()

    seen: dict[str, dict] = {}
    for dev in sorted(candidates, key=lambda d: d["priority"]):
        key = _norm(dev["name"])
        if key not in seen:
            seen[key] = dev

    result = [{"index": None, "label": "System Default"}]
    for dev in seen.values():
        result.append({"index": dev["index"], "label": f"{dev['name']}  [{dev['hostapi']}]"})
    return result


class AudioSettingsDialog(QDialog):
    def __init__(self, config: Config, parent=None):
        super().__init__(parent)
        self._config = config
        self._setup_window()
        self._setup_ui()
        self._populate()

    def _setup_window(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(400, 290)

        screen = QGuiApplication.primaryScreen().geometry()
        self.move(
            (screen.width() - 400) // 2,
            (screen.height() - 290) // 2,
        )

    def _setup_ui(self):
        self._container = QFrame(self)
        self._container.setObjectName("container")
        self._container.setFixedSize(400, 290)
        self._container.setStyleSheet(_SS_CONTAINER)

        root = QVBoxLayout(self._container)
        root.setContentsMargins(22, 18, 22, 18)
        root.setSpacing(0)

        # Title row
        title_row = QHBoxLayout()
        title = QLabel("AUDIO SETTINGS")
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

        root.addSpacing(12)

        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFixedHeight(1)
        divider.setStyleSheet("background: rgba(168,85,247,30); border: none;")
        root.addWidget(divider)

        root.addSpacing(18)

        # Input device
        lbl_in = QLabel("Input Device  (microphone)")
        lbl_in.setStyleSheet(_SS_LABEL)
        root.addWidget(lbl_in)
        root.addSpacing(6)
        self._input_combo = QComboBox()
        self._input_combo.setStyleSheet(_SS_COMBO)
        root.addWidget(self._input_combo)

        root.addSpacing(16)

        # Output device
        lbl_out = QLabel("Output Device  (speakers / headphones)")
        lbl_out.setStyleSheet(_SS_LABEL)
        root.addWidget(lbl_out)
        root.addSpacing(6)
        self._output_combo = QComboBox()
        self._output_combo.setStyleSheet(_SS_COMBO)
        root.addWidget(self._output_combo)

        root.addStretch()

        divider2 = QFrame()
        divider2.setFrameShape(QFrame.Shape.HLine)
        divider2.setFixedHeight(1)
        divider2.setStyleSheet("background: rgba(168,85,247,30); border: none;")
        root.addWidget(divider2)

        root.addSpacing(14)

        # Bottom row: status + buttons
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

    def _populate(self):
        current_in = self._config.get("mic_device", None)
        current_out = self._config.get("output_device", None)

        for combo, kind, current in [
            (self._input_combo, "input", current_in),
            (self._output_combo, "output", current_out),
        ]:
            try:
                devices = _get_devices(kind)
            except Exception as e:
                combo.addItem(f"Error: {e}")
                continue

            for dev in devices:
                combo.addItem(dev["label"], userData=dev["index"])

            # Select current
            for i in range(combo.count()):
                if combo.itemData(i) == current:
                    combo.setCurrentIndex(i)
                    break

    def _on_save(self):
        in_idx = self._input_combo.currentData()
        out_idx = self._output_combo.currentData()
        self._config.set("mic_device", in_idx)
        self._config.set("output_device", out_idx)
        try:
            self._config.save()
            self._status.setText("Saved.")
            self._status.setStyleSheet(f"color: #1d9e75; font-size: 11px; {_SS_BASE}")
            QTimer.singleShot(700, self.accept)
        except Exception as e:
            self._status.setText(f"Error: {e}")
            self._status.setStyleSheet(f"color: #ef4444; font-size: 11px; {_SS_BASE}")

    # Drag support
    def mousePressEvent(self, event):
        self._drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if hasattr(self, "_drag_pos"):
            delta = event.globalPosition().toPoint() - self._drag_pos
            self.move(self.pos() + delta)
            self._drag_pos = event.globalPosition().toPoint()

    def paintEvent(self, event):
        pass
