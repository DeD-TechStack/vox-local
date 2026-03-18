import sys
import threading
from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from listener import Listener
from brain import Brain
from speaker import Speaker
from executor import Executor
from ui.overlay import OverlayWindow
from utils.config import Config


class VoxApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        self.config = Config()
        self.speaker = Speaker(self.config)
        self.executor = Executor(self.config)
        self.brain = Brain(self.config, self.executor)
        self.listener = Listener(self.config)

        self.overlay = OverlayWindow()

        self._connect_signals()
        self._setup_tray()

    def _connect_signals(self):
        self.listener.transcription_ready.connect(self.on_transcription)
        self.listener.listening_started.connect(self.overlay.set_listening)
        self.listener.listening_stopped.connect(self.overlay.set_processing)

    def _setup_tray(self):
        tray = QSystemTrayIcon(self.app)
        tray.setToolTip("VOX")

        menu = QMenu()
        menu.addAction("Show", self.overlay.show)
        menu.addAction("Settings", lambda: print("TODO: settings"))
        menu.addSeparator()
        menu.addAction("Quit", self.app.quit)

        tray.setContextMenu(menu)
        tray.show()

    def on_transcription(self, text: str):
        self.overlay.set_transcript(text)
        response = self.brain.process(text)
        self.overlay.set_response(response)
        self.speaker.speak(response)
        self.overlay.set_idle()

    def run(self):
        self.overlay.show()
        self.listener.start()
        sys.exit(self.app.exec())


if __name__ == "__main__":
    vox = VoxApp()
    vox.run()
