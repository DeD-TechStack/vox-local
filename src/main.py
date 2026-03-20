import sys
import os
import traceback
import requests as _requests

from utils.config import Config
from utils.logger import get_logger

log = get_logger("VOX")


def load_whisper(config: Config):
    from faster_whisper import WhisperModel
    device       = config.get("whisper_device",       "cpu")
    compute_type = config.get("whisper_compute_type", "int8")
    log.info(f"Loading Whisper ({config.get('whisper_model', 'base')}, {device}/{compute_type})…")
    model = WhisperModel(config.get("whisper_model", "base"), device=device, compute_type=compute_type)
    log.info("Whisper ready.")
    return model


def run_app(config: Config, whisper_model):
    from PyQt6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
    from PyQt6.QtGui import QIcon
    from PyQt6.QtCore import QThread, QTimer, pyqtSignal

    from listener import Listener
    from brain    import Brain
    from speaker  import Speaker
    from executor import Executor
    from ui.overlay import OverlayWindow

    class BrainWorker(QThread):
        response_ready     = pyqtSignal(str, bool)
        token_received     = pyqtSignal(str)
        generating_started = pyqtSignal()

        def __init__(self, brain, text: str):
            super().__init__()
            self._brain = brain
            self._text  = text

        def run(self):
            try:
                text, is_action = self._brain.process(
                    self._text,
                    on_token=self.token_received.emit,
                    on_generating=self.generating_started.emit,
                )
            except Exception as e:
                traceback.print_exc()
                text, is_action = f"Internal error: {e}", False
            self.response_ready.emit(text, is_action)

    class VoxApp:
        def __init__(self):
            self.app = QApplication(sys.argv)
            self.app.setQuitOnLastWindowClosed(False)

            self.speaker  = Speaker(config)
            self.executor = Executor(config)
            self.brain    = Brain(config, self.executor)
            self.listener = Listener(config, whisper_model)

            self.overlay       = OverlayWindow()
            self._brain_worker = None

            self._connect_signals()
            self._setup_tray()

        def _connect_signals(self):
            self.listener.listening_started.connect(self.overlay.set_listening)
            self.listener.listening_stopped.connect(self.overlay.set_processing)
            self.listener.transcription_ready.connect(self.on_transcription)
            self.listener.language_detected.connect(self.overlay.show_detected_language)
            self.overlay.language_clicked.connect(self.on_language_cycle)
            self.overlay.set_language_mode(config.get("language", "auto"))

            self._ollama_timer = QTimer()
            self._ollama_timer.timeout.connect(self._ping_ollama)
            self._ollama_timer.start(12000)
            self._ping_ollama()  # check immediately on startup

        def _setup_tray(self):
            icon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "assets", "icon.ico",
            )
            icon = QIcon(icon_path) if os.path.exists(icon_path) else QIcon()
            tray = QSystemTrayIcon(icon, self.app)
            tray.setToolTip("VOX")

            menu = QMenu()
            menu.addAction("Show",           self.overlay.show)
            menu.addAction("Audio Settings", self._open_audio_settings)
            menu.addSeparator()
            menu.addAction("Quit",           self.app.quit)

            tray.setContextMenu(menu)
            tray.show()

        def _open_audio_settings(self):
            from ui.audio_settings import AudioSettingsDialog
            dlg = AudioSettingsDialog(config)
            if dlg.exec():
                self.speaker.reload_config()

        def on_transcription(self, text: str):
            # Ignore new command if still processing the previous one
            if self._brain_worker and self._brain_worker.isRunning():
                log.warning("Brain still processing — ignoring new transcription.")
                return
            self.overlay.set_transcript(text)
            self._brain_worker = BrainWorker(self.brain, text)
            self._brain_worker.token_received.connect(self.overlay.append_token)
            self._brain_worker.generating_started.connect(self.overlay.set_generating)
            self._brain_worker.response_ready.connect(self.on_response)
            self._brain_worker.start()

        def _ping_ollama(self):
            try:
                r = _requests.get(
                    f"{config.get('ollama_url', 'http://localhost:11434')}/api/tags",
                    timeout=2,
                )
                self.overlay.set_ollama_ok(r.status_code == 200)
            except Exception:
                self.overlay.set_ollama_ok(False)

        def on_language_cycle(self):
            order   = ["auto", "pt", "en"]
            current = config.get("language", "auto")
            nxt     = order[(order.index(current) + 1) % len(order)] if current in order else "auto"
            config.set("language", nxt)
            config.save()
            self.overlay.set_language_mode(nxt)
            log.info(f"Language switched to: {nxt}")

        def on_response(self, response: str, is_action: bool):
            if is_action:
                self.overlay.set_action(response)
            else:
                self.overlay.set_response(response)
            self.speaker.speak(response)
            self.overlay.set_idle()

        def run(self):
            self.overlay.show()
            self.listener.start()
            sys.exit(self.app.exec())

    vox = VoxApp()
    vox.run()


if __name__ == "__main__":
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vox_error.log")
    try:
        config        = Config()
        whisper_model = load_whisper(config)
        run_app(config, whisper_model)
    except Exception:
        err = traceback.format_exc()
        log.error(err)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(err)
        log.error(f"Error saved to: {os.path.abspath(log_path)}")
