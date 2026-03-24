import sys
import os
import traceback
import requests as _requests

from utils.config import Config
from utils.logger import get_logger

log = get_logger("VOX")


def _validate_startup(config: Config):
    """Warn about missing dependencies before the app starts. Never aborts."""
    # 1. Ollama reachability
    try:
        r = _requests.get(
            f"{config.get('ollama_url', 'http://localhost:11434')}/api/tags",
            timeout=3,
        )
        if r.status_code != 200:
            log.warning(f"Ollama responded with status {r.status_code}. LLM may not work.")
        else:
            log.info("Ollama is reachable.")
    except Exception as e:
        log.warning(f"Ollama is not reachable ({e}). Start it with: ollama serve")

    # 2. Piper binary
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    piper_raw = config.get("piper_path", "piper/piper/piper.exe")
    piper_path = (
        piper_raw if os.path.isabs(piper_raw)
        else os.path.normpath(os.path.join(project_root, piper_raw))
    )
    if not os.path.exists(piper_path):
        log.warning(f"Piper binary not found at '{piper_path}'. TTS will be silent.")

    # 3. Voice model
    voice_raw = config.get("voice_model", "voices/en_US-ryan-high.onnx")
    voice_path = (
        voice_raw if os.path.isabs(voice_raw)
        else os.path.normpath(os.path.join(project_root, voice_raw))
    )
    if not os.path.exists(voice_path):
        log.warning(f"Voice model not found at '{voice_path}'. TTS will be silent.")

    # 4. Audio device indices
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        n = len(devices)
        mic_idx = config.get("mic_device", None)
        out_idx = config.get("output_device", None)
        if mic_idx is not None and (not isinstance(mic_idx, int) or mic_idx >= n):
            log.warning(
                f"mic_device={mic_idx} is not a valid device index. Using system default."
            )
        if out_idx is not None and (not isinstance(out_idx, int) or out_idx >= n):
            log.warning(
                f"output_device={out_idx} is not a valid device index. Using system default."
            )
    except Exception as e:
        log.warning(f"Could not validate audio devices: {e}")


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
            self._brain     = brain
            self._text      = text
            self._cancelled = False

        def cancel(self):
            """Request cancellation of the running stream."""
            self._cancelled = True

        def run(self):
            try:
                text, is_action = self._brain.process(
                    self._text,
                    on_token=self._emit_token,
                    on_generating=self.generating_started.emit,
                    cancelled=lambda: self._cancelled,
                )
            except Exception as e:
                traceback.print_exc()
                text, is_action = f"Internal error: {e}", False
            if not self._cancelled:
                self.response_ready.emit(text, is_action)

        def _emit_token(self, token: str):
            if not self._cancelled:
                self.token_received.emit(token)

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
            self._apply_activation_mode_ui()

        def _connect_listener_signals(self):
            """Connect (or reconnect) listener-specific signals to overlay slots."""
            self.listener.listening_started.connect(self.overlay.set_listening)
            self.listener.listening_stopped.connect(self.overlay.set_processing)
            self.listener.transcription_ready.connect(self.on_transcription)
            self.listener.language_detected.connect(self.overlay.show_detected_language)

        def _connect_signals(self):
            self._connect_listener_signals()
            self.overlay.language_clicked.connect(self.on_language_cycle)
            self.overlay.set_language_mode(config.get("language", "auto"))

            self._ollama_timer = QTimer()
            self._ollama_timer.timeout.connect(self._ping_ollama)
            self._ollama_timer.start(12000)
            self._ping_ollama()  # check immediately on startup

        def _apply_activation_mode_ui(self):
            mode = config.get("activation_mode", "wake_word")
            key  = config.get("push_to_talk_key", "ctrl+shift")
            self.overlay.set_footer_mode_with_key(mode, key)

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
            menu.addAction("Settings",       self._open_settings)
            menu.addAction("Audio Settings", self._open_audio_settings)
            menu.addSeparator()
            menu.addAction("Quit",           self.app.quit)

            tray.setContextMenu(menu)
            tray.show()

        def _open_audio_settings(self):
            from ui.audio_settings import AudioSettingsDialog
            old_mic = config.get("mic_device", None)
            dlg = AudioSettingsDialog(config)
            if dlg.exec():
                self.speaker.reload_config()
                new_mic = config.get("mic_device", None)
                if new_mic != old_mic:
                    log.info("Mic device changed — restarting listener.")
                    self.overlay.show_info_notice(
                        "Microphone changed — restarting listener…"
                    )
                    self._restart_listener()

        def _open_settings(self):
            from ui.settings_dialog import SettingsDialog
            dlg = SettingsDialog(config)
            if dlg.exec():
                self.speaker.reload_config()
                self._apply_activation_mode_ui()
                self.overlay.set_language_mode(config.get("language", "auto"))
                # Activation mode and wake word are read dynamically by the
                # listener loop on the next iteration — no restart needed.
                # Voice model and TTS changes take effect immediately.

        def _restart_listener(self):
            """Stop and restart the listener thread.

            The listener stop flag is set; the thread checks it between audio
            calls and exits within at most chunk_duration seconds (≤ 2 s).
            Calling start() again on the same QThread object is safe once the
            thread has finished running.
            """
            self.listener.stop_listener()
            if not self.listener.wait(3000):
                log.warning("Listener did not stop cleanly within 3 s — proceeding anyway.")
            self.listener.start()
            log.info("Listener restarted.")

        def on_transcription(self, text: str):
            # If Brain is still processing, cancel it and start fresh.
            if self._brain_worker and self._brain_worker.isRunning():
                log.info("New transcription received — cancelling running worker.")
                self._brain_worker.cancel()
                self._brain_worker.wait(2000)
                self.overlay.set_cancelled()

            self.overlay.set_transcript(text)
            self._brain_worker = BrainWorker(self.brain, text)
            self._brain_worker.token_received.connect(self.overlay.append_token)
            self._brain_worker.generating_started.connect(self.overlay.set_generating)
            self._brain_worker.response_ready.connect(self.on_response)
            self._brain_worker.start()

            # Watchdog: if brain hangs for >35s, reset to idle
            self._brain_timeout = QTimer()
            self._brain_timeout.setSingleShot(True)
            self._brain_timeout.timeout.connect(self._on_brain_timeout)
            self._brain_timeout.start(35000)

        def _on_brain_timeout(self):
            if self._brain_worker and self._brain_worker.isRunning():
                log.error("Brain worker timed out — resetting to idle.")
                self._brain_worker.terminate()
                self.overlay.set_idle()

        def _ping_ollama(self):
            try:
                r = _requests.get(
                    f"{config.get('ollama_url', 'http://localhost:11434')}/api/tags",
                    timeout=2,
                )
                ok = r.status_code == 200
            except Exception:
                ok = False
            if not ok:
                log.warning("Ollama ping failed — LLM unavailable.")
            self.overlay.set_ollama_ok(ok)

        def on_language_cycle(self):
            order   = ["auto", "pt", "en"]
            current = config.get("language", "auto")
            nxt     = order[(order.index(current) + 1) % len(order)] if current in order else "auto"
            config.set("language", nxt)
            config.save()
            self.overlay.set_language_mode(nxt)
            log.info(f"Language switched to: {nxt}")

        def on_response(self, response: str, is_action: bool):
            if hasattr(self, "_brain_timeout"):
                self._brain_timeout.stop()
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
        _validate_startup(config)
        whisper_model = load_whisper(config)
        run_app(config, whisper_model)
    except Exception:
        err = traceback.format_exc()
        log.error(err)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(err)
        log.error(f"Error saved to: {os.path.abspath(log_path)}")
