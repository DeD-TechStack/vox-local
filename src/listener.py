import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel
import keyboard

from utils.config import Config


class Listener(QThread):
    transcription_ready = pyqtSignal(str)
    listening_started = pyqtSignal()
    listening_stopped = pyqtSignal()

    SAMPLE_RATE = 16000
    CHANNELS = 1

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hotkey = config.get("hotkey", "alt")
        self._audio_buffer = []
        self._recording = False

        print("[Listener] Loading Whisper model...")
        self.model = WhisperModel(
            config.get("whisper_model", "base"),
            device="cuda",
            compute_type="float16",
        )
        print("[Listener] Whisper ready.")

    def run(self):
        keyboard.on_press_key(self.hotkey, self._start_recording)
        keyboard.on_release_key(self.hotkey, self._stop_recording)
        print(f"[Listener] Hold '{self.hotkey.upper()}' to speak.")
        keyboard.wait()

    def _start_recording(self, _event=None):
        if self._recording:
            return
        self._recording = True
        self._audio_buffer = []
        self.listening_started.emit()

        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def _stop_recording(self, _event=None):
        if not self._recording:
            return
        self._recording = False
        self._stream.stop()
        self._stream.close()
        self.listening_stopped.emit()

        if self._audio_buffer:
            audio = np.concatenate(self._audio_buffer, axis=0).flatten()
            self._transcribe(audio)

    def _audio_callback(self, indata, frames, time, status):
        if self._recording:
            self._audio_buffer.append(indata.copy())

    def _transcribe(self, audio: np.ndarray):
        segments, _ = self.model.transcribe(
            audio,
            language=self.config.get("language", "pt"),
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            self.transcription_ready.emit(text)
