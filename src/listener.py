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

    def __init__(self, config: Config, model: WhisperModel):
        super().__init__()
        self.config = config
        self.model = model
        self.hotkey = config.get("hotkey", "alt")
        self._audio_buffer = []
        self._recording = False

    def run(self):
        self._print_devices()
        print(f"[Listener] Ready. Hold '{self.hotkey.upper()}' to speak.")
        keyboard.on_press_key(self.hotkey, self._start_recording)
        keyboard.on_release_key(self.hotkey, self._stop_recording)
        keyboard.wait()

    def _print_devices(self):
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        configured = self.config.get("mic_device", None)
        print("[Listener] Available microphones:")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                marker = "→ USING" if (configured is not None and i == configured) \
                         else ("→ DEFAULT" if (configured is None and i == default_in) else "  ")
                print(f"  [{i:2d}] {marker}  {d['name']}")
        print(f"[Listener] To change mic, set 'mic_device: <index>' in settings.yaml")

    def _start_recording(self, _event=None):
        if self._recording:
            return
        self._recording = True
        self._audio_buffer = []
        self.listening_started.emit()

        mic_device = self.config.get("mic_device", None)
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="float32",
            device=mic_device,
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
            language=self.config.get("language", "en"),
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if text:
            self.transcription_ready.emit(text)
