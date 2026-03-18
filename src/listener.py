import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel
import keyboard

from utils.config import Config
from utils.logger import get_logger

log = get_logger("Listener")


class Listener(QThread):
    transcription_ready = pyqtSignal(str)
    listening_started = pyqtSignal()
    listening_stopped = pyqtSignal()

    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_DURATION = 2      # seconds per wake-word detection chunk
    SILENCE_DURATION = 2    # seconds of silence to end a command
    SILENCE_THRESHOLD = 0.01

    def __init__(self, config: Config, model: WhisperModel):
        super().__init__()
        self.config = config
        self.model = model
        self.hotkey = config.get("hotkey", "alt")
        self._audio_buffer = []
        self._recording = False

    def run(self):
        self._print_devices()
        if self.config.get("wake_word_enabled", False):
            wake_word = self.config.get("wake_word", "hey vox").lower()
            log.info(f"Wake word mode active. Say '{wake_word}' to activate.")
            self._run_wake_word_loop(wake_word)
        else:
            log.info(f"Ready. Hold '{self.hotkey.upper()}' to speak.")
            keyboard.on_press_key(self.hotkey, self._start_recording)
            keyboard.on_release_key(self.hotkey, self._stop_recording)
            keyboard.wait()

    # ─── Wake word mode ──────────────────────────────────────────────────────

    def _run_wake_word_loop(self, wake_word: str):
        mic_device = self.config.get("mic_device", None)
        chunk_samples = self.SAMPLE_RATE * self.CHUNK_DURATION
        silence_samples = self.SAMPLE_RATE * self.SILENCE_DURATION

        while True:
            # Capture a short chunk and check for the wake word
            chunk = sd.rec(
                chunk_samples,
                samplerate=self.SAMPLE_RATE,
                channels=self.CHANNELS,
                dtype="float32",
                device=mic_device,
            )
            sd.wait()

            audio = chunk.flatten()
            segments, _ = self.model.transcribe(
                audio,
                language=self.config.get("language", "en"),
                beam_size=1,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip().lower()

            if wake_word in text:
                log.info("Wake word detected — listening for command...")
                self.listening_started.emit()
                command_audio = self._record_until_silence(mic_device, silence_samples)
                self.listening_stopped.emit()
                if command_audio is not None:
                    self._transcribe(command_audio)

    def _record_until_silence(self, mic_device, silence_samples: int):
        """Record audio until silence_samples consecutive silent samples, return ndarray."""
        all_chunks = []
        consecutive_silent = 0
        chunk_size = self.SAMPLE_RATE // 2  # 0.5-second read chunks

        stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype="float32",
            device=mic_device,
        )
        stream.start()
        try:
            while True:
                data, _ = stream.read(chunk_size)
                all_chunks.append(data.copy())
                rms = float(np.sqrt(np.mean(data ** 2)))
                if rms < self.SILENCE_THRESHOLD:
                    consecutive_silent += chunk_size
                    if consecutive_silent >= silence_samples:
                        break
                else:
                    consecutive_silent = 0
        finally:
            stream.stop()
            stream.close()

        if all_chunks:
            return np.concatenate(all_chunks, axis=0).flatten()
        return None

    # ─── Hotkey mode ─────────────────────────────────────────────────────────

    def _print_devices(self):
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        configured = self.config.get("mic_device", None)
        log.info("Available microphones:")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                marker = "→ USING" if (configured is not None and i == configured) \
                         else ("→ DEFAULT" if (configured is None and i == default_in) else "  ")
                log.info(f"  [{i:2d}] {marker}  {d['name']}")
        log.info("To change mic, set 'mic_device: <index>' in settings.yaml")

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
