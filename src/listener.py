import re

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel

from utils.config import Config
from utils.logger import get_logger

log = get_logger("Listener")


class Listener(QThread):
    transcription_ready = pyqtSignal(str)
    language_detected   = pyqtSignal(str)
    listening_started   = pyqtSignal()
    listening_stopped   = pyqtSignal()

    SAMPLE_RATE = 16000

    def __init__(self, config: Config, model: WhisperModel):
        super().__init__()
        self.config = config
        self.model  = model

        self._wake_word        = config.get("wake_word", "vox").lower().strip()
        self._chunk_duration   = float(config.get("chunk_duration",   2.0))
        self._silence_threshold = float(config.get("silence_threshold", 0.01))
        self._silence_duration  = float(config.get("silence_duration",  1.5))

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self._print_devices()
        self._validate_mic_device()
        log.info(f"Wake word mode active. Say '{self._wake_word}' to activate.")
        self._run_wake_word_loop()

    def _validate_mic_device(self):
        mic_device = self.config.get("mic_device", None)
        try:
            info = sd.query_devices(mic_device, "input") if mic_device is not None else sd.query_devices(sd.default.device[0])
            if info["max_input_channels"] == 0:
                log.error(f"mic_device {mic_device} ({info['name']}) has NO input channels — it is an output device! Fix mic_device in settings.yaml.")
            else:
                log.info(f"Using mic: [{mic_device}] {info['name']} ({info['max_input_channels']}ch)")
        except Exception as e:
            log.warning(f"Could not validate mic_device {mic_device}: {e}")

    # ── Wake word loop ────────────────────────────────────────────────────────

    def _get_device_channels(self, mic_device) -> int:
        """Return the number of input channels supported by the device (min 1)."""
        try:
            info = sd.query_devices(mic_device, "input") if mic_device is not None else sd.query_devices(sd.default.device[0])
            return max(1, int(info["max_input_channels"]))
        except Exception:
            return 1

    def _run_wake_word_loop(self):
        mic_device      = self.config.get("mic_device", None)
        channels        = self._get_device_channels(mic_device)
        chunk_samples   = int(self.SAMPLE_RATE * self._chunk_duration)
        silence_samples = int(self.SAMPLE_RATE * self._silence_duration)

        while True:
            try:
                # Capture a short chunk and check for the wake word
                chunk = sd.rec(
                    chunk_samples,
                    samplerate=self.SAMPLE_RATE,
                    channels=channels,
                    dtype="float32",
                    device=mic_device,
                )
                sd.wait()

                audio    = chunk.mean(axis=1) if chunk.ndim > 1 and chunk.shape[1] > 1 else chunk.flatten()
                rms = float(np.sqrt(np.mean(audio ** 2)))
                log.debug(f"[wake] chunk rms={rms:.4f}")
                lang_cfg  = self.config.get("language", "auto")
                lang_param = None if lang_cfg == "auto" else lang_cfg
                segments, _ = self.model.transcribe(
                    audio,
                    language=lang_param,
                    beam_size=1,
                )
                text = " ".join(s.text.strip() for s in segments).strip().lower()
                if text:
                    log.info(f"[wake] heard: '{text}'")

                if self._wake_word in text:
                    log.info(f"Wake word detected — listening for command…")
                    self.listening_started.emit()
                    command_audio = self._record_until_silence(mic_device, silence_samples)
                    self.listening_stopped.emit()
                    if command_audio is not None:
                        self._transcribe_and_emit(command_audio)

            except sd.PortAudioError as e:
                log.error(f"Audio device error: {e} — retrying with default device")
                mic_device = None
                channels   = self._get_device_channels(mic_device)
            except Exception as e:
                log.error(f"Wake word loop error: {e}")

    def _record_until_silence(self, mic_device, silence_samples: int):
        """Record audio until silence_samples consecutive silent samples. Returns flat ndarray."""
        all_chunks         = []
        consecutive_silent = 0
        chunk_size         = self.SAMPLE_RATE // 2  # 0.5 s read chunks
        max_samples        = int(self.SAMPLE_RATE * float(self.config.get("max_record_duration", 30)))
        min_samples        = int(self.SAMPLE_RATE * float(self.config.get("min_listen_duration", 1.0)))
        total_samples      = 0
        channels           = self._get_device_channels(mic_device)

        stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=channels,
            dtype="float32",
            device=mic_device,
        )
        stream.start()
        try:
            while True:
                data, _ = stream.read(chunk_size)
                all_chunks.append(data.copy())
                total_samples += chunk_size
                rms = float(np.sqrt(np.mean(data ** 2)))
                if total_samples >= min_samples and rms < self._silence_threshold:
                    consecutive_silent += chunk_size
                    if consecutive_silent >= silence_samples:
                        break
                else:
                    consecutive_silent = 0
                if total_samples >= max_samples:
                    log.warning("Max recording duration reached — stopping capture.")
                    break
        finally:
            stream.stop()
            stream.close()

        if all_chunks:
            audio = np.concatenate(all_chunks, axis=0)
            return audio.mean(axis=1) if audio.ndim > 1 and audio.shape[1] > 1 else audio.flatten()
        return None

    def _transcribe_and_emit(self, audio: np.ndarray):
        lang_cfg   = self.config.get("language", "auto")
        lang_param = None if lang_cfg == "auto" else lang_cfg
        segments, info = self.model.transcribe(
            audio,
            language=lang_param,
            beam_size=5,
        )
        detected = getattr(info, "language", lang_cfg) or "auto"
        self.language_detected.emit(detected)
        text = " ".join(s.text.strip() for s in segments).strip()

        # Strip leading wake word in case the mic picked it up in the command phase
        text = re.sub(
            rf"^\s*{re.escape(self._wake_word)}\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        if text:
            self.transcription_ready.emit(text)
        else:
            log.info("Empty command after wake word strip — ignoring.")

    # ── Device listing ────────────────────────────────────────────────────────

    def _print_devices(self):
        devices    = sd.query_devices()
        default_in = sd.default.device[0]
        configured = self.config.get("mic_device", None)
        log.info("Available microphones:")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                if configured is not None and i == configured:
                    marker = "→ USING"
                elif configured is None and i == default_in:
                    marker = "→ DEFAULT"
                else:
                    marker = "  "
                log.info(f"  [{i:2d}] {marker}  {d['name']}")
        log.info("To change mic, set 'mic_device: <index>' in settings.yaml")
