import re
import threading
import time

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QThread, pyqtSignal
from faster_whisper import WhisperModel

from utils.config import Config
from utils.logger import get_logger

log = get_logger("Listener")


def _extract_post_wake(text: str, wake_word: str) -> str:
    """Return text that follows the wake word in a transcription string.

    Uses a word-boundary regex so the wake word is matched as a whole word.
    Returns the stripped tail, or "" if the wake word is absent or nothing
    meaningful follows it.
    """
    pattern = rf'\b{re.escape(wake_word)}\b'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return text[match.end():].strip()
    return ""


class Listener(QThread):
    transcription_ready = pyqtSignal(str)
    language_detected   = pyqtSignal(str)
    listening_started   = pyqtSignal()
    listening_stopped   = pyqtSignal()

    SAMPLE_RATE = 16000
    CHANNELS    = 1

    def __init__(self, config: Config, model: WhisperModel):
        super().__init__()
        self.config     = config
        self.model      = model
        self._stop_flag = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop_listener(self):
        """Signal the listening loop to exit at the next opportunity.

        The loop is checked between blocking audio calls, so the actual stop
        may take up to chunk_duration seconds (≤ 2 s by default).
        """
        self._stop_flag.set()

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self):
        self._stop_flag.clear()
        self._print_devices()
        activation_mode = self.config.get("activation_mode", "wake_word")
        if activation_mode == "push_to_talk":
            ptt_key = self.config.get("push_to_talk_key", "ctrl+shift")
            log.info(f"Push-to-talk mode active. Press '{ptt_key}' to activate.")
            self._run_push_to_talk_loop()
        else:
            wake_word = self.config.get("wake_word", "vox").lower().strip()
            log.info(f"Wake word mode active. Say '{wake_word}' to activate.")
            self._run_wake_word_loop()

    # ── Wake word loop ────────────────────────────────────────────────────────

    def _run_wake_word_loop(self):
        _backoff = 0.0

        while not self._stop_flag.is_set():
            try:
                # Re-read settings each iteration so runtime changes apply.
                mic_device       = self.config.get("mic_device", None)
                wake_word        = self.config.get("wake_word", "vox").lower().strip()
                chunk_duration   = float(self.config.get("chunk_duration",   2.0))
                silence_duration = float(self.config.get("silence_duration",  1.5))
                chunk_samples    = int(self.SAMPLE_RATE * chunk_duration)
                silence_samples  = int(self.SAMPLE_RATE * silence_duration)

                chunk = sd.rec(
                    chunk_samples,
                    samplerate=self.SAMPLE_RATE,
                    channels=self.CHANNELS,
                    dtype="float32",
                    device=mic_device,
                )
                sd.wait()

                if self._stop_flag.is_set():
                    break

                audio     = chunk.flatten()
                lang_cfg  = self.config.get("language", "auto")
                lang_param = None if lang_cfg == "auto" else lang_cfg
                segments, _ = self.model.transcribe(
                    audio,
                    language=lang_param,
                    beam_size=1,
                )
                text = " ".join(s.text.strip() for s in segments).strip().lower()

                if wake_word in text:
                    # Extract any command text that followed the wake word within
                    # the same audio chunk.  If the user says "vox open spotify"
                    # in one breath, this tail ("open spotify") is preserved so
                    # it isn't silently lost when the command recording phase
                    # produces only silence.
                    tail_text = _extract_post_wake(text, wake_word)
                    # Discard single-character tails — likely transcription noise.
                    if len(tail_text) <= 1:
                        tail_text = ""

                    log.info("Wake word detected — listening for command…")
                    self.listening_started.emit()
                    command_audio = self._record_until_silence(mic_device, silence_samples)
                    self.listening_stopped.emit()

                    if command_audio is not None:
                        self._transcribe_and_emit(command_audio, fallback_text=tail_text)
                    elif tail_text:
                        # No additional audio — use same-chunk tail directly.
                        log.info(f"Using wake-word chunk tail as command: '{tail_text}'")
                        self.transcription_ready.emit(tail_text)

                _backoff = 0.0  # reset backoff on a successful iteration

            except Exception as e:
                log.error(f"Wake word loop error: {e}")
                _backoff = min(_backoff + 1.0, 5.0)
                if _backoff > 0:
                    log.info(f"Backing off {_backoff:.0f}s before retrying…")
                time.sleep(_backoff)

    # ── Push-to-talk loop ────────────────────────────────────────────────────

    def _run_push_to_talk_loop(self):
        try:
            import keyboard
        except ImportError:
            log.error(
                "push_to_talk mode requires the 'keyboard' package. "
                "Install it with: pip install keyboard"
            )
            return

        _backoff = 0.0

        while not self._stop_flag.is_set():
            try:
                mic_device       = self.config.get("mic_device", None)
                ptt_key          = self.config.get("push_to_talk_key", "ctrl+shift")
                silence_duration = float(self.config.get("silence_duration",  1.5))
                silence_samples  = int(self.SAMPLE_RATE * silence_duration)
                keys = [k.strip() for k in ptt_key.split("+")]

                # Block until ALL keys in the combo are pressed simultaneously.
                keyboard.wait(ptt_key)

                if self._stop_flag.is_set():
                    break

                log.info("Push-to-talk activated — listening…")
                self.listening_started.emit()

                command_audio = self._record_push_to_talk(
                    mic_device, silence_samples, keyboard, keys
                )
                self.listening_stopped.emit()

                if command_audio is not None:
                    self._transcribe_and_emit(command_audio)

                # Wait for all keys to be fully released before next trigger.
                for k in keys:
                    try:
                        keyboard.wait(k, suppress=False, trigger_on_release=True)
                    except Exception:
                        pass

                _backoff = 0.0

            except Exception as e:
                log.error(f"Push-to-talk loop error: {e}")
                _backoff = min(_backoff + 1.0, 5.0)
                if _backoff > 0:
                    log.info(f"Backing off {_backoff:.0f}s before retrying…")
                time.sleep(_backoff)

    def _record_push_to_talk(
        self, mic_device, silence_samples: int, keyboard, keys: list[str]
    ):
        """Record while keys are held; stop on silence or key release."""
        all_chunks         = []
        consecutive_silent = 0
        chunk_size         = self.SAMPLE_RATE // 2  # 0.5 s read chunks
        max_samples        = int(
            self.SAMPLE_RATE * float(self.config.get("max_record_duration", 30))
        )
        total_samples      = 0
        silence_threshold  = float(self.config.get("silence_threshold", 0.01))

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
                total_samples += chunk_size

                if not all(keyboard.is_pressed(k) for k in keys):
                    break

                rms = float(np.sqrt(np.mean(data ** 2)))
                if rms < silence_threshold:
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
            return np.concatenate(all_chunks, axis=0).flatten()
        return None

    # ── Shared recording helpers ──────────────────────────────────────────────

    def _record_until_silence(self, mic_device, silence_samples: int):
        """Record audio until silence_samples consecutive silent samples.

        Returns a flat float32 ndarray, or None if no audio was captured.
        """
        all_chunks         = []
        consecutive_silent = 0
        chunk_size         = self.SAMPLE_RATE // 2  # 0.5 s read chunks
        max_samples        = int(self.SAMPLE_RATE * float(self.config.get("max_record_duration", 30)))
        total_samples      = 0
        silence_threshold  = float(self.config.get("silence_threshold", 0.01))

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
                total_samples += chunk_size
                rms = float(np.sqrt(np.mean(data ** 2)))
                if rms < silence_threshold:
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
            return np.concatenate(all_chunks, axis=0).flatten()
        return None

    def _transcribe_and_emit(self, audio: np.ndarray, fallback_text: str = ""):
        """Transcribe audio and emit the result.

        If the transcription is empty (e.g. the user had already finished
        speaking before the command-recording phase began), fall back to
        ``fallback_text`` when provided — typically the text that followed
        the wake word in the same detection chunk.
        """
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

        # Strip a leading wake-word echo in case the mic captured it again
        # during the command-recording phase.
        wake_word = self.config.get("wake_word", "vox").lower().strip()
        text = re.sub(
            rf"^\s*{re.escape(wake_word)}\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        if not text and fallback_text:
            text = fallback_text
            log.info(f"Command audio empty — using wake-word chunk tail: '{text}'")

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
