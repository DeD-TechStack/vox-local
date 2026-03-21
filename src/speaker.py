import subprocess
import tempfile
import os
import wave
import threading

import numpy as np
import sounddevice as sd

from utils.config import Config
from utils.logger import get_logger

log = get_logger("Speaker")


class Speaker:
    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.Lock()
        self._reload()

    def _reload(self):
        self.enabled = self.config.get("tts_enabled", True)
        self.output_device = self.config.get("output_device", None)

        # Resolve relative paths against the project root (parent of config/).
        project_root = os.path.dirname(os.path.dirname(self.config._path))

        def _resolve(raw: str) -> str:
            if os.path.isabs(raw):
                return raw
            return os.path.normpath(os.path.join(project_root, raw))

        self.piper_path  = _resolve(self.config.get("piper_path",  "piper"))
        self.voice_model = _resolve(self.config.get("voice_model", "en_US-ryan-high.onnx"))

    def reload_config(self):
        self._reload()

    def speak(self, text: str):
        if not self.enabled or not text:
            return
        thread = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        thread.start()

    def _speak_blocking(self, text: str):
        with self._lock:
            wav_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    wav_path = f.name

                proc = subprocess.run(
                    [self.piper_path, "--model", self.voice_model, "--output_file", wav_path],
                    input=text.encode("utf-8"),
                    capture_output=True,
                    timeout=15,
                )

                if proc.returncode != 0:
                    stderr_msg = proc.stderr.decode("utf-8", errors="replace").strip()
                    if stderr_msg:
                        log.error(f"Piper error (exit {proc.returncode}): {stderr_msg}")
                    else:
                        log.error(f"Piper exited with code {proc.returncode} and no stderr output.")
                    return

                if not os.path.exists(wav_path):
                    log.error("Piper exited successfully but produced no output file.")
                    return

                self._play_wav(wav_path)

            except FileNotFoundError:
                log.error("Piper not found. Check piper_path in settings.yaml")
            except subprocess.TimeoutExpired:
                log.error("Piper timed out after 15 s.")
            except Exception as e:
                log.error(f"TTS error: {e}")
            finally:
                if wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)

    def _play_wav(self, path: str):
        with wave.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            raw        = wf.readframes(wf.getnframes())

        dtype_map = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_map.get(sampwidth, np.int16)
        audio = np.frombuffer(raw, dtype=dtype)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        audio = audio.astype(np.float32) / np.iinfo(dtype).max

        # target_rate tracks the actual sample rate of the audio array so that
        # both primary playback and the fallback path use the correct rate.
        target_rate = framerate

        if self.output_device is not None:
            try:
                dev_rate = int(sd.query_devices(self.output_device)["default_samplerate"])
                if dev_rate != framerate:
                    n_out = int(len(audio) * dev_rate / framerate)
                    audio = np.interp(
                        np.linspace(0, len(audio), n_out),
                        np.arange(len(audio)),
                        audio if audio.ndim == 1 else audio[:, 0],
                    ).astype(np.float32)
                    target_rate = dev_rate
            except Exception:
                pass

        try:
            sd.play(audio, samplerate=target_rate, device=self.output_device)
            sd.wait()
        except Exception as e:
            log.warning(f"Output device failed ({e}), falling back to default")
            # Use target_rate (the actual sample rate of the audio array) rather
            # than the original framerate — if the audio was already resampled,
            # playing it at framerate would produce wrong-pitch audio.
            sd.play(audio, samplerate=target_rate, device=None)
            sd.wait()
