import subprocess
import tempfile
import os
import threading

from utils.config import Config


class Speaker:
    def __init__(self, config: Config):
        self.config = config
        self.enabled = config.get("tts_enabled", True)
        self.piper_path = config.get("piper_path", "piper")
        self.voice_model = config.get("voice_model", "pt_BR-faber-medium")
        self._lock = threading.Lock()

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

                if proc.returncode == 0 and os.path.exists(wav_path):
                    if os.name == "nt":
                        import winsound
                        winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                    else:
                        subprocess.run(["aplay", wav_path], capture_output=True)
            except FileNotFoundError:
                print("[Speaker] Piper not found. Install it: https://github.com/rhasspy/piper")
            except Exception as e:
                print(f"[Speaker] TTS error: {e}")
            finally:
                if wav_path and os.path.exists(wav_path):
                    os.unlink(wav_path)
