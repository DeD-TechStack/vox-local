import os
import subprocess
import glob
from datetime import datetime
from typing import Any

from utils.config import Config
from utils.logger import get_logger

log = get_logger("Executor")


class Executor:
    def __init__(self, config: Config):
        self.config = config
        self.allowed_actions = set(config.get("allowed_actions", [
            "open_app", "close_app", "set_volume", "mute_volume",
            "play_pause_media", "next_track", "prev_track",
            "search_file", "open_url", "type_text",
            "take_screenshot", "show_time", "show_battery",
        ]))

        self._action_map = {
            "open_app":         self._open_app,
            "close_app":        self._close_app,
            "set_volume":       self._set_volume,
            "mute_volume":      self._mute_volume,
            "play_pause_media": self._play_pause_media,
            "next_track":       self._next_track,
            "prev_track":       self._prev_track,
            "search_file":      self._search_file,
            "open_url":         self._open_url,
            "type_text":        self._type_text,
            "take_screenshot":  self._take_screenshot,
            "show_time":        self._show_time,
            "show_battery":     self._show_battery,
        }

    def run(self, action: str, params: dict[str, Any]) -> str:
        if action not in self.allowed_actions:
            return f"Ação '{action}' não está permitida."

        handler = self._action_map.get(action)
        if not handler:
            return f"Ação '{action}' não está implementada."

        if params is None:
            params = {}

        try:
            return handler(**params)
        except TypeError as e:
            return f"Parâmetros inválidos para '{action}': {e}"
        except Exception as e:
            return f"Erro ao executar '{action}': {e}"

    # ─── Actions ────────────────────────────────────────────────────────────

    def _open_app(self, name: str) -> str:
        aliases = self.config.get("app_aliases", {})
        target = aliases.get(name.lower(), name)
        if os.name == "nt":
            # URI scheme (discord://, spotify:, etc.) — let Windows registry handle it
            if "://" in target or (target.endswith(":") and len(target) > 2):
                os.startfile(target)
            else:
                # Use Windows `start` which searches PATH + App Paths registry
                subprocess.Popen(f'start "" "{target}"', shell=True)
        else:
            # xdg-open handles apps and URIs on most Linux desktops
            subprocess.Popen(["xdg-open", target])
        return f"Abrindo {name}."

    def _close_app(self, name: str) -> str:
        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/F", "/IM", f"{name}.exe"],
                capture_output=True,
            )
            # taskkill exits with 128 when the process is not found
            if result.returncode == 128:
                return f"Nenhum processo '{name}' encontrado."
            if result.returncode != 0:
                return f"Não foi possível fechar '{name}'."
        else:
            result = subprocess.run(["pkill", "-f", name], capture_output=True)
            if result.returncode != 0:
                return f"Nenhum processo '{name}' encontrado."
        return f"Fechando {name}."

    def _set_volume(self, level) -> str:
        # Accept level as int or string (LLM may produce either)
        try:
            level = int(level)
        except (TypeError, ValueError):
            return "Nível de volume inválido. Informe um número entre 0 e 100."
        level = max(0, min(100, level))
        if os.name == "nt":
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(level / 100, None)
        else:
            # Try pactl first (PulseAudio / PipeWire), fall back to amixer
            pactl = subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                capture_output=True,
            )
            if pactl.returncode != 0:
                amixer = subprocess.run(
                    ["amixer", "-q", "sset", "Master", f"{level}%"],
                    capture_output=True,
                )
                if amixer.returncode != 0:
                    log.warning("set_volume: neither pactl nor amixer succeeded on Linux.")
                    return "Não foi possível ajustar o volume no Linux."
        return f"Volume em {level}%."

    def _mute_volume(self) -> str:
        if os.name == "nt":
            import keyboard
            keyboard.send("volume mute")
        else:
            # Try pactl first, then keyboard module as fallback
            pactl = subprocess.run(
                ["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"],
                capture_output=True,
            )
            if pactl.returncode != 0:
                try:
                    import keyboard
                    keyboard.send("volume mute")
                except Exception:
                    log.warning("mute_volume: neither pactl nor keyboard succeeded on Linux.")
                    return "Não foi possível silenciar no Linux."
        return "Mudo alternado."

    def _play_pause_media(self) -> str:
        import keyboard
        keyboard.send("play/pause media")
        return "Play/pause."

    def _next_track(self) -> str:
        import keyboard
        keyboard.send("next track")
        return "Próxima faixa."

    def _prev_track(self) -> str:
        import keyboard
        keyboard.send("previous track")
        return "Faixa anterior."

    def _search_file(self, query: str) -> str:
        search_dirs = self.config.get("search_dirs", [
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/Desktop"),
        ])
        results = []
        for directory in search_dirs:
            pattern = os.path.join(directory, "**", f"*{query}*")
            found = glob.glob(pattern, recursive=True)[:3]
            results.extend(found)
        if results:
            names = ", ".join(os.path.basename(r) for r in results[:3])
            return f"Encontrei: {names}"
        return f"Nenhum arquivo encontrado para '{query}'."

    def _open_url(self, url: str) -> str:
        import webbrowser
        webbrowser.open(url)
        return f"Abrindo {url}."

    def _type_text(self, text: str) -> str:
        import pyautogui
        pyautogui.write(text, interval=0.05)
        return "Texto digitado."

    def _take_screenshot(self) -> str:
        import pyautogui
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
        pyautogui.screenshot(path)
        return "Screenshot salvo na Área de Trabalho."

    def _show_time(self) -> str:
        now = datetime.now().strftime("%H:%M")
        return f"São {now}."

    def _show_battery(self) -> str:
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery:
                status = "carregando" if battery.power_plugged else "na bateria"
                return f"Bateria em {int(battery.percent)}%, {status}."
        except Exception:
            pass
        return "Não foi possível ler o status da bateria."
