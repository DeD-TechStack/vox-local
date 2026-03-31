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

    def reload_config(self) -> None:
        """Refresh the cached allowed_actions set from the current config.

        No longer required for runtime enforcement — run() reads the config
        live on every call.  Kept for backwards compatibility and so that
        self.allowed_actions remains accurate for external inspection (tests,
        diagnostics).
        """
        self.allowed_actions = set(self.config.get(
            "allowed_actions", list(self._action_map.keys())
        ))
        log.info(f"Executor: allowed_actions refreshed ({len(self.allowed_actions)} actions).")

    def run(self, action: str, params: dict[str, Any]) -> str:
        # Read allowed_actions live from config on every call.
        # This keeps the executor in sync with Brain, which also reads the
        # config fresh, so there is no silent drift between the LLM prompt
        # and what the executor will actually permit.
        allowed = set(self.config.get("allowed_actions", list(self._action_map.keys())))
        if action not in allowed:
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
            # URI scheme (discord://, spotify:, etc.) — let Windows registry handle it.
            if "://" in target or (target.endswith(":") and len(target) > 2):
                os.startfile(target)
            else:
                # Use Windows `start` which searches PATH + App Paths registry.
                subprocess.Popen(f'start "" "{target}"', shell=True)
        else:
            # On Linux, xdg-open works for URI schemes (discord://, etc.).
            # For plain executable names, prefer direct launch via subprocess.
            if "://" in target or (target.endswith(":") and len(target) > 2):
                subprocess.Popen(["xdg-open", target])
            else:
                subprocess.Popen([target])
        return f"Abrindo {name}."

    def _close_app(self, name: str) -> str:
        aliases = self.config.get("app_aliases", {})
        alias   = aliases.get(name.lower(), name)

        # Derive the process name from the alias.
        # URI-scheme aliases ("discord://", "spotify:", "steam://open/main")
        # map to their base name as the process executable.
        if "://" in alias:
            proc_name = alias.split("://")[0]
        elif alias.endswith(":") and len(alias) > 1:
            proc_name = alias[:-1]
        else:
            proc_name = alias

        if os.name == "nt":
            result = subprocess.run(
                ["taskkill", "/F", "/IM", f"{proc_name}.exe"],
                capture_output=True,
            )
            # taskkill exits 128 when the process is not found.
            if result.returncode == 128:
                return f"Nenhum processo '{name}' encontrado."
            if result.returncode != 0:
                return f"Não foi possível fechar '{name}'."
        else:
            result = subprocess.run(["pkill", "-f", proc_name], capture_output=True)
            if result.returncode != 0:
                return f"Nenhum processo '{name}' encontrado."
        return f"Fechando {name}."

    def _set_volume(self, level) -> str:
        # Accept level as int or string (LLM may produce either).
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
            # Try pactl first (PulseAudio / PipeWire), fall back to amixer.
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
            # Try pactl first, then keyboard module as fallback.
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
        if os.name != "nt":
            # keyboard.send() for media keys requires root on Linux and may fail silently.
            # playerctl is the recommended approach on Linux desktops.
            result = subprocess.run(["playerctl", "play-pause"], capture_output=True)
            if result.returncode != 0:
                # Fall back to keyboard module; may require elevated permissions.
                try:
                    import keyboard
                    keyboard.send("play/pause media")
                    return "Play/pause."
                except Exception as e:
                    log.warning(f"play_pause_media: keyboard fallback failed on Linux: {e}")
                    return (
                        "Controle de mídia não disponível. "
                        "Instale playerctl ou execute como administrador no Linux."
                    )
            return "Play/pause."
        import keyboard
        keyboard.send("play/pause media")
        return "Play/pause."

    def _next_track(self) -> str:
        if os.name != "nt":
            result = subprocess.run(["playerctl", "next"], capture_output=True)
            if result.returncode != 0:
                try:
                    import keyboard
                    keyboard.send("next track")
                    return "Próxima faixa."
                except Exception as e:
                    log.warning(f"next_track: keyboard fallback failed on Linux: {e}")
                    return (
                        "Controle de mídia não disponível. "
                        "Instale playerctl ou execute como administrador no Linux."
                    )
            return "Próxima faixa."
        import keyboard
        keyboard.send("next track")
        return "Próxima faixa."

    def _prev_track(self) -> str:
        if os.name != "nt":
            result = subprocess.run(["playerctl", "previous"], capture_output=True)
            if result.returncode != 0:
                try:
                    import keyboard
                    keyboard.send("previous track")
                    return "Faixa anterior."
                except Exception as e:
                    log.warning(f"prev_track: keyboard fallback failed on Linux: {e}")
                    return (
                        "Controle de mídia não disponível. "
                        "Instale playerctl ou execute como administrador no Linux."
                    )
            return "Faixa anterior."
        import keyboard
        keyboard.send("previous track")
        return "Faixa anterior."

    def _search_file(self, query: str) -> str:
        raw_dirs = self.config.get("search_dirs", [
            "~/Documents",
            "~/Downloads",
            "~/Desktop",
        ])
        # Expand ~ so paths like "~/Documents" resolve correctly regardless
        # of whether they come from config YAML or the fallback defaults.
        search_dirs = [os.path.expanduser(d) for d in raw_dirs]
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
