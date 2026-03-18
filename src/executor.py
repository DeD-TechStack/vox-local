import os
import subprocess
import glob
from datetime import datetime
from typing import Any

from utils.config import Config


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
            return f"Action '{action}' is not allowed."

        handler = self._action_map.get(action)
        if not handler:
            return f"Action '{action}' is not implemented."

        try:
            return handler(**params)
        except TypeError as e:
            return f"Invalid parameters for '{action}': {e}"
        except Exception as e:
            return f"Error running '{action}': {e}"

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
            subprocess.Popen(["xdg-open", target])
        return f"Opening {name}."

    def _close_app(self, name: str) -> str:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/IM", f"{name}.exe"], capture_output=True)
        else:
            subprocess.run(["pkill", "-f", name], capture_output=True)
        return f"Closing {name}."

    def _set_volume(self, level: int) -> str:
        level = max(0, min(100, int(level)))
        if os.name == "nt":
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(level / 100, None)
        return f"Volume set to {level}%."

    def _mute_volume(self) -> str:
        if os.name == "nt":
            import keyboard
            keyboard.send("volume mute")
        return "Muted."

    def _play_pause_media(self) -> str:
        import keyboard
        keyboard.send("play/pause media")
        return "Play/pause."

    def _next_track(self) -> str:
        import keyboard
        keyboard.send("next track")
        return "Next track."

    def _prev_track(self) -> str:
        import keyboard
        keyboard.send("previous track")
        return "Previous track."

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
            return f"Found: {names}"
        return f"No files found for '{query}'."

    def _open_url(self, url: str) -> str:
        import webbrowser
        webbrowser.open(url)
        return f"Opening {url}."

    def _type_text(self, text: str) -> str:
        import pyautogui
        pyautogui.write(text, interval=0.05)
        return "Text typed."

    def _take_screenshot(self) -> str:
        import pyautogui
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.expanduser(f"~/Desktop/screenshot_{ts}.png")
        pyautogui.screenshot(path)
        return f"Screenshot saved to Desktop."

    def _show_time(self) -> str:
        now = datetime.now().strftime("%H:%M")
        return f"It's {now}."

    def _show_battery(self) -> str:
        try:
            import psutil
            battery = psutil.sensors_battery()
            if battery:
                status = "charging" if battery.power_plugged else "on battery"
                return f"Battery at {int(battery.percent)}%, {status}."
        except Exception:
            pass
        return "Could not read battery status."
