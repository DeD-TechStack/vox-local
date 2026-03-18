import json
import re
import requests
from typing import Callable

from executor import Executor
from utils.config import Config


# Compact prompt — fewer tokens = faster TTFT
SYSTEM_PROMPT = """You are VOX, a voice assistant. Always reply in English. Be very concise — responses are spoken aloud.

To control the computer, output ONLY a single JSON line:
{"action":"<name>","params":{...}}

AVAILABLE ACTIONS:
open_app(name)      - launch an installed desktop app (discord, spotify, chrome, notepad, steam, etc.)
close_app(name)     - kill a running app by name
set_volume(level)   - set system volume, level is 0 to 100
mute_volume()       - toggle mute
play_pause_media()  - play or pause current media
next_track()        - skip to next track
prev_track()        - go to previous track
search_file(query)  - find a file on disk
open_url(url)       - open a FULL URL (only when user gives an explicit website address)
type_text(text)     - type text into active window
take_screenshot()   - capture screenshot
show_time()         - say the current time
show_battery()      - say battery percentage

FEW-SHOT EXAMPLES (follow these exactly):
User: open discord          → {"action":"open_app","params":{"name":"discord"}}
User: open spotify          → {"action":"open_app","params":{"name":"spotify"}}
User: open chrome           → {"action":"open_app","params":{"name":"chrome"}}
User: close discord         → {"action":"close_app","params":{"name":"discord"}}
User: set volume to 40      → {"action":"set_volume","params":{"level":40}}
User: mute                  → {"action":"mute_volume","params":{}}
User: next song             → {"action":"next_track","params":{}}
User: what time is it       → {"action":"show_time","params":{}}
User: open youtube          → {"action":"open_url","params":{"url":"https://youtube.com"}}
User: take a screenshot     → {"action":"take_screenshot","params":{}}

CRITICAL RULES:
- open_app is for INSTALLED APPS. NEVER use open_url for apps like discord, spotify, steam.
- open_url is ONLY for explicit website addresses the user mentions.
- When outputting an action: output ONLY the JSON. No explanation, no text before or after.
- For conversation/questions: reply with plain English, max 2 sentences.
"""

_JSON_RE = re.compile(r'\{[^{}]*"action"\s*:[^{}]*\}', re.DOTALL)


def _extract_action(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    for match in _JSON_RE.finditer(text):
        try:
            data = json.loads(match.group())
            if "action" in data:
                return data
        except json.JSONDecodeError:
            continue

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    return None


class Brain:
    def __init__(self, config: Config, executor: Executor):
        self.config = config
        self.executor = executor
        self.history: list[dict] = []
        self.base_url = config.get("ollama_url", "http://localhost:11434")
        self.model = config.get("ollama_model", "qwen2.5:3b")

    def process(
        self,
        user_input: str,
        on_token: Callable[[str], None] | None = None,
        on_generating: Callable[[], None] | None = None,
    ) -> tuple[str, bool]:
        self.history.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *self.history,
            ],
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 128,
                "num_ctx": 2048,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60,
                stream=True,
            )
            resp.raise_for_status()

            full_content = ""
            is_action_stream = None

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("message", {}).get("content", "")
                full_content += token

                if is_action_stream is None and full_content.strip():
                    is_action_stream = full_content.lstrip().startswith("{")
                    # Signal overlay to switch from "thinking" to "generating"
                    if on_generating:
                        on_generating()

                if on_token and token and is_action_stream is False:
                    on_token(token)

                if chunk.get("done"):
                    break

        except requests.exceptions.ConnectionError:
            return "Ollama is not running. Start it with: ollama serve", False
        except requests.exceptions.Timeout:
            return "Request timed out. Model may be overloaded.", False
        except Exception as e:
            return f"Error: {e}", False

        content = full_content.strip()
        self.history.append({"role": "assistant", "content": content})
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return self._handle_response(content)

    def _handle_response(self, content: str) -> tuple[str, bool]:
        data = _extract_action(content)
        if data:
            return self.executor.run(data.get("action", ""), data.get("params", {})), True
        return content, False
