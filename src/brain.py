import json
import re
import requests
from typing import Callable

from executor import Executor
from utils.config import Config


SYSTEM_PROMPT = """You are VOX, a local voice assistant running on the user's PC.
Always respond in English. Be extremely concise — responses are spoken aloud.

To control the computer, reply with ONLY a JSON object (no extra text, no markdown):
{"action": "<name>", "params": {<params>}}

For conversation, reply with plain text (max 2 sentences).

AVAILABLE ACTIONS (use exact keys):
{"action": "open_app",        "params": {"name": "spotify"}}
{"action": "close_app",       "params": {"name": "chrome"}}
{"action": "set_volume",      "params": {"level": 50}}
{"action": "mute_volume",     "params": {}}
{"action": "play_pause_media","params": {}}
{"action": "next_track",      "params": {}}
{"action": "prev_track",      "params": {}}
{"action": "search_file",     "params": {"query": "report"}}
{"action": "open_url",        "params": {"url": "https://google.com"}}
{"action": "type_text",       "params": {"text": "hello world"}}
{"action": "take_screenshot", "params": {}}
{"action": "show_time",       "params": {}}
{"action": "show_battery",    "params": {}}

RULES:
- Action request → reply ONLY with the JSON, nothing else
- Not an action → reply with short plain English text
- NEVER invent actions outside this list
"""

_JSON_RE = re.compile(r'\{[^{}]*"action"\s*:[^{}]*\}', re.DOTALL)


def _extract_action(text: str) -> dict | None:
    """Find the first valid action JSON anywhere in the LLM response."""
    # Direct parse (ideal case)
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Search anywhere in the response (handles markdown, extra text, etc.)
    for match in _JSON_RE.finditer(text):
        try:
            data = json.loads(match.group())
            if "action" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Last resort: find outermost { } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
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
        self.model = config.get("ollama_model", "qwen2.5:14b")

    def process(
        self,
        user_input: str,
        on_token: Callable[[str], None] | None = None,
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
                "num_predict": 256,
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
            is_action_stream = None  # determined after first non-whitespace token

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("message", {}).get("content", "")
                full_content += token

                # Decide streaming strategy once we know if it's JSON or text
                if is_action_stream is None and full_content.strip():
                    is_action_stream = full_content.lstrip().startswith("{")

                # Stream tokens to overlay only for text responses
                if on_token and token and is_action_stream is False:
                    on_token(token)

                if chunk.get("done"):
                    break

        except requests.exceptions.ConnectionError:
            return "Ollama não está rodando. Inicie com: ollama serve", False
        except requests.exceptions.Timeout:
            return "Tempo esgotado. O modelo está ocupado ou é muito pesado.", False
        except Exception as e:
            return f"Erro: {e}", False

        content = full_content.strip()
        self.history.append({"role": "assistant", "content": content})
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return self._handle_response(content)

    def _handle_response(self, content: str) -> tuple[str, bool]:
        data = _extract_action(content)
        if data:
            action = data.get("action", "")
            params = data.get("params", {})
            return self.executor.run(action, params), True
        return content, False
