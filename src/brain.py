import json
import requests
from typing import Any

from executor import Executor
from utils.config import Config


SYSTEM_PROMPT = """You are VOX, a local voice assistant running on the user's PC.
You understand commands in English.

You can control the computer by calling functions. Always respond in the same language the user used.
Be concise — your responses will be spoken aloud.

Available actions (call them by responding ONLY with valid JSON in this format):
{
  "action": "<action_name>",
  "params": { ... }
}

If no action is needed, respond with plain text.

Available actions:
- open_app: { "name": "spotify" }
- close_app: { "name": "chrome" }
- set_volume: { "level": 50 }
- mute_volume: {}
- play_pause_media: {}
- next_track: {}
- prev_track: {}
- search_file: { "query": "relatorio" }
- open_url: { "url": "https://..." }
- type_text: { "text": "hello world" }
- take_screenshot: {}
- show_time: {}
- show_battery: {}

NEVER perform actions outside this list. If the user asks for something not on the list, explain politely that you cannot do that yet.
"""


class Brain:
    def __init__(self, config: Config, executor: Executor):
        self.config = config
        self.executor = executor
        self.history = []
        self.base_url = config.get("ollama_url", "http://localhost:11434")
        self.model = config.get("ollama_model", "qwen2.5:14b")

    def process(self, user_input: str) -> str:
        self.history.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                *self.history,
            ],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 200,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            return "Ollama não está rodando. Inicie com: ollama serve"
        except Exception as e:
            return f"Erro ao processar: {e}"

        self.history.append({"role": "assistant", "content": content})

        if len(self.history) > 20:
            self.history = self.history[-20:]

        return self._handle_response(content)

    def _handle_response(self, content: str) -> str:
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                data = json.loads(stripped)
                action = data.get("action")
                params = data.get("params", {})
                return self.executor.run(action, params)
            except json.JSONDecodeError:
                pass
        return content
