import json
import re
import requests
from typing import Callable

from executor import Executor
from utils.config import Config
from utils.logger import get_logger

log = get_logger("Brain")


# Per-action documentation lines used to build the dynamic system prompt.
# Only actions present in the current `allowed_actions` config are included.
_ACTION_DOCS: dict[str, str] = {
    "open_app":         "open_app(name)      - abrir um aplicativo instalado (discord, spotify, chrome, notepad, steam, etc.)",
    "close_app":        "close_app(name)     - fechar um aplicativo pelo nome",
    "set_volume":       "set_volume(level)   - definir volume do sistema, de 0 a 100",
    "mute_volume":      "mute_volume()       - alternar mudo",
    "play_pause_media": "play_pause_media()  - play ou pause da mídia atual",
    "next_track":       "next_track()        - próxima faixa",
    "prev_track":       "prev_track()        - faixa anterior",
    "search_file":      "search_file(query)  - buscar um arquivo no disco",
    "open_url":         "open_url(url)       - abrir uma URL completa (somente quando o usuário diz um endereço de site)",
    "type_text":        "type_text(text)     - digitar texto na janela ativa",
    "take_screenshot":  "take_screenshot()   - capturar tela",
    "show_time":        "show_time()         - dizer a hora atual",
    "show_battery":     "show_battery()      - dizer a porcentagem da bateria",
}

_PROMPT_TEMPLATE = """\
Você é VOX, um assistente de voz bilíngue. Detecte o idioma do usuário (português ou inglês) e responda SEMPRE no mesmo idioma que ele usou. Seja muito conciso — as respostas são faladas em voz alta.

Para controlar o computador, gere APENAS uma linha JSON:
{{"action":"<nome>","params":{{...}}}}

AÇÕES DISPONÍVEIS:
{action_list}

EXEMPLOS (siga exatamente):
Usuário: abrir discord        → {{"action":"open_app","params":{{"name":"discord"}}}}
Usuário: abrir spotify        → {{"action":"open_app","params":{{"name":"spotify"}}}}
Usuário: fechar discord       → {{"action":"close_app","params":{{"name":"discord"}}}}
Usuário: volume 40            → {{"action":"set_volume","params":{{"level":40}}}}
Usuário: mutar                → {{"action":"mute_volume","params":{{}}}}
Usuário: próxima música       → {{"action":"next_track","params":{{}}}}
Usuário: que horas são        → {{"action":"show_time","params":{{}}}}
Usuário: abrir youtube        → {{"action":"open_url","params":{{"url":"https://youtube.com"}}}}
Usuário: tirar screenshot     → {{"action":"take_screenshot","params":{{}}}}

REGRAS CRÍTICAS:
- open_app é para APPS INSTALADOS. NUNCA use open_url para discord, spotify, steam.
- open_url é SOMENTE para endereços de sites que o usuário mencionar explicitamente.
- Ao gerar uma ação: gere APENAS o JSON. Sem explicação, sem texto antes ou depois.
- Para conversa/perguntas: responda no idioma do usuário, máximo 2 frases.
"""


def _build_system_prompt(allowed_actions: list[str]) -> str:
    """Build the system prompt filtered to the currently allowed actions.

    When no actions are allowed the model is still useful for conversation
    but will not be prompted to generate any JSON actions.
    """
    allowed_set = set(allowed_actions)
    lines = [
        doc for action, doc in _ACTION_DOCS.items()
        if action in allowed_set
    ]
    action_list = "\n".join(lines) if lines else "(none — all actions are currently disabled)"
    return _PROMPT_TEMPLATE.format(action_list=action_list)


# Matches a JSON object containing "action" key (handles fenced blocks too)
_JSON_RE = re.compile(r'\{[^{}]*"action"\s*:[^{}]*\}', re.DOTALL)
_FENCE_RE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)


def _extract_action(text: str) -> dict | None:
    """
    Try four strategies to extract a JSON action from model output:
    1. Entire response is a bare JSON object.
    2. Fenced code block: ```json { ... } ```.
    3. Regex scan for {"action": ...} anywhere in the text.
    4. Substring from first '{' to last '}'.
    Returns a dict with at least "action" key, or None.
    """
    stripped = text.strip()

    # Strategy 1: response is pure JSON
    if stripped.startswith("{"):
        try:
            data = json.loads(stripped)
            if "action" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Strategy 2: fenced code block
    for fence_match in _FENCE_RE.finditer(text):
        try:
            data = json.loads(fence_match.group(1))
            if "action" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Strategy 3: regex scan
    for match in _JSON_RE.finditer(text):
        try:
            data = json.loads(match.group())
            if "action" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Strategy 4: substring heuristic
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
        self.config   = config
        self.executor = executor
        self.history: list[dict] = []
        # NOTE: ollama_url and ollama_model are NOT cached here.
        # They are read from config on every process() call so that
        # changes saved from the Assistant tab take effect immediately
        # without restarting the application.

    def process(
        self,
        user_input: str,
        on_token: Callable[[str], None] | None = None,
        on_generating: Callable[[], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> tuple[str, bool]:
        # Read live config on every call — no stale cached values
        base_url    = self.config.get("ollama_url",   "http://localhost:11434")
        model       = self.config.get("ollama_model", "qwen2.5:14b")
        max_history = int(self.config.get("max_history", 20))

        # System prompt reflects the current allowed_actions at call time
        allowed       = self.config.get("allowed_actions", list(_ACTION_DOCS.keys()))
        system_prompt = _build_system_prompt(allowed)

        self.history.append({"role": "user", "content": user_input})

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *self.history,
            ],
            "stream": True,
            "options": {
                "temperature": 0.2,
                "num_predict": 100,
                "num_ctx": 1024,
            },
        }

        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=30,
                stream=True,
            )
            resp.raise_for_status()

            full_content = ""
            is_action_stream = None

            for raw_line in resp.iter_lines():
                if cancelled and cancelled():
                    log.info("Brain stream cancelled.")
                    break

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

        if cancelled and cancelled():
            # Do not store partial output in history
            self.history.pop()
            return "", False

        content = full_content.strip()

        # Store a clean summary in history — avoid raw JSON blobs in memory
        history_content = content
        action_data = _extract_action(content)
        if action_data:
            history_content = f"[action: {action_data.get('action', '?')}]"

        self.history.append({"role": "assistant", "content": history_content})
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

        return self._handle_response(content)

    def _handle_response(self, content: str) -> tuple[str, bool]:
        data = _extract_action(content)
        if data:
            return self.executor.run(data.get("action", ""), data.get("params") or {}), True
        return content, False
