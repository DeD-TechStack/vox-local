"""
tests/test_brain.py

Tests for src/brain.py — specifically _extract_action and Brain.process.
No Ollama or network calls are made; requests is mocked.
"""

import json
import sys
import yaml
import pytest
from unittest.mock import patch, MagicMock

from brain import _extract_action, Brain
from utils.config import Config
from executor import Executor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path, extra: dict = None) -> Config:
    data = {}
    if extra:
        data.update(extra)
    p = tmp_path / "settings.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return Config(path=str(p))


def _make_brain(tmp_path) -> Brain:
    cfg = _make_config(tmp_path)
    ex = Executor(cfg)
    return Brain(cfg, ex)


# ---------------------------------------------------------------------------
# _extract_action
# ---------------------------------------------------------------------------

class TestExtractAction:
    def test_clean_json(self):
        text = '{"action":"show_time","params":{}}'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "show_time"
        assert result["params"] == {}

    def test_clean_json_with_params(self):
        text = '{"action":"set_volume","params":{"level":50}}'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "set_volume"
        assert result["params"]["level"] == 50

    def test_fenced_code_block_json(self):
        text = '```json\n{"action":"open_app","params":{"name":"spotify"}}\n```'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "open_app"

    def test_fenced_code_block_no_lang(self):
        text = '```\n{"action":"mute_volume","params":{}}\n```'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "mute_volume"

    def test_json_embedded_in_text(self):
        text = 'Sure, here you go: {"action":"next_track","params":{}} — done!'
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "next_track"

    def test_plain_text_returns_none(self):
        text = "The weather today is sunny and warm."
        result = _extract_action(text)
        assert result is None

    def test_invalid_json_returns_none(self):
        text = '{"action": "show_time", broken json'
        result = _extract_action(text)
        assert result is None

    def test_json_without_action_key_returns_none(self):
        text = '{"name": "foo", "params": {}}'
        result = _extract_action(text)
        assert result is None

    def test_whitespace_padded_json(self):
        text = '   \n  {"action":"show_battery","params":{}}  \n  '
        result = _extract_action(text)
        assert result is not None
        assert result["action"] == "show_battery"


# ---------------------------------------------------------------------------
# Brain.process (mocked Ollama)
# ---------------------------------------------------------------------------

def _mock_stream_response(content: str):
    """Create a mock requests.Response that streams the given content as Ollama would."""
    lines = []
    for char in content:
        lines.append(json.dumps({"message": {"content": char}, "done": False}).encode())
    lines.append(json.dumps({"message": {"content": ""}, "done": True}).encode())

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.iter_lines = MagicMock(return_value=iter(lines))
    return mock_resp


class TestBrainProcess:
    def test_process_plain_text_response(self, tmp_path):
        brain = _make_brain(tmp_path)
        mock_resp = _mock_stream_response("São 14:30.")
        with patch("requests.post", return_value=mock_resp):
            text, is_action = brain.process("que horas são")
        assert is_action is False
        assert "14:30" in text

    def test_process_action_response(self, tmp_path):
        brain = _make_brain(tmp_path)
        payload = '{"action":"show_time","params":{}}'
        mock_resp = _mock_stream_response(payload)
        with patch("requests.post", return_value=mock_resp):
            text, is_action = brain.process("que horas são")
        assert is_action is True
        # show_time returns a string with ":"
        assert ":" in text

    def test_process_history_trimmed_on_cancel(self, tmp_path):
        brain = _make_brain(tmp_path)
        initial_len = len(brain.history)
        mock_resp = _mock_stream_response("partial")
        # cancelled immediately
        with patch("requests.post", return_value=mock_resp):
            text, is_action = brain.process(
                "hello", cancelled=lambda: True
            )
        # User message should be rolled back
        assert len(brain.history) == initial_len

    def test_process_connection_error(self, tmp_path):
        import requests as req
        brain = _make_brain(tmp_path)
        with patch("requests.post", side_effect=req.exceptions.ConnectionError):
            text, is_action = brain.process("hello")
        assert is_action is False
        assert "ollama" in text.lower() or "running" in text.lower()

    def test_process_history_not_stores_raw_json(self, tmp_path):
        brain = _make_brain(tmp_path)
        payload = '{"action":"show_time","params":{}}'
        mock_resp = _mock_stream_response(payload)
        with patch("requests.post", return_value=mock_resp):
            brain.process("que horas são")
        # The assistant history entry should be a short summary, not raw JSON
        assistant_entries = [
            h for h in brain.history if h["role"] == "assistant"
        ]
        assert len(assistant_entries) == 1
        assert assistant_entries[0]["content"].startswith("[action:")

    def test_max_history_respected(self, tmp_path):
        cfg = _make_config(tmp_path, {"max_history": 4})
        ex = Executor(cfg)
        brain = Brain(cfg, ex)

        for i in range(5):
            mock_resp = _mock_stream_response(f"reply {i}")
            with patch("requests.post", return_value=mock_resp):
                brain.process(f"message {i}")

        assert len(brain.history) <= 4
