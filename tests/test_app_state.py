"""Tests for AppState — central runtime state model."""
import pytest
from unittest.mock import MagicMock, patch

from utils.config import Config
from app_state import AppState


@pytest.fixture
def config(tmp_config_path):
    return Config(tmp_config_path)


@pytest.fixture
def state(config):
    return AppState(config)


# ── Status ────────────────────────────────────────────────────────────────────

def test_initial_status_is_idle(state):
    assert state.status == "idle"


def test_set_status_updates_value(state):
    state.set_status("listening")
    assert state.status == "listening"

    state.set_status("generating")
    assert state.status == "generating"

    state.set_status("idle")
    assert state.status == "idle"


# ── Ollama ok ─────────────────────────────────────────────────────────────────

def test_initial_ollama_ok_is_false(state):
    assert state.ollama_ok is False


def test_set_ollama_ok_true(state):
    state.set_ollama_ok(True)
    assert state.ollama_ok is True


def test_set_ollama_ok_false_adds_diagnostic(state):
    state.set_ollama_ok(True)   # go true first so transition to false fires
    state.set_ollama_ok(False)
    assert state.ollama_ok is False
    diag_msgs = [d["message"] for d in state.diagnostics]
    assert any("Ollama" in m for m in diag_msgs)


def test_set_ollama_ok_no_duplicate_diagnostic_when_unchanged(state):
    state.set_ollama_ok(False)  # was already False — no change
    # No new diagnostic should be added for unchanged False→False
    count_before = len(state.diagnostics)
    state.set_ollama_ok(False)
    assert len(state.diagnostics) == count_before


# ── Transcript / response / action ───────────────────────────────────────────

def test_set_transcript(state):
    state.set_transcript("open Spotify")
    assert state.transcript == "open Spotify"


def test_set_response(state):
    state.set_response("Opening Spotify now.")
    assert state.response == "Opening Spotify now."


def test_set_last_action(state):
    state.set_last_action("open_app(spotify)")
    assert state.last_action == "open_app(spotify)"


# ── Diagnostics ───────────────────────────────────────────────────────────────

def test_add_diagnostic_stores_entry(state):
    state.add_diagnostic("warning", "Ollama not found")
    assert len(state.diagnostics) == 1
    entry = state.diagnostics[0]
    assert entry["level"]   == "warning"
    assert entry["message"] == "Ollama not found"
    assert "timestamp" in entry


def test_add_multiple_diagnostics(state):
    state.add_diagnostic("info",    "Piper found.")
    state.add_diagnostic("warning", "Voice model missing.")
    state.add_diagnostic("error",   "Audio device invalid.")
    assert len(state.diagnostics) == 3
    levels = [d["level"] for d in state.diagnostics]
    assert levels == ["info", "warning", "error"]


def test_clear_diagnostics(state):
    state.add_diagnostic("info", "test")
    state.add_diagnostic("info", "test2")
    state.clear_diagnostics()
    assert state.diagnostics == []


def test_diagnostics_capped_at_500(state):
    for i in range(510):
        state.add_diagnostic("info", f"msg {i}")
    assert len(state.diagnostics) <= 500
    # Most recent entries should be kept
    assert state.diagnostics[-1]["message"] == "msg 509"


# ── History ───────────────────────────────────────────────────────────────────

def test_add_history_entry(state):
    state.add_history_entry("open Spotify", "Opening Spotify.")
    assert len(state.history) == 1
    entry = state.history[0]
    assert entry["transcript"] == "open Spotify"
    assert entry["response"]   == "Opening Spotify."
    assert entry["action"]     == ""
    assert "timestamp" in entry


def test_add_history_entry_with_action(state):
    state.add_history_entry("open Spotify", "", action="open_app(spotify)")
    entry = state.history[0]
    assert entry["action"] == "open_app(spotify)"
    assert entry["response"] == ""


def test_clear_history(state):
    state.add_history_entry("cmd", "resp")
    state.clear_history()
    assert state.history == []


def test_history_capped_at_200(state):
    for i in range(210):
        state.add_history_entry(f"cmd {i}", f"resp {i}")
    assert len(state.history) <= 200
    # Most recent entries kept
    assert state.history[-1]["transcript"] == "cmd 209"


def test_history_returns_copy(state):
    state.add_history_entry("cmd", "resp")
    h = state.history
    h.clear()
    assert len(state.history) == 1  # internal list unaffected


# ── Config property ───────────────────────────────────────────────────────────

def test_config_property_returns_config(state, config):
    assert state.config is config


# ── State transitions mirror real workflow ────────────────────────────────────

def test_full_interaction_workflow(state):
    """Simulate a realistic interaction and verify state transitions."""
    state.set_status("listening")
    assert state.status == "listening"

    state.set_transcript("what time is it")
    state.set_status("transcribing")
    assert state.transcript == "what time is it"

    state.set_status("generating")
    state.set_response("It is 14:30.")
    state.add_history_entry("what time is it", "It is 14:30.")
    state.set_status("idle")

    assert state.status    == "idle"
    assert state.response  == "It is 14:30."
    assert len(state.history) == 1
    assert state.history[0]["transcript"] == "what time is it"
