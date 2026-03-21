"""
tests/conftest.py

Shared pytest fixtures and import path setup.

Hardware-dependent modules (sounddevice, PyQt6, keyboard, pycaw) are
mocked at import time so tests run in any environment.
"""

import sys
import types
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stub out hardware / GUI dependencies before any project module is imported
# ---------------------------------------------------------------------------

def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# sounddevice
sd_stub = _stub("sounddevice")
sd_stub.query_devices = MagicMock(return_value=[])
sd_stub.query_hostapis = MagicMock(return_value=[])
sd_stub.default = MagicMock()
sd_stub.default.device = [0, 0]
sd_stub.rec = MagicMock()
sd_stub.wait = MagicMock()
sd_stub.play = MagicMock()
sd_stub.InputStream = MagicMock()
sys.modules.setdefault("sounddevice", sd_stub)

# PyQt6 — stub out the entire namespace with minimal working attributes.
# QThread, pyqtSignal, and pyqtSlot must behave well enough for modules that
# define listener/overlay classes to be imported without a real Qt installation.
_QThread_stub = type("QThread", (), {
    "__init__": lambda self, *a, **kw: None,
    "start": lambda self: None,
    "wait": lambda self, *a: True,
    "isRunning": lambda self: False,
    "quit": lambda self: None,
})

def _pyqtSignal(*args, **kwargs):
    """Stub factory that returns a descriptor-like object compatible with connect()."""
    class _Sig:
        def connect(self, *a, **kw): pass
        def emit(self, *a, **kw): pass
        def disconnect(self, *a, **kw): pass
    return _Sig()

def _pyqtSlot(*args, **kwargs):
    """Stub decorator that passes the decorated function through unchanged."""
    def _decorator(fn):
        return fn
    return _decorator

_qt_core_stub = _stub(
    "PyQt6.QtCore",
    QThread=_QThread_stub,
    pyqtSignal=_pyqtSignal,
    pyqtSlot=_pyqtSlot,
    QTimer=MagicMock(),
    Qt=MagicMock(),
    QRectF=MagicMock(),
)
sys.modules.setdefault("PyQt6.QtCore", _qt_core_stub)

for _qt_mod in ("PyQt6", "PyQt6.QtWidgets", "PyQt6.QtGui"):
    sys.modules.setdefault(_qt_mod, _stub(_qt_mod))

# keyboard
sys.modules.setdefault("keyboard", _stub("keyboard"))

# pycaw / comtypes
sys.modules.setdefault("pycaw", _stub("pycaw"))
sys.modules.setdefault("pycaw.pycaw", _stub("pycaw.pycaw"))
sys.modules.setdefault("comtypes", _stub("comtypes", CLSCTX_ALL=0))

# faster_whisper
sys.modules.setdefault("faster_whisper", _stub("faster_whisper", WhisperModel=MagicMock()))

# numpy — use real numpy; if unavailable fall back to stub
try:
    import numpy  # noqa: F401
except ImportError:
    sys.modules.setdefault("numpy", _stub("numpy"))

# requests — real library expected; if missing, stub
try:
    import requests  # noqa: F401
except ImportError:
    sys.modules.setdefault("requests", _stub("requests"))

# pyautogui
sys.modules.setdefault("pyautogui", _stub("pyautogui"))

# psutil — allow real import; tests that need it will mock specific calls
try:
    import psutil  # noqa: F401
except ImportError:
    sys.modules.setdefault("psutil", _stub("psutil"))


# ---------------------------------------------------------------------------
# Make src/ importable without installing the package
# ---------------------------------------------------------------------------

import os

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if _SRC not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_config_path(tmp_path):
    """Returns a path inside a fresh temp directory for a config file."""
    return str(tmp_path / "settings.yaml")
