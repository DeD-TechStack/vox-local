# VOX

A fully **offline**, **privacy-first** voice assistant for your PC — no cloud APIs, no data leaving your machine.

Built with [faster-whisper](https://github.com/guillaumekynast/faster-whisper), [Ollama](https://ollama.com), and [Piper TTS](https://github.com/rhasspy/piper).

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

- **Voice activation** — hold `Alt` to speak, release to send
- **Local LLM** — runs on Ollama (qwen2.5, llama3, mistral, etc.)
- **Local TTS** — Piper voices, zero latency
- **Permission system** — the AI can only run actions you explicitly allow in `settings.yaml`
- **Floating HUD** — minimal overlay, always on top, draggable
- **Fully configurable** — model, hotkey, language, aliases, allowed actions

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- [Piper TTS](https://github.com/rhasspy/piper/releases) binary in `piper/`
- NVIDIA GPU recommended (CUDA) — CPU works but is slower
- Windows 10/11 or Linux

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
pip install comtypes   # Windows only
```

### 2. Pull an Ollama model

```bash
ollama pull qwen2.5:14b   # recommended (requires ~9GB VRAM)
# or
ollama pull llama3.1:8b   # lighter option
```

### 3. Download Piper + voice

Download the Piper binary and place it in `piper/piper.exe`.
Download the voice files and place them in `voices/`:

```
vox-assistant/
├── piper/
│   └── piper.exe
└── voices/
    ├── en_US-ryan-high.onnx
    └── en_US-ryan-high.onnx.json
```

### 4. Run

```bash
cd src
python main.py
```

---

## Usage

| Action | How |
|--------|-----|
| Speak a command | Hold `Alt`, speak, release |
| Change hotkey | Edit `config/settings.yaml` → `hotkey` |
| Add an app alias | Edit `app_aliases` in settings |
| Block an action | Remove it from `allowed_actions` in settings |
| Move the overlay | Click and drag |

### Example commands

```
"Open Spotify"
"Close Discord"
"Volume 40"
"Mute"
"Next track"
"Take a screenshot"
"What time is it?"
"Search for file report"
"Open YouTube"
```

---

## Security Model

VOX uses a **declarative permission system**. The LLM cannot execute arbitrary shell commands — it can only call functions explicitly listed in `allowed_actions`. This prevents prompt injection attacks and accidental destructive operations.

```yaml
allowed_actions:
  - open_app       # allowed
  - set_volume     # allowed
  # delete_files   # not in the list = never runs
  # shell_command  # not in the list = never runs
```

---

## Architecture

```
Microphone
    ↓
faster-whisper (CUDA) — Speech-to-Text
    ↓
Brain (Ollama API) — LLM reasoning
    ↓
Executor — validates action against allowlist
    ↓
Action runs (open app / set volume / etc.)
    ↓
Piper TTS — speaks the response
    ↓
Overlay HUD — shows transcript + response
```

---

## Configuration (`config/settings.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `hotkey` | `alt` | Key to hold while speaking |
| `language` | `en` | Transcription language |
| `whisper_model` | `base` | Whisper model size |
| `ollama_model` | `qwen2.5:14b` | Ollama model to use |
| `tts_enabled` | `true` | Enable/disable voice responses |
| `voice_model` | `en_US-ryan-high.onnx` | Piper voice (.onnx path) |
| `app_aliases` | see file | Map spoken names to executables |
| `allowed_actions` | see file | Whitelist of executable actions |

---

## Roadmap

- [ ] Settings GUI
- [ ] Wake word support (always-on)
- [ ] Custom action plugins
- [ ] Conversation memory / context
- [ ] Linux audio (PipeWire) improvements

---

## License

MIT — use it, fork it, make it yours.
