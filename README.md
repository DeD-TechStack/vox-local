# VOX

**Topics:** voice-assistant, local-ai, ollama, whisper, python, offline-ai, piper-tts

A fully **offline**, **privacy-first** voice assistant for your PC — no cloud APIs, no data leaving your machine.

Built with [faster-whisper](https://github.com/guillaumekynast/faster-whisper), [Ollama](https://ollama.com), and [Piper TTS](https://github.com/rhasspy/piper).

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why VOX?

- **100% offline** — speech recognition, reasoning, and TTS all run locally
- **No cloud APIs** — your voice never leaves your machine
- **Works without internet** after initial model setup
- **Declarative permission system** — the AI can only run actions you explicitly allow; no arbitrary shell access

---

## Features

- **Voice activation** — hold `Alt` to speak, release to send
- **Local LLM** — runs on Ollama (qwen2.5, llama3, mistral, etc.)
- **Local TTS** — Piper voices, zero latency
- **Permission system** — the AI can only run actions you explicitly allow in `settings.yaml`
- **Floating HUD** — minimal overlay, always on top, draggable
- **Fully configurable** — model, hotkey, language, aliases, allowed actions
- **Wake word mode** — optional always-on mode, no external library needed

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- [Piper TTS](https://github.com/rhasspy/piper/releases) binary in `piper/`
- NVIDIA GPU recommended (CUDA) — CPU works but is slower
- Windows 10/11 or Linux

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/vox.git
cd vox
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
pip install comtypes   # Windows only
```

### 3. Pull an Ollama model

```bash
ollama pull qwen2.5:14b   # recommended (requires ~9GB VRAM)
# or
ollama pull llama3.1:8b   # lighter option
```

### 4. Download Piper + voice

Download the Piper binary and place it in `piper/piper.exe`.
Download the voice files and place them in `voices/`:

```
vox/
├── piper/
│   └── piper.exe
└── voices/
    ├── en_US-ryan-high.onnx
    └── en_US-ryan-high.onnx.json
```

### 5. Run

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
| `voice_model` | `voices/en_US-ryan-high.onnx` | Piper voice (.onnx path, relative to project root) |
| `wake_word_enabled` | `false` | Enable always-on wake word mode |
| `wake_word` | `hey vox` | Phrase to trigger wake word mode |
| `app_aliases` | see file | Map spoken names to executables |
| `allowed_actions` | see file | Whitelist of executable actions |

---

## Troubleshooting

**Ollama not running**
```
Error: connection refused
```
Start Ollama with:
```bash
ollama serve
```

**CUDA not available**
VOX falls back to CPU automatically. To force CPU explicitly:
```yaml
whisper_device: cpu
whisper_compute_type: int8
```

**Piper not found**
```
[Speaker] Piper not found. Check piper_path in settings.yaml
```
Check that `piper_path` in `config/settings.yaml` points to the correct binary. Default: `piper/piper/piper.exe` (relative to project root).

**No audio input / microphone not detected**
- Check microphone permissions in your OS settings
- Run VOX and check the console — it lists all detected devices with their indices
- Set `mic_device: <index>` in `settings.yaml` to use a specific device
- On Windows, verify `sounddevice` default input device with:
  ```python
  import sounddevice; print(sounddevice.query_devices())
  ```

---

## Roadmap

- [x] Wake word support (always-on)
- [ ] Settings GUI
- [ ] Custom action plugins
- [ ] Conversation memory / context
- [ ] Linux audio (PipeWire) improvements

---

## License

MIT — use it, fork it, make it yours.
