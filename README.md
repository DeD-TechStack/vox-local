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

- **Wake word activation** — say "VOX" to activate; no hotkey required
- **Push-to-talk mode** — optional `Ctrl+Shift` press-to-talk (configurable)
- **Same-utterance commands** — "vox open spotify" said in one breath is correctly captured
- **Local LLM** — runs on Ollama (qwen2.5, llama3, mistral, etc.)
- **Local TTS** — Piper voices, zero latency
- **Permission system** — the AI can only run actions you explicitly allow in `settings.yaml`
- **Floating HUD** — minimal overlay, always on top, draggable
- **Fully configurable** — model, activation mode, language, aliases, allowed actions

---

## Platform Support

| Feature | Windows | Linux |
|---------|---------|-------|
| Wake word / push-to-talk | ✅ Full | ✅ Full |
| Volume control | ✅ WASAPI/pycaw | ✅ pactl / amixer |
| Mute toggle | ✅ keyboard | ✅ pactl |
| Media keys (play/pause/next/prev) | ✅ keyboard | ⚠️ requires `playerctl` |
| Open/close apps | ✅ Full (PATH + registry) | ⚠️ URI schemes via xdg-open; plain apps via PATH |
| TTS (Piper) | ✅ `.exe` binary | ✅ native binary |
| Screenshot | ✅ | ✅ |

**Linux note for media keys:** install `playerctl` (`sudo apt install playerctl`) for reliable media control. The `keyboard` module requires root privileges on Linux and may not work without it.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- [Piper TTS](https://github.com/rhasspy/piper/releases) binary in `piper/piper/`
- NVIDIA GPU recommended (CUDA) — CPU works but is slower
- Windows 10/11 or Linux

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/DeD-TechStack/vox-local.git
cd vox-local
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

All dependencies (including `comtypes` for Windows audio control) are listed in `requirements.txt`.

### 3. Pull an Ollama model

```bash
ollama pull qwen2.5:7b    # recommended for most systems
# or
ollama pull qwen2.5:14b   # higher quality (requires ~9 GB VRAM)
# or
ollama pull llama3.1:8b   # alternative
```

### 4. Download Piper + voice model

Download the Piper binary and place it at `piper/piper/piper.exe` (Windows) or `piper/piper/piper` (Linux).
Download a voice model and place it in `voices/`:

```
vox-local/
├── piper/
│   └── piper/
│       └── piper.exe          ← binary here
└── voices/
    ├── en_US-ryan-high.onnx
    └── en_US-ryan-high.onnx.json
```

### 5. Run

```bash
python src/main.py
```

---

## Usage

| Action | How |
|--------|-----|
| Activate (wake word mode) | Say "VOX" — the overlay turns blue and listens |
| Activate (push-to-talk mode) | Press and hold `Ctrl+Shift`, speak, release |
| Switch activation mode | Settings → tray icon → Settings |
| Switch language | Click the `AUTO`/`PT`/`EN` badge in the overlay |
| Add an app alias | Edit `app_aliases` in `config/settings.yaml` |
| Block an action | Remove it from `allowed_actions` in `config/settings.yaml` |
| Move the overlay | Click and drag |
| Open Settings | Right-click the system tray icon → Settings |
| Open Audio Settings | Right-click the system tray icon → Audio Settings |

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

### Language badge

The badge in the lower-right corner shows the **configured** language mode (`AUTO`, `PT`, or `EN`). After each command, it briefly flashes the **detected** transcription language, then restores to the configured mode. Clicking it cycles through the modes and saves the selection immediately.

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
faster-whisper — Speech-to-Text (wake word detection)
    ↓
Listener — wake word / push-to-talk activation
    ↓
faster-whisper — Speech-to-Text (command transcription)
    ↓
Brain (Ollama API) — LLM reasoning (streamed)
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
| `activation_mode` | `wake_word` | How VOX is triggered: `wake_word` or `push_to_talk` |
| `wake_word` | `vox` | Phrase that activates wake-word mode |
| `push_to_talk_key` | `ctrl+shift` | Key combo for push-to-talk mode |
| `language` | `en` | Transcription language (`auto`, `pt`, `en`) |
| `whisper_model` | `base` | Whisper model size |
| `whisper_device` | `cpu` | Device for Whisper (`cpu` or `cuda`) |
| `whisper_compute_type` | `int8` | Compute type for Whisper |
| `ollama_model` | `qwen2.5:14b` | Ollama model to use |
| `ollama_url` | `http://localhost:11434` | Ollama API base URL |
| `tts_enabled` | `true` | Enable/disable voice responses |
| `piper_path` | `piper/piper/piper.exe` | Path to Piper binary (relative to project root) |
| `voice_model` | `voices/en_US-ryan-high.onnx` | Piper voice (.onnx path, relative to project root) |
| `mic_device` | `null` | Microphone device index (`null` = system default) |
| `output_device` | `null` | Speaker device index (`null` = system default) |
| `max_history` | `20` | Max conversation turns kept in memory |
| `chunk_duration` | `2.0` | Seconds per wake-word detection chunk |
| `silence_threshold` | `0.01` | RMS below which audio is considered silent |
| `silence_duration` | `1.5` | Seconds of silence that ends a command |
| `max_record_duration` | `30` | Max seconds to record a single command |
| `app_aliases` | see file | Map spoken names to executables or URI schemes |
| `allowed_actions` | see file | Allowlist of executable actions |
| `search_dirs` | `~/Documents`, `~/Downloads`, `~/Desktop` | Directories searched by `search_file` |

### Settings that apply immediately (no restart required)

- `language` — picked up on the next recognition cycle
- `wake_word` — picked up on the next detection chunk
- `tts_enabled`, `voice_model`, `piper_path` — picked up on the next TTS call
- `ollama_model` — picked up on the next LLM request

### Settings that require restarting VOX

- `activation_mode` — determines which listening loop runs
- `whisper_model`, `whisper_device`, `whisper_compute_type` — Whisper is loaded once at startup

### Audio device settings

Changing the **output device** via Audio Settings takes effect immediately for the next TTS call. Changing the **input (microphone) device** automatically restarts the listener thread — a brief notice appears in the overlay footer.

---

## Troubleshooting

**Ollama not running**
```
Ollama is not reachable. Start it with: ollama serve
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
- Or use **Audio Settings** from the tray icon to select a microphone

**VOX says "vox open spotify" instead of "open spotify"**
This is handled automatically — the listener strips any leading wake-word echo from the transcription. If it happens unexpectedly, lower `silence_threshold` so the command phase starts capturing sooner.

**Media keys not working on Linux**
Install `playerctl`:
```bash
sudo apt install playerctl   # Debian/Ubuntu
sudo pacman -S playerctl      # Arch
```

**search_file doesn't find my files**
Check that `search_dirs` in `config/settings.yaml` lists the correct directories. Paths with `~` are expanded automatically (e.g. `~/Documents` → `/home/user/Documents`).

---

## Known Limitations

- **Push-to-talk on Linux** requires the `keyboard` package to have root privileges (or `uinput` access). Wake word mode is recommended on Linux.
- **open_app on Linux** for plain executable names uses the PATH directly. Apps not in PATH must be aliased with their full path in `app_aliases`.
- **TTS interruption** is not implemented within a single response — Piper generates the full audio before playback begins.
- **Barge-in** (speaking while VOX is responding) cancels the LLM generation but the in-flight TTS audio plays to completion.

---

## Roadmap

- [x] Wake word support (always-on)
- [x] Push-to-talk activation mode
- [x] Settings GUI
- [x] Cancel/interrupt running LLM request
- [x] Same-utterance wake-word + command capture
- [x] Listener restart on microphone change
- [x] Truthful Linux platform behavior
- [ ] TTS barge-in / cancellation
- [ ] Custom action plugins
- [ ] Conversation memory persistence
- [ ] Linux audio (PipeWire) full integration

---

## License

MIT — use it, fork it, make it yours.
