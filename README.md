# Real-time Speech Transcription with Topic Explorer

A Python application that provides real-time speech transcription with interactive keyword-based topic explanations. Click on highlighted technical terms to get instant summaries, technical challenges, and useful commands.

## Features

- **Real-time Speech Recognition**: Continuous microphone listening with live transcription
- **Interactive Keywords**: Click on highlighted technical terms to get detailed explanations
- **Topic Explanations**: Get summaries, technical challenges, and command examples for various tech topics
- **Modern GUI**: Clean, responsive interface built with tkinter
- **AI-Enhanced Analysis**: OpenAI integration with Responses API, streaming, and adaptive question types
- **Two install profiles**: Lite (small, online STT) and Full (offline Whisper)

## Installation

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager
  - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Python 3.11 (installed automatically by uv via `.python-version`)

### Install profiles

| Profile | Command | Size | Speech recognition |
|---------|---------|------|-------------------|
| Lite (default) | `uv sync` | ~50MB | Google Speech (needs FLAC + network) |
| Full (offline) | `uv sync --extra whisper` | ~1-2GB | Whisper primary, Google fallback |

### Quick start

**Automated setup (recommended):**

```bash
git clone <repository-url>
cd agent-hoy

# Lite profile (default)
./setup_uv.sh          # Linux/macOS
setup_uv.bat           # Windows

# Full profile (offline Whisper)
./setup_uv.sh --full
setup_uv.bat --full
```

**Manual setup:**

```bash
uv python install 3.11
uv sync              # Lite
# or
uv sync --extra whisper   # Full

uv run python test_setup.py
uv run python main.py
```

### System dependencies

| OS | Lite | Full (additional) |
|----|------|-------------------|
| Linux | `portaudio19-dev`, `python3-tk`, `flac` | `ffmpeg` |
| macOS | `portaudio`, `flac` | `ffmpeg` |
| Windows | FLAC via `choco install flac` | `choco install ffmpeg` |

```bash
# Ubuntu/Debian example
sudo apt-get install portaudio19-dev python3-tk flac ffmpeg

# macOS example
brew install portaudio flac ffmpeg
```

### GPU-accelerated Whisper

`setup_uv.sh --full` and `setup_uv.bat --full` auto-detect NVIDIA GPUs and upgrade PyTorch to CUDA after install. Manual upgrade:

```bash
uv sync --extra whisper
uv pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

When CUDA is available, the app prefers Whisper for low-latency local transcription. Without a GPU, it prefers Google STT while online and falls back to CPU Whisper offline.

## Usage

```bash
uv run python main.py
```

Or use the profile-aware launcher (re-syncs the correct extras first):

```bash
run.bat          # Windows
./run.sh         # Linux/macOS
```

Set `OPENAI_API_KEY` for AI analysis. See [OPENAI_SETUP.md](OPENAI_SETUP.md) for model and API options.

Important: `uv sync` without `--extra whisper` installs the Lite profile and removes Whisper. After `setup_uv.bat --full`, use `run.bat` or always sync with `--extra whisper`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Model for analysis |
| `USE_RESPONSES_API` | `true` | Use Responses API (`false` for Chat Completions rollback) |
| `OPENAI_STREAM` | `true` | Stream AI responses to UI |
| `STT_PREFER_GOOGLE` | `false` | Force Google STT even when CUDA is available |
| `STT_PREFER_WHISPER` | `false` | Force Whisper even without CUDA |
| `WHISPER_MODEL` | auto | `base` on CPU, `small` on GPU |
| `WHISPER_FAST_MODE` | `true` | Greedy decoding, shorter prompts |
| `WHISPER_BEAM_SIZE` | `1` | Higher = slower but more accurate |
| `QUESTION_COMPLETION_TIMEOUT` | `1.5` | Seconds after speech before AI runs |
| `AI_ANALYSIS_THROTTLE_SECONDS` | `2` | Min gap between OpenAI calls |
| `LISTEN_PHRASE_TIME_LIMIT` | `3` | Max seconds per mic chunk |
| `VERBOSE` | `false` | Enable debug logging |
| `RECOGNITION_MAX_WORKERS` | `2` | Max parallel STT workers |
| `NETWORK_CHECK_TTL_SECONDS` | `30` | Network check cache TTL |

## Troubleshooting

**Recreate environment (rollback to Lite):**

```bash
rm -rf .venv    # Windows: rmdir /s /q .venv
uv sync
```

**Upgrade to Full profile:**

```bash
uv sync --extra whisper
```

**PyAudio issues:** Install system portaudio libraries first, then `uv sync --reinstall-package pyaudio`.

**Microphone:** Check OS permissions; ensure no other app holds the mic.

Run diagnostics: `uv run python test_setup.py`

## Cross-platform smoke test checklist

After setup on each platform:

- [ ] `uv run python test_setup.py` passes
- [ ] `uv run python test_openai.py` passes (or template fallback without API key)
- [ ] `uv run python main.py` starts GUI
- [ ] Microphone capture works
- [ ] Lite: Google STT transcribes (FLAC + network required)
- [ ] Full: Whisper transcribes offline
- [ ] AI analysis updates when keywords are spoken (with API key)

## License

MIT License - see [LICENSE](LICENSE).
