# Installation Guide

## Quick start with uv

### Lite profile (default)

```bash
./setup_uv.sh          # Linux/macOS
setup_uv.bat           # Windows
```

Or manually:

```bash
uv python install 3.11
uv sync
uv run python test_setup.py
```

### Full profile (offline Whisper)

```bash
./setup_uv.sh --full
setup_uv.bat --full
```

Or:

```bash
uv sync --extra whisper
```

## System dependencies

Install these before `uv sync` if PyAudio or FLAC fail:

**Ubuntu/Debian:**

```bash
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-tk flac ffmpeg build-essential
```

**macOS:**

```bash
brew install portaudio flac ffmpeg
```

**Windows:**

```powershell
choco install flac ffmpeg
```

## Alternative: pip-only install

If uv is unavailable, use the exported requirements file:

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For Whisper, install torch CPU wheels separately from https://pytorch.org then `pip install openai-whisper`.

## Troubleshooting PyAudio

1. Install system portaudio libraries (see above)
2. Reinstall: `uv sync --reinstall-package pyaudio`
3. Windows fallback: `pip install pipwin` then `pipwin install pyaudio`

## Docker example

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y portaudio19-dev flac ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
CMD ["uv", "run", "python", "main.py"]
```

## Testing

```bash
uv run python test_setup.py
uv run python -c "import pyaudio; print('PyAudio OK')"
uv run python -c "import speech_recognition as sr; print('SR OK')"
```

## Rollback

Switch from Full to Lite:

```bash
rm -rf .venv
uv sync
```
