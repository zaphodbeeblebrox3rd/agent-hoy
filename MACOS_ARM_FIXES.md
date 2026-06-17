# macOS ARM (Apple Silicon) Compatibility

## Setup

```bash
brew install portaudio flac ffmpeg
./setup_uv.sh          # Lite
./setup_uv.sh --full   # Full with Whisper
```

PyAudio installs via pip/uv wheels on ARM64 when portaudio is present.

## Verification

```bash
uv run python test_setup.py
uv run python main.py
```

## Notes

1. Install `portaudio` via Homebrew before `uv sync`
2. macOS prompts for microphone permission on first use
3. Use pip/uv for PyAudio on ARM, not legacy conda packages

## Rollback

```bash
rm -rf .venv && uv sync
```
