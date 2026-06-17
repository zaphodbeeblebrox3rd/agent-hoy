# Offline Setup Guide

Use the **Full** install profile for offline speech recognition with Whisper.

## Install Full profile

```bash
./setup_uv.sh --full
# or
uv sync --extra whisper
```

## Offline capabilities

1. **Whisper STT** (Full profile): Local transcription without internet
2. **Google fallback**: When online and FLAC is available
3. **Topic cache**: `cache/topic_cache.pkl` (24h topic / 1h troubleshooting TTL)
4. **OpenAI file cache**: `cache/openai_*.json` when API key is set

## Optional offline STT packages

```bash
uv sync --extra offline-stt
```

Installs vosk and pocketsphinx (not wired in main.py today; reserved for future use).

## Force offline mode

The app sets `use_offline = True` when network checks fail. To force offline at startup, set in code or wait for automatic detection.

## Cache maintenance

Delete the `cache/` directory to clear all cached data.

## Requirements by profile

| Feature | Lite | Full |
|---------|------|------|
| Google Speech | Yes (network + FLAC) | Fallback only |
| Whisper | No | Yes |
| AI analysis | Optional (API key) | Optional (API key) |
| Cached topics | Yes | Yes |
