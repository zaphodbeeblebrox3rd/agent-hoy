# Offline Setup Guide

Use the Full install profile for offline speech recognition with Whisper.

## Install Full profile

```bash
./setup_uv.sh --full
# or
uv sync --extra whisper
```

## Offline capabilities

1. Whisper STT (Full profile): local transcription without internet
2. Google fallback: when online and FLAC is available
3. Topic cache: `cache/topic_cache.pkl` (24h topic / 1h troubleshooting TTL)
4. OpenAI file cache: `cache/openai_*.json` when API key is set

## Prefer Whisper while online

The app sets offline behavior automatically when network checks fail. To prefer local Whisper even when online:

```bash
export STT_PREFER_WHISPER=true
uv run python main.py
```

Rollback: unset `STT_PREFER_WHISPER` or set `STT_PREFER_GOOGLE=true`.

## Optional offline STT packages

```bash
uv sync --extra offline-stt
```

Installs vosk and pocketsphinx (not wired in `main.py` today; reserved for future use).

## Cache maintenance

Delete the `cache/` directory to clear all cached data.

## Requirements by profile

| Feature | Lite | Full |
|---------|------|------|
| Google Speech | Yes (network + FLAC) | Fallback only |
| Whisper | No | Yes |
| AI analysis | Optional (API key) | Optional (API key) |
| Cached topics | Yes | Yes |
