# Transcription Quality Status

This document tracks transcription quality work. Phases 1-3 below are implemented; remaining items are optional.

## Current issues (mitigated)

| Issue | Mitigation |
|-------|------------|
| Google Speech accuracy / latency | Full profile uses Whisper; Google remains Lite default / fallback |
| FLAC dependency | Required only for Google STT; Whisper path does not need FLAC |
| Offline capability | `uv sync --extra whisper` or `./setup_uv.sh --full` |

## Implemented

### 1. Optimized Google Speech Recognition

- Faster ambient calibration and dynamic energy thresholds
- Language hints set to `en-US`
- Tunable via `LISTEN_TIMEOUT` and `LISTEN_PHRASE_TIME_LIMIT`

### 2. Configuration system

- `transcription_config.py`: STT preference, Whisper decode settings, phrase hints
- `conf/*.conf.example`: corrections, keywords, topics, question patterns (override with `conf/*.conf`)

### 3. Whisper integration (Phase 1)

- Optional extra: `openai-whisper`, `torch`, `torchaudio`
- Install: `./setup_uv.sh --full` or `uv sync --extra whisper`
- Auto device selection (CUDA when available); model via `WHISPER_MODEL` or auto (`base` CPU / `small` GPU)
- Preference logic: `STT_PREFER_WHISPER`, `STT_PREFER_GOOGLE`, or CUDA/online heuristics

### 4. OpenAI analysis (Phase 2)

- Responses API with streaming (rollback: `USE_RESPONSES_API=false`)
- Model switcher and `conf/openai_model.conf`
- See [OPENAI_SETUP.md](OPENAI_SETUP.md)

### 5. Hybrid recognition (Phase 3)

- Whisper preferred on CUDA or when offline
- Google used online without CUDA (or when forced)
- Transcription corrections applied after STT
- Cross-fallback when the primary path returns empty text

## Service comparison

| Service | Accuracy | Speed | Offline | Cost | Best for |
|---------|----------|-------|---------|------|----------|
| Google | Medium | Fast | No | Free | Lite profile / short online chunks |
| Whisper | High | Medium (CPU) / Fast (GPU) | Yes | Free | Technical content, offline |
| OpenAI analysis | N/A (post-STT) | Fast | No | Paid | Topic / troubleshooting explanations |

## Optional next steps

1. Wire `offline-stt` extra (vosk / pocketsphinx) into `main.py`
2. Context-aware correction ranking (beyond static `conf/corrections.conf`)
3. Azure or other cloud STT backends listed in `transcription_config.py`

## Quick start

```bash
# Lite (Google STT)
./setup_uv.sh
uv run python main.py

# Full (Whisper)
./setup_uv.sh --full
uv run python main.py

# Force local Whisper while online
STT_PREFER_WHISPER=true uv run python main.py
```

Rollback Full to Lite: `rm -rf .venv && uv sync` (or set `TORCH_CUDA_INDEX_URL` only if changing CUDA wheels; see README).
