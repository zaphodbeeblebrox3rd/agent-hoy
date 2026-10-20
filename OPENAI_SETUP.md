# OpenAI Integration Setup Guide

## Prerequisites

- OpenAI API account with credits
- OpenAI API key
- Internet connection for API calls

## Setup

### Environment variable (recommended)

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Linux/macOS:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Configuration file

Create `openai_key.txt` in the project root (gitignored).

## Default model and API

- Model: `gpt-4.1-mini` (change in-app or via `conf/openai_model.conf`)
- API: Responses API (rollback with `USE_RESPONSES_API=false`)
- Streaming: enabled by default (`OPENAI_STREAM=true`)

### In-app model switcher

Use the Model combobox in the control panel to switch between GPT-4.x, GPT-5.x, GPT-5.6, and GPT-6 models, including Pro and Astra premium tiers. The selection is saved to `conf/openai_model.conf` and restored on the next launch.

The combobox is disabled when OpenAI is not configured (template fallback mode).

Rollback: set `INCLUDE_LATEST_PREMIUM_MODELS = False` in `openai_config.py` to hide GPT-5.6, GPT-6, and Pro tiers from the combobox. You can still set those model ids via `OPENAI_MODEL` or `conf/openai_model.conf`.

### Model selection precedence

1. `OPENAI_MODEL` environment variable (startup override)
2. `conf/openai_model.conf` (persisted from UI)
3. Default: `gpt-4.1-mini`

Rollback: delete `conf/openai_model.conf` or set `OPENAI_MODEL`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | API key |
| `OPENAI_MODEL` | (from conf or `gpt-4.1-mini`) | Model name |
| `USE_RESPONSES_API` | `true` | Use Responses API |
| `OPENAI_STREAM` | `true` | Stream tokens to UI |
| `OPENAI_MAX_TOKENS` | `1000` | Max output tokens |
| `OPENAI_TEMPERATURE` | `0.7` | Sampling temperature (GPT-4.x only) |
| `OPENAI_REASONING_EFFORT` | `low` | GPT-5 / GPT-6 reasoning effort |
| `OPENAI_TEXT_VERBOSITY` | `medium` | GPT-5 / GPT-6 output verbosity |
| `OPENAI_CACHE_RESPONSES` | `true` | File cache in `cache/` |
| `OPENAI_CACHE_HOURS` | `24` | Cache TTL |

### GPT-5 and GPT-6 models

GPT-5, GPT-5.6, and GPT-6 family models use the Responses API with `reasoning.effort` and `text.verbosity` instead of `temperature`. Keep `USE_RESPONSES_API=true` when using these models.

Latest and premium options in the combobox include `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gpt-5.6-sol`, `gpt-5.5-pro`, and `gpt-5.4-pro`.

Optional reasoning tuning:

```bash
export OPENAI_REASONING_EFFORT=low
export OPENAI_TEXT_VERBOSITY=medium
```

### Rollback to Chat Completions

```bash
export USE_RESPONSES_API=false
export OPENAI_MODEL=gpt-4o-mini
uv run python main.py
```

## Cost reference (per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4.1-mini | $0.40 | $1.60 |
| gpt-4.1 | $2.00 | $8.00 |
| gpt-5-mini | $0.25 | $2.00 |
| gpt-5.4-mini | $0.75 | $4.50 |
| gpt-5.4 | $2.50 | $15.00 |
| gpt-5.5 | $5.00 | $30.00 |
| gpt-5.5-pro | $30.00 | $180.00 |
| gpt-5.6-luna | $0.20 | $1.20 |
| gpt-5.6-sol | $4.00 | $20.00 |
| gpt-6-luna | $0.10 | $0.50 |
| gpt-6.1-sol | $2.00 | $10.00 |
| gpt-6-astra | $10.00 | $50.00 |

## Testing

```bash
uv run python test_openai.py
```

Look for the active model id in the OpenAI status label in the application window.

## Security

- Never commit API keys
- Use environment variables in production
- Responses use `store=false` (no server-side retention)
