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

- Model: `gpt-4.1-mini` (override with `OPENAI_MODEL`)
- API: Responses API (rollback with `USE_RESPONSES_API=false`)
- Streaming: enabled by default (`OPENAI_STREAM=true`)

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (none) | API key |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Model name |
| `USE_RESPONSES_API` | `true` | Use Responses API |
| `OPENAI_STREAM` | `true` | Stream tokens to UI |
| `OPENAI_MAX_TOKENS` | `1000` | Max output tokens |
| `OPENAI_TEMPERATURE` | `0.7` | Sampling temperature |
| `OPENAI_CACHE_RESPONSES` | `true` | File cache in `cache/` |
| `OPENAI_CACHE_HOURS` | `24` | Cache TTL |

### Rollback to Chat Completions

```bash
export USE_RESPONSES_API=false
export OPENAI_MODEL=gpt-4o-mini
uv run python main.py
```

## Cost reference (per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| gpt-4.1-mini | $0.40 | $1.60 |
| gpt-4o-mini | $0.15 | $0.60 |
| gpt-4.1 | $2.00 | $8.00 |

## Testing

```bash
uv run python test_openai.py
```

Look for "OpenAI: Configured" in the application status bar.

## Security

- Never commit API keys
- Use environment variables in production
- Responses use `store=false` (no server-side retention)
