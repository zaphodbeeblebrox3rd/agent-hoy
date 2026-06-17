# Python Version Compatibility Guide

## Supported versions

- **Python 3.11** (recommended, pinned in `.python-version`)
- **Python 3.12** (supported via `requires-python = ">=3.11,<3.13"`)
- **Python 3.13+** not supported yet (PyAudio/torch wheel gaps)

## Setup with uv

```bash
uv python install 3.11
uv sync
uv run python test_setup.py
```

uv installs and pins Python 3.11 automatically from `.python-version`.

## If you have Python 3.13

Use uv to install 3.11:

```bash
uv python install 3.11
uv sync
```

Do not use system Python 3.13 directly.

## Alternative without uv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

```bash
uv run python test_setup.py
```
