#!/bin/bash
set -e

PROFILE="lite"
if [[ -f .install-profile ]]; then
    PROFILE="$(tr -d '[:space:]' < .install-profile)"
fi

if [[ "$PROFILE" == "full" ]]; then
    uv sync --extra whisper
else
    uv sync
fi

uv run python main.py "$@"
