@echo off
setlocal EnableExtensions

REM Run agent-hoy with the correct uv extras for the saved install profile.

set "PROFILE=lite"
if exist .install-profile (
    set /p PROFILE=<.install-profile
)

if /I "%PROFILE%"=="full" (
    uv sync --extra whisper
) else (
    uv sync
)

uv run python main.py %*
