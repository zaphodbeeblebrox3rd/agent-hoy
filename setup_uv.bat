@echo off
setlocal EnableExtensions

REM UV setup script for agent-hoy - Lite and Full profiles

set "PROFILE=lite"
if /I "%~1"=="--full" set "PROFILE=full"

echo Setting up agent-hoy with uv - %PROFILE% profile
echo ====================================================

where uv >nul 2>&1
if errorlevel 1 (
    echo uv not found. Install from: https://github.com/astral-sh/uv
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
)

for /f "delims=" %%v in ('uv --version 2^>nul') do echo Found %%v

uv python find 3.11 >nul 2>&1
if errorlevel 1 (
    echo Installing Python 3.11...
    uv python install 3.11
    if errorlevel 1 exit /b 1
) else (
    echo Python 3.11 is already available
)

if /I "%PROFILE%"=="full" goto sync_full
echo Syncing Lite profile - core dependencies only...
echo NOTE: Lite sync removes Whisper if it was previously installed.
uv sync
if errorlevel 1 exit /b 1
echo lite> .install-profile
goto run_tests

:sync_full
echo Syncing Full profile - Whisper with CPU PyTorch...
uv sync --extra whisper
if errorlevel 1 exit /b 1
echo full> .install-profile
where nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo NVIDIA GPU detected - upgrading to CUDA PyTorch...
    uv pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    if errorlevel 1 exit /b 1
)
echo Verifying Whisper install...
uv run python -c "import whisper, torch; print('Whisper OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
if errorlevel 1 (
    echo Whisper verification failed. Try: uv sync --extra whisper
    exit /b 1
)

:run_tests
echo Running setup tests...
uv run python test_setup.py
if errorlevel 1 exit /b 1

echo.
echo Setup complete - %PROFILE% profile
echo.
echo To use the application:
echo   uv run python main.py
echo.
echo Install profiles:
echo   Lite - default:  setup_uv.bat
echo   Full - Whisper:  setup_uv.bat --full
echo.
echo To remove the environment:
echo   rmdir /s /q .venv

endlocal
exit /b 0
