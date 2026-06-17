#!/bin/bash
# UV setup script for agent-hoy (Lite and Full profiles)

set -e

PROFILE="lite"
if [[ "${1:-}" == "--full" ]]; then
    PROFILE="full"
fi

echo "Setting up agent-hoy with uv (${PROFILE} profile)..."
echo "===================================================="

if ! command -v uv &> /dev/null; then
    echo "uv not found. Install from: https://github.com/astral-sh/uv"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv found: $(uv --version)"

if uv python find 3.11 >/dev/null 2>&1; then
    echo "Python 3.11 is already available"
else
    echo "Installing Python 3.11..."
    uv python install 3.11
fi

if [[ "$PROFILE" == "full" ]]; then
    echo "Syncing Full profile - Whisper with CPU PyTorch..."
    uv sync --extra whisper
    echo "full" > .install-profile
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected - upgrading to CUDA PyTorch..."
        uv pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
    fi
    echo "Verifying Whisper install..."
    uv run python -c "import whisper, torch; print('Whisper OK, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
else
    echo "NOTE: Lite sync removes Whisper if it was previously installed."
    echo "Syncing Lite profile (core dependencies only)..."
    uv sync
    echo "lite" > .install-profile
fi

echo "Running setup tests..."
uv run python test_setup.py

echo ""
echo "Setup complete (${PROFILE} profile)!"
echo ""
echo "To use the application:"
echo "  uv run python main.py"
echo ""
echo "Install profiles:"
echo "  Lite (default): ./setup_uv.sh"
echo "  Full (offline Whisper): ./setup_uv.sh --full"
echo ""
echo "To remove the environment:"
echo "  rm -rf .venv"
