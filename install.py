#!/usr/bin/env python3
"""
Installation script for the Speech Transcription Application (uv-based).
"""

import argparse
import os
import platform
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout.strip())
        print(f"OK {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED {description}: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_uv():
    """Verify uv is available."""
    try:
        result = subprocess.run(
            ["uv", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"OK uv found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("uv not found. Install from: https://github.com/astral-sh/uv")
        return False


def ensure_python_311():
    """Install Python 3.11 only if uv does not already have it."""
    try:
        subprocess.run(
            ["uv", "python", "find", "3.11"],
            check=True,
            capture_output=True,
            text=True,
        )
        print("OK Python 3.11 is already available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return run_command("uv python install 3.11", "Installing Python 3.11")


def _has_nvidia_gpu():
    """Return True when nvidia-smi is available."""
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_python_dependencies(full_profile=False):
    """Install Python dependencies using uv."""
    if not os.path.exists("pyproject.toml"):
        print("pyproject.toml not found")
        return False

    if not ensure_python_311():
        return False

    if full_profile:
        if not run_command("uv sync --extra whisper", "Syncing Full profile with Whisper"):
            return False
        if _has_nvidia_gpu():
            return run_command(
                "uv pip install --force-reinstall torch torchaudio "
                "--index-url https://download.pytorch.org/whl/cu128",
                "Upgrading to CUDA PyTorch",
            )
        return True

    return run_command("uv sync", "Syncing Lite profile (core dependencies)")


def install_system_dependencies():
    """Install system-specific dependencies."""
    system = platform.system().lower()

    if system == "linux":
        print("Installing Linux audio dependencies...")
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-tk portaudio19-dev flac ffmpeg",
        ]
    elif system == "darwin":
        print("Installing macOS dependencies...")
        commands = [
            "brew install portaudio flac ffmpeg",
        ]
    else:
        print("Windows detected - install FLAC via: choco install flac")
        print("Install ffmpeg via: choco install ffmpeg")
        return True

    for cmd in commands:
        if not run_command(cmd, cmd):
            print(f"Warning: Command failed: {cmd}")
            print("You may need to install dependencies manually")

    return True


def main():
    """Main installation process."""
    parser = argparse.ArgumentParser(description="Install agent-hoy with uv")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Install Full profile with Whisper offline STT (large download)",
    )
    parser.add_argument(
        "--skip-system-deps",
        action="store_true",
        help="Skip platform system dependency installation",
    )
    args = parser.parse_args()

    profile = "Full" if args.full else "Lite"
    print("Speech Transcription Application - Installation")
    print("=" * 50)
    print(f"Profile: {profile}")

    if sys.version_info < (3, 9):
        print(f"Python 3.9+ required. Current version: {sys.version}")
        return False

    print(f"Python version: {sys.version}")

    if not check_uv():
        return False

    if not install_python_dependencies(full_profile=args.full):
        print("Failed to install Python dependencies")
        return False

    if not args.skip_system_deps and not install_system_dependencies():
        print("Failed to install system dependencies")
        return False

    print("\n" + "=" * 50)
    print(f"Installation completed ({profile} profile)!")
    print("\nNext steps:")
    print("1. Run tests: uv run python test_setup.py")
    print("2. Start the application: uv run python main.py")
    print("\nProfiles:")
    print("  Lite:  python install.py")
    print("  Full:  python install.py --full")
    print("\nRollback to Lite after Full:")
    print("  rm -rf .venv && uv sync")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
