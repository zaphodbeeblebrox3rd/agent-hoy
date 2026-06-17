#!/usr/bin/env python3
"""
Test script to verify the speech transcription application setup.
"""

import importlib
import os
import sys


def detect_install_profile():
    """Detect Lite vs Full install profile."""
    try:
        importlib.import_module("whisper")
        return "full"
    except ImportError:
        return "lite"


def test_imports():
    """Test if all required modules can be imported."""
    required_modules = [
        "tkinter",
        "threading",
        "queue",
        "re",
        "json",
        "time",
        "typing",
        "numpy",
    ]

    optional_modules = [
        "speech_recognition",
        "pyaudio",
        "requests",
        "openai",
    ]

    print("Testing required modules...")
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"OK {module}")
        except ImportError as e:
            print(f"FAIL {module}: {e}")
            return False

    print("\nTesting optional modules...")
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"OK {module}")
        except ImportError as e:
            print(f"WARN {module}: {e}")
            print(f"  Install with: uv sync")

    profile = detect_install_profile()
    if profile == "full":
        try:
            importlib.import_module("whisper")
            importlib.import_module("torch")
            print("OK whisper (Full profile)")
            print("OK torch (Full profile)")
        except ImportError as e:
            print(f"WARN whisper/torch: {e}")
    else:
        print("INFO Lite profile - Whisper not installed (use setup_uv.sh --full)")

    return True


def test_microphone():
    """Test microphone access."""
    try:
        import speech_recognition as sr

        print("\nTesting microphone access...")
        r = sr.Recognizer()
        m = sr.Microphone()
        with m as source:
            r.adjust_for_ambient_noise(source, duration=1)
        print("OK Microphone access successful")
        return True
    except Exception as e:
        print(f"FAIL Microphone access failed: {e}")
        return False


def test_gui():
    """Test GUI components."""
    try:
        import tkinter as tk
        from tkinter import scrolledtext, ttk

        print("\nTesting GUI components...")
        root = tk.Tk()
        root.withdraw()

        frame = ttk.Frame(root)
        scrolledtext.ScrolledText(frame)
        ttk.Button(frame, text="Test")

        root.destroy()
        print("OK GUI components working")
        return True
    except Exception as e:
        print(f"FAIL GUI test failed: {e}")
        return False


def test_environment():
    """Report virtual environment and install profile."""
    profile = detect_install_profile()
    print(f"\nInstall profile: {profile.upper()}")

    if os.environ.get("VIRTUAL_ENV"):
        print(f"OK Virtual environment: {os.environ['VIRTUAL_ENV']}")
    elif os.path.isdir(".venv"):
        print("OK Project .venv directory present")
    else:
        print("WARN No virtual environment detected - run: uv sync")

    return True


def main():
    """Run all tests."""
    print("Speech Transcription Application - Setup Test")
    print("=" * 50)

    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print("FAIL Python 3.9+ required")
        return False
    if sys.version_info >= (3, 13):
        print("WARN Python 3.13 may have compatibility issues")
        print("  Recommended: Python 3.11 (see .python-version)")
    else:
        print("OK Python version is supported")

    test_environment()
    imports_ok = test_imports()
    mic_ok = test_microphone()
    gui_ok = test_gui()

    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Imports: {'OK' if imports_ok else 'FAIL'}")
    print(f"Microphone: {'OK' if mic_ok else 'FAIL'}")
    print(f"GUI: {'OK' if gui_ok else 'FAIL'}")
    print(f"Profile: {detect_install_profile().upper()}")

    if imports_ok and mic_ok and gui_ok:
        print("\nAll tests passed! Run: uv run python main.py")
        return True

    print("\nSome tests failed. Check dependencies and microphone permissions.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
