#!/usr/bin/env python3
"""
Test script to verify the speech transcription application setup
"""

import sys
import importlib
import os

def test_imports():
    """Test if all required modules can be imported"""
    required_modules = [
        'tkinter',
        'threading',
        'queue',
        're',
        'json',
        'time',
        'typing'
    ]
    
    optional_modules = [
        'speech_recognition',
        'pyaudio',
        'requests'
    ]
    
    print("Testing required modules...")
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            return False
    
    print("\nTesting optional modules...")
    for module in optional_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}")
        except ImportError as e:
            print(f"✗ {module}: {e}")
            print(f"  Install with: pip install {module}")
    
    return True

def test_microphone():
    """Test microphone access"""
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        m = sr.Microphone()
        
        print("\nTesting microphone access...")
        with m as source:
            r.adjust_for_ambient_noise(source, duration=1)
        print("✓ Microphone access successful")
        return True
    except Exception as e:
        print(f"✗ Microphone access failed: {e}")
        return False

def test_gui():
    """Test GUI components"""
    try:
        import tkinter as tk
        from tkinter import ttk, scrolledtext
        
        print("\nTesting GUI components...")
        root = tk.Tk()
        root.withdraw()  # Hide the window
        
        # Test basic widgets
        frame = ttk.Frame(root)
        text = scrolledtext.ScrolledText(frame)
        button = ttk.Button(frame, text="Test")
        
        root.destroy()
        print("✓ GUI components working")
        return True
    except Exception as e:
        print(f"✗ GUI test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Speech Transcription Application - Setup Test")
    print("=" * 50)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 9):
        print("✗ Error: Python 3.9+ required")
        return False
    elif sys.version_info >= (3, 13):
        print("⚠ Warning: Python 3.13 may have compatibility issues")
        print("  Recommended: Python 3.9-3.12")
    else:
        print("✓ Python version is recommended")
    
    # Check if we're in a conda environment
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print(f"✓ Running in conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
    else:
        print("⚠ Not in a conda environment - consider using: conda activate agent-hoy")
    
    # Run tests
    imports_ok = test_imports()
    mic_ok = test_microphone()
    gui_ok = test_gui()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Imports: {'✓' if imports_ok else '✗'}")
    print(f"Microphone: {'✓' if mic_ok else '✗'}")
    print(f"GUI: {'✓' if gui_ok else '✗'}")
    
    if imports_ok and mic_ok and gui_ok:
        print("\n🎉 All tests passed! You can run the application with: python main.py")
    else:
        print("\n⚠ Some tests failed. Please install missing dependencies and try again.")
        print("Install command: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
