#!/usr/bin/env python3
"""
Installation script for the Speech Transcription Application
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def install_requirements():
    """Install Python requirements using conda"""
    # Check if conda is available
    try:
        subprocess.run(["conda", "--version"], check=True, capture_output=True)
        print("✓ Conda found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Conda not found. Please install Miniconda or Anaconda first.")
        print("  Download from: https://docs.conda.io/en/latest/miniconda.html")
        return False
    
    # Create environment from yaml file
    if os.path.exists("environment.yaml"):
        return run_command(
            "conda env create -f environment.yaml",
            "Creating conda environment from environment.yaml"
        )
    else:
        print("✗ environment.yaml not found")
        return False

def install_system_dependencies():
    """Install system-specific dependencies"""
    import platform
    
    system = platform.system().lower()
    
    if system == "linux":
        print("Installing Linux audio dependencies...")
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-pyaudio portaudio19-dev",
            "sudo apt-get install -y python3-tk"
        ]
    elif system == "darwin":  # macOS
        print("Installing macOS dependencies...")
        commands = [
            "brew install portaudio",
            "brew install python-tk"
        ]
    else:
        print("Windows detected - PyAudio should install automatically")
        return True
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            print(f"⚠ Warning: Command failed: {cmd}")
            print("You may need to install dependencies manually")
    
    return True

def main():
    """Main installation process"""
    print("Speech Transcription Application - Installation")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("✗ Python 3.9+ required. Current version:", sys.version)
        return False
    
    print(f"✓ Python version: {sys.version}")
    
    # Install Python requirements
    if not install_requirements():
        print("✗ Failed to install Python requirements")
        return False
    
    # Install system dependencies
    if not install_system_dependencies():
        print("✗ Failed to install system dependencies")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Activate the conda environment: conda activate agent-hoy")
    print("2. Run the test script: python test_setup.py")
    print("3. Start the application: python main.py")
    print("\nIf you encounter issues:")
    print("- Check microphone permissions")
    print("- Ensure internet connection (for speech recognition)")
    print("- Run test_setup.py to diagnose problems")
    print("- Update conda: conda update conda")

if __name__ == "__main__":
    main()
