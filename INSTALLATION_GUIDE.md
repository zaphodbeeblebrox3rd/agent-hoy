# Installation Guide

This guide helps you install the application and resolve common dependency issues.

## Quick Start

### 1. Install System Dependencies First

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install build-essential
```

**CentOS/RHEL/Fedora:**
```bash
sudo yum install portaudio-devel
sudo yum groupinstall "Development Tools"
# or for newer versions:
sudo dnf install portaudio-devel
sudo dnf groupinstall "Development Tools"
```

**macOS:**
```bash
brew install portaudio
```

**Windows:**
- PyAudio should install automatically
- If issues occur, try: `pip install pipwin` then `pipwin install pyaudio`

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## Troubleshooting PyAudio Installation

### Error: "portaudio.h: No such file or directory"

This means the system audio libraries aren't installed. Follow the system dependency installation above.

### Alternative PyAudio Installation Methods

**Method 1: Use conda (Recommended)**
```bash
conda install pyaudio
```

**Method 2: Use pre-compiled wheels**
```bash
pip install --only-binary=all PyAudio
```

**Method 3: Install from source with system libraries**
```bash
# Install system dependencies first (see above)
pip install PyAudio
```

### Platform-Specific Solutions

#### Ubuntu/Debian
```bash
# Install all required system packages
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio python3-dev
sudo apt-get install build-essential
sudo apt-get install libasound2-dev

# Then install Python packages
pip install -r requirements.txt
```

#### CentOS/RHEL
```bash
# Install development tools and audio libraries
sudo yum groupinstall "Development Tools"
sudo yum install portaudio-devel
sudo yum install alsa-lib-devel

# Then install Python packages
pip install -r requirements.txt
```

#### macOS
```bash
# Install portaudio via Homebrew
brew install portaudio

# Then install Python packages
pip install -r requirements.txt
```

#### Windows
```bash
# PyAudio should work automatically
pip install -r requirements.txt

# If issues occur, try:
pip install pipwin
pipwin install pyaudio
```

## Alternative Installation Methods

### Using Conda (Easiest)
```bash
# Create conda environment
conda create -n agent-hoy python=3.11
conda activate agent-hoy

# Install packages via conda (handles system dependencies)
conda install pyaudio
conda install -c conda-forge speechrecognition
conda install -c conda-forge vosk
conda install -c conda-forge pocketsphinx

# Install remaining packages via pip
pip install requests
```

### Using Docker (Most Reliable)
```bash
# Create Dockerfile
cat > Dockerfile << EOF
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . .

# Run application
CMD ["python", "main.py"]
EOF

# Build and run
docker build -t agent-hoy .
docker run -it --rm -v /dev/snd:/dev/snd agent-hoy
```

## Testing Installation

### Test PyAudio
```python
import pyaudio
print("PyAudio installed successfully!")
```

### Test Speech Recognition
```python
import speech_recognition as sr
r = sr.Recognizer()
print("SpeechRecognition installed successfully!")
```

### Test Offline Recognition
```python
import vosk
print("Vosk installed successfully!")
```

## Common Issues and Solutions

### Issue 1: "No module named 'pyaudio'"
**Solution**: Install system dependencies first, then reinstall PyAudio

### Issue 2: "Permission denied" errors
**Solution**: Use `sudo` for system package installation, but not for pip

### Issue 3: "Compiler not found"
**Solution**: Install build tools:
- Ubuntu: `sudo apt-get install build-essential`
- CentOS: `sudo yum groupinstall "Development Tools"`

### Issue 4: "PortAudio not found"
**Solution**: Install portaudio development libraries:
- Ubuntu: `sudo apt-get install portaudio19-dev`
- CentOS: `sudo yum install portaudio-devel`

## Minimal Installation (No Audio)

If you can't get PyAudio working, you can still use the application for:
- Keyword detection and explanations
- Troubleshooting suggestions
- Cached content

Just comment out the audio-related imports in main.py:
```python
# import speech_recognition as sr
# import pyaudio
```

## Getting Help

If you're still having issues:

1. **Check your Python version**: `python --version` (should be 3.9-3.12)
2. **Check your system**: `uname -a` (Linux/macOS/Windows)
3. **Check installed packages**: `pip list`
4. **Try conda instead**: Often handles system dependencies better

## Platform-Specific Notes

### Linux (WSL)
- May need additional audio setup
- Consider using conda for easier dependency management

### macOS
- Homebrew is the easiest way to install system dependencies
- May need to allow microphone permissions

### Windows
- PyAudio usually installs without issues
- May need to install Visual Studio Build Tools if compilation fails

The key is installing the system audio libraries first, then the Python packages!
