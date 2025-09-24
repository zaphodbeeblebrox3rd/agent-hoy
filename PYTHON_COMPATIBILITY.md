# Python Version Compatibility Guide

## Recommended Python Versions

### ✅ **Best Compatibility**
- **Python 3.11** - Most stable, best package support
- **Python 3.10** - Excellent compatibility, widely tested
- **Python 3.9** - Very stable, good for production

### ⚠️ **Use with Caution**
- **Python 3.12** - Generally works but some packages may have issues

### ❌ **Not Recommended**
- **Python 3.13** - Too new, many packages don't have compatible wheels yet
- **Python 3.8 and below** - Not supported

## Common Issues with Python 3.13

### PyAudio Installation Problems
```bash
# Error you might see:
ERROR: Could not build wheels for pyaudio, which is required to install pyproject.toml based projects
```

**Solutions:**
1. Use Python 3.11 or 3.12 instead
2. Install pre-compiled wheels: `pip install --only-binary=all pyaudio`
3. Install system dependencies first (see main README)

### Speech Recognition Issues
```bash
# Error you might see:
ModuleNotFoundError: No module named 'speech_recognition'
```

**Solutions:**
1. Downgrade to Python 3.11: `pyenv install 3.11.7`
2. Use virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install with specific version: `pip install speech_recognition==3.10.0`

## Quick Fix for Python 3.13 Users

If you're stuck with Python 3.13, try these workarounds:

### Option 1: Use Conda
```bash
conda create -n speech-app python=3.11
conda activate speech-app
pip install -r requirements.txt
```

### Option 2: Use pyenv (Linux/macOS)
```bash
pyenv install 3.11.7
pyenv local 3.11.7
pip install -r requirements.txt
```

### Option 3: Use Docker
```bash
docker run -it --rm -v $(pwd):/app python:3.11 bash
cd /app
pip install -r requirements.txt
python main.py
```

## Testing Your Setup

Run the test script to check compatibility:
```bash
python test_setup.py
```

This will:
- Check your Python version
- Test all required modules
- Verify microphone access
- Test GUI components

## Package-Specific Issues

### PyAudio
- **Issue**: No pre-compiled wheels for Python 3.13
- **Solution**: Use Python 3.11 or install system dependencies

### Speech Recognition
- **Issue**: May have compatibility issues with very new Python versions
- **Solution**: Pin to specific version: `speech_recognition==3.10.0`

### Tkinter
- **Issue**: Built-in module, should work with any Python version
- **Note**: May need system packages on Linux

## Recommended Development Setup

```bash
# Create virtual environment with Python 3.11
python3.11 -m venv speech-app-env
source speech-app-env/bin/activate  # Linux/macOS
# or
speech-app-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_setup.py

# Run application
python main.py
```

**Note**: Python 3.9+ is required. Python 3.8 and below are not supported.

## Still Having Issues?

1. **Check your Python version**: `python --version`
2. **Run the test script**: `python test_setup.py`
3. **Try a different Python version**: Use Python 3.11
4. **Check system dependencies**: Install portaudio, tkinter dev packages
5. **Use virtual environment**: Isolate your dependencies

For more help, check the main README.md or create an issue in the repository.
