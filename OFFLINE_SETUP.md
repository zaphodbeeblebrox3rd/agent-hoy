# Offline Setup Guide

This guide explains how to set up the application for offline operation when internet connectivity is unreliable or unavailable.

## Offline Capabilities

The application now includes several offline features:

### 1. **Local Caching System**
- **Topic Explanations**: Cached for 24 hours
- **Troubleshooting Suggestions**: Cached for 1 hour
- **Session Data**: Persisted between application restarts
- **Cache Location**: `cache/topic_cache.pkl`

### 2. **Offline Speech Recognition**
- **Fallback Options**: Vosk, PocketSphinx, or local Whisper models
- **Automatic Detection**: Switches to offline mode when internet is unavailable
- **Status Indication**: Shows "Offline mode" in the status bar

### 3. **Network Resilience**
- **Connection Checking**: Automatic detection of network issues
- **Graceful Degradation**: Falls back to cached content when online
- **Error Handling**: Continues operation even with network failures

## Installation for Offline Support

### Basic Offline Setup (Recommended)
```bash
# Install core requirements
pip install -r requirements.txt

# Install offline speech recognition
pip install vosk pocketsphinx
```

### Advanced Offline Setup (Full Features)
```bash
# Install all requirements including Whisper
pip install -r requirements.txt
pip install -r requirements_offline.txt

# For Whisper support (optional)
pip install whisper torch torchaudio
```

## Offline Speech Recognition Options

### Option 1: Vosk (Recommended)
```bash
pip install vosk
```
- **Pros**: Fast, lightweight, good accuracy
- **Cons**: Requires model download
- **Usage**: Automatic fallback when online recognition fails

### Option 2: PocketSphinx
```bash
pip install pocketsphinx
```
- **Pros**: No internet required, works offline
- **Cons**: Lower accuracy than online services
- **Usage**: Built-in offline recognition

### Option 3: Whisper (Advanced)
```bash
pip install whisper torch torchaudio
```
- **Pros**: High accuracy, supports multiple languages
- **Cons**: Large model files, requires more storage
- **Usage**: Local transcription with excellent quality

## Cache Management

### Cache Structure
```
cache/
├── topic_cache.pkl          # Main cache file
├── session_data.json        # Session information
└── offline_models/          # Offline recognition models
```

### Cache Benefits
- **Faster Response**: Cached explanations load instantly
- **Offline Access**: Previously viewed content available offline
- **Reduced Network Usage**: Less dependency on internet connectivity
- **Session Persistence**: Content survives application restarts

### Cache Maintenance
- **Automatic Cleanup**: Old entries are automatically removed
- **Size Management**: Cache grows organically, cleaned periodically
- **Manual Clear**: Delete `cache/` folder to clear all cached data

## Configuration

### Network Settings
```python
# In main.py, you can adjust these settings:
self.cache_dir = "cache"                    # Cache directory
self.last_network_check = None              # Network check frequency
self.use_offline = False                    # Force offline mode
```

### Offline Mode Indicators
- **Status Bar**: Shows "Offline mode - using local recognition"
- **Console Output**: Logs offline recognition attempts
- **Error Messages**: Clear indication when falling back to offline

## Troubleshooting Offline Issues

### Common Problems

1. **"Offline recognition not fully implemented"**
   - **Solution**: Install Vosk or PocketSphinx
   - **Command**: `pip install vosk pocketsphinx`

2. **Cache not loading**
   - **Solution**: Check cache directory permissions
   - **Command**: `chmod 755 cache/`

3. **Offline models not found**
   - **Solution**: Download Vosk models manually
   - **Location**: Place models in `cache/offline_models/`

### Performance Optimization

1. **Cache Size**: Monitor cache directory size
2. **Model Selection**: Choose appropriate offline model size
3. **Network Checks**: Adjust network check frequency if needed

## Usage Examples

### Online Mode (Default)
- Uses Google Speech Recognition
- Real-time transcription
- Full feature set available

### Offline Mode (Automatic)
- Detects network issues
- Switches to local recognition
- Uses cached content
- Continues operation seamlessly

### Force Offline Mode
```python
# In main.py, set:
self.use_offline = True
```

## Best Practices

1. **Regular Cache Updates**: Let the application run online periodically
2. **Model Management**: Keep offline models updated
3. **Storage Monitoring**: Monitor cache directory size
4. **Backup Cache**: Consider backing up cache for important sessions

## Limitations

### Offline Mode Limitations
- **Lower Accuracy**: Offline recognition may be less accurate
- **Limited Vocabulary**: Some technical terms may not be recognized
- **Model Size**: Offline models require storage space
- **Setup Complexity**: Initial setup may require additional dependencies

### Cache Limitations
- **Storage Requirements**: Cache grows over time
- **Staleness**: Cached content may become outdated
- **Platform Specific**: Cache format may vary between systems

## Support

For offline setup issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Test with a simple offline recognition example
4. Review the cache directory for proper permissions

The application is designed to gracefully handle network issues and provide a seamless experience even when offline!
