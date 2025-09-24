#!/usr/bin/env python3
"""
Test script to verify Whisper installation and functionality
"""

def test_whisper_import():
    """Test if Whisper can be imported"""
    try:
        import whisper
        print("✅ Whisper imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Whisper import failed: {e}")
        return False

def test_whisper_model_loading():
    """Test if Whisper model can be loaded"""
    try:
        import whisper
        print("Loading Whisper base model...")
        model = whisper.load_model("base")
        print("✅ Whisper model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Whisper model loading failed: {e}")
        return False

def test_whisper_transcription():
    """Test Whisper transcription with a simple example"""
    try:
        import whisper
        import tempfile
        import os
        
        # Create a simple test audio file (silence)
        # This is just to test the transcription pipeline
        model = whisper.load_model("base")
        
        # Create a temporary WAV file with silence
        import wave
        import numpy as np
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Create a 1-second silence WAV file
            sample_rate = 16000
            duration = 1.0
            samples = np.zeros(int(sample_rate * duration), dtype=np.int16)
            
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())
            
            # Test transcription
            result = model.transcribe(temp_file.name)
            print(f"✅ Whisper transcription test completed: '{result['text']}'")
            
            # Clean up
            os.unlink(temp_file.name)
            return True
            
    except Exception as e:
        print(f"❌ Whisper transcription test failed: {e}")
        return False

def main():
    """Run all Whisper tests"""
    print("Testing Whisper Installation")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_whisper_import),
        ("Model Loading Test", test_whisper_model_loading),
        ("Transcription Test", test_whisper_transcription)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    if all(results):
        print("\n🎉 All tests passed! Whisper is ready to use.")
    else:
        print("\n⚠️ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()
