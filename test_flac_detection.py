#!/usr/bin/env python3
"""
Test script to verify FLAC detection and Google Speech Recognition availability
"""

def test_flac_availability():
    """Test if FLAC is available for Google Speech Recognition"""
    try:
        from speech_recognition.audio import get_flac_converter
        converter = get_flac_converter()
        if converter is not None:
            print("✅ FLAC is available - Google Speech Recognition will work")
            return True
        else:
            print("❌ FLAC converter is None - Google Speech Recognition will fail")
            return False
    except Exception as e:
        print(f"❌ FLAC not available: {e}")
        return False

def test_google_speech_recognition():
    """Test Google Speech Recognition with a simple example"""
    try:
        import speech_recognition as sr
        
        # Create a recognizer
        r = sr.Recognizer()
        
        # Test if we can create a microphone (this doesn't require FLAC)
        mic = sr.Microphone()
        print("✅ Microphone access works")
        
        # Test if we can get FLAC converter
        try:
            from speech_recognition.audio import get_flac_converter
            converter = get_flac_converter()
            if converter:
                print("✅ FLAC converter available - Google Speech Recognition should work")
                return True
            else:
                print("❌ FLAC converter not available - Google Speech Recognition will fail")
                return False
        except Exception as e:
            print(f"❌ FLAC converter test failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Google Speech Recognition test failed: {e}")
        return False

def main():
    """Run FLAC detection tests"""
    print("FLAC Detection Test")
    print("=" * 40)
    
    print("\n1. Testing FLAC availability:")
    flac_available = test_flac_availability()
    
    print("\n2. Testing Google Speech Recognition setup:")
    google_available = test_google_speech_recognition()
    
    print("\n" + "=" * 40)
    print("Test Summary:")
    print(f"FLAC Available: {'✅ YES' if flac_available else '❌ NO'}")
    print(f"Google Speech Recognition: {'✅ YES' if google_available else '❌ NO'}")
    
    if not flac_available:
        print("\n⚠️  Recommendation:")
        print("   - FLAC is not available, so Google Speech Recognition will fail")
        print("   - The application will use Whisper only (which is better anyway!)")
        print("   - No need to install FLAC - Whisper provides better accuracy")
    else:
        print("\n✅ Both FLAC and Google Speech Recognition are available")
        print("   - The application will use Whisper first, then Google as fallback")

if __name__ == "__main__":
    main()
