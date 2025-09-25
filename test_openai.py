#!/usr/bin/env python3
"""
Test script for OpenAI integration
"""

import os
import sys

def test_openai_imports():
    """Test if OpenAI modules can be imported"""
    print("Testing OpenAI integration...")
    
    try:
        from openai_config import openai_config
        print("✓ OpenAI config imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import openai_config: {e}")
        return False
    
    try:
        from openai_integration import openai_analyzer
        print("✓ OpenAI analyzer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import openai_analyzer: {e}")
        return False
    
    return True

def test_openai_configuration():
    """Test OpenAI configuration"""
    print("\nTesting OpenAI configuration...")
    
    try:
        from openai_config import openai_config
        
        print(f"API Key configured: {openai_config.is_configured()}")
        print(f"Model: {openai_config.model}")
        print(f"Max tokens: {openai_config.max_tokens}")
        print(f"Temperature: {openai_config.temperature}")
        
        if openai_config.is_configured():
            print("✓ OpenAI is properly configured")
            return True
        else:
            print("⚠ OpenAI not configured - will use template fallback")
            print("  Set OPENAI_API_KEY environment variable or create openai_key.txt")
            return False
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_openai_analyzer():
    """Test OpenAI analyzer functionality"""
    print("\nTesting OpenAI analyzer...")
    
    try:
        from openai_integration import openai_analyzer
        
        print(f"OpenAI available: {openai_analyzer.is_available()}")
        
        if openai_analyzer.is_available():
            print("✓ OpenAI analyzer is ready")
            return True
        else:
            print("⚠ OpenAI analyzer not available - will use template fallback")
            return False
            
    except Exception as e:
        print(f"✗ Analyzer test failed: {e}")
        return False

def test_template_fallback():
    """Test template fallback functionality"""
    print("\nTesting template fallback...")
    
    try:
        from openai_integration import openai_analyzer
        
        # Test topic analysis
        test_explanation = {
            'title': 'Python Programming',
            'summary': 'Python is a versatile programming language',
            'challenges': 'Memory management, performance optimization',
            'commands': 'python script.py, pip install package'
        }
        
        result = openai_analyzer.generate_topic_analysis('python', test_explanation)
        print("✓ Template fallback working for topic analysis")
        
        # Test troubleshooting analysis
        test_suggestions = {
            'approach': 'Systematic debugging',
            'steps': 'Check logs, verify configuration',
            'commands': 'grep error /var/log/app.log'
        }
        
        result = openai_analyzer.generate_troubleshooting_analysis(
            'My application is crashing', test_suggestions
        )
        print("✓ Template fallback working for troubleshooting analysis")
        
        return True
        
    except Exception as e:
        print(f"✗ Template fallback test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("OpenAI Integration Test Suite")
    print("=" * 40)
    
    tests = [
        test_openai_imports,
        test_openai_configuration,
        test_openai_analyzer,
        test_template_fallback
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! OpenAI integration is ready.")
    elif passed >= total - 1:
        print("✅ Most tests passed. OpenAI integration should work with fallback.")
    else:
        print("⚠️ Some tests failed. Check configuration and dependencies.")
    
    print("\nNext steps:")
    print("1. Set OPENAI_API_KEY environment variable for full functionality")
    print("2. Run the main application: python main.py")
    print("3. Check the status bar for OpenAI configuration status")

if __name__ == "__main__":
    main()
