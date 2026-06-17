#!/usr/bin/env python3
"""
Test script for OpenAI integration
"""

import os
import sys


def test_openai_imports():
    print("Testing OpenAI integration...")
    try:
        from openai_config import openai_config
        print("OK openai_config imported")
    except ImportError as e:
        print(f"FAIL openai_config: {e}")
        return False

    try:
        from openai_integration import openai_analyzer
        print("OK openai_analyzer imported")
    except ImportError as e:
        print(f"FAIL openai_analyzer: {e}")
        return False

    return True


def test_openai_configuration():
    print("\nTesting OpenAI configuration...")
    try:
        from openai_config import openai_config

        print(f"API Key configured: {openai_config.is_configured()}")
        print(f"Model: {openai_config.model}")
        print(f"Responses API: {openai_config.use_responses_api}")
        print(f"Streaming: {openai_config.stream_responses}")
        print(f"Max tokens: {openai_config.max_tokens}")

        if openai_config.is_configured():
            print("OK OpenAI is properly configured")
            return True

        print("WARN OpenAI not configured - will use template fallback")
        return False
    except Exception as e:
        print(f"FAIL configuration test: {e}")
        return False


def test_rollback_flags():
    print("\nTesting API rollback configuration...")
    try:
        from openai_config import OpenAIConfig

        os.environ["USE_RESPONSES_API"] = "false"
        os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
        cfg = OpenAIConfig()
        assert cfg.use_responses_api is False
        assert cfg.model == "gpt-4o-mini"
        print("OK rollback env vars respected")

        del os.environ["USE_RESPONSES_API"]
        del os.environ["OPENAI_MODEL"]
        return True
    except Exception as e:
        print(f"FAIL rollback test: {e}")
        return False


def test_openai_analyzer():
    print("\nTesting OpenAI analyzer...")
    try:
        from openai_integration import openai_analyzer

        print(f"OpenAI available: {openai_analyzer.is_available()}")
        if openai_analyzer.is_available():
            print("OK OpenAI analyzer is ready")
            return True

        print("WARN OpenAI analyzer not available - will use template fallback")
        return False
    except Exception as e:
        print(f"FAIL analyzer test: {e}")
        return False


def test_template_fallback():
    print("\nTesting template fallback...")
    try:
        from openai_integration import openai_analyzer

        test_explanation = {
            "title": "Python Programming",
            "summary": "Python is a versatile programming language",
            "challenges": "Memory management, performance optimization",
            "commands": "python script.py, pip install package",
        }
        result = openai_analyzer._get_fallback_topic_analysis("python", test_explanation)
        assert isinstance(result, str) and len(result) > 0
        print("OK template fallback for topic analysis")

        test_suggestions = {
            "approach": "Systematic debugging",
            "steps": "Check logs, verify configuration",
            "commands": "grep error /var/log/app.log",
        }
        result = openai_analyzer._get_fallback_troubleshooting_analysis(
            "My application is crashing", test_suggestions
        )
        assert isinstance(result, str) and len(result) > 0
        print("OK template fallback for troubleshooting analysis")
        return True
    except Exception as e:
        print(f"FAIL template fallback test: {e}")
        return False


def test_cost_calculation():
    print("\nTesting cost calculation...")
    try:
        from openai_integration import openai_analyzer
        from openai_config import openai_config

        cost = openai_analyzer._calculate_cost(1000, 500)
        assert cost >= 0
        pricing = openai_config.get_model_pricing()
        assert "input" in pricing and "output" in pricing
        print(f"OK cost calculation for {openai_config.model}: ${cost}")
        return True
    except Exception as e:
        print(f"FAIL cost calculation: {e}")
        return False


def main():
    print("OpenAI Integration Test Suite")
    print("=" * 40)

    tests = [
        test_openai_imports,
        test_openai_configuration,
        test_rollback_flags,
        test_openai_analyzer,
        test_template_fallback,
        test_cost_calculation,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"FAIL test exception: {e}")

    total = len(tests)
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("All tests passed.")
    elif passed >= total - 2:
        print("Most tests passed. Integration should work with fallback.")
    else:
        print("Some tests failed. Check configuration and dependencies.")

    return passed >= total - 2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
