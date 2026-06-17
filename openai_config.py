# openai_config.py
# OpenAI configuration settings for AI analysis

import os
from typing import Optional


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


class OpenAIConfig:
    """Configuration for OpenAI API integration."""

    MODEL_PRICING = {
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }

    def __init__(self):
        self.api_key = self._get_api_key()

        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))

        self.use_responses_api = _env_bool("USE_RESPONSES_API", True)
        self.stream_responses = _env_bool("OPENAI_STREAM", True)

        self.max_requests_per_minute = int(os.getenv("OPENAI_MAX_REQUESTS_PER_MINUTE", "20"))
        self.max_tokens_per_minute = int(os.getenv("OPENAI_MAX_TOKENS_PER_MINUTE", "40000"))

        self.cache_responses = _env_bool("OPENAI_CACHE_RESPONSES", True)
        self.cache_expiration_hours = int(os.getenv("OPENAI_CACHE_HOURS", "24"))

        self.use_fallback_on_error = _env_bool("OPENAI_USE_FALLBACK_ON_ERROR", True)
        self.fallback_to_template = _env_bool("OPENAI_FALLBACK_TO_TEMPLATE", True)

        self.system_instructions = (
            "You are a technical expert providing detailed analysis and recommendations "
            "for software development, system administration, and troubleshooting."
        )

    def _get_api_key(self) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key

        config_file = os.path.join(os.path.dirname(__file__), "openai_key.txt")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    return f.read().strip()
            except OSError as e:
                print(f"Error reading OpenAI key from config file: {e}")

        return None

    def is_configured(self) -> bool:
        return self.api_key is not None and len(self.api_key.strip()) > 0

    def get_model_pricing(self) -> dict:
        return self.MODEL_PRICING.get(
            self.model,
            self.MODEL_PRICING["gpt-4.1-mini"],
        )

    def get_model_info(self) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "configured": self.is_configured(),
            "use_responses_api": self.use_responses_api,
            "stream_responses": self.stream_responses,
        }


openai_config = OpenAIConfig()
