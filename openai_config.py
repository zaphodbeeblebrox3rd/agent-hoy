# openai_config.py
# OpenAI configuration settings for AI analysis

import os
from typing import Dict, List, Optional, Tuple

CONF_DIR = "conf"
DEFAULT_MODEL = "gpt-4.1-mini"
MODEL_CONFIG_FILE = "openai_model.conf"

# (combobox label, API model id)
AVAILABLE_MODELS: List[Tuple[str, str]] = [
    ("GPT-4o Mini", "gpt-4o-mini"),
    ("GPT-4o", "gpt-4o"),
    ("GPT-4.1 Nano", "gpt-4.1-nano"),
    ("GPT-4.1 Mini", "gpt-4.1-mini"),
    ("GPT-4.1", "gpt-4.1"),
    ("GPT-5 Nano", "gpt-5-nano"),
    ("GPT-5 Mini", "gpt-5-mini"),
    ("GPT-5", "gpt-5"),
    ("GPT-5.4 Nano", "gpt-5.4-nano"),
    ("GPT-5.4 Mini", "gpt-5.4-mini"),
    ("GPT-5.4", "gpt-5.4"),
    ("GPT-5.5", "gpt-5.5"),
]

MODEL_ID_BY_LABEL = {label: model_id for label, model_id in AVAILABLE_MODELS}
LABEL_BY_MODEL_ID = {model_id: label for label, model_id in AVAILABLE_MODELS}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def is_reasoning_model(model: str) -> bool:
    """GPT-5 reasoning models do not support temperature."""
    normalized = model.lower()
    if not normalized.startswith("gpt-5"):
        return False
    return "chat" not in normalized


def _resolve_model_config_path() -> str:
    config_path = os.path.join(CONF_DIR, MODEL_CONFIG_FILE)
    if os.path.exists(config_path):
        return config_path
    example_path = f"{config_path}.example"
    if os.path.exists(example_path):
        return example_path
    return config_path


def load_persisted_model() -> Optional[str]:
    """Load model id from conf/openai_model.conf."""
    path = _resolve_model_config_path()
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = (part.strip() for part in stripped.split("=", 1))
                if key.lower() == "model" and value:
                    return value
    except OSError as exc:
        print(f"Error loading model from {path}: {exc}")

    return None


def save_persisted_model(model_id: str) -> None:
    """Save model id to conf/openai_model.conf."""
    os.makedirs(CONF_DIR, exist_ok=True)
    path = os.path.join(CONF_DIR, MODEL_CONFIG_FILE)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# OpenAI model selection (overrides default; OPENAI_MODEL env still wins on startup)\n")
            handle.write(f"model = {model_id}\n")
    except OSError as exc:
        print(f"Error saving model to {path}: {exc}")


def resolve_startup_model() -> str:
    """OPENAI_MODEL env > persisted conf > default."""
    env_model = os.getenv("OPENAI_MODEL")
    if env_model:
        return env_model

    persisted = load_persisted_model()
    if persisted:
        return persisted

    return DEFAULT_MODEL


class OpenAIConfig:
    """Configuration for OpenAI API integration."""

    MODEL_PRICING: Dict[str, Dict[str, float]] = {
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-5-nano": {"input": 0.10, "output": 0.40},
        "gpt-5-mini": {"input": 0.25, "output": 2.00},
        "gpt-5": {"input": 1.25, "output": 10.00},
        "gpt-5.4-nano": {"input": 0.20, "output": 0.80},
        "gpt-5.4-mini": {"input": 0.75, "output": 4.50},
        "gpt-5.4": {"input": 2.50, "output": 15.00},
        "gpt-5.5": {"input": 5.00, "output": 30.00},
    }

    def __init__(self):
        self.api_key = self._get_api_key()

        self.model = resolve_startup_model()
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "1000"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        self.reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "low")
        self.text_verbosity = os.getenv("OPENAI_TEXT_VERBOSITY", "medium")

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

    def get_model(self) -> str:
        return self.model

    def set_model(self, model_id: str) -> None:
        self.model = model_id

    def get_model_label(self) -> str:
        return LABEL_BY_MODEL_ID.get(self.model, self.model)

    def is_current_model_reasoning(self) -> bool:
        return is_reasoning_model(self.model)

    def get_model_pricing(self) -> dict:
        return self.MODEL_PRICING.get(
            self.model,
            self.MODEL_PRICING[DEFAULT_MODEL],
        )

    def get_model_info(self) -> dict:
        return {
            "model": self.model,
            "model_label": self.get_model_label(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "text_verbosity": self.text_verbosity,
            "configured": self.is_configured(),
            "use_responses_api": self.use_responses_api,
            "stream_responses": self.stream_responses,
            "is_reasoning_model": self.is_current_model_reasoning(),
        }


openai_config = OpenAIConfig()
