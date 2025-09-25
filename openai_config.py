# openai_config.py
# OpenAI configuration settings for AI analysis

import os
from typing import Optional

class OpenAIConfig:
    """Configuration for OpenAI API integration"""
    
    def __init__(self):
        # API Key - can be set via environment variable or config file
        self.api_key = self._get_api_key()
        
        # Model settings
        self.model = "gpt-4o-mini"  # Cost-effective model for analysis
        self.max_tokens = 1000
        self.temperature = 0.7
        
        # Rate limiting
        self.max_requests_per_minute = 20
        self.max_tokens_per_minute = 40000
        
        # Caching settings
        self.cache_responses = True
        self.cache_expiration_hours = 24
        
        # Fallback settings
        self.use_fallback_on_error = True
        self.fallback_to_template = True
    
    def _get_api_key(self) -> Optional[str]:
        """Get OpenAI API key from environment or config"""
        # Try environment variable first
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Try config file
        config_file = os.path.join(os.path.dirname(__file__), 'openai_key.txt')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Error reading OpenAI key from config file: {e}")
        
        return None
    
    def is_configured(self) -> bool:
        """Check if OpenAI is properly configured"""
        return self.api_key is not None and len(self.api_key.strip()) > 0
    
    def get_model_info(self) -> dict:
        """Get model information for display"""
        return {
            'model': self.model,
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'configured': self.is_configured()
        }

# Global config instance
openai_config = OpenAIConfig()
