"""LLM Factory for creating different language model instances."""

from typing import Optional, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from ..config.config_loader import ModelConfig


class LLMFactory:
    """Factory class for creating LLM instances based on provider configuration."""
    
    @staticmethod
    def create_llm(model_config: ModelConfig, provider: str = "openai") -> BaseLanguageModel:
        """
        Create an LLM instance based on the provider and configuration.
        
        Args:
            model_config: Model configuration containing API details
            provider: Provider name (openai, deepseek, panshi)
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider in ["openai", "deepseek", "panshi"]:
            # All these providers use OpenAI-compatible API
            return ChatOpenAI(
                model=model_config.model_name,
                temperature=model_config.temperature,
                api_key=model_config.api_key,
                base_url=model_config.base_url,
                timeout=model_config.timeout,
                # Add provider-specific headers if needed
                default_headers=_get_provider_headers(provider)
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}. "
                           f"Supported providers: openai, deepseek, panshi")
    
    @staticmethod
    def get_supported_providers() -> list[str]:
        """Get list of supported LLM providers."""
        return ["openai", "deepseek", "panshi"]


def _get_provider_headers(provider: str) -> Optional[Dict[str, str]]:
    """
    Get provider-specific headers if needed.
    
    Args:
        provider: Provider name
        
    Returns:
        Dictionary of headers or None
    """
    headers = {}
    
    if provider == "panshi":
        # Add any Panshi-specific headers here
        headers["User-Agent"] = "CryoAgent/1.0"
    elif provider == "deepseek":
        # Add any DeepSeek-specific headers here
        headers["User-Agent"] = "CryoAgent/1.0"
    elif provider == "openai":
        # Add any OpenAI-specific headers here
        headers["User-Agent"] = "CryoAgent/1.0"
    
    return headers if headers else None
