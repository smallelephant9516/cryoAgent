"""LLM Factory for creating different language model instances."""

from typing import Optional, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from ..config.config_loader import ModelConfig
from .deepseek_adapter import DeepSeekChatAdapter


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
            model_kwargs: Dict[str, Any] = {}
            extra_body = dict(getattr(model_config, "extra_body", {}) or {})

            # Provider-specific request overrides for DeepSeek V4 models.
            if provider == "deepseek":
                model_name = (model_config.model_name or "").lower()
                if "deepseek-v4" in model_name:
                    v4_overrides = dict(getattr(model_config, "deepseek_v4_extra_body", {}) or {})
                    if v4_overrides:
                        extra_body.update(v4_overrides)
            if extra_body:
                model_kwargs["extra_body"] = extra_body

            model_cls = DeepSeekChatAdapter if provider == "deepseek" else ChatOpenAI

            # All providers here use an OpenAI-compatible API surface.
            return model_cls(
                model=model_config.model_name,
                temperature=model_config.temperature,
                api_key=model_config.api_key,
                base_url=model_config.base_url,
                timeout=model_config.timeout,
                max_retries=getattr(model_config, "max_retries", 2),
                # Add provider-specific headers if needed
                default_headers=_get_provider_headers(provider),
                **model_kwargs
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
