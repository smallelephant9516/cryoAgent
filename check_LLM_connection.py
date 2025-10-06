#!/usr/bin/env python3
"""Test script to verify the configured LLM API connection.

This script reads the active provider from config.json and performs a simple
connectivity check using the corresponding API credentials.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Add the parent directory to the path so we can import cryoagent
current_dir = Path.cwd()
sys.path.insert(0, str(current_dir))

from cryoagent.config.config_loader import ConfigLoader, CryoAgentConfig, ModelConfig
from cryoagent.core.llm_factory import LLMFactory


PROVIDER_METADATA: Dict[str, Dict[str, str]] = {
    "deepseek": {
        "display": "DeepSeek",
        "env_var": "DEEPSEEK_API_KEY",
        "default_base_url": "https://api.deepseek.com",
    },
    "openai": {
        "display": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "default_base_url": "https://api.openai.com/v1",
    },
    "panshi": {
        "display": "Panshi",
        "env_var": "PANSHI_API_KEY",
        "default_base_url": "https://uni-api.cstcloud.cn/v1",
    },
}


def _mask_key(value: str) -> str:
    """Return a masked representation of an API key for display purposes."""
    value = value.strip()
    if len(value) <= 8:
        return "*" * len(value) if value else ""
    return f"{value[:8]}...{value[-4:]}"


def _get_provider_metadata(provider: str) -> Dict[str, str]:
    """Return metadata for the given provider with safe fallbacks."""
    provider = (provider or "").lower()
    return PROVIDER_METADATA.get(
        provider,
        {
            "display": provider or "Unknown",
            "env_var": "",
            "default_base_url": "",
        },
    )


def _ensure_provider_with_api_key(config: CryoAgentConfig) -> Tuple[str, ModelConfig, bool]:
    """Ensure the agent configuration has a provider with a valid API key."""
    agent_settings = config.agent
    provider = (agent_settings.provider or "").lower()
    model_config = agent_settings.get_current_model_config()

    if agent_settings._is_api_key_valid(model_config.api_key):
        return provider, model_config, False

    try:
        selected_provider = agent_settings.auto_select_provider()
        provider = selected_provider.lower()
        model_config = agent_settings.get_current_model_config()
        if not agent_settings._is_api_key_valid(model_config.api_key):
            raise ValueError("Selected provider does not have a valid API key")
        return provider, model_config, True
    except ValueError as err:
        raise RuntimeError(str(err)) from err


def _create_llm(provider: str, model_config: ModelConfig):
    """Create an LLM instance compatible with the configured provider."""
    provider = (provider or "").lower()
    if provider in PROVIDER_METADATA:
        return LLMFactory.create_llm(model_config, provider)

    # Fallback for legacy/unknown providers using OpenAI-compatible API surface
    return ChatOpenAI(
        model=model_config.model_name,
        temperature=model_config.temperature,
        api_key=model_config.api_key,
        base_url=model_config.base_url,
        timeout=model_config.timeout,
    )


def test_llm_connection(config: CryoAgentConfig) -> bool:
    """Test the connection to the configured LLM provider."""
    print("🔗 Testing LLM API Connection")
    print("=" * 50)

    try:
        provider, model_config, auto_selected = _ensure_provider_with_api_key(config)
        provider_meta = _get_provider_metadata(provider)
        provider_display = provider_meta["display"].strip() or provider.capitalize()

        print("✅ Configuration loaded successfully")
        if auto_selected:
            print(f"   - Provider auto-selected: {provider_display}")
        else:
            print(f"   - Provider: {provider_display}")
        print(f"   - Model: {model_config.model_name}")
        base_url = model_config.base_url or provider_meta.get("default_base_url", "")
        print(f"   - Base URL: {base_url}")
        print(f"   - Temperature: {model_config.temperature}")
        print()

        api_key = (model_config.api_key or "").strip()
        env_var = provider_meta.get("env_var", "")
        if not api_key:
            print("❌ API key not found or not set")
            if env_var:
                print(f"   Please set the {env_var} environment variable")
            print("   or update the config.json file with your API key")
            return False

        print(f"✅ API key found: {_mask_key(api_key)}")
        print()

        print(f"🤖 Initializing {provider_display} LLM...")
        llm = _create_llm(provider, model_config)
        print("✅ LLM initialized successfully")
        print()

        print("🧪 Testing API connection...")
        test_message = HumanMessage(
            content="Hello! Please respond with 'Connection successful' to confirm the API is working."
        )

        try:
            response = llm.invoke([test_message])
            print("✅ API connection successful!")
            print(f"   Response: {response.content}")
            print()

            print("🧪 Testing with a more complex query...")
            complex_message = HumanMessage(
                content="What is 2+2? Please respond with just the number."
            )

            response = llm.invoke([complex_message])
            print("✅ Complex query successful!")
            print(f"   Response: {response.content}")
            print()

            return True

        except Exception as exc:  # pragma: no cover - network errors handled at runtime
            print(f"❌ API connection failed: {exc}")
            print()
            print("🔍 Troubleshooting tips:")
            print("   1. Check if your API key is valid")
            print("   2. Verify the base URL is correct")
            print("   3. Check your internet connection")
            print("   4. Ensure you have sufficient API credits")
            return False

    except RuntimeError as err:
        provider_meta = _get_provider_metadata(config.agent.provider)
        env_var = provider_meta.get("env_var", "")
        print(f"❌ {err}")
        if env_var:
            print(f"   Set the {env_var} environment variable or update config.json")
        else:
            print("   Update config.json with a valid provider and API key")
        return False

    except Exception as exc:  # pragma: no cover - unexpected failures
        print(f"❌ Error during connection test: {exc}")
        return False


def test_environment_variables(active_provider: str) -> None:
    """Report environment variable status for supported providers."""
    print("🔧 Testing Environment Variables")
    print("-" * 40)

    active_provider = (active_provider or "").lower()
    for provider, metadata in PROVIDER_METADATA.items():
        env_var = metadata["env_var"]
        env_value = os.environ.get(env_var, "").strip()
        display = metadata["display"]
        if env_value:
            print(f"✅ {env_var} ({display}) found: {_mask_key(env_value)}")
        else:
            prefix = "❌" if provider == active_provider else "ℹ️"
            print(f"{prefix} {env_var} ({display}) not set")

    license_id = os.environ.get("LICENSE_ID", "").strip()
    if license_id:
        print(f"✅ LICENSE_ID found: {_mask_key(license_id)}")
    else:
        print("ℹ️ LICENSE_ID not found in environment variables")

    print()


def main() -> int:
    """Main function to run all tests."""
    print("🚀 LLM Connection Test Suite")
    print("=" * 60)
    print()

    try:
        config_loader = ConfigLoader("configs/master_config.json")
        config = config_loader.load_config()
    except FileNotFoundError:
        print("❌ Configuration file not found: configs/master_config.json")
        print("   Please ensure configs/master_config.json exists in the project root")
        return 1
    except Exception as exc:
        print(f"❌ Failed to load configuration: {exc}")
        return 1

    test_environment_variables(config.agent.provider)

    print("🧪 Testing with active configuration...")
    success = test_llm_connection(config)

    print()
    print("📊 Test Results")
    print("-" * 40)
    if success:
        print("✅ All tests passed! Configured LLM API connection is working correctly.")
        print()
        print("🎉 You can now use CryoAgent with the selected LLM provider!")
        print()
        print("💡 Next steps:")
        print("   1. Run the basic workflow: python cryoagent_workflow.py --workflow basic")
        print("   2. Test CryoSPARC connection: python check_cryosparc_connection.py")
        print("   3. Try the ReAct workflow example: python examples/react_workflow_example.py")
    else:
        print("❌ Some tests failed. Please check the configuration and try again.")
        print()
        print("🔧 Common issues:")
        print("   1. Invalid or missing API key")
        print("   2. Incorrect base URL")
        print("   3. Network connectivity issues")
        print("   4. Insufficient API credits")
        print()
        print("🔧 To fix:")
        print("   1. Set the appropriate environment variable (e.g., DEEPSEEK_API_KEY)")
        print("   2. Or update config.json with your API key")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
