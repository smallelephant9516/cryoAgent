#!/usr/bin/env python3
"""
Test script to verify DeepSeek API connection.

This script tests whether the DeepSeek API is accessible with the current configuration.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import cryoagent
current_dir = Path.cwd()
sys.path.insert(0, str(current_dir))

from cryoagent.config.config_loader import ConfigLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def test_deepseek_connection():
    """Test the connection to DeepSeek API."""
    print("🔗 Testing DeepSeek API Connection")
    print("=" * 50)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config_loader = ConfigLoader("config.json")
        config = config_loader.load_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Model: {config.agent.model_name}")
        print(f"   - Base URL: {config.agent.base_url}")
        print(f"   - Temperature: {config.agent.temperature}")
        print()
        
        # Check if API key is available
        api_key = config.agent.api_key
        if not api_key or api_key.startswith("${"):
            print("❌ API key not found or not set")
            print("   Please set the DEEPSEEK_API_KEY environment variable")
            print("   or update the config.json file with your API key")
            print()
            print("🔧 To set the environment variable:")
            print("   export DEEPSEEK_API_KEY='your-api-key-here'")
            print()
            print("🔧 Or update config.json directly:")
            print('   "api_key": "your-api-key-here"')
            return False
        
        print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
        print()
        
        # Initialize the LLM
        print("🤖 Initializing DeepSeek LLM...")
        llm = ChatOpenAI(
            model=config.agent.model_name,
            temperature=config.agent.temperature,
            api_key=api_key,
            base_url=config.agent.base_url
        )
        print("✅ LLM initialized successfully")
        print()
        
        # Test the connection with a simple message
        print("🧪 Testing API connection...")
        test_message = HumanMessage(content="Hello! Please respond with 'Connection successful' to confirm the API is working.")
        
        try:
            response = llm.invoke([test_message])
            print("✅ API connection successful!")
            print(f"   Response: {response.content}")
            print()
            
            # Test with a more complex message
            print("🧪 Testing with a more complex query...")
            complex_message = HumanMessage(content="What is 2+2? Please respond with just the number.")
            
            response = llm.invoke([complex_message])
            print("✅ Complex query successful!")
            print(f"   Response: {response.content}")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ API connection failed: {str(e)}")
            print()
            print("🔍 Troubleshooting tips:")
            print("   1. Check if your API key is valid")
            print("   2. Verify the base URL is correct")
            print("   3. Check your internet connection")
            print("   4. Ensure you have sufficient API credits")
            return False
            
    except FileNotFoundError:
        print("❌ Configuration file not found: config.json")
        print("   Please ensure config.json exists in the project root")
        return False
        
    except Exception as e:
        print(f"❌ Error during connection test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_variables():
    """Test if environment variables are properly set."""
    print("🔧 Testing Environment Variables")
    print("-" * 40)
    
    # Check for DEEPSEEK_API_KEY
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if api_key:
        print(f"✅ DEEPSEEK_API_KEY found: {api_key[:8]}...{api_key[-4:]}")
    else:
        print("❌ DEEPSEEK_API_KEY not found in environment variables")
        print("   You can set it with: export DEEPSEEK_API_KEY='your-api-key'")
    
    # Check for LICENSE_ID
    license_id = os.environ.get("LICENSE_ID")
    if license_id:
        print(f"✅ LICENSE_ID found: {license_id[:8]}...{license_id[-4:]}")
    else:
        print("❌ LICENSE_ID not found in environment variables")
        print("   You can set it with: export LICENSE_ID='your-license-id'")
    
    print()


def test_simple_connection():
    """Test connection with minimal setup."""
    print("🧪 Simple Connection Test")
    print("-" * 40)
    
    # Check if API key is in environment
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY not found in environment variables")
        return False
    
    print(f"✅ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    try:
        # Test with minimal LLM setup
        llm = ChatOpenAI(
            model="deepseek-chat",
            temperature=0.1,
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        print("✅ LLM initialized successfully")
        
        # Test simple message
        from langchain_core.messages import HumanMessage
        message = HumanMessage(content="Hello! Please respond with 'Test successful'.")
        
        response = llm.invoke([message])
        print(f"✅ API response: {response.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {str(e)}")
        return False


def main():
    """Main function to run all tests."""
    print("🚀 DeepSeek Connection Test Suite")
    print("=" * 60)
    print()
    
    # Test environment variables first
    test_environment_variables()
    
    # Try full configuration test first
    print("🧪 Testing with full configuration...")
    success = test_deepseek_connection()
    
    # If that fails, try simple connection test
    if not success:
        print()
        print("🧪 Trying simple connection test...")
        success = test_simple_connection()
    
    print()
    print("📊 Test Results")
    print("-" * 40)
    if success:
        print("✅ All tests passed! DeepSeek API connection is working correctly.")
        print()
        print("🎉 You can now use CryoAgent with DeepSeek!")
        print()
        print("💡 Next steps:")
        print("   1. Run the basic workflow: python cryoagent_workflow.py --workflow basic")
        print("   2. Test CryoSPARC connection: python test_cryosparc_connection.py")
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
        print("   1. Set environment variable: export DEEPSEEK_API_KEY='your-key'")
        print("   2. Or update config.json with your API key")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
