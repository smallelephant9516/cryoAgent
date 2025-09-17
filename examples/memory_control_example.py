#!/usr/bin/env python3
"""
Example demonstrating the LLM memory control functionality.

This example shows how to use the new memory control parameters to control
whether the LLM maintains memory between conversations or starts fresh each time.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import cryoagent
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryoagent import (
    ReActCryoEMAgent, 
    ReActCryoEMWorkflow, 
    CryoSPARCTools,
    ConfigLoader
)


def demonstrate_memory_control():
    """Demonstrate the memory control functionality."""
    print("🧠 LLM Memory Control Example")
    print("=" * 50)
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config_loader = ConfigLoader("config.json")
        config = config_loader.load_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Memory Control Settings:")
        print(f"     * Clear memory on new conversation: {config.memory_control.clear_memory_on_new_conversation}")
        print(f"     * Maintain context between interactions: {config.memory_control.maintain_context_between_interactions}")
        print()
        
        # Initialize CryoSPARC tools
        print("🔧 Initializing CryoSPARC tools...")
        cryosparc_tools = CryoSPARCTools(config.cryosparc)
        print("✅ CryoSPARC tools initialized")
        print()
        
        # Initialize ReAct agent
        print("🤖 Initializing ReAct CryoEM agent...")
        agent = ReActCryoEMAgent(
            cryosparc_tools=cryosparc_tools,
            config=config
        )
        print("✅ ReAct agent initialized")
        print()
        
        # Demonstrate memory control
        print("🧪 Demonstrating Memory Control")
        print("-" * 40)
        
        # Show initial memory status
        memory_status = agent.get_memory_status()
        print(f"Initial memory status: {memory_status}")
        print()
        
        # Example 1: Run workflow with conversation ID
        print("📝 Example 1: Running workflow with conversation ID")
        print("   This will demonstrate how memory is managed between conversations")
        
        try:
            result = agent.run_react_workflow(
                "Please analyze the current workflow state and provide a summary", 
                conversation_id="demo_conversation_1"
            )
            print(f"   Workflow result: {result[:100]}...")
        except Exception as e:
            print(f"   Expected error (CryoSPARC not running): {str(e)[:100]}...")
        
        memory_status = agent.get_memory_status()
        print(f"   Memory status after first workflow: {memory_status}")
        print()
        
        # Example 2: Change memory control settings dynamically
        print("📝 Example 2: Dynamic Memory Control Changes")
        print("   Changing settings to maintain context between interactions...")
        
        agent.set_memory_control(
            clear_on_new_conversation=False,
            maintain_context=True
        )
        
        memory_status = agent.get_memory_status()
        print(f"   Memory status after changing settings: {memory_status}")
        print()
        
        # Example 3: Force clear memory
        print("📝 Example 3: Force Clear Memory")
        print("   Manually clearing all memory...")
        
        agent.force_clear_memory()
        
        memory_status = agent.get_memory_status()
        print(f"   Memory status after force clear: {memory_status}")
        print()
        
        print("✅ Memory control demonstration completed!")
        print()
        print("📚 How to Use Memory Control:")
        print("   1. Configure in config.json:")
        print("      \"memory_control\": {")
        print("        \"clear_memory_on_new_conversation\": true,")
        print("        \"maintain_context_between_interactions\": false")
        print("      }")
        print()
        print("   2. Use conversation IDs to group related interactions:")
        print("      agent.run_react_workflow(input, conversation_id='my_workflow')")
        print()
        print("   3. Change settings dynamically:")
        print("      agent.set_memory_control(clear_on_new_conversation=False)")
        print()
        print("   4. Force clear memory when needed:")
        print("      agent.force_clear_memory()")
        print()
        print("💡 Use Cases:")
        print("   - Debugging: Set clear_memory_on_new_conversation=true for fresh starts")
        print("   - Continuous workflows: Set maintain_context_between_interactions=true")
        print("   - Session management: Use conversation_id to group related interactions")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demonstrate_memory_control()
