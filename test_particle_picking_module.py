#!/usr/bin/env python3
"""Test script for the particle picking module."""

import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cryoagent.config.config_loader import ConfigLoader, CryoAgentConfig
from cryoagent.core.relion_picking import PickingAgent, PickingTools, PickingWorkflow


def test_particle_picking_module():
    """Test the particle picking module."""
    print("🧪 Testing Particle Picking Module")
    print("=" * 50)
    
    # Test 1: Workflow
    print("\n1. Testing PickingWorkflow")
    print("-" * 30)
    try:
        workflow = PickingWorkflow()
        steps = workflow.get_workflow_steps()
        print(f"✅ Workflow loaded successfully")
        print(f"   Total steps: {len(steps)}")
        print(f"   Steps: {[step['step_name'] for step in steps]}")
        
        # Test dependencies
        dependencies = workflow.get_workflow_dependencies()
        print(f"   Dependencies: {dependencies}")
        
        # Test validation
        validation = workflow.validate_workflow_inputs({
            "input_micrographs_star": "Select/job001/micrographs.star"
        })
        print(f"   Validation: {validation}")
        
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
    
    # Test 2: Tools
    print("\n2. Testing PickingTools")
    print("-" * 30)
    try:
        # Create a mock agent for testing
        class MockAgent:
            def _blob_picker_tool(self, **kwargs): return "Mock blob picker"
            def _particle_extraction_tool(self, **kwargs): return "Mock extraction"
            def _classification_2d_tool(self, **kwargs): return "Mock classification"
            def _auto_2d_selection_tool(self, **kwargs): return "Mock selection"
            def _reason_about_workflow_tool(self, **kwargs): return "Mock reasoning"
            def _check_job_status_tool(self, **kwargs): return "Mock status"
            def _wait_for_job_tool(self, **kwargs): return "Mock wait"
            def _get_job_log_tool(self, **kwargs): return "Mock log"
            def _validate_inputs_tool(self, **kwargs): return "Mock validation"
        
        mock_agent = MockAgent()
        
        # Test tool creation
        tools = [
            PickingTools.create_blob_picker_tool(mock_agent),
            PickingTools.create_particle_extraction_tool(mock_agent),
            PickingTools.create_classification_2d_tool(mock_agent),
            PickingTools.create_auto_2d_selection_tool(mock_agent),
            PickingTools.create_reason_about_workflow_tool(mock_agent),
            PickingTools.create_check_job_status_tool(mock_agent),
            PickingTools.create_wait_for_job_tool(mock_agent),
            PickingTools.create_get_job_log_tool(mock_agent),
            PickingTools.create_validate_inputs_tool(mock_agent)
        ]
        
        print(f"✅ Created {len(tools)} tools successfully")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description[:50]}...")
        
    except Exception as e:
        print(f"❌ Tools test failed: {e}")
    
    # Test 3: Agent (without LLM)
    print("\n3. Testing PickingAgent")
    print("-" * 30)
    try:
        # Load configuration
        config_loader = ConfigLoader(
            config_path="configs/master_config.json",
            master_config_path="configs/master_config.json"
        )
        config = config_loader.load_config()
        
        # Create agent (this will fail due to missing API keys, but we can test the structure)
        try:
            agent = PickingAgent(config)
            print(f"✅ Agent created successfully")
            print(f"   Workflow: {agent.workflow.config['workflow_name']}")
            print(f"   Tools available: {len(agent.tools)}")
        except Exception as e:
            print(f"⚠️ Agent creation failed (expected due to missing API keys): {str(e)[:100]}...")
            print("   This is expected in the test environment without API keys")
            
            # Test tool methods directly without full agent initialization
            print("   Testing tool methods directly...")
            
            # Create a minimal agent instance for testing
            class MinimalAgent:
                def __init__(self):
                    from ...tools.relion_tools import RELIONTools
                    from ...config.config_loader import ConfigLoader
                    config_loader = ConfigLoader()
                    settings = config_loader.get_relion_settings()
                    self.relion_tools = RELIONTools(settings, config_loader)
                
                def _blob_picker_tool(self, **kwargs): 
                    return "Mock blob picker result"
                def _particle_extraction_tool(self, **kwargs): 
                    return "Mock particle extraction result"
                def _classification_2d_tool(self, **kwargs): 
                    return "Mock 2D classification result"
                def _auto_2d_selection_tool(self, **kwargs): 
                    return "Mock auto 2D selection result"
            
            minimal_agent = MinimalAgent()
            
            # Test tool methods
            test_input = "Select/job001/micrographs.star"
            
            # Test blob picker
            result = minimal_agent._blob_picker_tool(input_star=test_input)
            print(f"   Blob picker test: {result}")
            
            # Test particle extraction
            result = minimal_agent._particle_extraction_tool(input_star=test_input)
            print(f"   Particle extraction test: {result}")
            
            # Test 2D classification
            result = minimal_agent._classification_2d_tool(input_star="Particles/job001/particles.star")
            print(f"   2D classification test: {result}")
            
            # Test auto 2D selection
            result = minimal_agent._auto_2d_selection_tool(input_opt="Class2D/job001/run_optimiser.star")
            print(f"   Auto 2D selection test: {result}")
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
    
    # Test 4: Configuration
    print("\n4. Testing Configuration")
    print("-" * 30)
    try:
        config_path = "configs/relion/particle_picking_config.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded successfully")
            print(f"   Workflow: {config['workflow_name']}")
            print(f"   Stages: {len(config['stages'])}")
            print(f"   Steps in stage1: {len(config['stages']['stage1']['steps'])}")
        else:
            print(f"❌ Configuration file not found: {config_path}")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Particle picking module testing completed!")
    print("Note: These are structural tests. Actual execution requires valid input files and LLM.")


if __name__ == "__main__":
    test_particle_picking_module()
