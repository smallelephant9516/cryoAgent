#!/usr/bin/env python3
"""Test script for complete RELION preprocessing workflow with ReAct logic."""

import os
import sys
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, '/home/daoyi/Github/cryoagent')

from cryoagent.core.relion_preprocessing.preprocessing_agent import PreprocessingAgent
from cryoagent.core.relion_preprocessing.preprocessing_workflow import PreprocessingWorkflow
from cryoagent.config.config_loader import ConfigLoader

def test_complete_workflow():
    """Test the complete RELION preprocessing workflow."""
    print("🧪 Testing Complete RELION Preprocessing Workflow")
    print("=" * 60)
    
    try:
        # Load configuration
        config_loader = ConfigLoader(
            config_path="configs/relion/preprocessing_config.json",
            master_config_path="configs/master_config.json"
        )
        
        # Create CryoAgentConfig
        config = config_loader.load_config()
        
        # Initialize preprocessing agent
        agent = PreprocessingAgent(config)
        print("✅ RELION preprocessing agent initialized successfully!")
        
        # Create workflow
        workflow = PreprocessingWorkflow(agent, config)
        print("✅ Workflow created successfully!")
        
        # Test workflow input
        workflow_input = workflow._create_workflow_input()
        print("✅ Workflow input created successfully!")
        print(f"Workflow input length: {len(workflow_input)} characters")
        
        # Test individual tools
        print("\n🔧 Testing individual tools...")
        
        # Test reason_about_workflow
        print("\n🤔 Testing reason_about_workflow...")
        reason_result = agent._reason_about_workflow_tool("test input")
        print(f"Result: {reason_result}")
        
        # Test validate_inputs
        print("\n🔍 Testing validate_inputs...")
        movies_path = agent.microscope_config.get('movies_path', 'Micrographs/*.tif')
        validate_result = agent._validate_inputs_tool(f'{{"input_type": "movies", "input_path": "{movies_path}"}}')
        print(f"Result: {validate_result}")
        
        # Test import_movies (if movies exist)
        if "Found" in validate_result and "movie files" in validate_result:
            print("\n🎬 Testing import_movies...")
            import_result = agent._import_movies_tool('{"wait_for_completion": "true", "timeout": "300"}')
            print(f"Result: {import_result}")
            
            # Test motion_correction if import was successful
            if "Successfully imported" in import_result:
                print("\n🎯 Testing motion_correction...")
                motion_result = agent._motion_correction_tool('{"wait_for_completion": "true", "timeout": "600"}')
                print(f"Result: {motion_result}")
                
                # Test ctf_estimation if motion correction was successful
                if "Successfully performed motion correction" in motion_result:
                    print("\n🔬 Testing ctf_estimation...")
                    ctf_result = agent._ctf_estimation_tool('{"wait_for_completion": "true", "timeout": "600"}')
                    print(f"Result: {ctf_result}")
                    
                    # Test micrograph_selection if CTF estimation was successful
                    if "Successfully estimated CTF parameters" in ctf_result:
                        print("\n📊 Testing micrograph_selection...")
                        selection_result = agent._micrograph_selection_tool('{"wait_for_completion": "true", "timeout": "300"}')
                        print(f"Result: {selection_result}")
        
        # Test final workflow state
        print("\n📋 Final workflow state:")
        for step, state in agent.workflow_state.items():
            status = "✅ COMPLETED" if state["completed"] else "⏳ PENDING"
            print(f"  {step}: {status}")
            if state["job_dir"]:
                print(f"    Job directory: {state['job_dir']}")
            if state["output_file"]:
                print(f"    Output file: {state['output_file']}")
        
        print("\n🎉 Workflow testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Workflow testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_react_workflow():
    """Test the ReAct workflow execution."""
    print("\n🤖 Testing ReAct Workflow Execution")
    print("=" * 40)
    
    try:
        # Load configuration
        config_loader = ConfigLoader(
            config_path="configs/relion/preprocessing_config.json",
            master_config_path="configs/master_config.json"
        )
        
        # Create CryoAgentConfig
        config = config_loader.load_config()
        
        # Initialize preprocessing agent
        agent = PreprocessingAgent(config)
        
        # Create workflow
        workflow = PreprocessingWorkflow(agent, config)
        
        # Test ReAct workflow execution
        print("🚀 Starting ReAct workflow execution...")
        results = workflow.run()
        
        print(f"✅ ReAct workflow completed with {len(results)} results")
        
        # Display results
        for result in results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"  {result.step.value}: {status}")
            if result.message:
                print(f"    Message: {result.message}")
            if result.error:
                print(f"    Error: {result.error}")
        
        return True
        
    except Exception as e:
        print(f"❌ ReAct workflow testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🧪 Testing RELION Preprocessing Workflow with ReAct Logic")
    print("=" * 70)
    
    # Test 1: Complete workflow
    #success1 = test_complete_workflow()
    
    # Test 2: ReAct workflow (optional, might take longer)
    print("\n" + "="*70)
    success2 = test_react_workflow()
    
    print("\n" + "="*70)
    print("🎯 Test Summary:")
    #print(f"✅ Complete workflow test: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ ReAct workflow test: {'PASSED' if success2 else 'FAILED'}")

if __name__ == "__main__":
    main()
