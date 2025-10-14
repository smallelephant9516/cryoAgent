#!/usr/bin/env python3
"""Test script for homogeneous refinement using J103 (ab initio job) with default parameters."""

import sys
import os
sys.path.append('/home/daoyi/Github/cryoagent')

from cryoagent.config.config_loader import CryoAgentConfig, ConfigLoader
from cryoagent.tools.cryosparc_tools import CryoSPARCTools
from cryoagent.core.cryosparc_reconstruction.reconstruction_agent import ReconstructionAgent
from cryoagent.core.llm_factory import LLMFactory

def test_homogeneous_refinement():
    """Test homogeneous refinement with J101 using default parameters."""
    
    print("🧪 Testing Homogeneous Refinement with J101 (Ab Initio Job)")
    print("=" * 60)
    
    # Load configuration
    master_config_path = "/home/daoyi/Github/cryoagent/configs/master_config.json"
    stage_config_path = "/home/daoyi/Github/cryoagent/configs/3d_reconstruction_config.json"
    config_loader = ConfigLoader(master_config_path, stage_config_path)
    config = config_loader.load_config()
    
    print(f"✅ Loaded configuration from: {master_config_path}")
    print(f"✅ Stage configuration from: {stage_config_path}")
    print(f"📋 Project UID: {config.workflow.project_uid}")
    print(f"📋 Workspace UID: {config.workflow.workspace_uid}")
    
    # Initialize CryoSPARC tools
    try:
        cryosparc_tools = CryoSPARCTools(config.cryosparc)
        print("✅ CryoSPARC tools initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize CryoSPARC tools: {e}")
        return
    
    # Initialize reconstruction agent
    try:
        agent = ReconstructionAgent(cryosparc_tools, config)
        print("✅ Reconstruction agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize reconstruction agent: {e}")
        return
    
    # Test workflow input
    workflow_input = """
    Run homogeneous refinement using the ab initio job J101 with default parameters.
    
    The ab initio job J101 should have completed and produced initial 3D model(s).
    Use default parameters for homogeneous refinement:
    - No specific refinement resolution (let CryoSPARC auto-determine)
    - Use C1 symmetry (no symmetry)
    - Enable defocus refinement
    - Enable global CTF refinement
    
    Please:
    1. First check the status of J101 to ensure it completed successfully
    2. If J101 is completed, start homogeneous refinement using J101 as both particles and volume source
    3. Wait for the refinement job to complete
    4. Provide analysis of the results
    
    IMPORTANT: If the job fails at any step, please:
    - Analyze the error logs using get_job_log tool
    - Think through alternative approaches
    - Suggest potential solutions based on the error analysis
    - Consider different parameter combinations or alternative workflows
    """
    
    print("\n🚀 Starting Homogeneous Refinement Workflow")
    print("-" * 40)
    
    try:
        # Run the workflow with self-reflection
        result = agent.run_react_workflow(
            workflow_input=workflow_input,
            conversation_id="test_homogeneous_refinement_j101"
        )
        
        print("\n📊 Workflow Result:")
        print("=" * 40)
        print(result)
        
        # Get tool execution log for analysis
        tool_log = agent.get_tool_execution_log()
        print(f"\n🔧 Tool Execution Summary:")
        print(f"   Total tools executed: {len(tool_log)}")
        
        for i, entry in enumerate(tool_log, 1):
            tool_name = entry.get('tool', 'unknown')
            timestamp = entry.get('timestamp', 0)
            error = entry.get('error')
            result_info = entry.get('result')
            
            print(f"   {i}. {tool_name}")
            if error:
                print(f"      ❌ Error: {error}")
            elif result_info:
                print(f"      ✅ Success: {str(result_info)[:100]}...")
        
        # Get conversation log file
        log_file = agent.get_conversation_log_file()
        if log_file:
            print(f"\n📝 Conversation log saved to: {log_file}")
        
    except Exception as e:
        print(f"❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Test completed")

if __name__ == "__main__":
    test_homogeneous_refinement()
