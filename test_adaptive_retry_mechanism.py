#!/usr/bin/env python3
"""Test script with adaptive retry mechanism for CryoEM jobs."""

import sys
import os
sys.path.append('/home/daoyi/Github/cryoagent')

from cryoagent.config.config_loader import CryoAgentConfig, ConfigLoader
from cryoagent.tools.cryosparc_tools import CryoSPARCTools
from cryoagent.core.cryosparc_reconstruction.reconstruction_agent import ReconstructionAgent

def test_adaptive_retry_mechanism():
    """Test homogeneous refinement with adaptive retry mechanism."""
    
    print("🧪 Testing Adaptive Retry Mechanism for Homogeneous Refinement")
    print("=" * 70)
    
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
    
    # Enhanced workflow input with adaptive retry mechanism
    workflow_input = """
    Run homogeneous refinement using the ab initio job J101 with adaptive retry mechanism.
    
    CRITICAL REQUIREMENTS:
    1. Start with default parameters for homogeneous refinement
    2. If the job FAILS, you MUST try at least 3 different parameter combinations
    3. For each failure, follow this protocol:
       a. Read the job log using get_job_log tool
       b. Analyze the error patterns and root cause
       c. Based on the error analysis, try a different parameter strategy:
          
          STRATEGY 1 (CTF Refinement Issues):
          - If CTF refinement fails: refine_defocus_refine=false, refine_ctf_global_refine=false
          
          STRATEGY 2 (Resolution Issues):
          - If resolution too aggressive: refinement_resolution=15.0 (more conservative)
          
          STRATEGY 3 (Convergence Issues):
          - If convergence fails: symmetry=C1, refinement_resolution=None (auto)
          
          STRATEGY 4 (Alternative Approach):
          - If all refinement attempts fail: Try homogeneous reconstruction instead
          
          STRATEGY 5 (Conservative Approach):
          - Use minimal parameters: Only volume_job_uid, no CTF refinement, no resolution target
     
    4. Document each attempt and the reasoning behind parameter choices
    5. Continue until successful or all strategies exhausted
    6. Provide comprehensive analysis of all attempts
    
    ADAPTIVE RETRY PROTOCOL:
    - Start with default parameters
    - For each failure: analyze log → choose strategy → retry
    - Try at least 3 different parameter combinations
    - Don't give up until all reasonable options exhausted
    - Learn from each failure to improve next attempt
    """
    
    print("\n🚀 Starting Adaptive Retry Mechanism Workflow")
    print("-" * 50)
    
    try:
        # Run the adaptive workflow
        result = agent.run_react_workflow(
            workflow_input=workflow_input,
            conversation_id="test_adaptive_retry_mechanism"
        )
        
        print("\n📊 Adaptive Retry Result:")
        print("=" * 50)
        print(result)
        
        # Analyze the execution for retry patterns
        tool_log = agent.get_tool_execution_log()
        retry_analysis = analyze_retry_patterns(tool_log)
        print(f"\n🔄 Retry Pattern Analysis:")
        print(retry_analysis)
        
        # Get conversation log file
        log_file = agent.get_conversation_log_file()
        if log_file:
            print(f"\n📝 Conversation log saved to: {log_file}")
        
    except Exception as e:
        print(f"❌ Adaptive retry workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Adaptive retry mechanism test completed")

def analyze_retry_patterns(tool_log):
    """Analyze the tool execution log for retry patterns and strategies used."""
    analysis = {
        "total_attempts": 0,
        "strategies_used": [],
        "failures_analyzed": 0,
        "parameter_adjustments": [],
        "success_achieved": False,
        "final_strategy": None
    }
    
    attempts = []
    current_attempt = {}
    
    for entry in tool_log:
        tool_name = entry.get('tool', 'unknown')
        result = entry.get('result', {})
        error = entry.get('error')
        
        if 'homogeneous_refinement' in tool_name:
            analysis["total_attempts"] += 1
            current_attempt = {
                "attempt_number": analysis["total_attempts"],
                "parameters": entry.get('params', {}),
                "result": result,
                "error": error
            }
            attempts.append(current_attempt)
            
            # Analyze parameter strategy used
            params = entry.get('params', {})
            strategy = determine_strategy(params)
            analysis["strategies_used"].append(strategy)
            analysis["parameter_adjustments"].append({
                "attempt": analysis["total_attempts"],
                "strategy": strategy,
                "parameters": params
            })
            
            if result.get('success') == True:
                analysis["success_achieved"] = True
                analysis["final_strategy"] = strategy
        
        if 'get_job_log' in tool_name:
            analysis["failures_analyzed"] += 1
    
    # Generate analysis report
    report = f"""
   📊 Total Refinement Attempts: {analysis['total_attempts']}
   🔄 Strategies Used: {', '.join(set(analysis['strategies_used']))}
   📋 Failures Analyzed: {analysis['failures_analyzed']}
   ✅ Success Achieved: {'Yes' if analysis['success_achieved'] else 'No'}
   🎯 Final Strategy: {analysis['final_strategy'] or 'None'}
   
   📈 Retry Effectiveness:
   - Adaptive parameter adjustment: {'✅ Working' if len(set(analysis['strategies_used'])) > 1 else '❌ Limited'}
   - Error log analysis: {'✅ Active' if analysis['failures_analyzed'] > 0 else '❌ Not triggered'}
   - Multiple strategies: {'✅ Implemented' if len(set(analysis['strategies_used'])) >= 3 else '❌ Insufficient'}
   
   🔧 Parameter Evolution:
"""
    
    for i, adjustment in enumerate(analysis['parameter_adjustments'], 1):
        report += f"   {i}. {adjustment['strategy']}: {adjustment['parameters']}\n"
    
    return report

def determine_strategy(params):
    """Determine which strategy was used based on parameters."""
    refine_defocus = params.get('refine_defocus_refine', True)
    refine_ctf_global = params.get('refine_ctf_global_refine', True)
    refinement_resolution = params.get('refinement_resolution')
    symmetry = params.get('symmetry', 'C1')
    
    if not refine_defocus and not refine_ctf_global:
        return "CTF_Disabled"
    elif refinement_resolution == 15.0:
        return "Conservative_Resolution"
    elif refinement_resolution is None:
        return "Auto_Resolution"
    elif symmetry == 'C1' and not refine_defocus:
        return "Minimal_Parameters"
    else:
        return "Default_Parameters"

if __name__ == "__main__":
    test_adaptive_retry_mechanism()
