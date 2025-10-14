#!/usr/bin/env python3
"""Enhanced test script with automatic failure recovery for homogeneous refinement."""

import sys
import os
sys.path.append('/home/daoyi/Github/cryoagent')

from cryoagent.config.config_loader import CryoAgentConfig, ConfigLoader
from cryoagent.tools.cryosparc_tools import CryoSPARCTools
from cryoagent.core.cryosparc_reconstruction.reconstruction_agent import ReconstructionAgent

def test_enhanced_failure_recovery():
    """Test homogeneous refinement with automatic failure recovery and parameter adjustment."""
    
    print("🧪 Testing Enhanced Failure Recovery for Homogeneous Refinement")
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
    
    # Enhanced workflow input with automatic failure recovery
    workflow_input = """
    Run homogeneous refinement using the ab initio job J101 with automatic failure recovery.
    
    The ab initio job J101 should have completed and produced initial 3D model(s).
    
    CRITICAL REQUIREMENTS:
    1. First check the status of J101 to ensure it completed successfully
    2. If J101 is completed, start homogeneous refinement using J101 as both particles and volume source
    3. If the homogeneous refinement job FAILS:
       a. IMMEDIATELY read the job log using get_job_log tool
       b. Analyze the error patterns and identify the root cause
       c. Based on the error analysis, automatically retry with adjusted parameters:
          - If CTF refinement error: Disable CTF refinement (refine_defocus_refine=False, refine_ctf_global_refine=False)
          - If resolution error: Use more conservative resolution parameters
          - If convergence error: Increase max iterations or adjust symmetry
          - If parameter error: Use default CryoSPARC parameters
    4. Wait for the retry job to complete
    5. If retry also fails, try alternative approaches:
       - Try homogeneous reconstruction instead of refinement
       - Try ab initio reconstruction with different parameters
    6. Provide comprehensive analysis of all attempts and results
    
    AUTOMATIC FAILURE RECOVERY PROTOCOL:
    - NEVER give up after first failure
    - ALWAYS read job logs to understand the failure
    - ALWAYS retry with adjusted parameters based on error analysis
    - ALWAYS provide alternative approaches if retries fail
    - ALWAYS document the reasoning behind parameter adjustments
    
    Example parameter adjustments for common failures:
    - CTF refinement failure → Disable CTF refinement
    - Resolution too aggressive → Use auto-resolution or higher target resolution
    - Convergence failure → Increase iterations or change symmetry
    - Memory issues → Reduce batch size or use CPU instead of GPU
    """
    
    print("\n🚀 Starting Enhanced Failure Recovery Workflow")
    print("-" * 50)
    
    try:
        # Run the enhanced workflow with automatic failure recovery
        result = agent.run_react_workflow(
            workflow_input=workflow_input,
            conversation_id="test_enhanced_failure_recovery_j101"
        )
        
        print("\n📊 Enhanced Workflow Result:")
        print("=" * 50)
        print(result)
        
        # Get tool execution log for analysis
        tool_log = agent.get_tool_execution_log()
        print(f"\n🔧 Tool Execution Summary:")
        print(f"   Total tools executed: {len(tool_log)}")
        
        # Analyze the execution log for failure recovery patterns
        failure_recovery_analysis = analyze_failure_recovery(tool_log)
        print(f"\n🔄 Failure Recovery Analysis:")
        print(failure_recovery_analysis)
        
        # Get conversation log file
        log_file = agent.get_conversation_log_file()
        if log_file:
            print(f"\n📝 Conversation log saved to: {log_file}")
        
    except Exception as e:
        print(f"❌ Enhanced workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Enhanced failure recovery test completed")

def analyze_failure_recovery(tool_log):
    """Analyze the tool execution log for failure recovery patterns."""
    analysis = {
        "total_attempts": 0,
        "failures_detected": 0,
        "logs_analyzed": 0,
        "retries_attempted": 0,
        "parameter_adjustments": [],
        "alternative_approaches": [],
        "success_rate": 0
    }
    
    for entry in tool_log:
        tool_name = entry.get('tool', 'unknown')
        
        if 'homogeneous_refinement' in tool_name:
            analysis["total_attempts"] += 1
        
        if 'get_job_log' in tool_name:
            analysis["logs_analyzed"] += 1
        
        if 'wait_for_job' in tool_name:
            result = entry.get('result', {})
            if result.get('status') == 'failed':
                analysis["failures_detected"] += 1
        
        if 'error' in entry:
            analysis["failures_detected"] += 1
    
    # Calculate success rate
    if analysis["total_attempts"] > 0:
        successful_attempts = analysis["total_attempts"] - analysis["failures_detected"]
        analysis["success_rate"] = (successful_attempts / analysis["total_attempts"]) * 100
    
    # Generate analysis report
    report = f"""
   📊 Total Refinement Attempts: {analysis['total_attempts']}
   ❌ Failures Detected: {analysis['failures_detected']}
   📋 Logs Analyzed: {analysis['logs_analyzed']}
   🔄 Retries Attempted: {analysis['retries_attempted']}
   ✅ Success Rate: {analysis['success_rate']:.1f}%
   
   🧠 Failure Recovery Effectiveness:
   - Automatic log analysis: {'✅ Working' if analysis['logs_analyzed'] > 0 else '❌ Not triggered'}
   - Parameter adjustment: {'✅ Implemented' if analysis['parameter_adjustments'] else '❌ Not detected'}
   - Alternative approaches: {'✅ Attempted' if analysis['alternative_approaches'] else '❌ Not tried'}
"""
    
    return report

if __name__ == "__main__":
    test_enhanced_failure_recovery()
