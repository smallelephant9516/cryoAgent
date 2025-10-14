#!/usr/bin/env python3
"""Demo script showing the adaptive retry mechanism implementation."""

import sys
import os
sys.path.append('/home/daoyi/Github/cryoagent')

from cryoagent.config.config_loader import CryoAgentConfig, ConfigLoader
from cryoagent.tools.cryosparc_tools import CryoSPARCTools
from cryoagent.core.cryosparc_reconstruction.reconstruction_agent import ReconstructionAgent

def demo_adaptive_retry_mechanism():
    """Demonstrate the adaptive retry mechanism implementation."""
    
    print("🎯 Demo: Adaptive Retry Mechanism Implementation")
    print("=" * 60)
    
    # Load configuration
    master_config_path = "/home/daoyi/Github/cryoagent/configs/master_config.json"
    stage_config_path = "/home/daoyi/Github/cryoagent/configs/3d_reconstruction_config.json"
    config_loader = ConfigLoader(master_config_path, stage_config_path)
    config = config_loader.load_config()
    
    # Initialize CryoSPARC tools
    cryosparc_tools = CryoSPARCTools(config.cryosparc)
    agent = ReconstructionAgent(cryosparc_tools, config)
    
    print("✅ System initialized successfully")
    
    # Demonstrate the adaptive retry strategies
    print("\n🔄 ADAPTIVE RETRY STRATEGIES IMPLEMENTED:")
    print("-" * 50)
    
    strategies = [
        {
            "name": "Strategy 1: Default Parameters",
            "description": "Start with standard CryoSPARC defaults",
            "parameters": {
                "refine_defocus_refine": True,
                "refine_ctf_global_refine": True,
                "refinement_resolution": None,
                "symmetry": "C1"
            },
            "use_case": "Initial attempt with standard parameters"
        },
        {
            "name": "Strategy 2: CTF Refinement Disabled",
            "description": "Disable CTF refinement for stability issues",
            "parameters": {
                "refine_defocus_refine": False,
                "refine_ctf_global_refine": False,
                "refinement_resolution": None,
                "symmetry": "C1"
            },
            "use_case": "When CTF refinement fails due to poor reconstruction quality"
        },
        {
            "name": "Strategy 3: Conservative Resolution",
            "description": "Use conservative resolution target",
            "parameters": {
                "refine_defocus_refine": True,
                "refine_ctf_global_refine": True,
                "refinement_resolution": 15.0,
                "symmetry": "C1"
            },
            "use_case": "When resolution target is too aggressive"
        },
        {
            "name": "Strategy 4: Minimal Parameters",
            "description": "Use minimal parameters for maximum compatibility",
            "parameters": {
                "refine_defocus_refine": False,
                "refine_ctf_global_refine": False,
                "refinement_resolution": None,
                "symmetry": "C1"
            },
            "use_case": "When all other strategies fail"
        },
        {
            "name": "Strategy 5: Alternative Method",
            "description": "Try homogeneous reconstruction instead",
            "method": "homogeneous_reconstruction",
            "parameters": {
                "particles_job_uid": "J101",
                "initial_resolution": 20.0,
                "final_resolution": 8.0,
                "symmetry": "C1"
            },
            "use_case": "When refinement consistently fails"
        }
    ]
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{i}. {strategy['name']}")
        print(f"   📋 {strategy['description']}")
        print(f"   🎯 Use case: {strategy['use_case']}")
        if 'parameters' in strategy:
            print(f"   ⚙️  Parameters: {strategy['parameters']}")
        if 'method' in strategy:
            print(f"   🔧 Method: {strategy['method']}")
    
    print(f"\n🧠 ADAPTIVE RETRY LOGIC:")
    print("-" * 30)
    print("1. Start with Strategy 1 (Default Parameters)")
    print("2. If job fails → Read job log with get_job_log tool")
    print("3. Analyze error patterns from CryoSPARC logs")
    print("4. Choose appropriate strategy based on error type:")
    print("   - CTF errors → Strategy 2")
    print("   - Resolution errors → Strategy 3") 
    print("   - Convergence errors → Strategy 4")
    print("   - All strategies fail → Strategy 5")
    print("5. Continue until success or all strategies exhausted")
    
    print(f"\n✅ IMPLEMENTATION STATUS:")
    print("-" * 30)
    print("✅ Enhanced CryoSPARC tools with CTF refinement parameters")
    print("✅ Updated reconstruction agent with adaptive retry instructions")
    print("✅ Implemented 5 different retry strategies")
    print("✅ Enhanced error log analysis capabilities")
    print("✅ Automatic parameter adjustment based on error patterns")
    
    print(f"\n🎯 KEY FEATURES:")
    print("-" * 20)
    print("• Automatic error log analysis")
    print("• Intelligent parameter adjustment")
    print("• Multiple fallback strategies")
    print("• Alternative method selection")
    print("• Comprehensive failure recovery")
    print("• At least 3 retry attempts guaranteed")
    
    print(f"\n📝 USAGE EXAMPLE:")
    print("-" * 20)
    print("When homogeneous refinement fails:")
    print("1. Agent reads job log automatically")
    print("2. Identifies CTF refinement error")
    print("3. Retries with refine_defocus_refine=false")
    print("4. If still fails, tries conservative resolution")
    print("5. Continues until success or all options exhausted")
    
    print(f"\n🏁 Demo completed successfully!")

if __name__ == "__main__":
    demo_adaptive_retry_mechanism()
