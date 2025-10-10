#!/usr/bin/env python3
"""
Example script demonstrating the modular agent architecture for CryoAgent.

This script shows how to use the new modular agents:
1. PreprocessingAgent - handles import, motion correction, CTF estimation, and micrograph selection
2. PickingAgent - handles particle detection using blob picker

The modular design provides better code organization, reusability, and separation of concerns.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryoagent.config.config_loader import ConfigLoader
from cryoagent.tools.cryosparc_tools import CryoSPARCTools
from cryoagent.core.cryosparc_preprocessing import PreprocessingAgent, PreprocessingWorkflow
from cryoagent.core.cryosparc_picking import PickingAgent, PickingWorkflow


def main():
    """Main function demonstrating modular agent usage."""
    
    print("=" * 80)
    print("CryoAgent Modular Architecture Example")
    print("=" * 80)
    print()
    
    # Load configuration
    print("📋 Loading configuration...")
    config_loader = ConfigLoader("configs/master_config.json")
    config = config_loader.load_config()
    print(f"✅ Configuration loaded successfully")
    print(f"   Provider: {config.agent.provider}")
    print(f"   Model: {config.agent.get_current_model_config().model_name}")
    print()
    
    # Initialize CryoSPARC tools
    print("🔌 Connecting to CryoSPARC...")
    try:
        cryosparc_tools = CryoSPARCTools(config.cryosparc)
        print("✅ Connected to CryoSPARC successfully")
        print()
    except Exception as e:
        print(f"❌ Failed to connect to CryoSPARC: {e}")
        return 1
    
    # ========================================================================
    # Stage 1: Preprocessing with PreprocessingAgent
    # ========================================================================
    print("=" * 80)
    print("STAGE 1: Preprocessing")
    print("=" * 80)
    print()
    
    print("🤖 Initializing Preprocessing Agent...")
    preprocessing_agent = PreprocessingAgent(cryosparc_tools, config)
    preprocessing_workflow = PreprocessingWorkflow(preprocessing_agent, config)
    print("✅ Preprocessing agent initialized")
    print()
    
    print("🚀 Running preprocessing workflow...")
    print("   This will execute: Import → Motion Correction → CTF Estimation → Micrograph Selection")
    print()
    
    try:
        # Run the preprocessing workflow
        preprocessing_results = preprocessing_workflow.run(conversation_id="preprocessing_run_1")
        
        # Display results
        print()
        print("📊 Preprocessing Results:")
        print("-" * 80)
        for result in preprocessing_results:
            status_icon = "✅" if result.success else "❌"
            print(f"{status_icon} {result.step.value}:")
            print(f"   Job UID: {result.job_uid}")
            print(f"   Message: {result.message}")
            if result.error:
                print(f"   Error: {result.error}")
            print()
        
        # Get workflow summary
        summary = preprocessing_workflow.get_workflow_summary()
        print(f"Total steps: {summary['total_steps']}")
        print(f"Successful: {summary['successful_steps']}")
        print(f"Failed: {summary['failed_steps']}")
        print()
        
        # Check if preprocessing was successful
        if summary['failed_steps'] > 0:
            print("❌ Preprocessing failed. Cannot proceed to particle picking.")
            return 1
        
        # Get the micrograph selection job UID for next stage
        micrograph_job_uid = None
        for result in preprocessing_results:
            if result.step.value == "micrograph_selection" and result.success:
                micrograph_job_uid = result.job_uid
                break
        
        if not micrograph_job_uid:
            print("❌ Could not find micrograph selection job UID. Cannot proceed to particle picking.")
            return 1
        
        print(f"✅ Preprocessing completed successfully!")
        print(f"   Micrograph selection job: {micrograph_job_uid}")
        print()
        
    except Exception as e:
        print(f"❌ Preprocessing workflow failed: {e}")
        return 1
    
    # ========================================================================
    # Stage 2: Particle Picking with PickingAgent
    # ========================================================================
    print("=" * 80)
    print("STAGE 2: Particle Picking")
    print("=" * 80)
    print()
    
    print("🤖 Initializing Particle Picking Agent...")
    picking_agent = PickingAgent(cryosparc_tools, config)
    picking_workflow = PickingWorkflow(picking_agent, config)
    print("✅ Particle picking agent initialized")
    print()
    
    # Set particle picking parameters
    particle_diameter = 180.0  # Angstroms
    min_separation = None  # Will default to 0.8 * diameter
    
    print(f"🚀 Running particle picking workflow...")
    print(f"   Input micrographs: {micrograph_job_uid}")
    print(f"   Particle diameter: {particle_diameter} Å")
    print(f"   Min separation: {'Auto (0.8 × diameter)' if min_separation is None else f'{min_separation} Å'}")
    print()
    
    try:
        # Run the particle picking workflow
        picking_results = picking_workflow.run(
            micrographs_job_uid=micrograph_job_uid,
            particle_diameter=particle_diameter,
            min_separation=min_separation,
            conversation_id="picking_run_1"
        )
        
        # Display results
        print()
        print("📊 Particle Picking Results:")
        print("-" * 80)
        for result in picking_results:
            status_icon = "✅" if result.success else "❌"
            print(f"{status_icon} {result.step.value}:")
            print(f"   Job UID: {result.job_uid}")
            print(f"   Message: {result.message}")
            if result.error:
                print(f"   Error: {result.error}")
            print()
        
        # Get workflow summary
        summary = picking_workflow.get_workflow_summary()
        print(f"Total steps: {summary['total_steps']}")
        print(f"Successful: {summary['successful_steps']}")
        print(f"Failed: {summary['failed_steps']}")
        print()
        
        if summary['failed_steps'] > 0:
            print("❌ Particle picking failed.")
            return 1
        
        print("✅ Particle picking completed successfully!")
        
        # Get the blob picker job UID
        picker_job_uid = None
        for result in picking_results:
            if result.step.value == "blob_picker" and result.success:
                picker_job_uid = result.job_uid
                break
        
        if picker_job_uid:
            print(f"   Blob picker job: {picker_job_uid}")
        print()
        
    except Exception as e:
        print(f"❌ Particle picking workflow failed: {e}")
        return 1
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print()
    print("✅ Both preprocessing and particle picking stages completed successfully!")
    print()
    print("📁 Output Summary:")
    print(f"   - Micrograph selection: {micrograph_job_uid}")
    print(f"   - Particle picking: {picker_job_uid}")
    print()
    print("Next steps:")
    print("   - Review picked particles in CryoSPARC")
    print("   - Proceed with 2D classification or 3D reconstruction")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

