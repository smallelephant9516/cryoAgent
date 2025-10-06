#!/usr/bin/env python3
"""
CryoAgent Master Workflow - Multi-Stage CryoEM Processing Pipeline

This script provides a comprehensive agentic workflow for the complete cryoEM processing
pipeline using separate specialized ReAct agents for each stage.

Architecture:
- Master Orchestrator coordinates separate stage agents
- Each stage has its own configuration file and ReAct agent
- Stage 1 (Pre-processing): Fully implemented
- Stage 2 (Particle Picking): Placeholder for future implementation
- Stage 3 (3D Reconstruction): Placeholder for future implementation

Usage:
    python cryoagent_master_workflow.py [options]

Options:
    --config CONFIG_FILE    Path to master configuration file (default: configs/master_config.json)
    --workflow WORKFLOW     Workflow type: complete, preprocessing, custom (default: complete)
    --stages STAGES         Comma-separated list of stages for custom workflow
    --verbose               Enable verbose output
    --dry-run               Show what would be done without executing
    --test                  Run test mode to verify setup
"""

import sys
import argparse
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from cryoagent.core.master_orchestrator import (
    MasterOrchestrator,
    WorkflowStage
)


def setup_logging(verbose: bool = False):
    """Setup logging for the master workflow."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('cryoagent_master.log')
        ]
    )


class CryoAgentMasterWorkflow:
    """Main master workflow orchestrator for CryoAgent."""
    
    def __init__(self, master_config_path: str = "configs/master_config.json"):
        """
        Initialize the CryoAgent master workflow.
        
        Args:
            master_config_path: Path to the master configuration file
        """
        self.master_config_path = master_config_path
        self.orchestrator = None
        self.start_time = None
        
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("🚀 Initializing CryoAgent Master Workflow")
            print("=" * 60)
            
            # Initialize master orchestrator
            print("🎭 Initializing master orchestrator...")
            self.orchestrator = MasterOrchestrator(self.master_config_path)
            
            if not self.orchestrator.initialize():
                print("❌ Failed to initialize master orchestrator")
                return False
            
            print("✅ Master orchestrator initialized")
            
            # Display configuration summary
            self._display_config_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _display_config_summary(self):
        """Display a summary of the current configuration."""
        print("\n📊 Configuration Summary:")
        print(f"   Master Config: {self.master_config_path}")
        
        # Display available stages
        stages = self.orchestrator.master_config["master_workflow"]["stages"]
        print(f"   Available Stages: {len(stages)}")
        for stage in stages:
            status = "✅ ENABLED" if stage["enabled"] else "🚧 DISABLED"
            print(f"     - {stage['name']}: {stage['description']} {status}")
        
        print()
    
    def run_complete_workflow(self, dry_run: bool = False, conversation_id: Optional[str] = None) -> bool:
        """
        Run the complete 3-stage cryoEM workflow.
        
        Args:
            dry_run: If True, show what would be done without executing
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        if dry_run:
            print("🔍 DRY RUN: Complete Workflow")
            stages = [stage.value.replace('_', ' ').title() for stage in WorkflowStage]
            workflow_stages = " → ".join(stages)
            print(f"Would execute: {workflow_stages}")
            return True
        
        try:
            print("🎯 Starting Complete 3-Stage CryoEM Workflow")
            print("=" * 70)
            print("This will execute all three stages:")
            print("1. Pre-processing (Import → Motion Correction → CTF → Selection)")
            print("2. Particle Picking (Detection → Extraction → Quality Assessment)")
            print("3. 3D Reconstruction (Initial Model → Refinement → Validation)")
            print()
            
            self.start_time = time.time()
            
            # Execute complete workflow
            result = self.orchestrator.execute_complete_workflow(conversation_id)
            
            # Check if workflow completed successfully
            success = result['successful_stages'] == result['total_stages']
            
            if success:
                print("🎉 Complete workflow executed successfully!")
                print("   ✅ All stages completed with proper monitoring")
                print("   🔗 Stage dependencies properly handled")
                print("   ⏱️ Each stage waited for completion before claiming success")
            else:
                print("❌ Complete workflow failed!")
                print("   ⚠️ One or more stages did not complete successfully")
            
            return success
            
        except Exception as e:
            print(f"❌ Complete workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_preprocessing_workflow(self, dry_run: bool = False, conversation_id: Optional[str] = None) -> bool:
        """
        Run only the pre-processing stage workflow.
        
        Args:
            dry_run: If True, show what would be done without executing
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        if dry_run:
            print("🔍 DRY RUN: Pre-processing Workflow")
            print("Would execute: Pre-processing (Import → Motion Correction → CTF → Selection)")
            return True
        
        try:
            print("🎯 Starting Pre-processing Workflow")
            print("=" * 50)
            print("This will execute the pre-processing stage:")
            print("1. Import Movies")
            print("2. Motion Correction")
            print("3. CTF Estimation")
            print("4. Micrograph Selection")
            print()
            
            self.start_time = time.time()
            
            # Execute pre-processing stage only
            stages = [WorkflowStage.PREPROCESSING]
            result = self.orchestrator.execute_stage_workflow(stages, conversation_id)
            
            # Check if workflow completed successfully
            success = result['successful_stages'] == result['total_stages']
            
            if success:
                print("🎉 Pre-processing workflow completed successfully!")
                print("   ✅ All pre-processing steps completed")
                print("   🔗 Step dependencies properly handled")
                print("   ⏱️ Each step waited for completion before claiming success")
            else:
                print("❌ Pre-processing workflow failed!")
                print("   ⚠️ One or more steps did not complete successfully")
            
            return success
            
        except Exception as e:
            print(f"❌ Pre-processing workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_custom_workflow(self, stages: List[str], dry_run: bool = False, conversation_id: Optional[str] = None) -> bool:
        """
        Run a custom workflow with specified stages.
        
        Args:
            stages: List of stage names to execute
            dry_run: If True, show what would be done without executing
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        if dry_run:
            print(f"🔍 DRY RUN: Custom Workflow")
            print(f"Would execute: {' → '.join(stages)}")
            return True
        
        try:
            print("🎯 Starting Custom CryoEM Workflow")
            print("=" * 50)
            print(f"Stages: {' → '.join(stages)}")
            print()
            
            # Convert stage names to WorkflowStage enums
            workflow_stages = []
            for stage_name in stages:
                try:
                    stage = WorkflowStage(stage_name.lower())
                    workflow_stages.append(stage)
                except ValueError:
                    print(f"❌ Invalid stage: {stage_name}")
                    print(f"   Valid stages: {[s.value for s in WorkflowStage]}")
                    return False
            
            self.start_time = time.time()
            
            # Execute custom workflow
            result = self.orchestrator.execute_stage_workflow(workflow_stages, conversation_id)
            
            # Check if workflow completed successfully
            success = result['successful_stages'] == result['total_stages']
            
            if success:
                print("🎉 Custom workflow completed successfully!")
            else:
                print("❌ Custom workflow failed!")
                print("   ⚠️ One or more stages did not complete successfully")
            
            return success
            
        except Exception as e:
            print(f"❌ Custom workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_setup(self) -> bool:
        """
        Test the setup and configuration.
        
        Returns:
            True if setup is correct, False otherwise
        """
        try:
            print("🧪 Testing CryoAgent Master Workflow Setup")
            print("=" * 60)
            
            # Test stage agents
            print("🤖 Testing Stage Agents...")
            for stage_name, agent in self.orchestrator.stage_agents.items():
                print(f"   ✅ {stage_name}: {agent.get_stage_description()}")
                print(f"      Required inputs: {agent.get_required_inputs()}")
                print(f"      Config file: {agent.config_path}")
            
            print("\n🎉 All tests passed! Master workflow is ready to use.")
            return True
            
        except Exception as e:
            print(f"❌ Setup test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        if self.orchestrator:
            return self.orchestrator.get_workflow_status()
        else:
            return {"status": "not_initialized"}


def main():
    """Main function to run the CryoAgent master workflow."""
    parser = argparse.ArgumentParser(
        description="CryoAgent Master Workflow - Multi-Stage CryoEM Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete workflow (all 3 stages)
  python cryoagent_master_workflow.py

  # Run pre-processing only
  python cryoagent_master_workflow.py --workflow preprocessing

  # Run custom stages
  python cryoagent_master_workflow.py --workflow custom --stages preprocessing,particle_picking

  # Test setup
  python cryoagent_master_workflow.py --workflow test

  # Dry run
  python cryoagent_master_workflow.py --dry-run

  # Verbose mode
  python cryoagent_master_workflow.py --verbose
        """
    )
    
    parser.add_argument(
        "--config", 
        default="configs/master_config.json",
        help="Path to master configuration file (default: configs/master_config.json)"
    )
    
    parser.add_argument(
        "--workflow",
        choices=["complete", "preprocessing", "custom", "test"],
        default="complete",
        help="Workflow type (default: complete)"
    )
    
    parser.add_argument(
        "--stages",
        help="Comma-separated list of stages for custom workflow (preprocessing,particle_picking,reconstruction)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Initialize master workflow
    master_workflow = CryoAgentMasterWorkflow(args.config)
    
    if not master_workflow.initialize():
        print("❌ Failed to initialize CryoAgent master workflow")
        sys.exit(1)
    
    # Execute workflow based on type
    success = False
    
    try:
        if args.workflow == "test":
            success = master_workflow.test_setup()
            
        elif args.workflow == "complete":
            # Use a unique conversation ID to ensure fresh start
            conversation_id = f"complete_workflow_{int(time.time())}"
            success = master_workflow.run_complete_workflow(args.dry_run, conversation_id)
            
        elif args.workflow == "preprocessing":
            # Use a unique conversation ID to ensure fresh start
            conversation_id = f"preprocessing_workflow_{int(time.time())}"
            success = master_workflow.run_preprocessing_workflow(args.dry_run, conversation_id)
            
        elif args.workflow == "custom":
            if not args.stages:
                print("❌ --stages required for custom workflow")
                print("   Valid stages: preprocessing, particle_picking, reconstruction")
                sys.exit(1)
            stages = [s.strip() for s in args.stages.split(",")]
            # Use a unique conversation ID to ensure fresh start
            conversation_id = f"custom_workflow_{int(time.time())}"
            success = master_workflow.run_custom_workflow(stages, args.dry_run, conversation_id)
        
        # Exit with appropriate code
        if success:
            print("\n🎉 CryoAgent master workflow completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ CryoAgent master workflow failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
