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
import json
import glob
from pathlib import Path
from typing import List, Optional, Dict, Any

from cryoagent.core.master_orchestrator import (
    MasterOrchestrator,
    WorkflowStage
)
from cryoagent.utils.general_llm_logger import GeneralLLMLogger


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


def check_stage_output_exists(stage: WorkflowStage, outputs_dir: Optional[str] = None, default_outputs_dir: str = "outputs") -> Optional[Dict[str, Any]]:
    """
    Check if output file for a given stage already exists.
    
    Args:
        stage: The workflow stage to check
        outputs_dir: Custom directory where output files are stored. When provided,
                     only this directory is checked (no fallback to default).
        default_outputs_dir: Default directory to check when outputs_dir is not provided
        
    Returns:
        Dictionary with output file information if exists, None otherwise
    """
    # When a custom outputs directory is explicitly provided, only check that directory
    if outputs_dir:
        outputs_path = Path(outputs_dir)
        if not outputs_path.exists():
            return None
        return _check_output_in_directory(stage, outputs_path)
    
    # No custom dir specified: check the default outputs directory
    outputs_path = Path(default_outputs_dir)
    if not outputs_path.exists():
        return None
    
    return _check_output_in_directory(stage, outputs_path)


def _check_output_in_directory(stage: WorkflowStage, outputs_path: Path) -> Optional[Dict[str, Any]]:
    """Helper function to check for outputs in a specific directory."""
    
    # Map stage names to output file patterns
    stage_patterns = {
        WorkflowStage.PREPROCESSING: "preprocessing_results_*.json",
        WorkflowStage.PARTICLE_PICKING: "particle_picking_results_*.json",
        WorkflowStage.OPTIMIZATION_2D: "2d_optimization_results_*.json",
        WorkflowStage.RECONSTRUCTION: "reconstruction_results_*.json"
    }
    
    pattern = stage_patterns.get(stage)
    if not pattern:
        return None
    
    # Search for matching output files
    search_pattern = str(outputs_path / pattern)
    matching_files = glob.glob(search_pattern)
    
    if not matching_files:
        return None
    
    # Get the most recent file (sorted by modification time)
    latest_file = max(matching_files, key=lambda f: Path(f).stat().st_mtime)
    
    try:
        # Read and validate the output file
        with open(latest_file, 'r') as f:
            output_data = json.load(f)
        
        # Check if the stage was completed successfully
        if output_data.get("status") == "completed":
            return {
                "file_path": latest_file,
                "timestamp": output_data.get("timestamp"),
                "status": output_data.get("status"),
                "project_uid": output_data.get("project_uid"),
                "workspace_uid": output_data.get("workspace_uid"),
                "data": output_data
            }
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to read output file {latest_file}: {e}")
        return None
    
    return None


class CryoAgentMasterWorkflow:
    """Main master workflow orchestrator for CryoAgent."""
    
    def __init__(self, master_config_path: str = "configs/master_config.json", outputs_dir: str = "outputs"):
        """
        Initialize the CryoAgent master workflow.
        
        Args:
            master_config_path: Path to the master configuration file
            outputs_dir: Directory where output files will be saved
        """
        self.master_config_path = master_config_path
        self.outputs_dir = outputs_dir
        self.orchestrator = None
        self.start_time = None
        self.llm_logger = GeneralLLMLogger(outputs_dir=outputs_dir)
        
    def initialize(self, force_all_stages: bool = False) -> bool:
        """
        Initialize all components.

        Args:
            force_all_stages: When True, initialize every stage agent regardless
                of its ``enabled`` flag (used by full_dynamic mode so the single
                agent has the full cross-stage tool set).

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("🚀 Initializing CryoAgent Master Workflow")
            print("=" * 60)
            
            # Initialize master orchestrator
            print("🎭 Initializing master orchestrator...")
            self.orchestrator = MasterOrchestrator(self.master_config_path, outputs_dir=self.outputs_dir)
            
            if not self.orchestrator.initialize(force_all_stages=force_all_stages):
                print("❌ Failed to initialize master orchestrator")
                return False
            
            # Set the general LLM logger for all stage agents
            self.orchestrator.set_general_llm_logger(self.llm_logger)
            
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
    
    def run_complete_workflow(self, dry_run: bool = False, conversation_id: Optional[str] = None, mode: str = "guided", goal: Optional[str] = None, improve: bool = False) -> bool:
        """
        Run the complete cryoEM workflow.

        Args:
            dry_run: If True, show what would be done without executing
            conversation_id: Optional conversation ID for tracking
            mode: "guided" (configured stage order, LLM re-plans on failure) or
                "dynamic" (LLM planner picks each next stage from prior JSON outputs)
            goal: Optional high-level goal statement for the planner

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
            # Start LLM conversation logging
            llm_log_file = self.llm_logger.start_workflow_log(conversation_id)
            print(f"💬 LLM conversation log: {llm_log_file}")

            # Dynamic mode: the LLM planner chooses each next stage from prior
            # stages' JSON outputs; skip the fixed pre-flight stage-skip check.
            if mode == "dynamic":
                print("🤖 Starting DYNAMIC CryoEM Workflow (LLM-planned stage order)")
                print("=" * 70)
                self.start_time = time.time()
                result = self.orchestrator.execute_complete_workflow(
                    conversation_id, mode="dynamic", goal=goal
                )
                success = result.get('successful_stages', 0) > 0 and 'error' not in result
                if success:
                    print("🎉 Dynamic workflow finished.")
                    self.llm_logger.end_workflow_log(True, "Dynamic workflow finished")
                else:
                    print("❌ Dynamic workflow did not complete any stage successfully.")
                    self.llm_logger.end_workflow_log(False, "Dynamic workflow failed")
                if 'final_report_path' in result:
                    print(f"\n📊 Workflow Report: {result['final_report_path']}")
                if success and improve:
                    print("\n🔬 Running dynamic improvement on the completed run...")
                    try:
                        imp = self.orchestrator.execute_improvement(conversation_id=conversation_id, goal=goal)
                        if imp.get("success"):
                            print("✅ Improvement phase finished.")
                            print(imp.get("output", ""))
                        else:
                            print(f"⚠️ Improvement phase did not run: {imp.get('error')}")
                    except Exception as e:
                        print(f"⚠️ Improvement phase error: {e}")
                return success

            # Full-dynamic mode: a single from-scratch agent with the full tool
            # set drives the whole run, guided only by tool descriptions + the
            # CryoSPARC guide. Skip the fixed pre-flight stage-skip checks.
            if mode == "full_dynamic":
                print("🤖 Starting FULL-DYNAMIC CryoEM Workflow (single from-scratch agent)")
                print("=" * 70)
                self.start_time = time.time()
                result = self.orchestrator.execute_complete_workflow(
                    conversation_id, mode="full_dynamic", goal=goal
                )
                success = result.get('successful_stages', 0) > 0 and 'error' not in result
                if success:
                    print("🎉 Full-dynamic run finished.")
                    print(result.get("output", ""))
                    self.llm_logger.end_workflow_log(True, "Full-dynamic run finished")
                else:
                    print(f"❌ Full-dynamic run did not complete: {result.get('error', 'unknown error')}")
                    self.llm_logger.end_workflow_log(False, "Full-dynamic run failed")
                return success

            print("🎯 Starting Complete CryoEM Workflow (guided mode)")
            print("=" * 70)
            print("This will execute the enabled stages in configured order,")
            print("re-planning with the LLM only when a stage fails.")
            print()
            
            # Check which stages have already been completed
            # Only check enabled stages from config
            print("🔍 Checking for existing stage outputs...")
            stages_to_skip = []
            enabled_stages = []
            
            # Get enabled stages from master config
            if self.orchestrator and hasattr(self.orchestrator, 'master_config'):
                master_config = self.orchestrator.master_config
                for stage_info in master_config.get("master_workflow", {}).get("stages", []):
                    if stage_info.get("enabled", False):
                        try:
                            stage = WorkflowStage(stage_info["name"])
                            enabled_stages.append(stage)
                        except ValueError:
                            pass
            
            # Fallback to all stages if config not available
            if not enabled_stages:
                enabled_stages = list(WorkflowStage)
            
            for stage in enabled_stages:
                existing_output = check_stage_output_exists(stage, outputs_dir=self.outputs_dir)
                if existing_output:
                    stages_to_skip.append(stage)
                    print(f"   ✅ {stage.value.replace('_', ' ').title()}: Already completed ({existing_output['timestamp']})")
                else:
                    print(f"   ⏳ {stage.value.replace('_', ' ').title()}: Will be executed")
            
            if stages_to_skip:
                print()
                print("ℹ️  Some stages already completed and will be skipped.")
                print(f"💡 To re-run all stages, delete or move output files in the {self.outputs_dir}/ folder.")
                print()
            
            if len(stages_to_skip) == len(enabled_stages):
                print("✅ All enabled stages already completed! No execution needed.")
                return True
            
            self.start_time = time.time()

            # Execute complete workflow
            result = self.orchestrator.execute_complete_workflow(conversation_id, mode="guided", goal=goal)
            
            # Check if workflow completed successfully
            success = result['successful_stages'] == result['total_stages']
            
            if success:
                print("🎉 Complete workflow executed successfully!")
                print("   ✅ All stages completed with proper monitoring")
                print("   🔗 Stage dependencies properly handled")
                print("   ⏱️ Each stage waited for completion before claiming success")
                
                # Display final report information if available
                if 'final_report_path' in result:
                    print(f"\n📊 Final Workflow Report:")
                    print(f"   JSON: {result['final_report_path']}")
                    md_report = result['final_report_path'].replace('.json', '.md')
                    print(f"   Markdown: {md_report}")
                
                self.llm_logger.end_workflow_log(True, "Complete workflow executed successfully")
            else:
                print("❌ Complete workflow failed!")
                print("   ⚠️ One or more stages did not complete successfully")
                
                # Display final report information even if workflow failed
                if 'final_report_path' in result:
                    print(f"\n📊 Workflow Report (partial):")
                    print(f"   JSON: {result['final_report_path']}")
                    md_report = result['final_report_path'].replace('.json', '.md')
                    print(f"   Markdown: {md_report}")
                
                self.llm_logger.end_workflow_log(False, "Workflow failed - one or more stages did not complete")

            # Opt-in dynamic improvement: only after a successful guided run.
            if success and improve:
                print("\n🔬 Running dynamic improvement on the completed run...")
                try:
                    imp = self.orchestrator.execute_improvement(conversation_id=conversation_id, goal=goal)
                    if imp.get("success"):
                        print("✅ Improvement phase finished.")
                        print(imp.get("output", ""))
                    else:
                        print(f"⚠️ Improvement phase did not run: {imp.get('error')}")
                except Exception as e:
                    print(f"⚠️ Improvement phase error: {e}")

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
            # Check if preprocessing output already exists
            existing_output = check_stage_output_exists(WorkflowStage.PREPROCESSING, outputs_dir=self.outputs_dir)
            
            if existing_output:
                print("✅ Pre-processing stage already completed!")
                print("=" * 50)
                print(f"📄 Output file found: {existing_output['file_path']}")
                print(f"📅 Completed at: {existing_output['timestamp']}")
                print(f"🎯 Status: {existing_output['status']}")
                print(f"📦 Project: {existing_output['project_uid']}, Workspace: {existing_output['workspace_uid']}")
                print()
                print("ℹ️  Skipping pre-processing stage to avoid re-running.")
                print(f"💡 To re-run, delete or move the output file in the {self.outputs_dir}/ folder.")
                return True
            
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
            
            # Check which stages have already been completed
            print("🔍 Checking for existing stage outputs...")
            stages_to_skip = []
            stages_to_run = []
            for stage in workflow_stages:
                existing_output = check_stage_output_exists(stage, outputs_dir=self.outputs_dir)
                if existing_output:
                    stages_to_skip.append(stage)
                    print(f"   ✅ {stage.value.replace('_', ' ').title()}: Already completed ({existing_output['timestamp']})")
                else:
                    stages_to_run.append(stage)
                    print(f"   ⏳ {stage.value.replace('_', ' ').title()}: Will be executed")
            
            if stages_to_skip:
                print()
                print("ℹ️  Some stages already completed and will be skipped.")
                print(f"💡 To re-run these stages, delete or move output files in the {self.outputs_dir}/ folder.")
                print()
            
            if len(stages_to_skip) == len(workflow_stages):
                print("✅ All requested stages already completed! No execution needed.")
                return True
            
            if not stages_to_run:
                print("✅ No stages to execute.")
                return True
            
            self.start_time = time.time()
            
            # IMPORTANT: Pass ALL requested stages to orchestrator, not just stages_to_run
            # The orchestrator will handle skipping and loading cached outputs correctly
            # This ensures that dependent stages have access to previous stage outputs
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

  # Interactive agentic GUI (chat to shape and run the plan):
  streamlit run cryoagent_gui.py
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
    
    parser.add_argument(
        "--conversation-id",
        help="Optional custom conversation ID for this workflow run (default: auto-generated)"
    )
    
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Directory where output files will be saved (default: outputs)"
    )

    parser.add_argument(
        "--mode",
        choices=["guided", "dynamic", "full_dynamic"],
        default="guided",
        help="Workflow control mode for the complete workflow: 'guided' runs the "
             "configured stage order and lets the LLM re-plan only on failure; "
             "'dynamic' lets the LLM planner choose each next stage from prior "
             "stages' JSON outputs; 'full_dynamic' runs a single from-scratch agent "
             "with the full tool set that drives the entire micrograph -> 3D density "
             "run itself, guided only by tool descriptions and the CryoSPARC guide "
             "(it sees only microscope_config.json params + the session project/"
             "workspace ids) (default: guided)."
    )

    parser.add_argument(
        "--improve",
        action="store_true",
        help="After the workflow completes, run the dynamic improvement agent: it "
             "reads the blackboard (all stages' real metrics), diagnoses the "
             "limiting factor and root cause across stages, and acts with atomic "
             "tools to further improve the result (reusing prior good jobs), "
             "stopping when no significant FSC/cFAR gain remains."
    )

    parser.add_argument(
        "--goal",
        default=None,
        help="Optional high-level goal statement passed to the workflow planner."
    )

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
            # Initialize master workflow
    master_workflow = CryoAgentMasterWorkflow(args.config, outputs_dir=args.outputs_dir)
    
    # full_dynamic mode needs every stage agent initialized so the single agent
    # has the full cross-stage tool set, regardless of stage enabled flags.
    force_all_stages = (getattr(args, "mode", "guided") == "full_dynamic"
                        and args.workflow == "complete")
    if not master_workflow.initialize(force_all_stages=force_all_stages):
        print("❌ Failed to initialize CryoAgent master workflow")
        sys.exit(1)
    
    # Execute workflow based on type
    success = False
    
    try:
        if args.workflow == "test":
            success = master_workflow.test_setup()
            
        elif args.workflow == "complete":
            # Use provided conversation ID or generate a unique one to ensure fresh start
            conversation_id = args.conversation_id or f"complete_workflow_{int(time.time())}"
            success = master_workflow.run_complete_workflow(args.dry_run, conversation_id, mode=args.mode, goal=args.goal, improve=args.improve)
            
        elif args.workflow == "preprocessing":
            # Use provided conversation ID or generate a unique one to ensure fresh start
            conversation_id = args.conversation_id or f"preprocessing_workflow_{int(time.time())}"
            success = master_workflow.run_preprocessing_workflow(args.dry_run, conversation_id)

        elif args.workflow == "custom":
            if not args.stages:
                print("❌ --workflow custom requires --stages (comma-separated stage names)")
                sys.exit(1)
            stage_list = [s.strip() for s in args.stages.split(",") if s.strip()]
            conversation_id = args.conversation_id or f"custom_workflow_{int(time.time())}"
            success = master_workflow.run_custom_workflow(stage_list, args.dry_run, conversation_id)

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
