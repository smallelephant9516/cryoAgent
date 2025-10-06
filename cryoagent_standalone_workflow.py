#!/usr/bin/env python3
"""
CryoAgent Standalone Workflow

This module provides standalone testing capabilities for individual workflow agents.
It allows testing specific agents (preprocessing, particle_picking, reconstruction) 
independently without running the full master workflow.

Usage:
    python cryoagent_standalone_workflow.py --agent preprocessing [--dry-run]
    python cryoagent_standalone_workflow.py --agent particle_picking [--dry-run]
    python cryoagent_standalone_workflow.py --agent reconstruction [--dry-run]
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from cryoagent.core.master_orchestrator import (
    PreprocessingAgent, 
    ParticlePickingAgent, 
    ReconstructionAgent,
    WorkflowContext,
    WorkflowStage
)
from cryoagent.config.config_loader import ConfigLoader


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("cryoagent_standalone.log"),
            logging.StreamHandler()
        ]
    )


class CryoAgentStandaloneWorkflow:
    """Standalone workflow orchestrator for testing individual agents."""
    
    def __init__(self, master_config_path: str = "configs/master_config.json"):
        """
        Initialize the standalone workflow.
        
        Args:
            master_config_path: Path to the master configuration file
        """
        self.master_config_path = master_config_path
        self.master_config = None
        self.agent = None
        self.agent_type = None
        self.start_time = None
        self.logger = logging.getLogger("StandaloneWorkflow")
        
    def initialize(self) -> bool:
        """
        Initialize the standalone workflow.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("🚀 Initializing CryoAgent Standalone Workflow")
            print("=" * 60)
            
            # Load master configuration
            print("📋 Loading master configuration...")
            config_loader = ConfigLoader(self.master_config_path)
            self.master_config = config_loader.load_config()
            print(f"✅ Master configuration loaded from {self.master_config_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize standalone workflow: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def initialize_agent(self, agent_type: str) -> bool:
        """
        Initialize a specific agent type.
        
        Args:
            agent_type: Type of agent to initialize (preprocessing, particle_picking, reconstruction)
            
        Returns:
            True if agent initialized successfully, False otherwise
        """
        try:
            print(f"🤖 Initializing {agent_type} agent...")
            
            # Determine agent configuration file
            agent_configs = {
                "preprocessing": "configs/preprocessing_config.json",
                "particle_picking": "configs/particle_picking_config.json", 
                "reconstruction": "configs/reconstruction_config.json"
            }
            
            if agent_type not in agent_configs:
                print(f"❌ Unknown agent type: {agent_type}")
                print(f"   Valid types: {list(agent_configs.keys())}")
                return False
            
            config_path = agent_configs[agent_type]
            
            # Create appropriate agent
            if agent_type == "preprocessing":
                self.agent = PreprocessingAgent(config_path)
            elif agent_type == "particle_picking":
                self.agent = ParticlePickingAgent(config_path)
            elif agent_type == "reconstruction":
                self.agent = ReconstructionAgent(config_path)
            
            # Initialize the agent
            if self.agent.initialize():
                self.agent_type = agent_type
                print(f"✅ {agent_type.title()} agent initialized successfully")
                self._display_agent_summary()
                return True
            else:
                print(f"❌ Failed to initialize {agent_type} agent")
                return False
                
        except Exception as e:
            print(f"❌ Error initializing {agent_type} agent: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _display_agent_summary(self):
        """Display agent configuration summary."""
        if not self.agent:
            return
            
        print(f"\n📊 {self.agent_type.title()} Agent Summary:")
        print("-" * 40)
        print(f"   Stage: {self.agent.get_stage_description()}")
        print(f"   Required inputs: {self.agent.get_required_inputs()}")
        
        if hasattr(self.agent, 'config') and self.agent.config:
            print(f"   Project UID: {self.agent.config.workflow.project_uid}")
            print(f"   Workspace UID: {self.agent.config.workflow.workspace_uid}")
            
            # Display agent-specific parameters
            if self.agent_type == "preprocessing":
                print(f"   Movies path: {self.agent.config.workflow.movies_path}")
                print(f"   Pixel size: {self.agent.config.workflow.pixel_size} Å")
                print(f"   Voltage: {self.agent.config.workflow.voltage} kV")
        print()
    
    def run_agent_workflow(self, agent_type: str, dry_run: bool = False, conversation_id: Optional[str] = None) -> bool:
        """
        Run workflow for a specific agent.
        
        Args:
            agent_type: Type of agent to run (preprocessing, particle_picking, reconstruction)
            dry_run: If True, show what would be done without executing
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        if dry_run:
            print(f"🔍 DRY RUN: {agent_type.title()} Workflow")
            self._show_dry_run_info(agent_type)
            return True
        
        try:
            print(f"🎯 Starting {agent_type.title()} Workflow")
            print("=" * 60)
            self._show_workflow_info(agent_type)
            print()
            
            self.start_time = time.time()
            
            # Initialize the specific agent
            if not self.initialize_agent(agent_type):
                return False
            
            # Create workflow context
            context = self._create_workflow_context(agent_type, conversation_id)
            
            # Execute the agent workflow
            result = self.agent.execute_stage(context, conversation_id)
            
            # Check results
            success = result.success
            execution_time = time.time() - self.start_time
            
            if success:
                print(f"🎉 {agent_type.title()} workflow completed successfully!")
                print(f"   ✅ Stage completed in {execution_time:.2f}s")
                print(f"   📊 Stage outputs: {len(result.stage_outputs)} entries")
                if result.reasoning:
                    print(f"   🧠 Reasoning: {result.reasoning[:100]}...")
            else:
                print(f"❌ {agent_type.title()} workflow failed!")
                print(f"   ⚠️ Error: {result.error}")
                print(f"   ⏱️ Execution time: {execution_time:.2f}s")
            
            return success
            
        except Exception as e:
            print(f"❌ {agent_type.title()} workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _show_dry_run_info(self, agent_type: str):
        """Show what would be executed in dry run mode."""
        workflow_descriptions = {
            "preprocessing": "Pre-processing (Import → Motion Correction → CTF → Selection)",
            "particle_picking": "Particle Picking (Detection → Extraction → Quality Assessment)",
            "reconstruction": "3D Reconstruction (Initial Model → Refinement → Validation)"
        }
        
        description = workflow_descriptions.get(agent_type, f"{agent_type.title()} workflow")
        print(f"Would execute: {description}")
        
        if agent_type == "preprocessing":
            print("   Steps: Import Movies, Motion Correction, CTF Estimation, Micrograph Selection")
        elif agent_type == "particle_picking":
            print("   Steps: Particle Detection, Particle Extraction, Quality Assessment")
        elif agent_type == "reconstruction":
            print("   Steps: Initial Model Generation, 3D Refinement, Structure Validation")
    
    def _show_workflow_info(self, agent_type: str):
        """Show detailed workflow information."""
        if agent_type == "preprocessing":
            print("This will execute the pre-processing stage:")
            print("1. Import Movies")
            print("2. Motion Correction") 
            print("3. CTF Estimation")
            print("4. Micrograph Selection")
        elif agent_type == "particle_picking":
            print("This will execute the particle picking stage:")
            print("1. Particle Detection")
            print("2. Particle Extraction")
            print("3. Quality Assessment")
        elif agent_type == "reconstruction":
            print("This will execute the 3D reconstruction stage:")
            print("1. Initial Model Generation")
            print("2. 3D Refinement")
            print("3. Structure Validation")
    
    def _create_workflow_context(self, agent_type: str, conversation_id: Optional[str] = None) -> WorkflowContext:
        """Create workflow context for the agent."""
        # Get project and workspace UIDs from master config
        project_uid = self.master_config.workflow.project_uid if hasattr(self.master_config, 'workflow') else "P1"
        workspace_uid = self.master_config.workflow.workspace_uid if hasattr(self.master_config, 'workflow') else "W1"
        
        return WorkflowContext(
            project_uid=project_uid,
            workspace_uid=workspace_uid,
            stage_outputs={},
            metadata={
                "workflow_type": f"standalone_{agent_type}",
                "start_time": self.start_time,
                "conversation_id": conversation_id,
                "agent_type": agent_type
            }
        )
    
    def test_agent_setup(self, agent_type: str) -> bool:
        """
        Test agent setup without executing workflow.
        
        Args:
            agent_type: Type of agent to test
            
        Returns:
            True if setup test passed, False otherwise
        """
        try:
            print(f"🧪 Testing {agent_type.title()} Agent Setup")
            print("=" * 50)
            
            # Test agent initialization
            if not self.initialize_agent(agent_type):
                return False
            
            # Test workflow context creation
            context = self._create_workflow_context(agent_type)
            print(f"✅ Workflow context created successfully")
            
            # Test agent methods
            description = self.agent.get_stage_description()
            required_inputs = self.agent.get_required_inputs()
            
            print(f"✅ Stage description: {description}")
            print(f"✅ Required inputs: {required_inputs}")
            
            # Test workflow input creation (if available)
            if hasattr(self.agent, '_create_preprocessing_input'):
                workflow_input = self.agent._create_preprocessing_input(context)
                print(f"✅ Workflow input created ({len(workflow_input)} characters)")
            
            print(f"🎉 {agent_type.title()} agent setup test passed!")
            return True
            
        except Exception as e:
            print(f"❌ {agent_type.title()} agent setup test failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main function to run the CryoAgent standalone workflow."""
    parser = argparse.ArgumentParser(
        description="CryoAgent Standalone Workflow - Test individual agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cryoagent_standalone_workflow.py --agent preprocessing
  python cryoagent_standalone_workflow.py --agent particle_picking --dry-run
  python cryoagent_standalone_workflow.py --agent reconstruction --test-setup
  python cryoagent_standalone_workflow.py --agent preprocessing --verbose
        """
    )
    
    parser.add_argument(
        "--agent", 
        required=True,
        choices=["preprocessing", "particle_picking", "reconstruction"],
        help="Type of agent to run"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    
    parser.add_argument(
        "--test-setup",
        action="store_true", 
        help="Test agent setup without executing workflow"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--conversation-id",
        type=str,
        help="Conversation ID for tracking (auto-generated if not provided)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create standalone workflow
    standalone_workflow = CryoAgentStandaloneWorkflow()
    
    # Initialize workflow
    if not standalone_workflow.initialize():
        print("❌ Failed to initialize standalone workflow")
        sys.exit(1)
    
    # Generate conversation ID if not provided
    conversation_id = args.conversation_id or f"standalone_{args.agent}_{int(time.time())}"
    
    # Execute based on mode
    success = False
    
    try:
        if args.test_setup:
            success = standalone_workflow.test_agent_setup(args.agent)
        else:
            success = standalone_workflow.run_agent_workflow(
                args.agent, 
                args.dry_run, 
                conversation_id
            )
        
        # Exit with appropriate code
        if success:
            print(f"\n🎉 CryoAgent standalone {args.agent} workflow completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ CryoAgent standalone {args.agent} workflow failed!")
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
