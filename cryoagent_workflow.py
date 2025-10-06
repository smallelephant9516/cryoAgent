#!/usr/bin/env python3
"""
CryoAgent - Intelligent CryoEM Workflow Orchestrator

This script provides a comprehensive agentic workflow for cryoEM image processing
using the ReAct (Reasoning + Acting) framework with CryoSPARC integration.

Features:
- Intelligent workflow orchestration with reasoning
- Automatic job monitoring and dependency management
- Comprehensive error handling and retry logic
- Real-time status updates and progress tracking
- Flexible configuration management
- Support for custom workflows

Usage:
    python cryoagent_workflow.py [options]

Options:
    --config CONFIG_FILE    Path to configuration file (default: config.json)
    --workflow WORKFLOW     Workflow type: basic, custom, or single (default: basic)
    --steps STEPS           Comma-separated list of steps for custom workflow
    --timeout TIMEOUT       Job timeout in seconds (default: from config)
    --verbose               Enable verbose output
    --dry-run               Show what would be done without executing
"""

import sys
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from cryoagent import (
    ReActCryoEMAgent, 
    ReActCryoEMWorkflow, 
    CryoSPARCTools,
    ConfigLoader
)
from cryoagent.core.react_workflow import WorkflowStep


class CryoAgentWorkflow:
    """Main workflow orchestrator for CryoAgent."""
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the CryoAgent workflow.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = None
        self.cryosparc_tools = None
        self.agent = None
        self.workflow = None
        self.start_time = None
        
    def initialize(self) -> bool:
        """
        Initialize all components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("🚀 Initializing CryoAgent Workflow")
            print("=" * 60)
            
            # Load configuration
            print("📋 Loading configuration...")
            config_loader = ConfigLoader(self.config_path)
            self.config = config_loader.load_config()
            print(f"✅ Configuration loaded from {self.config_path}")
            
            # Initialize CryoSPARC tools
            print("🔧 Initializing CryoSPARC tools...")
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            print("✅ CryoSPARC tools initialized")
            
            # Initialize ReAct agent
            print("🤖 Initializing ReAct CryoEM agent...")
            self.agent = ReActCryoEMAgent(
                cryosparc_tools=self.cryosparc_tools,
                config=self.config
            )
            print("✅ ReAct agent initialized")
            
            # Initialize ReAct workflow
            print("⚙️ Initializing ReAct workflow...")
            self.workflow = ReActCryoEMWorkflow(
                agent=self.agent,
                config=self.config
            )
            print("✅ ReAct workflow initialized")
            
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
        print(f"   Project UID: {self.config.workflow.project_uid}")
        print(f"   Workspace UID: {self.config.workflow.workspace_uid}")
        print(f"   Movies Path: {self.config.workflow.movies_path}")
        print(f"   Pixel Size: {self.config.workflow.pixel_size} Å")
        print(f"   Voltage: {self.config.workflow.voltage} kV")
        print(f"   CS: {self.config.workflow.cs_mm} mm")
        print(f"   Dose: {self.config.workflow.dose} e-/Å²")
        
        # Display model information
        model_info = self.agent.get_current_model_info()
        available_providers = self.config.agent.get_available_providers()
        
        print(f"   LLM Provider: {model_info['provider']}")
        print(f"   Model: {model_info['model_name']}")
        print(f"   Base URL: {model_info['base_url']}")
        print(f"   Temperature: {model_info['temperature']}")
        print(f"   Available Providers: {', '.join(available_providers) if available_providers else 'None (no valid API keys)'}")
        
        if not available_providers:
            print("   ⚠️ Warning: No valid API keys found. Please set one of: DEEPSEEK_API_KEY, OPENAI_API_KEY, or PANSHI_API_KEY")
        
        print(f"   Timeout: {self.config.job_management.default_timeout}s")
        print()
    
    def run_basic_workflow(self, dry_run: bool = False, conversation_id: Optional[str] = None) -> bool:
        """
        Run the basic cryoEM workflow using ReAct agent with job monitoring.
        
        Args:
            dry_run: If True, show what would be done without executing
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        if dry_run:
            print("🔍 DRY RUN: Basic Workflow")
            # Get steps from configuration
            steps = []
            for step_config in self.config.react_workflow.steps:
                step_name = step_config.name.replace('_', ' ').title()
                steps.append(step_name)
            
            workflow_steps = " → ".join(steps)
            print(f"Would execute: {workflow_steps}")
            return True
        
        try:
            print("🎯 Starting ReAct-Based CryoEM Workflow with Job Monitoring")
            print("=" * 70)
            print("This will use the ReAct (Reasoning + Acting) framework")
            
            # Get steps from configuration dynamically
            steps = []
            for step_config in self.config.react_workflow.steps:
                step_name = step_config.name.replace('_', ' ').title()
                steps.append(step_name)
            
            workflow_steps = " → ".join(steps)
            print(f"to intelligently execute: {workflow_steps}")
            print("with automatic job monitoring and dependency management")
            print()
            print("📋 Workflow Steps:")
            for i, step_config in enumerate(self.config.react_workflow.steps, 1):
                step_name = step_config.name.replace('_', ' ').title()
                description = step_config.description
                print(f"   {i}. {step_name} - {description}")
            print()
            
            self.start_time = time.time()
            
            # Create a fresh agent instance to prevent hallucination
            print("🧠 Creating fresh agent instance to ensure clean state...")
            self.agent = self.agent.create_fresh_agent()
            print("✅ Fresh agent instance created")
            
            # Use ReAct agent to orchestrate the entire workflow with monitoring
            workflow_input = self._create_workflow_input()
            
            print("🤖 ReAct Agent Starting Workflow Execution...")
            print("The agent will reason through each step and monitor job completion")
            print()
            
            # Execute the workflow using ReAct approach
            result = self.agent.run_react_workflow(workflow_input, conversation_id)
            
            # Display the ReAct agent's execution result
            print("📊 ReAct Agent Execution Result:")
            print("=" * 50)
            print(result)
            print()
            
            # Check if the workflow completed successfully
            success = self._analyze_workflow_result(result)
            
            # Display timing information
            if self.start_time:
                elapsed = time.time() - self.start_time
                print(f"⏱️ Total Execution Time: {elapsed:.2f} seconds")
                print()
            
            # Display reasoning history if available
            reasoning_history = self.agent.get_reasoning_history()
            if reasoning_history:
                print("🧠 ReAct Reasoning History:")
                for i, reasoning in enumerate(reasoning_history, 1):
                    print(f"   {i}. {reasoning}")
                print()
            
            if success:
                print("🎉 ReAct-based workflow completed successfully!")
                print("   ✅ All jobs completed with proper monitoring")
                print("   🔗 Workflow dependencies properly handled")
                print("   ⏱️ Each tool waited for completion before claiming success")
            else:
                print("❌ ReAct-based workflow failed!")
                print("   ⚠️ Check the reasoning history above for details")
            
            return success
            
        except Exception as e:
            print(f"❌ ReAct workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_custom_workflow(self, steps: List[str], dry_run: bool = False, conversation_id: Optional[str] = None) -> bool:
        """
        Run a custom workflow with specified steps using ReAct agent.
        
        Args:
            steps: List of workflow steps to execute
            dry_run: If True, show what would be done without executing
            
        Returns:
            True if workflow completed successfully, False otherwise
        """
        if dry_run:
            print(f"🔍 DRY RUN: Custom Workflow")
            print(f"Would execute: {' → '.join(steps)}")
            return True
        
        try:
            print("🎯 Starting Custom ReAct-Based CryoEM Workflow")
            print("=" * 60)
            print(f"Steps: {' → '.join(steps)}")
            print("Using ReAct (Reasoning + Acting) framework with job monitoring")
            print()
            
            # Validate steps
            valid_steps = [s.value for s in WorkflowStep]
            for step_str in steps:
                if step_str.lower() not in valid_steps:
                    print(f"❌ Invalid workflow step: {step_str}")
                    print(f"   Valid steps: {valid_steps}")
                    return False
            
            self.start_time = time.time()
            
            # Create a fresh agent instance to prevent hallucination
            print("🧠 Creating fresh agent instance to ensure clean state...")
            self.agent = self.agent.create_fresh_agent()
            print("✅ Fresh agent instance created")
            
            # Create custom workflow input
            workflow_input = self._create_custom_workflow_input(steps)
            
            print("🤖 ReAct Agent Starting Custom Workflow Execution...")
            print("The agent will reason through each step and monitor job completion")
            print()
            
            # Execute the workflow using ReAct approach
            result = self.agent.run_react_workflow(workflow_input, conversation_id)
            
            # Display the ReAct agent's execution result
            print("📊 ReAct Agent Custom Workflow Result:")
            print("=" * 50)
            print(result)
            print()
            
            # Check if the workflow completed successfully
            success = self._analyze_workflow_result(result)
            
            # Display timing information
            if self.start_time:
                elapsed = time.time() - self.start_time
                print(f"⏱️ Total Execution Time: {elapsed:.2f} seconds")
                print()
            
            # Display reasoning history if available
            reasoning_history = self.agent.get_reasoning_history()
            if reasoning_history:
                print("🧠 ReAct Reasoning History:")
                for i, reasoning in enumerate(reasoning_history, 1):
                    print(f"   {i}. {reasoning}")
                print()
            
            if success:
                print("🎉 Custom ReAct-based workflow completed successfully!")
            else:
                print("❌ Custom ReAct-based workflow failed!")
                print("   ⚠️ Check the reasoning history above for details")
            
            return success
            
        except Exception as e:
            print(f"❌ Custom workflow failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_single_step(self, step: str, dry_run: bool = False, conversation_id: Optional[str] = None) -> bool:
        """
        Run a single workflow step using ReAct agent.
        
        Args:
            step: The step to execute
            dry_run: If True, show what would be done without executing
            
        Returns:
            True if step completed successfully, False otherwise
        """
        if dry_run:
            print(f"🔍 DRY RUN: Single Step")
            print(f"Would execute: {step}")
            return True
        
        try:
            print(f"🎯 Starting Single Step with ReAct Agent: {step}")
            print("=" * 60)
            print("Using ReAct (Reasoning + Acting) framework with job monitoring")
            print()
            
            self.start_time = time.time()
            
            # Create a fresh agent instance to prevent hallucination
            print("🧠 Creating fresh agent instance to ensure clean state...")
            self.agent = self.agent.create_fresh_agent()
            print("✅ Fresh agent instance created")
            
            # Create a focused workflow input for the single step
            workflow_input = f"""
Execute the following single cryoEM processing step:

**Step**: {step}

**Configuration**:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}
- Movies Path: {self.config.workflow.movies_path}
- Pixel Size: {self.config.workflow.pixel_size} Å
- Voltage: {self.config.workflow.voltage} kV
- CS: {self.config.workflow.cs_mm} mm
- Dose: {self.config.workflow.dose} e-/Å²

**Important**: 
- Execute the step with proper reasoning
- Wait for job completion if applicable
- Provide clear status updates
- Handle any errors gracefully

Start by reasoning about what needs to be done and then execute the step.
"""
            
            print("🤖 ReAct Agent Starting Single Step Execution...")
            print("The agent will reason through the step and monitor completion")
            print()
            
            # Execute using ReAct approach
            result = self.agent.run_react_workflow(workflow_input, conversation_id)
            
            print("📊 ReAct Agent Single Step Result:")
            print("=" * 50)
            print(result)
            print()
            
            # Check if successful
            success = self._analyze_workflow_result(result)
            
            # Display timing information
            if self.start_time:
                elapsed = time.time() - self.start_time
                print(f"⏱️ Total Execution Time: {elapsed:.2f} seconds")
                print()
            
            # Display reasoning history if available
            reasoning_history = self.agent.get_reasoning_history()
            if reasoning_history:
                print("🧠 ReAct Reasoning History:")
                for i, reasoning in enumerate(reasoning_history, 1):
                    print(f"   {i}. {reasoning}")
                print()
            
            if success:
                print("✅ Single step completed successfully!")
            else:
                print("❌ Single step failed or did not complete")
                print("   ⚠️ Check the reasoning history above for details")
            
            return success
            
        except Exception as e:
            print(f"❌ Single step failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_workflow_input(self) -> str:
        """Create the workflow input for the ReAct agent."""
        return f"""
Execute the complete cryoEM processing workflow with these steps:

1. **Import Movies**: Import movie files from {self.config.workflow.movies_path}
   - Pixel size: {self.config.workflow.pixel_size} Å
   - Voltage: {self.config.workflow.voltage} kV
   - CS: {self.config.workflow.cs_mm} mm
   - Dose: {self.config.workflow.dose} e-/Å²
   - Project: {self.config.workflow.project_uid}
   - Workspace: {self.config.workflow.workspace_uid}

2. **Motion Correction**: Correct motion in the imported movies
   - Binning: {self.config.workflow.motion_correction_binning}
   - Patch size: {self.config.workflow.motion_correction_patch_size}

3. **CTF Estimation**: Estimate CTF parameters for micrographs
   - Min resolution: {self.config.workflow.ctf_min_res} Å
   - Max resolution: {self.config.workflow.ctf_max_res} Å

4. **Micrograph Selection**: Select micrographs with resolution better than 5 Å
   - Min resolution threshold: 5.0 Å
   - Filters out low-quality micrographs

**Important**: 
- Each step must complete successfully before the next begins
- Always check job status and wait for completion
- Handle any errors gracefully
- Provide clear status updates throughout the process

Start by reasoning about the workflow state and then proceed step by step.
"""
    
    def _create_custom_workflow_input(self, steps: List[str]) -> str:
        """Create custom workflow input for specified steps."""
        step_descriptions = []
        
        for i, step in enumerate(steps, 1):
            if step.lower() == "import_movies":
                step_descriptions.append(f"""
{i}. **Import Movies**: Import movie files from {self.config.workflow.movies_path}
   - Pixel size: {self.config.workflow.pixel_size} Å
   - Voltage: {self.config.workflow.voltage} kV
   - CS: {self.config.workflow.cs_mm} mm
   - Dose: {self.config.workflow.dose} e-/Å²
""")
            elif step.lower() == "motion_correction":
                step_descriptions.append(f"""
{i}. **Motion Correction**: Correct motion in imported movies
   - Binning: {self.config.workflow.motion_correction_binning}
   - Patch size: {self.config.workflow.motion_correction_patch_size}
""")
            elif step.lower() == "ctf_estimation":
                step_descriptions.append(f"""
{i}. **CTF Estimation**: Estimate CTF parameters for micrographs
   - Min resolution: {self.config.workflow.ctf_min_res} Å
   - Max resolution: {self.config.workflow.ctf_max_res} Å
""")
            elif step.lower() == "micrograph_selection":
                step_descriptions.append(f"""
{i}. **Micrograph Selection**: Select micrographs with resolution better than 5 Å
   - Min resolution threshold: 5.0 Å
   - Filters out low-quality micrographs
""")
        
        return f"""
Execute the following custom cryoEM workflow:

{''.join(step_descriptions)}

**Important**: 
- Each step must complete successfully before the next begins
- Always check job status and wait for completion
- Handle any errors gracefully
- Provide clear status updates throughout the process

Start by reasoning about the workflow state and then proceed step by step.
"""
    
    def _analyze_workflow_result(self, result: str) -> bool:
        """Analyze the workflow result to determine success."""
        execution_log = self.agent.get_tool_execution_log()

        if not execution_log:
            print("⚠️ No CryoSPARC tool activity was recorded during this run. The agent likely hallucinated the workflow.")
            print("🔧 This indicates the agent's internal state wasn't properly reset. Try running again.")
            return False
        
        # Additional validation: Check if any tools were actually invoked
        cryosparc_tools = {"import_movies", "motion_correction", "ctf_estimation", "micrograph_selection", "wait_for_job", "get_job_status"}
        actual_tool_calls = [entry for entry in execution_log if entry.get("tool") in cryosparc_tools]
        
        if not actual_tool_calls:
            print("⚠️ No actual CryoSPARC tool calls were recorded. The agent likely hallucinated the workflow.")
            print("🔧 This indicates the agent's internal state wasn't properly reset. Try running again.")
            return False

        # If any critical tool reported an error, flag the workflow as failed immediately
        critical_tools = {"import_movies", "motion_correction", "ctf_estimation", "micrograph_selection", "wait_for_job"}
        critical_errors = [
            entry for entry in execution_log
            if entry.get("error") and entry.get("tool") in critical_tools
        ]
        if critical_errors:
            print("⚠️ Encountered errors while executing CryoSPARC tools:")
            for entry in critical_errors:
                print(f"   - {entry['tool']}: {entry['error']}")
            return False

        step_requirements = {
            "import_movies": {"job_uid": None},
            "motion_correction": {"job_uid": None},
            "ctf_estimation": {"job_uid": None},
            "micrograph_selection": {"job_uid": None}
        }
        wait_results: Dict[str, Dict[str, Any]] = {}

        for entry in execution_log:
            tool = entry.get("tool")
            if tool in step_requirements and entry.get("result"):
                job_uid = entry["result"].get("job_uid")
                if job_uid:
                    step_requirements[tool]["job_uid"] = job_uid
            if tool == "wait_for_job" and entry.get("result"):
                job_uid = entry.get("params", {}).get("job_uid")
                if job_uid:
                    wait_results[job_uid] = entry["result"]

        missing_steps = [step for step, info in step_requirements.items() if not info.get("job_uid")]
        if missing_steps:
            readable = ", ".join(missing_steps)
            print(f"⚠️ The agent did not execute the following required steps: {readable}.")
            print("   Treating the workflow as failed.")
            return False

        for step, info in step_requirements.items():
            job_uid = info.get("job_uid")
            wait_info = wait_results.get(job_uid)
            if not wait_info:
                print(f"⚠️ No wait_for_job call was recorded for {step} job {job_uid}.")
                print("   The workflow cannot be marked successful without confirming job completion.")
                return False
            status = wait_info.get("status")
            if status != "completed":
                print(f"⚠️ Job {job_uid} ({step}) finished with status '{status}'.")
                return False

        return True
    
    def _process_results(self, results: List, workflow_type: str) -> bool:
        """
        Process and display workflow results.
        
        Args:
            results: List of workflow results
            workflow_type: Type of workflow executed
            
        Returns:
            True if all steps successful, False otherwise
        """
        print(f"📊 {workflow_type} Results:")
        print("=" * 50)
        
        all_success = True
        for i, result in enumerate(results, 1):
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"{i}. {result.step.value}: {status}")
            
            if result.job_uid:
                print(f"   Job UID: {result.job_uid}")
            if result.message:
                print(f"   Message: {result.message}")
            if result.error:
                print(f"   Error: {result.error}")
                all_success = False
            if result.reasoning:
                print(f"   Reasoning: {result.reasoning[:100]}...")
            print()
        
        # Display workflow summary
        summary = self.workflow.get_workflow_summary()
        print("📈 Workflow Summary:")
        print(f"   Total Steps: {summary['total_steps']}")
        print(f"   Successful: {summary['successful_steps']}")
        print(f"   Failed: {summary['failed_steps']}")
        
        # Display timing information
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"   Execution Time: {elapsed:.2f} seconds")
        
        print()
        
        # Display reasoning history if available
        reasoning_history = self.agent.get_reasoning_history()
        if reasoning_history:
            print("🧠 ReAct Reasoning History:")
            for i, reasoning in enumerate(reasoning_history, 1):
                print(f"   {i}. {reasoning}")
            print()
        
        # Display current workflow state
        current_state = self.workflow.get_current_state()
        print("🔍 Current Workflow State:")
        print(f"   Status: {current_state['workflow_state'].get('workflow_status', 'unknown')}")
        print(f"   Active Jobs: {len(current_state['current_job_uids'])}")
        for step, job_uid in current_state['current_job_uids'].items():
            print(f"     {step.value}: {job_uid}")
        print()
        
        if all_success:
            print(f"🎉 {workflow_type} completed successfully!")
            print("   ✅ All steps completed with proper monitoring")
            print("   🔗 Workflow dependencies properly handled")
            print("   ⏱️ Each tool waited for completion before claiming success")
        else:
            print(f"❌ {workflow_type} failed!")
            print("   ⚠️ One or more steps did not complete successfully")
        
        return all_success
    
    def test_connection(self) -> bool:
        """
        Test the CryoSPARC connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            print("🔌 Testing CryoSPARC Connection")
            print("=" * 40)
            
            # Test basic connection
            projects = self.cryosparc_tools.list_projects()
            print(f"✅ Connected to CryoSPARC successfully")
            print(f"   Found {len(projects)} projects")
            
            # Test project access
            project_uid = self.config.workflow.project_uid
            workspaces = self.cryosparc_tools.list_workspaces(project_uid)
            print(f"✅ Project {project_uid} accessible")
            print(f"   Found {len(workspaces)} workspaces")
            
            return True
            
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


def main():
    """Main function to run the CryoAgent workflow."""
    parser = argparse.ArgumentParser(
        description="CryoAgent - Intelligent CryoEM Workflow Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic workflow
  python cryoagent_workflow.py

  # Run custom workflow
  python cryoagent_workflow.py --workflow custom --steps import_movies,motion_correction,ctf_estimation,micrograph_selection

  # Run single step
  python cryoagent_workflow.py --workflow single --steps "Import movies and wait for completion"

  # Test connection
  python cryoagent_workflow.py --workflow test

  # Dry run
  python cryoagent_workflow.py --dry-run

  # Clear AI memory and run workflow
  python cryoagent_workflow.py --clear-memory
        """
    )
    
    parser.add_argument(
        "--config", 
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    
    parser.add_argument(
        "--workflow",
        choices=["basic", "custom", "single", "test"],
        default="basic",
        help="Workflow type (default: basic)"
    )
    
    parser.add_argument(
        "--steps",
        help="Comma-separated list of steps for custom workflow, or description for single step"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        help="Job timeout in seconds (overrides config)"
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
        "--clear-memory",
        action="store_true",
        help="Force clear AI memory before starting workflow"
    )
    
    parser.add_argument(
        "--model",
        choices=["deepseek", "openai", "panshi"],
        help="Override the LLM model provider (deepseek, openai, panshi)"
    )
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = CryoAgentWorkflow(args.config)
    
    if not workflow.initialize():
        print("❌ Failed to initialize CryoAgent workflow")
        sys.exit(1)
    
    # Override timeout if specified
    if args.timeout:
        workflow.config.job_management.default_timeout = args.timeout
        print(f"⏱️ Timeout set to {args.timeout} seconds")
    
    # Set verbose mode
    if args.verbose:
        workflow.config.agent.verbose = True
        print("🔊 Verbose mode enabled")
    
    # Force clear memory if requested
    if args.clear_memory:
        print("🧠 Force clearing AI memory...")
        workflow.agent.force_clear_memory()
        print("✅ AI memory cleared")
    
    # Override model provider if specified
    if args.model:
        print(f"🔄 Switching to model provider: {args.model}")
        try:
            workflow.agent.switch_model_provider(args.model)
            print(f"✅ Model provider switched to: {args.model}")
        except ValueError as e:
            print(f"❌ Failed to switch model provider: {e}")
            print("💡 Available providers with valid API keys:")
            available = workflow.config.agent.get_available_providers()
            if available:
                for provider in available:
                    print(f"   - {provider}")
            else:
                print("   None - please set one of: DEEPSEEK_API_KEY, OPENAI_API_KEY, or PANSHI_API_KEY")
            sys.exit(1)
    
    # Always create a fresh agent instance to prevent hallucination
    print("🧠 Ensuring fresh agent state to prevent hallucination...")
    workflow.agent = workflow.agent.create_fresh_agent()
    print("✅ Fresh agent state ensured")
    
    # Execute workflow based on type
    success = False
    
    try:
        if args.workflow == "test":
            success = workflow.test_connection()
            
        elif args.workflow == "basic":
            # Use a unique conversation ID to ensure fresh start
            import time
            conversation_id = f"workflow_{int(time.time())}"
            success = workflow.run_basic_workflow(args.dry_run, conversation_id)
            
        elif args.workflow == "custom":
            if not args.steps:
                print("❌ --steps required for custom workflow")
                print("   Valid steps: import_movies, motion_correction, ctf_estimation, micrograph_selection")
                sys.exit(1)
            steps = [s.strip() for s in args.steps.split(",")]
            # Use a unique conversation ID to ensure fresh start
            import time
            conversation_id = f"custom_{int(time.time())}"
            success = workflow.run_custom_workflow(steps, args.dry_run, conversation_id)
            
        elif args.workflow == "single":
            if not args.steps:
                print("❌ --steps required for single step workflow")
                print("   Example: --steps 'Import movies and wait for completion'")
                sys.exit(1)
            # Use a unique conversation ID to ensure fresh start
            import time
            conversation_id = f"single_{int(time.time())}"
            success = workflow.run_single_step(args.steps, args.dry_run, conversation_id)
        
        # Exit with appropriate code
        if success:
            print("\n🎉 CryoAgent workflow completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ CryoAgent workflow failed!")
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
