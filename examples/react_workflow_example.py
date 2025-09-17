#!/usr/bin/env python3
"""
Example script demonstrating the ReAct-based CryoEM workflow.

This example shows how to use the new ReAct agent with configuration file
for automated cryoEM processing workflows.
"""

import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import cryoagent
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryoagent import (
    ReActCryoEMAgent, 
    ReActCryoEMWorkflow, 
    CryoSPARCTools,
    ConfigLoader
)


def main():
    """Main function to demonstrate ReAct workflow."""
    print("🚀 CryoAgent ReAct Workflow Example")
    print("=" * 50)
    
    try:
        # Load configuration from JSON file
        print("📋 Loading configuration...")
        config_loader = ConfigLoader("config.json")
        config = config_loader.load_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   - Project: {config.workflow.project_uid}")
        print(f"   - Workspace: {config.workflow.workspace_uid}")
        print(f"   - Movies Path: {config.workflow.movies_path}")
        print(f"   - Model: {config.agent.model_name}")
        print()
        
        # Initialize CryoSPARC tools
        print("🔧 Initializing CryoSPARC tools...")
        cryosparc_tools = CryoSPARCTools(config.cryosparc)
        print("✅ CryoSPARC tools initialized")
        print()
        
        # Initialize ReAct agent
        print("🤖 Initializing ReAct CryoEM agent...")
        agent = ReActCryoEMAgent(
            cryosparc_tools=cryosparc_tools,
            config=config
        )
        print("✅ ReAct agent initialized")
        print()
        
        # Initialize ReAct workflow
        print("⚙️ Initializing ReAct workflow...")
        workflow = ReActCryoEMWorkflow(
            agent=agent,
            config=config
        )
        print("✅ ReAct workflow initialized")
        print()
        
        # Run the basic workflow
        print("🎯 Starting ReAct-based cryoEM workflow...")
        print("   This will execute: Import Movies → Motion Correction → CTF Estimation")
        print("   Using ReAct (Reasoning + Acting) approach")
        print()
        
        results = workflow.run_basic_workflow()
        
        # Display results
        print("📊 Workflow Results:")
        print("=" * 30)
        
        for i, result in enumerate(results, 1):
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"{i}. {result.step.value}: {status}")
            if result.job_uid:
                print(f"   Job UID: {result.job_uid}")
            if result.message:
                print(f"   Message: {result.message}")
            if result.error:
                print(f"   Error: {result.error}")
            print()
        
        # Get workflow summary
        summary = workflow.get_workflow_summary()
        print("📈 Workflow Summary:")
        print(f"   Total Steps: {summary['total_steps']}")
        print(f"   Successful: {summary['successful_steps']}")
        print(f"   Failed: {summary['failed_steps']}")
        print()
        
        # Display reasoning history if available
        reasoning_history = agent.get_reasoning_history()
        if reasoning_history:
            print("🧠 Reasoning History:")
            for i, reasoning in enumerate(reasoning_history, 1):
                print(f"   {i}. {reasoning}")
            print()
        
        print("🎉 ReAct workflow example completed!")
        
    except FileNotFoundError as e:
        print(f"❌ Configuration file not found: {e}")
        print("   Please ensure config.json exists in the project root")
        return 1
        
    except Exception as e:
        print(f"❌ Error running ReAct workflow: {e}")
        return 1
    
    return 0


def demonstrate_custom_workflow():
    """Demonstrate custom workflow with specific steps."""
    print("\n" + "=" * 50)
    print("🎯 Custom Workflow Example")
    print("=" * 50)
    
    try:
        # Load configuration
        config_loader = ConfigLoader("config.json")
        config = config_loader.load_config()
        
        # Initialize components
        cryosparc_tools = CryoSPARCTools(config.cryosparc)
        agent = ReActCryoEMAgent(cryosparc_tools=cryosparc_tools, config=config)
        workflow = ReActCryoEMWorkflow(agent=agent, config=config)
        
        # Run custom workflow (only import movies and motion correction)
        from cryoagent.core.react_workflow import WorkflowStep
        
        custom_steps = [WorkflowStep.IMPORT_MOVIES, WorkflowStep.MOTION_CORRECTION]
        print(f"Running custom workflow with steps: {[step.value for step in custom_steps]}")
        
        results = workflow.run_custom_workflow(custom_steps)
        
        print("📊 Custom Workflow Results:")
        for result in results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            print(f"   {result.step.value}: {status}")
        
    except Exception as e:
        print(f"❌ Error in custom workflow: {e}")


if __name__ == "__main__":
    exit_code = main()
    
    # Optionally run custom workflow example
    if exit_code == 0:
        demonstrate_custom_workflow()
    
    sys.exit(exit_code)
