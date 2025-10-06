"""
Master Orchestrator for Multi-Stage CryoEM Workflow

This module provides the master orchestrator that coordinates separate specialized
ReAct agents for each stage of the cryoEM processing pipeline.
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from .react_agent import ReActCryoEMAgent
from .react_workflow import ReActCryoEMWorkflow
from ..config.config_loader import ConfigLoader, CryoAgentConfig
from ..tools.cryosparc_tools import CryoSPARCTools


class WorkflowStage(Enum):
    """Enumeration of workflow stages."""
    PREPROCESSING = "preprocessing"
    PARTICLE_PICKING = "particle_picking"
    RECONSTRUCTION = "reconstruction"


@dataclass
class StageResult:
    """Result of a workflow stage execution."""
    stage: WorkflowStage
    success: bool
    stage_outputs: Dict[str, Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    reasoning: Optional[str] = None
    
    def __post_init__(self):
        if self.stage_outputs is None:
            self.stage_outputs = {}


@dataclass
class WorkflowContext:
    """Context information passed between workflow stages."""
    project_uid: str
    workspace_uid: str
    stage_outputs: Dict[WorkflowStage, Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.stage_outputs is None:
            self.stage_outputs = {}
        if self.metadata is None:
            self.metadata = {}


class StageAgent:
    """Base class for specialized stage agents."""
    
    def __init__(self, stage_name: str, config_path: str):
        """
        Initialize the stage agent.
        
        Args:
            stage_name: Name of the stage
            config_path: Path to the stage-specific configuration file
        """
        self.stage_name = stage_name
        self.config_path = config_path
        self.config = None
        self.cryosparc_tools = None
        self.react_agent = None
        self.react_workflow = None
        self.logger = logging.getLogger(f"StageAgent-{stage_name}")
        
    def initialize(self) -> bool:
        """
        Initialize the stage agent with its configuration.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load stage-specific configuration with master config
            master_config_path = "configs/master_config.json"
            config_loader = ConfigLoader(self.config_path, master_config_path)
            self.config = config_loader.load_config()
            
            # Initialize CryoSPARC tools
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            
            # Initialize ReAct agent
            self.react_agent = ReActCryoEMAgent(
                cryosparc_tools=self.cryosparc_tools,
                config=self.config
            )
            
            # Initialize ReAct workflow
            self.react_workflow = ReActCryoEMWorkflow(
                agent=self.react_agent,
                config=self.config
            )
            
            self.logger.info(f"Stage agent {self.stage_name} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """
        Execute the stage with given context.
        
        Args:
            context: Workflow context with previous stage outputs
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            StageResult with execution details
        """
        raise NotImplementedError("Subclasses must implement execute_stage")
    
    def get_stage_description(self) -> str:
        """Get description of what this stage does."""
        raise NotImplementedError("Subclasses must implement get_stage_description")
    
    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs from previous stages."""
        raise NotImplementedError("Subclasses must implement get_required_inputs")


class PreprocessingAgent(StageAgent):
    """Specialized agent for pre-processing stage."""
    
    def __init__(self, config_path: str):
        super().__init__("preprocessing", config_path)
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the pre-processing stage."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting {self.get_stage_description()}")
            
            # Create stage-specific workflow input
            workflow_input = self._create_preprocessing_input(context)
            
            # Execute the pre-processing workflow using ReAct
            result = self.react_agent.run_react_workflow(workflow_input, conversation_id)
            
            # Parse results and extract stage outputs
            stage_outputs = self._parse_preprocessing_results(result)
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=WorkflowStage.PREPROCESSING,
                success=True,
                stage_outputs=stage_outputs,
                execution_time=execution_time,
                reasoning=result
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Pre-processing stage failed: {e}")
            
            return StageResult(
                stage=WorkflowStage.PREPROCESSING,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def get_stage_description(self) -> str:
        return "Pre-processing: Import movies, motion correction, CTF estimation, and micrograph selection"
    
    def get_required_inputs(self) -> List[str]:
        return ["movies_path", "pixel_size", "voltage", "cs_mm", "dose"]
    
    def _create_preprocessing_input(self, context: WorkflowContext) -> str:
        """Create workflow input for pre-processing stage."""
        return f"""
Execute the complete cryoEM pre-processing workflow with these steps:

1. **Import Movies**: Import movie files from {self.config.workflow.movies_path}
   - Pixel size: {self.config.workflow.pixel_size} Å
   - Voltage: {self.config.workflow.voltage} kV
   - CS: {self.config.workflow.cs_mm} mm
   - Dose: {self.config.workflow.dose} e-/Å²
   - Project: {context.project_uid}
   - Workspace: {context.workspace_uid}

2. **Motion Correction**: Correct motion in the imported movies
   - Binning: {self.config.workflow.motion_correction_binning}
   - Patch size: {self.config.workflow.motion_correction_patch_size}

3. **CTF Estimation**: Estimate CTF parameters for micrographs
   - Min resolution: {self.config.workflow.ctf_min_res} Å
   - Max resolution: {self.config.workflow.ctf_max_res} Å

4. **Micrograph Selection**: Select micrographs with resolution better than 5 Å
   - Min resolution threshold: 5.0 Å
   - Quality threshold: 0.8

**Important**: 
- Each step must complete successfully before the next begins
- Always check job status and wait for completion
- Handle any errors gracefully
- Provide clear status updates throughout the process
- Store job UIDs and outputs for the next stage

Start by reasoning about the workflow state and then proceed step by step.
"""
    
    def _parse_preprocessing_results(self, result: str) -> Dict[str, Any]:
        """Parse pre-processing results to extract stage outputs."""
        execution_log = self.react_agent.get_tool_execution_log()
        
        stage_outputs = {
            "movies_job_uid": None,
            "motion_correction_job_uid": None,
            "ctf_job_uid": None,
            "micrograph_selection_job_uid": None,
            "selected_micrographs": None,
            "ctf_parameters": None
        }
        
        if execution_log:
            for entry in execution_log:
                tool = entry.get("tool")
                result_data = entry.get("result", {})
                
                if tool == "import_movies" and result_data.get("job_uid"):
                    stage_outputs["movies_job_uid"] = result_data["job_uid"]
                elif tool == "motion_correction" and result_data.get("job_uid"):
                    stage_outputs["motion_correction_job_uid"] = result_data["job_uid"]
                elif tool == "ctf_estimation" and result_data.get("job_uid"):
                    stage_outputs["ctf_job_uid"] = result_data["job_uid"]
                elif tool == "micrograph_selection" and result_data.get("job_uid"):
                    stage_outputs["micrograph_selection_job_uid"] = result_data["job_uid"]
        
        return stage_outputs


class ParticlePickingAgent(StageAgent):
    """Specialized agent for particle picking stage (placeholder)."""
    
    def __init__(self, config_path: str):
        super().__init__("particle_picking", config_path)
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the particle picking stage."""
        start_time = time.time()
        
        # Placeholder implementation
        self.logger.info("Particle picking stage not yet implemented")
        
        return StageResult(
            stage=WorkflowStage.PARTICLE_PICKING,
            success=False,
            error="Particle picking stage not yet implemented",
            execution_time=time.time() - start_time
        )
    
    def get_stage_description(self) -> str:
        return "Particle Picking: Detect and extract particles from micrographs"
    
    def get_required_inputs(self) -> List[str]:
        return ["selected_micrographs", "ctf_parameters"]


class ReconstructionAgent(StageAgent):
    """Specialized agent for 3D reconstruction stage (placeholder)."""
    
    def __init__(self, config_path: str):
        super().__init__("reconstruction", config_path)
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the 3D reconstruction stage."""
        start_time = time.time()
        
        # Placeholder implementation
        self.logger.info("3D reconstruction stage not yet implemented")
        
        return StageResult(
            stage=WorkflowStage.RECONSTRUCTION,
            success=False,
            error="3D reconstruction stage not yet implemented",
            execution_time=time.time() - start_time
        )
    
    def get_stage_description(self) -> str:
        return "3D Reconstruction: Generate initial models and refine 3D structures"
    
    def get_required_inputs(self) -> List[str]:
        return ["picked_particles", "particle_coordinates"]


class MasterOrchestrator:
    """Master orchestrator for the complete cryoEM workflow."""
    
    def __init__(self, master_config_path: str):
        """
        Initialize the master orchestrator.
        
        Args:
            master_config_path: Path to the master configuration file
        """
        self.master_config_path = master_config_path
        self.master_config = None
        self.stage_agents = {}
        self.stage_results = []
        self.workflow_context = None
        self.start_time = None
        self.logger = logging.getLogger("MasterOrchestrator")
        
    def initialize(self) -> bool:
        """
        Initialize the master orchestrator and all stage agents.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Load master configuration
            with open(self.master_config_path, 'r') as f:
                self.master_config = json.load(f)
            
            # Initialize stage agents
            for stage_info in self.master_config["master_workflow"]["stages"]:
                stage_name = stage_info["name"]
                config_path = stage_info["config_file"]
                agent_class = stage_info["agent_class"]
                
                self.logger.info(f"Initializing stage agent: {stage_name}")
                
                # Create appropriate agent based on class name
                if agent_class == "PreprocessingAgent":
                    agent = PreprocessingAgent(config_path)
                elif agent_class == "ParticlePickingAgent":
                    agent = ParticlePickingAgent(config_path)
                elif agent_class == "ReconstructionAgent":
                    agent = ReconstructionAgent(config_path)
                else:
                    self.logger.error(f"Unknown agent class: {agent_class}")
                    return False
                
                # Initialize the agent
                if agent.initialize():
                    self.stage_agents[stage_name] = agent
                    self.logger.info(f"Stage agent {stage_name} initialized successfully")
                else:
                    self.logger.error(f"Failed to initialize stage agent {stage_name}")
                    return False
            
            self.logger.info("Master orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize master orchestrator: {e}")
            return False
    
    def execute_complete_workflow(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the complete cryoEM workflow.
        
        Args:
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            Dictionary with workflow results and summary
        """
        self.start_time = time.time()
        self.stage_results = []
        
        # Initialize workflow context
        self.workflow_context = WorkflowContext(
            project_uid="P1",  # Default, will be updated from config
            workspace_uid="W1",  # Default, will be updated from config
            stage_outputs={},
            metadata={
                "workflow_type": "complete_cryoem",
                "start_time": self.start_time,
                "conversation_id": conversation_id
            }
        )
        
        self.logger.info("Starting complete cryoEM workflow")
        print("🚀 Starting Complete CryoEM Workflow")
        print("=" * 60)
        
        # Execute stages in sequence
        stages_to_execute = [
            WorkflowStage.PREPROCESSING,
            WorkflowStage.PARTICLE_PICKING,
            WorkflowStage.RECONSTRUCTION
        ]
        
        for stage in stages_to_execute:
            stage_name = stage.value
            self.logger.info(f"Executing stage: {stage_name}")
            print(f"\n🎯 Executing Stage: {stage_name.replace('_', ' ').title()}")
            print("-" * 40)
            
            # Get stage agent
            if stage_name not in self.stage_agents:
                self.logger.error(f"Stage agent not found: {stage_name}")
                continue
                
            stage_agent = self.stage_agents[stage_name]
            print(f"📋 {stage_agent.get_stage_description()}")
            
            # Execute stage
            stage_result = stage_agent.execute_stage(self.workflow_context, conversation_id)
            self.stage_results.append(stage_result)
            
            # Update context with stage outputs
            if stage_result.success:
                self.workflow_context.stage_outputs[stage] = stage_result.stage_outputs
                print(f"✅ Stage {stage_name} completed successfully")
                print(f"   Execution time: {stage_result.execution_time:.2f} seconds")
            else:
                print(f"❌ Stage {stage_name} failed: {stage_result.error}")
                # For now, continue with other stages even if one fails
                # In production, you might want to stop here
                if stage == WorkflowStage.PREPROCESSING:
                    print("⚠️ Pre-processing failed - stopping workflow")
                    break
        
        # Generate workflow summary
        total_time = time.time() - self.start_time
        summary = self._generate_workflow_summary(total_time)
        
        # Display results
        self._display_workflow_results(summary)
        
        return summary
    
    def execute_stage_workflow(self, stages: List[WorkflowStage], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a subset of workflow stages.
        
        Args:
            stages: List of stages to execute
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            Dictionary with workflow results and summary
        """
        self.start_time = time.time()
        self.stage_results = []
        
        # Initialize workflow context
        self.workflow_context = WorkflowContext(
            project_uid="P1",
            workspace_uid="W1",
            stage_outputs={},
            metadata={
                "workflow_type": "partial_cryoem",
                "stages": [s.value for s in stages],
                "start_time": self.start_time,
                "conversation_id": conversation_id
            }
        )
        
        self.logger.info(f"Starting partial workflow: {[s.value for s in stages]}")
        print(f"🚀 Starting Partial CryoEM Workflow")
        print(f"Stages: {[s.value.replace('_', ' ').title() for s in stages]}")
        print("=" * 60)
        
        # Execute specified stages
        for stage in stages:
            stage_name = stage.value
            self.logger.info(f"Executing stage: {stage_name}")
            print(f"\n🎯 Executing Stage: {stage_name.replace('_', ' ').title()}")
            print("-" * 40)
            
            # Get stage agent
            if stage_name not in self.stage_agents:
                self.logger.error(f"Stage agent not found: {stage_name}")
                continue
                
            stage_agent = self.stage_agents[stage_name]
            print(f"📋 {stage_agent.get_stage_description()}")
            
            # Execute stage
            stage_result = stage_agent.execute_stage(self.workflow_context, conversation_id)
            self.stage_results.append(stage_result)
            
            # Update context with stage outputs
            if stage_result.success:
                self.workflow_context.stage_outputs[stage] = stage_result.stage_outputs
                print(f"✅ Stage {stage_name} completed successfully")
                print(f"   Execution time: {stage_result.execution_time:.2f} seconds")
            else:
                print(f"❌ Stage {stage_name} failed: {stage_result.error}")
                break
        
        # Generate workflow summary
        total_time = time.time() - self.start_time
        summary = self._generate_workflow_summary(total_time)
        
        # Display results
        self._display_workflow_results(summary)
        
        return summary
    
    def _generate_workflow_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate a summary of the workflow execution."""
        successful_stages = [r for r in self.stage_results if r.success]
        failed_stages = [r for r in self.stage_results if not r.success]
        
        summary = {
            "workflow_type": self.workflow_context.metadata.get("workflow_type", "unknown"),
            "total_stages": len(self.stage_results),
            "successful_stages": len(successful_stages),
            "failed_stages": len(failed_stages),
            "total_execution_time": total_time,
            "stage_results": [
                {
                    "stage": result.stage.value,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "error": result.error,
                    "outputs": result.stage_outputs
                }
                for result in self.stage_results
            ],
            "workflow_context": {
                "project_uid": self.workflow_context.project_uid,
                "workspace_uid": self.workflow_context.workspace_uid,
                "stage_outputs": self.workflow_context.stage_outputs,
                "metadata": self.workflow_context.metadata
            }
        }
        
        return summary
    
    def _display_workflow_results(self, summary: Dict[str, Any]):
        """Display workflow results in a formatted way."""
        print("\n📊 Workflow Execution Summary")
        print("=" * 50)
        print(f"Workflow Type: {summary['workflow_type']}")
        print(f"Total Stages: {summary['total_stages']}")
        print(f"Successful: {summary['successful_stages']}")
        print(f"Failed: {summary['failed_stages']}")
        print(f"Total Time: {summary['total_execution_time']:.2f} seconds")
        print()
        
        print("📋 Stage Results:")
        for i, result in enumerate(summary['stage_results'], 1):
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            print(f"   {i}. {result['stage'].replace('_', ' ').title()}: {status}")
            print(f"      Execution Time: {result['execution_time']:.2f}s")
            if result['error']:
                print(f"      Error: {result['error']}")
            print()
        
        if summary['successful_stages'] == summary['total_stages']:
            print("🎉 Complete workflow executed successfully!")
        else:
            print("⚠️ Workflow completed with some failures")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            "stage_results": self.stage_results,
            "workflow_context": self.workflow_context,
            "total_execution_time": time.time() - self.start_time if self.start_time else 0
        }
    
    def reset_workflow(self):
        """Reset the workflow state."""
        self.stage_results = []
        self.workflow_context = None
        self.start_time = None
        
        # Reset all stage agents
        for agent in self.stage_agents.values():
            if hasattr(agent, 'react_agent') and agent.react_agent:
                agent.react_agent.clear_reasoning_history()
