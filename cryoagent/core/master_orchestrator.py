"""
Master Orchestrator for Multi-Stage CryoEM Workflow

This module provides the master orchestrator that coordinates separate specialized
ReAct agents for each stage of the cryoEM processing pipeline.

Updated to use modular agent architecture.
"""

import json
import time
import logging
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Import modular agents
from .cryosparc_preprocessing import PreprocessingAgent as ModularPreprocessingAgent, PreprocessingWorkflow
from .cryosparc_picking import PickingAgent as ModularPickingAgent, PickingWorkflow
from ..config.config_loader import ConfigLoader, CryoAgentConfig
from ..tools.cryosparc_tools import CryoSPARCTools


class WorkflowStage(Enum):
    """Enumeration of workflow stages."""
    PREPROCESSING = "preprocessing"
    PARTICLE_PICKING = "particle_picking"
    RECONSTRUCTION = "reconstruction"


def check_stage_output_exists(stage: WorkflowStage, outputs_dir: str = "outputs") -> Optional[Dict[str, Any]]:
    """
    Check if output file for a given stage already exists.
    
    Args:
        stage: The workflow stage to check
        outputs_dir: Directory where output files are stored
        
    Returns:
        Dictionary with output file information if exists, None otherwise
    """
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        return None
    
    # Map stage names to output file patterns
    stage_patterns = {
        WorkflowStage.PREPROCESSING: "preprocessing_results_*.json",
        WorkflowStage.PARTICLE_PICKING: "particle_picking_results_*.json",
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
    """Base class for specialized stage agents using modular architecture."""
    
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
        self.modular_agent = None  # Changed from react_agent to modular_agent
        self.modular_workflow = None  # Changed from react_workflow to modular_workflow
        self.logger = logging.getLogger(f"StageAgent-{stage_name}")
        
    def initialize(self) -> bool:
        """
        Initialize the stage agent with its configuration.
        Must be implemented by subclasses to use specific modular agents.
        
        Returns:
            True if initialization successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement initialize with specific modular agent")
    
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
    """Specialized agent for pre-processing stage using modular architecture."""
    
    def __init__(self, config_path: str):
        super().__init__("preprocessing", config_path)
    
    def initialize(self) -> bool:
        """Initialize the preprocessing agent with modular architecture."""
        try:
            # Load stage-specific configuration with master config
            master_config_path = "configs/master_config.json"
            config_loader = ConfigLoader(self.config_path, master_config_path)
            self.config = config_loader.load_config()
            
            # Initialize CryoSPARC tools
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            
            # Initialize modular preprocessing agent
            self.modular_agent = ModularPreprocessingAgent(
                cryosparc_tools=self.cryosparc_tools,
                config=self.config
            )
            
            # Initialize preprocessing workflow
            self.modular_workflow = PreprocessingWorkflow(
                agent=self.modular_agent,
                config=self.config
            )
            
            self.logger.info(f"Stage agent {self.stage_name} initialized successfully with modular architecture")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the pre-processing stage using modular workflow."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting {self.get_stage_description()}")
            
            # Execute the preprocessing workflow using modular architecture
            results = self.modular_workflow.run(conversation_id=conversation_id)
            
            # Parse results and extract stage outputs
            stage_outputs = self._parse_modular_preprocessing_results(results)
            
            # Validate that jobs were actually executed
            validation_result = self._validate_preprocessing_results(stage_outputs)
            
            # Save preprocessing results to JSON file
            result_file_path = self._save_preprocessing_results(stage_outputs, context, validation_result["success"])
            self.logger.info(f"Preprocessing results saved to: {result_file_path}")
            print(f"📄 Preprocessing results saved to: {result_file_path}")
            
            # Add result file path to stage outputs
            stage_outputs["result_file"] = result_file_path
            
            execution_time = time.time() - start_time
            
            # Return result based on validation
            if not validation_result["success"]:
                self.logger.error(f"Pre-processing validation failed: {validation_result['error']}")
                return StageResult(
                    stage=WorkflowStage.PREPROCESSING,
                    success=False,
                    stage_outputs=stage_outputs,
                    error=validation_result["error"],
                    execution_time=execution_time
                )
            
            return StageResult(
                stage=WorkflowStage.PREPROCESSING,
                success=True,
                stage_outputs=stage_outputs,
                execution_time=execution_time
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
    
    def _parse_modular_preprocessing_results(self, results: List) -> Dict[str, Any]:
        """Parse modular preprocessing results to extract stage outputs."""
        stage_outputs = {
            "movies_job_uid": None,
            "motion_correction_job_uid": None,
            "ctf_job_uid": None,
            "micrograph_selection_job_uid": None,
            "selected_micrographs": None,
            "ctf_parameters": None
        }
        
        # Extract job UIDs from modular workflow results
        for result in results:
            step_name = result.step.value
            if result.success and result.job_uid:
                if step_name == "import_movies":
                    stage_outputs["movies_job_uid"] = result.job_uid
                elif step_name == "motion_correction":
                    stage_outputs["motion_correction_job_uid"] = result.job_uid
                elif step_name == "ctf_estimation":
                    stage_outputs["ctf_job_uid"] = result.job_uid
                elif step_name == "micrograph_selection":
                    stage_outputs["micrograph_selection_job_uid"] = result.job_uid
        
        return stage_outputs
    
    def _validate_preprocessing_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the preprocessing workflow actually completed successfully.
        
        Args:
            stage_outputs: Dictionary of stage outputs to validate
            
        Returns:
            Dictionary with 'success' boolean and 'error' message if failed
        """
        # Check if any jobs were executed
        required_jobs = [
            ("import_movies", stage_outputs.get("movies_job_uid")),
            ("motion_correction", stage_outputs.get("motion_correction_job_uid")),
            ("ctf_estimation", stage_outputs.get("ctf_job_uid")),
            ("micrograph_selection", stage_outputs.get("micrograph_selection_job_uid"))
        ]
        
        missing_jobs = []
        for job_name, job_uid in required_jobs:
            if job_uid is None:
                missing_jobs.append(job_name)
        
        if missing_jobs:
            error_msg = f"Preprocessing workflow failed - the following jobs were not executed: {', '.join(missing_jobs)}. " \
                       f"The agent may have completed without actually running the CryoSPARC jobs. " \
                       f"Check the agent's reasoning and ensure all tools are being called correctly."
            return {
                "success": False,
                "error": error_msg
            }
        
        # All required jobs have UIDs, validation passed
        return {
            "success": True,
            "error": None
        }
    
    def _save_preprocessing_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """
        Save preprocessing results to a JSON file.
        
        Args:
            stage_outputs: Dictionary of stage outputs
            context: Workflow context
            success: Whether the preprocessing was successful
            
        Returns:
            Path to the saved JSON file
        """
        import datetime
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Get the final selection job output directory
        final_job_uid = stage_outputs.get("micrograph_selection_job_uid")
        micrograph_output_directory = None
        output_summary = None
        
        if final_job_uid:
            try:
                job_dir_info = self.cryosparc_tools.get_job_output_directory(
                    context.project_uid,
                    final_job_uid
                )
                micrograph_output_directory = job_dir_info.get("job_directory")
                output_summary = job_dir_info.get("outputs", [])
                self.logger.info(f"Retrieved job directory for {final_job_uid}: {micrograph_output_directory}")
            except Exception as e:
                self.logger.warning(f"Could not retrieve job output directory: {e}")
                micrograph_output_directory = None
        
        # Create preprocessing results dictionary
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "completed" if success else "failed"
        
        preprocessing_results = {
            "stage": "preprocessing",
            "status": status,
            "timestamp": timestamp,
            "project_uid": context.project_uid,
            "workspace_uid": context.workspace_uid,
            "micrograph_location": {
                "description": "Location of processed micrographs in CryoSPARC",
                "project_uid": context.project_uid,
                "workspace_uid": context.workspace_uid,
                "final_selection_job_uid": final_job_uid,
                "output_directory": micrograph_output_directory,
                "path_pattern": f"Project {context.project_uid}, Workspace {context.workspace_uid}, Job {final_job_uid}",
                "output_summary": output_summary
            },
            "job_uids": {
                "import_movies": stage_outputs.get("movies_job_uid"),
                "motion_correction": stage_outputs.get("motion_correction_job_uid"),
                "ctf_estimation": stage_outputs.get("ctf_job_uid"),
                "micrograph_selection": stage_outputs.get("micrograph_selection_job_uid")
            },
            "outputs": {
                "selected_micrographs": stage_outputs.get("selected_micrographs"),
                "ctf_parameters": stage_outputs.get("ctf_parameters")
            },
            "usage_notes": {
                "next_stage": "particle_picking",
                "final_selection_job_uid_usage": "Use the final_selection_job_uid as input for particle picking stage",
                "micrograph_location_usage": "This job UID contains the selected micrographs for further processing",
                "output_directory_usage": "The output_directory contains the filesystem path to the micrograph files"
            }
        }
        
        # Save to JSON file
        output_file = output_dir / f"preprocessing_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(preprocessing_results, f, indent=2)
        
        self.logger.info(f"Preprocessing results saved to {output_file}")
        return str(output_file)


class ParticlePickingAgent(StageAgent):
    """Specialized agent for particle picking stage using modular architecture."""
    
    def __init__(self, config_path: str):
        super().__init__("particle_picking", config_path)
    
    def initialize(self) -> bool:
        """Initialize the particle picking agent with modular architecture."""
        try:
            # Load stage-specific configuration with master config
            master_config_path = "configs/master_config.json"
            config_loader = ConfigLoader(self.config_path, master_config_path)
            self.config = config_loader.load_config()
            
            # Initialize CryoSPARC tools
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            
            # Initialize modular picking agent
            self.modular_agent = ModularPickingAgent(
                cryosparc_tools=self.cryosparc_tools,
                config=self.config
            )
            
            # Initialize picking workflow
            self.modular_workflow = PickingWorkflow(
                agent=self.modular_agent,
                config=self.config
            )
            
            self.logger.info(f"Stage agent {self.stage_name} initialized successfully with modular architecture")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the particle picking stage using modular workflow."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting {self.get_stage_description()}")
            
            # Get micrograph job UID from preprocessing stage outputs
            preprocessing_outputs = context.stage_outputs.get(WorkflowStage.PREPROCESSING, {})
            micrographs_job_uid = preprocessing_outputs.get("micrograph_selection_job_uid")
            
            if not micrographs_job_uid:
                error_msg = "No micrograph_selection_job_uid found in preprocessing outputs"
                self.logger.error(error_msg)
                return StageResult(
                    stage=WorkflowStage.PARTICLE_PICKING,
                    success=False,
                    error=error_msg,
                    execution_time=time.time() - start_time
                )
            
            # Get particle diameter from config
            particle_diameter = getattr(self.config.workflow, "particle_diameter", 180.0)
            diameter_max = getattr(self.config.workflow, "diameter_max", None)
            
            self.logger.info(f"Running blob picker GPU with diameter={particle_diameter}Å (max={diameter_max or particle_diameter*2.0}Å), micrographs={micrographs_job_uid}")
            
            # Execute the particle picking workflow using modular architecture
            results = self.modular_workflow.run(
                micrographs_job_uid=micrographs_job_uid,
                particle_diameter=particle_diameter,
                diameter_max=diameter_max,
                conversation_id=conversation_id
            )
            
            # Parse results and extract stage outputs
            stage_outputs = self._parse_modular_picking_results(results)
            stage_outputs["micrographs_job_uid"] = micrographs_job_uid
            stage_outputs["particle_diameter"] = particle_diameter
            
            # Validate that jobs were actually executed
            validation_result = self._validate_picking_results(stage_outputs)
            
            # Save picking results to JSON file
            result_file_path = self._save_picking_results(stage_outputs, context, validation_result["success"])
            self.logger.info(f"Particle picking results saved to: {result_file_path}")
            print(f"📄 Particle picking results saved to: {result_file_path}")
            
            # Add result file path to stage outputs
            stage_outputs["result_file"] = result_file_path
            
            execution_time = time.time() - start_time
            
            # Return result based on validation
            if not validation_result["success"]:
                self.logger.error(f"Particle picking validation failed: {validation_result['error']}")
                return StageResult(
                    stage=WorkflowStage.PARTICLE_PICKING,
                    success=False,
                    stage_outputs=stage_outputs,
                    error=validation_result["error"],
                    execution_time=execution_time
                )
            
            return StageResult(
                stage=WorkflowStage.PARTICLE_PICKING,
                success=True,
                stage_outputs=stage_outputs,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Particle picking stage failed: {e}")
            
            return StageResult(
                stage=WorkflowStage.PARTICLE_PICKING,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def get_stage_description(self) -> str:
        return "Particle Picking: Detect and extract particles from micrographs using blob picker"
    
    def get_required_inputs(self) -> List[str]:
        return ["micrograph_selection_job_uid"]
    
    def _parse_modular_picking_results(self, results: List) -> Dict[str, Any]:
        """Parse modular picking results to extract stage outputs."""
        stage_outputs = {
            "blob_picker_job_uid": None,
            "picked_particles": None
        }
        
        # Extract job UIDs from modular workflow results
        for result in results:
            step_name = result.step.value
            if result.success and result.job_uid:
                if step_name == "blob_picker":
                    stage_outputs["blob_picker_job_uid"] = result.job_uid
        
        return stage_outputs
    
    def _validate_picking_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the particle picking workflow completed successfully."""
        picker_job_uid = stage_outputs.get("blob_picker_job_uid")
        
        if not picker_job_uid:
            return {
                "success": False,
                "error": "Blob picker job was not executed - no job UID found"
            }
        
        return {
            "success": True,
            "error": None
        }
    
    def _save_picking_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """Save particle picking results to a JSON file."""
        import datetime
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Create picking results dictionary
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "completed" if success else "failed"
        
        picking_results = {
            "stage": "particle_picking",
            "status": status,
            "timestamp": timestamp,
            "project_uid": context.project_uid,
            "workspace_uid": context.workspace_uid,
            "input_micrographs_job_uid": stage_outputs.get("micrographs_job_uid"),
            "particle_diameter": stage_outputs.get("particle_diameter"),
            "job_uids": {
                "blob_picker": stage_outputs.get("blob_picker_job_uid")
            },
            "outputs": {
                "picked_particles": stage_outputs.get("picked_particles")
            },
            "usage_notes": {
                "next_stage": "2d_classification or extraction",
                "picker_job_uid_usage": "Use the blob_picker job UID for particle extraction or 2D classification",
                "particle_coordinates": "Particle coordinates are stored in the blob_picker job output"
            }
        }
        
        # Save to JSON file
        output_file = output_dir / f"particle_picking_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(picking_results, f, indent=2)
        
        self.logger.info(f"Particle picking results saved to {output_file}")
        return str(output_file)


class ReconstructionAgent(StageAgent):
    """Specialized agent for 3D reconstruction stage (placeholder - not yet implemented)."""
    
    def __init__(self, config_path: str):
        super().__init__("reconstruction", config_path)
    
    def initialize(self) -> bool:
        """Initialize the reconstruction agent (placeholder)."""
        try:
            # Load basic configuration
            master_config_path = "configs/master_config.json"
            config_loader = ConfigLoader(self.config_path, master_config_path)
            self.config = config_loader.load_config()
            
            # Initialize CryoSPARC tools
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            
            # Note: No modular reconstruction agent yet - this is a placeholder
            self.modular_agent = None
            self.modular_workflow = None
            
            self.logger.info(f"Stage agent {self.stage_name} initialized (placeholder - not yet implemented)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
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
    
    def _reconstruct_stage_outputs_from_cache(self, stage: WorkflowStage, job_uids: Dict[str, str]) -> Dict[str, Any]:
        """
        Reconstruct stage_outputs from cached job_uids with proper key names.
        
        The saved JSON has keys like "micrograph_selection", but the agents expect
        keys like "micrograph_selection_job_uid". This method adds the "_job_uid" suffix.
        
        Args:
            stage: The workflow stage
            job_uids: Dictionary from cached JSON (keys without "_job_uid" suffix)
            
        Returns:
            Dictionary with properly formatted keys that agents expect
        """
        if stage == WorkflowStage.PREPROCESSING:
            return {
                "movies_job_uid": job_uids.get("import_movies"),
                "motion_correction_job_uid": job_uids.get("motion_correction"),
                "ctf_job_uid": job_uids.get("ctf_estimation"),
                "micrograph_selection_job_uid": job_uids.get("micrograph_selection"),
                "selected_micrographs": None,
                "ctf_parameters": None
            }
        elif stage == WorkflowStage.PARTICLE_PICKING:
            return {
                "blob_picker_job_uid": job_uids.get("blob_picker"),
                "picked_particles": None
            }
        else:
            # For other stages, return as-is
            return job_uids
    
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
            
            # Check if stage output already exists
            existing_output = check_stage_output_exists(stage)
            if existing_output:
                self.logger.info(f"Stage {stage_name} already completed, skipping execution")
                print(f"✅ Stage {stage_name} already completed!")
                print(f"   📄 Output file: {existing_output['file_path']}")
                print(f"   📅 Completed at: {existing_output['timestamp']}")
                print(f"   ℹ️  Skipping execution to avoid re-running")
                
                # Reconstruct stage_outputs with proper key names
                job_uids = existing_output['data'].get('job_uids', {})
                stage_outputs = self._reconstruct_stage_outputs_from_cache(stage, job_uids)
                
                # Create a successful stage result from existing output
                stage_result = StageResult(
                    stage=stage,
                    success=True,
                    stage_outputs=stage_outputs,
                    execution_time=0.0,
                    reasoning="Stage skipped - output already exists"
                )
                self.stage_results.append(stage_result)
                self.workflow_context.stage_outputs[stage] = stage_result.stage_outputs
                continue
            
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
            
            # Check if stage output already exists
            existing_output = check_stage_output_exists(stage)
            if existing_output:
                self.logger.info(f"Stage {stage_name} already completed, skipping execution")
                print(f"✅ Stage {stage_name} already completed!")
                print(f"   📄 Output file: {existing_output['file_path']}")
                print(f"   📅 Completed at: {existing_output['timestamp']}")
                print(f"   ℹ️  Skipping execution to avoid re-running")
                
                # Reconstruct stage_outputs with proper key names
                job_uids = existing_output['data'].get('job_uids', {})
                stage_outputs = self._reconstruct_stage_outputs_from_cache(stage, job_uids)
                
                # Create a successful stage result from existing output
                stage_result = StageResult(
                    stage=stage,
                    success=True,
                    stage_outputs=stage_outputs,
                    execution_time=0.0,
                    reasoning="Stage skipped - output already exists"
                )
                self.stage_results.append(stage_result)
                self.workflow_context.stage_outputs[stage] = stage_result.stage_outputs
                continue
            
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
            if hasattr(agent, 'modular_agent') and agent.modular_agent:
                # Modular agents don't have clear_reasoning_history method
                # They handle memory differently
                pass
