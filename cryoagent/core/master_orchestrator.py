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

# Import modular agents (imported dynamically to support both RELION and CryoSPARC)
from ..config.config_loader import ConfigLoader, CryoAgentConfig
from ..tools.cryosparc_tools import CryoSPARCTools
from ..utils.general_llm_logger import GeneralLLMLogger


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
        self.backend_type = None  # Will be set during initialization
    
    def initialize(self) -> bool:
        """Initialize the preprocessing agent with modular architecture."""
        try:
            # Load stage-specific configuration with master config
            master_config_path = "configs/master_config.json"
            config_loader = ConfigLoader(self.config_path, master_config_path)
            self.config = config_loader.load_config()
            
            # Initialize the modular agent based on config (determines backend automatically)
            self._initialize_modular_agent(config_loader)
            
            # Initialize preprocessing workflow based on backend type
            self._initialize_preprocessing_workflow()
            
            self.logger.info(f"Stage agent {self.stage_name} initialized successfully with {self.backend_type} backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def _initialize_modular_agent(self, config_loader):
        """Initialize the appropriate modular agent based on configuration."""
        # The ConfigLoader determines the backend type from the config path
        # RELION configs are in configs/relion/, CryoSPARC in configs/cryosparc/
        
        if "relion" in self.config_path.lower():
            # Import and create RELION-specific agent
            from .relion_preprocessing.preprocessing_agent import PreprocessingAgent as RelionPreprocessingAgent
            
            self.cryosparc_tools = None  # No CryoSPARC tools for RELION
            self.modular_agent = RelionPreprocessingAgent(
                config=self.config
            )
            self.backend_type = "RELION"
            
        else:
            # Default to CryoSPARC preprocessing
            from .cryosparc_preprocessing.preprocessing_agent import PreprocessingAgent as CryoSPARCPreprocessingAgent
            
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            self.modular_agent = CryoSPARCPreprocessingAgent(
                cryosparc_tools=self.cryosparc_tools,
                config=self.config
            )
            self.backend_type = "CryoSPARC"
        
        # Set stage name and workflow type for conversation logging
        self.modular_agent.stage_name = "preprocessing"
        self.modular_agent.workflow_type = "cryoem"
    
    def _initialize_preprocessing_workflow(self):
        """Initialize the appropriate preprocessing workflow based on backend type."""
        if self.backend_type == "RELION":
            from .relion_preprocessing.preprocessing_workflow import PreprocessingWorkflow as RelionPreprocessingWorkflow
            self.modular_workflow = RelionPreprocessingWorkflow(
                agent=self.modular_agent,
                config=self.config
            )
        else:
            from .cryosparc_preprocessing.preprocessing_workflow import PreprocessingWorkflow as CryoSPARCPreprocessingWorkflow
            self.modular_workflow = CryoSPARCPreprocessingWorkflow(
                agent=self.modular_agent,
                config=self.config
            )
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the pre-processing stage using modular workflow."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting {self.get_stage_description()}")
            
            # Execute the preprocessing workflow using modular architecture
            results = self.modular_workflow.run(conversation_id=conversation_id)
            
            # Delegate result handling to the modular agent (backend-specific logic)
            # This keeps the master orchestrator backend-agnostic
            stage_outputs = self.modular_agent.process_workflow_results(
                results=results,
                context=context
            )
            
            # Validate that jobs were actually executed (backend-specific)
            validation_result = self.modular_agent.validate_results(stage_outputs)
            
            # Save results (backend-specific)
            result_file_path = self.modular_agent.save_results(
                stage_outputs=stage_outputs,
                context=context,
                success=validation_result["success"]
            )
            
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


class RelionPreprocessingAgent(StageAgent):
    """Specialized agent for RELION pre-processing stage using modular architecture."""
    
    def __init__(self, config_path: str):
        super().__init__("preprocessing", config_path)
    
    def initialize(self) -> bool:
        """Initialize the RELION preprocessing agent with modular architecture."""
        try:
            # Load stage-specific configuration with master config
            master_config_path = "configs/master_config.json"
            config_loader = ConfigLoader(self.config_path, master_config_path)
            self.config = config_loader.load_config()
            
            # Initialize RELION preprocessing agent (no CryoSPARC tools needed)
            from .relion_preprocessing.preprocessing_agent import PreprocessingAgent as RelionPreprocessingAgent
            self.modular_agent = RelionPreprocessingAgent(
                config=self.config
            )
            
            # Set stage name and workflow type for conversation logging
            self.modular_agent.stage_name = "preprocessing"
            self.modular_agent.workflow_type = "cryoem"
            
            # Initialize preprocessing workflow
            self.modular_workflow = RelionPreprocessingWorkflow(
                agent=self.modular_agent,
                config=self.config
            )
            
            self.logger.info(f"RELION stage agent {self.stage_name} initialized successfully with modular architecture")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RELION stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the RELION pre-processing stage using modular workflow."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting {self.get_stage_description()}")
            
            # Execute the preprocessing workflow using modular architecture
            results = self.modular_workflow.run(conversation_id=conversation_id)
            
            # Delegate result handling to the modular agent (backend-specific logic)
            stage_outputs = self.modular_agent.process_workflow_results(
                results=results,
                context=context
            )
            
            # Validate that jobs were actually executed (backend-specific)
            validation_result = self.modular_agent.validate_results(stage_outputs)
            
            # Save results (backend-specific)
            result_file_path = self.modular_agent.save_results(
                stage_outputs=stage_outputs,
                context=context,
                success=validation_result["success"]
            )
            
            self.logger.info(f"RELION preprocessing results saved to: {result_file_path}")
            print(f"📄 RELION preprocessing results saved to: {result_file_path}")
            
            # Add result file path to stage outputs
            stage_outputs["result_file"] = result_file_path
            
            execution_time = time.time() - start_time
            
            # Return result based on validation
            if not validation_result["success"]:
                self.logger.error(f"RELION pre-processing validation failed: {validation_result['error']}")
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
            self.logger.error(f"RELION pre-processing stage failed: {e}")
            
            return StageResult(
                stage=WorkflowStage.PREPROCESSING,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    def get_stage_description(self) -> str:
        return "RELION Pre-processing: Import movies, motion correction, CTF estimation, and micrograph selection"
    
    def get_required_inputs(self) -> List[str]:
        return ["movies_path", "pixel_size", "voltage", "cs_mm", "q0", "beamtilt_x", "beamtilt_y"]


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
            
            # Import and initialize modular picking agent dynamically
            from .cryosparc_picking.picking_agent import PickingAgent as ModularPickingAgent
            from .cryosparc_picking.picking_workflow import PickingWorkflow
            
            self.modular_agent = ModularPickingAgent(
                cryosparc_tools=self.cryosparc_tools,
                config=self.config
            )
            
            # Set stage name and workflow type for conversation logging
            self.modular_agent.stage_name = "particle_picking"
            self.modular_agent.workflow_type = "cryoem"
            
            # Initialize picking workflow with stage config path
            self.modular_workflow = PickingWorkflow(
                agent=self.modular_agent,
                config=self.config,
                stage_config_path=self.config_path
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
            
            self.logger.info(f"Running particle picking workflow (parameters from {self.config_path})")
            
            # Execute the particle picking workflow using modular architecture
            # The workflow will read all parameters from the stage config file
            results = self.modular_workflow.run(
                micrographs_job_uid=micrographs_job_uid,
                conversation_id=conversation_id
            )
            
            # Delegate result handling to the modular agent (backend-specific logic)
            stage_outputs = self.modular_agent.process_workflow_results(
                results=results,
                context=context
            )
            stage_outputs["micrographs_job_uid"] = micrographs_job_uid
            
            # Validate that jobs were actually executed (backend-specific)
            validation_result = self.modular_agent.validate_results(stage_outputs)
            
            # Save results (backend-specific)
            result_file_path = self.modular_agent.save_results(
                stage_outputs=stage_outputs,
                context=context,
                success=validation_result["success"]
            )
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
        return "Particle Picking: Detect particles (blob picker), extract them, and perform 2D classification"
    
    def get_required_inputs(self) -> List[str]:
        return ["micrograph_selection_job_uid"]


class ReconstructionAgent(StageAgent):
    """Specialized agent for 3D reconstruction stage."""
    
    def __init__(self, config_path: str):
        super().__init__("reconstruction", config_path)
    
    def initialize(self) -> bool:
        """Initialize the reconstruction agent."""
        try:
            from .cryosparc_reconstruction import ReconstructionAgent as ModularReconstructionAgent
            from .cryosparc_reconstruction import ReconstructionWorkflow
            
            # Load basic configuration
            master_config_path = "configs/master_config.json"
            config_loader = ConfigLoader(self.config_path, master_config_path)
            self.config = config_loader.load_config()
            
            # Initialize CryoSPARC tools
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            
            # Initialize modular reconstruction agent and workflow
            self.modular_agent = ModularReconstructionAgent(self.cryosparc_tools, self.config)
            self.modular_workflow = ReconstructionWorkflow(self.modular_agent, self.config, stage_config_path=self.config_path)
            
            # Set stage name and workflow type for conversation logging
            self.modular_agent.stage_name = "reconstruction"
            self.modular_agent.workflow_type = "cryoem"
            
            self.logger.info(f"Stage agent {self.stage_name} initialized with modular reconstruction agent")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the 3D reconstruction stage."""
        start_time = time.time()
        
        try:
            # Get particles job UID from context (from picking stage)
            particles_job_uid = context.stage_outputs.get("picked_particles")
            if not particles_job_uid:
                # Try alternative sources
                particles_job_uid = context.stage_outputs.get("selected_particles")
                if not particles_job_uid:
                    particles_job_uid = context.stage_outputs.get("final_selection_job_uid")
            
            # If still not found, try to read from particle picking output JSON file
            if not particles_job_uid:
                particles_job_uid = self._get_particles_from_output_file()
            
            if not particles_job_uid:
                return StageResult(
                    stage=WorkflowStage.RECONSTRUCTION,
                    success=False,
                    error="No particles job UID found in context or output files. Please complete particle picking first.",
                    execution_time=time.time() - start_time
                )
            
            self.logger.info(f"Starting 3D reconstruction with particles from job {particles_job_uid}")
            
            # Check if refinement is enabled in config
            refinement_type = self.modular_workflow.workflow_params.get('refinement_type', 'none')
            run_refinement = refinement_type != 'none'
            
            if run_refinement:
                self.logger.info(f"Refinement enabled: {refinement_type}")
            else:
                self.logger.info("Refinement disabled (set refinement.type in config to enable)")
            
            # Run the modular workflow
            results = self.modular_workflow.run(
                particles_job_uid=particles_job_uid,
                conversation_id=conversation_id,
                run_refinement=run_refinement
            )
            
            # Extract stage outputs
            stage_outputs = self._extract_reconstruction_outputs(results)
            
            # Validate results
            validation = self._validate_reconstruction_results(stage_outputs)
            
            if not validation["success"]:
                return StageResult(
                    stage=WorkflowStage.RECONSTRUCTION,
                    success=False,
                    error=validation["error"],
                    stage_outputs=stage_outputs,
                    execution_time=time.time() - start_time
                )
            
            # Save results to JSON
            output_file = self._save_reconstruction_results(stage_outputs, context, success=True)
            
            # Add output file path to stage_outputs for reference
            stage_outputs["output_file"] = output_file
            
            return StageResult(
                stage=WorkflowStage.RECONSTRUCTION,
                success=True,
                stage_outputs=stage_outputs,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            self.logger.error(f"3D reconstruction stage failed: {e}")
            return StageResult(
                stage=WorkflowStage.RECONSTRUCTION,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def get_stage_description(self) -> str:
        return "3D Reconstruction: Generate initial models using ab initio or homogeneous reconstruction"
    
    def get_required_inputs(self) -> List[str]:
        return ["picked_particles", "selected_particles"]
    
    def _get_particles_from_output_file(self) -> Optional[str]:
        """
        Read particles job UID from particle picking output JSON file.
        
        Returns:
            Particles job UID if found, None otherwise
        """
        import json
        import glob
        from pathlib import Path
        
        try:
            # Find the most recent particle picking results file
            outputs_path = Path("outputs")
            if not outputs_path.exists():
                return None
            
            # Search for particle picking result files
            result_files = list(outputs_path.glob("particle_picking_results_*.json"))
            if not result_files:
                return None
            
            # Get the most recent file
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            self.logger.info(f"Reading particles job UID from {latest_file}")
            
            # Read the JSON file
            with open(latest_file, 'r') as f:
                picking_results = json.load(f)
            
            # Try to get particles job UID from various fields
            # First priority: final_selection_job_uid (from select_2d_classes)
            particles_job_uid = picking_results.get("job_uids", {}).get("final_selection")
            
            if not particles_job_uid:
                # Second priority: selected_particles_job_uid in outputs
                particles_job_uid = picking_results.get("outputs", {}).get("selected_particles_job_uid")
            
            if not particles_job_uid:
                # Third priority: classified_particles (2D classification output)
                particles_job_uid = picking_results.get("outputs", {}).get("classified_particles")
            
            if not particles_job_uid:
                # Fourth priority: extracted_particles
                particles_job_uid = picking_results.get("outputs", {}).get("extracted_particles")
            
            if particles_job_uid:
                self.logger.info(f"Found particles job UID from output file: {particles_job_uid}")
            else:
                self.logger.warning("No particles job UID found in particle picking output file")
            
            return particles_job_uid
            
        except Exception as e:
            self.logger.warning(f"Failed to read particles job UID from output file: {e}")
            return None
    
    def _validate_reconstruction_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the 3D reconstruction workflow completed successfully."""
        ab_initio_job_uid = stage_outputs.get("ab_initio_job_uid")
        
        if not ab_initio_job_uid:
            return {
                "success": False,
                "error": "Ab initio reconstruction did not complete successfully"
            }
        
        return {
            "success": True,
            "error": None
        }
    
    def _save_reconstruction_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """Save 3D reconstruction results to a JSON file."""
        import datetime
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Create reconstruction results dictionary
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "completed" if success else "failed"
        
        reconstruction_results = {
            "stage": "3d_reconstruction",
            "status": status,
            "timestamp": timestamp,
            "project_uid": context.project_uid,
            "workspace_uid": context.workspace_uid,
            "input_particles_job_uid": context.metadata.get("input_particles_job_uid"),
            "reconstruction_type": stage_outputs.get("reconstruction_type"),
            "job_uids": {
                "ab_initio": stage_outputs.get("ab_initio_job_uid"),
                "homogeneous_reconstruction": stage_outputs.get("homogeneous_reconstruction_job_uid"),
                "homogeneous_refinement": stage_outputs.get("homogeneous_refinement_job_uid"),
                "heterogeneous_refinement": stage_outputs.get("heterogeneous_refinement_job_uid"),
                "final_volume": stage_outputs.get("final_volume_job_uid")
            },
            "outputs": {
                "final_volume_job_uid": stage_outputs.get("final_volume_job_uid"),
                "volume_location": stage_outputs.get("volume_location"),
                "final_volume_absolute_path": stage_outputs.get("final_volume_absolute_path")
            },
            "usage_notes": {
                "next_stage": "refinement_or_analysis",
                "volume_usage": "Use the final_volume_job_uid for further refinement or analysis",
                "final_volume_path": "The final_volume_absolute_path field contains the absolute path to the job directory with the reconstructed volume"
            }
        }
        
        # Save to JSON file
        output_file = output_dir / f"3d_reconstruction_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(reconstruction_results, f, indent=2)
        
        self.logger.info(f"3D reconstruction results saved to {output_file}")
        return str(output_file)
    
    def _extract_reconstruction_outputs(self, results: List) -> Dict[str, Any]:
        """Extract job UIDs and metadata from reconstruction workflow results."""
        stage_outputs = {
            "ab_initio_job_uid": None,
            "homogeneous_reconstruction_job_uid": None,
            "homogeneous_refinement_job_uid": None,
            "heterogeneous_refinement_job_uid": None,
            "final_volume_job_uid": None,
            "reconstruction_type": "unknown"
        }
        
        # Extract job UIDs from results
        for result in results:
            step_name = result.step.value
            if result.success and result.job_uid:
                if step_name == "ab_initio_reconstruction":
                    stage_outputs["ab_initio_job_uid"] = result.job_uid
                    stage_outputs["final_volume_job_uid"] = result.job_uid
                    stage_outputs["reconstruction_type"] = "ab_initio"
                elif step_name == "homogeneous_reconstruction":
                    stage_outputs["homogeneous_reconstruction_job_uid"] = result.job_uid
                    stage_outputs["final_volume_job_uid"] = result.job_uid
                    stage_outputs["reconstruction_type"] = "homogeneous_reconstruction"
                elif step_name == "homogeneous_refinement":
                    stage_outputs["homogeneous_refinement_job_uid"] = result.job_uid
                    stage_outputs["final_volume_job_uid"] = result.job_uid
                    # Only update type if it's still initial reconstruction
                    if stage_outputs["reconstruction_type"] in ["ab_initio", "homogeneous_reconstruction"]:
                        stage_outputs["reconstruction_type"] = "refined_" + stage_outputs["reconstruction_type"]
                elif step_name == "heterogeneous_refinement":
                    stage_outputs["heterogeneous_refinement_job_uid"] = result.job_uid
                    stage_outputs["final_volume_job_uid"] = result.job_uid
                    stage_outputs["reconstruction_type"] = "heterogeneous_refined"
        
        # Get volume output directory if available
        final_volume_job_uid = stage_outputs.get("final_volume_job_uid")
        project_uid = getattr(self.config.workflow, "project_uid", None)
        
        if final_volume_job_uid and project_uid:
            try:
                job_info = self.cryosparc_tools.get_job_output_directory(project_uid, final_volume_job_uid)
                job_directory = job_info.get("job_directory")
                stage_outputs["volume_location"] = job_directory
                stage_outputs["volume_job_metadata"] = job_info
                
                # Add absolute paths for final volume
                if job_directory:
                    from pathlib import Path
                    job_path = Path(job_directory)
                    stage_outputs["final_volume_absolute_path"] = str(job_path.absolute())
                    
            except Exception as exc:
                self.logger.warning(
                    "Failed to resolve volume job directory for %s: %s",
                    final_volume_job_uid,
                    exc
                )
        
        return stage_outputs


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
            
            # Initialize stage agents (only for enabled stages)
            for stage_info in self.master_config["master_workflow"]["stages"]:
                stage_name = stage_info["name"]
                agent_group = stage_info["agent_group"]
                agent_class = stage_info["agent_class"]
                enabled = stage_info.get("enabled", False)
                
                # Skip disabled stages
                if not enabled:
                    self.logger.info(f"Skipping disabled stage: {stage_name}")
                    continue
                
                # Dynamically construct config file path
                config_path = f"configs/{agent_group}/{stage_name}_config.json"
                
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
                "extraction_job_uid": job_uids.get("particle_extraction"),
                "classification_2d_job_uid": job_uids.get("2d_classification"),
                "picked_particles": None,
                "extracted_particles": None,
                "classified_particles": None
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
        
        # Collect conversation log files from all stages
        conversation_log_files = self._collect_conversation_logs()
        
        # Generate workflow summary
        total_time = time.time() - self.start_time
        summary = self._generate_workflow_summary(total_time)
        
        # Add conversation log files to summary
        summary['conversation_log_files'] = conversation_log_files
        
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
        
        # Display conversation log files
        if 'conversation_log_files' in summary and summary['conversation_log_files']:
            print("\n💬 LLM Conversation Logs:")
            print("=" * 50)
            for stage_name, log_file in summary['conversation_log_files'].items():
                print(f"   {stage_name.replace('_', ' ').title()}: {log_file}")
            print()
    
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
    
    def _collect_conversation_logs(self) -> Dict[str, str]:
        """
        Collect conversation log files from all stage agents.
        
        Returns:
            Dictionary mapping stage names to conversation log file paths
        """
        conversation_logs = {}
        
        for stage_name, stage_agent in self.stage_agents.items():
            try:
                # Check if the stage agent has a modular agent with conversation logging
                if (hasattr(stage_agent, 'modular_agent') and 
                    stage_agent.modular_agent and 
                    hasattr(stage_agent.modular_agent, 'get_conversation_log_file')):
                    
                    log_file = stage_agent.modular_agent.get_conversation_log_file()
                    if log_file:
                        conversation_logs[stage_name] = log_file
                        self.logger.info(f"Found conversation log for {stage_name}: {log_file}")
                        
            except Exception as e:
                self.logger.warning(f"Could not collect conversation log for {stage_name}: {e}")
        
        return conversation_logs
    
    def set_general_llm_logger(self, logger: GeneralLLMLogger):
        """Set the general LLM logger for all stage agents."""
        for stage_agent in self.stage_agents.values():
            if hasattr(stage_agent, 'modular_agent') and stage_agent.modular_agent:
                stage_agent.modular_agent.set_general_llm_logger(logger)
