"""
Master Orchestrator for Multi-Stage CryoEM Workflow

This module provides the master orchestrator that coordinates separate specialized
ReAct agents for each stage of the cryoEM processing pipeline.

Updated to use modular agent architecture.
"""

import json
import os
import time
import logging
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Import modular agents (imported dynamically to support both RELION and CryoSPARC)
from ..config.config_loader import ConfigLoader, CryoAgentConfig
from ..config.microscope_override_updater import apply_microscope_overrides_if_enabled, apply_cryosift_overrides_if_enabled
from ..tools.cryosparc_tools import CryoSPARCTools
from ..utils.general_llm_logger import GeneralLLMLogger
from .transition_agent import TransitionAgent
from .summary_agent import SummaryAgent


class WorkflowStage(Enum):
    """Enumeration of workflow stages."""
    PREPROCESSING = "preprocessing"
    PARTICLE_PICKING = "particle_picking"
    OPTIMIZATION_2D = "optimization_2d"
    RECONSTRUCTION = "reconstruction"
    OPTIMIZATION = "optimization"
    HETEROGENEITY = "heterogeneity"
    POLISH = "polish"


def check_stage_output_exists(stage: WorkflowStage, outputs_dir: Optional[str] = None, default_outputs_dir: str = "outputs") -> Optional[Dict[str, Any]]:
    """
    Check if output file for a given stage already exists.
    Checks custom outputs_dir first, then falls back to default_outputs_dir.
    
    Args:
        stage: The workflow stage to check
        outputs_dir: Custom directory where output files are stored (checked first)
        default_outputs_dir: Default directory to check if outputs_dir is None or files not found
        
    Returns:
        Dictionary with output file information if exists, None otherwise
    """
    # First check custom outputs directory if provided
    if outputs_dir:
        outputs_path = Path(outputs_dir)
        if outputs_path.exists():
            result = _check_output_in_directory_orch(stage, outputs_path)
            if result:
                return result
    
    # Fall back to default outputs directory
    outputs_path = Path(default_outputs_dir)
    if not outputs_path.exists():
        return None
    
    return _check_output_in_directory_orch(stage, outputs_path)


def _check_output_in_directory_orch(stage: WorkflowStage, outputs_path: Path) -> Optional[Dict[str, Any]]:
    """Helper function to check for outputs in a specific directory."""
    
    # Map stage names to output file patterns
    # Note: Reconstruction files are named:
    # - reconstruction_results_cryosparc_{timestamp}.json (CryoSPARC from master orchestrator)
    # - 3d_reconstruction_results_relion_{timestamp}.json (RELION from master orchestrator)
    stage_patterns = {
        WorkflowStage.PREPROCESSING: "preprocessing_results_*.json",
        WorkflowStage.PARTICLE_PICKING: "particle_picking_results_*.json",
        WorkflowStage.OPTIMIZATION_2D: "2d_optimization_results_*.json",
        WorkflowStage.RECONSTRUCTION: ["reconstruction_results_cryosparc_*.json", "3d_reconstruction_results_relion_*.json"],
        WorkflowStage.OPTIMIZATION: "optimization_results_*.json",
        WorkflowStage.HETEROGENEITY: "heterogeneity_analysis_results_*.json",
        WorkflowStage.POLISH: "polish_results_*.json"
    }
    
    pattern = stage_patterns.get(stage)
    if not pattern:
        return None
    
    # Handle multiple patterns (for reconstruction which has different naming conventions)
    if isinstance(pattern, list):
        matching_files = []
        for p in pattern:
            search_pattern = str(outputs_path / p)
            matching_files.extend(glob.glob(search_pattern))
    else:
        # Single pattern
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
    
    def __init__(self, stage_name: str, config_path: str, master_config_path: Optional[str] = None):
        """
        Initialize the stage agent.
        
        Args:
            stage_name: Name of the stage
            config_path: Path to the stage-specific configuration file
        """
        self.stage_name = stage_name
        self.config_path = config_path
        self.master_config_path = master_config_path or "configs/master_config.json"
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
    
    def __init__(self, config_path: str, master_config_path: Optional[str] = None):
        super().__init__("preprocessing", config_path, master_config_path)
        self.backend_type = None  # Will be set during initialization
    
    def initialize(self) -> bool:
        """Initialize the preprocessing agent with modular architecture."""
        try:
            # Load stage-specific configuration with master config
            config_loader = ConfigLoader(self.config_path, self.master_config_path)
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
                config=self.config,
                master_config_path=self.master_config_path
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
    
    def __init__(self, config_path: str, master_config_path: Optional[str] = None):
        super().__init__("preprocessing", config_path, master_config_path)
    
    def initialize(self) -> bool:
        """Initialize the RELION preprocessing agent with modular architecture."""
        try:
            # Load stage-specific configuration with master config
            config_loader = ConfigLoader(self.config_path, self.master_config_path)
            self.config = config_loader.load_config()
            
            # Initialize RELION preprocessing agent (no CryoSPARC tools needed)
            from .relion_preprocessing.preprocessing_agent import PreprocessingAgent as RelionPreprocessingAgent
            from .relion_preprocessing.preprocessing_workflow import PreprocessingWorkflow as RelionPreprocessingWorkflow
            
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
    
    def __init__(self, config_path: str, master_config_path: Optional[str] = None):
        super().__init__("particle_picking", config_path, master_config_path)
    
    def initialize(self) -> bool:
        """Initialize the particle picking agent with modular architecture."""
        try:
            # Load stage-specific configuration with master config
            config_loader = ConfigLoader(self.config_path, self.master_config_path)
            self.config = config_loader.load_config()
            
            # Detect backend type from config path
            if "relion" in self.config_path.lower():
                # RELION backend
                from .relion_picking.picking_agent import PickingAgent as RelionPickingAgent
                from .relion_picking.picking_workflow import PickingWorkflow as RelionPickingWorkflow
                
                self.cryosparc_tools = None  # No CryoSPARC tools for RELION
                self.modular_agent = RelionPickingAgent(
                    config=self.config,
                    master_config_path=self.master_config_path
                )
                self.backend_type = "RELION"
                
                # Initialize RELION picking workflow
                self.modular_workflow = RelionPickingWorkflow(
                    agent=self.modular_agent,
                    config=self.config
                )
                
            else:
                # CryoSPARC backend (default)
                self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
                
                # Import and initialize modular picking agent dynamically
                from .cryosparc_picking.picking_agent import PickingAgent as ModularPickingAgent
                from .cryosparc_picking.picking_workflow import PickingWorkflow
                
                self.modular_agent = ModularPickingAgent(
                    cryosparc_tools=self.cryosparc_tools,
                    config=self.config
                )
                self.backend_type = "CryoSPARC"
                
                # Initialize picking workflow with stage config path
                self.modular_workflow = PickingWorkflow(
                    agent=self.modular_agent,
                    config=self.config,
                    stage_config_path=self.config_path
                )
            
            # Set stage name and workflow type for conversation logging
            self.modular_agent.stage_name = "particle_picking"
            self.modular_agent.workflow_type = "cryoem"
            
            self.logger.info(f"Stage agent {self.stage_name} initialized successfully with {self.backend_type} backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the particle picking stage using modular workflow."""
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting {self.get_stage_description()}")
            
            # Get micrograph input from preprocessing stage outputs
            preprocessing_outputs = context.stage_outputs.get(WorkflowStage.PREPROCESSING, {})
            
            # Get backend type (set during initialization)
            backend_type = getattr(self, 'backend_type', None)
            if not backend_type:
                # Fallback: detect from config path
                backend_type = "RELION" if "relion" in self.config_path.lower() else "CryoSPARC"
            
            # Handle input based on backend type
            if backend_type == "RELION":
                # RELION uses file path (selected_micrographs_star)
                selected_micrographs_star = preprocessing_outputs.get("selected_micrographs_star")
                
                if not selected_micrographs_star:
                    error_msg = "No selected_micrographs_star found in preprocessing outputs (required for RELION)"
                    self.logger.error(error_msg)
                    return StageResult(
                        stage=WorkflowStage.PARTICLE_PICKING,
                        success=False,
                        error=error_msg,
                        execution_time=time.time() - start_time
                    )
                
                if not os.path.exists(selected_micrographs_star):
                    error_msg = f"Selected micrographs file does not exist: {selected_micrographs_star}"
                    self.logger.error(error_msg)
                    return StageResult(
                        stage=WorkflowStage.PARTICLE_PICKING,
                        success=False,
                        error=error_msg,
                        execution_time=time.time() - start_time
                    )
                
                self.logger.info(f"Using RELION format: selected_micrographs_star={selected_micrographs_star}")
                
                # Execute the particle picking workflow using modular architecture
                # The workflow will read all parameters from the stage config file
                results = self.modular_workflow.run(
                    selected_micrographs_star=selected_micrographs_star,
                    conversation_id=conversation_id
                )
                
                # Store results for later processing
                self._relion_picking_results = results
                
            else:
                # CryoSPARC uses job UID
                micrographs_job_uid = preprocessing_outputs.get("micrograph_selection_job_uid")
                
                if not micrographs_job_uid:
                    error_msg = "No micrograph_selection_job_uid found in preprocessing outputs (required for CryoSPARC)"
                    self.logger.error(error_msg)
                    return StageResult(
                        stage=WorkflowStage.PARTICLE_PICKING,
                        success=False,
                        error=error_msg,
                        execution_time=time.time() - start_time
                    )
                
                self.logger.info(f"Using CryoSPARC format: micrographs_job_uid={micrographs_job_uid}")
                self.logger.info(f"Running particle picking workflow (parameters from {self.config_path})")
                
                # Execute the particle picking workflow using modular architecture
                # The workflow will read all parameters from the stage config file
                results = self.modular_workflow.run(
                    micrographs_job_uid=micrographs_job_uid,
                    conversation_id=conversation_id
                )
            
            # Delegate result handling to the modular agent (backend-specific logic)
            if backend_type == "RELION":
                # For RELION, check if agent has result processing methods
                if hasattr(self.modular_agent, 'process_workflow_results'):
                    # Extract workflow results if agent tracks workflow_state
                    workflow_results = []
                    if hasattr(self.modular_agent, 'workflow_state') and self.modular_agent.workflow_state:
                        for step_name, step_state in self.modular_agent.workflow_state.items():
                            workflow_results.append({
                                'step': step_name,
                                'completed': step_state.get('completed', False),
                                'job_dir': step_state.get('job_dir'),
                                'output_file': step_state.get('output_file')
                            })
                    
                    stage_outputs = self.modular_agent.process_workflow_results(
                        results=workflow_results if workflow_results else self._relion_picking_results,
                        context=context
                    )
                else:
                    # RELION picking - process workflow results
                    stage_outputs = self.modular_agent.process_workflow_results(
                        results=results,
                        context=context
                    )
                    stage_outputs["selected_micrographs_star"] = selected_micrographs_star
                
                # Validate that jobs were actually executed (backend-specific)
                validation_result = self.modular_agent.validate_results(stage_outputs)
                
                # Save results (backend-specific)
                result_file_path = self.modular_agent.save_results(
                    stage_outputs=stage_outputs,
                    context=context,
                    success=validation_result["success"]
                )
                
            else:
                # CryoSPARC
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
            if result_file_path:
                self.logger.info(f"Particle picking results saved to: {result_file_path}")
                print(f"📄 Particle picking results saved to: {result_file_path}")
                # Add result file path to stage outputs
                stage_outputs["result_file"] = result_file_path
            elif backend_type == "RELION":
                self.logger.info("RELION picking workflow completed (results saving not yet implemented)")
            
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
    
    def __init__(self, config_path: str, master_config_path: Optional[str] = None):
        super().__init__("reconstruction", config_path, master_config_path)
        self.backend_type = None  # Will be set during initialization
    
    def initialize(self) -> bool:
        """Initialize the reconstruction agent with modular architecture."""
        try:
            # Load basic configuration
            config_loader = ConfigLoader(self.config_path, self.master_config_path)
            self.config = config_loader.load_config()
            
            # Detect backend type from config path
            # RELION configs are in configs/relion/, CryoSPARC in configs/cryosparc/
            
            if "relion" in self.config_path.lower():
                # Import and create RELION-specific agent
                from .relion_reconstruction.reconstruction_agent import ReconstructionAgent as RelionReconstructionAgent
                from .relion_reconstruction.reconstruction_workflow import ReconstructionWorkflow
                
                self.cryosparc_tools = None  # No CryoSPARC tools for RELION
                self.modular_agent = RelionReconstructionAgent(
                    config=self.config,
                    master_config_path=self.master_config_path
                )
                self.backend_type = "RELION"
                
                # Initialize RELION reconstruction workflow
                self.modular_workflow = ReconstructionWorkflow(
                    agent=self.modular_agent,
                    config=self.config
                )
                
            else:
                # Default to CryoSPARC reconstruction
                from .cryosparc_reconstruction import ReconstructionAgent as ModularReconstructionAgent
                from .cryosparc_reconstruction import ReconstructionWorkflow
                
                # Initialize CryoSPARC tools
                self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
                
                # Initialize modular reconstruction agent and workflow
                self.modular_agent = ModularReconstructionAgent(self.cryosparc_tools, self.config)
                self.modular_workflow = ReconstructionWorkflow(self.modular_agent, self.config, stage_config_path=self.config_path)
                self.backend_type = "CryoSPARC"
            
            # Set stage name and workflow type for conversation logging
            self.modular_agent.stage_name = "reconstruction"
            self.modular_agent.workflow_type = "cryoem"
            
            self.logger.info(f"Stage agent {self.stage_name} initialized with {self.backend_type} backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """Execute the 3D reconstruction stage."""
        start_time = time.time()
        
        try:
            # Detect backend type
            backend_type = getattr(self, 'backend_type', None)
            if not backend_type:
                # Fallback: detect from config path
                backend_type = "RELION" if "relion" in self.config_path.lower() else "CryoSPARC"

            # Share upstream stage outputs with modular agent for context-aware auto-detection
            if hasattr(self.modular_agent, "set_context_stage_outputs"):
                self.modular_agent.set_context_stage_outputs(context.stage_outputs)
            elif hasattr(self.modular_agent, "context_stage_outputs"):
                self.modular_agent.context_stage_outputs = context.stage_outputs or {}
            
            if backend_type == "RELION":
                # For RELION, get final_star_file from picking results
                final_star_file = context.stage_outputs.get("final_star_file")
                if not final_star_file and isinstance(context.stage_outputs, dict):
                    for stage_dict in context.stage_outputs.values():
                        if isinstance(stage_dict, dict):
                            final_star_file = (
                                stage_dict.get("final_star_file")
                                or stage_dict.get("particles_star")
                                or stage_dict.get("star_file")
                            )
                            if final_star_file:
                                break
                if not final_star_file:
                    # Try to read from particle picking output JSON file
                    final_star_file = self._get_relion_particles_from_output_file()
                
                if not final_star_file:
                    return StageResult(
                        stage=WorkflowStage.RECONSTRUCTION,
                        success=False,
                        error="No final_star_file found in context or output files. Please complete particle picking first.",
                        execution_time=time.time() - start_time
                    )
                
                self.logger.info(f"Starting RELION 3D reconstruction with particles from: {final_star_file}")
                
                # Run the RELION reconstruction workflow with final_star_file
                results = self.modular_workflow.run(final_star_file=final_star_file, conversation_id=conversation_id)
                
                # Extract stage outputs from workflow results
                stage_outputs = {}
                if hasattr(self.modular_agent, 'workflow_state'):
                    # Get outputs from workflow state
                    if self.modular_agent.workflow_state.get("ab_initio_reconstruction", {}).get("completed"):
                        ab_initio_state = self.modular_agent.workflow_state["ab_initio_reconstruction"]
                        stage_outputs["initial_model"] = ab_initio_state.get("initial_model")
                        stage_outputs["ab_initio_job_dir"] = ab_initio_state.get("job_dir")
                    
                    if self.modular_agent.workflow_state.get("refinement_3d", {}).get("completed"):
                        refinement_state = self.modular_agent.workflow_state["refinement_3d"]
                        stage_outputs["refined_map"] = refinement_state.get("output_file")
                        stage_outputs["refinement_job_dir"] = refinement_state.get("job_dir")
                    
                    # Extract validation information if available
                    validation_state = self.modular_agent.workflow_state.get("validation", {})
                    if validation_state.get("import_job_uid_a") or validation_state.get("import_job_uid_b") or validation_state.get("job_dir"):
                        stage_outputs["validation"] = {}
                        if validation_state.get("import_job_uid_a"):
                            stage_outputs["validation"]["import_job_uid_a"] = validation_state.get("import_job_uid_a")
                        if validation_state.get("import_job_uid_b"):
                            stage_outputs["validation"]["import_job_uid_b"] = validation_state.get("import_job_uid_b")
                        if validation_state.get("job_dir"):
                            stage_outputs["validation"]["fsc_validation_job_uid"] = validation_state.get("job_dir")
                        if validation_state.get("fsc_results"):
                            stage_outputs["validation"]["fsc_results"] = validation_state.get("fsc_results")
                
                stage_outputs["final_star_file"] = final_star_file
                
                # Save results to JSON (RELION format)
                output_file = self._save_reconstruction_results(stage_outputs, context, success=True, backend_type="RELION")
                stage_outputs["output_file"] = output_file
                
                return StageResult(
                    stage=WorkflowStage.RECONSTRUCTION,
                    success=True,
                    stage_outputs=stage_outputs,
                    execution_time=time.time() - start_time
                )
            
            else:
                # CryoSPARC path - original logic with context-aware lookup
                # Check if 2D optimization stage ran and use its output
                optimization_2d_outputs = context.stage_outputs.get(WorkflowStage.OPTIMIZATION_2D)
                if isinstance(optimization_2d_outputs, StageResult):
                    optimization_2d_outputs = optimization_2d_outputs.stage_outputs
                
                # Prefer particles from 2D optimization if available
                if optimization_2d_outputs and optimization_2d_outputs.get("final_particles_job_uid"):
                    particles_job_uid = optimization_2d_outputs.get("final_particles_job_uid")
                    self.logger.info(f"Using particles from 2D optimization stage: {particles_job_uid}")
                else:
                    # Fallback to particle picking stage
                    particles_job_uid = self._resolve_particles_job_uid(context)
                
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
                
                # Save results to JSON (CryoSPARC format)
                output_file = self._save_reconstruction_results(stage_outputs, context, success=True, backend_type="CryoSPARC")
                
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
    
    def _resolve_particles_job_uid(self, context: WorkflowContext) -> Optional[str]:
        """
        Resolve the particles job UID needed for reconstruction by inspecting
        the workflow context and cached particle picking outputs.
        """
        stage_outputs_map = getattr(context, "stage_outputs", {}) or {}
        candidate_mappings: List[Dict[str, Any]] = []
        
        # Primary: particle picking stage outputs stored under enum key
        picking_outputs = stage_outputs_map.get(WorkflowStage.PARTICLE_PICKING)
        if isinstance(picking_outputs, StageResult):
            picking_outputs = picking_outputs.stage_outputs
        if picking_outputs:
            candidate_mappings.append(picking_outputs)
        
        # Secondary: some flows may store using string key
        if "particle_picking" in stage_outputs_map:
            alt_outputs = stage_outputs_map.get("particle_picking")
            if isinstance(alt_outputs, StageResult):
                alt_outputs = alt_outputs.stage_outputs
            if alt_outputs and alt_outputs not in candidate_mappings:
                candidate_mappings.append(alt_outputs)
        
        # Fallback: inspect the entire stage_outputs map for legacy keys
        candidate_mappings.append(stage_outputs_map)
        
        for mapping in candidate_mappings:
            value = self._extract_particles_uid_from_mapping(mapping)
            if value:
                self.logger.info(f"Resolved particles job UID '{value}' from workflow context")
                return value
        
        # Fallback: try to read from the most recent particle picking JSON output
        particles_job_uid = self._get_particles_from_output_file()
        if particles_job_uid:
            self.logger.info(f"Resolved particles job UID '{particles_job_uid}' from cached particle picking JSON")
        return particles_job_uid
    
    def _extract_particles_uid_from_mapping(self, mapping: Dict[str, Any]) -> Optional[str]:
        """Search for a usable particles job UID within a mapping structure."""
        if isinstance(mapping, StageResult):
            mapping = mapping.stage_outputs
        if not isinstance(mapping, dict):
            return None
        
        primary_keys = [
            "selected_particles_job_uid",
            "final_selection_job_uid",
            "selected_particles",
            "classified_particles",
            "extracted_particles",
            "picked_particles"
        ]
        
        for key in primary_keys:
            value = mapping.get(key)
            if value:
                return value
        
        outputs = mapping.get("outputs")
        if isinstance(outputs, dict):
            for key in primary_keys:
                value = outputs.get(key)
                if value:
                    return value
        
        job_uids = mapping.get("job_uids")
        if isinstance(job_uids, dict):
            for key in ["final_selection", "selected_particles", "classified_particles", "extracted_particles", "blob_picker"]:
                value = job_uids.get(key)
                if value:
                    return value
        
        result_file = mapping.get("result_file")
        if result_file:
            value = self._extract_particles_uid_from_file(result_file)
            if value:
                return value
        
        stage_outputs = mapping.get("stage_outputs")
        if isinstance(stage_outputs, dict):
            value = self._extract_particles_uid_from_mapping(stage_outputs)
            if value:
                return value
        
        return None
    
    def _extract_particles_uid_from_file(self, file_path: Union[str, Path]) -> Optional[str]:
        """Read a JSON file and attempt to extract a particles job UID."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning(f"Unable to read particle picking result file '{file_path}': {exc}")
            return None
        
        if not isinstance(data, dict):
            return None
        
        # Prefer outputs or stage_outputs sections if present
        for key in ["outputs", "stage_outputs", "data", "stageOutputs"]:
            section = data.get(key)
            value = self._extract_particles_uid_from_mapping(section) if isinstance(section, dict) else None
            if value:
                return value
        
        return self._extract_particles_uid_from_mapping(data)
    
    def _get_relion_particles_from_output_file(self) -> Optional[str]:
        """
        Read final_star_file from RELION particle picking output JSON file.
        
        Returns:
            Path to final STAR file if found, None otherwise
        """
        import json
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
            self.logger.info(f"Reading final_star_file from {latest_file}")
            
            # Read the JSON file
            with open(latest_file, 'r') as f:
                picking_results = json.load(f)
            
            # Check if this is a RELION output (has agent_type field)
            agent_type = picking_results.get("agent_type", "")
            if agent_type != "relion":
                self.logger.warning(f"Particle picking output is not from RELION (agent_type: {agent_type})")
                return None
            
            # Get final_star_file for RELION
            final_star_file = picking_results.get("final_star_file")
            
            if final_star_file:
                self.logger.info(f"Found final_star_file from output file: {final_star_file}")
            else:
                self.logger.warning("No final_star_file found in RELION particle picking output file")
            
            return final_star_file
            
        except Exception as e:
            self.logger.warning(f"Failed to read final_star_file from output file: {e}")
            return None
    
    def _get_particles_from_output_file(self) -> Optional[str]:
        """
        Read particles job UID from particle picking output JSON file (CryoSPARC format).
        
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
                particles_job_uid = picking_results.get("job_uids", {}).get("classified_particles")
            
            if not particles_job_uid:
                # Fourth priority: extracted_particles (extraction output)
                particles_job_uid = picking_results.get("job_uids", {}).get("extracted_particles")
            
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
                        stage_outputs["reconstruction_type"] = "homogeneous_refinement"
                elif step_name == "heterogeneous_refinement":
                    stage_outputs["heterogeneous_refinement_job_uid"] = result.job_uid
                    stage_outputs["final_volume_job_uid"] = result.job_uid
                    stage_outputs["reconstruction_type"] = "heterogeneous_refinement"
        
        # Get volume location from the final volume job
        if stage_outputs["final_volume_job_uid"]:
            try:
                job = self.cryosparc_tools.find_job(self.config.workflow.project_uid, stage_outputs["final_volume_job_uid"])
                if job:
                    job.refresh()
                    doc = getattr(job, "doc", {})
                    output_group = doc.get("output_result_groups", [{}])[0]
                    volume_location = output_group.get("output_files", {}).get("volume", [None])[0]
                    if volume_location:
                        stage_outputs["volume_location"] = volume_location
                        # Get absolute path
                        job_dir = getattr(job, "dir", "")
                        if job_dir and volume_location:
                            from pathlib import Path
                            volume_path = Path(job_dir) / volume_location
                            stage_outputs["final_volume_absolute_path"] = str(volume_path.absolute())
            except Exception as e:
                self.logger.warning(f"Could not get volume location: {e}")
        
        return stage_outputs
    
    def _save_reconstruction_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True, backend_type: str = "CryoSPARC") -> str:
        """Save 3D reconstruction results to a JSON file."""
        import datetime
        import os
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Create reconstruction results dictionary
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "completed" if success else "failed"
        
        if backend_type == "RELION":
            # RELION format: simplified structure matching particle_picking format
            # Get the folder containing the final density (refinement job directory)
            final_volume_folder = stage_outputs.get("refinement_job_dir")
            if not final_volume_folder:
                # Fallback to ab_initio job directory if refinement not available
                final_volume_folder = stage_outputs.get("ab_initio_job_dir")
            
            # Convert to absolute path if it's relative
            if final_volume_folder:
                if not os.path.isabs(final_volume_folder):
                    # Try to get RELION directory from modular_agent's relion_tools
                    relion_dir = None
                    if hasattr(self, 'modular_agent') and hasattr(self.modular_agent, 'relion_tools'):
                        relion_dir = self.modular_agent.relion_tools.relion_dir
                    elif hasattr(self, 'relion_dir'):
                        relion_dir = self.relion_dir
                    else:
                        # Try to get from config
                        try:
                            from ..config.config_loader import ConfigLoader
                            config_loader = ConfigLoader(self.config_path, self.master_config_path)
                            config = config_loader.load_config()
                            relion_dir = config.relion.relion_dir
                        except:
                            relion_dir = "."
                    
                    final_volume_folder = os.path.join(relion_dir, final_volume_folder)
                    final_volume_folder = os.path.abspath(final_volume_folder)
            
            reconstruction_results = {
                "timestamp": timestamp,
                "status": status,
                "stage": "3d_reconstruction",
                "agent_type": "relion",
                "project_uid": context.project_uid,
                "workspace_uid": context.workspace_uid,
                "final_volume_folder": final_volume_folder,
                "metadata": {
                    "workflow_type": getattr(context, 'workflow_type', context.metadata.get("workflow_type", "unknown")),
                    "start_time": getattr(context, 'start_time', context.metadata.get("start_time", None)),
                    "conversation_id": getattr(context, 'conversation_id', context.metadata.get("conversation_id", None))
                }
            }
            
            # Add validation information if available
            validation_info = stage_outputs.get("validation")
            if validation_info:
                validation_data = {}
                
                # Get job directory paths from CryoSPARC if available
                try:
                    # Try to get cryosparc_tools from modular_agent (for RELION with validation) or from self
                    cryosparc_tools = None
                    if hasattr(self, 'modular_agent') and hasattr(self.modular_agent, 'cryosparc_tools'):
                        cryosparc_tools = self.modular_agent.cryosparc_tools
                    elif hasattr(self, 'cryosparc_tools') and self.cryosparc_tools:
                        cryosparc_tools = self.cryosparc_tools
                    
                    if cryosparc_tools:
                        project_uid = context.project_uid
                        
                        # Import job A
                        import_job_uid_a = validation_info.get("import_job_uid_a")
                        if import_job_uid_a:
                            try:
                                job_info = cryosparc_tools.get_job_output_directory(project_uid, import_job_uid_a)
                                validation_data["import_job_a"] = {
                                    "job_uid": import_job_uid_a,
                                    "job_number": import_job_uid_a,  # Job UID contains job number (e.g., "J329")
                                    "job_folder": job_info.get("job_directory", "")
                                }
                            except Exception as e:
                                self.logger.warning(f"Could not get job directory for import job A {import_job_uid_a}: {e}")
                                validation_data["import_job_a"] = {
                                    "job_uid": import_job_uid_a,
                                    "job_number": import_job_uid_a
                                }
                        
                        # Import job B
                        import_job_uid_b = validation_info.get("import_job_uid_b")
                        if import_job_uid_b:
                            try:
                                job_info = cryosparc_tools.get_job_output_directory(project_uid, import_job_uid_b)
                                validation_data["import_job_b"] = {
                                    "job_uid": import_job_uid_b,
                                    "job_number": import_job_uid_b,
                                    "job_folder": job_info.get("job_directory", "")
                                }
                            except Exception as e:
                                self.logger.warning(f"Could not get job directory for import job B {import_job_uid_b}: {e}")
                                validation_data["import_job_b"] = {
                                    "job_uid": import_job_uid_b,
                                    "job_number": import_job_uid_b
                                }
                        
                        # FSC validation job
                        fsc_job_uid = validation_info.get("fsc_validation_job_uid")
                        if fsc_job_uid:
                            try:
                                job_info = cryosparc_tools.get_job_output_directory(project_uid, fsc_job_uid)
                                validation_data["fsc_validation"] = {
                                    "job_uid": fsc_job_uid,
                                    "job_number": fsc_job_uid,
                                    "job_folder": job_info.get("job_directory", "")
                                }
                            except Exception as e:
                                self.logger.warning(f"Could not get job directory for FSC validation job {fsc_job_uid}: {e}")
                                validation_data["fsc_validation"] = {
                                    "job_uid": fsc_job_uid,
                                    "job_number": fsc_job_uid
                                }
                        
                        # Add FSC results if available
                        if validation_info.get("fsc_results"):
                            validation_data["fsc_results"] = validation_info.get("fsc_results")
                        
                        if validation_data:
                            reconstruction_results["validation"] = validation_data
                except Exception as e:
                    self.logger.warning(f"Could not retrieve validation job information: {e}")
                    # Fallback: include basic validation info without job folders
                    validation_data = {}
                    if validation_info.get("import_job_uid_a"):
                        validation_data["import_job_a"] = {"job_uid": validation_info.get("import_job_uid_a"), "job_number": validation_info.get("import_job_uid_a")}
                    if validation_info.get("import_job_uid_b"):
                        validation_data["import_job_b"] = {"job_uid": validation_info.get("import_job_uid_b"), "job_number": validation_info.get("import_job_uid_b")}
                    if validation_info.get("fsc_validation_job_uid"):
                        validation_data["fsc_validation"] = {"job_uid": validation_info.get("fsc_validation_job_uid"), "job_number": validation_info.get("fsc_validation_job_uid")}
                    if validation_info.get("fsc_results"):
                        validation_data["fsc_results"] = validation_info.get("fsc_results")
                    if validation_data:
                        reconstruction_results["validation"] = validation_data
            
            # Save to JSON file with RELION naming convention
            output_file = output_dir / f"3d_reconstruction_results_relion_{timestamp}.json"
            
        else:
            # CryoSPARC format: original structure
            reconstruction_results = {
                "stage": "3d_reconstruction",
                "status": status,
                "timestamp": timestamp,
                "agent_type": "cryosparc",
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
            
            # Save to JSON file with CryoSPARC naming convention: reconstruction_results_cryosparc_{timestamp}.json
            output_file = output_dir / f"reconstruction_results_cryosparc_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(reconstruction_results, f, indent=2)
        
        self.logger.info(f"3D reconstruction results saved to {output_file}")
        return str(output_file)


class Optimizer2DAgent(StageAgent):
    """Specialized agent for 2D classification optimization stage."""
    
    def __init__(self, config_path: str, master_config_path: Optional[str] = None):
        super().__init__("optimization_2d", config_path, master_config_path)
        self.backend_type = None  # Will be set during initialization
    
    def initialize(self) -> bool:
        """Initialize the 2D optimization agent with modular architecture."""
        try:
            # Load basic configuration
            config_loader = ConfigLoader(self.config_path, self.master_config_path)
            self.config = config_loader.load_config()
            
            # 2D Optimization is only available for CryoSPARC
            from .cryosparc_2Doptimize.optimizer_2d_agent import Optimizer2DAgent as ModularOptimizer2DAgent
            from .cryosparc_2Doptimize.optimizer_2d_workflow import Optimizer2DWorkflow
            
            # Initialize CryoSPARC tools
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            
            # Initialize modular 2D optimizer agent and workflow
            self.modular_agent = ModularOptimizer2DAgent(self.cryosparc_tools, self.config)
            self.modular_workflow = Optimizer2DWorkflow(self.modular_agent, self.config, stage_config_path=self.config_path)
            self.backend_type = "CryoSPARC"
            
            # Set stage name and workflow type for conversation logging
            self.modular_agent.stage_name = "optimization_2d"
            self.modular_agent.workflow_type = "cryoem"
            
            self.logger.info(f"Stage agent {self.stage_name} initialized with {self.backend_type} backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """
        Execute the 2D optimization stage.
        
        Args:
            context: Workflow context with previous stage outputs
            conversation_id: Optional conversation identifier
            
        Returns:
            StageResult with 2D optimization outputs
        """
        start_time = time.time()
        try:
            # Get required inputs from previous stages
            stage_outputs_map = getattr(context, "stage_outputs", {}) or {}
            picking_outputs = stage_outputs_map.get(WorkflowStage.PARTICLE_PICKING)
            if isinstance(picking_outputs, StageResult):
                picking_outputs = picking_outputs.stage_outputs
            
            # Get particles job UID from particle picking stage
            particles_job_uid = (
                picking_outputs.get("final_selection_job_uid")
                or picking_outputs.get("selected_particles_job_uid")
                or picking_outputs.get("classification_2d_job_uid")
                or picking_outputs.get("extraction_job_uid")
                or picking_outputs.get("blob_picker_job_uid")
            )
            
            # Also check job_uids structure (from JSON files)
            if not particles_job_uid:
                job_uids = picking_outputs.get("job_uids", {})
                if isinstance(job_uids, dict):
                    particles_job_uid = (
                        job_uids.get("final_selection")
                        or job_uids.get("selected_particles")
                        or job_uids.get("classified_particles")
                        or job_uids.get("extracted_particles")
                    )
            
            # Also check outputs structure
            if not particles_job_uid:
                outputs = picking_outputs.get("outputs", {})
                if isinstance(outputs, dict):
                    particles_job_uid = outputs.get("selected_particles_job_uid")
            
            if not particles_job_uid:
                execution_time = time.time() - start_time
                return StageResult(
                    stage=WorkflowStage.OPTIMIZATION_2D,
                    success=False,
                    error="Missing required input: particles_job_uid from particle picking stage",
                    stage_outputs={},
                    execution_time=execution_time
                )
            
            # Execute 2D optimization workflow
            result = self.modular_workflow.execute_2d_optimization(
                particles_job_uid=particles_job_uid,
                conversation_id=conversation_id
            )
            
            # Extract outputs
            stage_outputs = {
                "final_particles_job_uid": result.final_particles_job_uid,
                "final_good_particles_count": result.final_good_particles_count,
                "final_good_particles_percentage": result.final_good_particles_percentage,
                "total_rounds": result.total_rounds,
                "workflow_summary": result.workflow_summary
            }
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Save results
            output_file = self._save_2d_optimization_results(stage_outputs, context, result.success, execution_time)
            stage_outputs["output_file"] = output_file
            
            return StageResult(
                stage=WorkflowStage.OPTIMIZATION_2D,
                success=result.success,
                error=result.error,
                stage_outputs=stage_outputs,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Failed to execute 2D optimization stage: {e}")
            return StageResult(
                stage=WorkflowStage.OPTIMIZATION_2D,
                success=False,
                error=str(e),
                stage_outputs={},
                execution_time=execution_time
            )
    
    def _save_2d_optimization_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True, execution_time: float = 0.0) -> str:
        """Save 2D optimization results to a JSON file."""
        import datetime
        from pathlib import Path
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "completed" if success else "failed"
        
        optimization_results = {
            "stage": "optimization_2d",
            "status": status,
            "timestamp": timestamp,
            "agent_type": "cryosparc",
            "project_uid": context.project_uid,
            "workspace_uid": context.workspace_uid,
            "execution_time": execution_time,
            "final_particles_job_uid": stage_outputs.get("final_particles_job_uid"),
            "final_good_particles_count": stage_outputs.get("final_good_particles_count"),
            "final_good_particles_percentage": stage_outputs.get("final_good_particles_percentage"),
            "total_rounds": stage_outputs.get("total_rounds"),
            "workflow_summary": stage_outputs.get("workflow_summary", {})
        }
        
        output_file = output_dir / f"2d_optimization_results_cryosparc_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        self.logger.info(f"2D optimization results saved to {output_file}")
        return str(output_file)
    
    def get_stage_description(self) -> str:
        return "2D Classification Optimization: Iteratively refine particle selection using 2D classification and CryoSift until 90% good particles"
    
    def get_required_inputs(self) -> List[str]:
        return ["particles_job_uid"]


class OptimizerAgent(StageAgent):
    """Specialized agent for box size optimization stage."""
    
    def __init__(self, config_path: str, master_config_path: Optional[str] = None):
        super().__init__("optimization", config_path, master_config_path)
        self.backend_type = None  # Will be set during initialization
    
    def initialize(self) -> bool:
        """Initialize the optimization agent with modular architecture."""
        try:
            # Load basic configuration
            config_loader = ConfigLoader(self.config_path, self.master_config_path)
            self.config = config_loader.load_config()
            
            # Optimization is only available for CryoSPARC
            from .cryosparc_optimize.optimizer_agent import OptimizerAgent as ModularOptimizerAgent
            from .cryosparc_optimize.optimizer_workflow import OptimizerWorkflow
            
            # Initialize CryoSPARC tools
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            
            # Initialize modular optimizer agent and workflow
            self.modular_agent = ModularOptimizerAgent(self.cryosparc_tools, self.config)
            self.modular_workflow = OptimizerWorkflow(self.modular_agent, self.config, stage_config_path=self.config_path)
            self.backend_type = "CryoSPARC"
            
            # Set stage name and workflow type for conversation logging
            self.modular_agent.stage_name = "optimization"
            self.modular_agent.workflow_type = "cryoem"
            
            self.logger.info(f"Stage agent {self.stage_name} initialized with {self.backend_type} backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """
        Execute the optimization stage.
        
        Args:
            context: Workflow context with previous stage outputs
            conversation_id: Optional conversation identifier
            
        Returns:
            StageResult with optimization outputs
        """
        start_time = time.time()
        try:
            # Get required inputs from previous stages
            stage_outputs_map = getattr(context, "stage_outputs", {}) or {}
            reconstruction_outputs = stage_outputs_map.get(WorkflowStage.RECONSTRUCTION)
            if isinstance(reconstruction_outputs, StageResult):
                reconstruction_outputs = reconstruction_outputs.stage_outputs
            
            # Get final_volume_job_uid first (this is the key identifier)
            # Check both direct fields and nested job_uids/outputs structures (from JSON files)
            final_volume_job_uid = None
            if reconstruction_outputs:
                final_volume_job_uid = (
                    reconstruction_outputs.get("final_volume_job_uid")
                    or (reconstruction_outputs.get("job_uids", {}).get("final_volume") if isinstance(reconstruction_outputs.get("job_uids"), dict) else None)
                    or (reconstruction_outputs.get("outputs", {}).get("final_volume_job_uid") if isinstance(reconstruction_outputs.get("outputs"), dict) else None)
                    or reconstruction_outputs.get("homogeneous_refinement_job_uid")
                    or (reconstruction_outputs.get("job_uids", {}).get("homogeneous_refinement") if isinstance(reconstruction_outputs.get("job_uids"), dict) else None)
                )
            
            # refinement_job_uid and volume_job_uid should both be the final_volume_job_uid
            # (the final refined volume is used both as the source of initial resolution/box size
            #  and as the initial volume for new refinements)
            refinement_job_uid = final_volume_job_uid
            volume_job_uid = final_volume_job_uid
            
            # Get particles job UID (picking job for re-extraction)
            picking_outputs = stage_outputs_map.get(WorkflowStage.PARTICLE_PICKING)
            if isinstance(picking_outputs, StageResult):
                picking_outputs = picking_outputs.stage_outputs
            
            particles_job_uid = None
            if picking_outputs:
                particles_job_uid = (
                    picking_outputs.get("blob_picker_job_uid")
                    or picking_outputs.get("picking_job_uid")
                    or picking_outputs.get("particle_picking_job_uid")
                    or picking_outputs.get("final_selection_job_uid")  # From JSON: final_selection_job_uid
                    or picking_outputs.get("selected_particles_job_uid")
                )
            
            # Get micrographs job UID
            preprocessing_outputs = stage_outputs_map.get(WorkflowStage.PREPROCESSING)
            if isinstance(preprocessing_outputs, StageResult):
                preprocessing_outputs = preprocessing_outputs.stage_outputs
            
            micrographs_job_uid = None
            if preprocessing_outputs:
                micrographs_job_uid = (
                    preprocessing_outputs.get("micrograph_selection_job_uid")
                    or preprocessing_outputs.get("final_micrographs_job_uid")
                )
            
            if not all([refinement_job_uid, particles_job_uid, micrographs_job_uid, volume_job_uid]):
                missing = []
                if not refinement_job_uid:
                    missing.append("refinement_job_uid")
                if not particles_job_uid:
                    missing.append("particles_job_uid")
                if not micrographs_job_uid:
                    missing.append("micrographs_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                
                execution_time = time.time() - start_time
                return StageResult(
                    stage=WorkflowStage.OPTIMIZATION,
                    success=False,
                    error=f"Missing required inputs from previous stages: {', '.join(missing)}",
                    stage_outputs={},
                    execution_time=execution_time
                )
            
            # Execute optimization workflow
            result = self.modular_workflow.execute_optimization(
                refinement_job_uid=refinement_job_uid,
                particles_job_uid=particles_job_uid,
                micrographs_job_uid=micrographs_job_uid,
                volume_job_uid=volume_job_uid,
                conversation_id=conversation_id
            )
            
            # Extract outputs
            # Note: OptimizationResult uses 'job_uid' not 'best_job_uid'
            stage_outputs = {
                "optimization_job_uid": result.job_uid,
                "best_box_size": result.best_box_size,
                "best_resolution_angstroms": result.best_resolution,
                "tested_combinations": result.tested_combinations or [],
                "iterations": len(result.tested_combinations) if result.tested_combinations else 0
            }
            
            # Add final refinement job UID if available (non-uniform refinement with CTF refinement)
            if result.final_refinement_job_uid:
                stage_outputs["final_refinement_job_uid"] = result.final_refinement_job_uid
                stage_outputs["final_refinement_resolution"] = result.final_refinement_resolution
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Save results
            output_file = self._save_optimization_results(stage_outputs, context, result.success, execution_time)
            stage_outputs["output_file"] = output_file
            
            return StageResult(
                stage=WorkflowStage.OPTIMIZATION,
                success=result.success,
                error=result.error,
                stage_outputs=stage_outputs,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Failed to execute optimization stage: {e}")
            return StageResult(
                stage=WorkflowStage.OPTIMIZATION,
                success=False,
                error=str(e),
                stage_outputs={},
                execution_time=execution_time
            )
    
    def _save_optimization_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True, execution_time: float = 0.0) -> str:
        """Save optimization results to a JSON file."""
        import datetime
        from pathlib import Path
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "completed" if success else "failed"
        
        optimization_results = {
            "stage": "optimization",
            "status": status,
            "timestamp": timestamp,
            "agent_type": "cryosparc",
            "project_uid": context.project_uid,
            "workspace_uid": context.workspace_uid,
            "execution_time": execution_time,
            "best_box_size": stage_outputs.get("best_box_size"),
            "best_resolution_angstroms": stage_outputs.get("best_resolution_angstroms"),
            "best_job_uid": stage_outputs.get("optimization_job_uid"),
            "iterations": stage_outputs.get("iterations", 0),
            "tested_combinations": stage_outputs.get("tested_combinations", [])
        }
        
        output_file = output_dir / f"optimization_results_cryosparc_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        self.logger.info(f"Optimization results saved to {output_file}")
        return str(output_file)
    
    def get_stage_description(self) -> str:
        return "Box Size Optimization: Optimize particle extraction box size for improved 3D reconstruction resolution"
    
    def get_required_inputs(self) -> List[str]:
        return ["homogeneous_refinement_job_uid", "picked_particles", "micrograph_selection_job_uid", "volume_job_uid"]


class PolishAgent(StageAgent):
    """Specialized agent for polish refinement stage."""
    
    def __init__(self, config_path: str, master_config_path: Optional[str] = None):
        super().__init__("polish", config_path, master_config_path)
        self.backend_type = None  # Will be set during initialization
    
    def initialize(self) -> bool:
        """Initialize the polish agent with modular architecture."""
        try:
            # Load basic configuration
            config_loader = ConfigLoader(self.config_path, self.master_config_path)
            self.config = config_loader.load_config()
            
            # Polish is only available for CryoSPARC
            from .cryosparc_polish.polish_agent import PolishAgent as ModularPolishAgent
            from .cryosparc_polish.polish_workflow import PolishWorkflow
            
            # Initialize CryoSPARC tools
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            
            # Initialize modular polish agent and workflow
            self.modular_agent = ModularPolishAgent(self.cryosparc_tools, self.config)
            self.modular_workflow = PolishWorkflow(self.modular_agent, self.config, stage_config_path=self.config_path)
            self.backend_type = "CryoSPARC"
            
            # Set stage name and workflow type for conversation logging
            self.modular_agent.stage_name = "polish"
            self.modular_agent.workflow_type = "cryoem"
            
            self.logger.info(f"Stage agent {self.stage_name} initialized with {self.backend_type} backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """
        Execute the polish stage.
        
        Args:
            context: Workflow context with previous stage outputs
            conversation_id: Optional conversation identifier
            
        Returns:
            StageResult with polish outputs
        """
        start_time = time.time()
        try:
            # Polish stage verifies inputs internally by reading JSON files
            # It needs optimization_results and preprocessing_results
            
            # Execute polish workflow (it will verify inputs internally)
            results = self.modular_workflow.run(conversation_id=conversation_id)
            
            # Check if all steps completed successfully
            all_successful = all(r.success for r in results)
            
            # Find final refinement result
            final_refinement_result = None
            for result in reversed(results):
                if result.step.value == "final_refinement":
                    final_refinement_result = result
                    break
            
            best_job_uid = final_refinement_result.job_uid if final_refinement_result and final_refinement_result.success else None
            
            # Get final resolution from workflow summary
            workflow_summary = self.modular_workflow.get_workflow_summary()
            final_resolution = None
            if best_job_uid:
                try:
                    fsc_info = self.cryosparc_tools.get_refinement_fsc_info(
                        self.config.workflow.project_uid,
                        best_job_uid
                    )
                    if fsc_info.get("success"):
                        final_resolution = fsc_info.get("resolution_angstroms")
                except Exception:
                    pass
            
            # Extract outputs
            stage_outputs = {
                "best_job_uid": best_job_uid,
                "final_resolution": final_resolution,
                "workflow_summary": workflow_summary
            }
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Save results
            output_file = self.modular_workflow.save_results(execution_time)
            stage_outputs["output_file"] = output_file
            
            # Determine error message if any step failed
            error = None
            if not all_successful:
                failed_steps = [r for r in results if not r.success]
                error = f"Failed steps: {[r.step.value for r in failed_steps]}"
            
            return StageResult(
                stage=WorkflowStage.POLISH,
                success=all_successful,
                error=error,
                stage_outputs=stage_outputs,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Failed to execute polish stage: {e}")
            return StageResult(
                stage=WorkflowStage.POLISH,
                success=False,
                error=str(e),
                stage_outputs={},
                execution_time=execution_time
            )
    
    def get_stage_description(self) -> str:
        return "Polish Refinement: Final refinement steps with CTF refinement and motion correction to achieve best possible resolution"
    
    def get_required_inputs(self) -> List[str]:
        return ["optimization_results", "preprocessing_results"]


class HeterogeneityAgent(StageAgent):
    """Specialized agent for heterogeneity analysis stage."""
    
    def __init__(self, config_path: str, master_config_path: Optional[str] = None):
        super().__init__("heterogeneity", config_path, master_config_path)
        self.backend_type = None  # Will be set during initialization
    
    def initialize(self) -> bool:
        """Initialize the heterogeneity analysis agent with modular architecture."""
        try:
            # Load basic configuration
            config_loader = ConfigLoader(self.config_path, self.master_config_path)
            self.config = config_loader.load_config()
            
            # Heterogeneity analysis is only available for CryoSPARC
            from .cryosparc_heterogeneity.heterogeneity_agent import HeterogeneityAgent as ModularHeterogeneityAgent
            from .cryosparc_heterogeneity.heterogeneity_workflow import HeterogeneityWorkflow
            
            # Initialize CryoSPARC tools
            self.cryosparc_tools = CryoSPARCTools(self.config.cryosparc)
            
            # Initialize modular heterogeneity agent and workflow
            self.modular_agent = ModularHeterogeneityAgent(self.cryosparc_tools, self.config)
            self.modular_workflow = HeterogeneityWorkflow(self.modular_agent, self.config, stage_config_path=self.config_path)
            self.backend_type = "CryoSPARC"
            
            # Set stage name and workflow type for conversation logging
            self.modular_agent.stage_name = "heterogeneity"
            self.modular_agent.workflow_type = "cryoem"
            
            self.logger.info(f"Stage agent {self.stage_name} initialized with {self.backend_type} backend")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stage agent {self.stage_name}: {e}")
            return False
    
    def execute_stage(self, context: WorkflowContext, conversation_id: Optional[str] = None) -> StageResult:
        """
        Execute the heterogeneity analysis stage.
        
        Args:
            context: Workflow context with previous stage outputs
            conversation_id: Optional conversation identifier
            
        Returns:
            StageResult with heterogeneity analysis outputs
        """
        start_time = time.time()
        try:
            # Get required inputs from previous stages
            stage_outputs_map = getattr(context, "stage_outputs", {}) or {}
            reconstruction_outputs = stage_outputs_map.get(WorkflowStage.RECONSTRUCTION)
            if isinstance(reconstruction_outputs, StageResult):
                reconstruction_outputs = reconstruction_outputs.stage_outputs
            
            # Get final_volume_job_uid from reconstruction stage
            final_volume_job_uid = None
            if reconstruction_outputs:
                final_volume_job_uid = (
                    reconstruction_outputs.get("final_volume_job_uid")
                    or (reconstruction_outputs.get("job_uids", {}).get("final_volume") if isinstance(reconstruction_outputs.get("job_uids"), dict) else None)
                    or (reconstruction_outputs.get("outputs", {}).get("final_volume_job_uid") if isinstance(reconstruction_outputs.get("outputs"), dict) else None)
                    or reconstruction_outputs.get("homogeneous_refinement_job_uid")
                    or (reconstruction_outputs.get("job_uids", {}).get("homogeneous_refinement") if isinstance(reconstruction_outputs.get("job_uids"), dict) else None)
                )
            
            # refinement_job_uid and volume_job_uid should both be the final_volume_job_uid
            refinement_job_uid = final_volume_job_uid
            volume_job_uid = final_volume_job_uid
            
            # Get particles job UID from reconstruction stage (use the same particles that reconstruction used)
            particles_job_uid = None
            
            # First, try to get from reconstruction outputs (input_particles_job_uid)
            if reconstruction_outputs:
                particles_job_uid = reconstruction_outputs.get("input_particles_job_uid")
                if particles_job_uid:
                    self.logger.info(f"Using particles from reconstruction stage outputs: {particles_job_uid}")
            
            # If not in outputs, try to get from reconstruction job itself (ab_initio job has particles as input)
            if not particles_job_uid and final_volume_job_uid:
                try:
                    ab_initio_job_uid = (
                        reconstruction_outputs.get("ab_initio_job_uid")
                        or (reconstruction_outputs.get("job_uids", {}).get("ab_initio") if isinstance(reconstruction_outputs.get("job_uids"), dict) else None)
                    )
                    if ab_initio_job_uid:
                        # Get particles job UID from ab_initio job (it has particles as input)
                        job = self.cryosparc_tools.cs.find_job(self.config.workflow.project_uid, ab_initio_job_uid)
                        if job:
                            job.refresh()
                            doc = getattr(job, "doc", {})
                            input_connections = doc.get("input_connections", {})
                            particles_connection = input_connections.get("particles", [])
                            if particles_connection and len(particles_connection) > 0:
                                # particles_connection is a list of [job_uid, group_name] tuples
                                particles_job_uid = particles_connection[0][0] if isinstance(particles_connection[0], (list, tuple)) else particles_connection[0]
                                if particles_job_uid:
                                    self.logger.info(f"Using particles from reconstruction ab_initio job input: {particles_job_uid}")
                except Exception as e:
                    self.logger.warning(f"Could not get particles from reconstruction job: {e}")
            
            # Fallback: use same logic as reconstruction (2D optimization first, then particle picking)
            if not particles_job_uid:
                # Check 2D optimization stage (if enabled and completed)
                optimization_2d_outputs = stage_outputs_map.get(WorkflowStage.OPTIMIZATION_2D)
                if isinstance(optimization_2d_outputs, StageResult):
                    optimization_2d_outputs = optimization_2d_outputs.stage_outputs
                
                if optimization_2d_outputs and optimization_2d_outputs.get("final_particles_job_uid"):
                    particles_job_uid = optimization_2d_outputs.get("final_particles_job_uid")
                    self.logger.info(f"Using particles from 2D optimization stage (fallback): {particles_job_uid}")
                else:
                    # Fallback to particle picking stage
                    picking_outputs = stage_outputs_map.get(WorkflowStage.PARTICLE_PICKING)
                    if isinstance(picking_outputs, StageResult):
                        picking_outputs = picking_outputs.stage_outputs
                    
                    if picking_outputs:
                        particles_job_uid = (
                            picking_outputs.get("final_selection_job_uid")
                            or picking_outputs.get("selected_particles_job_uid")
                            or picking_outputs.get("blob_picker_job_uid")
                            or picking_outputs.get("picking_job_uid")
                            or picking_outputs.get("particle_picking_job_uid")
                        )
                        if particles_job_uid:
                            self.logger.info(f"Using particles from particle picking stage (fallback): {particles_job_uid}")
            
            # Get micrographs job UID
            preprocessing_outputs = stage_outputs_map.get(WorkflowStage.PREPROCESSING)
            if isinstance(preprocessing_outputs, StageResult):
                preprocessing_outputs = preprocessing_outputs.stage_outputs
            
            micrographs_job_uid = None
            if preprocessing_outputs:
                micrographs_job_uid = (
                    preprocessing_outputs.get("micrograph_selection_job_uid")
                    or preprocessing_outputs.get("final_micrographs_job_uid")
                )
            
            if not all([refinement_job_uid, particles_job_uid, micrographs_job_uid, volume_job_uid]):
                missing = []
                if not refinement_job_uid:
                    missing.append("refinement_job_uid")
                if not particles_job_uid:
                    missing.append("particles_job_uid")
                if not micrographs_job_uid:
                    missing.append("micrographs_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                
                execution_time = time.time() - start_time
                return StageResult(
                    stage=WorkflowStage.HETEROGENEITY,
                    success=False,
                    error=f"Missing required inputs from previous stages: {', '.join(missing)}",
                    stage_outputs={},
                    execution_time=execution_time
                )
            
            # Execute heterogeneity analysis workflow
            # Use "outputs" as default output directory (same as other agents)
            output_dir = "outputs"
            result = self.modular_workflow.execute_heterogeneity_analysis(
                refinement_job_uid=refinement_job_uid,
                particles_job_uid=particles_job_uid,
                micrographs_job_uid=micrographs_job_uid,
                volume_job_uid=volume_job_uid,
                conversation_id=conversation_id,
                output_dir=output_dir
            )
            
            # Extract outputs
            stage_outputs = {
                "converged_k": result.converged_k,
                "true_num_classes": result.true_num_classes,
                "filtered_groups": result.filtered_groups or [],
                "final_refinement_jobs": result.final_refinement_jobs or [],
                "output_json_path": result.output_json_path
            }
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Save results
            output_file = self._save_heterogeneity_results(stage_outputs, context, result.success, execution_time)
            stage_outputs["output_file"] = output_file
            
            return StageResult(
                stage=WorkflowStage.HETEROGENEITY,
                success=result.success,
                error=result.error,
                stage_outputs=stage_outputs,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Failed to execute heterogeneity analysis stage: {e}")
            return StageResult(
                stage=WorkflowStage.HETEROGENEITY,
                success=False,
                error=str(e),
                stage_outputs={},
                execution_time=execution_time
            )
    
    def _save_heterogeneity_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True, execution_time: float = 0.0) -> str:
        """Save heterogeneity analysis results to a JSON file."""
        import datetime
        from pathlib import Path
        
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "completed" if success else "failed"
        
        heterogeneity_results = {
            "stage": "heterogeneity",
            "status": status,
            "timestamp": timestamp,
            "agent_type": "cryosparc",
            "project_uid": context.project_uid,
            "workspace_uid": context.workspace_uid,
            "execution_time": execution_time,
            "converged_k": stage_outputs.get("converged_k"),
            "true_num_classes": stage_outputs.get("true_num_classes"),
            "filtered_groups": stage_outputs.get("filtered_groups", []),
            "final_refinement_jobs": stage_outputs.get("final_refinement_jobs", []),
            "output_json_path": stage_outputs.get("output_json_path")
        }
        
        output_file = output_dir / f"heterogeneity_analysis_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(heterogeneity_results, f, indent=2)
        
        self.logger.info(f"Heterogeneity analysis results saved to {output_file}")
        return str(output_file)
    
    def get_stage_description(self) -> str:
        return "Heterogeneity Analysis: Determine true number of classes using ab initio + heterogeneous refinement and density comparison"
    
    def get_required_inputs(self) -> List[str]:
        return ["refinement_job_uid", "particles_job_uid", "micrographs_job_uid", "volume_job_uid"]


class MasterOrchestrator:
    """Master orchestrator for the complete cryoEM workflow."""
    
    def __init__(self, master_config_path: str, outputs_dir: str = "outputs"):
        """
        Initialize the master orchestrator.
        
        Args:
            master_config_path: Path to the master configuration file
            outputs_dir: Directory where output files will be saved
        """
        self.master_config_path = master_config_path
        self.outputs_dir = outputs_dir
        self.master_config = None
        self.stage_agents = {}
        self.stage_results = []
        self.workflow_context = None
        self.start_time = None
        self.logger = logging.getLogger("MasterOrchestrator")
        self.transition_agent = None
        self.summary_agent = SummaryAgent(outputs_dir=outputs_dir)
        
    def initialize(self) -> bool:
        """
        Initialize the master orchestrator and all stage agents.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Apply microscope overrides before loading any stage configurations
            try:
                apply_microscope_overrides_if_enabled()
            except Exception as override_exc:
                self.logger.warning(f"Failed to apply microscope overrides: {override_exc}")
            
            # Apply CryoSift overrides from master_config.json to stage configs
            try:
                apply_cryosift_overrides_if_enabled()
            except Exception as cryosift_exc:
                self.logger.warning(f"Failed to apply CryoSift overrides: {cryosift_exc}")

            # Load master configuration
            with open(self.master_config_path, 'r') as f:
                self.master_config = json.load(f)
            
            # Load and merge session.json if it exists (session.json takes precedence)
            session_config_path = Path(self.master_config_path).parent / "session.json"
            if session_config_path.exists():
                self.logger.info(f"Loading session configuration from {session_config_path}")
                with open(session_config_path, 'r') as f:
                    session_config = json.load(f)
                # Merge session config into master config (session config takes precedence)
                self.master_config = self._merge_configs(self.master_config, session_config)
                self.logger.info("Session configuration merged successfully")
            else:
                self.logger.info(f"Session configuration not found at {session_config_path}, using master config only")
            
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
                    agent = PreprocessingAgent(config_path, self.master_config_path)
                elif agent_class == "ParticlePickingAgent":
                    agent = ParticlePickingAgent(config_path, self.master_config_path)
                elif agent_class == "Optimizer2DAgent":
                    agent = Optimizer2DAgent(config_path, self.master_config_path)
                elif agent_class == "ReconstructionAgent":
                    agent = ReconstructionAgent(config_path, self.master_config_path)
                elif agent_class == "OptimizerAgent":
                    agent = OptimizerAgent(config_path, self.master_config_path)
                elif agent_class == "HeterogeneityAgent":
                    agent = HeterogeneityAgent(config_path, self.master_config_path)
                elif agent_class == "PolishAgent":
                    agent = PolishAgent(config_path, self.master_config_path)
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
            
            # Initialize transition agent
            # Try to get CryoSparc and Relion tools from initialized stage agents
            cryosparc_tools = None
            relion_tools = None
            
            # Look for CryoSparc tools in any stage agent
            for agent in self.stage_agents.values():
                if hasattr(agent, 'cryosparc_tools') and agent.cryosparc_tools:
                    cryosparc_tools = agent.cryosparc_tools
                    self.logger.info("Found CryoSparc tools from stage agent")
                    break
                # Also check modular_agent
                if hasattr(agent, 'modular_agent') and hasattr(agent.modular_agent, 'cryosparc_tools') and agent.modular_agent.cryosparc_tools:
                    cryosparc_tools = agent.modular_agent.cryosparc_tools
                    self.logger.info("Found CryoSparc tools from modular agent")
                    break
            
            # Look for Relion tools in any stage agent
            for agent in self.stage_agents.values():
                if hasattr(agent, 'modular_agent') and hasattr(agent.modular_agent, 'relion_tools') and agent.modular_agent.relion_tools:
                    relion_tools = agent.modular_agent.relion_tools
                    self.logger.info("Found Relion tools from modular agent")
                    break
            
            try:
                self.transition_agent = TransitionAgent(
                    self.master_config_path,
                    cryosparc_tools=cryosparc_tools,
                    relion_tools=relion_tools
                )
                self.logger.info("Transition agent initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize transition agent: {e}")
                self.transition_agent = None
            
            # Summary agent is already initialized in __init__
            self.logger.info("Summary agent initialized")
            
            self.logger.info("Master orchestrator initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize master orchestrator: {e}")
            return False
    
    def _merge_configs(self, master_config: Dict[str, Any], session_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge session configuration into master configuration.
        Session config takes precedence for overlapping keys.
        
        Args:
            master_config: The master configuration dictionary
            session_config: The session configuration dictionary to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = master_config.copy()
        
        for key, value in session_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Session config takes precedence
                merged[key] = value
        
        return merged
    
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

    def _reconstruct_stage_outputs_from_minimal_data(self, stage: WorkflowStage, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct stage_outputs from simplified result files produced by the
        updated parsers. Ensures downstream stages receive the identifiers they expect.
        """
        stage_outputs: Dict[str, Any] = {}

        if stage == WorkflowStage.PREPROCESSING:
            final_job_uid = data.get("final_micrographs_job_uid") or data.get("micrograph_selection_job_uid")
            micrograph_dir = data.get("micrograph_directory") or data.get("micrographs_directory")
            micrograph_location = data.get("micrograph_location")
            micrographs_folder = data.get("micrographs_folder")
            selected_micrographs_star = data.get("selected_micrographs_star")
            micrograph_selection_job_dir = data.get("micrograph_selection_job_dir")
            relion_dir = data.get("relion_dir")

            if final_job_uid:
                stage_outputs["micrograph_selection_job_uid"] = final_job_uid
            if micrograph_dir:
                stage_outputs["micrograph_directory"] = micrograph_dir
                stage_outputs["selected_micrographs"] = micrograph_dir
            if micrograph_location:
                stage_outputs["micrograph_location"] = micrograph_location
            if micrographs_folder:
                stage_outputs["micrograph_directory"] = micrographs_folder
                stage_outputs["selected_micrographs"] = micrographs_folder
                stage_outputs["micrograph_location"] = micrographs_folder
                try:
                    stage_outputs["motion_correction_job_dir"] = str(Path(micrographs_folder).resolve().parent)
                except Exception:
                    stage_outputs["motion_correction_job_dir"] = str(Path(micrographs_folder).parent)
            if selected_micrographs_star:
                stage_outputs["selected_micrographs_star"] = selected_micrographs_star
            if micrograph_selection_job_dir:
                stage_outputs["micrograph_selection_job_dir"] = micrograph_selection_job_dir
            if relion_dir:
                stage_outputs["relion_dir"] = relion_dir

        elif stage == WorkflowStage.PARTICLE_PICKING:
            final_selection_uid = data.get("final_selection_job_uid") or data.get("selected_particles_job_uid")
            selected_dir = data.get("selected_particles_directory")
            selected_file = data.get("selected_particles_file")
            transition_metadata = data.get("transition_metadata", {})
            transition_config = transition_metadata.get("transition_config") or data.get("transition_config")
            transition_outputs = transition_metadata.get("transition_config_outputs")
            transition_transitions = transition_metadata.get("transition_config_transitions")
            transition_info = transition_metadata.get("transition_info")

            if final_selection_uid:
                stage_outputs["final_selection_job_uid"] = final_selection_uid
                stage_outputs["selected_particles_job_uid"] = final_selection_uid
            if selected_dir:
                stage_outputs["selected_particles_location"] = selected_dir
            if selected_file:
                stage_outputs["final_particles_cs_file"] = selected_file
            if transition_config:
                stage_outputs["transition_config"] = transition_config
            if transition_outputs:
                stage_outputs["transition_config_outputs"] = transition_outputs
            if transition_transitions:
                stage_outputs["transition_config_transitions"] = transition_transitions
            if transition_info:
                stage_outputs["transition_info"] = transition_info

        elif stage == WorkflowStage.OPTIMIZATION_2D:
            final_particles_uid = data.get("final_particles_job_uid")
            final_good_count = data.get("final_good_particles_count")
            final_good_percentage = data.get("final_good_particles_percentage")
            total_rounds = data.get("total_rounds")
            workflow_summary = data.get("workflow_summary", {})
            
            if final_particles_uid:
                stage_outputs["final_particles_job_uid"] = final_particles_uid
                # Also set as selected_particles_job_uid for compatibility with reconstruction stage
                stage_outputs["selected_particles_job_uid"] = final_particles_uid
            if final_good_count is not None:
                stage_outputs["final_good_particles_count"] = final_good_count
            if final_good_percentage is not None:
                stage_outputs["final_good_particles_percentage"] = final_good_percentage
            if total_rounds is not None:
                stage_outputs["total_rounds"] = total_rounds
            if workflow_summary:
                stage_outputs["workflow_summary"] = workflow_summary

        elif stage == WorkflowStage.RECONSTRUCTION:
            final_volume_dir = data.get("final_volume_directory")
            final_volume_uid = data.get("final_volume_job_uid")
            final_star_file = data.get("final_star_file")
            
            # Extract from nested job_uids structure (CryoSPARC format)
            job_uids = data.get("job_uids", {})
            if isinstance(job_uids, dict):
                if job_uids.get("ab_initio"):
                    stage_outputs["ab_initio_job_uid"] = job_uids["ab_initio"]
                if job_uids.get("homogeneous_reconstruction"):
                    stage_outputs["homogeneous_reconstruction_job_uid"] = job_uids["homogeneous_reconstruction"]
                if job_uids.get("homogeneous_refinement"):
                    stage_outputs["homogeneous_refinement_job_uid"] = job_uids["homogeneous_refinement"]
                if job_uids.get("final_volume"):
                    stage_outputs["final_volume_job_uid"] = job_uids["final_volume"]
            
            # Extract from nested outputs structure (CryoSPARC format)
            outputs = data.get("outputs", {})
            if isinstance(outputs, dict):
                if outputs.get("final_volume_job_uid"):
                    stage_outputs["final_volume_job_uid"] = outputs["final_volume_job_uid"]
            
            # Direct fields (fallback)
            if final_volume_dir:
                stage_outputs["final_volume_directory"] = final_volume_dir
            if final_volume_uid:
                stage_outputs["final_volume_job_uid"] = final_volume_uid
            if final_star_file:
                stage_outputs["final_star_file"] = final_star_file

        elif stage == WorkflowStage.HETEROGENEITY:
            converged_k = data.get("converged_k")
            true_num_classes = data.get("true_num_classes")
            filtered_groups = data.get("filtered_groups", [])
            final_refinement_jobs = data.get("final_refinement_jobs", [])
            output_json_path = data.get("output_json_path")
            
            if converged_k is not None:
                stage_outputs["converged_k"] = converged_k
            if true_num_classes is not None:
                stage_outputs["true_num_classes"] = true_num_classes
            if filtered_groups:
                stage_outputs["filtered_groups"] = filtered_groups
            if final_refinement_jobs:
                stage_outputs["final_refinement_jobs"] = final_refinement_jobs
            if output_json_path:
                stage_outputs["output_json_path"] = output_json_path

        return stage_outputs
    
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

        # Update workflow context with actual project/workspace from master config if available
        workflow_config = (self.master_config or {}).get("workflow", {})
        if workflow_config:
            self.workflow_context.project_uid = workflow_config.get("project_uid", self.workflow_context.project_uid)
            self.workflow_context.workspace_uid = workflow_config.get("workspace_uid", self.workflow_context.workspace_uid)
        else:
            # Fallback to stage agent configs if master config missing workflow section
            try:
                default_agent = next(iter(self.stage_agents.values()), None)
                if default_agent and hasattr(default_agent, "config") and default_agent.config:
                    cfg_workflow = getattr(default_agent.config, "workflow", None)
                    if cfg_workflow:
                        self.workflow_context.project_uid = getattr(cfg_workflow, "project_uid", self.workflow_context.project_uid)
                        self.workflow_context.workspace_uid = getattr(cfg_workflow, "workspace_uid", self.workflow_context.workspace_uid)
            except Exception:
                pass
        
        # Set workflow context in summary agent
        self.summary_agent.set_workflow_context(self.workflow_context)
        
        self.logger.info("Starting complete cryoEM workflow")
        print("🚀 Starting Complete CryoEM Workflow")
        print("=" * 60)
        
        # Execute stages in sequence - dynamically determine from master config
        stages_to_execute = []
        for stage_info in self.master_config["master_workflow"]["stages"]:
            if stage_info.get("enabled", False):
                try:
                    stage = WorkflowStage(stage_info["name"])
                    stages_to_execute.append(stage)
                except ValueError:
                    self.logger.warning(f"Unknown stage name in config: {stage_info['name']}")
        
        # Fallback to default stages if none found in config
        if not stages_to_execute:
            self.logger.warning("No enabled stages found in config, using default stages")
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
            
            # Check if stage output already exists (check custom outputs_dir first, then default)
            existing_output = check_stage_output_exists(stage, self.outputs_dir)
            if existing_output:
                self.logger.info(f"Stage {stage_name} already completed, skipping execution")
                print(f"✅ Stage {stage_name} already completed!")
                print(f"   📄 Output file: {existing_output['file_path']}")
                print(f"   📅 Completed at: {existing_output['timestamp']}")
                print(f"   ℹ️  Skipping execution to avoid re-running")
                
                # Get stage_outputs from the saved JSON file
                # RELION saves directly in stage_outputs, CryoSPARC might use job_uids
                data = existing_output['data']
                if 'stage_outputs' in data:
                    # RELION format: use stage_outputs directly
                    stage_outputs = data['stage_outputs']
                elif stage == WorkflowStage.RECONSTRUCTION:
                    # For reconstruction, use _reconstruct_stage_outputs_from_minimal_data
                    # to properly extract from nested job_uids and outputs structures
                    stage_outputs = self._reconstruct_stage_outputs_from_minimal_data(stage, data)
                elif 'job_uids' in data:
                    # CryoSPARC format: reconstruct from job_uids (for other stages)
                    job_uids = data.get('job_uids', {})
                    stage_outputs = self._reconstruct_stage_outputs_from_cache(stage, job_uids)
                    # Ensure project_uid and workspace_uid are included
                    if 'project_uid' in data:
                        stage_outputs['project_uid'] = data['project_uid']
                    if 'workspace_uid' in data:
                        stage_outputs['workspace_uid'] = data['workspace_uid']
                    # Preserve micrograph_location data if it exists (needed for transition)
                    if 'micrograph_location' in data:
                        stage_outputs['micrograph_location'] = data['micrograph_location']
                else:
                    # Fallback: try to use the data directly
                    self.logger.warning(f"No stage_outputs or job_uids found in {existing_output['file_path']}")
                    stage_outputs = self._reconstruct_stage_outputs_from_minimal_data(stage, data)
                    # Try to get project/workspace from data
                    if 'project_uid' in data:
                        stage_outputs['project_uid'] = data['project_uid']
                    if 'workspace_uid' in data:
                        stage_outputs['workspace_uid'] = data['workspace_uid']
                    # Preserve micrograph_location if it exists
                    if 'micrograph_location' in data:
                        stage_outputs['micrograph_location'] = data['micrograph_location']
                
                # Check if transition is needed before next stage (even when loading from cache)
                stage_index = stages_to_execute.index(stage)
                if stage_index < len(stages_to_execute) - 1:
                    next_stage = stages_to_execute[stage_index + 1]
                    next_stage_name = next_stage.value
                    
                    if self.transition_agent:
                        try:
                            # Perform transition if needed
                            stage_outputs = self.transition_agent.perform_transition(
                                current_stage=stage_name,
                                next_stage=next_stage_name,
                                current_stage_outputs=stage_outputs,
                                project_uid=self.workflow_context.project_uid,
                                workspace_uid=self.workflow_context.workspace_uid
                            )
                            
                            transition_info = stage_outputs.get("transition_info")
                            if transition_info:
                                print(f"🔄 Format transition: {transition_info['from_agent']} -> {transition_info['to_agent']}")
                                self.logger.info(f"Transition completed (from cache): {transition_info}")
                                # Print additional info about created files
                                if 'relion_job_dir' in transition_info:
                                    print(f"   📁 Relion job directory: {transition_info['relion_job_dir']}")
                                if 'config_file' in transition_info:
                                    print(f"   📄 Config file: {transition_info['config_file']}")
                        except Exception as e:
                            self.logger.error(f"Transition failed (from cache): {e}")
                            import traceback
                            self.logger.error(f"Transition error traceback: {traceback.format_exc()}")
                            print(f"⚠️ Warning: Transition failed: {e}")
                
                # Create a successful stage result from existing output
                stage_result = StageResult(
                    stage=stage,
                    success=True,
                    stage_outputs=stage_outputs,
                    execution_time=0.0,
                    reasoning="Stage skipped - output already exists"
                )
                self.stage_results.append(stage_result)
                
                # Add stage summary to summary agent (even for cached stages)
                if stage_name in self.stage_agents:
                    stage_agent = self.stage_agents[stage_name]
                    self.summary_agent.add_stage_summary(stage_result, stage_agent)
                
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
            
            # Add stage summary to summary agent
            self.summary_agent.add_stage_summary(stage_result, stage_agent)
            
            # Update context with stage outputs
            if stage_result.success:
                stage_outputs = stage_result.stage_outputs
                
                # Check if transition is needed before next stage
                stage_index = stages_to_execute.index(stage)
                if stage_index < len(stages_to_execute) - 1:
                    next_stage = stages_to_execute[stage_index + 1]
                    next_stage_name = next_stage.value
                    
                    if self.transition_agent:
                        try:
                            # Perform transition if needed
                            stage_outputs = self.transition_agent.perform_transition(
                                current_stage=stage_name,
                                next_stage=next_stage_name,
                                current_stage_outputs=stage_outputs,
                                project_uid=self.workflow_context.project_uid,
                                workspace_uid=self.workflow_context.workspace_uid
                            )
                            
                            transition_info = stage_outputs.get("transition_info")
                            if transition_info:
                                print(f"🔄 Format transition: {transition_info['from_agent']} -> {transition_info['to_agent']}")
                                self.logger.info(f"Transition completed: {transition_info}")
                                # Print additional info about created files
                                if 'relion_job_dir' in transition_info:
                                    print(f"   📁 Relion job directory: {transition_info['relion_job_dir']}")
                                if 'config_file' in transition_info:
                                    print(f"   📄 Config file: {transition_info['config_file']}")
                        except Exception as e:
                            self.logger.error(f"Transition failed: {e}")
                            import traceback
                            self.logger.error(f"Transition error traceback: {traceback.format_exc()}")
                            print(f"⚠️ Warning: Transition failed: {e}")
                
                self.workflow_context.stage_outputs[stage] = stage_outputs
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
        
        # Set workflow end time for summary agent
        workflow_end_time = time.time()
        self.summary_agent.set_workflow_end_time(workflow_end_time)
        
        # Generate final comprehensive report
        final_report_path = self.summary_agent.generate_final_report(conversation_id)
        print(f"\n📊 Final workflow report generated: {final_report_path}")
        
        # Generate workflow summary
        total_time = workflow_end_time - self.start_time
        summary = self._generate_workflow_summary(total_time)
        
        # Add conversation log files to summary
        summary['conversation_log_files'] = conversation_log_files
        
        # Add final report path to summary
        summary['final_report_path'] = final_report_path
        
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
        
        # Initialize summary agent for this workflow
        self.summary_agent.clear_summaries()
        self.summary_agent.set_workflow_start_time(self.start_time)
        
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

        workflow_config = (self.master_config or {}).get("workflow", {})
        if workflow_config:
            self.workflow_context.project_uid = workflow_config.get("project_uid", self.workflow_context.project_uid)
            self.workflow_context.workspace_uid = workflow_config.get("workspace_uid", self.workflow_context.workspace_uid)
        else:
            try:
                default_agent = next(iter(self.stage_agents.values()), None)
                if default_agent and hasattr(default_agent, "config") and default_agent.config:
                    cfg_workflow = getattr(default_agent.config, "workflow", None)
                    if cfg_workflow:
                        self.workflow_context.project_uid = getattr(cfg_workflow, "project_uid", self.workflow_context.project_uid)
                        self.workflow_context.workspace_uid = getattr(cfg_workflow, "workspace_uid", self.workflow_context.workspace_uid)
            except Exception:
                pass
        
        # Set workflow context in summary agent
        self.summary_agent.set_workflow_context(self.workflow_context)
        
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
            
            # Check if stage output already exists (check custom outputs_dir first, then default)
            existing_output = check_stage_output_exists(stage, self.outputs_dir)
            if existing_output:
                self.logger.info(f"Stage {stage_name} already completed, skipping execution")
                print(f"✅ Stage {stage_name} already completed!")
                print(f"   📄 Output file: {existing_output['file_path']}")
                print(f"   📅 Completed at: {existing_output['timestamp']}")
                print(f"   ℹ️  Skipping execution to avoid re-running")
                
                # Get stage_outputs from the saved JSON file
                # RELION saves directly in stage_outputs, CryoSPARC might use job_uids
                data = existing_output['data']
                if 'stage_outputs' in data:
                    # RELION format: use stage_outputs directly
                    stage_outputs = data['stage_outputs']
                elif stage == WorkflowStage.RECONSTRUCTION:
                    # For reconstruction, use _reconstruct_stage_outputs_from_minimal_data
                    # to properly extract from nested job_uids and outputs structures
                    stage_outputs = self._reconstruct_stage_outputs_from_minimal_data(stage, data)
                elif 'job_uids' in data:
                    # CryoSPARC format: reconstruct from job_uids (for other stages)
                    job_uids = data.get('job_uids', {})
                    stage_outputs = self._reconstruct_stage_outputs_from_cache(stage, job_uids)
                    # Ensure project_uid and workspace_uid are included
                    if 'project_uid' in data:
                        stage_outputs['project_uid'] = data['project_uid']
                    if 'workspace_uid' in data:
                        stage_outputs['workspace_uid'] = data['workspace_uid']
                    # Preserve micrograph_location data if it exists (needed for transition)
                    if 'micrograph_location' in data:
                        stage_outputs['micrograph_location'] = data['micrograph_location']
                else:
                    # Fallback: try to use the data directly
                    self.logger.warning(f"No stage_outputs or job_uids found in {existing_output['file_path']}")
                    stage_outputs = self._reconstruct_stage_outputs_from_minimal_data(stage, data)
                    # Try to get project/workspace from data
                    if 'project_uid' in data:
                        stage_outputs['project_uid'] = data['project_uid']
                    if 'workspace_uid' in data:
                        stage_outputs['workspace_uid'] = data['workspace_uid']
                    # Preserve micrograph_location if it exists
                    if 'micrograph_location' in data:
                        stage_outputs['micrograph_location'] = data['micrograph_location']
                
                # Check if transition is needed before next stage (even when loading from cache)
                stage_index = stages.index(stage)
                if stage_index < len(stages) - 1:
                    next_stage = stages[stage_index + 1]
                    next_stage_name = next_stage.value
                    
                    if self.transition_agent:
                        # Check if transition is actually needed
                        needs_transition, _, _ = self.transition_agent.check_transition_needed(
                            current_stage=stage_name,
                            next_stage=next_stage_name
                        )
                        
                        if needs_transition:
                            try:
                                # Perform transition if needed
                                transition_result = self.transition_agent.perform_transition(
                                    current_stage=stage_name,
                                    next_stage=next_stage_name,
                                    current_stage_outputs=stage_outputs,
                                    project_uid=self.workflow_context.project_uid,
                                    workspace_uid=self.workflow_context.workspace_uid
                                )
                                
                                # Check if transition was successful
                                if not transition_result.get("success", False):
                                    error_msg = transition_result.get("error", "Unknown transition error")
                                    self.logger.error(f"Transition failed (from cache): {error_msg}")
                                    print(f"❌ Transition failed: {error_msg}")
                                    print(f"⚠️  Stopping workflow - transition is required but failed")
                                    # Create a failed stage result to stop the workflow
                                    stage_result = StageResult(
                                        stage=stage,
                                        success=False,
                                        stage_outputs=stage_outputs,
                                        execution_time=0.0,
                                        error=f"Transition failed: {error_msg}",
                                        reasoning="Transition required but failed"
                                    )
                                    self.stage_results.append(stage_result)
                                    break
                                
                                # Merge converted outputs into stage_outputs
                                converted_outputs = transition_result.get("converted_outputs", {})
                                stage_outputs.update(converted_outputs)
                                
                                transition_info = stage_outputs.get("transition_info")
                                if transition_info:
                                    print(f"🔄 Format transition: {transition_info['from_agent']} -> {transition_info['to_agent']}")
                                    self.logger.info(f"Transition completed (from cache): {transition_info}")
                                    # Print additional info about created files
                                    if 'relion_job_dir' in transition_info:
                                        print(f"   📁 Relion job directory: {transition_info['relion_job_dir']}")
                                    if 'config_file' in transition_info:
                                        print(f"   📄 Config file: {transition_info['config_file']}")
                            except Exception as e:
                                self.logger.error(f"Transition failed (from cache): {e}")
                                import traceback
                                self.logger.error(f"Transition error traceback: {traceback.format_exc()}")
                                print(f"❌ Transition failed: {e}")
                                print(f"⚠️  Stopping workflow - transition is required but failed")
                                # Create a failed stage result to stop the workflow
                                stage_result = StageResult(
                                    stage=stage,
                                    success=False,
                                    stage_outputs=stage_outputs,
                                    execution_time=0.0,
                                    error=f"Transition failed: {e}",
                                    reasoning="Transition required but failed"
                                )
                                self.stage_results.append(stage_result)
                                break
                
                # Create a successful stage result from existing output
                stage_result = StageResult(
                    stage=stage,
                    success=True,
                    stage_outputs=stage_outputs,
                    execution_time=0.0,
                    reasoning="Stage skipped - output already exists"
                )
                self.stage_results.append(stage_result)
                
                # Add stage summary to summary agent (even for cached stages)
                if stage_name in self.stage_agents:
                    stage_agent = self.stage_agents[stage_name]
                    self.summary_agent.add_stage_summary(stage_result, stage_agent)
                
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
            
            # Add stage summary to summary agent
            self.summary_agent.add_stage_summary(stage_result, stage_agent)
            
            # Update context with stage outputs
            if stage_result.success:
                stage_outputs = stage_result.stage_outputs
                
                # Check if transition is needed before next stage
                stage_index = stages.index(stage)
                if stage_index < len(stages) - 1:
                    next_stage = stages[stage_index + 1]
                    next_stage_name = next_stage.value
                    
                    if self.transition_agent:
                        # Check if transition is actually needed
                        needs_transition, _, _ = self.transition_agent.check_transition_needed(
                            current_stage=stage_name,
                            next_stage=next_stage_name
                        )
                        
                        if needs_transition:
                            try:
                                # Perform transition if needed
                                transition_result = self.transition_agent.perform_transition(
                                    current_stage=stage_name,
                                    next_stage=next_stage_name,
                                    current_stage_outputs=stage_outputs,
                                    project_uid=self.workflow_context.project_uid,
                                    workspace_uid=self.workflow_context.workspace_uid
                                )
                                
                                # Check if transition was successful
                                if not transition_result.get("success", False):
                                    error_msg = transition_result.get("error", "Unknown transition error")
                                    self.logger.error(f"Transition failed: {error_msg}")
                                    print(f"❌ Transition failed: {error_msg}")
                                    print(f"⚠️  Stopping workflow - transition is required but failed")
                                    # Mark stage as failed to stop workflow
                                    stage_result.success = False
                                    stage_result.error = f"Transition failed: {error_msg}"
                                    break
                                
                                # Merge converted outputs into stage_outputs
                                converted_outputs = transition_result.get("converted_outputs", {})
                                stage_outputs.update(converted_outputs)
                                
                                transition_info = stage_outputs.get("transition_info")
                                if transition_info:
                                    print(f"🔄 Format transition: {transition_info['from_agent']} -> {transition_info['to_agent']}")
                                    self.logger.info(f"Transition completed: {transition_info}")
                                    # Print additional info about created files
                                    if 'relion_job_dir' in transition_info:
                                        print(f"   📁 Relion job directory: {transition_info['relion_job_dir']}")
                                    if 'config_file' in transition_info:
                                        print(f"   📄 Config file: {transition_info['config_file']}")
                            except Exception as e:
                                self.logger.error(f"Transition failed: {e}")
                                import traceback
                                self.logger.error(f"Transition error traceback: {traceback.format_exc()}")
                                print(f"❌ Transition failed: {e}")
                                print(f"⚠️  Stopping workflow - transition is required but failed")
                                # Mark stage as failed to stop workflow
                                stage_result.success = False
                                stage_result.error = f"Transition failed: {e}"
                                break
                
                self.workflow_context.stage_outputs[stage] = stage_outputs
                print(f"✅ Stage {stage_name} completed successfully")
                print(f"   Execution time: {stage_result.execution_time:.2f} seconds")
            else:
                print(f"❌ Stage {stage_name} failed: {stage_result.error}")
                break
        
        # Set workflow end time for summary agent
        workflow_end_time = time.time()
        self.summary_agent.set_workflow_end_time(workflow_end_time)
        
        # Generate final comprehensive report
        final_report_path = self.summary_agent.generate_final_report(conversation_id)
        print(f"\n📊 Final workflow report generated: {final_report_path}")
        
        # Generate workflow summary
        total_time = workflow_end_time - self.start_time
        summary = self._generate_workflow_summary(total_time)
        
        # Add final report path to summary
        summary['final_report_path'] = final_report_path
        
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
        
        # Display final report path
        if 'final_report_path' in summary:
            print("\n📊 Final Workflow Report:")
            print("=" * 50)
            print(f"   JSON Report: {summary['final_report_path']}")
            # Markdown report has same name but .md extension
            md_report = summary['final_report_path'].replace('.json', '.md')
            print(f"   Markdown Report: {md_report}")
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
