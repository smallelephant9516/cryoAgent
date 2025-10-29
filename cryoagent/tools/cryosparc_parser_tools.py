"""
CryoSPARC Parser Tools

This module provides utilities for parsing, validating, and saving CryoSPARC workflow results.
These tools are designed to be used by the modular agents to handle backend-specific result processing.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .cryosparc_tools import CryoSPARCTools


logger = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """Context information for workflow execution."""
    project_uid: str
    workspace_uid: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CryoSPARCPreprocessingParser:
    """Parser for CryoSPARC preprocessing workflow results."""
    
    def __init__(self, cryosparc_tools: CryoSPARCTools, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the CryoSPARC preprocessing parser.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            logger_instance: Optional logger instance
        """
        self.cryosparc_tools = cryosparc_tools
        self.logger = logger_instance or logger
    
    def process_workflow_results(self, results: List, context: WorkflowContext) -> Dict[str, Any]:
        """
        Process workflow results and extract stage outputs.
        
        Args:
            results: List of preprocessing workflow results
            context: Workflow context
            
        Returns:
            Dictionary of stage outputs with standardized keys
        """
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
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """
        Save preprocessing results to a JSON file.
        
        Args:
            stage_outputs: Dictionary of stage outputs
            context: Workflow context
            success: Whether the preprocessing was successful
            
        Returns:
            Path to the saved JSON file
        """
        from datetime import datetime
        
        try:
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
            output_file = output_dir / f"preprocessing_results_cryosparc_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(preprocessing_results, f, indent=2)
            
            self.logger.info(f"Preprocessing results saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save preprocessing results: {e}")
            return ""


class CryoSPARCPickingParser:
    """Parser for CryoSPARC particle picking workflow results."""
    
    def __init__(self, cryosparc_tools: CryoSPARCTools, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the CryoSPARC particle picking parser.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            logger_instance: Optional logger instance
        """
        self.cryosparc_tools = cryosparc_tools
        self.logger = logger_instance or logger
    
    def process_workflow_results(self, results: List, context: WorkflowContext) -> Dict[str, Any]:
        """Process particle picking workflow results."""
        stage_outputs = {
            "blob_picker_job_uid": None,
            "extraction_job_uid": None,
            "classification_2d_job_uid": None,
            "template_picker_job_uid": None,
            "extraction_job_uid_round2": None,
            "classification_2d_job_uid_round2": None,
            "initial_selection_job_uid": None,
            "final_selection_job_uid": None,
            "picked_particles": None,
            "extracted_particles": None,
            "classified_particles": None,
            "selected_particles_location": None,
            "selected_particles_job_metadata": None
        }

        # Extract job UIDs from modular workflow results
        for result in results:
            step_name = result.step.value
            if result.success and result.job_uid:
                if step_name == "blob_picker":
                    stage_outputs["blob_picker_job_uid"] = result.job_uid
                    stage_outputs["picked_particles"] = result.job_uid
                elif step_name == "extract_particles" or step_name == "extract_particles_2":
                    if stage_outputs["extraction_job_uid"] is None:
                        stage_outputs["extraction_job_uid"] = result.job_uid
                    else:
                        stage_outputs["extraction_job_uid_round2"] = result.job_uid
                    stage_outputs["extracted_particles"] = result.job_uid
                elif step_name == "class_2d" or step_name == "class_2d_2":
                    if stage_outputs["classification_2d_job_uid"] is None:
                        stage_outputs["classification_2d_job_uid"] = result.job_uid
                    else:
                        stage_outputs["classification_2d_job_uid_round2"] = result.job_uid
                    stage_outputs["classified_particles"] = result.job_uid
                elif step_name == "select_2d_classes" or step_name == "select_final_classes":
                    if stage_outputs["initial_selection_job_uid"] is None:
                        stage_outputs["initial_selection_job_uid"] = result.job_uid
                    else:
                        stage_outputs["final_selection_job_uid"] = result.job_uid
                elif step_name == "template_picker":
                    stage_outputs["template_picker_job_uid"] = result.job_uid
                elif step_name == "final_extraction" and stage_outputs.get("final_selection_job_uid") is None:
                    stage_outputs["final_selection_job_uid"] = result.job_uid

        final_job_uid = stage_outputs.get("final_selection_job_uid")
        project_uid = context.project_uid

        if final_job_uid and project_uid:
            try:
                job_info = self.cryosparc_tools.get_job_output_directory(project_uid, final_job_uid)
                job_directory = job_info.get("job_directory")
                stage_outputs["selected_particles_location"] = job_directory
                stage_outputs["selected_particles_job_metadata"] = job_info
                
                if job_directory:
                    job_path = Path(job_directory)
                    stage_outputs["final_particles_absolute_path"] = str(job_path.absolute())
                    
                    particles_cs_file = job_path / "particles_selected.cs"
                    if particles_cs_file.exists():
                        stage_outputs["final_particles_cs_file"] = str(particles_cs_file.absolute())
                    
                    passthrough_file = job_path / "particles_selected_passthrough.cs"
                    if passthrough_file.exists():
                        stage_outputs["final_particles_passthrough_file"] = str(passthrough_file.absolute())
                        
            except Exception as exc:
                self.logger.warning(f"Failed to resolve selected particle job directory for {final_job_uid}: {exc}")

        return stage_outputs
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that the complete particle picking workflow completed successfully."""
        required_jobs = [
            ("blob_picker", stage_outputs.get("blob_picker_job_uid")),
            ("particle_extraction", stage_outputs.get("extraction_job_uid")),
            ("2d_classification", stage_outputs.get("classification_2d_job_uid"))
        ]
        
        missing_jobs = []
        for job_name, job_uid in required_jobs:
            if job_uid is None:
                missing_jobs.append(job_name)
        
        if missing_jobs:
            error_msg = f"Particle picking workflow failed - the following jobs were not executed: {', '.join(missing_jobs)}"
            return {
                "success": False,
                "error": error_msg
            }
        
        return {
            "success": True,
            "error": None
        }
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """Save particle picking results to a JSON file."""
        from datetime import datetime
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                "blob_picker": stage_outputs.get("blob_picker_job_uid"),
                "particle_extraction": stage_outputs.get("extraction_job_uid"),
                "2d_classification": stage_outputs.get("classification_2d_job_uid"),
                "template_picker": stage_outputs.get("template_picker_job_uid"),
                "particle_extraction_round2": stage_outputs.get("extraction_job_uid_round2"),
                "2d_classification_round2": stage_outputs.get("classification_2d_job_uid_round2"),
                "final_selection": stage_outputs.get("final_selection_job_uid")
            },
            "outputs": {
                "picked_particles": stage_outputs.get("picked_particles"),
                "extracted_particles": stage_outputs.get("extracted_particles"),
                "classified_particles": stage_outputs.get("classified_particles"),
                "selected_particles_job_uid": stage_outputs.get("final_selection_job_uid"),
                "selected_particles_location": stage_outputs.get("selected_particles_location"),
                "final_particles_absolute_path": stage_outputs.get("final_particles_absolute_path"),
                "final_particles_cs_file": stage_outputs.get("final_particles_cs_file"),
                "final_particles_passthrough_file": stage_outputs.get("final_particles_passthrough_file")
            },
            "usage_notes": {
                "next_stage": "3d_reconstruction",
                "classification_2d_job_uid_usage": "Use the 2d_classification job UID for 3D reconstruction or further refinement",
                "particle_classes": "2D classes are stored in the classification job output and can be used for particle selection",
                "final_particles_path": "The final_particles_absolute_path field contains the absolute path to the job directory with final selected particles",
                "final_particles_files": "The final_particles_cs_file and final_particles_passthrough_file fields contain absolute paths to specific particle data files if they exist"
            }
        }
        
        output_file = output_dir / f"particle_picking_results_cryosparc_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(picking_results, f, indent=2)
        
        self.logger.info(f"Particle picking results saved to {output_file}")
        return str(output_file)


class CryoSPARCReconstructionParser:
    """Parser for CryoSPARC 3D reconstruction workflow results."""
    
    def __init__(self, cryosparc_tools: CryoSPARCTools, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the CryoSPARC reconstruction parser.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            logger_instance: Optional logger instance
        """
        self.cryosparc_tools = cryosparc_tools
        self.logger = logger_instance or logger
    
    def process_workflow_results(self, results: List, context: WorkflowContext) -> Dict[str, Any]:
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
                    if stage_outputs["reconstruction_type"] in ["ab_initio", "homogeneous_reconstruction"]:
                        stage_outputs["reconstruction_type"] = "refined_" + stage_outputs["reconstruction_type"]
                elif step_name == "heterogeneous_refinement":
                    stage_outputs["heterogeneous_refinement_job_uid"] = result.job_uid
                    stage_outputs["final_volume_job_uid"] = result.job_uid
                    stage_outputs["reconstruction_type"] = "heterogeneous_refined"
        
        # Get volume output directory if available
        final_volume_job_uid = stage_outputs.get("final_volume_job_uid")
        project_uid = context.project_uid
        
        if final_volume_job_uid and project_uid:
            try:
                job_info = self.cryosparc_tools.get_job_output_directory(project_uid, final_volume_job_uid)
                job_directory = job_info.get("job_directory")
                stage_outputs["volume_location"] = job_directory
                stage_outputs["volume_job_metadata"] = job_info
                
                if job_directory:
                    job_path = Path(job_directory)
                    stage_outputs["final_volume_absolute_path"] = str(job_path.absolute())
                    
            except Exception as exc:
                self.logger.warning(f"Failed to resolve volume job directory for {final_volume_job_uid}: {exc}")
        
        return stage_outputs
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
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
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """Save 3D reconstruction results to a JSON file."""
        from datetime import datetime
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        
        output_file = output_dir / f"reconstruction_results_cryosparc_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(reconstruction_results, f, indent=2)
        
        self.logger.info(f"3D reconstruction results saved to {output_file}")
        return str(output_file)
