"""
RELION Parser Tools

This module provides utilities for parsing, validating, and saving RELION workflow results.
These tools are designed to be used by the modular agents to handle backend-specific result processing.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


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


class RelionPreprocessingParser:
    """Parser for RELION preprocessing workflow results."""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the RELION preprocessing parser.
        
        Args:
            logger_instance: Optional logger instance
        """
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
            "import_job_dir": None,
            "motion_correction_job_dir": None,
            "ctf_job_dir": None,
            "selection_job_dir": None,
            "movies_star_file": None,
            "corrected_micrographs_star": None,
            "ctf_star_file": None,
            "selected_micrographs_star": None
        }
        
        # Extract job directories and output files from workflow results
        for result in results:
            step_name = result.step.value
            if result.success:
                if step_name == "import_movies":
                    stage_outputs["import_job_dir"] = result.job_dir
                    stage_outputs["movies_star_file"] = result.output_file
                elif step_name == "motion_correction":
                    stage_outputs["motion_correction_job_dir"] = result.job_dir
                    stage_outputs["corrected_micrographs_star"] = result.output_file
                elif step_name == "ctf_estimation":
                    stage_outputs["ctf_job_dir"] = result.job_dir
                    stage_outputs["ctf_star_file"] = result.output_file
                elif step_name == "micrograph_selection":
                    stage_outputs["selection_job_dir"] = result.job_dir
                    stage_outputs["selected_micrographs_star"] = result.output_file
                    # Extract job ID from job directory path for compatibility with master orchestrator
                    if result.job_dir:
                        job_id = os.path.basename(result.job_dir)
                        stage_outputs["micrograph_selection_job_uid"] = job_id
        
        return stage_outputs
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that RELION preprocessing jobs were executed successfully.
        
        Args:
            stage_outputs: Dictionary of stage outputs to validate
            
        Returns:
            Dictionary with 'success' boolean and 'error' message if failed
        """
        try:
            selected_micrographs_star = stage_outputs.get("selected_micrographs_star")
            if selected_micrographs_star:
                star_path = Path(selected_micrographs_star)
                if not star_path.is_absolute():
                    selection_dir = stage_outputs.get("selection_job_dir")
                    if selection_dir:
                        base_dir = Path(selection_dir).resolve()
                        if base_dir.is_dir():
                            candidate = (base_dir / star_path).resolve()
                            if candidate.exists():
                                return {"success": True, "error": None}
                    relion_dir = stage_outputs.get("relion_dir")
                    if relion_dir:
                        candidate = (Path(relion_dir).resolve() / star_path).resolve()
                        if candidate.exists():
                            return {"success": True, "error": None}
                elif star_path.exists():
                    return {"success": True, "error": None}

            required_files = [
                stage_outputs.get("movies_star_file"),
                stage_outputs.get("corrected_micrographs_star"),
                stage_outputs.get("ctf_star_file"),
                stage_outputs.get("selected_micrographs_star")
            ]

            missing_files = [
                f for f in required_files
                if not f or not Path(f).exists()
            ]
            if missing_files:
                return {
                    "success": False,
                    "error": f"Missing output files: {missing_files}"
                }

            return {"success": True, "error": None}

        except Exception as exc:
            return {
                "success": False,
                "error": f"Validation error: {str(exc)}"
            }
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """
        Save RELION preprocessing results to JSON file.
        
        Args:
            stage_outputs: Dictionary of stage outputs
            context: Workflow context
            success: Whether the preprocessing was successful
            
        Returns:
            Path to the saved JSON file
        """
        import time
        
        try:
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            result_file = outputs_dir / f"preprocessing_results_relion_{timestamp}.json"

            selection_job_dir = stage_outputs.get("selection_job_dir")
            relion_dir = stage_outputs.get("relion_dir")
            if not relion_dir and selection_job_dir:
                selection_path = Path(selection_job_dir).resolve()
                if selection_path.is_dir():
                    relion_dir = str(selection_path.parent.parent)

            relion_dir_path = Path(relion_dir) if relion_dir else None

            def resolve_path(value: Optional[str]) -> Optional[str]:
                if not value:
                    return None
                path_obj = Path(value)
                try:
                    if path_obj.is_absolute():
                        return str(path_obj.resolve())
                    if relion_dir_path:
                        candidate = (relion_dir_path / path_obj).resolve()
                        return str(candidate)
                    return str(path_obj.resolve())
                except Exception:
                    if relion_dir_path:
                        return str(relion_dir_path / path_obj)
                    return str(path_obj)

            selected_star = stage_outputs.get("selected_micrographs_star")
            if selected_star:
                selected_star = resolve_path(selected_star)

            motion_correction_job_dir = resolve_path(stage_outputs.get("motion_correction_job_dir"))
            selection_job_dir_resolved = resolve_path(selection_job_dir)

            micrographs_folder = None
            if motion_correction_job_dir:
                motion_corr_path = Path(motion_correction_job_dir)
                movies_dir = motion_corr_path / "Movies"
                micrographs_dir = motion_corr_path / "Micrographs"
                if movies_dir.exists():
                    micrographs_folder = str(movies_dir)
                elif micrographs_dir.exists():
                    micrographs_folder = str(micrographs_dir)
                else:
                    micrographs_folder = str(micrographs_dir)

            results_data = {
                "timestamp": timestamp,
                "status": "completed" if success else "failed",
                "stage": "preprocessing",
                "agent_type": "relion",
                "relion_dir": relion_dir,
                "selected_micrographs_star": selected_star,
                "micrograph_selection_job_dir": selection_job_dir_resolved
            }

            if micrographs_folder:
                results_data["micrographs_folder"] = micrographs_folder

            with open(result_file, "w") as handle:
                json.dump(results_data, handle, indent=2)

            self.logger.info(f"RELION preprocessing results saved to {result_file}")
            return str(result_file)

        except Exception as exc:
            self.logger.error(f"Failed to save RELION preprocessing results: {exc}")
            return ""


class RelionPickingParser:
    """Parser for RELION particle picking workflow results."""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the RELION particle picking parser.
        
        Args:
            logger_instance: Optional logger instance
        """
        self.logger = logger_instance or logger
    
    def process_workflow_results(self, results: List, context: WorkflowContext) -> Dict[str, Any]:
        """Process RELION particle picking workflow results."""
        # Implementation for RELION picking
        pass
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate RELION particle picking results."""
        pass
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """Save RELION particle picking results."""
        pass


class RelionReconstructionParser:
    """Parser for RELION 3D reconstruction workflow results."""
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize the RELION reconstruction parser.
        
        Args:
            logger_instance: Optional logger instance
        """
        self.logger = logger_instance or logger
    
    def process_workflow_results(self, results: List, context: WorkflowContext) -> Dict[str, Any]:
        """Process RELION 3D reconstruction workflow results."""
        # Implementation for RELION reconstruction
        pass
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate RELION reconstruction results."""
        pass
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """Save RELION reconstruction results."""
        pass
