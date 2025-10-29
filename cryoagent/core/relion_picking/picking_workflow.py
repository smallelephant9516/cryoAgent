"""Particle picking workflow for RELION CryoEM data processing."""

import json
import os
import time
from typing import Dict, Any, List, Optional
from pathlib import Path


class PickingWorkflow:
    """Workflow for particle picking and 2D classification."""
    
    def __init__(self, config_path: str = "configs/relion/particle_picking_config.json"):
        """Initialize the particle picking workflow."""
        self.config_path = config_path
        self.config = self._load_config()
        self.current_stage = None
        self.current_step = None
        self.workflow_results = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load particle picking configuration."""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            "workflow_name": "particle_picking_workflow",
            "stages": {
                "stage1": {
                    "name": "particle_picking_and_classification",
                    "steps": [
                        {"step_name": "blob_picker", "tool": "blob_picker"},
                        {"step_name": "particle_extraction", "tool": "particle_extraction"},
                        {"step_name": "classification_2d", "tool": "classification_2d"},
                        {"step_name": "auto_2d_selection", "tool": "auto_2d_selection"}
                    ]
                }
            }
        }
    
    def get_workflow_steps(self) -> List[Dict[str, Any]]:
        """Get all workflow steps."""
        steps = []
        for stage_name, stage_data in self.config["stages"].items():
            for step in stage_data["steps"]:
                steps.append({
                    "stage": stage_name,
                    "step_name": step["step_name"],
                    "tool": step["tool"]
                })
        return steps
    
    def get_step_parameters(self, step_name: str) -> Dict[str, Any]:
        """Get default parameters for a specific step."""
        for stage_name, stage_data in self.config["stages"].items():
            for step in stage_data["steps"]:
                if step["step_name"] == step_name:
                    return step.get("parameters", {})
        return {}
    
    def validate_workflow_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate workflow inputs."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required inputs
        required_inputs = ["input_micrographs_star"]
        for req_input in required_inputs:
            if req_input not in inputs:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Missing required input: {req_input}")
        
        # Validate file existence
        if "input_micrographs_star" in inputs:
            if not os.path.exists(inputs["input_micrographs_star"]):
                validation_results["valid"] = False
                validation_results["errors"].append(f"Input file does not exist: {inputs['input_micrographs_star']}")
        
        return validation_results
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow."""
        return {
            "workflow_name": self.config["workflow_name"],
            "total_stages": len(self.config["stages"]),
            "total_steps": sum(len(stage["steps"]) for stage in self.config["stages"].values()),
            "stages": list(self.config["stages"].keys())
        }
    
    def get_next_step(self, completed_steps: List[str]) -> Optional[Dict[str, Any]]:
        """Get the next step to execute based on completed steps."""
        all_steps = self.get_workflow_steps()
        
        for step in all_steps:
            if step["step_name"] not in completed_steps:
                return step
        
        return None
    
    def is_workflow_complete(self, completed_steps: List[str]) -> bool:
        """Check if the workflow is complete."""
        all_steps = self.get_workflow_steps()
        return len(completed_steps) >= len(all_steps)
    
    def get_workflow_dependencies(self) -> Dict[str, List[str]]:
        """Get workflow dependencies between steps."""
        return {
            "blob_picker": [],
            "particle_extraction": ["blob_picker"],
            "classification_2d": ["particle_extraction"],
            "auto_2d_selection": ["classification_2d"]
        }
    
    def get_step_outputs(self, step_name: str) -> Dict[str, str]:
        """Get expected outputs for a step."""
        outputs = {
            "blob_picker": {
                "coordinate_files": "AutoPick/jobXXX/*_autopick.star files",
                "pipeline_control": "AutoPick/jobXXX/RELION_JOB_EXIT_SUCCESS"
            },
            "particle_extraction": {
                "particle_stack": "Particles/jobXXX/particles.mrcs",
                "particle_star": "Particles/jobXXX/particles.star",
                "pipeline_control": "Particles/jobXXX/RELION_JOB_EXIT_SUCCESS"
            },
            "classification_2d": {
                "optimiser_star": "Class2D/jobXXX/run_optimiser.star",
                "class_averages": "Class2D/jobXXX/run_class001.mrc",
                "data_star": "Class2D/jobXXX/run_data.star",
                "pipeline_control": "Class2D/jobXXX/RELION_JOB_EXIT_SUCCESS"
            },
            "auto_2d_selection": {
                "selected_particles": "Select/jobXXX/particles.star",
                "selected_class_averages": "Select/jobXXX/class_averages.star",
                "pipeline_control": "Select/jobXXX/RELION_JOB_EXIT_SUCCESS"
            }
        }
        
        return outputs.get(step_name, {})
    
    def get_workflow_parameters(self) -> Dict[str, Any]:
        """Get workflow-level parameters."""
        return self.config.get("workflow_parameters", {})
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules for the workflow."""
        return self.config.get("validation_rules", {})
    
    def get_error_handling_config(self) -> Dict[str, Any]:
        """Get error handling configuration."""
        return self.config.get("error_handling", {})
    
    def get_output_analysis_config(self) -> Dict[str, Any]:
        """Get output analysis configuration."""
        return self.config.get("output_analysis", {})
