"""ReAct-based particle picking workflow orchestrator."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .picking_agent import PickingAgent
from ...config.config_loader import CryoAgentConfig


class PickingStep(Enum):
    """Enumeration of particle picking workflow steps."""
    BLOB_PICKER = "blob_picker"
    EXTRACT_PARTICLES = "extract_particles"
    CLASS_2D = "class_2d"


@dataclass
class PickingResult:
    """Result of a particle picking workflow execution."""
    step: PickingStep
    success: bool
    job_uid: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None


class PickingWorkflow:
    """ReAct-based orchestrator for particle picking workflows."""
    
    def __init__(self, agent: PickingAgent, config: CryoAgentConfig, stage_config_path: Optional[str] = None):
        """
        Initialize the particle picking workflow.
        
        Args:
            agent: Particle picking agent instance
            config: Complete configuration object
            stage_config_path: Path to stage-specific configuration file
        """
        self.agent = agent
        self.config = config
        self.results: List[PickingResult] = []
        self.current_job_uids: Dict[PickingStep, str] = {}
        self.workflow_state: Dict[str, Any] = {}
        
        # Load stage-specific configuration
        self.stage_config = self._load_stage_config(stage_config_path)
        self.workflow_params = self._parse_workflow_params()
    
    def _load_stage_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load stage-specific configuration from JSON file."""
        import json
        from pathlib import Path
        
        if not config_path:
            return {}
        
        stage_config_path = Path(config_path)
        if not stage_config_path.exists():
            return {}
        
        with open(stage_config_path, 'r') as f:
            return json.load(f)
    
    def _parse_workflow_params(self) -> Dict[str, Any]:
        """Parse workflow parameters from stage config."""
        workflow_config = self.stage_config.get("workflow", {})
        
        # Blob picker parameters
        blob_picker_config = workflow_config.get("blob_picker", {})
        particle_diameter = blob_picker_config.get("particle_diameter", 180.0)
        diameter_max = blob_picker_config.get("diameter_max", None)
        
        # Particle extraction parameters
        extraction_config = workflow_config.get("particle_extraction", {})
        box_size_pix = extraction_config.get("box_size_pix", 256)
        extract_downscale_factor = extraction_config.get("extract_downscale_factor", 1.0)
        bg_radius = extraction_config.get("bg_radius", None)
        invert_contrast = extraction_config.get("invert_contrast", True)
        
        # 2D classification parameters
        classification_config = workflow_config.get("2d_classification", {})
        num_classes = classification_config.get("num_classes", 50)
        max_iterations = classification_config.get("max_iterations", 20)
        initial_resolution = classification_config.get("initial_resolution", 12.0)
        final_resolution = classification_config.get("final_resolution", 6.0)
        batch_size_per_class = classification_config.get("batch_size_per_class", 200)
        force_max_res = classification_config.get("force_max_res", False)
        
        return {
            "particle_diameter": particle_diameter,
            "diameter_max": diameter_max,
            "box_size_pix": box_size_pix,
            "extract_downscale_factor": extract_downscale_factor,
            "bg_radius": bg_radius,
            "invert_contrast": invert_contrast,
            "num_classes": num_classes,
            "max_iterations": max_iterations,
            "initial_resolution": initial_resolution,
            "final_resolution": final_resolution,
            "batch_size_per_class": batch_size_per_class,
            "force_max_res": force_max_res
        }
    
    def run(
        self,
        micrographs_job_uid: str,
        conversation_id: Optional[str] = None
    ) -> List[PickingResult]:
        """
        Run the complete particle picking workflow using ReAct approach.
        Parameters are loaded from the stage configuration file.
        
        Args:
            micrographs_job_uid: Job UID from micrograph selection or CTF estimation
            conversation_id: Optional conversation identifier for memory control
            
        Returns:
            List of picking results for all steps
        """
        self.results = []
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "active_jobs": {},
            "workflow_status": "starting"
        }
        
        workflow_input = self._create_workflow_input(micrographs_job_uid)
        
        try:
            result = self.agent.run_react_workflow(workflow_input, conversation_id)
            self._parse_workflow_result(result)
            
        except Exception as e:
            error_result = PickingResult(
                step=PickingStep.BLOB_PICKER,
                success=False,
                error=f"Particle picking workflow execution failed: {str(e)}",
                message="ReAct workflow failed to execute"
            )
            self.results.append(error_result)
        
        return self.results
    
    def _create_workflow_input(self, micrographs_job_uid: str) -> str:
        """Create the workflow input for the ReAct agent using config parameters."""
        # Get parameters from workflow_params
        particle_diameter = self.workflow_params.get("particle_diameter", 180.0)
        diameter_max = self.workflow_params.get("diameter_max")
        box_size_pix = self.workflow_params.get("box_size_pix", 256)
        num_classes = self.workflow_params.get("num_classes", 50)
        max_iterations = self.workflow_params.get("max_iterations", 20)
        initial_resolution = self.workflow_params.get("initial_resolution", 12.0)
        final_resolution = self.workflow_params.get("final_resolution", 6.0)
        
        diameter_range = f"{particle_diameter}-{diameter_max}" if diameter_max else f"{particle_diameter}-{particle_diameter * 2.0}"
        diameter_max_text = f"   - Max diameter: {diameter_max} Å" if diameter_max else f"   - Max diameter: Auto (2.0 × min diameter)"
        
        return f"""
Execute the complete particle picking workflow with 3 steps:

**Input**: Micrographs from job {micrographs_job_uid}

**Task**: Detect particles, extract them, and perform 2D classification

**Parameters** (from config):
   - Micrographs Job UID: {micrographs_job_uid}
   - Min particle diameter: {particle_diameter} Å
{diameter_max_text}
   - Diameter search range: {diameter_range} Å
   - Box size for extraction: {box_size_pix} pixels
   - Number of 2D classes: {num_classes}
   - Max iterations: {max_iterations}
   - Initial resolution: {initial_resolution} Å
   - Final resolution: {final_resolution} Å
   - Project: {self.config.workflow.project_uid}
   - Workspace: {self.config.workflow.workspace_uid}

**Workflow Steps** (execute in order):

1. **Blob Picker**: Detect particles using GPU-accelerated blob detection
   - Execute blob_picker with particle_diameter={particle_diameter} and diameter_max={diameter_max or particle_diameter * 2.0}
   - Use micrographs_job_uid={micrographs_job_uid}
   - Wait for the blob picker job to complete
   - Record the blob picker job UID

2. **Particle Extraction**: Extract particles from micrographs
   - Execute extract_particles using:
     * particles_job_uid (from blob picker)
     * micrographs_job_uid={micrographs_job_uid} (same as blob picker input)
     * box_size_pix={box_size_pix}
   - Wait for the extraction job to complete
   - Record the extraction job UID

3. **2D Classification**: Classify extracted particles
   - Execute class_2d using the extraction job UID
   - Set num_classes={num_classes}
   - Wait for the classification job to complete
   - Report final results

**Important**: 
- Each step must complete successfully before proceeding to the next
- Always wait for each job to complete before starting the next one
- Particle extraction requires BOTH particles_job_uid (from blob picker) AND micrographs_job_uid={micrographs_job_uid}
- Handle any errors gracefully and report them
- Blob picker uses GPU-accelerated Gaussian blob detection
- Box size should be ~1.5-2x the particle diameter
- 2D classification helps assess particle quality and remove junk

Start by reasoning about the workflow parameters and then proceed step by step.
"""
    
    def _parse_workflow_result(self, result: str) -> None:
        """Parse the workflow result to extract results for all picking steps."""
        execution_log = self.agent.get_tool_execution_log()

        if not execution_log:
            error_result = PickingResult(
                step=PickingStep.BLOB_PICKER,
                success=False,
                error="No CryoSPARC tool calls were recorded during workflow execution",
                message="Agent response did not trigger any tool invocations",
                reasoning=result
            )
            self.results.append(error_result)
            return

        waits: Dict[str, Dict[str, Any]] = {}
        tool_entries: Dict[str, List[Dict[str, Any]]] = {}

        for entry in execution_log:
            tool_name = entry.get("tool")
            tool_entries.setdefault(tool_name, []).append(entry)
            if tool_name == "wait_for_job" and entry.get("result"):
                job_uid = entry.get("params", {}).get("job_uid")
                if job_uid:
                    waits[job_uid] = entry["result"]

        # Check all three steps in order
        for step in [PickingStep.BLOB_PICKER, PickingStep.EXTRACT_PARTICLES, PickingStep.CLASS_2D]:
            records = tool_entries.get(step.value, [])
            
            if not records:
                self.results.append(
                    PickingResult(
                        step=step,
                        success=False,
                        error=f"{step.value} was never executed",
                        message="No tool invocation recorded",
                        reasoning=result
                    )
                )
                continue

            latest_record = records[-1]
            error_message = latest_record.get("error")
            result_payload = latest_record.get("result", {})
            job_uid = result_payload.get("job_uid") if isinstance(result_payload, dict) else None

            if error_message:
                self.results.append(
                    PickingResult(
                        step=step,
                        success=False,
                        job_uid=job_uid,
                        error=error_message,
                        message="Tool execution reported an error",
                        reasoning=result
                    )
                )
                continue

            if not job_uid:
                self.results.append(
                    PickingResult(
                        step=step,
                        success=False,
                        error="Tool did not return a job UID",
                        message="Unable to confirm CryoSPARC job submission",
                        reasoning=result
                    )
                )
                continue

            wait_info = waits.get(job_uid)
            if not wait_info:
                self.results.append(
                    PickingResult(
                        step=step,
                        success=False,
                        job_uid=job_uid,
                        error="Job completion was not confirmed",
                        message="Missing wait_for_job invocation",
                        reasoning=result
                    )
                )
                continue

            status = wait_info.get("status")
            success = status == "completed"
            step_name = step.value.replace('_', ' ').title()
            message = f"CryoSPARC {step_name} job {job_uid} completed successfully" if success else f"CryoSPARC {step_name} job {job_uid} finished with status '{status}'"
            error = None if success else f"Job status: {status}"

            self.results.append(
                PickingResult(
                    step=step,
                    success=success,
                    job_uid=job_uid,
                    message=message,
                    error=error,
                    reasoning=result
                )
            )
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the particle picking workflow execution."""
        summary = {
            "total_steps": len(self.results),
            "successful_steps": sum(1 for r in self.results if r.success),
            "failed_steps": sum(1 for r in self.results if not r.success),
            "steps": [],
            "workflow_state": self.workflow_state,
            "reasoning_history": self.agent.get_reasoning_history()
        }
        
        for result in self.results:
            step_summary = {
                "step": result.step.value,
                "success": result.success,
                "job_uid": result.job_uid,
                "message": result.message,
                "reasoning": result.reasoning
            }
            if result.error:
                step_summary["error"] = result.error
            
            summary["steps"].append(step_summary)
        
        return summary
    
    def reset_workflow(self):
        """Reset the workflow state."""
        self.results = []
        self.current_job_uids = {}
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "active_jobs": {},
            "workflow_status": "reset"
        }
        self.agent.clear_reasoning_history()

