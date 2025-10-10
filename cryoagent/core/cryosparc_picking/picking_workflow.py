"""ReAct-based particle picking workflow orchestrator."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .picking_agent import PickingAgent
from ...config.config_loader import CryoAgentConfig


class PickingStep(Enum):
    """Enumeration of particle picking workflow steps."""
    BLOB_PICKER = "blob_picker"


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
    
    def __init__(self, agent: PickingAgent, config: CryoAgentConfig):
        """
        Initialize the particle picking workflow.
        
        Args:
            agent: Particle picking agent instance
            config: Complete configuration object
        """
        self.agent = agent
        self.config = config
        self.results: List[PickingResult] = []
        self.current_job_uids: Dict[PickingStep, str] = {}
        self.workflow_state: Dict[str, Any] = {}
    
    def run(
        self,
        micrographs_job_uid: str,
        particle_diameter: float,
        diameter_max: Optional[float] = None,
        conversation_id: Optional[str] = None
    ) -> List[PickingResult]:
        """
        Run the particle picking workflow using ReAct approach.
        
        Args:
            micrographs_job_uid: Job UID from micrograph selection or CTF estimation
            particle_diameter: Minimum particle diameter in Angstroms
            diameter_max: Maximum particle diameter in Angstroms (optional, defaults to 2x particle_diameter)
            conversation_id: Optional conversation identifier for memory control
            
        Returns:
            List of picking results
        """
        self.results = []
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "active_jobs": {},
            "workflow_status": "starting"
        }
        
        workflow_input = self._create_workflow_input(
            micrographs_job_uid,
            particle_diameter,
            diameter_max
        )
        
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
    
    def _create_workflow_input(
        self,
        micrographs_job_uid: str,
        particle_diameter: float,
        diameter_max: Optional[float] = None
    ) -> str:
        """Create the workflow input for the ReAct agent."""
        diameter_range = f"{particle_diameter}-{diameter_max}" if diameter_max else f"{particle_diameter}-{particle_diameter * 2.0}"
        diameter_max_text = f"   - Max diameter: {diameter_max} Å" if diameter_max else f"   - Max diameter: Auto (2.0 × min diameter)"
        
        return f"""
Execute the particle picking workflow with GPU-accelerated blob picker:

**Input**: Micrographs from job {micrographs_job_uid}

**Task**: Detect and pick particles using Gaussian blob detection

**Parameters**:
   - Micrographs Job UID: {micrographs_job_uid}
   - Min particle diameter: {particle_diameter} Å
{diameter_max_text}
   - Diameter search range: {diameter_range} Å
   - Project: {self.config.workflow.project_uid}
   - Workspace: {self.config.workflow.workspace_uid}

**Steps**:
1. Verify that the input micrographs job ({micrographs_job_uid}) exists and is completed
2. Execute blob_picker with the specified diameter range
3. Wait for the blob picker GPU job to complete
4. Report the final status and particle picking statistics

**Important**: 
- Blob picker uses GPU-accelerated Gaussian blob detection to identify circular features
- Particle diameter range should encompass the expected size of your protein complex
- The blob picker will search for particles with diameters between min and max values
- Always wait for job completion before finishing
- Handle any errors gracefully

Start by reasoning about the particle picking parameters and then proceed with the blob picker.
"""
    
    def _parse_workflow_result(self, result: str) -> None:
        """Parse the workflow result to extract blob picker results."""
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

        # Check blob_picker step
        step = PickingStep.BLOB_PICKER
        records = tool_entries.get(step.value, [])
        
        if not records:
            self.results.append(
                PickingResult(
                    step=step,
                    success=False,
                    error="Blob picker was never executed",
                    message="No tool invocation recorded",
                    reasoning=result
                )
            )
            return

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
            return

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
            return

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
            return

        status = wait_info.get("status")
        success = status == "completed"
        message = f"CryoSPARC blob picker job {job_uid} completed successfully" if success else f"CryoSPARC blob picker job {job_uid} finished with status '{status}'"
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

