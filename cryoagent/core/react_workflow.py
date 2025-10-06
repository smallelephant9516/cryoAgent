"""ReAct-based CryoEM workflow orchestrator."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .react_agent import ReActCryoEMAgent
from ..config.config_loader import CryoAgentConfig


class WorkflowStep(Enum):
    """Enumeration of workflow steps."""
    IMPORT_MOVIES = "import_movies"
    MOTION_CORRECTION = "motion_correction"
    CTF_ESTIMATION = "ctf_estimation"
    MICROGRAPH_SELECTION = "micrograph_selection"


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""
    step: WorkflowStep
    success: bool
    job_uid: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None


class ReActCryoEMWorkflow:
    """ReAct-based orchestrator for cryoEM processing workflows."""
    
    def __init__(self, agent: ReActCryoEMAgent, config: CryoAgentConfig):
        """
        Initialize the ReAct workflow.
        
        Args:
            agent: ReAct CryoEM agent instance
            config: Complete configuration object
        """
        self.agent = agent
        self.config = config
        self.results: List[WorkflowResult] = []
        self.current_job_uids: Dict[WorkflowStep, str] = {}
        self.workflow_state: Dict[str, Any] = {}
    
    def run_basic_workflow(self, conversation_id: Optional[str] = None) -> List[WorkflowResult]:
        """
        Run the basic cryoEM workflow using ReAct approach.
        
        Returns:
            List of workflow results for each step
        """
        self.results = []
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "active_jobs": {},
            "workflow_status": "starting"
        }
        
        # Use ReAct agent to orchestrate the entire workflow
        workflow_input = self._create_workflow_input()
        
        try:
            # Execute the workflow using ReAct approach
            result = self.agent.run_react_workflow(workflow_input, conversation_id)
            
            # Parse the result to extract individual step results
            self._parse_workflow_result(result)
            
        except Exception as e:
            # Create error result
            error_result = WorkflowResult(
                step=WorkflowStep.IMPORT_MOVIES,  # Default to first step
                success=False,
                error=f"Workflow execution failed: {str(e)}",
                message="ReAct workflow failed to execute"
            )
            self.results.append(error_result)
        
        return self.results
    
    def _create_workflow_input(self) -> str:
        """Create the workflow input for the ReAct agent."""
        return f"""
Execute the complete cryoEM processing workflow with these steps:

1. **Import Movies**: Import movie files from {self.config.workflow.movies_path}
   - Pixel size: {self.config.workflow.pixel_size} Å
   - Voltage: {self.config.workflow.voltage} kV
   - CS: {self.config.workflow.cs_mm} mm
   - Dose: {self.config.workflow.dose} e-/Å²
   - Project: {self.config.workflow.project_uid}
   - Workspace: {self.config.workflow.workspace_uid}

2. **Motion Correction**: Correct motion in the imported movies
   - Binning: {self.config.workflow.motion_correction_binning}
   - Patch size: {self.config.workflow.motion_correction_patch_size}

3. **CTF Estimation**: Estimate CTF parameters for micrographs
   - Min resolution: {self.config.workflow.ctf_min_res} Å
   - Max resolution: {self.config.workflow.ctf_max_res} Å

4. **Micrograph Selection**: Select micrographs with resolution better than 5 Å
   - Min resolution threshold: 5.0 Å
   - Filters out low-quality micrographs

**Important**: 
- Each step must complete successfully before the next begins
- Always check job status and wait for completion
- Handle any errors gracefully
- Provide clear status updates throughout the process

Start by reasoning about the workflow state and then proceed step by step.
"""
    
    def _parse_workflow_result(self, result: str) -> None:
        """Parse the workflow result to extract individual step results."""
        execution_log = self.agent.get_tool_execution_log()

        if not execution_log:
            error_result = WorkflowResult(
                step=WorkflowStep.IMPORT_MOVIES,
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

        for step in [WorkflowStep.IMPORT_MOVIES, WorkflowStep.MOTION_CORRECTION, WorkflowStep.CTF_ESTIMATION, WorkflowStep.MICROGRAPH_SELECTION]:
            records = tool_entries.get(step.value, [])
            if not records:
                self.results.append(
                    WorkflowResult(
                        step=step,
                        success=False,
                        error="Step was never executed",
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
                    WorkflowResult(
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
                    WorkflowResult(
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
                    WorkflowResult(
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
            message = f"CryoSPARC job {job_uid} completed successfully" if success else f"CryoSPARC job {job_uid} finished with status '{status}'"
            error = None if success else f"Job status: {status}"

            self.results.append(
                WorkflowResult(
                    step=step,
                    success=success,
                    job_uid=job_uid,
                    message=message,
                    error=error,
                    reasoning=result
                )
            )
    
    def run_custom_workflow(self, steps: List[WorkflowStep]) -> List[WorkflowResult]:
        """
        Run a custom workflow with specified steps using ReAct approach.
        
        Args:
            steps: List of workflow steps to execute
            
        Returns:
            List of workflow results
        """
        self.results = []
        
        # Create custom workflow input
        workflow_input = self._create_custom_workflow_input(steps)
        
        try:
            result = self.agent.run_react_workflow(workflow_input)
            self._parse_workflow_result(result)
        except Exception as e:
            error_result = WorkflowResult(
                step=steps[0] if steps else WorkflowStep.IMPORT_MOVIES,
                success=False,
                error=f"Custom workflow execution failed: {str(e)}",
                message="ReAct custom workflow failed to execute"
            )
            self.results.append(error_result)
        
        return self.results
    
    def _create_custom_workflow_input(self, steps: List[WorkflowStep]) -> str:
        """Create custom workflow input for specified steps."""
        step_descriptions = []
        
        for i, step in enumerate(steps, 1):
            if step == WorkflowStep.IMPORT_MOVIES:
                step_descriptions.append(f"""
{i}. **Import Movies**: Import movie files from {self.config.workflow.movies_path}
   - Pixel size: {self.config.workflow.pixel_size} Å
   - Voltage: {self.config.workflow.voltage} kV
   - CS: {self.config.workflow.cs_mm} mm
   - Dose: {self.config.workflow.dose} e-/Å²
""")
            elif step == WorkflowStep.MOTION_CORRECTION:
                step_descriptions.append(f"""
{i}. **Motion Correction**: Correct motion in imported movies
   - Binning: {self.config.workflow.motion_correction_binning}
   - Patch size: {self.config.workflow.motion_correction_patch_size}
""")
            elif step == WorkflowStep.CTF_ESTIMATION:
                step_descriptions.append(f"""
{i}. **CTF Estimation**: Estimate CTF parameters for micrographs
   - Min resolution: {self.config.workflow.ctf_min_res} Å
   - Max resolution: {self.config.workflow.ctf_max_res} Å
""")
            elif step == WorkflowStep.MICROGRAPH_SELECTION:
                step_descriptions.append(f"""
{i}. **Micrograph Selection**: Select micrographs with resolution better than 5 Å
   - Min resolution threshold: 5.0 Å
   - Filters out low-quality micrographs
""")
        
        return f"""
Execute the following custom cryoEM workflow:

{''.join(step_descriptions)}

**Important**: 
- Each step must complete successfully before the next begins
- Always check job status and wait for completion
- Handle any errors gracefully
- Provide clear status updates throughout the process

Start by reasoning about the workflow state and then proceed step by step.
"""
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow execution."""
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
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get the current workflow state."""
        return {
            "workflow_state": self.workflow_state,
            "current_job_uids": self.current_job_uids,
            "results": [
                {
                    "step": r.step.value,
                    "success": r.success,
                    "job_uid": r.job_uid,
                    "message": r.message
                }
                for r in self.results
            ]
        }
    
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
