"""ReAct-based preprocessing workflow orchestrator."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .preprocessing_agent import PreprocessingAgent
from ...config.config_loader import CryoAgentConfig


class PreprocessingStep(Enum):
    """Enumeration of preprocessing workflow steps."""
    IMPORT_MOVIES = "import_movies"
    IMPORT_MICROGRAPHS = "import_micrographs"
    MOTION_CORRECTION = "motion_correction"
    CTF_ESTIMATION = "ctf_estimation"
    MICROGRAPH_SELECTION = "micrograph_selection"


@dataclass
class PreprocessingResult:
    """Result of a preprocessing workflow execution."""
    step: PreprocessingStep
    success: bool
    job_uid: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None


class PreprocessingWorkflow:
    """ReAct-based orchestrator for cryoEM preprocessing workflows."""
    
    def __init__(self, agent: PreprocessingAgent, config: CryoAgentConfig):
        """
        Initialize the preprocessing workflow.
        
        Args:
            agent: Preprocessing agent instance
            config: Complete configuration object
        """
        self.agent = agent
        self.config = config
        self.results: List[PreprocessingResult] = []
        self.current_job_uids: Dict[PreprocessingStep, str] = {}
        self.workflow_state: Dict[str, Any] = {}
    
    def run(self, conversation_id: Optional[str] = None) -> List[PreprocessingResult]:
        """
        Run the complete preprocessing workflow using ReAct approach.
        
        Args:
            conversation_id: Optional conversation identifier for memory control
            
        Returns:
            List of preprocessing results for each step
        """
        self.results = []
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "active_jobs": {},
            "workflow_status": "starting"
        }
        
        workflow_input = self._create_workflow_input()
        
        try:
            result = self.agent.run_react_workflow(workflow_input, conversation_id)
            self._parse_workflow_result(result)
            
        except Exception as e:
            error_result = PreprocessingResult(
                step=PreprocessingStep.IMPORT_MOVIES,
                success=False,
                error=f"Preprocessing workflow execution failed: {str(e)}",
                message="ReAct workflow failed to execute"
            )
            self.results.append(error_result)
        
        return self.results
    
    def _create_workflow_input(self) -> str:
        """Create the workflow input for the ReAct agent."""
        # Get microscope config from the agent
        microscope_config = getattr(self.agent, 'microscope_config', {})
        
        # Get preprocessing config from the agent to read workflow parameters
        preprocessing_config = getattr(self.agent, 'preprocessing_config', {})
        workflow_config = preprocessing_config.get('workflow', {})
        motion_correction_config = workflow_config.get('motion_correction', {})
        ctf_config = workflow_config.get('ctf_estimation', {})
        micrograph_selection_config = workflow_config.get('micrograph_selection', {})
        
        # Check if micrographs_path is available (indicates direct micrograph import)
        micrographs_path = microscope_config.get('micrographs_path')
        movie_sets = getattr(self.agent, "_get_movie_sets", lambda: [])()
        if movie_sets:
            movies_path = "; ".join(
                f"{movie_set.get('name', f'set_{index + 1}')}: {movie_set.get('movies_path', 'N/A')}"
                for index, movie_set in enumerate(movie_sets)
            )
        else:
            movies_path = microscope_config.get('movies_path', 'N/A')
        
        # Get min_resolution from config, default to 5.0 if not found
        min_resolution = micrograph_selection_config.get('min_resolution', 5.0)
        
        return f"""
Execute the complete cryoEM preprocessing workflow. Choose the appropriate path based on your input data:

**Option A: If you have raw movie files:**
1. **Import Movies**: Import movie files from {movies_path}
   - Pixel size: {microscope_config.get('pixel_size', 'N/A')} Å
   - Voltage: {microscope_config.get('voltage', 'N/A')} kV
   - CS: {microscope_config.get('cs_mm', 'N/A')} mm
   - Dose: {microscope_config.get('dose', 'N/A')} e-/Å²
   - Project: {self.config.workflow.project_uid}
   - Workspace: {self.config.workflow.workspace_uid}
   - If movies_path is a list, import all paths in one import_movies call

2. **Motion Correction**: Correct motion in the imported movies
   - Connect all import job UIDs to a single motion correction job when multiple sets were imported
   - Binning: {motion_correction_config.get('binning', 1)}
   - Patch size: {motion_correction_config.get('patch_size', 5)}

3. **CTF Estimation**: Estimate CTF parameters for micrographs
   - Min resolution: {ctf_config.get('min_res', 30.0)} Å
   - Max resolution: {ctf_config.get('max_res', 4.0)} Å

**Option B: If you have already motion-corrected micrographs:**
1. **Import Micrographs**: Import micrograph files directly from {micrographs_path or movies_path}
   - Pixel size: {microscope_config.get('pixel_size', 'N/A')} Å
   - Voltage: {microscope_config.get('voltage', 'N/A')} kV
   - CS: {microscope_config.get('cs_mm', 'N/A')} mm
   - Dose: {microscope_config.get('dose', 'N/A')} e-/Å²
   - Project: {self.config.workflow.project_uid}
   - Workspace: {self.config.workflow.workspace_uid}
   - **CRITICAL**: Skip motion correction and proceed directly to CTF estimation

2. **CTF Estimation**: Estimate CTF parameters for micrographs
   - Min resolution: {ctf_config.get('min_res', 30.0)} Å
   - Max resolution: {ctf_config.get('max_res', 4.0)} Å

**Common Final Step:**
4. **Micrograph Selection**: Select micrographs with resolution better than {min_resolution} Å
   - Min resolution threshold: {min_resolution} Å
   - Filters out low-quality micrographs

**Important**: 
- Each step must complete successfully before the next begins
- If using import_micrographs, DO NOT run motion_correction
- Always check job status and wait for completion
- Handle any errors gracefully
- Provide clear status updates throughout the process

Start by reasoning about the workflow state and then proceed step by step.
"""
    
    def _parse_workflow_result(self, result: str) -> None:
        """Parse the workflow result to extract individual step results."""
        execution_log = self.agent.get_tool_execution_log()

        if not execution_log:
            error_result = PreprocessingResult(
                step=PreprocessingStep.IMPORT_MOVIES,
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

        for step in [PreprocessingStep.IMPORT_MOVIES, PreprocessingStep.IMPORT_MICROGRAPHS, 
                     PreprocessingStep.MOTION_CORRECTION, PreprocessingStep.CTF_ESTIMATION, 
                     PreprocessingStep.MICROGRAPH_SELECTION]:
            records = tool_entries.get(step.value, [])
            if not records:
                self.results.append(
                    PreprocessingResult(
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
            job_uids_to_check: List[str] = []
            if isinstance(result_payload, dict):
                if result_payload.get("job_uids"):
                    job_uids_to_check = list(result_payload["job_uids"])
                elif job_uid:
                    job_uids_to_check = [job_uid]

            if error_message:
                self.results.append(
                    PreprocessingResult(
                        step=step,
                        success=False,
                        job_uid=job_uid,
                        error=error_message,
                        message="Tool execution reported an error",
                        reasoning=result
                    )
                )
                continue

            if not job_uids_to_check:
                self.results.append(
                    PreprocessingResult(
                        step=step,
                        success=False,
                        error="Tool did not return a job UID",
                        message="Unable to confirm CryoSPARC job submission",
                        reasoning=result
                    )
                )
                continue

            missing_waits = [uid for uid in job_uids_to_check if uid not in waits]
            if missing_waits:
                self.results.append(
                    PreprocessingResult(
                        step=step,
                        success=False,
                        job_uid=job_uid,
                        error="Job completion was not confirmed",
                        message=f"Missing wait_for_job invocation for: {', '.join(missing_waits)}",
                        reasoning=result
                    )
                )
                continue

            statuses = [waits[uid].get("status") for uid in job_uids_to_check]
            success = all(status == "completed" for status in statuses)
            if len(job_uids_to_check) == 1:
                message = (
                    f"CryoSPARC job {job_uids_to_check[0]} completed successfully"
                    if success
                    else f"CryoSPARC job {job_uids_to_check[0]} finished with status '{statuses[0]}'"
                )
            else:
                message = (
                    f"CryoSPARC jobs {', '.join(job_uids_to_check)} completed successfully"
                    if success
                    else f"CryoSPARC jobs finished with statuses: {', '.join(statuses)}"
                )
            error = None if success else f"Job statuses: {', '.join(str(s) for s in statuses)}"

            self.results.append(
                PreprocessingResult(
                    step=step,
                    success=success,
                    job_uid=job_uid,
                    message=message,
                    error=error,
                    reasoning=result
                )
            )
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the preprocessing workflow execution."""
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

