"""ReAct-based preprocessing workflow orchestrator."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .preprocessing_agent import PreprocessingAgent
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt
from ..stage_result_parser import index_execution_log, check_step


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
        microscope_config = getattr(self.agent, "microscope_config", {})
        preprocessing_config = getattr(self.agent, "preprocessing_config", {})
        workflow_config = preprocessing_config.get("workflow", {})
        ctf_config = workflow_config.get("ctf_estimation", {})
        micrograph_selection_config = workflow_config.get("micrograph_selection", {})

        micrographs_path = microscope_config.get("micrographs_path")
        movie_sets = getattr(self.agent, "_get_movie_sets", lambda: [])()
        if movie_sets:
            movies_path = "; ".join(
                f"{movie_set.get('name', f'set_{index + 1}')}: {movie_set.get('movies_path', 'N/A')}"
                for index, movie_set in enumerate(movie_sets)
            )
        else:
            movies_path = microscope_config.get("movies_path", "N/A")

        min_resolution = micrograph_selection_config.get("min_resolution", 5.0)
        return load_prompt(
            "cryosparc/preprocessing/task.md",
            {
                "movies_path": movies_path,
                "micrographs_or_movies_path": micrographs_path or movies_path,
                "pixel_size": microscope_config.get("pixel_size", "N/A"),
                "voltage": microscope_config.get("voltage", "N/A"),
                "cs_mm": microscope_config.get("cs_mm", "N/A"),
                "dose": microscope_config.get("dose", "N/A"),
                "project_uid": self.config.workflow.project_uid,
                "workspace_uid": self.config.workflow.workspace_uid,
                "ctf_min_res": ctf_config.get("min_res", 30.0),
                "ctf_max_res": ctf_config.get("max_res", 4.0),
                "min_resolution": min_resolution,
            },
        )
    
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

        tool_entries, waits = index_execution_log(execution_log)

        for step in [PreprocessingStep.IMPORT_MOVIES, PreprocessingStep.IMPORT_MICROGRAPHS,
                     PreprocessingStep.MOTION_CORRECTION, PreprocessingStep.CTF_ESTIMATION,
                     PreprocessingStep.MICROGRAPH_SELECTION]:
            outcome = check_step(tool_entries, waits, step.value)
            self.results.append(
                PreprocessingResult(
                    step=step,
                    success=outcome.success,
                    job_uid=outcome.job_uid,
                    message=outcome.message,
                    error=outcome.error,
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

