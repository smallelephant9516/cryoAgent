"""ReAct-based polish workflow orchestrator."""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .polish_agent import PolishAgent
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt
from ..stage_result_parser import index_execution_log, outcome_for_record
from ...tools.cryosparc_tools import CryoSPARCTools


class PolishStep(Enum):
    """Enumeration of polish workflow steps."""
    VERIFY_INPUTS = "verify_inputs"
    INITIAL_REFINEMENT = "initial_refinement"
    MOTION_CORRECTION = "motion_correction"
    FINAL_REFINEMENT = "final_refinement"


@dataclass
class PolishResult:
    """Result of a polish workflow execution."""
    step: PolishStep
    success: bool
    job_uid: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None


class PolishWorkflow:
    """ReAct-based orchestrator for polish workflows."""
    
    def __init__(self, agent: PolishAgent, config: CryoAgentConfig, stage_config_path: Optional[str] = None):
        """
        Initialize the polish workflow.
        
        Args:
            agent: Polish agent instance
            config: Complete configuration object
            stage_config_path: Optional path to stage-specific configuration file
        """
        self.agent = agent
        self.config = config
        self.results: List[PolishResult] = []
        self.workflow_state: Dict[str, Any] = {}
        self.stage_config_path = stage_config_path
    
    def run(
        self,
        conversation_id: Optional[str] = None
    ) -> List[PolishResult]:
        """
        Run the polish workflow using ReAct approach.
        
        Args:
            conversation_id: Optional conversation identifier for memory control
            
        Returns:
            List of polish results for all steps
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
            error_result = PolishResult(
                step=PolishStep.VERIFY_INPUTS,
                success=False,
                error=f"Polish workflow execution failed: {str(e)}",
                message="ReAct workflow failed to execute"
            )
            self.results.append(error_result)
        
        return self.results
    
    def _create_workflow_input(self) -> str:
        """Create the workflow input for the ReAct agent."""
        symmetry = self.agent._get_refinement_symmetry()
        return load_prompt(
            "cryosparc/polish/task.md",
            {
                "project_uid": self.config.workflow.project_uid,
                "workspace_uid": self.config.workflow.workspace_uid,
                "symmetry": symmetry,
            },
        )
    
    def _parse_workflow_result(self, result: str) -> None:
        """Parse the workflow result to extract results for polish steps."""
        execution_log = self.agent.get_tool_execution_log()
        
        if not execution_log:
            error_result = PolishResult(
                step=PolishStep.VERIFY_INPUTS,
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
        
        # Check each step
        self._check_step_result(PolishStep.VERIFY_INPUTS, tool_entries, waits, result)
        self._check_step_result(PolishStep.INITIAL_REFINEMENT, tool_entries, waits, result)
        self._check_step_result(PolishStep.MOTION_CORRECTION, tool_entries, waits, result)
        self._check_step_result(PolishStep.FINAL_REFINEMENT, tool_entries, waits, result)
    
    def _check_step_result(
        self,
        step: PolishStep,
        tool_entries: Dict[str, List[Dict[str, Any]]],
        waits: Dict[str, Dict[str, Any]],
        reasoning: str
    ) -> None:
        """Check the result for a specific polish step."""
        # Map steps to tool names
        step_tool_map = {
            PolishStep.VERIFY_INPUTS: "verify_inputs",
            PolishStep.INITIAL_REFINEMENT: "nonuniform_refine_new",
            PolishStep.MOTION_CORRECTION: "reference_motion_correction",
            PolishStep.FINAL_REFINEMENT: "nonuniform_refine_new"
        }
        
        tool_name = step_tool_map.get(step)
        if not tool_name:
            return
        
        records = tool_entries.get(tool_name, [])
        if step in (PolishStep.INITIAL_REFINEMENT, PolishStep.FINAL_REFINEMENT):
            # Accept the former tool name in execution logs from older runs.
            records = records or tool_entries.get("homogeneous_refinement", [])
            motion_job_uids = {
                outcome.job_uid
                for entry in tool_entries.get("reference_motion_correction", [])
                if (outcome := outcome_for_record(entry, waits)).success
            }
            is_final = step == PolishStep.FINAL_REFINEMENT
            records = [
                entry for entry in records
                if (entry.get("params", {}).get("particles_job_uid") in motion_job_uids) == is_final
            ]

        if not records:
            self.results.append(
                PolishResult(
                    step=step,
                    success=False,
                    error=f"{step.value} was never executed",
                    message="No tool invocation recorded",
                    reasoning=reasoning
                )
            )
            return
        
        latest_record = records[-1]
        if step != PolishStep.VERIFY_INPUTS:
            latest_record = next(
                (entry for entry in reversed(records) if outcome_for_record(entry, waits).success),
                latest_record,
            )
        error_message = latest_record.get("error")
        result_payload = latest_record.get("result", {})

        if error_message:
            self.results.append(
                PolishResult(
                    step=step,
                    success=False,
                    job_uid=result_payload.get("job_uid") if isinstance(result_payload, dict) else None,
                    error=error_message,
                    message="Tool execution reported an error",
                    reasoning=reasoning
                )
            )
            return

        if step == PolishStep.VERIFY_INPUTS:
            # For verify_inputs, success is determined by the result itself
            if isinstance(result_payload, dict) and result_payload.get("success"):
                self.results.append(
                    PolishResult(
                        step=step,
                        success=True,
                        message="Inputs verified successfully",
                        reasoning=reasoning
                    )
                )
            else:
                self.results.append(
                    PolishResult(
                        step=step,
                        success=False,
                        error="Input verification failed",
                        message="Could not verify required inputs",
                        reasoning=reasoning
                    )
                )
            return

        # Job steps: run the shared completion ladder on the selected record.
        outcome = outcome_for_record(latest_record, waits)
        self.results.append(
            PolishResult(
                step=step,
                success=outcome.success,
                job_uid=outcome.job_uid,
                message=outcome.message,
                error=outcome.error,
                reasoning=reasoning
            )
        )
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the polish workflow execution."""
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
    
    def save_results(self, execution_time: float = 0.0, output_dir: Optional[str] = None) -> str:
        """Save polish results to a JSON file."""
        import datetime
        
        output_dir = Path(output_dir) if output_dir else Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Find final refinement job (last step)
        final_refinement_result = None
        for result in reversed(self.results):
            if result.step == PolishStep.FINAL_REFINEMENT:
                final_refinement_result = result
                break
        
        best_job_uid = final_refinement_result.job_uid if final_refinement_result and final_refinement_result.success else None
        
        # Get final resolution from the best job
        final_resolution = None
        if best_job_uid:
            try:
                fsc_info = self.agent.cryosparc_tools.get_refinement_fsc_info(
                    self.config.workflow.project_uid,
                    best_job_uid
                )
                if fsc_info.get("success"):
                    final_resolution = fsc_info.get("resolution_angstroms")
            except Exception:
                pass
        
        # Determine overall status
        all_successful = all(r.success for r in self.results)
        status = "completed" if all_successful else "failed"
        
        polish_results = {
            "stage": "polish",
            "status": status,
            "timestamp": timestamp,
            "agent_type": "cryosparc",
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
            "execution_time": execution_time,
            "best_job_uid": best_job_uid,
            "final_resolution": final_resolution,
            "workflow_summary": self.get_workflow_summary()
        }
        
        output_file = output_dir / f"polish_results_cryosparc_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(polish_results, f, indent=2)
        
        return str(output_file)


