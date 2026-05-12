"""ReAct-based polish workflow orchestrator."""

import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .polish_agent import PolishAgent
from ...config.config_loader import CryoAgentConfig
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
        # Get symmetry from reconstruction config via agent
        symmetry = self.agent._get_refinement_symmetry()
        
        return f"""
Execute the polish refinement workflow to achieve the best possible resolution:

**Project**: {self.config.workflow.project_uid} | **Workspace**: {self.config.workflow.workspace_uid}
**Symmetry**: {symmetry} (from reconstruction_config.json)

**Workflow Steps** (execute in order):

═══ STEP 1: Verify Inputs ═══

1. **Verify Inputs** - Check that optimization and preprocessing are complete
   - Tool: verify_inputs
   - No parameters required
   - This will read optimization_results_cryosparc_*.json to get best_job_uid
   - This will read preprocessing_results_cryosparc_*.json to get final_micrographs_job_uid
   - Wait for verification to complete

═══ STEP 2: Initial Homogeneous Refinement with CTF ═══

2. **Initial Homogeneous Refinement** - Refine with CTF refinement enabled
   - Tool: homogeneous_refinement
   - Parameters:
     * particles_job_uid=[best_job_uid from Step 1]
     * volume_job_uid=[best_job_uid from Step 1]
     * symmetry={symmetry} (from reconstruction_config.json)
     * refine_defocus_refine=true (enable local CTF refinement)
     * refine_ctf_global_refine=true (enable global CTF refinement)
   - Wait for completion and record job UID (J-A)

═══ STEP 3: Reference Motion Correction ═══

3. **Reference Motion Correction** - Correct particle motion using reference volume
   - Tool: reference_motion_correction
   - Parameters:
     * micrographs_job_uid=[final_micrographs_job_uid from Step 1]
     * particles_job_uid=[J-A from Step 2]
     * volume_job_uid=[J-A from Step 2]
   - Wait for completion and record job UID (J-B)
   - Output particles group name is "particles_0"

═══ STEP 4: Final Homogeneous Refinement with CTF ═══

4. **Final Homogeneous Refinement** - Final refinement with CTF refinement enabled
   - Tool: homogeneous_refinement
   - Parameters:
     * particles_job_uid=[J-B from Step 3]
     * particles_group_name="particles_0" (use particles_0 group from motion correction)
     * volume_job_uid=[J-A from Step 2] (use volume from initial refinement, NOT from motion correction)
     * symmetry={symmetry} (from reconstruction_config.json)
     * refine_defocus_refine=true (enable local CTF refinement)
     * refine_ctf_global_refine=true (enable global CTF refinement)
   - Wait for completion and record job UID (J-C)
   - J-C is the final best job UID

**Critical Instructions**:
- Execute ALL steps in order - do not skip any steps
- Each step MUST complete successfully before proceeding
- Always wait_for_job after each CryoSPARC job
- Track all job UIDs - each step depends on the previous step's output
- Polish refinement jobs can take significant time (minutes to hours)
- Use symmetry={symmetry} from reconstruction_config.json for all refinement steps

**Expected Outcome**:
- Final refined structure (J-C) with best resolution
- All CTF refinement enabled for optimal results
- Motion-corrected particles from reference-based correction

Begin by executing step 1 (verify_inputs) and proceed through all steps sequentially.
"""
    
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
        
        for entry in execution_log:
            tool_name = entry.get("tool")
            tool_entries.setdefault(tool_name, []).append(entry)
            if tool_name == "wait_for_job" and entry.get("result"):
                job_uid = entry.get("params", {}).get("job_uid")
                if job_uid:
                    waits[job_uid] = entry["result"]
        
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
            PolishStep.INITIAL_REFINEMENT: "homogeneous_refinement",
            PolishStep.MOTION_CORRECTION: "reference_motion_correction",
            PolishStep.FINAL_REFINEMENT: "homogeneous_refinement"
        }
        
        tool_name = step_tool_map.get(step)
        if not tool_name:
            return
        
        records = tool_entries.get(tool_name, [])
        
        # For FINAL_REFINEMENT, we want the LAST homogeneous_refinement call
        # For INITIAL_REFINEMENT, we want the FIRST homogeneous_refinement call
        if step == PolishStep.FINAL_REFINEMENT and len(records) >= 2:
            latest_record = records[-1]
        elif step == PolishStep.INITIAL_REFINEMENT and len(records) >= 1:
            latest_record = records[0]
        elif records:
            latest_record = records[-1]
        else:
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
        
        error_message = latest_record.get("error")
        result_payload = latest_record.get("result", {})
        job_uid = result_payload.get("job_uid") if isinstance(result_payload, dict) else None
        
        if error_message:
            self.results.append(
                PolishResult(
                    step=step,
                    success=False,
                    job_uid=job_uid,
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
        
        if not job_uid:
            self.results.append(
                PolishResult(
                    step=step,
                    success=False,
                    error="Tool did not return a job UID",
                    message="Unable to confirm CryoSPARC job submission",
                    reasoning=reasoning
                )
            )
            return
        
        wait_info = waits.get(job_uid)
        if not wait_info:
            self.results.append(
                PolishResult(
                    step=step,
                    success=False,
                    job_uid=job_uid,
                    error="Job completion was not confirmed",
                    message="Missing wait_for_job invocation",
                    reasoning=reasoning
                )
            )
            return
        
        status = wait_info.get("status")
        success = status == "completed"
        step_name = step.value.replace('_', ' ').title()
        message = f"CryoSPARC {step_name} job {job_uid} completed successfully" if success else f"CryoSPARC {step_name} job {job_uid} finished with status '{status}'"
        error = None if success else f"Job status: {status}"
        
        self.results.append(
            PolishResult(
                step=step,
                success=success,
                job_uid=job_uid,
                message=message,
                error=error,
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


