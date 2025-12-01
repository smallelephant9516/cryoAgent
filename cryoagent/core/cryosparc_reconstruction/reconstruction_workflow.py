"""ReAct-based 3D reconstruction workflow orchestrator."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .reconstruction_agent import ReconstructionAgent
from ...config.config_loader import CryoAgentConfig


class ReconstructionStep(Enum):
    """Enumeration of 3D reconstruction workflow steps."""
    AB_INITIO = "ab_initio_reconstruction"
    HOMOGENEOUS_RECONSTRUCTION = "homogeneous_reconstruction"
    HOMOGENEOUS_REFINEMENT = "homogeneous_refinement"
    HETEROGENEOUS_REFINEMENT = "heterogeneous_refinement"


@dataclass
class ReconstructionResult:
    """Result of a reconstruction workflow execution."""
    step: ReconstructionStep
    success: bool
    job_uid: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None


class ReconstructionWorkflow:
    """ReAct-based orchestrator for 3D reconstruction workflows."""
    
    def __init__(self, agent: ReconstructionAgent, config: CryoAgentConfig, stage_config_path: Optional[str] = None):
        """
        Initialize the 3D reconstruction workflow.
        
        Args:
            agent: Reconstruction agent instance
            config: Complete configuration object
            stage_config_path: Path to stage-specific configuration file
        """
        self.agent = agent
        self.config = config
        self.results: List[ReconstructionResult] = []
        self.current_job_uids: Dict[ReconstructionStep, str] = {}
        self.workflow_state: Dict[str, Any] = {}
        self.dynamic_defaults: Dict[str, Any] = {}
        
        # Load stage-specific configuration
        self.stage_config = self._load_stage_config(stage_config_path)
        self.workflow_params = self._parse_workflow_params()
        if hasattr(self.agent, "update_workflow_defaults"):
            try:
                self.agent.update_workflow_defaults(self.workflow_params)
            except Exception:
                # Non-fatal; agent may decline to store defaults
                pass
    
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
        
        # Reconstruction method
        reconstruction_method_config = workflow_config.get("reconstruction_method", {})
        reconstruction_method = reconstruction_method_config.get("type", "ab_initio")  # "ab_initio" or "homogeneous"
        
        # Ab initio parameters
        ab_initio_config = workflow_config.get("ab_initio", {})
        num_classes = ab_initio_config.get("num_classes", 1)
        ab_initial_resolution = ab_initio_config.get("initial_resolution", 20.0)
        ab_final_resolution = ab_initio_config.get("final_resolution", 10.0)
        max_iterations = ab_initio_config.get("max_iterations", 50)
        ab_symmetry = ab_initio_config.get("symmetry", "C1")
        
        # Homogeneous reconstruction parameters
        homo_recon_config = workflow_config.get("homogeneous_reconstruction", {})
        homo_initial_resolution = homo_recon_config.get("initial_resolution", 20.0)
        homo_final_resolution = homo_recon_config.get("final_resolution", 8.0)
        homo_symmetry = homo_recon_config.get("symmetry", "C1")
        
        # Refinement parameters
        refinement_config = workflow_config.get("refinement", {})
        refinement_type = refinement_config.get("type", "none")  # "none", "homogeneous", or "heterogeneous"
        refinement_resolution = refinement_config.get("resolution", None)
        refinement_symmetry = refinement_config.get("symmetry", "C1")
        
        return {
            # Reconstruction method selection
            "reconstruction_method": reconstruction_method,
            
            # Ab initio parameters
            "num_classes": num_classes,
            "ab_initial_resolution": ab_initial_resolution,
            "ab_final_resolution": ab_final_resolution,
            "max_iterations": max_iterations,
            "ab_symmetry": ab_symmetry,
            
            # Homogeneous reconstruction parameters
            "homo_initial_resolution": homo_initial_resolution,
            "homo_final_resolution": homo_final_resolution,
            "homo_symmetry": homo_symmetry,
            
            # Refinement parameters
            "refinement_type": refinement_type,
            "refinement_resolution": refinement_resolution,
            "refinement_symmetry": refinement_symmetry
        }
    
    def run(
        self,
        particles_job_uid: str,
        conversation_id: Optional[str] = None,
        run_refinement: bool = False
    ) -> List[ReconstructionResult]:
        """
        Run the 3D reconstruction workflow using ReAct approach.
        
        Args:
            particles_job_uid: Job UID from particle selection or 2D classification
            conversation_id: Optional conversation identifier for memory control
            run_refinement: Whether to run refinement after ab initio (default: False)
            
        Returns:
            List of reconstruction results for all steps
        """
        self.results = []
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "active_jobs": {},
            "workflow_status": "starting",
            "input_particles_job_uid": particles_job_uid
        }
        self.dynamic_defaults = {
            "particles_job_uid": particles_job_uid,
            "refinement_resolution": self.workflow_params.get("refinement_resolution"),
            "refinement_symmetry": self.workflow_params.get("refinement_symmetry"),
        }
        self._refresh_agent_defaults()
        
        workflow_input = self._create_workflow_input(particles_job_uid, run_refinement)
        
        try:
            result = self.agent.run_react_workflow(workflow_input, conversation_id)
            self._parse_workflow_result(result, run_refinement)
            
        except Exception as e:
            error_result = ReconstructionResult(
                step=ReconstructionStep.AB_INITIO,
                success=False,
                error=f"3D reconstruction workflow execution failed: {str(e)}",
                message="ReAct workflow failed to execute"
            )
            self.results.append(error_result)
        
        return self.results
    
    def _create_workflow_input(self, particles_job_uid: str, run_refinement: bool) -> str:
        """Create the workflow input for the ReAct agent using config parameters."""
        p = self.workflow_params  # shorthand
        
        # Determine which reconstruction method to use
        recon_method = p.get('reconstruction_method', 'ab_initio')
        
        if recon_method == 'homogeneous':
            method_name = "Homogeneous Reconstruction"
            tool_name = "homogeneous_reconstruction"
            initial_res = p['homo_initial_resolution']
            final_res = p['homo_final_resolution']
            symmetry = p['homo_symmetry']
            method_description = "This uses an optimized algorithm for single structure reconstruction"
            params_str = f"""     * particles_job_uid={particles_job_uid}
     * initial_resolution={initial_res}
     * final_resolution={final_res}
     * symmetry={symmetry}"""
        else:  # ab_initio
            method_name = "Ab Initio Reconstruction"
            tool_name = "ab_initio_reconstruction"
            initial_res = p['ab_initial_resolution']
            final_res = p['ab_final_resolution']
            symmetry = p['ab_symmetry']
            method_description = "This generates 3D structures without requiring a reference model"
            num_classes_note = f"Generating {p['num_classes']} class" + ("es" if p['num_classes'] > 1 else "") + " to handle potential heterogeneity" if p['num_classes'] > 1 else "Generating a single homogeneous 3D model"
            params_str = f"""     * particles_job_uid={particles_job_uid}
     * num_classes={p['num_classes']}
     * initial_resolution={initial_res}
     * final_resolution={final_res}
     * max_iterations={p['max_iterations']}
     * symmetry={symmetry}"""
        
        workflow_description = f"""
Execute the 3D reconstruction workflow starting with {method_name.lower()}:

**Input**: Particles from job {particles_job_uid}

**Task**: Generate initial 3D model(s) from 2D particles using {method_name.lower()}

**Project**: {self.config.workflow.project_uid} | **Workspace**: {self.config.workflow.workspace_uid}

**Workflow Steps** (execute in order):

═══ PHASE 1: Initial Model Generation ({method_name}) ═══

1. **{method_name}** - Generate initial 3D model(s)
   - Tool: {tool_name}
   - Parameters: 
{params_str}
   - {method_description}
   - {"" if recon_method == 'homogeneous' else num_classes_note}
   - Wait for completion and record job UID
"""
        
        if run_refinement and p['refinement_type'] != 'none':
            if p['refinement_type'] == 'homogeneous':
                # Use symmetry from the refinement configuration
                symmetry = p['refinement_symmetry']
                workflow_description += f"""
═══ PHASE 2: Homogeneous Refinement ═══

2. **Homogeneous Refinement** - Refine the single 3D structure
   - Tool: homogeneous_refinement
   - Parameters:
     * particles_job_uid=[ORIGINAL input - same particles_job_uid used in step 1, e.g., {particles_job_uid}]
     * volume_job_uid=[from step 1 - the ab initio job UID that produced the volume]
     * symmetry={symmetry}
     * refinement_resolution={p['refinement_resolution'] if p['refinement_resolution'] else "auto"}
   - CRITICAL: particles_job_uid and volume_job_uid must be DIFFERENT
     - particles_job_uid: Use the ORIGINAL input particles (same as used in step 1)
     - volume_job_uid: Use the ab initio job UID from step 1
   - Improves resolution and quality of the structure
   - Wait for completion and record job UID
"""
            elif p['refinement_type'] == 'heterogeneous':
                workflow_description += f"""

**Critical Instructions**:
- Execute ALL steps in order - do not skip any steps
- Each step MUST complete successfully before proceeding
- Always wait_for_job after each CryoSPARC job
- Track all job UIDs - refinement depends on initial reconstruction output
- Reconstruction can take significant time (minutes to hours)

**Expected Outcome**:
- Initial 3D model(s) from {method_name.lower()}
{"- Refined structure(s) with improved resolution" if run_refinement else ""}
- Ready for further refinement or analysis

Begin by executing step 1 ({tool_name}){"and proceed to refinement after completion" if run_refinement else ""}.
"""
        
        return workflow_description
    
    def _parse_workflow_result(self, result: str, run_refinement: bool) -> None:
        """Parse the workflow result to extract results for reconstruction steps."""
        execution_log = self.agent.get_tool_execution_log()

        # Determine which reconstruction method was used
        recon_method = self.workflow_params.get('reconstruction_method', 'ab_initio')
        initial_step = ReconstructionStep.HOMOGENEOUS_RECONSTRUCTION if recon_method == 'homogeneous' else ReconstructionStep.AB_INITIO

        if not execution_log:
            error_result = ReconstructionResult(
                step=initial_step,
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

        # Check initial reconstruction step (ab initio or homogeneous reconstruction)
        self._check_step_result(
            initial_step,
            tool_entries,
            waits,
            result
        )
        
        # Check refinement steps if requested
        if run_refinement:
            refinement_type = self.workflow_params.get('refinement_type', 'none')
            
            if refinement_type == 'homogeneous':
                self._check_step_result(
                    ReconstructionStep.HOMOGENEOUS_REFINEMENT,
                    tool_entries,
                    waits,
                    result
                )
            elif refinement_type == 'heterogeneous':
                self._check_step_result(
                    ReconstructionStep.HETEROGENEOUS_REFINEMENT,
                    tool_entries,
                    waits,
                    result
                )
    
    def _check_step_result(
        self,
        step: ReconstructionStep,
        tool_entries: Dict[str, List[Dict[str, Any]]],
        waits: Dict[str, Dict[str, Any]],
        reasoning: str
    ) -> None:
        """Check the result for a specific reconstruction step."""
        records = tool_entries.get(step.value, [])
        
        if not records:
            self.results.append(
                ReconstructionResult(
                    step=step,
                    success=False,
                    error=f"{step.value} was never executed",
                    message="No tool invocation recorded",
                    reasoning=reasoning
                )
            )
            return

        latest_record = records[-1]
        error_message = latest_record.get("error")
        result_payload = latest_record.get("result", {})
        job_uid = result_payload.get("job_uid") if isinstance(result_payload, dict) else None

        if error_message:
            self.results.append(
                ReconstructionResult(
                    step=step,
                    success=False,
                    job_uid=job_uid,
                    error=error_message,
                    message="Tool execution reported an error",
                    reasoning=reasoning
                )
            )
            return

        if not job_uid:
            self.results.append(
                ReconstructionResult(
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
                ReconstructionResult(
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
            ReconstructionResult(
                step=step,
                success=success,
                job_uid=job_uid,
                message=message,
                error=error,
                reasoning=reasoning
            )
        )
        
        if success and job_uid:
            self.current_job_uids[step] = job_uid
            self._update_dynamic_defaults(step, job_uid)
            self._refresh_agent_defaults()
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get a summary of the 3D reconstruction workflow execution."""
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
        self.dynamic_defaults = {}
        self._refresh_agent_defaults()

    def _update_dynamic_defaults(self, step: ReconstructionStep, job_uid: Optional[str]) -> None:
        if not job_uid:
            return
        if step == ReconstructionStep.AB_INITIO:
            self.dynamic_defaults["ab_init_job_uid"] = job_uid
            self.dynamic_defaults["last_volume_job_uid"] = job_uid
        elif step == ReconstructionStep.HOMOGENEOUS_RECONSTRUCTION:
            self.dynamic_defaults["homogeneous_reconstruction_job_uid"] = job_uid
            self.dynamic_defaults["last_volume_job_uid"] = job_uid
        elif step == ReconstructionStep.HOMOGENEOUS_REFINEMENT:
            self.dynamic_defaults["homogeneous_refinement_job_uid"] = job_uid
            self.dynamic_defaults["last_volume_job_uid"] = job_uid
        elif step == ReconstructionStep.HETEROGENEOUS_REFINEMENT:
            self.dynamic_defaults["heterogeneous_refinement_job_uid"] = job_uid
            self.dynamic_defaults["last_volume_job_uid"] = job_uid

    def _refresh_agent_defaults(self) -> None:
        if hasattr(self.agent, "update_workflow_defaults"):
            combined = dict(self.workflow_params)
            combined.update(self.dynamic_defaults)
            try:
                self.agent.update_workflow_defaults(combined)
            except Exception:
                pass

