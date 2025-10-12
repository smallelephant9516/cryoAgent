"""ReAct-based 3D reconstruction workflow orchestrator."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .reconstruction_agent import ReconstructionAgent
from ...config.config_loader import CryoAgentConfig


class ReconstructionStep(Enum):
    """Enumeration of 3D reconstruction workflow steps."""
    AB_INITIO = "ab_initio_reconstruction"
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
        
        # Ab initio parameters
        ab_initio_config = workflow_config.get("ab_initio", {})
        num_classes = ab_initio_config.get("num_classes", 1)
        initial_resolution = ab_initio_config.get("initial_resolution", 20.0)
        final_resolution = ab_initio_config.get("final_resolution", 10.0)
        max_iterations = ab_initio_config.get("max_iterations", 50)
        symmetry = ab_initio_config.get("symmetry", "C1")
        
        # Refinement parameters
        refinement_config = workflow_config.get("refinement", {})
        refinement_type = refinement_config.get("type", "none")  # "none", "homogeneous", or "heterogeneous"
        refinement_resolution = refinement_config.get("resolution", None)
        
        return {
            # Ab initio parameters
            "num_classes": num_classes,
            "initial_resolution": initial_resolution,
            "final_resolution": final_resolution,
            "max_iterations": max_iterations,
            "symmetry": symmetry,
            
            # Refinement parameters
            "refinement_type": refinement_type,
            "refinement_resolution": refinement_resolution
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
            "workflow_status": "starting"
        }
        
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
        
        workflow_description = f"""
Execute the 3D reconstruction workflow starting with ab initio reconstruction:

**Input**: Particles from job {particles_job_uid}

**Task**: Generate initial 3D model(s) from 2D particles using ab initio reconstruction

**Project**: {self.config.workflow.project_uid} | **Workspace**: {self.config.workflow.workspace_uid}

**Workflow Steps** (execute in order):

═══ PHASE 1: Initial Model Generation (Ab Initio) ═══

1. **Ab Initio Reconstruction** - Generate initial 3D model(s) de novo
   - Tool: ab_initio_reconstruction
   - Parameters: 
     * particles_job_uid={particles_job_uid}
     * num_classes={p['num_classes']}
     * initial_resolution={p['initial_resolution']}
     * final_resolution={p['final_resolution']}
     * max_iterations={p['max_iterations']}
     * symmetry={p['symmetry']}
   - This generates 3D structures without requiring a reference model
   - {"Generating " + str(p['num_classes']) + " class" + ("es" if p['num_classes'] > 1 else "") + " to handle potential heterogeneity" if p['num_classes'] > 1 else "Generating a single homogeneous 3D model"}
   - Wait for completion and record job UID
"""
        
        if run_refinement and p['refinement_type'] != 'none':
            if p['refinement_type'] == 'homogeneous':
                workflow_description += f"""
═══ PHASE 2: Homogeneous Refinement ═══

2. **Homogeneous Refinement** - Refine the single 3D structure
   - Tool: homogeneous_refinement
   - Parameters:
     * particles_job_uid={particles_job_uid}
     * volume_job_uid=[from step 1]
     * symmetry={p['symmetry']}
     * refinement_resolution={p['refinement_resolution'] if p['refinement_resolution'] else "auto"}
   - Improves resolution and quality of the structure
   - Wait for completion and record job UID
"""
            elif p['refinement_type'] == 'heterogeneous':
                workflow_description += f"""
═══ PHASE 2: Heterogeneous Refinement ═══

2. **Heterogeneous Refinement** - Refine multiple 3D structures simultaneously
   - Tool: heterogeneous_refinement
   - Parameters:
     * particles_job_uid={particles_job_uid}
     * volume_job_uids=[all volumes from step 1]
     * num_classes={p['num_classes']}
   - Simultaneously refines structures and classifies particles
   - Wait for completion and record job UID
"""
        
        workflow_description += f"""
**Critical Instructions**:
- Execute ALL steps in order - do not skip any steps
- Each step MUST complete successfully before proceeding
- Always wait_for_job after each CryoSPARC job
- Track all job UIDs - refinement depends on ab initio output
- Ab initio reconstruction can take significant time (minutes to hours)

**Expected Outcome**:
- Initial 3D model(s) from ab initio reconstruction
{"- Refined structure(s) with improved resolution" if run_refinement else ""}
- Ready for further refinement or analysis

Begin by executing step 1 (ab_initio_reconstruction){"and proceed to refinement after completion" if run_refinement else ""}.
"""
        
        return workflow_description
    
    def _parse_workflow_result(self, result: str, run_refinement: bool) -> None:
        """Parse the workflow result to extract results for reconstruction steps."""
        execution_log = self.agent.get_tool_execution_log()

        if not execution_log:
            error_result = ReconstructionResult(
                step=ReconstructionStep.AB_INITIO,
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

        # Check ab initio step
        self._check_step_result(
            ReconstructionStep.AB_INITIO,
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

