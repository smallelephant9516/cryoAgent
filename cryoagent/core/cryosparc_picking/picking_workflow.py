"""ReAct-based particle picking workflow orchestrator."""

import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .picking_agent import PickingAgent
from ...config.config_loader import CryoAgentConfig


class PickingStep(Enum):
    """Enumeration of particle picking workflow steps."""
    BLOB_PICKER = "blob_picker"
    EXTRACT_PARTICLES = "extract_particles"
    CLASS_2D = "class_2d"
    SELECT_2D_CLASSES = "select_2d_classes"
    TEMPLATE_PICKER = "template_picker"
    EXTRACT_PARTICLES_2 = "extract_particles_2"
    CLASS_2D_2 = "class_2d_2"
    SELECT_FINAL_CLASSES = "select_final_classes"
    FINAL_EXTRACTION = "final_extraction"


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
        
        # Select 2D classes parameters (used for BOTH selections)
        select_config = workflow_config.get("select_2d_classes", {})
        top_n_classes = select_config.get("top_n_classes", 5)
        selection_mode = select_config.get("selection_mode", "top_n")
        cryosift_threshold = select_config.get("cryosift_threshold")
        cryosift_env = select_config.get("cryosift_env")
        cryosift_weights = select_config.get("cryosift_weights_path")
        cryosift_output_dir = select_config.get("cryosift_output_dir")
        cryosift_output_subdir = select_config.get("cryosift_output_subdir")
        cryosift_python = select_config.get("cryosift_python_executable")
        cryosift_fallback = select_config.get("cryosift_fallback_strategy")
        
        # Template picker parameters
        template_config = workflow_config.get("template_picker", {})
        lowpass_resolution = template_config.get("lowpass_resolution", 20.0)
        angle_search_range = template_config.get("angle_search_range", 180)
        
        return {
            # Blob picker parameters
            "particle_diameter": particle_diameter,
            "diameter_max": diameter_max,
            
            # Particle extraction parameters (used for BOTH extractions)
            "box_size_pix": box_size_pix,
            "extract_downscale_factor": extract_downscale_factor,
            "bg_radius": bg_radius,
            "invert_contrast": invert_contrast,
            
            # 2D classification parameters (used for BOTH classifications)
            "num_classes": num_classes,
            "max_iterations": max_iterations,
            "initial_resolution": initial_resolution,
            "final_resolution": final_resolution,
            "batch_size_per_class": batch_size_per_class,
            "force_max_res": force_max_res,
            
            # Selection parameters (used for BOTH selections)
            "top_n_classes": top_n_classes,
            "selection_mode": selection_mode,
            "cryosift_threshold": cryosift_threshold,
            "cryosift_env": cryosift_env,
            "cryosift_weights_path": cryosift_weights,
            "cryosift_output_dir": cryosift_output_dir,
            "cryosift_output_subdir": cryosift_output_subdir,
            "cryosift_python_executable": cryosift_python,
            "cryosift_fallback_strategy": cryosift_fallback,
            
            # Template picker parameters
            "lowpass_resolution": lowpass_resolution,
            "angle_search_range": angle_search_range
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
            execution_log = self.agent.get_tool_execution_log()

            if not self._log_contains_required_calls(execution_log):
                # Fall back to deterministic execution when the LLM cannot call tools (e.g., DeepSeek)
                self.agent.clear_tool_execution_log()
                self._run_direct_workflow(micrographs_job_uid, result)
            else:
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
        p = self.workflow_params  # shorthand
        
        diameter_range = f"{p['particle_diameter']}-{p['diameter_max']}" if p['diameter_max'] else f"{p['particle_diameter']}-{p['particle_diameter'] * 2.0}"

        selection_mode = (p.get("selection_mode") or "top_n").lower()

        select_params = ["class_2d_job_uid=[from step 3]"]
        final_select_params = ["class_2d_job_uid=[from step 7]"]

        if selection_mode == "cryosift":
            select_step_title = "**Select CryoSift Classes** - Evaluate CryoSift scores to choose templates"
            final_select_step_title = "**Select Final CryoSift Classes** - Re-evaluate CryoSift scores after refinement"
            select_params.append("selection_mode=cryosift")
            final_select_params.append("selection_mode=cryosift")
            if p.get("cryosift_threshold") is not None:
                threshold = p['cryosift_threshold']
                select_params.append(f"cryosift_threshold={threshold}")
                final_select_params.append(f"cryosift_threshold={threshold}")
            if p.get("cryosift_env"):
                env = p['cryosift_env']
                select_params.append(f"cryosift_env={env}")
                final_select_params.append(f"cryosift_env={env}")
            if p.get("cryosift_weights_path"):
                select_params.append("cryosift_weights_path=[configured]")
                final_select_params.append("cryosift_weights_path=[configured]")
            if p.get("cryosift_output_dir"):
                select_params.append("cryosift_output_dir=[configured]")
                final_select_params.append("cryosift_output_dir=[configured]")
            elif p.get("cryosift_output_subdir"):
                subdir = p['cryosift_output_subdir']
                select_params.append(f"cryosift_output_subdir={subdir}")
                final_select_params.append(f"cryosift_output_subdir={subdir}")
        else:
            select_step_title = f"**Select Top Classes** - Select best {p['top_n_classes']} classes as templates"
            final_select_step_title = f"**Select Final Classes** - Select top {p['top_n_classes']} classes from round 2"
            select_params.append(f"top_n_classes={p['top_n_classes']}")
            final_select_params.append(f"top_n_classes={p['top_n_classes']}")

        return f"""
Execute the complete ADVANCED particle picking workflow with 9 steps (template-based refinement):

**Input**: Micrographs from job {micrographs_job_uid}

**Task**: Use blob picker for initial detection, then refine with template-based picking for high-quality particles

**Project**: {self.config.workflow.project_uid} | **Workspace**: {self.config.workflow.workspace_uid}

**Workflow Steps** (execute in order):

═══ PHASE 1: Initial Blob-Based Picking ═══

1. **Blob Picker** - Initial particle detection
   - Tool: blob_picker
   - Parameters: micrographs_job_uid={micrographs_job_uid}, particle_diameter={p['particle_diameter']}, diameter_max={p['diameter_max'] or p['particle_diameter'] * 2.0}
   - Wait for completion and record job UID

2. **Extract Particles (Round 1)** - Extract blob-picked particles
   - Tool: extract_particles
   - Parameters: particles_job_uid=[from step 1], micrographs_job_uid={micrographs_job_uid}, box_size_pix={p['box_size_pix']}
   - Wait for completion and record job UID

3. **2D Classification (Round 1)** - Classify initial particles
   - Tool: class_2d
   - Parameters: particles_job_uid=[from step 2], num_classes={p['num_classes']}
   - Wait for completion and record job UID

═══ PHASE 2: Template-Based Refinement ═══

4. {select_step_title}
   - Tool: select_2d_classes
   - Parameters: {', '.join(select_params)}
   - Wait for completion and record job UID

5. **Template Picker** - Re-pick particles using class averages as templates
   - Tool: template_picker
   - Parameters: micrographs_job_uid={micrographs_job_uid}, template_job_uid=[from step 4], lowpass_resolution={p['lowpass_resolution']}
   - More accurate than blob picker - uses actual particle images
   - Wait for completion and record job UID

6. **Extract Particles (Round 2)** - Extract template-picked particles
   - Tool: extract_particles
   - Parameters: particles_job_uid=[from step 5], micrographs_job_uid={micrographs_job_uid}, box_size_pix={p['box_size_pix']} (same as round 1)
   - Wait for completion and record job UID

7. **2D Classification (Round 2)** - Classify refined particles
   - Tool: class_2d
   - Parameters: particles_job_uid=[from step 6], num_classes={p['num_classes']} (same as round 1)
   - Wait for completion and record job UID

═══ PHASE 3: Final Selection ═══

8. {final_select_step_title}
   - Tool: select_2d_classes
   - Parameters: {', '.join(final_select_params)}
   - These are the highest quality particles
   - Wait for completion and record job UID

9. **Final Particles** - Particles ready for 3D reconstruction
   - No additional tool needed - step 8 output contains final selected particles
   - Report the select job UID from step 8 as final output

**Critical Instructions**:
- Execute ALL 9 steps in order - do not skip any steps
- Each step MUST complete successfully before proceeding
- Always wait_for_job after each CryoSPARC job
- Track all job UIDs - each step depends on previous outputs
- Template picking (step 5) requires BOTH micrographs AND templates
- Both extractions (steps 2 & 6) require BOTH particles AND micrographs

**Expected Outcome**:
- High-quality particles from 2 rounds of picking and classification
- Template-based refinement improves particle quality significantly
- Final selected particles ready for 3D reconstruction

Begin by executing step 1 (blob_picker) and proceed sequentially through all 9 steps.
"""
    
    def _parse_workflow_result(self, result: str) -> None:
        """Parse the workflow result to extract results for all 9 picking steps."""
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

        # Map workflow steps to tool names and their index in the tool call sequence
        # Format: (step_enum, tool_name, tool_call_index)
        step_mapping = [
            (PickingStep.BLOB_PICKER, "blob_picker", 0),
            (PickingStep.EXTRACT_PARTICLES, "extract_particles", 0),
            (PickingStep.CLASS_2D, "class_2d", 0),
            (PickingStep.SELECT_2D_CLASSES, "select_2d_classes", 0),
            (PickingStep.TEMPLATE_PICKER, "template_picker", 0),
            (PickingStep.EXTRACT_PARTICLES_2, "extract_particles", 1),
            (PickingStep.CLASS_2D_2, "class_2d", 1),
            (PickingStep.SELECT_FINAL_CLASSES, "select_2d_classes", 1),
        ]

        # Check all steps in order
        for step_enum, tool_name, tool_index in step_mapping:
            records = tool_entries.get(tool_name, [])
            
            # Check if we have enough invocations for this tool
            if len(records) <= tool_index:
                self.results.append(
                    PickingResult(
                        step=step_enum,
                        success=False,
                        error=f"{tool_name} invocation #{tool_index+1} was never executed",
                        message="No tool invocation recorded",
                        reasoning=result
                    )
                )
                continue

            # Get the specific invocation for this step
            record = records[tool_index]
            error_message = record.get("error")
            result_payload = record.get("result", {})
            job_uid = result_payload.get("job_uid") if isinstance(result_payload, dict) else None

            if error_message:
                self.results.append(
                    PickingResult(
                        step=step_enum,
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
                        step=step_enum,
                        success=False,
                        error="Tool did not return a job UID",
                        message="Unable to confirm CryoSPARC job submission",
                        reasoning=result
                    )
                )
                continue

            wait_info = waits.get(job_uid)
            
            # select_2d_classes completes synchronously (waits internally), so it doesn't require wait_for_job
            is_select_2d_classes = step_enum in (PickingStep.SELECT_2D_CLASSES, PickingStep.SELECT_FINAL_CLASSES)
            
            if not wait_info:
                if is_select_2d_classes:
                    # For select_2d_classes, if we have a job_uid and the tool returned successfully,
                    # assume it completed (the tool waits internally)
                    # Check if the result payload indicates success
                    if isinstance(result_payload, dict) and "job_uid" in result_payload:
                        # The tool completed synchronously, so mark as successful
                        step_name = step_enum.value.replace('_', ' ').title()
                        self.results.append(
                            PickingResult(
                                step=step_enum,
                                success=True,
                                job_uid=job_uid,
                                message=f"CryoSPARC {step_name} job {job_uid} completed successfully (synchronous operation)",
                                error=None,
                                reasoning=result
                            )
                        )
                        continue
                
                # For other steps, require wait_for_job
                self.results.append(
                    PickingResult(
                        step=step_enum,
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
            step_name = step_enum.value.replace('_', ' ').title()
            message = f"CryoSPARC {step_name} job {job_uid} completed successfully" if success else f"CryoSPARC {step_name} job {job_uid} finished with status '{status}'"
            error = None if success else f"Job status: {status}"

            self.results.append(
                PickingResult(
                    step=step_enum,
                    success=success,
                    job_uid=job_uid,
                    message=message,
                    error=error,
                    reasoning=result
                )
            )
        
        # Add final extraction step (no job, just uses output from select_final_classes)
        if self.results and self.results[-1].success:
            self.results.append(
                PickingResult(
                    step=PickingStep.FINAL_EXTRACTION,
                    success=True,
                    job_uid=self.results[-1].job_uid,  # Same as select_final_classes
                    message="Final particles ready for 3D reconstruction (from select_final_classes output)",
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

    # ------------------------------------------------------------------
    # Fallback execution helpers
    # ------------------------------------------------------------------

    def _log_contains_required_calls(self, execution_log: List[Dict[str, Any]]) -> bool:
        """Check whether the execution log contains any core CryoSPARC tool calls."""
        if not execution_log:
            return False

        required_tools = {
            "blob_picker",
            "extract_particles",
            "class_2d",
            "template_picker",
            "select_2d_classes"
        }

        return any(entry.get("tool") in required_tools for entry in execution_log)

    def _run_direct_workflow(self, micrographs_job_uid: str, agent_reasoning: Optional[str]) -> None:
        """Execute the particle picking workflow without relying on tool-calling LLM support."""
        fallback_note = "Deterministic fallback execution triggered because no CryoSPARC tool calls were produced by the LLM."
        combined_reasoning = (agent_reasoning.strip() + "\n\n" if agent_reasoning and agent_reasoning.strip() else "") + fallback_note

        self.workflow_state["fallback_mode"] = True
        self.workflow_state["fallback_details"] = {
            "reason": fallback_note,
            "trigger": "missing_tool_calls"
        }
        self.workflow_state["workflow_status"] = "fallback_running"

        job_timeout = int(getattr(self.config.job_management, "default_timeout", 3600))
        check_interval = int(getattr(self.config.job_management, "status_check_interval", 30))

        p = self.workflow_params

        # Step 1: Blob picker
        blob_success, blob_job = self._execute_direct_step(
            step=PickingStep.BLOB_PICKER,
            tool_callable=self.agent._blob_picker_tool,
            params=self._build_blob_picker_params(micrographs_job_uid, p, job_timeout, check_interval),
            wait_timeout=job_timeout,
            wait_interval=check_interval,
            reasoning=combined_reasoning
        )
        if not blob_success:
            self.workflow_state["workflow_status"] = "fallback_failed"
            return

        # Step 2: Particle extraction (round 1)
        extract_success, extract_job = self._execute_direct_step(
            step=PickingStep.EXTRACT_PARTICLES,
            tool_callable=self.agent._extract_particles_tool,
            params=self._build_extract_params(blob_job, micrographs_job_uid, p, job_timeout, check_interval),
            wait_timeout=job_timeout,
            wait_interval=check_interval,
            reasoning=combined_reasoning
        )
        if not extract_success:
            self.workflow_state["workflow_status"] = "fallback_failed"
            return

        # Step 3: 2D classification (round 1)
        class_success, class_job = self._execute_direct_step(
            step=PickingStep.CLASS_2D,
            tool_callable=self.agent._class_2d_tool,
            params=self._build_class2d_params(extract_job, p, job_timeout * 2, check_interval),
            wait_timeout=job_timeout * 2,
            wait_interval=check_interval,
            reasoning=combined_reasoning
        )
        if not class_success:
            self.workflow_state["workflow_status"] = "fallback_failed"
            return

        # Step 4: Select top classes
        select_success, select_job = self._execute_direct_step(
            step=PickingStep.SELECT_2D_CLASSES,
            tool_callable=self.agent._select_2d_classes_tool,
            params=self._build_select_params(class_job, p, job_timeout, check_interval),
            wait_timeout=job_timeout,
            wait_interval=check_interval,
            reasoning=combined_reasoning
        )
        if not select_success:
            self.workflow_state["workflow_status"] = "fallback_failed"
            return

        # Step 5: Template picker
        template_success, template_job = self._execute_direct_step(
            step=PickingStep.TEMPLATE_PICKER,
            tool_callable=self.agent._template_picker_tool,
            params=self._build_template_picker_params(micrographs_job_uid, select_job, p, job_timeout, check_interval),
            wait_timeout=job_timeout,
            wait_interval=check_interval,
            reasoning=combined_reasoning
        )
        if not template_success:
            self.workflow_state["workflow_status"] = "fallback_failed"
            return

        # Step 6: Particle extraction (round 2)
        extract2_success, extract2_job = self._execute_direct_step(
            step=PickingStep.EXTRACT_PARTICLES_2,
            tool_callable=self.agent._extract_particles_tool,
            params=self._build_extract_params(template_job, micrographs_job_uid, p, job_timeout, check_interval),
            wait_timeout=job_timeout,
            wait_interval=check_interval,
            reasoning=combined_reasoning
        )
        if not extract2_success:
            self.workflow_state["workflow_status"] = "fallback_failed"
            return

        # Step 7: 2D classification (round 2)
        class2_success, class2_job = self._execute_direct_step(
            step=PickingStep.CLASS_2D_2,
            tool_callable=self.agent._class_2d_tool,
            params=self._build_class2d_params(extract2_job, p, job_timeout * 2, check_interval),
            wait_timeout=job_timeout * 2,
            wait_interval=check_interval,
            reasoning=combined_reasoning
        )
        if not class2_success:
            self.workflow_state["workflow_status"] = "fallback_failed"
            return

        # Step 8: Final class selection
        final_select_success, final_select_job = self._execute_direct_step(
            step=PickingStep.SELECT_FINAL_CLASSES,
            tool_callable=self.agent._select_2d_classes_tool,
            params=self._build_select_params(class2_job, p, job_timeout, check_interval),
            wait_timeout=job_timeout,
            wait_interval=check_interval,
            reasoning=combined_reasoning
        )
        if not final_select_success:
            self.workflow_state["workflow_status"] = "fallback_failed"
            return

        # Step 9: Final extraction summary
        self.results.append(
            PickingResult(
                step=PickingStep.FINAL_EXTRACTION,
                success=True,
                job_uid=final_select_job,
                message="Final particles ready for 3D reconstruction (fallback mode)",
                reasoning=combined_reasoning
            )
        )
        self.workflow_state["completed_steps"].append(PickingStep.FINAL_EXTRACTION.value)
        self.workflow_state["workflow_status"] = "fallback_completed"

    def _execute_direct_step(
        self,
        *,
        step: PickingStep,
        tool_callable,
        params: Dict[str, Any],
        wait_timeout: int,
        wait_interval: int,
        reasoning: str
    ) -> Tuple[bool, Optional[str]]:
        """Execute a single CryoSPARC tool via the agent wrappers and wait for completion."""
        self.workflow_state["current_step"] = step.value

        job_uid: Optional[str] = None

        try:
            job_uid, _ = self._invoke_agent_tool(tool_callable, params)
        except Exception as exc:
            self.results.append(
                PickingResult(
                    step=step,
                    success=False,
                    message="Tool execution failed in fallback mode",
                    error=str(exc),
                    reasoning=reasoning
                )
            )
            self.workflow_state["failed_steps"].append(step.value)
            return False, None

        try:
            wait_result = self._invoke_wait_for_job(job_uid, wait_timeout, wait_interval)
        except Exception as exc:
            self.results.append(
                PickingResult(
                    step=step,
                    success=False,
                    job_uid=job_uid,
                    message="Failed while waiting for CryoSPARC job completion (fallback mode)",
                    error=str(exc),
                    reasoning=reasoning
                )
            )
            self.workflow_state["failed_steps"].append(step.value)
            return False, job_uid

        status = wait_result.get("status")
        success = status == "completed"
        step_name = step.value.replace('_', ' ').title()

        if success:
            message = f"CryoSPARC {step_name} job {job_uid} completed successfully (fallback mode)"
            error = None
            self.workflow_state["completed_steps"].append(step.value)
        else:
            message = f"CryoSPARC {step_name} job {job_uid} finished with status '{status}' (fallback mode)"
            error = f"Job status: {status}"
            self.workflow_state["failed_steps"].append(step.value)

        self.workflow_state["active_jobs"][step.value] = {
            "job_uid": job_uid,
            "status": status,
            "details": wait_result
        }

        self.results.append(
            PickingResult(
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
            return True, job_uid

        return False, job_uid

    def _invoke_agent_tool(self, tool_callable, params: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Invoke an agent tool directly and return the recorded result payload."""
        input_str = json.dumps(params)
        prior_len = len(self.agent.tool_execution_log)
        tool_callable(input_str)

        if len(self.agent.tool_execution_log) <= prior_len:
            raise RuntimeError("Tool execution was not recorded in the execution log")

        entry = self.agent.tool_execution_log[prior_len]
        if entry.get("error"):
            raise RuntimeError(entry["error"])

        result_payload = entry.get("result", {})
        if not isinstance(result_payload, dict):
            raise RuntimeError("Tool result payload missing or malformed")

        job_uid = result_payload.get("job_uid")
        if not job_uid:
            raise RuntimeError("Tool did not return a job UID")

        return job_uid, result_payload

    def _invoke_wait_for_job(self, job_uid: str, timeout: int, check_interval: int) -> Dict[str, Any]:
        """Invoke the wait_for_job tool and return its status payload."""
        wait_params = {
            "job_uid": job_uid,
            "timeout": timeout,
            "check_interval": check_interval
        }
        input_str = json.dumps(wait_params)
        prior_len = len(self.agent.tool_execution_log)
        self.agent._wait_for_job_tool(input_str)

        if len(self.agent.tool_execution_log) <= prior_len:
            raise RuntimeError("wait_for_job did not produce a log entry")

        entry = self.agent.tool_execution_log[prior_len]
        if entry.get("error"):
            raise RuntimeError(entry["error"])

        result_payload = entry.get("result", {})
        if not isinstance(result_payload, dict):
            raise RuntimeError("wait_for_job result payload missing or malformed")

        return result_payload

    def _build_blob_picker_params(
        self,
        micrographs_job_uid: str,
        params: Dict[str, Any],
        timeout: int,
        check_interval: int
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "micrographs_job_uid": micrographs_job_uid,
            "particle_diameter": params.get("particle_diameter"),
            "wait_for_completion": "true",
            "timeout": timeout,
            "check_interval": check_interval
        }
        if params.get("diameter_max"):
            payload["diameter_max"] = params.get("diameter_max")
        return payload

    def _build_extract_params(
        self,
        particles_job_uid: str,
        micrographs_job_uid: str,
        params: Dict[str, Any],
        timeout: int,
        check_interval: int
    ) -> Dict[str, Any]:
        return {
            "particles_job_uid": particles_job_uid,
            "micrographs_job_uid": micrographs_job_uid,
            "box_size_pix": params.get("box_size_pix"),
            "wait_for_completion": "true",
            "timeout": timeout,
            "check_interval": check_interval
        }

    def _build_class2d_params(
        self,
        particles_job_uid: str,
        params: Dict[str, Any],
        timeout: int,
        check_interval: int
    ) -> Dict[str, Any]:
        return {
            "particles_job_uid": particles_job_uid,
            "num_classes": params.get("num_classes"),
            "wait_for_completion": "true",
            "timeout": timeout,
            "check_interval": check_interval
        }

    def _build_select_params(
        self,
        class_2d_job_uid: str,
        params: Dict[str, Any],
        timeout: int,
        check_interval: int
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "class_2d_job_uid": class_2d_job_uid,
            "selection_mode": params.get("selection_mode"),
            "top_n_classes": params.get("top_n_classes"),
            "wait_for_completion": "true",
            "timeout": timeout,
            "check_interval": check_interval
        }

        if params.get("cryosift_threshold") is not None:
            payload["cryosift_threshold"] = params.get("cryosift_threshold")
        if params.get("cryosift_env"):
            payload["cryosift_env"] = params.get("cryosift_env")
        if params.get("cryosift_weights_path"):
            payload["cryosift_weights_path"] = params.get("cryosift_weights_path")
        if params.get("cryosift_output_dir"):
            payload["cryosift_output_dir"] = params.get("cryosift_output_dir")
        if params.get("cryosift_output_subdir"):
            payload["cryosift_output_subdir"] = params.get("cryosift_output_subdir")
        if params.get("cryosift_python_executable"):
            payload["cryosift_python_executable"] = params.get("cryosift_python_executable")
        if params.get("cryosift_fallback_strategy"):
            payload["cryosift_fallback_strategy"] = params.get("cryosift_fallback_strategy")

        return payload

    def _build_template_picker_params(
        self,
        micrographs_job_uid: str,
        template_job_uid: str,
        params: Dict[str, Any],
        timeout: int,
        check_interval: int
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "micrographs_job_uid": micrographs_job_uid,
            "template_job_uid": template_job_uid,
            "lowpass_resolution": params.get("lowpass_resolution"),
            "particle_diameter": params.get("particle_diameter"),
            "lowpass_micrograph": params.get("lowpass_resolution"),
            "blob_picker_job_uid": self.current_job_uids.get(PickingStep.BLOB_PICKER),
            "wait_for_completion": "true",
            "timeout": timeout,
            "check_interval": check_interval
        }
        if params.get("angle_search_range") is not None:
            payload["angular_spacing_deg"] = params.get("angle_search_range")
        if params.get("min_distance") is not None:
            payload["min_distance"] = params.get("min_distance")
        return payload
