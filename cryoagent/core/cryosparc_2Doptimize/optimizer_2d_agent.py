"""ReAct-based 2D classification optimization agent for iterative particle selection."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..base_react_agent import BaseReActAgent
from .optimizer_2d_tools import Optimizer2DTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...config.config_loader import CryoAgentConfig


class Optimizer2DAgent(BaseReActAgent):
    """ReAct-based agent for optimizing particle selection through iterative 2D classification."""
    
    # Class-level lock to prevent concurrent classification job creation
    _class_2d_lock = None
    # Class-level flag to track if a job creation is in progress (prevents race condition with parallel tool calls)
    _class_2d_creation_in_progress = False
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the 2D optimization agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        # Initialize stage_config BEFORE calling super().__init__() because
        # BaseReActAgent.__init__() calls _create_tools() which may access stage_config
        self.workflow_defaults: Dict[str, Any] = {}
        self.stage_config = self._load_stage_config()
        self.stage_workflow = self.stage_config.get("workflow", {})
        
        # Initialize class-level lock if not already done
        import threading
        if Optimizer2DAgent._class_2d_lock is None:
            Optimizer2DAgent._class_2d_lock = threading.Lock()
            Optimizer2DAgent._class_2d_creation_in_progress = False
        
        # Now call super().__init__() which will call _create_tools()
        super().__init__(cryosparc_tools, config, llm)
        
        # Initialize logger for this agent
        self.logger = logging.getLogger("Optimizer2DAgent")
    
    def _create_tools(self) -> List[Tool]:
        """Create 2D optimization-specific tools."""
        return [
            Optimizer2DTools.create_class_2d_tool(self),
            Optimizer2DTools.create_select_2d_classes_tool(self),
            Optimizer2DTools.create_get_particle_count_tool(self),
            Optimizer2DTools.create_merge_particles_tool(self),
            Optimizer2DTools.create_get_job_status_tool(self),
            Optimizer2DTools.create_wait_for_job_tool(self),
        ]
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load 2D optimization stage configuration."""
        config_path = Path("configs/cryosparc/optimization_2d_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Fetch a parameter from the stage workflow configuration."""
        opt_config = self.stage_workflow.get("2d_optimization", {})
        if section == "2d_optimization":
            return opt_config.get(key, default)
        elif section == "2d_classification":
            class_config = opt_config.get("2d_classification", {})
            return class_config.get(key, default)
        elif section == "select_2d_classes":
            select_config = opt_config.get("select_2d_classes", {})
            return select_config.get(key, default)
        return default
    
    def _get_react_system_prompt(self) -> str:
        """Get the 2D optimization-specific ReAct system prompt."""
        enable_f1 = self._get_stage_param("2d_optimization", "enable_function1_iterative", True)
        enable_f2 = self._get_stage_param("2d_optimization", "enable_function2_rescue", True)
        max_rounds = self._get_stage_param("2d_optimization", "max_iterative_rounds", 5)
        threshold = self._get_stage_param("2d_optimization", "good_particles_threshold", 0.9)
        
        threshold_pct = int(threshold * 100)
        
        return f"""You are a CryoEM 2D classification optimization assistant using the ReAct (Reasoning + Acting) framework.
You specialize in optimizing particle selection through iterative 2D classification and CryoSift evaluation.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}
- Function 1 (Iterative): {'ENABLED' if enable_f1 else 'DISABLED'}
- Function 2 (Rescue): {'ENABLED' if enable_f2 else 'DISABLED'}
- Max Iterative Rounds: {max_rounds}
- Good Particles Threshold: {threshold_pct}%

## Workflow Overview:

**Step A: Initial 2D Classification + Selection**
1. Run 2D classification on input particles using `class_2d` tool
2. Select good classes using CryoSift with `select_2d_classes` tool
3. Get particle count from selected particles using `get_particle_count` tool

**Step B: Function 2 - Rescue Excluded Particles** {'(ENABLED)' if enable_f2 else '(DISABLED)'}
{'**This step runs ONLY if Function 2 is enabled.**' if enable_f2 else '**This step is SKIPPED when Function 2 is disabled.**'}
1. After Step A completes, you have a select_2D job (e.g., J116) with two output groups:
   - particles_selected: Good particles (already used in Step A)
   - particles_excluded: Excluded particles (need to rescue in Step B)
2. **CRITICAL FOR RESCUE**: Run 2D classification on EXCLUDED particles ONLY:
   - Use: `class_2d` with particles_job_uid="[select_2D_job_uid]" AND particles_group_name="particles_excluded"
   - Example: class_2d with particles_job_uid="J116" and particles_group_name="particles_excluded"
   - **DO NOT** run class_2d on the select_2D job without particles_group_name - this will classify the WRONG particles (selected ones instead of excluded ones)!
   - **DO NOT** use particles_job_uid from the original input - use the select_2D job UID from Step A
3. Select good classes from excluded set using CryoSift with `select_2d_classes`
4. Merge good particles from Step A + good particles from excluded set using `merge_particles` tool

**Step C: Function 1 - Iterative Refinement** {'(ENABLED)' if enable_f1 else '(DISABLED)'}
{'**This step runs after Step B (or Step A if Function 2 is disabled).**' if enable_f1 else '**This step is SKIPPED when Function 1 is disabled.**'}
Iterative loop (max {max_rounds} rounds):
1. Check if ≥{threshold_pct}% of current input particles are good
2. If yes: STOP and return final particles
3. If no: 
   - Run 2D classification on current good particles
   - Select good classes using CryoSift
   - Check percentage again
   - Repeat until ≥{threshold_pct}% OR max {max_rounds} rounds reached

**CRITICAL: When BOTH Function 1 and Function 2 are enabled:**
- Step A: Initial 2D classification → select_2d_classes (creates J157 with particles_selected and particles_excluded)
- Step B: 2D classification on excluded particles from J157 → select_2d_classes (creates J159 with particles_selected)
- After Step B completes, you will have TWO select_2d jobs:
  - J157 from Step A with particles_selected (good particles from initial classification)
  - J159 from Step B with particles_selected (good particles rescued from excluded set)
- **For Step C (Iterative): Call class_2d ONCE - the tool automatically handles both jobs**
  - **CRITICAL: Call class_2d ONLY ONCE, not multiple times**
  - The tool will automatically detect that both functions are enabled and connect both J157.particles_selected and J159.particles_selected in a SINGLE call
  - DO NOT call class_2d multiple times - ONE call handles both jobs automatically
  - This avoids creating an intermediate merge job and connects both particle sets directly to the next classification round

## Execution Logic:

**If F1=OFF, F2=ON**: Run Step A → Step B → Return merged particles
**If F1=ON, F2=OFF**: Run Step A → Step C → Return final particles
**If F1=ON, F2=ON**: Run Step A → Step B → Step C → Return final particles
  * **CRITICAL for Step C**: Call class_2d ONCE - the tool automatically connects both select_2d jobs (from Step A and Step B) in a single call. DO NOT call class_2d multiple times.

## Tool Usage:

- **class_2d**: Run 2D classification on particles
  * Required: particles_job_uid (e.g., "J123") OR job_uid (when passing just "J123")
  * Optional: num_classes (default from config), particles_group_name (e.g., "particles_excluded" to use excluded particles from a select_2D job)
  * Returns: job_uid, status
  * **CRITICAL for Function 2 (Rescue)**: 
    - You MUST use the select_2D job_uid from Step A (e.g., "J116")
    - You MUST specify particles_group_name="particles_excluded" 
    - Correct format: class_2d with particles_job_uid="J116" and particles_group_name="particles_excluded"
    - WRONG: class_2d with particles_job_uid="J116" (missing particles_group_name - will classify selected particles instead!)
    - The tool will reject attempts to classify a select_2D job without particles_group_name
  * **CRITICAL when BOTH Function 1 and Function 2 are enabled (starting Step C)**:
    - **Call class_2d ONCE with any one of the select_2d job UIDs (e.g., J157)**
    - The tool will AUTOMATICALLY detect that both functions are enabled
    - The tool will AUTOMATICALLY connect both select_2d jobs (J157 from Step A and J159 from Step B) in this SINGLE call
    - **DO NOT call class_2d multiple times - ONE call handles both jobs automatically**
    - You don't need to specify both jobs manually - just call class_2d once normally and the tool handles connecting both
    - This connects both J157.particles_selected (from Step A) and J159.particles_selected (from Step B) directly instead of merging them first

- **select_2d_classes**: Select good 2D classes using CryoSift
  * Required: class_2d_job_uid (e.g., "J123")
  * Optional: selection_mode (default: "cryosift"), cryosift_threshold
  * Returns: job_uid, selected_template_indices, selection_metadata
  * **IMPORTANT**: The select_2D job outputs:
    - particles_selected: Good particles (selected classes) - use this group name when referencing selected particles
    - particles_excluded: Excluded particles (non-selected classes) - use this group name when referencing excluded particles
  * **For Function 2 (Rescue)**: To get excluded particles, use the select_2D job_uid with particles_group_name="particles_excluded" in get_particle_count, 
    and use the same job_uid with particles_excluded group when running class_2d on excluded particles

- **get_particle_count**: Get number of particles in a job
  * Required: particles_job_uid (e.g., "J123")
  * Optional: particles_group_name (default: "particles")
  * Returns: num_particles, particles_group_name

- **merge_particles**: Merge particles from multiple jobs
  * Required: particles_job_uids (comma-separated, e.g., "J123,J124")
  * Returns: merged job_uid, status
  * **When BOTH Function 1 and Function 2 are enabled**: DO NOT merge after Step B! 
    Instead, connect both select_2d jobs directly to the iterative class_2d job.
    Only use merge_particles if Function 1 is disabled (F1=OFF, F2=ON scenario).

- **get_job_status**: Check status of a job
- **wait_for_job**: Wait for job completion

## Key Workflow Steps:

1. **Always start with Step A**: Run initial 2D classification and selection
2. **Check Function 2**: If enabled, run rescue workflow (Step B)
3. **Check Function 1**: If enabled, run iterative refinement (Step C)
4. **Calculate percentages**: Use get_particle_count to check if threshold is met
5. **Final output**: Return final particles_job_uid and log summary

## Important Notes:

- **CRITICAL: Sequential Execution Only - ONE Call Per Step**
  - **NEVER run multiple 2D classification jobs simultaneously**
  - **ALWAYS wait for one classification job to complete before starting the next**
  - **ONLY call class_2d tool ONCE per step - even when you have multiple input jobs, call it ONCE and the tool handles it**
  - **When both Function 1 and Function 2 are enabled, call class_2d ONCE in Step C - the tool automatically connects both jobs**
  - If you see an error about a job already running, wait for it to complete first
  - Check job status using `get_job_status` or `wait_for_job` before starting new jobs

- **Particle Count Calculation**: 
  - Good particles percentage = (selected_particles_count / input_particles_count) × 100
  - Always use the CURRENT input particles count for percentage calculation
  - After each round, the input becomes the selected particles from previous round

- **Excluded Particles**: 
  - Use particles_excluded group from select_2D job to get excluded particles
  - This is needed for Function 2 (rescue workflow)

- **Stopping Conditions**:
  - Function 1 stops when: ≥{threshold_pct}% good particles OR max {max_rounds} rounds reached
  - Always check particle count after each selection to determine if threshold is met

- **Final Summary**: 
  - Log: "Final Good Particles: X (Y% of current input). Total Rounds: Z."
  - Return final particles_job_uid

Remember: Always follow the Thought → Action → Observation pattern!
Think carefully about the workflow order and which functions are enabled before proceeding.
**NEVER make multiple parallel tool calls - execute tools one at a time, waiting for each to complete!**"""

    def update_workflow_defaults(self, defaults: Dict[str, Any]) -> None:
        """Store workflow-level default parameters for later tool invocations."""
        if defaults:
            if not hasattr(self, "workflow_defaults") or self.workflow_defaults is None:
                self.workflow_defaults = {}
            self.workflow_defaults.update(defaults)
    
    # =================================================================
    # Tool Implementation Methods
    # =================================================================
    
    def _determine_particles_group_name(self, particles_job_uid: str) -> Optional[str]:
        """
        Intelligently determine which particles group name to use based on workflow context.
        
        Note: For the first round (Step A), the _class_2d_tool method now connects both
        particles_selected and particles_excluded simultaneously, so this method is only
        used for subsequent rounds.
        
        Logic:
        1. First round (Step A): Handled separately - connects both particles_selected and particles_excluded
        2. Rescue (Step B): Input is select_2d job from Step A → use "particles_excluded"
        3. Iterative (Step C): Input is select_2d job from previous round → use "particles_selected"
        
        Args:
            particles_job_uid: The job UID to classify
            
        Returns:
            particles_group_name to use, or None to let the connection logic try defaults
        """
        # Get the original input particles_job_uid from workflow defaults
        original_input_job_uid = self.workflow_defaults.get("particles_job_uid")
        
        # If this is the original input job (from picking), this method shouldn't be called
        # for the first round anymore (handled directly in _class_2d_tool with both groups)
        # But keep this for backward compatibility
        if particles_job_uid == original_input_job_uid:
            # This case is now handled in _class_2d_tool, but return None as fallback
            return None
        
        # Check if this is a select_2d job by looking at execution log
        recent_select_jobs = [
            entry for entry in self.tool_execution_log[-20:]
            if entry.get("tool") == "select_2d_classes"
        ]
        
        # Find if particles_job_uid matches any select_2d job
        is_select_2d_job = False
        select_job_index = -1
        for i, select_entry in enumerate(reversed(recent_select_jobs)):
            select_result = select_entry.get("result")
            if isinstance(select_result, dict):
                select_job_uid = select_result.get("job_uid")
                if select_job_uid == particles_job_uid:
                    is_select_2d_job = True
                    select_job_index = len(recent_select_jobs) - 1 - i
                    break
        
        if not is_select_2d_job:
            # Not a select_2d job, use default connection logic
            return None
        
        # This is a select_2d job - determine if it's rescue or iterative
        # Count how many class_2d jobs have been run
        class_2d_count = sum(
            1 for entry in self.tool_execution_log[-20:]
            if entry.get("tool") == "class_2d"
        )
        
        # Count how many select_2d_classes jobs have been run
        select_2d_count = len(recent_select_jobs)
        
        # Logic to determine if this is rescue or iterative:
        # - Step B (Rescue): First select_2d job (index 0) AND exactly 1 class_2d job has run
        #   This means: Step A (class_2d) → Step A (select_2d) → Step B (class_2d on excluded)
        # - Step C (Iterative): Any other select_2d job
        #   This means: Previous round's select_2d → Next round's class_2d on selected particles
        
        # Check if Function 2 (Rescue) is enabled
        enable_f2 = self._get_stage_param("2d_optimization", "enable_function2_rescue", True)
        
        # Determine if this is rescue:
        # - Rescue is enabled AND
        # - This is the first select_2d job (index 0) AND  
        # - Only 1 class_2d job has run so far (Step A's initial classification)
        is_rescue = enable_f2 and select_job_index == 0 and class_2d_count == 1
        
        if is_rescue:
            # This is Step B (Rescue) - classify the EXCLUDED particles from Step A
            self.logger.info(
                f"Auto-detected rescue workflow (Step B): Using particles_group_name='particles_excluded' "
                f"for select_2d job {particles_job_uid} (select_2d index={select_job_index}, class_2d count={class_2d_count})"
            )
            return "particles_excluded"
        else:
            # This is Step C (Iterative) or any later round
            # Classify the SELECTED particles from the previous round
            self.logger.info(
                f"Auto-detected iterative workflow (Step C or later): Using particles_group_name='particles_selected' "
                f"for select_2d job {particles_job_uid} (select_2d index={select_job_index}, class_2d count={class_2d_count})"
            )
            return "particles_selected"
    
    def _check_running_class_2d_jobs(self, project_uid: str, workspace_uid: str) -> Optional[str]:
        """
        Check if there are any running class_2d jobs in the workspace.
        
        This method checks both:
        1. The execution log for recently created class_2d jobs (to catch parallel tool calls)
        2. CryoSPARC for actual running jobs
        
        Returns:
            Job UID of a running job if found, None otherwise
        """
        try:
            # CRITICAL: First check execution log for ANY recent class_2d jobs
            # This catches jobs that were just created by parallel tool calls but haven't
            # been verified in CryoSPARC yet. This prevents the race condition where
            # multiple parallel calls all pass the check before any job is recorded.
            recent_class_2d_jobs = [
                entry for entry in self.tool_execution_log[-20:]
                if entry.get("tool") == "class_2d"
            ]
            
            # If there's a recent class_2d job in the log, check if it's still running
            # This prevents creating multiple jobs when the LLM makes parallel tool calls
            if recent_class_2d_jobs:
                # Get the most recent class_2d job
                most_recent_entry = recent_class_2d_jobs[-1]
                result = most_recent_entry.get("result")
                
                if isinstance(result, dict):
                    job_uid = result.get("job_uid")
                    if job_uid:
                        # Check if this job has completed - if not, it's still running
                        log_status = result.get("status", "")
                        
                        # If status is not "completed", "failed", or "cancelled", it's still running
                        if log_status not in ("completed", "failed", "cancelled"):
                            # Verify status in CryoSPARC if possible
                            try:
                                job_status = self.cryosparc_tools.get_job_status(
                                    job_uid,
                                    project_uid=project_uid,
                                    workspace_uid=workspace_uid
                                )
                                actual_status = job_status.get("status", "")
                                job_type = job_status.get("job_type", "")
                                
                                # Check if it's a class_2d job and still running
                                if job_type in ("class_2D", "class_2d") and actual_status in ("queued", "launched", "running", "waiting"):
                                    self.logger.info(f"Found running 2D classification job {job_uid} with status: {actual_status}")
                                    if self.enable_conversation_logging and self.realtime_logger.current_log_file:
                                        self.realtime_logger.log_tool_execution(
                                            "get_job_status",
                                            {"job_uid": job_uid},
                                            f"Job {job_uid} is currently {actual_status}"
                                        )
                                    return job_uid
                            except Exception:
                                # If we can't check status in CryoSPARC, but the log shows it's not completed,
                                # assume it's still running to be safe
                                if log_status in ("queued", "launched", "running", "waiting", "unknown", ""):
                                    self.logger.info(f"Found recent 2D classification job {job_uid} (from execution log) with status: {log_status}")
                                    return job_uid
                        else:
                            # Job has completed, check all other recent jobs
                            for entry in reversed(recent_class_2d_jobs[:-1]):  # Check all except the most recent
                                result = entry.get("result")
                                if isinstance(result, dict):
                                    job_uid = result.get("job_uid")
                                    if job_uid:
                                        try:
                                            job_status = self.cryosparc_tools.get_job_status(
                                                job_uid,
                                                project_uid=project_uid,
                                                workspace_uid=workspace_uid
                                            )
                                            actual_status = job_status.get("status", "")
                                            job_type = job_status.get("job_type", "")
                                            if job_type in ("class_2D", "class_2d") and actual_status in ("queued", "launched", "running", "waiting"):
                                                self.logger.info(f"Found running 2D classification job {job_uid} with status: {actual_status}")
                                                return job_uid
                                        except Exception:
                                            log_status = result.get("status", "")
                                            if log_status in ("queued", "launched", "running", "waiting"):
                                                self.logger.info(f"Found running 2D classification job {job_uid} (from execution log) with status: {log_status}")
                                                return job_uid
            
            # Also check workspace directly for any class_2d jobs (more comprehensive)
            try:
                project = self.cryosparc_tools.cs.find_project(project_uid)
                workspace = project.find_workspace(workspace_uid)
                
                # Get all jobs in the workspace
                for job_uid in workspace.list_job_uids():
                    try:
                        job = workspace.find_job(job_uid)
                        if job:
                            job.refresh()
                            doc = getattr(job, "doc", {})
                            job_type = doc.get("job_type", "")
                            status = doc.get("status", getattr(job, "status", "unknown"))
                            
                            # Check if it's a class_2d job in a running state
                            if job_type in ("class_2D", "class_2d") and status in ("queued", "launched", "running", "waiting"):
                                return job_uid
                    except Exception:
                        # Skip jobs we can't check
                        continue
            except Exception:
                # If workspace check fails, we already checked execution log
                pass
            
            return None
        except Exception:
            return None
    
    def _class_2d_tool(self, tool_input: str) -> str:

        """Tool wrapper for 2D classification."""
        # Use lock to prevent concurrent job creation when LLM makes parallel tool calls
        with Optimizer2DAgent._class_2d_lock:
            try:
                params = self._parse_tool_input(tool_input)
                project_uid = params.get("project_uid", self.config.workflow.project_uid)
                workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
                
                # CRITICAL: Check for pending job creation attempts first
                # This prevents race condition when multiple parallel tool calls enter the lock
                if Optimizer2DAgent._class_2d_creation_in_progress:
                    return json.dumps({
                        "success": False,
                        "error": "A 2D classification job creation is already in progress. Wait for it to complete before starting a new one. Use 'wait_for_job' or 'get_job_status' to check its status."
                    })
                
                # CRITICAL: Check execution log for ANY recent class_2d job (even if just created)
                # This catches jobs created by parallel tool calls that haven't been recorded yet
                recent_class_2d_jobs = [
                    entry for entry in self.tool_execution_log[-10:]
                    if entry.get("tool") == "class_2d"
                ]
                if recent_class_2d_jobs:
                    # Get the most recent class_2d job
                    most_recent = recent_class_2d_jobs[-1]
                    result = most_recent.get("result")
                    if isinstance(result, dict):
                        job_uid = result.get("job_uid")
                        status = result.get("status", "")
                        # If status is "creating" (pending entry) or job exists and hasn't completed/failed, reject new creation
                        if status == "creating" or (job_uid and status not in ("completed", "failed", "cancelled")):
                            if status == "creating":
                                return json.dumps({
                                    "success": False,
                                    "error": "A 2D classification job creation is already in progress (detected from execution log). Wait for it to complete before starting a new one."
                                })
                            else:
                                return json.dumps({
                                    "success": False,
                                    "error": f"A 2D classification job was recently created (job: {job_uid}, status: {status}). Wait for it to complete before starting a new one. Use 'wait_for_job' or 'get_job_status' to check its status.",
                                    "recent_job_uid": job_uid
                                })
                
                # CRITICAL: Check for running class_2d jobs BEFORE creating a new one
                # This prevents multiple parallel tool calls from all creating jobs
                running_job_uid = self._check_running_class_2d_jobs(project_uid, workspace_uid)
                if running_job_uid:
                    return json.dumps({
                        "success": False,
                        "error": f"A 2D classification job is already running (job: {running_job_uid}). Wait for it to complete before starting a new one. Use 'wait_for_job' or 'get_job_status' to check its status.",
                        "running_job_uid": running_job_uid
                    })
                
                # Mark that we're about to create a job (prevents other parallel calls from creating jobs)
                Optimizer2DAgent._class_2d_creation_in_progress = True
                
                try:
                    # Check if both Function 1 and Function 2 are enabled
                    enable_f1 = self._get_stage_param("2d_optimization", "enable_function1_iterative", True)
                    enable_f2 = self._get_stage_param("2d_optimization", "enable_function2_rescue", True)
                
                    # Support both particles_job_uid and job_uid (when LLM passes just "J88")
                    particles_job_uid = params.get("particles_job_uid") or params.get("job_uid")
                    
                    # Check if we need to connect multiple jobs directly (both functions enabled, starting iterative step)
                    particles_job_uids = None
                    particles_group_names = None
                    
                    if enable_f1 and enable_f2:
                        # Check if we're at the start of Step C (iterative step) after Step B
                        # Step A: class_2d → select_2d_classes (creates J157 with particles_selected and particles_excluded)
                        # Step B: class_2d on excluded particles from J157 → select_2d_classes (creates J159 with particles_selected)
                        # Step C: Should connect BOTH J157.particles_selected and J159.particles_selected directly
                        
                        recent_select_jobs = [
                            entry for entry in self.tool_execution_log[-20:]
                            if entry.get("tool") == "select_2d_classes"
                        ]
                        
                        # Check if we have exactly 2 select_2d jobs (J157 from Step A, J159 from Step B)
                        # This indicates Step B has just completed
                        if len(recent_select_jobs) == 2:
                            # Get the two select_2d jobs (J157 from Step A, J159 from Step B)
                            select_job_A_result = recent_select_jobs[-2].get("result")  # J157 from Step A
                            select_job_B_result = recent_select_jobs[-1].get("result")  # J159 from Step B
                            
                            if isinstance(select_job_A_result, dict) and isinstance(select_job_B_result, dict):
                                job_A_uid = select_job_A_result.get("job_uid")  # J157
                                job_B_uid = select_job_B_result.get("job_uid")  # J159
                                
                                # Check if this is the first class_2d call after Step B (starting iterative step)
                                # Count how many class_2d jobs have been run - should be 2 (Step A and Step B)
                                recent_class_2d_jobs = [
                                    entry for entry in self.tool_execution_log[-20:]
                                    if entry.get("tool") == "class_2d"
                                ]
                                
                                # If we have exactly 2 class_2d jobs and 2 select_2d jobs, we're starting Step C
                                if len(recent_class_2d_jobs) == 2 and job_A_uid and job_B_uid:
                                    # Connect both jobs directly instead of merging
                                    # Order: J157 (Step A selected) first, then J159 (Step B selected)
                                    particles_job_uids = [job_A_uid, job_B_uid]
                                    particles_group_names = ["particles_selected", "particles_selected"]
                                    self.logger.info(
                                        f"Both Function 1 and Function 2 enabled: Connecting both select_2d jobs directly "
                                        f"(J157={job_A_uid} from Step A and J159={job_B_uid} from Step B) to iterative class_2d instead of merging them."
                                    )
                    
                    if not particles_job_uids:
                        # Single particle input (original behavior)
                        if not particles_job_uid:
                            return json.dumps({
                                "success": False,
                                "error": "Missing required parameter: particles_job_uid (or job_uid)"
                            })
                        
                        # Check if this is the first round (original input job)
                        # If so, use both particles_selected and particles_excluded simultaneously
                        original_input_job_uid = self.workflow_defaults.get("particles_job_uid")
                        is_first_round = (particles_job_uid == original_input_job_uid)
                        
                        if is_first_round:
                            # First round: Connect both particles_selected and particles_excluded from the same job
                            particles_job_uids = [particles_job_uid, particles_job_uid]
                            particles_group_names = ["particles_selected", "particles_excluded"]
                            self.logger.info(
                                f"First round (Step A): Connecting both particles_selected and particles_excluded "
                                f"from job {particles_job_uid} to class_2d simultaneously"
                            )
                        else:
                            # Not first round: Use single particle input with auto-determined group name
                            # Get particles_group_name if specified (for excluded particles)
                            particles_group_name = params.get("particles_group_name")
                            
                            # Auto-determine particles_group_name based on workflow context if not specified
                            if not particles_group_name:
                                particles_group_name = self._determine_particles_group_name(particles_job_uid)
                            
                            # Validation: If particles_job_uid is a select_2d job, particles_group_name must be specified
                            # This prevents accidentally classifying the wrong particles
                            if not particles_group_name:
                                recent_select_jobs = [
                                    entry for entry in self.tool_execution_log[-20:]
                                    if entry.get("tool") == "select_2d_classes"
                                ]
                                for select_entry in reversed(recent_select_jobs):
                                    select_result = select_entry.get("result")
                                    if isinstance(select_result, dict):
                                        select_job_uid = select_result.get("job_uid")
                                        if select_job_uid == particles_job_uid:
                                            # This is a select_2d job but no group name determined - this shouldn't happen
                                            # but if it does, default to particles_selected (safer than particles_excluded)
                                            particles_group_name = "particles_selected"
                                            self.logger.warning(
                                                f"Auto-selected particles_group_name='particles_selected' for select_2d job {particles_job_uid}. "
                                                f"Consider explicitly specifying particles_group_name in future calls."
                                            )
                                            break
                    
                    # Get num_classes from params or config
                    num_classes = params.get("num_classes")
                    if not num_classes:
                        num_classes = self._get_stage_param("2d_classification", "num_classes", 50)
                    
                    used_params = {
                        "project_uid": project_uid,
                        "workspace_uid": workspace_uid,
                        "num_classes": int(num_classes),
                        "wait_for_completion": params.get("wait_for_completion", "true").lower() == "true",
                        "timeout": int(params.get("timeout", self.config.job_management.default_timeout * 2)),
                        "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
                    }
                    
                    if particles_job_uids:
                        # Multiple particle inputs - connect both jobs directly
                        used_params["particles_job_uids"] = particles_job_uids
                        used_params["particles_group_names"] = particles_group_names
                    else:
                        # Single particle input
                        used_params["particles_job_uid"] = particles_job_uid
                        if particles_group_name:
                            used_params["particles_group_name"] = particles_group_name
                    
                    # Record job creation attempt IMMEDIATELY in execution log (before creating job)
                    # This allows parallel tool calls to detect pending job creation
                    pending_result = {"status": "creating", "job_uid": None}
                    self._record_tool_execution("class_2d", used_params, result=pending_result)
                    
                    result = self.cryosparc_tools.class_2d(**used_params)
                    
                    # Get job UID and current status for logging
                    job_uid = result.get("job_uid")
                    job_status = result.get("status", "unknown")
                    
                    # Check actual job status from CryoSPARC if job was launched
                    if job_uid and job_status in ("queued", "launched", "running", "waiting"):
                        try:
                            # Get current status from CryoSPARC to ensure accurate logging
                            actual_status = self.cryosparc_tools.get_job_status(
                                job_uid,
                                project_uid=project_uid,
                                workspace_uid=workspace_uid
                            )
                            job_status = actual_status.get("status", job_status)
                            # Update result with actual status
                            result["status"] = job_status
                        except Exception:
                            # If status check fails, use the status from class_2d result
                            pass
                    
                    # Update the execution log entry with actual job result (replaces the pending entry)
                    # Remove the pending entry and add the real one
                    if self.tool_execution_log and self.tool_execution_log[-1].get("tool") == "class_2d":
                        # Update the last entry (which should be our pending entry)
                        self.tool_execution_log[-1]["result"] = result
                    else:
                        # Fallback: record normally if pending entry wasn't found
                        self._record_tool_execution("class_2d", used_params, result=result)
                    
                    # Log job launch with clear status message
                    if job_uid:
                        status_msg = f"2D classification job {job_uid} launched with status: {job_status}"
                        if job_status in ("queued", "launched", "running", "waiting"):
                            self.logger.info(status_msg)
                            # Also log to realtime logger if available
                            if self.enable_conversation_logging and self.realtime_logger.current_log_file:
                                self.realtime_logger.log_tool_execution(
                                    "class_2d",
                                    {"job_uid": job_uid, "status": job_status, "num_classes": num_classes},
                                    f"Job {job_uid} is {job_status}"
                                )
                    
                    if result.get("status") == "completed":
                        return json.dumps({
                            "success": True,
                            "job_uid": job_uid,
                            "status": "completed",
                            "num_classes": num_classes
                        })
                    else:
                        return json.dumps({
                            "success": True,
                            "job_uid": job_uid,
                            "status": job_status
                        })
                finally:
                    # Always clear the creation flag when done (success or error)
                    Optimizer2DAgent._class_2d_creation_in_progress = False
                    
            except Exception as e:
                # Clean up creation flag in case of exception
                Optimizer2DAgent._class_2d_creation_in_progress = False
                
                error_result = {"success": False, "error": str(e)}
                self._record_tool_execution("class_2d", params if 'params' in locals() else {}, error=str(e))
                return json.dumps(error_result)
    
    def _select_2d_classes_tool(self, tool_input: str) -> str:
        """Tool wrapper for 2D class selection."""
        try:
            params = self._parse_tool_input(tool_input)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Support both class_2d_job_uid and job_uid (when LLM passes just "J88")
            class_2d_job_uid = params.get("class_2d_job_uid") or params.get("job_uid")
            if not class_2d_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: class_2d_job_uid (or job_uid)"
                })
            
            # Get selection parameters from config
            selection_mode = params.get("selection_mode") or self._get_stage_param("select_2d_classes", "selection_mode", "cryosift")
            
            cryosift_options = {}
            if selection_mode.lower() == "cryosift":
                cryosift_threshold = params.get("cryosift_threshold")
                if cryosift_threshold is None:
                    cryosift_threshold = self._get_stage_param("select_2d_classes", "cryosift_threshold", 3)
                if cryosift_threshold is not None:
                    cryosift_options["threshold"] = float(cryosift_threshold)
                
                cryosift_env = params.get("cryosift_env") or self._get_stage_param("select_2d_classes", "cryosift_env")
                if cryosift_env:
                    cryosift_options["conda_env"] = cryosift_env
                
                cryosift_weights_path = params.get("cryosift_weights_path") or self._get_stage_param("select_2d_classes", "cryosift_weights_path")
                if cryosift_weights_path:
                    cryosift_options["weights_path"] = cryosift_weights_path
                
                cryosift_output_subdir = params.get("cryosift_output_subdir") or self._get_stage_param("select_2d_classes", "cryosift_output_subdir")
                if cryosift_output_subdir:
                    cryosift_options["output_subdir"] = cryosift_output_subdir
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "class_2d_job_uid": class_2d_job_uid,
                "selection_mode": selection_mode,
                "wait_for_completion": params.get("wait_for_completion", "true").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }
            
            if cryosift_options:
                used_params["cryosift_options"] = cryosift_options
            
            result = self.cryosparc_tools.select_2d_classes(**used_params)
            
            # Get job UID and current status for logging
            job_uid = result.get("job_uid")
            job_status = result.get("status", "unknown")
            
            # Check actual job status from CryoSPARC if job was launched
            if job_uid and job_status in ("queued", "launched", "running", "waiting"):
                try:
                    # Get current status from CryoSPARC to ensure accurate logging
                    actual_status = self.cryosparc_tools.get_job_status(
                        job_uid,
                        project_uid=project_uid,
                        workspace_uid=workspace_uid
                    )
                    job_status = actual_status.get("status", job_status)
                except Exception:
                    # If status check fails, use the status from select_2d_classes result
                    pass
            
            # Format result for LLM and execution log
            if result.get("status") == "completed":
                formatted_result = {
                    "success": True,
                    "job_uid": job_uid,
                    "status": "completed",
                    "selected_template_indices": result.get("selected_template_indices", []),
                    "selection_metadata": result.get("selection_metadata", {})
                }
            else:
                formatted_result = {
                    "success": True,
                    "job_uid": job_uid,
                    "status": job_status
                }
            
            # Record tool execution with updated status
            self._record_tool_execution("select_2d_classes", used_params, result=formatted_result)
            
            # Log job launch with clear status message
            if job_uid:
                status_msg = f"2D class selection job {job_uid} launched with status: {job_status}"
                if job_status in ("queued", "launched", "running", "waiting"):
                    self.logger.info(status_msg)
                    # Also log to realtime logger if available
                    if self.enable_conversation_logging and self.realtime_logger.current_log_file:
                        self.realtime_logger.log_tool_execution(
                            "select_2d_classes",
                            {"job_uid": job_uid, "status": job_status, "selection_mode": selection_mode},
                            f"Job {job_uid} is {job_status}"
                        )
            
            return json.dumps(formatted_result)
                
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("select_2d_classes", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _get_particle_count_tool(self, tool_input: str) -> str:
        """Tool wrapper for getting particle count."""
        try:
            params = self._parse_tool_input(tool_input)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            
            # Support both particles_job_uid and job_uid (when LLM passes just "J88")
            particles_job_uid = params.get("particles_job_uid") or params.get("job_uid")
            if not particles_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: particles_job_uid (or job_uid)"
                })
            
            particles_group_name = params.get("particles_group_name", "particles")
            
            result = self.cryosparc_tools.get_particle_count(
                project_uid,
                particles_job_uid,
                particles_group_name
            )
            self._record_tool_execution("get_particle_count", params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("get_particle_count", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _merge_particles_tool(self, tool_input: str) -> str:
        """Tool wrapper for merging particles."""
        try:
            params = self._parse_tool_input(tool_input)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            particles_job_uids = params.get("particles_job_uids")
            if not particles_job_uids:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: particles_job_uids (should be a list)"
                })
            
            if not isinstance(particles_job_uids, list):
                return json.dumps({
                    "success": False,
                    "error": "particles_job_uids must be a list"
                })
            
            if len(particles_job_uids) < 2:
                return json.dumps({
                    "success": False,
                    "error": "Need at least 2 particle jobs to merge"
                })
            
            particles_group_names = params.get("particles_group_names")
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uids": particles_job_uids,
                "particles_group_names": particles_group_names,
                "wait_for_completion": params.get("wait_for_completion", "true").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }
            
            result = self.cryosparc_tools.merge_particles(**used_params)
            self._record_tool_execution("merge_particles", used_params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("merge_particles", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)

