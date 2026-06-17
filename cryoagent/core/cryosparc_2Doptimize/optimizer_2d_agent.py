"""ReAct-based 2D classification optimization agent for iterative particle selection."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..cryosparc_common_tools import CryoSPARCCommonTools
from ..base_react_agent import BaseReActAgent
from ...tools.cryosparc_tools import CryoSPARCTools
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


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
        
        # Track previous round's median cryosift score for quality degradation detection
        self._previous_round_median_score: Optional[float] = None
    
    def _create_tools(self) -> List[Tool]:
        """Create 2D optimization-specific tools."""
        from ..cryosparc_tool_registry import build_tools, AGENT_TOOL_SETS
        return build_tools(self, AGENT_TOOL_SETS["optimization_2d"])
    
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
    
    def _get_2d_optimization_flags(self) -> Dict[str, Any]:
        """Read 2D optimization config flags used by system and task prompts."""
        enable_f1 = self._get_stage_param("2d_optimization", "enable_function1_iterative", True)
        enable_f2 = self._get_stage_param("2d_optimization", "enable_function2_rescue", True)
        max_rounds = self._get_stage_param("2d_optimization", "max_iterative_rounds", 5)
        threshold = self._get_stage_param("2d_optimization", "good_particles_threshold", 0.9)
        enable_select_all = self._get_stage_param("2d_optimization", "enable_select_all_after_last_round", False)
        stop_on_degradation = self._get_stage_param("2d_optimization", "stop_on_quality_degradation", True)
        threshold_pct = int(threshold * 100)
        return {
            "enable_f1": enable_f1,
            "enable_f2": enable_f2,
            "max_rounds": max_rounds,
            "threshold_pct": threshold_pct,
            "enable_select_all": enable_select_all,
            "stop_on_degradation": stop_on_degradation,
        }

    def _get_system_prompt_context(self) -> Dict[str, Any]:
        """Build template variables for cryosparc/optimization_2d/system.md."""
        flags = self._get_2d_optimization_flags()
        enable_f1 = flags["enable_f1"]
        enable_f2 = flags["enable_f2"]
        stop_on_degradation = flags["stop_on_degradation"]
        return {
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
            "enable_f1_status": "ENABLED" if enable_f1 else "DISABLED",
            "enable_f2_status": "ENABLED" if enable_f2 else "DISABLED",
            "enable_f1_paren_status": "(ENABLED)" if enable_f1 else "(DISABLED)",
            "enable_f2_paren_status": "(ENABLED)" if enable_f2 else "(DISABLED)",
            "enable_select_all_status": "ENABLED" if flags["enable_select_all"] else "DISABLED",
            "max_rounds": flags["max_rounds"],
            "threshold_pct": flags["threshold_pct"],
            "enable_f2_step_note": (
                "**This step runs ONLY if Function 2 is enabled.**"
                if enable_f2
                else "**This step is SKIPPED when Function 2 is disabled.**"
            ),
            "enable_f1_step_note": (
                "**This step runs after Step B (or Step A if Function 2 is disabled).**"
                if enable_f1
                else "**This step is SKIPPED when Function 1 is disabled.**"
            ),
            "stop_on_degradation_paren_status": "(ENABLED)" if stop_on_degradation else "(DISABLED)",
            "stop_on_degradation_word": "enabled" if stop_on_degradation else "disabled",
        }

    def _get_react_system_prompt(self) -> str:
        """Get the 2D optimization-specific ReAct system prompt."""
        return self._compose_stage_system_prompt(
            "cryosparc/optimization_2d/system.md",
            self._get_system_prompt_context(),
        )

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
                    
                    # Get force_max from params or config
                    force_max = params.get("force_max")
                    if force_max is None:
                        force_max_config = self._get_stage_param("2d_classification", "force_max")
                        if force_max_config is not None:
                            force_max = bool(force_max_config)
                    
                    # Get batchsize_per_class from params or config
                    batchsize_per_class = params.get("batchsize_per_class")
                    if batchsize_per_class is None:
                        batchsize_per_class_config = self._get_stage_param("2d_classification", "batchsize_per_class")
                        if batchsize_per_class_config is not None:
                            batchsize_per_class = int(batchsize_per_class_config)
                    
                    used_params = {
                        "project_uid": project_uid,
                        "workspace_uid": workspace_uid,
                        "num_classes": int(num_classes),
                        "wait_for_completion": self._parse_bool_param(params.get("wait_for_completion"), True),
                        "timeout": int(params.get("timeout", self.config.job_management.default_timeout * 2)),
                        "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
                    }
                    
                    # Add force_max if specified
                    if force_max is not None:
                        used_params["force_max"] = force_max
                    
                    # Add batchsize_per_class if specified
                    if batchsize_per_class is not None:
                        used_params["batchsize_per_class"] = batchsize_per_class
                    
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

                    passthrough = self._extract_passthrough_params(
                        params,
                        consumed_keys=["particles_job_uid", "job_uid", "particles_group_name", "num_classes", "force_max", "batchsize_per_class"],
                    )
                    if passthrough:
                        used_params["params"] = passthrough

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
                
                cryosift_evaluator_script_path = params.get("cryosift_evaluator_script_path") or self._get_stage_param("select_2d_classes", "cryosift_evaluator_script_path")
                if cryosift_evaluator_script_path:
                    cryosift_options["evaluator_script_path"] = cryosift_evaluator_script_path
                
                cryosift_output_subdir = params.get("cryosift_output_subdir") or self._get_stage_param("select_2d_classes", "cryosift_output_subdir")
                if cryosift_output_subdir:
                    cryosift_options["output_subdir"] = cryosift_output_subdir
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "class_2d_job_uid": class_2d_job_uid,
                "selection_mode": selection_mode,
                "wait_for_completion": self._parse_bool_param(params.get("wait_for_completion"), True),
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
                
                # Check for quality degradation using cryosift median scores
                # This is only relevant for iterative rounds (Function 1, Step C)
                selection_metadata = result.get("selection_metadata", {})
                if selection_metadata.get("selection_mode") == "cryosift":
                    scores = selection_metadata.get("scores", {})
                    if scores:
                        # Calculate median score of selected classes
                        score_values = [score for score in scores.values() if score is not None]
                        if score_values:
                            import statistics
                            current_median_score = statistics.median(score_values)
                            
                            # Check if quality degradation check is enabled
                            stop_on_degradation = self._get_stage_param("2d_optimization", "stop_on_quality_degradation", True)
                            
                            # Determine if we should compare scores
                            enable_f1 = self._get_stage_param("2d_optimization", "enable_function1_iterative", True)
                            is_iterative_round = enable_f1 and self._previous_round_median_score is not None
                            
                            if is_iterative_round and stop_on_degradation:
                                # We have a baseline from a previous round—compare and flag degradation
                                if current_median_score > self._previous_round_median_score:
                                    formatted_result["quality_degradation_detected"] = True
                                    formatted_result["current_median_score"] = current_median_score
                                    formatted_result["previous_median_score"] = self._previous_round_median_score
                                    formatted_result["quality_warning"] = (
                                        f"⚠️ Quality degradation detected: Current round median score ({current_median_score:.3f}) "
                                        f"is higher (worse) than previous round ({self._previous_round_median_score:.3f}). "
                                        f"Consider stopping further rounds of 2D classification."
                                    )
                                    self.logger.warning(
                                        "Quality degradation detected in 2D classification: "
                                        f"median score increased from {self._previous_round_median_score:.3f} to {current_median_score:.3f}"
                                    )
                                else:
                                    formatted_result["quality_degradation_detected"] = False
                                    formatted_result["current_median_score"] = current_median_score
                                    formatted_result["previous_median_score"] = self._previous_round_median_score
                                    self.logger.info(
                                        "Quality maintained or improved: "
                                        f"median score changed from {self._previous_round_median_score:.3f} to {current_median_score:.3f}"
                                    )
                            
                            # Always update baseline so the next round compares against the latest score
                            self._previous_round_median_score = current_median_score
                            formatted_result["current_median_score"] = current_median_score
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
            # Accept a comma-separated string (e.g. "J123,J124") as well as a list,
            # so this works as a plain Tool without the former StructuredTool adapter.
            if isinstance(particles_job_uids, str):
                particles_job_uids = [
                    uid.strip() for uid in particles_job_uids.split(",") if uid.strip()
                ]
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
                "wait_for_completion": self._parse_bool_param(params.get("wait_for_completion"), True),
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

