"""ReAct-based preprocessing agent for CryoEM data processing."""

from typing import Dict, Any, List
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from typing import Optional

from ..base_react_agent import BaseReActAgent
from .preprocessing_tools import PreprocessingTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...config.config_loader import CryoAgentConfig


class PreprocessingAgent(BaseReActAgent):
    """ReAct-based agent for CryoEM preprocessing operations."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the preprocessing agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        super().__init__(cryosparc_tools, config, llm)
    
    def _create_tools(self) -> List[Tool]:
        """Create preprocessing-specific tools."""
        return [
            PreprocessingTools.create_import_movies_tool(self),
            PreprocessingTools.create_motion_correction_tool(self),
            PreprocessingTools.create_ctf_estimation_tool(self),
            PreprocessingTools.create_micrograph_selection_tool(self),
            PreprocessingTools.create_get_job_status_tool(self),
            PreprocessingTools.create_wait_for_job_tool(self),
            PreprocessingTools.create_reason_about_workflow_tool(self)
        ]
    
    def _get_react_system_prompt(self) -> str:
        """Get the preprocessing-specific ReAct system prompt."""
        return f"""You are a CryoEM preprocessing assistant using the ReAct (Reasoning + Acting) framework. 
You specialize in the initial stages of cryoEM data processing: movie import, motion correction, CTF estimation, and micrograph selection.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Preprocessing Workflow Steps (in order):
1. **Import Movies**: Import raw movie files into CryoSPARC
   - Required: movies_path, pixel_size, voltage, cs_mm, dose
   - Optional: gain_ref_path, project_uid, workspace_uid
   
2. **Motion Correction**: Correct beam-induced motion in movies
   - Required: movies_job_uid (from import_movies)
   - Optional: binning, patch_size, project_uid, workspace_uid
   
3. **CTF Estimation**: Estimate Contrast Transfer Function parameters
   - Required: micrographs_job_uid (from motion_correction)
   - Optional: min_res, max_res, project_uid, workspace_uid
   
4. **Micrograph Selection**: Filter micrographs based on quality metrics
   - Required: ctf_job_uid (from ctf_estimation)
   - Optional: min_resolution, project_uid, workspace_uid

## ReAct Process:
For each step, you MUST follow this pattern:

**Thought**: [Your reasoning about what needs to be done, why, and how]
**Action**: [The specific tool to use with exact parameters]
**Observation**: [What happened as a result of the action]

## CRITICAL: Job Monitoring and Waiting
- After starting any job, you MUST wait for it to complete using wait_for_job
- Do NOT proceed to the next step until the current job is completed
- If a job fails, report the error and stop the workflow

## Tool Usage Guidelines:
- import_movies: Start the import, then wait for completion
- motion_correction: Requires movies_job_uid from completed import_movies job
- ctf_estimation: Requires micrographs_job_uid from completed motion_correction job
- micrograph_selection: Requires ctf_job_uid from completed ctf_estimation job
- get_job_status: Check status of a specific job (use job UID only, e.g., "J81")
- wait_for_job: Wait for job completion (use job UID only, e.g., "J81")
- reason_about_workflow: Analyze current preprocessing state and dependencies

## Job UID Format:
- Job UIDs are strings like "J81", "J82", etc.
- When calling get_job_status or wait_for_job, pass ONLY the job UID
- Do NOT use JSON format or complex parameters for these tools

## Workflow Dependencies:
1. Import movies → Motion correction → CTF estimation → Micrograph selection
2. Each step must complete successfully before the next can begin
3. Always verify job completion before proceeding

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}
- Movies Path: {self.config.workflow.movies_path}
- Pixel Size: {self.config.workflow.pixel_size} Å
- Voltage: {self.config.workflow.voltage} kV

Remember: Always follow the Thought → Action → Observation pattern and WAIT for each job to complete!"""
    
    # Tool implementation methods
    def _import_movies_tool(self, input_str: str) -> str:
        """Tool wrapper for importing movies."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "movies_path": params.get("movies_path", self.config.workflow.movies_path),
                "gain_ref_path": params.get("gain_ref_path", self.config.workflow.gain_ref_path),
                "pixel_size": float(params.get("pixel_size", self.config.workflow.pixel_size)),
                "voltage": float(params.get("voltage", self.config.workflow.voltage)),
                "cs_mm": float(params.get("cs_mm", self.config.workflow.cs_mm)),
                "dose": float(params.get("dose", self.config.workflow.dose)),
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            result = self.cryosparc_tools.import_movies(**used_params)
            self._record_tool_execution("import_movies", used_params, result=result)
            return f"✅ Successfully queued import movies job: {result['job_uid']}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("import_movies", context, error=str(e))
            return f"❌ Error importing movies: {str(e)}"
    
    def _motion_correction_tool(self, input_str: str) -> str:
        """Tool wrapper for motion correction."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "movies_job_uid": params.get("movies_job_uid"),
                "binning": int(params.get("binning", self.config.workflow.motion_correction_binning)),
                "patch_size": int(params.get("patch_size", self.config.workflow.motion_correction_patch_size)),
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            result = self.cryosparc_tools.motion_correction(**used_params)
            self._record_tool_execution("motion_correction", used_params, result=result)
            return f"✅ Successfully queued motion correction job: {result['job_uid']}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("motion_correction", context, error=str(e))
            return f"❌ Error starting motion correction: {str(e)}"
    
    def _ctf_estimation_tool(self, input_str: str) -> str:
        """Tool wrapper for CTF estimation."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "micrographs_job_uid": params.get("micrographs_job_uid"),
                "min_res": float(params.get("min_res", self.config.workflow.ctf_min_res)),
                "max_res": float(params.get("max_res", self.config.workflow.ctf_max_res)),
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            result = self.cryosparc_tools.ctf_estimation(**used_params)
            self._record_tool_execution("ctf_estimation", used_params, result=result)
            return f"✅ Successfully queued CTF estimation job: {result['job_uid']}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("ctf_estimation", context, error=str(e))
            return f"❌ Error starting CTF estimation: {str(e)}"
    
    def _micrograph_selection_tool(self, input_str: str) -> str:
        """Tool wrapper for micrograph selection."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "ctf_job_uid": params.get("ctf_job_uid"),
                "min_resolution": float(params.get("min_resolution", 5.0)),
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            result = self.cryosparc_tools.micrograph_selection(**used_params)
            self._record_tool_execution("micrograph_selection", used_params, result=result)
            return f"✅ Successfully queued micrograph selection job: {result['job_uid']}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("micrograph_selection", context, error=str(e))
            return f"❌ Error starting micrograph selection: {str(e)}"
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about preprocessing workflow state."""
        try:
            reasoning = f"""
🤔 **Preprocessing Workflow Analysis**:

**Current State**: {input_str}

**Workflow Dependencies**:
1. Import Movies → Motion Correction → CTF Estimation → Micrograph Selection
2. Each step must complete before the next can begin

**Next Steps Analysis**:
- If no jobs are running: Start with import_movies
- If import job is running: Wait for completion, then start motion_correction
- If motion correction is running: Wait for completion, then start ctf_estimation
- If CTF estimation is running: Wait for completion, then start micrograph_selection
- If micrograph selection is running: Wait for completion, preprocessing is done

**Recommended Actions**:
- Always check job status before proceeding
- Use wait_for_job for critical dependencies
- Verify each step completes successfully before moving to the next
"""
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": reasoning})
            return reasoning
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error in workflow reasoning: {str(e)}"

