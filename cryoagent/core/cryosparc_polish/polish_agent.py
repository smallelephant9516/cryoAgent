"""ReAct-based polish agent for final refinement after optimization."""

import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..base_react_agent import BaseReActAgent
from .polish_tools import PolishTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...config.config_loader import CryoAgentConfig


class PolishAgent(BaseReActAgent):
    """ReAct-based agent for final polish refinement operations."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the polish agent.
        
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
        stage_defaults = self.stage_config.get("microscope_parameters", {})
        
        # Now call super().__init__() which will call _create_tools()
        super().__init__(cryosparc_tools, config, llm)
        
        # Set microscope_config after super().__init__() since it uses methods from base class
        self.microscope_config = self._resolve_microscope_defaults(stage_defaults, update_cache=True)
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load polish stage configuration."""
        config_path = Path("configs/cryosparc/polish_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _load_reconstruction_config(self) -> Dict[str, Any]:
        """Load reconstruction stage configuration to get symmetry."""
        config_path = Path("configs/cryosparc/reconstruction_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _get_refinement_symmetry(self) -> str:
        """Get symmetry from reconstruction_config.json for homogeneous refinement."""
        recon_config = self._load_reconstruction_config()
        
        # First try to get from workflow.refinement.symmetry (preferred)
        refinement_symmetry = recon_config.get("workflow", {}).get("refinement", {}).get("symmetry")
        if refinement_symmetry:
            return refinement_symmetry
        
        # Fall back to workflow.ab_initio.symmetry
        ab_initio_symmetry = recon_config.get("workflow", {}).get("ab_initio", {}).get("symmetry")
        if ab_initio_symmetry:
            return ab_initio_symmetry
        
        # Fall back to microscope_parameters.symmetry
        microscope_symmetry = recon_config.get("microscope_parameters", {}).get("symmetry")
        if microscope_symmetry:
            return microscope_symmetry
        
        # Fall back to polish config
        polish_symmetry = self._get_stage_param("polish", "initial_refinement", {}).get("symmetry")
        if polish_symmetry:
            return polish_symmetry
        
        # Default to C1 if not found
        return "C1"
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Fetch a parameter from the stage workflow configuration."""
        return self.stage_workflow.get(section, {}).get(key, default)
    
    def _create_tools(self) -> List[Tool]:
        """Create polish-specific tools."""
        return [
            PolishTools.create_homogeneous_refinement_tool(self),
            PolishTools.create_reference_motion_correction_tool(self),
            PolishTools.create_get_job_status_tool(self),
            PolishTools.create_wait_for_job_tool(self),
            PolishTools.create_verify_inputs_tool(self)
        ]
    
    def _get_react_system_prompt(self) -> str:
        """Get the polish-specific ReAct system prompt."""
        return f"""You are a CryoEM polish refinement assistant using the ReAct (Reasoning + Acting) framework.
You specialize in final refinement steps after optimization to achieve the best possible resolution.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Polish Workflow Steps:

### Step 1: Verify Inputs
1. Use `verify_inputs` tool to check that optimization and preprocessing are complete
2. This will read the optimization JSON file to get `best_job_uid` (from optimization)
3. This will read the preprocessing JSON file to get `final_micrographs_job_uid`
4. Verify that all required job UIDs exist and are accessible

### Step 2: Initial Homogeneous Refinement with CTF Refinement
1. Run `homogeneous_refinement` with:
   - particles_job_uid: from best optimization job (best_job_uid)
   - volume_job_uid: from best optimization job (best_job_uid)
   - refine_defocus_refine: true (enable local CTF refinement)
   - refine_ctf_global_refine: true (enable global CTF refinement)
   - Wait for completion and record job UID (J-A)

### Step 3: Reference Motion Correction
1. Run `reference_motion_correction` with:
   - micrographs_job_uid: from preprocessing JSON (final_micrographs_job_uid)
   - particles_job_uid: from Step 2 (J-A)
   - volume_job_uid: from Step 2 (J-A)
   - Wait for completion and record job UID (J-B)
   - The output particles group name is "particles_0"

### Step 4: Final Homogeneous Refinement with CTF Refinement
1. Run `homogeneous_refinement` with:
   - particles_job_uid: from Step 3 (J-B), use group name "particles_0"
   - volume_job_uid: from Step 2 (J-A) (use the volume from initial refinement, not from local motion correction)
   - refine_defocus_refine: true (enable local CTF refinement)
   - refine_ctf_global_refine: true (enable global CTF refinement)
   - Wait for completion and record job UID (J-C)
   - This is the final best job UID

## ReAct Process:
For each step, you MUST follow this pattern:

**Thought**: [Your reasoning about what needs to be done, why, and how]
**Action**: [The specific tool to use with exact parameters]
**Observation**: [What happened as a result of the action]

## CRITICAL: Job Monitoring and Failure Recovery
- After starting ANY job, you MUST wait for it to complete
- Use wait_for_job with the job UID to wait for completion
- Do NOT proceed to the next step until the current job is completed
- Polish refinement jobs can take significant time

## Tool Usage Guidelines:

- **verify_inputs**: Verify optimization and preprocessing are complete
  * No parameters required
  * Returns: best_job_uid (from optimization), final_micrographs_job_uid (from preprocessing)
  * Use this FIRST before starting any refinement steps
  
- **homogeneous_refinement**: Refine single structure with CTF refinement
  * Required: particles_job_uid, volume_job_uid
  * Optional: refinement_resolution, symmetry
  * IMPORTANT: Set refine_defocus_refine=true and refine_ctf_global_refine=true for CTF refinement
  * Optional: particles_group_name (e.g., "particles_0" for motion correction output)
  * Start the job, then wait for completion
  
- **reference_motion_correction**: Run reference-based motion correction
  * Required: micrographs_job_uid, particles_job_uid, volume_job_uid
  * Optional: All reference_motion_correction parameters can be passed
  * Start the job, then wait for completion
  * Output particles group name is "particles_0"
  
- **get_job_status**: Check status of a specific job (use job UID only, e.g., "J113")
- **wait_for_job**: Wait for job completion (use job UID only, e.g., "J113")

## Job UID Format:
- Job UIDs are strings like "J113", "J114", etc.
- When calling get_job_status or wait_for_job, pass ONLY the job UID
- Do NOT use JSON format or complex parameters for these tools

## Workflow Dependencies:
1. Step 1: Verify inputs (must complete successfully)
2. Step 2: Initial refinement (J-A) must complete before Step 3
3. Step 3: Motion correction (J-B) must complete before Step 4
4. Step 4: Final refinement (J-C) is the final output
5. Each step must complete successfully before the next can begin
6. Always verify job completion before proceeding

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}

## Example Workflow:

**Complete Polish Workflow**:
1. verify_inputs → Get best_job_uid and final_micrographs_job_uid
2. homogeneous_refinement (best_job_uid particles + volume, CTF enabled) → Wait → J-A
3. reference_motion_correction (final_micrographs_job_uid, J-A particles, J-A volume) → Wait → J-B
4. homogeneous_refinement (J-B particles_0, J-A volume, CTF enabled) → Wait → J-C
5. J-C is the final best job UID

Remember: Always follow the Thought → Action → Observation pattern and WAIT for each job to complete!
Polish refinement jobs can take significant time, especially with CTF refinement enabled."""
    
    def _homogeneous_refinement_tool(self, tool_input: str) -> str:
        """Execute homogeneous refinement with CTF refinement enabled."""
        try:
            params = dict(self._parse_tool_input(tool_input))

            print(f"Homogeneous refinement params: {params}")
            
            particles_job_uid = params.get("particles_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            
            if not particles_job_uid or not volume_job_uid:
                missing = []
                if not particles_job_uid:
                    missing.append("particles_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}"
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Extract CTF refinement parameters
            refine_defocus_refine = params.get("refine_defocus_refine", "true").lower() == "true"
            refine_ctf_global_refine = params.get("refine_ctf_global_refine", "true").lower() == "true"
            
            # Extract other parameters
            refinement_resolution = params.get("refinement_resolution")
            # Get symmetry from reconstruction config if not explicitly provided
            symmetry = params.get("symmetry")
            if not symmetry:
                symmetry = self._get_refinement_symmetry()
            if not symmetry:
                symmetry = "C1"
            refine_do_init_scale_est = params.get("refine_do_init_scale_est", "true").lower() == "true"
            refine_highpass_res = params.get("refine_highpass_res", None)
            refine_num_final_iterations = params.get("refine_num_final_iterations", None)
            refine_res_init = params.get("refine_res_init", None)
            refine_symmetry_do_align = params.get("refine_symmetry_do_align", "true").lower() == "true"
            
            # Handle particles_group_name if specified (for motion correction output)
            particles_group_name = params.get("particles_group_name")
            
            wait_for_completion = params.get("wait_for_completion", "false").lower() == "true"
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            # Execute homogeneous refinement with CTF parameters
            result = self.cryosparc_tools.homogeneous_refinement(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particles_job_uid=particles_job_uid,
                volume_job_uid=volume_job_uid,
                refinement_resolution=refinement_resolution,
                symmetry=symmetry,
                refine_do_init_scale_est=refine_do_init_scale_est,
                refine_highpass_res=refine_highpass_res,
                refine_num_final_iterations=refine_num_final_iterations,
                refine_res_init=refine_res_init,
                refine_symmetry_do_align=refine_symmetry_do_align,
                refine_defocus_refine=refine_defocus_refine,
                refine_ctf_global_refine=refine_ctf_global_refine,
                particles_group_name=particles_group_name,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )
            
            self._record_tool_execution("homogeneous_refinement", params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("homogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _reference_motion_correction_tool(self, tool_input: str) -> str:
        """Execute reference-based motion correction."""
        try:
            params = self._parse_tool_input(tool_input)
            
            micrographs_job_uid = params.get("micrographs_job_uid")
            particles_job_uid = params.get("particles_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            
            if not micrographs_job_uid or not particles_job_uid or not volume_job_uid:
                missing = []
                if not micrographs_job_uid:
                    missing.append("micrographs_job_uid")
                if not particles_job_uid:
                    missing.append("particles_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}"
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            wait_for_completion = params.get("wait_for_completion", "false").lower() == "true"
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            # Extract any additional parameters for reference_motion_correction
            motion_correction_params = {k: v for k, v in params.items() 
                                      if k not in ["micrographs_job_uid", "particles_job_uid", "volume_job_uid",
                                                   "project_uid", "workspace_uid", "wait_for_completion", 
                                                   "timeout", "check_interval"]}
            
            result = self.cryosparc_tools.reference_motion_correction(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                micrographs_job_uid=micrographs_job_uid,
                particles_job_uid=particles_job_uid,
                volume_job_uid=volume_job_uid,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval,
                **motion_correction_params
            )
            
            self._record_tool_execution("reference_motion_correction", params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("reference_motion_correction", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _verify_inputs_tool(self, tool_input: str) -> str:
        """Verify optimization and preprocessing results exist and read job UIDs."""
        try:
            outputs_path = Path("outputs")
            if not outputs_path.exists():
                return json.dumps({
                    "success": False,
                    "error": "Outputs directory does not exist"
                })
            
            # Find optimization results file
            optimization_files = list(outputs_path.glob("**/optimization_results_cryosparc_*.json"))
            if not optimization_files:
                return json.dumps({
                    "success": False,
                    "error": "No optimization results file found"
                })
            
            latest_optimization = max(optimization_files, key=lambda f: f.stat().st_mtime)
            with open(latest_optimization, 'r') as f:
                optimization_data = json.load(f)
            
            if optimization_data.get("status") != "completed":
                return json.dumps({
                    "success": False,
                    "error": f"Optimization not completed. Status: {optimization_data.get('status')}"
                })
            
            best_job_uid = optimization_data.get("best_job_uid")
            if not best_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "best_job_uid not found in optimization results"
                })
            
            # Find preprocessing results file
            preprocessing_files = list(outputs_path.glob("**/preprocessing_results_cryosparc_*.json"))
            if not preprocessing_files:
                return json.dumps({
                    "success": False,
                    "error": "No preprocessing results file found"
                })
            
            latest_preprocessing = max(preprocessing_files, key=lambda f: f.stat().st_mtime)
            with open(latest_preprocessing, 'r') as f:
                preprocessing_data = json.load(f)
            
            if preprocessing_data.get("status") != "completed":
                return json.dumps({
                    "success": False,
                    "error": f"Preprocessing not completed. Status: {preprocessing_data.get('status')}"
                })
            
            final_micrographs_job_uid = preprocessing_data.get("final_micrographs_job_uid")
            if not final_micrographs_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "final_micrographs_job_uid not found in preprocessing results"
                })
            
            result = {
                "success": True,
                "best_job_uid": best_job_uid,
                "final_micrographs_job_uid": final_micrographs_job_uid,
                "optimization_file": str(latest_optimization),
                "preprocessing_file": str(latest_preprocessing),
                "message": "All inputs verified successfully"
            }
            
            self._record_tool_execution("verify_inputs", {}, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("verify_inputs", {}, error=str(e))
            return json.dumps(error_result)


