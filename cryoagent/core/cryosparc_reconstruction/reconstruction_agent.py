"""ReAct-based 3D reconstruction agent for CryoEM data processing."""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..base_react_agent import BaseReActAgent
from .reconstruction_tools import ReconstructionTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


class ReconstructionAgent(BaseReActAgent):
    """ReAct-based agent for CryoEM 3D reconstruction operations."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the 3D reconstruction agent.
        
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
    
    def update_workflow_defaults(self, defaults: Dict[str, Any]) -> None:
        """Store workflow-level default parameters for later tool invocations."""
        if defaults:
            if not hasattr(self, "workflow_defaults") or self.workflow_defaults is None:
                self.workflow_defaults = {}
            self.workflow_defaults.update(defaults)
    
    def _create_tools(self) -> List[Tool]:
        """Create 3D reconstruction-specific tools."""
        return [
            ReconstructionTools.create_ab_initio_tool(self),
            ReconstructionTools.create_homogeneous_refinement_tool(self),
            ReconstructionTools.create_heterogeneous_refinement_tool(self),
            ReconstructionTools.create_get_job_status_tool(self),
            ReconstructionTools.create_wait_for_job_tool(self),
            ReconstructionTools.create_get_job_log_tool(self),
            ReconstructionTools.create_reason_about_workflow_tool(self)
        ]
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load reconstruction stage configuration."""
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
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Fetch a parameter from the stage workflow configuration."""
        return self.stage_workflow.get(section, {}).get(key, default)
    
    def _get_system_prompt_context(self) -> Dict[str, Any]:
        """Build template variables for cryosparc/reconstruction/system.md."""
        return {
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
        }

    def _get_react_system_prompt(self) -> str:
        """Get the 3D reconstruction-specific ReAct system prompt."""
        return load_prompt(
            "cryosparc/reconstruction/system.md",
            self._get_system_prompt_context(),
        )
    
    # =================================================================
    # Tool Implementation Methods
    # =================================================================
    
    def _ab_initio_tool(self, tool_input: str) -> str:
        """Execute ab initio reconstruction."""
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            # Support both "particles_job_uid" and "job_uid" for flexibility
            particles_job_uid = params.get("particles_job_uid") or params.get("job_uid")
            if not particles_job_uid and "__arg1" in params:
                arg_value = str(params.get("__arg1", "")).strip()
                if arg_value:
                    particles_job_uid = arg_value.split(",")[0].strip()
            if not particles_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: particles_job_uid or job_uid"
                })
            
            # Get project and workspace UIDs
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Extract optional parameters (stage config defaults first)
            ab_initio_defaults = self.stage_workflow.get("ab_initio", {})
            num_classes = params.get("num_classes", ab_initio_defaults.get("num_classes", 1))
            initial_resolution = params.get("initial_resolution", ab_initio_defaults.get("initial_resolution", 20.0))
            final_resolution = params.get("final_resolution", ab_initio_defaults.get("final_resolution", 10.0))
            max_iterations = params.get("max_iterations", ab_initio_defaults.get("max_iterations", 50))
            symmetry = params.get("symmetry")
            if not symmetry:
                symmetry = self._get_microscope_parameter("symmetry") or ab_initio_defaults.get("symmetry") or "C1"
            params["symmetry"] = symmetry

            scaled_diameter = self._get_scaled_particle_diameter(1.2)
            if scaled_diameter:
                params["particle_diameter_angstroms"] = float(scaled_diameter)
                if hasattr(self, "workflow_defaults") and isinstance(self.workflow_defaults, dict):
                    self.workflow_defaults["ab_initio_particle_diameter"] = float(scaled_diameter)
            
            # Job control parameters
            wait_for_completion = params.get("wait_for_completion", "false").lower() == "true"
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            # Execute ab initio reconstruction
            result = self.cryosparc_tools.ab_initio_reconstruction(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particles_job_uid=particles_job_uid,
                num_classes=num_classes,
                initial_resolution=initial_resolution,
                final_resolution=final_resolution,
                max_iterations=max_iterations,
                symmetry=symmetry,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )
            
            # Log the tool execution
            self._record_tool_execution("ab_initio_reconstruction", params, result=result)
            
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("ab_initio_reconstruction", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _homogeneous_refinement_tool(self, tool_input: str) -> str:
        """Execute homogeneous refinement."""
        
        print(f"Homogeneous refinement tool input: {tool_input}")

        try:
            params = dict(self._parse_tool_input(tool_input))

            print(f"Homogeneous refinement params: {params}")
            # Extract required parameters
            particles_job_uid = params.get("particles_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            
            # Support legacy single job_uid parameter (treat as volume_job_uid for ab initio case)
            if not volume_job_uid:
                volume_job_uid = params.get("job_uid")
            
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
            
            # CRITICAL VALIDATION: particles_job_uid and volume_job_uid should be DIFFERENT
            if particles_job_uid == volume_job_uid:
                error_msg = (
                    f"ERROR: particles_job_uid and volume_job_uid cannot be the same (both are '{particles_job_uid}'). "
                    f"For homogeneous refinement:\n"
                    f"  - particles_job_uid should be from the ORIGINAL input (Select 2D job or import particle job)\n"
                    f"  - volume_job_uid should be from the ab initio reconstruction job\n"
                    f"Example: If ab initio used particles from J100 and produced volume in J138, "
                    f"use particles_job_uid=J100, volume_job_uid=J138"
                )
                print(f"⚠️  {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": error_msg
                })
            
            # Get project and workspace UIDs
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Extract optional parameters
            defaults = getattr(self, "workflow_defaults", {}) or {}
            refinement_resolution = params.get("refinement_resolution")
            if refinement_resolution in ("", None):
                refinement_resolution = defaults.get("refinement_resolution")
            symmetry = params.get("symmetry")
            if not symmetry:
                symmetry = self._get_microscope_parameter("symmetry")
            if not symmetry:
                symmetry = defaults.get("refinement_symmetry")
            if not symmetry:
                symmetry = "C1"
            params["symmetry"] = symmetry
            if refinement_resolution is not None:
                params["refinement_resolution"] = refinement_resolution
            
            # Advanced refinement parameters
            refine_do_init_scale_est = params.get("refine_do_init_scale_est", "true").lower() == "true"
            refine_highpass_res = params.get("refine_highpass_res", None)
            refine_num_final_iterations = params.get("refine_num_final_iterations", None)
            refine_res_init = params.get("refine_res_init", None)
            refine_symmetry_do_align = params.get("refine_symmetry_do_align", "true").lower() == "true"
            
            # CTF refinement parameters - extract from params if provided, otherwise default to False
            # (to match the agent's intent when it says CTF refinement is disabled)
            refine_defocus_refine = params.get("refine_defocus_refine")
            if refine_defocus_refine is not None:
                refine_defocus_refine = str(refine_defocus_refine).lower() == "true"
            else:
                refine_defocus_refine = False  # Default to False when not specified
            
            refine_ctf_global_refine = params.get("refine_ctf_global_refine")
            if refine_ctf_global_refine is not None:
                refine_ctf_global_refine = str(refine_ctf_global_refine).lower() == "true"
            else:
                refine_ctf_global_refine = False  # Default to False when not specified
            
            # Job control parameters
            wait_for_completion = params.get("wait_for_completion", "false").lower() == "true"
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            # Execute homogeneous refinement
            result = self.cryosparc_tools.homogeneous_refinement(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particles_job_uid=particles_job_uid,
                particles_group_name='particles_selected',
                volume_group_name='volume_class_0',
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
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )
            
            # Log the tool execution
            self._record_tool_execution("homogeneous_refinement", params, result=result)
            
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("homogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _heterogeneous_refinement_tool(self, tool_input: str) -> str:
        """Execute heterogeneous refinement."""
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            particles_job_uid = params.get("particles_job_uid")
            volume_job_uids = params.get("volume_job_uids")
            
            if not particles_job_uid or not volume_job_uids:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameters: particles_job_uid and volume_job_uids"
                })
            
            # Ensure volume_job_uids is a list
            if isinstance(volume_job_uids, str):
                volume_job_uids = [v.strip() for v in volume_job_uids.split(",")]
            
            # Get project and workspace UIDs
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Extract optional parameters
            num_classes = params.get("num_classes", len(volume_job_uids))
            
            # Job control parameters
            wait_for_completion = params.get("wait_for_completion", "false").lower() == "true"
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            # Execute heterogeneous refinement
            result = self.cryosparc_tools.heterogeneous_refinement(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particles_job_uid=particles_job_uid,
                volume_job_uids=volume_job_uids,
                num_classes=num_classes,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )
            
            # Log the tool execution
            self._record_tool_execution("heterogeneous_refinement", params, result=result)
            
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("heterogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about 3D reconstruction workflow state."""
        try:
            reasoning = f"""
🤔 **3D Reconstruction Workflow Analysis**:

**Current State**: {input_str}

**Workflow Dependencies**:
1. Initial Model Generation:
   - Ab Initio Reconstruction (requires particles from 2D class selection or extraction)
   - OR Homogeneous Reconstruction (alternative to ab initio for homogeneous datasets)
2. Optional: Homogeneous Refinement (requires volume + particles)
3. Optional: Heterogeneous Refinement (requires multiple volumes + particles)

**Ab Initio Reconstruction Parameters**:
- **num_classes**: 1 for homogeneous, 2-4 for heterogeneous datasets
- **initial_resolution**: Start at 20-30 Å for stability
- **final_resolution**: Target 8-12 Å for ab initio (don't go < 8 Å)
- **symmetry**: C1 (no symmetry) is safest default
- **max_iterations**: 50 is typical, 100 for difficult cases

**Homogeneous Reconstruction Parameters**:
- **initial_resolution**: Start at 20-30 Å for stability
- **final_resolution**: Target 8.0 Å (can be more aggressive than ab initio)
- **symmetry**: C1 (no symmetry) is safest default
- Often faster and more robust than ab initio for single structure datasets

**Homogeneous Refinement Parameters**:
- **particles_job_uid**: From ORIGINAL input - Select 2D job or import particle job (the SAME particles_job_uid that was used for ab initio reconstruction)
- **volume_job_uid**: From ab initio reconstruction job (the ab initio job UID that produced the initial 3D volume)
- **refinement_resolution**: Target resolution in Angstroms (optional)
- **symmetry**: C1 (no symmetry) is safest default
- **CRITICAL**: particles_job_uid and volume_job_uid must be DIFFERENT
  * Example: If ab initio used particles from J100 and produced volume in J138, use particles_job_uid=J100, volume_job_uid=J138
  * DO NOT use the same job UID for both parameters

**Next Steps Analysis**:
- If no reconstruction jobs are running: Choose between:
  * ab_initio_reconstruction (standard approach, supports multiple classes)
  * homogeneous_refinement (faster alternative for single structure)
- If reconstruction is running: Wait for completion, then assess results
- If initial model completed successfully: Decide on refinement strategy
  * Single good class → homogeneous refinement
  * Multiple distinct classes → heterogeneous refinement
  * Poor results → try alternative method or re-run with adjusted parameters

**Recommended Actions**:
- Always wait for ab initio to complete before refinement
- Check FSC curves in CryoSPARC to assess resolution
- Review 3D volumes visually before proceeding
- Use conservative parameters initially
- Save refinement for validated structures

**Common Issues**:
- Ab initio fails to converge: Increase initial_resolution, reduce num_classes
- Multiple classes look identical: Dataset may be homogeneous, use num_classes=1
- Low resolution: Expected for ab initio; refinement improves resolution
- Artifacts in volume: Check symmetry (try C1), verify particle quality
"""
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": reasoning})
            return reasoning
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error in workflow reasoning: {str(e)}"
    
