"""MCP tools for CryoSPARC polish operations."""

from typing import Dict, Any, Optional
from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class PolishTools:
    """Factory for creating polish-specific MCP tools."""
    
    @staticmethod
    def create_homogeneous_refinement_tool(agent) -> StructuredTool:
        """Create tool for homogeneous refinement with CTF refinement."""
        class HomogeneousRefinementInput(BaseModel):
            particles_job_uid: str = Field(description="UID of particles job")
            volume_job_uid: str = Field(description="UID of volume job")
            refinement_resolution: Optional[float] = Field(default=None, description="Target resolution in Angstroms")
            symmetry: Optional[str] = Field(default=None, description="Symmetry group (e.g., C1, C2, D7)")
            refine_defocus_refine: Optional[bool] = Field(default=True, description="Enable local CTF refinement (defocus)")
            refine_ctf_global_refine: Optional[bool] = Field(default=True, description="Enable global CTF refinement")
            refine_do_init_scale_est: Optional[bool] = Field(default=True, description="Enable initial scale estimation")
            refine_highpass_res: Optional[float] = Field(default=None, description="High-pass filter resolution in Angstroms")
            refine_num_final_iterations: Optional[int] = Field(default=None, description="Number of final refinement iterations")
            refine_res_init: Optional[float] = Field(default=None, description="Initial resolution for refinement in Angstroms")
            refine_symmetry_do_align: Optional[bool] = Field(default=True, description="Enable symmetry alignment")
            particles_group_name: Optional[str] = Field(default=None, description="Specific particles group name to use (e.g., particles_0)")
            project_uid: Optional[str] = Field(default="", description="Optional project UID")
            workspace_uid: Optional[str] = Field(default="", description="Optional workspace UID")
            wait_for_completion: Optional[bool] = Field(default=False, description="Whether to wait for job completion")
            timeout: Optional[int] = Field(default=None, description="Maximum time to wait for completion in seconds")
            check_interval: Optional[int] = Field(default=None, description="Time between status checks in seconds")
        
        def homogeneous_refinement_wrapper(
            particles_job_uid: str,
            volume_job_uid: str,
            refinement_resolution: Optional[float] = None,
            symmetry: Optional[str] = None,
            refine_defocus_refine: Optional[bool] = True,
            refine_ctf_global_refine: Optional[bool] = True,
            refine_do_init_scale_est: Optional[bool] = True,
            refine_highpass_res: Optional[float] = None,
            refine_num_final_iterations: Optional[int] = None,
            refine_res_init: Optional[float] = None,
            refine_symmetry_do_align: Optional[bool] = True,
            particles_group_name: Optional[str] = None,
            project_uid: str = "",
            workspace_uid: str = "",
            wait_for_completion: bool = False,
            timeout: Optional[int] = None,
            check_interval: Optional[int] = None
        ) -> str:
            """Wrapper to convert structured input to JSON string format."""
            import json
            params = {
                "particles_job_uid": particles_job_uid,
                "volume_job_uid": volume_job_uid
            }
            if refinement_resolution is not None:
                params["refinement_resolution"] = refinement_resolution
            if symmetry:
                params["symmetry"] = symmetry
            if refine_defocus_refine is not None:
                params["refine_defocus_refine"] = str(refine_defocus_refine).lower()
            if refine_ctf_global_refine is not None:
                params["refine_ctf_global_refine"] = str(refine_ctf_global_refine).lower()
            if refine_do_init_scale_est is not None:
                params["refine_do_init_scale_est"] = str(refine_do_init_scale_est).lower()
            if refine_highpass_res is not None:
                params["refine_highpass_res"] = refine_highpass_res
            if refine_num_final_iterations is not None:
                params["refine_num_final_iterations"] = refine_num_final_iterations
            if refine_res_init is not None:
                params["refine_res_init"] = refine_res_init
            if refine_symmetry_do_align is not None:
                params["refine_symmetry_do_align"] = str(refine_symmetry_do_align).lower()
            if particles_group_name:
                params["particles_group_name"] = particles_group_name
            if project_uid:
                params["project_uid"] = project_uid
            if workspace_uid:
                params["workspace_uid"] = workspace_uid
            if wait_for_completion is not None:
                params["wait_for_completion"] = str(wait_for_completion).lower()
            if timeout is not None:
                params["timeout"] = timeout
            if check_interval is not None:
                params["check_interval"] = check_interval
            json_input = json.dumps(params)
            return agent._homogeneous_refinement_tool(json_input)
        
        return StructuredTool.from_function(
            func=homogeneous_refinement_wrapper,
            name="homogeneous_refinement",
            description="Refine a single 3D structure with local and global CTF refinement enabled. "
                       "Required parameters: particles_job_uid, volume_job_uid. "
                       "Optional parameters: refinement_resolution, symmetry, "
                       "refine_defocus_refine (enable local CTF refinement, default: True), "
                       "refine_ctf_global_refine (enable global CTF refinement, default: True), "
                       "refine_do_init_scale_est, refine_highpass_res, refine_num_final_iterations, "
                       "refine_res_init, refine_symmetry_do_align, particles_group_name, "
                       "project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            args_schema=HomogeneousRefinementInput
        )
    
    @staticmethod
    def create_reference_motion_correction_tool(agent) -> Tool:
        """Create tool for reference-based motion correction."""
        return Tool(
            name="reference_motion_correction",
            description="Run reference-based motion correction on particles using a reference volume. "
                       "Required parameters: micrographs_job_uid, particles_job_uid, volume_job_uid. "
                       "Optional parameters: All reference_motion_correction job parameters can be passed via kwargs. "
                       "project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._reference_motion_correction_tool
        )
    
    @staticmethod
    def create_get_job_status_tool(agent) -> Tool:
        """Create tool for checking job status."""
        return Tool(
            name="get_job_status",
            description="Check the status of a CryoSPARC job. Required parameters: job_uid.",
            func=agent._get_job_status_tool
        )
    
    @staticmethod
    def create_wait_for_job_tool(agent) -> Tool:
        """Create tool for waiting for job completion."""
        return Tool(
            name="wait_for_job",
            description="Wait for a job to complete and return final status. "
                       "Required parameters: job_uid. Optional parameters: timeout.",
            func=agent._wait_for_job_tool
        )
    
    @staticmethod
    def create_verify_inputs_tool(agent) -> Tool:
        """Create tool for verifying optimization and preprocessing results."""
        return Tool(
            name="verify_inputs",
            description="Verify that optimization and preprocessing stages are complete and read required job UIDs. "
                       "This checks for optimization_results_cryosparc_*.json and preprocessing_results_cryosparc_*.json files. "
                       "No parameters required.",
            func=agent._verify_inputs_tool
        )


