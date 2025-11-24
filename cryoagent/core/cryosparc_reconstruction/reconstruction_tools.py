"""MCP tools for CryoSPARC 3D reconstruction operations."""

from typing import Dict, Any, Optional
from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ReconstructionTools:
    """Factory for creating 3D reconstruction-specific MCP tools."""
    
    @staticmethod
    def create_ab_initio_tool(agent) -> Tool:
        """Create tool for ab initio reconstruction."""
        return Tool(
            name="ab_initio_reconstruction",
            description="Generate initial 3D models from 2D particles using ab initio reconstruction. "
                       "Required parameters: particles_job_uid (from 2D class selection or extraction). "
                       "Optional parameters: num_classes (number of 3D classes, default: 1), "
                       "initial_resolution (starting resolution in Å, default: 20.0), "
                       "final_resolution (target resolution in Å, default: 10.0), "
                       "max_iterations (default: 50), symmetry (default: C1), "
                       "project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._ab_initio_tool
        )
    
    @staticmethod
    def create_homogeneous_reconstruction_tool(agent) -> Tool:
        """Create tool for homogeneous reconstruction."""
        return Tool(
            name="homogeneous_reconstruction",
            description="Generate a 3D model from 2D particles using homogeneous reconstruction. "
                       "This is an alternative to ab initio that's often faster and more robust for homogeneous datasets. "
                       "Required parameters: particles_job_uid (from 2D class selection or extraction). "
                       "Optional parameters: initial_resolution (starting resolution in Å, default: 20.0), "
                       "final_resolution (target resolution in Å, default: 8.0), "
                       "symmetry (default: C1), project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._homogeneous_reconstruction_tool
        )
    
    @staticmethod
    def create_homogeneous_refinement_tool(agent) -> StructuredTool:
        """Create tool for homogeneous refinement."""
        # Define the input schema for StructuredTool
        class HomogeneousRefinementInput(BaseModel):
            particles_job_uid: str = Field(description="UID of particles job (from ab initio job, e.g., 'J425')")
            volume_job_uid: str = Field(description="UID of volume job (from ab initio job, e.g., 'J425'). Both particles and volume typically come from the same ab initio job.")
            refinement_resolution: Optional[float] = Field(default=None, description="Target resolution in Angstroms (optional)")
            symmetry: Optional[str] = Field(default=None, description="Symmetry group (e.g., C1, C2, D7, default: C1)")
            refine_do_init_scale_est: Optional[bool] = Field(default=True, description="Enable initial scale estimation")
            refine_highpass_res: Optional[float] = Field(default=None, description="High-pass filter resolution in Angstroms")
            refine_num_final_iterations: Optional[int] = Field(default=None, description="Number of final refinement iterations")
            refine_res_init: Optional[float] = Field(default=None, description="Initial resolution for refinement in Angstroms")
            refine_symmetry_do_align: Optional[bool] = Field(default=True, description="Enable symmetry alignment")
            project_uid: Optional[str] = Field(default="", description="Optional project UID")
            workspace_uid: Optional[str] = Field(default="", description="Optional workspace UID")
            wait_for_completion: Optional[bool] = Field(default=False, description="Whether to wait for job completion")
            timeout: Optional[int] = Field(default=None, description="Maximum time to wait for completion in seconds")
            check_interval: Optional[int] = Field(default=None, description="Time between status checks in seconds")
        
        # Create a wrapper function that converts structured input to JSON string format
        def homogeneous_refinement_wrapper(
            particles_job_uid: str,
            volume_job_uid: str,
            refinement_resolution: Optional[float] = None,
            symmetry: Optional[str] = None,
            refine_do_init_scale_est: Optional[bool] = True,
            refine_highpass_res: Optional[float] = None,
            refine_num_final_iterations: Optional[int] = None,
            refine_res_init: Optional[float] = None,
            refine_symmetry_do_align: Optional[bool] = True,
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
            # Convert to JSON string and call the original function
            json_input = json.dumps(params)
            return agent._homogeneous_refinement_tool(json_input)
        
        return StructuredTool.from_function(
            func=homogeneous_refinement_wrapper,
            name="homogeneous_refinement",
            description="Refine a single 3D structure with all particles. "
                       "Required parameters: particles_job_uid, volume_job_uid (both from ab initio job, typically the same job UID). "
                       "Optional parameters: refinement_resolution (target resolution in Å), symmetry, "
                       "refine_do_init_scale_est (enable initial scale estimation), "
                       "refine_highpass_res (high-pass filter resolution in Å), "
                       "refine_num_final_iterations (number of final iterations), "
                       "refine_res_init (initial resolution in Å), "
                       "refine_symmetry_do_align (enable symmetry alignment), "
                       "project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            args_schema=HomogeneousRefinementInput
        )
    
    @staticmethod
    def create_heterogeneous_refinement_tool(agent) -> Tool:
        """Create tool for heterogeneous refinement."""
        return Tool(
            name="heterogeneous_refinement",
            description="Simultaneously refine multiple 3D classes with particles. "
                       "Required parameters: particles_job_uid, volume_job_uids (list of initial volumes). "
                       "Optional parameters: num_classes (default: 3), project_uid, workspace_uid, "
                       "wait_for_completion, timeout, check_interval.",
            func=agent._heterogeneous_refinement_tool
        )
    
    @staticmethod
    def create_get_job_status_tool(agent) -> Tool:
        """Create tool for checking job status."""
        return Tool(
            name="get_job_status",
            description="Check the status of a CryoSPARC job. "
                       "Required parameters: job_uid.",
            func=agent._get_job_status_tool
        )
    
    @staticmethod
    def create_wait_for_job_tool(agent) -> Tool:
        """Create tool for waiting for job completion."""
        return Tool(
            name="wait_for_job",
            description="Wait for a job to complete and return final status. "
                       "Required parameters: job_uid. "
                       "Optional parameters: timeout.",
            func=agent._wait_for_job_tool
        )
    
    @staticmethod
    def create_get_job_log_tool(agent) -> Tool:
        """Create tool for reading job logs and analyzing errors."""
        return Tool(
            name="get_job_log",
            description="Read and analyze the log file of a CryoSPARC job to understand failures and get suggestions. "
                       "Required parameters: job_uid. "
                       "Optional parameters: project_uid, workspace_uid. "
                       "This tool helps diagnose why a job failed and provides suggestions for fixing the issues.",
            func=agent._get_job_log_tool
        )
    
    @staticmethod
    def create_reason_about_workflow_tool(agent) -> Tool:
        """Create tool for reasoning about 3D reconstruction workflow."""
        return Tool(
            name="reason_about_workflow",
            description="Analyze the current 3D reconstruction workflow state and determine next steps. "
                       "Use this to think through reconstruction parameters and job dependencies.",
            func=agent._reason_about_workflow_tool
        )

