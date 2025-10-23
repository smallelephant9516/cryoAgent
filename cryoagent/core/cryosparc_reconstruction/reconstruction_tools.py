"""MCP tools for CryoSPARC 3D reconstruction operations."""

from typing import Dict, Any
from langchain.tools import Tool


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
    def create_homogeneous_refinement_tool(agent) -> Tool:
        """Create tool for homogeneous refinement."""
        return Tool(
            name="homogeneous_refinement",
            description="Refine a single 3D structure with all particles. "
                       "Required parameters: particles_job_uid, volume_job_uid (from ab initio). "
                       "Optional parameters: refinement_resolution (target resolution in Å), symmetry, "
                       "refine_do_init_scale_est (enable initial scale estimation), "
                       "refine_highpass_res (high-pass filter resolution in Å), "
                       "refine_num_final_iterations (number of final iterations), "
                       "refine_res_init (initial resolution in Å), "
                       "refine_symmetry_do_align (enable symmetry alignment), "
                       "project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._homogeneous_refinement_tool
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

