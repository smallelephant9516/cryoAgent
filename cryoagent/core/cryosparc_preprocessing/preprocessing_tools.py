"""MCP tools for CryoSPARC preprocessing operations."""

from typing import Dict, Any
from langchain.tools import Tool


class PreprocessingTools:
    """Factory for creating preprocessing-specific MCP tools."""
    
    @staticmethod
    def create_import_movies_tool(agent) -> Tool:
        """Create tool for importing movies."""
        return Tool(
            name="import_movies",
            description="Import movie files into CryoSPARC for processing. "
                       "Required parameters: movies_path, pixel_size, voltage, cs_mm, dose. "
                       "Optional parameters: gain_ref_path, project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._import_movies_tool
        )
    
    @staticmethod
    def create_motion_correction_tool(agent) -> Tool:
        """Create tool for motion correction."""
        return Tool(
            name="motion_correction",
            description="Perform motion correction on imported movies. "
                       "Required parameters: movies_job_uid. "
                       "Optional parameters: binning, patch_size, max_shift, project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._motion_correction_tool
        )
    
    @staticmethod
    def create_ctf_estimation_tool(agent) -> Tool:
        """Create tool for CTF estimation."""
        return Tool(
            name="ctf_estimation",
            description="Estimate CTF parameters for micrographs. "
                       "Required parameters: micrographs_job_uid. "
                       "Optional parameters: min_res, max_res, defocus_range, project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._ctf_estimation_tool
        )
    
    @staticmethod
    def create_micrograph_selection_tool(agent) -> Tool:
        """Create tool for micrograph selection."""
        return Tool(
            name="micrograph_selection",
            description="Select micrographs with resolution better than specified threshold. "
                       "Required parameters: ctf_job_uid. "
                       "Optional parameters: min_resolution, project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._micrograph_selection_tool
        )
    
    @staticmethod
    def create_reason_about_workflow_tool(agent) -> Tool:
        """Create tool for reasoning about preprocessing workflow."""
        return Tool(
            name="reason_about_workflow",
            description="Analyze the current preprocessing workflow state and determine next steps. "
                       "Use this to think through the workflow progression and identify dependencies.",
            func=agent._reason_about_workflow_tool
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

