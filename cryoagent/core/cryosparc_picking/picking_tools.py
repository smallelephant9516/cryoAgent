"""MCP tools for CryoSPARC particle picking operations."""

from typing import Dict, Any
from langchain.tools import Tool


class PickingTools:
    """Factory for creating particle picking-specific MCP tools."""
    
    @staticmethod
    def create_blob_picker_tool(agent) -> Tool:
        """Create tool for blob picker particle detection."""
        return Tool(
            name="blob_picker",
            description="Detect and pick particles from micrographs using GPU-accelerated blob detection. "
                       "Required parameters: micrographs_job_uid, particle_diameter. "
                       "Optional parameters: diameter_max (default: 2x particle_diameter), project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._blob_picker_tool
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
    def create_extract_particles_tool(agent) -> Tool:
        """Create tool for particle extraction."""
        return Tool(
            name="extract_particles",
            description="Extract particles from micrographs using particle coordinates from picking. "
                       "Required parameters: particles_job_uid (from blob picker), micrographs_job_uid (from CTF/selection), box_size_pix (box size in pixels). "
                       "Optional parameters: project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._extract_particles_tool
        )
    
    @staticmethod
    def create_class_2d_tool(agent) -> Tool:
        """Create tool for 2D classification."""
        return Tool(
            name="class_2d",
            description="Perform 2D classification on extracted particles. "
                       "Required parameters: particles_job_uid (from extraction). "
                       "Optional parameters: num_classes (default: 20), project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._class_2d_tool
        )
    
    @staticmethod
    def create_select_2d_classes_tool(agent) -> Tool:
        """Create tool for 2D class selection."""
        return Tool(
            name="select_2d_classes",
            description="Select 2D classes to use as templates. "
                       "Required parameters: class_2d_job_uid (from 2D classification). "
                       "Optional parameters: selection_mode (top_n or cryosift), top_n_classes (default: 5 when top_n), "
                       "cryosift_threshold, cryosift_env, cryosift_weights_path, cryosift_output_dir, project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._select_2d_classes_tool
        )
    
    @staticmethod
    def create_template_picker_tool(agent) -> Tool:
        """Create tool for template-based particle picking."""
        return Tool(
            name="template_picker",
            description="Template-based particle picking using 2D class averages as templates. More accurate than blob picker. "
                       "Required parameters: micrographs_job_uid (from CTF/selection), template_job_uid (from select_2d_classes). "
                       "Optional parameters: lowpass_resolution (default: 20.0 Å), project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
            func=agent._template_picker_tool
        )
    
    @staticmethod
    def create_reason_about_workflow_tool(agent) -> Tool:
        """Create tool for reasoning about particle picking workflow."""
        return Tool(
            name="reason_about_workflow",
            description="Analyze the current particle picking workflow state and determine next steps. "
                       "Use this to think through particle detection parameters and job dependencies.",
            func=agent._reason_about_workflow_tool
        )

