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
    def create_reason_about_workflow_tool(agent) -> Tool:
        """Create tool for reasoning about particle picking workflow."""
        return Tool(
            name="reason_about_workflow",
            description="Analyze the current particle picking workflow state and determine next steps. "
                       "Use this to think through particle detection parameters and job dependencies.",
            func=agent._reason_about_workflow_tool
        )

