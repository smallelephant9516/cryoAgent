"""Tools for CryoSPARC box size optimization operations."""

from typing import Dict, Any
from langchain.tools import Tool


class OptimizerTools:
    """Factory for creating box size optimization-specific tools."""
    
    @staticmethod
    def create_optimize_diameter_tool(agent) -> Tool:
        """Create tool for optimizing box size/diameter based on FSC resolution."""
        return Tool(
            name="optimize_diameter",
            description="Optimize box size/diameter by testing different box sizes and comparing FSC resolutions. "
                       "This tool automatically tests 10% less and 10% more box sizes, extracts particles using refined coordinates from the refinement job, runs refinement, "
                       "and iteratively finds the optimal box size. "
                       "Required parameters: refinement_job_uid (first refinement job, used for refined particle coordinates), particles_job_uid (picking job, kept for compatibility), "
                       "micrographs_job_uid (micrographs for re-extraction), volume_job_uid (initial volume). "
                       "Note: Particle re-extraction uses coordinates from the refinement_job_uid (refined positions/orientations), not the picking job. "
                       "Optional parameters: project_uid, workspace_uid. "
                       "The tool will try to get missing parameters from workflow defaults if not provided.",
            func=agent._optimize_diameter_tool
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
        """Create tool for reasoning about optimization workflow."""
        return Tool(
            name="reason_about_workflow",
            description="Analyze the current box size optimization workflow state and determine next steps. "
                       "Use this to think through optimization parameters and job dependencies.",
            func=agent._reason_about_workflow_tool
        )

