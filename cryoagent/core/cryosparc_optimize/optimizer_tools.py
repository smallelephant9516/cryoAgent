"""Tools for CryoSPARC box size optimization operations."""

from typing import Dict, Any
from langchain.tools import Tool


class OptimizerTools:
    """Factory for creating box size optimization-specific tools."""
    
    @staticmethod
    def create_test_box_size_tool(agent) -> Tool:
        """Create tool for testing a specific box size."""
        return Tool(
            name="test_box_size",
            description="Test a specific box size by extracting particles, running refinement, and getting FSC resolution. "
                       "This tool: 1) Extracts particles with the specified box_size_pix using refined coordinates from refinement_job_uid, "
                       "2) Runs homogeneous refinement, 3) Gets FSC resolution. "
                       "Required parameters: box_size_pix (box size in pixels to test), refinement_job_uid (source of refined particle coordinates), "
                       "micrographs_job_uid (micrographs for re-extraction), volume_job_uid (initial volume for refinement). "
                       "Optional parameters: refinement_resolution (target resolution in Angstroms), project_uid, workspace_uid. "
                       "Returns: job_uid, box_size, resolution_angstroms, and status.",
            func=agent._test_box_size_tool
        )
    
    @staticmethod
    def create_get_fsc_info_tool(agent) -> Tool:
        """Create tool for getting FSC resolution and box size from a refinement job."""
        return Tool(
            name="get_fsc_info",
            description="Get FSC resolution and box size information from a refinement job. "
                       "You can pass just the job UID (e.g., 'J357') or JSON with refinement_job_uid parameter. "
                       "Optional parameters: project_uid, workspace_uid. "
                       "Returns: box_size (in pixels), resolution_angstroms (FSC resolution), and success status.",
            func=agent._get_fsc_info_tool
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
    
    @staticmethod
    def create_get_hetero_class_resolutions_tool(agent) -> Tool:
        """Create tool for getting class resolutions from heterogeneous refinement job."""
        return Tool(
            name="get_hetero_class_resolutions",
            description="Get resolution information for each class in a heterogeneous refinement job. "
                       "You can pass just the job UID (e.g., 'J357') or JSON with job_uid parameter. "
                       "Returns a list of classes with resolution_angstroms and fsc_loosemask_last for each class. "
                       "Optional parameters: project_uid, workspace_uid. "
                       "Returns: classes (list with class_id, resolution_angstroms, fsc_loosemask_last), num_classes, and success status.",
            func=agent._get_hetero_class_resolutions_tool
        )
    
    @staticmethod
    def create_test_heterogeneous_refinement_tool(agent) -> Tool:
        """Create tool for testing heterogeneous refinement with a specific K value."""
        return Tool(
            name="test_heterogeneous_refinement",
            description="Test heterogeneous refinement with K classes. "
                       "This tool: 1) Repeats the volume from refinement_job_uid K times as initial densities, "
                       "2) Runs heterogeneous refinement using particles from refinement_job_uid, "
                       "3) Gets resolution for each class, 4) Selects best class (smallest resolution value), "
                       "5) Runs homogeneous refinement on selected class particles, 6) Gets final FSC resolution. "
                       "Input format: JSON string with required parameters: {\"k\": 3, \"refinement_job_uid\": \"J357\"}. "
                       "Required parameters: k (number of classes, e.g., 3 or 5), refinement_job_uid (source of particles and volume, e.g., \"J357\"). "
                       "Optional parameters: project_uid, workspace_uid. "
                       "Example: '{\"k\": 3, \"refinement_job_uid\": \"J357\"}' "
                       "Returns: hetero_job_uid, best_class_id, best_class_resolution, refine_job_uid, final_resolution_angstroms, and status.",
            func=agent._test_heterogeneous_refinement_tool
        )

