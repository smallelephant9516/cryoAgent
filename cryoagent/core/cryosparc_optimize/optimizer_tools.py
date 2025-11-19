"""Tools for CryoSPARC box size optimization operations."""

from typing import Dict, Any, Optional
from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class OptimizerTools:
    """Factory for creating box size optimization-specific tools."""
    
    @staticmethod
    def create_test_box_size_tool(agent):
        """Create tool for testing a specific box size."""
        # Define the input schema for StructuredTool
        class TestBoxSizeInput(BaseModel):
            box_size_pix: int = Field(description="Box size in pixels to test (e.g., 292)")
            refinement_job_uid: str = Field(description="Source of refined particle coordinates (e.g., 'J53')")
            micrographs_job_uid: str = Field(default="", description="Micrographs for re-extraction (e.g., 'J43'). Optional if set in workflow defaults.")
            volume_job_uid: str = Field(default="", description="Initial volume for refinement (e.g., 'J53'). Optional if set in workflow defaults.")
            refinement_resolution: Optional[float] = Field(default=None, description="Optional target resolution in Angstroms")
            project_uid: str = Field(default="", description="Optional project UID")
            workspace_uid: str = Field(default="", description="Optional workspace UID")
        
        # Create a wrapper function that converts structured input to string format
        def test_box_size_wrapper(
            box_size_pix: int,
            refinement_job_uid: str,
            micrographs_job_uid: str = "",
            volume_job_uid: str = "",
            refinement_resolution: Optional[float] = None,
            project_uid: str = "",
            workspace_uid: str = ""
        ) -> str:
            """Wrapper to convert structured input to JSON string format."""
            import json
            params = {
                "box_size_pix": box_size_pix,
                "refinement_job_uid": refinement_job_uid
            }
            if micrographs_job_uid:
                params["micrographs_job_uid"] = micrographs_job_uid
            if volume_job_uid:
                params["volume_job_uid"] = volume_job_uid
            if refinement_resolution is not None:
                params["refinement_resolution"] = refinement_resolution
            if project_uid:
                params["project_uid"] = project_uid
            if workspace_uid:
                params["workspace_uid"] = workspace_uid
            # Convert to JSON string and call the original function
            json_input = json.dumps(params)
            return agent._test_box_size_tool(json_input)
        
        return StructuredTool.from_function(
            func=test_box_size_wrapper,
            name="test_box_size",
            description="Test a specific box size by extracting particles, running refinement, and getting FSC resolution. "
                       "This tool: 1) Extracts particles with the specified box_size_pix using refined coordinates from refinement_job_uid, "
                       "2) Runs homogeneous refinement, 3) Gets FSC resolution. "
                       "Required parameters: box_size_pix (box size in pixels to test), refinement_job_uid (source of refined particle coordinates). "
                       "Optional parameters: micrographs_job_uid (micrographs for re-extraction), volume_job_uid (initial volume for refinement), "
                       "refinement_resolution (target resolution in Angstroms), project_uid, workspace_uid. "
                       "Returns: job_uid, box_size, resolution_angstroms, and status.",
            args_schema=TestBoxSizeInput
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
    def create_test_heterogeneous_refinement_tool(agent):
        """Create tool for testing heterogeneous refinement with a specific K value."""
        # Define the input schema for StructuredTool
        class TestHeterogeneousRefinementInput(BaseModel):
            k: int = Field(description="Number of classes to test (e.g., 3 or 5)")
            refinement_job_uid: str = Field(description="Source of particles and volume (e.g., 'J357')")
            project_uid: str = Field(default="", description="Optional project UID")
            workspace_uid: str = Field(default="", description="Optional workspace UID")
        
        # Create a wrapper function that converts structured input to string format
        def test_heterogeneous_refinement_wrapper(k: int, refinement_job_uid: str, project_uid: str = "", workspace_uid: str = "") -> str:
            """Wrapper to convert structured input to JSON string format."""
            import json
            params = {
                "k": k,
                "refinement_job_uid": refinement_job_uid
            }
            if project_uid:
                params["project_uid"] = project_uid
            if workspace_uid:
                params["workspace_uid"] = workspace_uid
            # Convert to JSON string and call the original function
            json_input = json.dumps(params)
            return agent._test_heterogeneous_refinement_tool(json_input)
        
        return StructuredTool.from_function(
            func=test_heterogeneous_refinement_wrapper,
            name="test_heterogeneous_refinement",
            description="Test heterogeneous refinement with K classes. "
                       "This tool: 1) Repeats the volume from refinement_job_uid K times as initial densities, "
                       "2) Runs heterogeneous refinement using particles from refinement_job_uid, "
                       "3) Runs regroup to regroup K classes into 2 superclasses (job name: regroup_3D_new), "
                       "4) Gets num_items for each superclass from regroup job.json, "
                       "5) Selects the superclass with more particles, "
                       "6) Runs homogeneous refinement on selected superclass particles and volumes, "
                       "7) Gets final FSC resolution. "
                       "Required parameters: k (number of classes, e.g., 3 or 5), refinement_job_uid (source of particles and volume, e.g., 'J357'). "
                       "Optional parameters: project_uid, workspace_uid. "
                       "Returns: hetero_job_uid, regroup_job_uid, best_superclass_id, best_superclass_num_items, refine_job_uid, final_resolution_angstroms, and status.",
            args_schema=TestHeterogeneousRefinementInput
        )

