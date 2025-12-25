"""Tools for CryoSPARC heterogeneity analysis operations."""

from typing import Dict, Any, Optional
from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class HeterogeneityTools:
    """Factory for creating heterogeneity analysis-specific tools."""
    
    @staticmethod
    def create_run_ab_initio_hetero_combo_tool(agent):
        """Create tool for running ab initio + heterogeneous refinement combo."""
        class RunAbInitioHeteroComboInput(BaseModel):
            k: int = Field(description="Number of classes to test (e.g., 3 or 5)")
            particles_job_uid: str = Field(description="Source of particles (e.g., 'JXXX')")
            project_uid: str = Field(default="", description="Optional project UID")
            workspace_uid: str = Field(default="", description="Optional workspace UID")
        
        def run_ab_initio_hetero_combo_wrapper(
            k: int,
            particles_job_uid: str,
            project_uid: str = "",
            workspace_uid: str = ""
        ) -> str:
            """Wrapper to convert structured input to JSON string format."""
            import json
            params = {
                "k": k,
                "particles_job_uid": particles_job_uid
            }
            if project_uid:
                params["project_uid"] = project_uid
            if workspace_uid:
                params["workspace_uid"] = workspace_uid
            json_input = json.dumps(params)
            return agent._run_ab_initio_hetero_combo_tool(json_input)
        
        return StructuredTool.from_function(
            func=run_ab_initio_hetero_combo_wrapper,
            name="run_ab_initio_hetero_combo",
            description="Run ab initio reconstruction + heterogeneous refinement combo with K classes. "
                       "This tool: 1) Runs ab initio reconstruction with K classes, "
                       "2) Runs heterogeneous refinement using the ab initio volumes, "
                       "3) Returns the heterogeneous refinement job UID. "
                       "Required parameters: k (number of classes, e.g., 3 or 5), particles_job_uid (source of particles, e.g., 'JXXX'). "
                       "Optional parameters: project_uid, workspace_uid. "
                       "Returns: ab_initio_job_uid, hetero_job_uid, and status.",
            args_schema=RunAbInitioHeteroComboInput
        )
    
    @staticmethod
    def create_extract_density_maps_tool(agent) -> Tool:
        """Create tool for getting the job directory containing density maps from a heterogeneous refinement job."""
        return Tool(
            name="extract_density_maps",
            description="Get the job directory containing density map files (*_volume.mrc) from a heterogeneous refinement job. "
                       "Returns the job directory directly without copying files (Docker-accessible). "
                       "You can pass just the job UID (e.g., 'JXXX') or JSON with hetero_job_uid parameter. "
                       "Optional parameters: project_uid. "
                       "Returns: output_folder (job directory path), num_maps_extracted, map_files list (full paths), and success status.",
            func=agent._extract_density_maps_tool
        )
    
    @staticmethod
    def create_get_hetero_class_resolutions_tool(agent) -> Tool:
        """Create tool for getting class resolutions from heterogeneous refinement job."""
        return Tool(
            name="get_hetero_class_resolutions",
            description="Get resolution information for each class in a heterogeneous refinement job. "
                       "You can pass just the job UID (e.g., 'JXXX') or JSON with job_uid parameter. "
                       "Returns a list of classes with resolution_angstroms and fsc_loosemask_last for each class. "
                       "Optional parameters: project_uid, workspace_uid. "
                       "Returns: classes (list with class_id, resolution_angstroms, fsc_loosemask_last), num_classes, and success status.",
            func=agent._get_hetero_class_resolutions_tool
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
    def create_run_non_uniform_refinement_tool(agent) -> Tool:
        """Create tool for running non-uniform refinement for a specific group."""
        return Tool(
            name="run_non_uniform_refinement",
            description="Run non-uniform homogeneous refinement for a specific group of particles. "
                       "This tool refines a single group using particles from one or more classes and a volume from the best-resolution class. "
                       "Required parameters: hetero_job_uid (heterogeneous refinement job UID), particles_group_names (list of particle group names like 'particles_class_0'), "
                       "volume_group_name (volume group name like 'volume_class_0' from the class with best resolution). "
                       "Optional parameters: project_uid, workspace_uid, refine_res_init (initial lowpass resolution in Angstroms). "
                       "Returns: job_uid, job_type, and status.",
            func=agent._run_non_uniform_refinement_tool
        )

