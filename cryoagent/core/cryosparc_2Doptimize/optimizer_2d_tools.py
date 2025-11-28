"""Tools for CryoSPARC 2D optimization operations."""

from typing import Dict, Any
from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class Optimizer2DTools:
    """Factory for creating 2D optimization-specific tools."""
    
    @staticmethod
    def create_class_2d_tool(agent) -> Tool:
        """Create tool for running 2D classification."""
        return Tool(
            name="class_2d",
            description="Run 2D classification on particles. "
                       "Required parameters: particles_job_uid (e.g., 'J123') OR job_uid (when passing just 'J123'). "
                       "Optional parameters: num_classes (default from config), particles_group_name (e.g., 'particles_excluded' for excluded particles from select_2D job), project_uid, workspace_uid. "
                       "Returns: job_uid and status. "
                       "CRITICAL for Function 2 (Rescue): When running rescue workflow, you MUST: "
                       "(1) Use the select_2D job_uid from Step A (e.g., 'J116'), "
                       "(2) Specify particles_group_name='particles_excluded' to classify excluded particles. "
                       "Example: class_2d with particles_job_uid='J116' and particles_group_name='particles_excluded'. "
                       "DO NOT call class_2d on a select_2D job without particles_group_name - this will classify the wrong particles!",
            func=agent._class_2d_tool
        )
    
    @staticmethod
    def create_select_2d_classes_tool(agent) -> Tool:
        """Create tool for selecting 2D classes using CryoSift."""
        return Tool(
            name="select_2d_classes",
            description="Select 2D classes using various selection modes. "
                       "Required parameters: class_2d_job_uid (e.g., 'J123'). "
                       "Optional parameters: selection_mode (default: 'cryosift', options: 'cryosift', 'top_n', 'all'), cryosift_threshold, project_uid, workspace_uid. "
                       "Selection modes: 'cryosift' (selects classes using CryoSift evaluation), 'top_n' (selects top N classes by particle count), 'all' (selects all classes). "
                       "Returns: job_uid, selected_template_indices, selection_metadata, and status.",
            func=agent._select_2d_classes_tool
        )
    
    @staticmethod
    def create_get_particle_count_tool(agent) -> Tool:
        """Create tool for getting particle count from a job."""
        return Tool(
            name="get_particle_count",
            description="Get the number of particles in a particles job. "
                       "Required parameters: particles_job_uid (e.g., 'J123'). "
                       "Optional parameters: particles_group_name (default: 'particles'), project_uid. "
                       "Returns: num_particles, particles_group_name, and success status.",
            func=agent._get_particle_count_tool
        )
    
    @staticmethod
    def create_merge_particles_tool(agent):
        """Create tool for merging particles from multiple jobs."""
        class MergeParticlesInput(BaseModel):
            particles_job_uids: str = Field(description="Comma-separated list of particle job UIDs to merge (e.g., 'J123,J124')")
            project_uid: str = Field(default="", description="Optional project UID")
            workspace_uid: str = Field(default="", description="Optional workspace UID")
        
        def merge_particles_wrapper(particles_job_uids: str, project_uid: str = "", workspace_uid: str = "") -> str:
            """Wrapper to convert structured input to JSON string format."""
            import json
            # Parse comma-separated list
            job_uids_list = [uid.strip() for uid in particles_job_uids.split(",") if uid.strip()]
            params = {
                "particles_job_uids": job_uids_list
            }
            if project_uid:
                params["project_uid"] = project_uid
            if workspace_uid:
                params["workspace_uid"] = workspace_uid
            json_input = json.dumps(params)
            return agent._merge_particles_tool(json_input)
        
        return StructuredTool.from_function(
            func=merge_particles_wrapper,
            name="merge_particles",
            description="Merge particles from multiple jobs into a single particles set. "
                       "Required parameters: particles_job_uids (comma-separated list, e.g., 'J123,J124'). "
                       "Optional parameters: project_uid, workspace_uid. "
                       "Returns: merged job_uid and status.",
            args_schema=MergeParticlesInput
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

