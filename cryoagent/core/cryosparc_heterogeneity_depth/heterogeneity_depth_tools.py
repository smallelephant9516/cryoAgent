"""Tools for CryoSPARC heterogeneity depth analysis operations."""

from typing import Dict, Any, Optional, List
from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class HeterogeneityDepthTools:
    """Factory for creating heterogeneity depth analysis-specific tools."""
    
    @staticmethod
    def create_read_input_json_tool(agent):
        """Create tool for reading input JSON from refinement or heterogeneity job."""
        class ReadInputJsonInput(BaseModel):
            config_path: Optional[str] = Field(default="", description="Optional path to config file specifying which JSON to read")
        
        def read_input_json_wrapper(config_path: str = "") -> str:
            """Wrapper to convert structured input to JSON string format."""
            import json
            params = {}
            if config_path:
                params["config_path"] = config_path
            json_input = json.dumps(params)
            return agent._read_input_json_tool(json_input)
        
        return StructuredTool.from_function(
            func=read_input_json_wrapper,
            name="read_input_json",
            description="Read JSON file from either refinement job (reconstruction_results_*.json) or heterogeneity job (heterogeneity_analysis_results_*.json). "
                       "If heterogeneity job JSON exists, it will be preferred. "
                       "When reading from heterogeneity_analysis_results_*.json, returns all clusters from final_refinement_jobs with: "
                       "num_clusters, clusters (array with refinement_job_uid, particles_job_uid, volume_job_uid, particles_group_name, volume_group_name for each cluster). "
                       "When reading from reconstruction_results_*.json, returns: refinement_job_uid, particles_job_uid, volume_job_uid. "
                       "Optional parameter: config_path (path to config file).",
            args_schema=ReadInputJsonInput
        )
    
    @staticmethod
    def create_run_heterogeneous_refinement_tool(agent):
        """Create tool for running heterogeneous refinement with K classes."""
        class RunHeterogeneousRefinementInput(BaseModel):
            particles_job_uid: str = Field(description="Source of particles (e.g., 'JXXX')")
            volume_job_uid: str = Field(description="Source of volume (e.g., 'JXXX')")
            k: int = Field(description="Number of classes (default: 4)")
            particles_group_names: Optional[List[str]] = Field(default=None, description="Optional list of particles group names (e.g., ['particles_class_0', 'particles_class_2'] for multiple classes, or ['particles_class_1'] for single class)")
            particles_group_name: Optional[str] = Field(default="", description="Optional single particles group name (legacy, use particles_group_names for multiple classes)")
            volume_group_name: Optional[str] = Field(default="", description="Optional volume group name (e.g., 'volume_class_0')")
            project_uid: str = Field(default="", description="Optional project UID")
            workspace_uid: str = Field(default="", description="Optional workspace UID")
        
        def run_heterogeneous_refinement_wrapper(
            particles_job_uid: str,
            volume_job_uid: str,
            k: int = 4,
            particles_group_names: Optional[List[str]] = None,
            particles_group_name: str = "",
            volume_group_name: str = "",
            project_uid: str = "",
            workspace_uid: str = ""
        ) -> str:
            """Wrapper to convert structured input to JSON string format."""
            import json
            params = {
                "particles_job_uid": particles_job_uid,
                "volume_job_uid": volume_job_uid,
                "k": k
            }
            # Prefer particles_group_names (list) over particles_group_name (string)
            if particles_group_names:
                params["particles_group_names"] = particles_group_names
            elif particles_group_name:
                params["particles_group_name"] = particles_group_name
            if volume_group_name:
                params["volume_group_name"] = volume_group_name
            if project_uid:
                params["project_uid"] = project_uid
            if workspace_uid:
                params["workspace_uid"] = workspace_uid
            json_input = json.dumps(params)
            return agent._run_heterogeneous_refinement_tool(json_input)
        
        return StructuredTool.from_function(
            func=run_heterogeneous_refinement_wrapper,
            name="run_heterogeneous_refinement",
            description="Run heterogeneous refinement with K classes. Waits for completion, then auto-runs "
                       "class resolutions + density comparison. Use the returned result to decide: "
                       "Case A (zero good classes → fallback non-uniform on particles_all_classes), "
                       "Case B1 (one KEPT cluster → non-uniform refinement), "
                       "Case B2 (multiple KEPT clusters → hetero per KEPT cluster; discard FILTERED OUT clusters). "
                       "Required: particles_job_uid, volume_job_uid, k. "
                       "Optional: particles_group_names, particles_group_name, volume_group_name, project_uid, workspace_uid. "
                       "Returns: hetero_job_uid, good_classes, bad_classes, density_comparison, next_action, fallback_non_uniform.",
            args_schema=RunHeterogeneousRefinementInput
        )
    
    @staticmethod
    def create_extract_density_maps_tool(agent) -> Tool:
        """Create tool for getting the job directory containing density maps from a heterogeneous refinement job."""
        return Tool(
            name="extract_density_maps",
            description="Get the job directory containing density map files (*_volume.mrc) from a heterogeneous refinement job. "
                       "Returns the job directory directly without copying files. "
                       "You can pass just the job UID (e.g., 'JXXX') or JSON with hetero_job_uid parameter. "
                       "Optional parameters: project_uid. "
                       "Returns: output_folder (job directory path), num_maps_extracted, map_files list (full paths), and success status.",
            func=agent._extract_density_maps_tool
        )
    
    @staticmethod
    def create_run_homogeneous_refinement_tool(agent):
        """Create tool for running homogeneous refinement."""
        class RunHomogeneousRefinementInput(BaseModel):
            particles_job_uid: str = Field(description="Source of particles (e.g., 'JXXX')")
            volume_job_uid: str = Field(description="Source of volume (e.g., 'JXXX')")
            particles_group_name: Optional[str] = Field(default="", description="Optional particles group name (e.g., 'particles_class_0')")
            volume_group_name: Optional[str] = Field(default="", description="Optional volume group name (e.g., 'volume_class_0')")
            refine_defocus_refine: Optional[bool] = Field(default=True, description="Enable defocus refinement during CTF refinement (default: True)")
            refine_ctf_global_refine: Optional[bool] = Field(default=True, description="Enable global CTF refinement (default: True)")
            project_uid: str = Field(default="", description="Optional project UID")
            workspace_uid: str = Field(default="", description="Optional workspace UID")
        
        def run_homogeneous_refinement_wrapper(
            particles_job_uid: str,
            volume_job_uid: str,
            particles_group_name: str = "",
            volume_group_name: str = "",
            refine_defocus_refine: bool = True,
            refine_ctf_global_refine: bool = True,
            project_uid: str = "",
            workspace_uid: str = ""
        ) -> str:
            """Wrapper to convert structured input to JSON string format."""
            import json
            params = {
                "particles_job_uid": particles_job_uid,
                "volume_job_uid": volume_job_uid
            }
            if particles_group_name:
                params["particles_group_name"] = particles_group_name
            if volume_group_name:
                params["volume_group_name"] = volume_group_name
            if refine_defocus_refine is not None:
                params["refine_defocus_refine"] = refine_defocus_refine
            if refine_ctf_global_refine is not None:
                params["refine_ctf_global_refine"] = refine_ctf_global_refine
            if project_uid:
                params["project_uid"] = project_uid
            if workspace_uid:
                params["workspace_uid"] = workspace_uid
            json_input = json.dumps(params)
            return agent._run_homogeneous_refinement_tool(json_input)
        
        return StructuredTool.from_function(
            func=run_homogeneous_refinement_wrapper,
            name="run_homogeneous_refinement",
            description="Run homogeneous refinement using particles and volume from a job. "
                       "Required parameters: particles_job_uid, volume_job_uid. "
                       "Optional parameters: particles_group_name (e.g., 'particles_class_0' or 'particles_all_classes'), "
                       "volume_group_name (e.g., 'volume_class_0'), "
                       "refine_defocus_refine (enable defocus refinement during CTF refinement, default: True), "
                       "refine_ctf_global_refine (enable global CTF refinement, default: True), "
                       "project_uid, workspace_uid. "
                       "Returns: job_uid, status.",
            args_schema=RunHomogeneousRefinementInput
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
    def create_get_hetero_class_resolutions_tool(agent) -> Tool:
        """Create tool for getting class resolutions from heterogeneous refinement job."""
        return Tool(
            name="get_hetero_class_resolutions",
            description="Get resolution for each class and label GOOD vs BAD relative to the depth threshold. "
                       "Normally auto-run inside run_heterogeneous_refinement — call manually only to re-analyze an older job UID. "
                       "You can pass just the job UID (e.g., 'JXXX') or JSON with job_uid parameter. "
                       "Returns: good_classes, bad_classes, fallback_non_uniform (when zero good classes), "
                       "resolution_threshold_angstroms, next_action, and per-class quality labels.",
            func=agent._get_hetero_class_resolutions_tool
        )

    @staticmethod
    def create_compare_all_densities_tool(agent) -> Tool:
        """Create resolution-aware density comparison tool for depth analysis."""
        return Tool(
            name="compare_all_densities",
            description="Compare density maps in a folder and filter clusters by resolution. "
                       "Normally auto-run inside run_heterogeneous_refinement — call manually only to re-analyze an older job UID. "
                       "Required: folder (path to hetero job directory). "
                       "Do NOT pass the full get_hetero_class_resolutions JSON — only folder + optional class_resolutions. "
                       "KEPT clusters continue; FILTERED OUT (BAD) clusters must be thrown away with no further processing.",
            func=agent._compare_all_densities_tool
        )

    @staticmethod
    def create_run_non_uniform_refinement_tool(agent) -> Tool:
        """Create tool for final non-uniform refinement when a good branch converges."""
        return Tool(
            name="run_non_uniform_refinement",
            description="Run non-uniform refinement to terminate a branch. "
                       "Converged good cluster: hetero_job_uid + particles_group_names (good class(es)) + best volume. "
                       "Zero good classes fallback: hetero_job_uid + particles_group_names=['particles_all_classes'] + best volume_class_X. "
                       "Required: hetero_job_uid, particles_group_names (list), volume_group_name. "
                       "Optional: project_uid, workspace_uid, refine_res_init. "
                       "After completion, call wait_for_job then get_fsc_info to report final resolution.",
            func=agent._run_non_uniform_refinement_tool
        )

    @staticmethod
    def create_get_fsc_info_tool(agent) -> Tool:
        """Create tool for reporting FSC resolution after non-uniform refinement."""
        return Tool(
            name="get_fsc_info",
            description="Get FSC resolution and box size from a completed non-uniform refinement job. "
                       "Pass job UID (e.g. 'JXXX') or JSON with refinement_job_uid. "
                       "MUST be called after wait_for_job on the final non-uniform refinement job.",
            func=agent._get_fsc_info_tool
        )

