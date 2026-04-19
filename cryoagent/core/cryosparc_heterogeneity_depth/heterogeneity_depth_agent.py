"""ReAct-based heterogeneity depth analysis agent for CryoEM 3D reconstruction."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..base_react_agent import BaseReActAgent
from .heterogeneity_depth_tools import HeterogeneityDepthTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...tools.alignment_tools.compare_all_densities_tool import CompareAllDensitiesTool
from ...config.config_loader import CryoAgentConfig


class HeterogeneityDepthAgent(BaseReActAgent):
    """ReAct-based agent for heterogeneity depth analysis in CryoEM 3D reconstruction."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the heterogeneity depth analysis agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        # Initialize stage_config BEFORE calling super().__init__() because
        # BaseReActAgent.__init__() calls _create_tools() which may access stage_config
        self.workflow_defaults: Dict[str, Any] = {}
        self.stage_config = self._load_stage_config()
        self.stage_workflow = self.stage_config.get("workflow", {})
        
        # Now call super().__init__() which will call _create_tools()
        super().__init__(cryosparc_tools, config, llm)
        
        # Initialize logger for this agent
        self.logger = logging.getLogger("HeterogeneityDepthAgent")
    
    def _create_tools(self) -> List[Tool]:
        """Create heterogeneity depth analysis-specific tools."""
        tools = [
            HeterogeneityDepthTools.create_read_input_json_tool(self),
            HeterogeneityDepthTools.create_run_heterogeneous_refinement_tool(self),
            HeterogeneityDepthTools.create_extract_density_maps_tool(self),
            HeterogeneityDepthTools.create_get_hetero_class_resolutions_tool(self),
            HeterogeneityDepthTools.create_run_homogeneous_refinement_tool(self),
            HeterogeneityDepthTools.create_get_job_status_tool(self),
            HeterogeneityDepthTools.create_wait_for_job_tool(self),
            HeterogeneityDepthTools.create_get_job_log_tool(self),
        ]
        
        # Add compare_all_densities tool
        compare_tool = CompareAllDensitiesTool.create_compare_all_densities_tool(
            compare_script=self._get_stage_param("script_paths", "compare_all_densities_script", None),
            align_script=self._get_stage_param("script_paths", "align_and_compare_script", None),
            default_voxel_size=self._get_stage_param("density_comparison", "voxel_size", 5.0),
            default_alg_type=self._get_stage_param("density_comparison", "alg_type", "global"),
            default_resolution_threshold=self._get_stage_param("density_comparison", "resolution_threshold", None)
        )
        tools.append(compare_tool)
        
        return tools
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load heterogeneity depth analysis stage configuration."""
        config_path = Path("configs/cryosparc/heterogeneity_depth_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _load_reconstruction_config(self) -> Dict[str, Any]:
        """Load reconstruction stage configuration to get symmetry settings."""
        config_path = Path("configs/cryosparc/reconstruction_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _get_refinement_symmetry(self) -> str:
        """Get symmetry from reconstruction_config.json for refinement."""
        recon_config = self._load_reconstruction_config()
        
        # First try to get from workflow.refinement.symmetry (preferred)
        refinement_symmetry = recon_config.get("workflow", {}).get("refinement", {}).get("symmetry")
        if refinement_symmetry:
            return refinement_symmetry
        
        # Fall back to microscope_parameters.symmetry
        microscope_symmetry = recon_config.get("microscope_parameters", {}).get("symmetry")
        if microscope_symmetry:
            return microscope_symmetry
        
        # Default to C1 if not found
        return "C1"
    
    def _get_refinement_res_init(self) -> Optional[float]:
        """Get initial lowpass resolution for refinement from config."""
        return self._get_stage_param("heterogeneity_depth_analysis", "refine_res_init", None)
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Fetch a parameter from the stage workflow configuration."""
        if section == "heterogeneity_depth_analysis":
            return self.stage_workflow.get("heterogeneity_depth_analysis", {}).get(key, default)
        elif section == "density_comparison":
            density_config = self.stage_workflow.get("heterogeneity_depth_analysis", {}).get("density_comparison", {})
            return density_config.get(key, default)
        elif section == "external_tools":
            external_config = self.stage_workflow.get("heterogeneity_depth_analysis", {}).get("external_tools", {})
            return external_config.get(key, default)
        elif section == "script_paths":
            script_config = self.stage_workflow.get("heterogeneity_depth_analysis", {}).get("script_paths", {})
            return script_config.get(key, default)
        return default
    
    def _get_react_system_prompt(self) -> str:
        """Get the heterogeneity depth analysis-specific ReAct system prompt."""
        k_value = self._get_stage_param("heterogeneity_depth_analysis", "k", 4)
        resolution_threshold = self._get_stage_param("heterogeneity_depth_analysis", "resolution_threshold", 12.0)
        
        return f"""You are a CryoEM heterogeneity depth analysis assistant using the ReAct (Reasoning + Acting) framework.
You specialize in performing deep heterogeneity analysis by:
1. Reading input from refinement or heterogeneity job JSON files
2. Running heterogeneous refinement with K={k_value} classes
3. Validating clusters using density comparison
4. Iteratively refining until only one cluster remains
5. Running final homogeneous refinement

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}
- K value: {k_value}
- Resolution threshold: {resolution_threshold} Å

## Heterogeneity Depth Analysis Workflow:

**Step 1: Read Input JSON (MANDATORY FIRST STEP)**
1. **CRITICAL: You MUST start by reading the input JSON file. DO NOT run any reconstruction or refinement jobs until you have read the JSON file first.**
2. Use `read_input_json` tool to read JSON file from either:
   - Heterogeneity job: heterogeneity_analysis_results_*.json (preferred if available)
   - Refinement job: reconstruction_results_*.json (fallback if heterogeneity not available)
3. If reading from heterogeneity analysis results:
   - The tool will return all clusters from `final_refinement_jobs`
   - Each cluster has: refinement_job_uid (e.g., J83, J84), particles_job_uid, volume_job_uid, particles_group_names, volume_group_name
   - particles_job_uid and volume_job_uid are from the refinement job (not the hetero job)
   - **IMPORTANT**: Refinement jobs (non-uniform or homogeneous) use standard group names:
     * particles_group_names: ["particles"] (standard group, NOT particles_class_X)
     * volume_group_name: "volume" (standard group, NOT volume_class_X)
   - Class-specific group names (particles_class_X, volume_class_X) only exist in heterogeneous refinement jobs
   - You will process EACH cluster separately
   - **DO NOT proceed to Step 2 until you have successfully read the JSON file**

**Step 2: Process Each Cluster**
For EACH cluster from the heterogeneity analysis results:
1. **Initial Heterogeneous Refinement**:
   - Use particles_job_uid and volume_job_uid from the cluster (these are from the refinement job)
   - Use particles_group_name and volume_group_name from the cluster
   - Run heterogeneous refinement with K={k_value} using:
     * Particles from refinement_job_uid with particles_group_name (e.g., "particles" or "particles_class_0")
     * Volume from refinement_job_uid with volume_group_name (e.g., "volume" or "volume_class_0")
   - Wait for completion

2. **Extract and Compare Densities**:
   - Extract density maps from heterogeneous refinement job
   - Compare all density maps using `compare_all_densities` tool
   - Check the number of clusters from the comparison results
   - **CRITICAL**: Extract which classes belong to each cluster from the comparison output
     * Map names contain class IDs: "J39_class_00_00042_volume.mrc" → class 0, "J39_class_02_00042_volume.mrc" → class 2
     * The comparison results show which maps (and thus which classes) belong to each cluster
     * Use this information to determine particles_group_names for the next refinement

3. **Tree Structure Expansion (Recursive)**:
   - **If comparison shows only 1 cluster:**
     * Extract which classes belong to this cluster from the comparison results (map names like "J34_class_03_volume.mrc" → class 3)
     * **CRITICAL**: Determine how many classes are in this cluster:
       - If cluster has only 1 class (e.g., only class 3): Use `particles_class_3` (NOT particles_all_classes)
       - If cluster has multiple classes: Use particles from all classes in this cluster (particles_class_X for each class in the cluster)
     * Use the best volume from the classes in this cluster (volume_class_X where X has best resolution)
     * Run homogeneous refinement using the appropriate particles group(s) and best volume
     * Record this final refinement job UID - this branch is COMPLETE
     * Move to the next starting cluster
   
   - **If comparison shows multiple clusters (e.g., 2, 3, or more):**
     * This creates a TREE STRUCTURE - the hetero job splits into multiple branches
     * For EACH new cluster found in the comparison:
       a. Extract which classes belong to this cluster from the comparison results (map names like "J39_class_00_volume.mrc" → class 0)
       b. **CRITICAL**: Determine how many classes are in this cluster:
          - If cluster has only 1 class (e.g., class 3): Use `particles_group_names=["particles_class_3"]` (list with single element)
          - If cluster has multiple classes (e.g., classes 0 and 2): Use `particles_group_names=["particles_class_0", "particles_class_2"]` (list with all classes)
       c. Get volume from the best class in that cluster (volume_class_X where X has best resolution)
       d. Run heterogeneous refinement with K={k_value} using:
          * particles_group_names: List of particles_class_X for each class in the cluster
          * volume_group_name: volume_class_X with best resolution
       e. Wait for completion → get new hetero job
       f. **CRITICAL - DO NOT STOP HERE**: After the heterogeneous refinement job completes, you MUST continue:
          - Extract density maps from the new hetero job using `extract_density_maps` tool
          - Compare all density maps using `compare_all_densities` tool
          - Get class resolutions using `get_hetero_class_resolutions` tool
          - Check if any class passes the resolution threshold
       g. **Recursively apply the same logic based on comparison and resolution results:**
          - If NO cluster passes resolution threshold → use particles_all_classes from that hetero job + best volume → run homogeneous refinement → record final job UID → branch COMPLETE
          - If 1 cluster → use particles_all_classes from that hetero job + best volume → run homogeneous refinement → record final job UID → branch COMPLETE
          - If multiple clusters AND at least one passes threshold → split into more branches → repeat steps a-g recursively for EACH new cluster
     * **CRITICAL**: Do NOT stop after a heterogeneous refinement job completes. ALWAYS extract density maps, compare, check resolutions, and continue recursively until the branch terminates.
     * Continue this recursive expansion until EVERY branch reaches only 1 cluster OR no cluster passes resolution threshold
     * Each branch that reaches 1 cluster OR no cluster passes threshold gets a final homogeneous refinement job UID

4. **Final Homogeneous Refinement** (for each terminal branch):
   - A branch terminates when comparison shows only 1 cluster in a heterogeneous refinement job
   - Extract which classes belong to this cluster from the comparison results
   - **CRITICAL**: Use particles based on cluster composition:
     * If cluster has only 1 class: Use `particles_class_X` for that specific class
     * If cluster has multiple classes: Use particles from all classes in this cluster (particles_class_X for each class)
   - Use the best volume from the classes in this cluster (volume_class_X with best resolution)
   - Run homogeneous refinement using the appropriate particles group(s) and best volume
   - Record the final refinement job UID for this branch
   - This completes the depth analysis for this branch

**IMPORTANT TREE STRUCTURE LOGIC:**
- Each heterogeneous refinement can split into multiple branches if multiple clusters are found
- Each branch is processed independently and recursively
- The tree expands until every branch reaches 1 cluster
- Each terminal branch (reaching 1 cluster) gets a final homogeneous refinement job UID
- You must track and record ALL final refinement job UIDs from ALL terminal branches
- Process ALL starting clusters from the heterogeneity analysis results independently

## Tool Usage:

- **read_input_json**: Read JSON file from refinement or heterogeneity job
  * Optional: config_path
  * If reading from heterogeneity_analysis_results_*.json:
    * Returns: success, source, num_clusters, clusters (array with refinement_job_uid, particles_job_uid, volume_job_uid, particles_group_name, volume_group_name for each cluster)
  * If reading from reconstruction_results_*.json:
    * Returns: refinement_job_uid, particles_job_uid, volume_job_uid

- **run_heterogeneous_refinement**: Run heterogeneous refinement with K classes
  * Required: particles_job_uid, volume_job_uid, k (default: {k_value})
  * Optional: particles_group_names (list, e.g., ["particles_class_0", "particles_class_2"] for multiple classes, or ["particles_class_1"] for single class)
  * Optional: particles_group_name (legacy single group name), volume_group_name, project_uid, workspace_uid
  * **CRITICAL**: For clusters with multiple classes, use particles_group_names with a list of particles_class_X for each class
  * Returns: hetero_job_uid, status

- **extract_density_maps**: Get job directory containing density maps
  * Required: hetero_job_uid (can pass just "JXXX")
  * Returns: output_folder, num_maps_extracted, map_files

- **compare_all_densities**: Compare all density maps in a folder
  * Required: folder (path to folder with *_volume.mrc files)
  * Optional: voxel_size, alg_type, resolution_threshold, n_clusters, cluster_method
  * Returns: clustering results with number of clusters and which maps belong to each cluster
  * **IMPORTANT**: Map names contain class IDs (e.g., "J34_class_03_00042_volume.mrc" → class 3)
  * Extract class IDs from map names to determine which classes belong to each cluster
  * Use this to select the correct particles group: particles_class_X for single class, or particles_class_X for each class if multiple

- **run_homogeneous_refinement**: Run homogeneous refinement
  * Required: particles_job_uid, volume_job_uid
  * Optional: particles_group_name (e.g., "particles_all_classes"), volume_group_name (e.g., "volume_class_0")
  * Returns: job_uid, status

- **get_job_status**: Check status of a job
- **wait_for_job**: Wait for job completion
- **get_job_log**: Read and analyze job logs

## CRITICAL: Tree-Based Recursive Refinement Logic
- Always validate with comparison tool after EACH heterogeneous refinement
- Extract which classes belong to each cluster from comparison results (map names contain class IDs)
- **CRITICAL WORKFLOW**: For EACH cluster from initial heterogeneity analysis:
  * ALWAYS start with heterogeneous refinement (do NOT skip to homogeneous refinement)
  * After heterogeneous refinement completes, you MUST:
    1. Extract density maps using `extract_density_maps` tool
    2. Compare all density maps using `compare_all_densities` tool
    3. Get class resolutions using `get_hetero_class_resolutions` tool
    4. Check if any class passes the resolution threshold (default: 10.0 Å, check config for actual value)
  * **DO NOT STOP** after a heterogeneous refinement job completes - you MUST continue with extraction, comparison, and recursive refinement
  * **If comparison shows multiple clusters**: Create branches - one for each cluster, continue with heterogeneous refinement for each
  * **If comparison shows only 1 cluster**: This is when you do homogeneous refinement using `particles_all_classes` from that hetero job + best volume
  * **If NO cluster passes resolution threshold** (all classes have resolution worse than threshold): Also run homogeneous refinement using `particles_all_classes` from that hetero job + best volume (even if multiple clusters exist)
- **CRITICAL PARTICLE SELECTION FOR HETEROGENEOUS REFINEMENT**:
  * When running heterogeneous refinement for a cluster: Use particles_class_X for each class in the cluster
  * If cluster has only 1 class: Use `particles_group_names=["particles_class_X"]` (list with single element)
  * If cluster has multiple classes: Use `particles_group_names=["particles_class_X", "particles_class_Y", ...]` (list with all classes)
- **CRITICAL PARTICLE SELECTION FOR HOMOGENEOUS REFINEMENT**:
  * When a heterogeneous refinement job shows only 1 cluster: Use `particles_all_classes` from that hetero job (NOT particles_class_X)
  * When NO cluster passes resolution threshold: Use `particles_all_classes` from that hetero job (NOT particles_class_X)
  * This uses ALL particles from the heterogeneous refinement job, not just specific classes
- Each branch recursively continues with heterogeneous refinement until it reaches only 1 cluster OR no cluster passes resolution threshold
- When a branch reaches 1 cluster OR no cluster passes threshold, use particles_all_classes from the hetero job + best volume → homogeneous refinement
- Track ALL final refinement job UIDs from ALL terminal branches
- The workflow creates a tree structure where each split represents multiple clusters found

Remember: Always follow the Thought → Action → Observation pattern!
Think carefully about cluster validation and iteration logic before proceeding.

**CRITICAL REMINDER**: After EACH heterogeneous refinement job completes, you MUST:
1. Extract density maps (extract_density_maps)
2. Compare densities (compare_all_densities)
3. Get class resolutions (get_hetero_class_resolutions)
4. Continue recursively based on results

DO NOT stop after a heterogeneous refinement job completes - always continue with the recursive workflow until the branch terminates (1 cluster OR no cluster passes threshold)."""
    
    # =================================================================
    # Tool Implementation Methods
    # =================================================================
    
    def _read_input_json_tool(self, tool_input: str) -> str:
        """
        Read JSON file from either refinement job or heterogeneity job.
        Prefers heterogeneity job if available.
        """
        try:
            params = self._parse_tool_input(tool_input)
            config_path = params.get("config_path", "")
            
            # If config_path is provided, read from config
            if config_path:
                config_file = Path(config_path)
                if config_file.exists():
                    with open(config_file, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                        # Check if config specifies which JSON to read
                        if "input_json_path" in config_data:
                            json_path = Path(config_data["input_json_path"])
                            if json_path.exists():
                                with open(json_path, "r", encoding="utf-8") as jf:
                                    return json.dumps(json.load(jf))
            
            # Otherwise, search for JSON files in outputs directory
            outputs_path = Path("outputs")
            if not outputs_path.exists():
                return json.dumps({
                    "success": False,
                    "error": "Outputs directory not found"
                })
            
            # First, try to find heterogeneity analysis results (preferred)
            hetero_files = list(outputs_path.glob("heterogeneity_analysis_results_*.json"))
            if hetero_files:
                latest_hetero = max(hetero_files, key=lambda f: f.stat().st_mtime)
                self.logger.info(f"📄 Found heterogeneity analysis results: {latest_hetero}")
                with open(latest_hetero, "r", encoding="utf-8") as f:
                    hetero_data = json.load(f)
                    
                    # Extract all clusters from final_refinement_jobs
                    final_refinement_jobs = hetero_data.get("final_refinement_jobs", [])
                    if not final_refinement_jobs:
                        # Try filtered_groups as fallback
                        final_refinement_jobs = hetero_data.get("filtered_groups", [])
                        self.logger.info(f"📊 Using filtered_groups as fallback, found {len(final_refinement_jobs)} groups")
                    
                    if final_refinement_jobs:
                        # Return all clusters for depth analysis
                        # Use refinement_job_uid (J83/J84) as the job source for both particles and volume
                        # The refinement job contains the final refined particles and volumes
                        # IMPORTANT: Refinement jobs (non-uniform or homogeneous) use standard group names:
                        # - particles (not particles_class_X)
                        # - volume (not volume_class_X)
                        clusters = []
                        for cluster in final_refinement_jobs:
                            refinement_job_uid = cluster.get("refinement_job_uid")  # J83/J84 - final refinement job
                            hetero_job_uid = cluster.get("hetero_job_uid")  # J82 - heterogeneous refinement job (for reference)
                            
                            # For refinement jobs, use standard group names (not class-specific)
                            # Class-specific names (particles_class_X, volume_class_X) only exist in hetero jobs
                            clusters.append({
                                "group_id": cluster.get("group_id"),
                                "refinement_job_uid": refinement_job_uid,  # J83/J84 - final refinement job
                                "particles_job_uid": refinement_job_uid,  # Particles from refinement job (J83/J84)
                                "volume_job_uid": refinement_job_uid,  # Volume from refinement job (J83/J84)
                                "particles_group_names": ["particles"],  # Refinement jobs use standard "particles" group
                                "particles_group_name": "particles",  # Standard group name for refinement jobs
                                "volume_group_name": "volume",  # Standard group name for refinement jobs
                                "hetero_job_uid": hetero_job_uid,  # J82 - the source hetero job (for reference)
                                "class_ids": cluster.get("class_ids", [])  # e.g., [0] or [0, 2] - kept for reference
                            })
                        
                        result = {
                            "success": True,
                            "source": "heterogeneity_analysis",
                            "num_clusters": len(clusters),
                            "clusters": clusters,
                            "project_uid": hetero_data.get("project_uid"),
                            "workspace_uid": hetero_data.get("workspace_uid")
                        }
                        self.logger.info(f"📊 Found {len(clusters)} clusters for depth analysis")
                        self.logger.info(f"📊 Clusters will use refinement jobs: {[c.get('refinement_job_uid') for c in clusters]}")
                        return json.dumps(result)
                    else:
                        self.logger.warning(f"⚠️ Heterogeneity analysis results file found but no final_refinement_jobs or filtered_groups found in {latest_hetero}")
                        return json.dumps({
                            "success": False,
                            "error": f"Heterogeneity analysis results file found but no clusters available in {latest_hetero}"
                        })
            
            # If no heterogeneity results, try reconstruction results
            self.logger.warning("⚠️ No heterogeneity_analysis_results_*.json files found, falling back to reconstruction results")
            recon_files = list(outputs_path.glob("reconstruction_results_*.json"))
            if recon_files:
                latest_recon = max(recon_files, key=lambda f: f.stat().st_mtime)
                self.logger.info(f"📄 Found reconstruction results: {latest_recon}")
                with open(latest_recon, "r", encoding="utf-8") as f:
                    recon_data = json.load(f)
                    
                    # Extract relevant information
                    job_uids = recon_data.get("job_uids", {})
                    result = {
                        "success": True,
                        "source": "reconstruction",
                        "refinement_job_uid": job_uids.get("homogeneous_refinement") or job_uids.get("homogeneous_reconstruction"),
                        "particles_job_uid": recon_data.get("input_particles_job_uid"),
                        "volume_job_uid": job_uids.get("final_volume") or job_uids.get("homogeneous_refinement") or job_uids.get("homogeneous_reconstruction"),
                        "project_uid": recon_data.get("project_uid"),
                        "workspace_uid": recon_data.get("workspace_uid")
                    }
                    return json.dumps(result)
            
            return json.dumps({
                "success": False,
                "error": "No suitable JSON files found in outputs directory"
            })
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("read_input_json", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _run_heterogeneous_refinement_tool(self, tool_input: str) -> str:
        """
        Run heterogeneous refinement with K classes using particles and volume from a refinement job.
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            particles_job_uid = params.get("particles_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            k = int(params.get("k", 4))
            particles_group_names = params.get("particles_group_names")  # List of particle group names
            particles_group_name = params.get("particles_group_name", "")  # Legacy single group name
            volume_group_name = params.get("volume_group_name", "")
            
            # Convert single particles_group_name to list if particles_group_names not provided
            if not particles_group_names and particles_group_name:
                particles_group_names = [particles_group_name]
            
            if not particles_job_uid or not volume_job_uid:
                missing = []
                if not particles_job_uid:
                    missing.append("particles_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}"
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            symmetry = self._get_refinement_symmetry()
            # For heterogeneous refinement, use C1 (no symmetry)
            hetero_symmetry = "C1"
            
            self.logger.info(f"🔬 Running heterogeneous refinement with K={k}")
            
            # For heterogeneous refinement, we need to connect:
            # - Particles from the refinement job (particles_all_classes or particles group)
            # - Volume from the refinement job (volume or volume_class_0), repeated K times
            
            try:
                from cryosparc.tools import CryoSPARC
                project = self.cryosparc_tools.cs.find_project(project_uid)
                workspace = project.find_workspace(workspace_uid)
                
                # Handle multiple particle groups (for clusters with multiple classes)
                if particles_group_names and len(particles_group_names) > 1:
                    # Multiple particle groups - create list of connections
                    self.logger.info(f"📦 Using particles from {len(particles_group_names)} classes: {particles_group_names}")
                    particle_connections = [(particles_job_uid, group_name) for group_name in particles_group_names]
                elif particles_group_names and len(particles_group_names) == 1:
                    # Single particle group from list
                    particles_slot = particles_group_names[0]
                    particle_connections = None  # Will use single connection
                elif particles_group_name:
                    # Legacy single group name
                    particles_slot = particles_group_name
                    particle_connections = None  # Will use single connection
                else:
                    # Auto-detect particles slot
                    particles_slot = "particles_all_classes"  # Default for hetero jobs
                    try:
                        particles_job = project.find_job(particles_job_uid)
                        particles_job.refresh()
                        particles_doc = getattr(particles_job, "doc", {}) or {}
                        particles_outputs = particles_doc.get("output_result_groups", []) or []
                        for group in particles_outputs:
                            name = group.get("name") or ""
                            group_type = (group.get("type") or "").lower()
                            if "particle" in group_type:
                                if "all_classes" in name.lower():
                                    particles_slot = name
                                    break
                                elif particles_slot == "particles_all_classes":
                                    particles_slot = name  # Fallback to first particle group
                    except Exception:
                        pass  # Use default
                    particle_connections = None  # Will use single connection
                
                # Get volume slot from volume job
                if volume_group_name:
                    volume_slot = volume_group_name
                else:
                    volume_slot = "volume"  # Default
                    try:
                        volume_job = project.find_job(volume_job_uid)
                        volume_job.refresh()
                        volume_doc = getattr(volume_job, "doc", {}) or {}
                        volume_outputs = volume_doc.get("output_result_groups", []) or []
                        for group in volume_outputs:
                            name = group.get("name") or ""
                            group_type = (group.get("type") or "").lower()
                            if "volume" in group_type:
                                # Prefer volume_class_0, but accept any volume
                                if "class_0" in name.lower() or volume_slot == "volume":
                                    volume_slot = name
                    except Exception:
                        pass  # Use default
                
                # Create volume connections: same volume job, same volume slot, repeated K times
                volume_connections = [
                    (volume_job_uid, volume_slot) 
                    for _ in range(k)
                ]
                
                # Build connections - handle multiple particle groups if needed
                if particle_connections:
                    # Multiple particle connections (for clusters with multiple classes)
                    connections = {
                        "particles": particle_connections,  # List of (job_uid, group_name) tuples
                        "volume": volume_connections
                    }
                    self.logger.info(f"🔗 Connecting heterogeneous refinement:")
                    self.logger.info(f"   Particles: from {particles_job_uid} (groups: {particles_group_names})")
                    self.logger.info(f"   Volumes: from {volume_job_uid} (group: {volume_slot}, repeated {k} times)")
                else:
                    # Single particle connection
                    connections = {
                        "particles": (particles_job_uid, particles_slot),
                        "volume": volume_connections
                    }
                    self.logger.info(f"🔗 Connecting heterogeneous refinement:")
                    self.logger.info(f"   Particles: from {particles_job_uid} (group: {particles_slot})")
                    self.logger.info(f"   Volumes: from {volume_job_uid} (group: {volume_slot}, repeated {k} times)")
                
                job_params = {}
                
                hetero_job = workspace.create_job(
                    "hetero_refine",
                    connections=connections,
                    params=job_params
                )
                
                # Queue the job
                used_lane = self.cryosparc_tools._queue_job_with_lane_fallback(
                    hetero_job,
                    log_prefix="⚙️ No lane specified; using lane",
                    logger=self.logger,
                )
                
                hetero_job_uid = hetero_job.uid
                self.logger.info(f"✅ Queued heterogeneous refinement job: {hetero_job_uid}")
                
                # Record tool execution
                hetero_params = {
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "particles_job_uid": particles_job_uid,
                    "volume_job_uid": volume_job_uid,
                    "volume_group": volume_slot,
                    "num_classes": k,
                    "symmetry": "C1"
                }
                if particle_connections:
                    hetero_params["particles_groups"] = particles_group_names
                else:
                    hetero_params["particles_group"] = particles_slot
                self._record_tool_execution("heterogeneous_refinement", hetero_params)
                
                # Wait for completion
                hetero_result = self.cryosparc_tools.wait_for_job_completion(
                    project_uid=project_uid,
                    job_uid=hetero_job_uid,
                    timeout=self.config.job_management.default_timeout,
                    check_interval=self.config.job_management.status_check_interval
                )
                
                hetero_status = hetero_result.get("status", "unknown")
                if hetero_status != "completed":
                    error_msg = hetero_result.get("error") or f"Status: {hetero_status}"
                    self.logger.error(f"❌ Heterogeneous refinement did not complete for K={k}: {error_msg}")
                    hetero_result = {
                        "success": False,
                        "error": f"Heterogeneous refinement did not complete: {error_msg}",
                        "job_uid": hetero_job_uid,
                        "status": hetero_status
                    }
                else:
                    hetero_result = {
                        "success": True,
                        "job_uid": hetero_job_uid,
                        "status": "completed"
                    }
                
                self._record_tool_execution("heterogeneous_refinement", hetero_params, result=hetero_result)
                
            except Exception as e:
                self.logger.error(f"❌ Failed to create heterogeneous refinement: {str(e)}")
                hetero_result = {"success": False, "error": str(e)}
            
            if not hetero_result.get("success", False):
                error_msg = hetero_result.get("error") or "Unknown error"
                return json.dumps({
                    "success": False,
                    "error": f"Heterogeneous refinement failed: {error_msg}",
                    "k": k
                })
            
            hetero_job_uid = hetero_result["job_uid"]
            self.logger.info(f"✅ Heterogeneous refinement completed for K={k}, job: {hetero_job_uid}")
            
            return json.dumps({
                "success": True,
                "k": k,
                "hetero_job_uid": hetero_job_uid,
                "status": "completed"
            })
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("run_heterogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _extract_density_maps_tool(self, tool_input: str) -> str:
        """
        Get the job directory containing density map files (*_volume.mrc) from a heterogeneous refinement job.
        Returns the job directory directly without copying files.
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            hetero_job_uid = params.get("hetero_job_uid") or params.get("job_uid")
            
            if not hetero_job_uid:
                input_stripped = tool_input.strip().strip('"\'')
                if input_stripped.startswith("J") and len(input_stripped) <= 10:
                    hetero_job_uid = input_stripped
            
            if not hetero_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: hetero_job_uid. You can pass just the job UID (e.g., 'JXXX') or JSON with hetero_job_uid parameter."
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            
            job_info = self.cryosparc_tools.get_job_output_directory(project_uid, hetero_job_uid)
            job_dir = Path(job_info["job_directory"])
            
            self.logger.info(f"📦 Using job directory directly for density maps: {job_dir}")
            
            volume_files = list(job_dir.glob("*_volume.mrc"))
            if not volume_files:
                volume_files = list(job_dir.rglob("*_volume.mrc"))
            
            if not volume_files:
                return json.dumps({
                    "success": False,
                    "error": f"No *_volume.mrc files found in job directory: {job_dir}",
                    "job_directory": str(job_dir)
                })
            
            map_files = [str(vol_file) for vol_file in volume_files]
            self.logger.info(f"  Found {len(map_files)} density map(s) in job directory")
            
            result = {
                "success": True,
                "hetero_job_uid": hetero_job_uid,
                "output_folder": str(job_dir),
                "num_maps_extracted": len(map_files),
                "map_files": map_files
            }
            
            self._record_tool_execution("extract_density_maps", {"hetero_job_uid": hetero_job_uid}, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("extract_density_maps", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _get_hetero_class_resolutions_tool(self, tool_input: str) -> str:
        """
        Get resolution information for each class in a heterogeneous refinement job.
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            job_uid = params.get("job_uid") or params.get("hetero_job_uid") or params.get("refinement_job_uid")
            
            if not job_uid:
                input_stripped = tool_input.strip().strip('"\'')
                if input_stripped.startswith("J") and len(input_stripped) <= 10:
                    job_uid = input_stripped
            
            if not job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: job_uid. You can pass just the job UID (e.g., 'JXXX') or JSON with job_uid parameter."
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            
            class_resolutions = self.cryosparc_tools.get_heterogeneous_refinement_class_resolutions(project_uid, job_uid)
            
            if not class_resolutions.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get class resolutions: {class_resolutions.get('error', 'Unknown error')}"
                })
            
            result = {
                "success": True,
                "job_uid": job_uid,
                "num_classes": class_resolutions.get("num_classes", 0),
                "classes": class_resolutions.get("classes", [])
            }
            
            self._record_tool_execution("get_hetero_class_resolutions", {"job_uid": job_uid, "project_uid": project_uid}, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("get_hetero_class_resolutions", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _run_homogeneous_refinement_tool(self, tool_input: str) -> str:
        """
        Run homogeneous refinement using particles and volume from a job.
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            particles_job_uid = params.get("particles_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            particles_group_name = params.get("particles_group_name", "")
            volume_group_name = params.get("volume_group_name", "")
            refine_defocus_refine = params.get("refine_defocus_refine", True)
            refine_ctf_global_refine = params.get("refine_ctf_global_refine", True)
            
            if not particles_job_uid or not volume_job_uid:
                missing = []
                if not particles_job_uid:
                    missing.append("particles_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}"
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            symmetry = self._get_refinement_symmetry()
            refine_res_init = self._get_refinement_res_init()
            
            # If particles_group_name is "particles_all_classes" or empty, use particles_all_classes from hetero job
            # This is the case when a heterogeneous refinement job shows only 1 cluster
            if not particles_group_name or particles_group_name == "particles_all_classes":
                particles_group_name = "particles_all_classes"
                self.logger.info(f"🔧 Running homogeneous refinement")
                self.logger.info(f"   Particles: from {particles_job_uid} (group: particles_all_classes - using ALL particles from hetero job)")
                self.logger.info(f"   Volume: from {volume_job_uid}" + (f" (group: {volume_group_name})" if volume_group_name else ""))
            else:
                self.logger.info(f"🔧 Running homogeneous refinement")
                self.logger.info(f"   Particles: from {particles_job_uid} (group: {particles_group_name})")
                self.logger.info(f"   Volume: from {volume_job_uid}" + (f" (group: {volume_group_name})" if volume_group_name else ""))
            
            # Prepare refinement parameters
            refine_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": particles_job_uid,
                "volume_job_uid": volume_job_uid,
                "symmetry": symmetry,
                "refine_defocus_refine": refine_defocus_refine,
                "refine_ctf_global_refine": refine_ctf_global_refine,
                "wait_for_completion": False,
                "timeout": self.config.job_management.default_timeout,
                "check_interval": self.config.job_management.status_check_interval
            }
            
            if refine_res_init is not None:
                refine_params["refine_res_init"] = float(refine_res_init)
            
            # Always set particles_group_name (either particles_all_classes or specific class)
            refine_params["particles_group_name"] = particles_group_name
            if volume_group_name:
                refine_params["volume_group_name"] = volume_group_name
            
            # Run homogeneous refinement
            refine_result = self.cryosparc_tools.homogeneous_refinement(**refine_params)
            
            if not refine_result.get("success", False):
                error_msg = refine_result.get("error") or "Unknown error"
                self.logger.error(f"❌ Homogeneous refinement failed: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Homogeneous refinement failed: {error_msg}",
                    "refinement_result": refine_result
                })
            
            job_uid = refine_result.get("job_uid")
            self.logger.info(f"✅ Homogeneous refinement job queued: {job_uid}")
            
            result = {
                "success": True,
                "job_uid": job_uid,
                "particles_job_uid": particles_job_uid,
                "volume_job_uid": volume_job_uid,
                "particles_group_name": particles_group_name,
                "volume_group_name": volume_group_name,
                "status": "queued",
                "refinement_result": refine_result
            }
            
            tool_params = {
                "particles_job_uid": particles_job_uid,
                "volume_job_uid": volume_job_uid,
                "particles_group_name": particles_group_name,
                "volume_group_name": volume_group_name,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            if refine_res_init is not None:
                tool_params["refine_res_init"] = refine_res_init
            
            self._record_tool_execution("run_homogeneous_refinement", tool_params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("run_homogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)

