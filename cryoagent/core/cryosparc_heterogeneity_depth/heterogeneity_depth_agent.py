"""ReAct-based heterogeneity depth analysis agent for CryoEM 3D reconstruction."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..cryosparc_common_tools import CryoSPARCCommonTools
from ..base_react_agent import BaseReActAgent
from .heterogeneity_depth_tools import HeterogeneityDepthTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...tools.alignment_tools.compare_all_densities_tool import CompareAllDensitiesTool
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


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
        self._last_class_resolutions: Optional[Dict[str, Any]] = None
        self._compare_densities_delegate = CompareAllDensitiesTool.create_compare_all_densities_tool(
            compare_script=self._get_stage_param("script_paths", "compare_all_densities_script", None),
            align_script=self._get_stage_param("script_paths", "align_and_compare_script", None),
            default_voxel_size=self._get_stage_param("density_comparison", "voxel_size", 5.0),
            default_alg_type=self._get_stage_param("density_comparison", "alg_type", "global"),
            default_resolution_threshold=self._get_stage_param("density_comparison", "resolution_threshold", None),
        )
        
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
            HeterogeneityDepthTools.create_run_non_uniform_refinement_tool(self),
            HeterogeneityDepthTools.create_get_fsc_info_tool(self),
            HeterogeneityDepthTools.create_get_job_status_tool(self),
            HeterogeneityDepthTools.create_wait_for_job_tool(self),
            HeterogeneityDepthTools.create_get_job_log_tool(self),
            CryoSPARCCommonTools.create_search_cryosparc_forum_tool(self),
        ]

        tools.append(HeterogeneityDepthTools.create_compare_all_densities_tool(self))

        return tools
    
    def _get_resolution_filter_threshold(self) -> float:
        """Resolution cutoff (Å): classes/clusters must be better (lower) than this to continue."""
        return float(self._get_stage_param("heterogeneity_depth_analysis", "resolution_threshold", 10.0))
    
    def _classify_resolution(self, resolution_angstroms: float) -> Dict[str, Any]:
        """Label a class GOOD/BAD relative to the depth-analysis resolution threshold."""
        threshold = self._get_resolution_filter_threshold()
        is_good = resolution_angstroms < threshold
        return {
            "passes_threshold": is_good,
            "quality": "GOOD" if is_good else "BAD",
            "action": "continue_hetero_or_final_non_uniform" if is_good else "discard_no_further_processing",
        }
    
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
    
    def _get_system_prompt_context(self) -> Dict[str, Any]:
        """Build template variables for cryosparc/heterogeneity_depth/system.md."""
        k_value = self._get_stage_param("heterogeneity_depth_analysis", "k", 4)
        resolution_threshold = self._get_stage_param("heterogeneity_depth_analysis", "resolution_threshold", 10.0)
        return {
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
            "k_value": k_value,
            "resolution_threshold": resolution_threshold,
        }

    def _get_react_system_prompt(self) -> str:
        """Get the heterogeneity depth analysis-specific ReAct system prompt."""
        return self._compose_stage_system_prompt(
            "cryosparc/heterogeneity_depth/system.md",
            self._get_system_prompt_context(),
        )
    
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
            outputs_path = Path(getattr(self, 'outputs_dir', 'outputs'))
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

            self.logger.info("📊 Auto-running class resolutions + density comparison after hetero job...")
            post_analysis = self._analyze_hetero_densities_after_completion(hetero_job_uid)

            result: Dict[str, Any] = {
                "success": True,
                "k": k,
                "hetero_job_uid": hetero_job_uid,
                "status": "completed",
                "auto_density_analysis": True,
                "density_folder": post_analysis.get("density_folder"),
                "density_maps": post_analysis.get("density_maps", []),
                "density_comparison": post_analysis.get("density_comparison"),
            }
            class_resolutions = post_analysis.get("class_resolutions")
            if class_resolutions:
                result["class_resolutions"] = class_resolutions
                result["good_classes"] = class_resolutions.get("good_classes", [])
                result["bad_classes"] = class_resolutions.get("bad_classes", [])
                result["next_action"] = class_resolutions.get("next_action")
                if class_resolutions.get("fallback_non_uniform"):
                    result["fallback_non_uniform"] = class_resolutions["fallback_non_uniform"]

            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("run_heterogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)

    def _analyze_hetero_densities_after_completion(self, hetero_job_uid: str) -> Dict[str, Any]:
        """Run class resolutions, map extraction, and density comparison after hetero completes."""
        analysis: Dict[str, Any] = {
            "class_resolutions": None,
            "density_folder": None,
            "density_maps": [],
            "density_comparison": None,
        }
        try:
            res_raw = self._get_hetero_class_resolutions_tool(
                json.dumps({"job_uid": hetero_job_uid})
            )
            res_data = json.loads(res_raw) if isinstance(res_raw, str) else res_raw
            if res_data.get("success"):
                analysis["class_resolutions"] = res_data
            else:
                self.logger.warning(
                    f"Auto class resolutions failed for {hetero_job_uid}: {res_data.get('error')}"
                )
                return analysis

            extract_raw = self._extract_density_maps_tool(
                json.dumps({"hetero_job_uid": hetero_job_uid})
            )
            extract_data = json.loads(extract_raw) if isinstance(extract_raw, str) else extract_raw
            if not extract_data.get("success"):
                self.logger.warning(
                    f"Auto density extraction failed for {hetero_job_uid}: {extract_data.get('error')}"
                )
                return analysis

            analysis["density_folder"] = extract_data.get("output_folder")
            analysis["density_maps"] = extract_data.get("map_files", [])

            compare_raw = self._compare_all_densities_tool(
                json.dumps({"folder": analysis["density_folder"]})
            )
            analysis["density_comparison"] = (
                compare_raw if isinstance(compare_raw, str) else json.dumps(compare_raw)
            )
        except Exception as exc:
            self.logger.warning(
                f"Post-hetero density analysis failed for {hetero_job_uid}: {exc}"
            )
        return analysis
    
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
    
    def _compare_all_densities_tool(self, tool_input: str) -> str:
        """
        Compare density maps and filter clusters using the depth-analysis resolution threshold.
        Auto-injects class resolutions from the most recent get_hetero_class_resolutions call.
        """
        try:
            params = self._parse_tool_input(tool_input)
            threshold = self._get_resolution_filter_threshold()

            # LLM may pass folder path as "input" instead of "folder"
            if not params.get("folder") and params.get("input"):
                candidate = params["input"]
                if isinstance(candidate, str) and not candidate.strip().startswith("{"):
                    params["folder"] = candidate

            if params.get("class_resolutions") is None and self._last_class_resolutions:
                params["class_resolutions"] = self._last_class_resolutions.get("classes", [])

            # Compare only needs class_id + resolution; drop bulky FSC curves from LLM copies
            if params.get("class_resolutions"):
                slim = []
                for class_data in params["class_resolutions"]:
                    if isinstance(class_data, dict):
                        slim.append({
                            "class_id": class_data.get("class_id"),
                            "resolution_angstroms": class_data.get("resolution_angstroms"),
                            "group_name": class_data.get("group_name"),
                        })
                params["class_resolutions"] = [c for c in slim if c.get("class_id") is not None]

            if params.get("resolution_filter_threshold") is None:
                params["resolution_filter_threshold"] = threshold

            if params.get("voxel_size") is None:
                params["voxel_size"] = self._get_stage_param("density_comparison", "voxel_size", 5.0)
            if params.get("alg_type") is None:
                params["alg_type"] = self._get_stage_param("density_comparison", "alg_type", "global")
            if params.get("n_clusters") is None:
                n_clusters = self._get_stage_param("density_comparison", "n_clusters", None)
                if n_clusters is not None:
                    params["n_clusters"] = n_clusters
            if params.get("cluster_method") is None:
                params["cluster_method"] = self._get_stage_param("density_comparison", "cluster_method", "spectral")

            if not params.get("class_resolutions"):
                self.logger.warning(
                    "compare_all_densities called without class_resolutions — "
                    "call get_hetero_class_resolutions on the same hetero job first"
                )

            result = self._compare_densities_delegate.func(json.dumps(params))
            self._record_tool_execution("compare_all_densities", params, result=result)
            return result
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution(
                "compare_all_densities",
                params if "params" in locals() else {},
                error=str(e),
            )
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
            
            threshold = self._get_resolution_filter_threshold()
            raw_classes = class_resolutions.get("classes", [])
            annotated_classes = []
            good_classes = []
            bad_classes = []

            for class_data in raw_classes:
                entry = dict(class_data)
                resolution = entry.get("resolution_angstroms")
                if resolution is not None:
                    classification = self._classify_resolution(float(resolution))
                    entry.update(classification)
                    class_id = entry.get("class_id")
                    summary = {
                        "class_id": class_id,
                        "group_name": entry.get("group_name"),
                        "resolution_angstroms": resolution,
                        **classification,
                    }
                    if classification["passes_threshold"]:
                        good_classes.append(summary)
                    else:
                        bad_classes.append(summary)
                annotated_classes.append(entry)

            self._last_class_resolutions = {
                "job_uid": job_uid,
                "resolution_threshold_angstroms": threshold,
                "classes": annotated_classes,
            }

            best_class = None
            if not good_classes:
                best_class = min(
                    (c for c in annotated_classes if c.get("resolution_angstroms") is not None),
                    key=lambda c: c["resolution_angstroms"],
                    default=None,
                )
                next_action = "terminate_non_uniform_all_particles_best_volume"
            elif len(good_classes) == 1:
                next_action = "compare_densities_then_non_uniform_if_one_kept_cluster"
            else:
                next_action = "compare_densities_then_hetero_per_kept_cluster_only"

            result = {
                "success": True,
                "job_uid": job_uid,
                "num_classes": class_resolutions.get("num_classes", 0),
                "resolution_threshold_angstroms": threshold,
                "classes": annotated_classes,
                "good_classes": good_classes,
                "bad_classes": bad_classes,
                "num_good_classes": len(good_classes),
                "num_bad_classes": len(bad_classes),
                "filter_rule": f"GOOD if resolution < {threshold} Å; BAD if ≥ {threshold} Å — discard bad only when good classes exist",
                "next_action": next_action,
            }
            if not good_classes and best_class is not None:
                result["fallback_non_uniform"] = {
                    "particles_group_names": ["particles_all_classes"],
                    "volume_group_name": best_class.get("group_name"),
                    "best_class_id": best_class.get("class_id"),
                    "best_resolution_angstroms": best_class.get("resolution_angstroms"),
                    "reason": "No good classes — refine all particles with best available volume",
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

    def _run_non_uniform_refinement_tool(self, tool_input: str) -> str:
        """Run non-uniform refinement for a converged good cluster."""
        try:
            params = self._parse_tool_input(tool_input)

            hetero_job_uid = params.get("hetero_job_uid")
            particles_group_names = params.get("particles_group_names")
            volume_group_name = params.get("volume_group_name")

            if not hetero_job_uid or not particles_group_names or not volume_group_name:
                missing = []
                if not hetero_job_uid:
                    missing.append("hetero_job_uid")
                if not particles_group_names:
                    missing.append("particles_group_names")
                if not volume_group_name:
                    missing.append("volume_group_name")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}"
                })

            if isinstance(particles_group_names, str):
                try:
                    parsed = json.loads(particles_group_names)
                    particles_group_names = parsed if isinstance(parsed, list) else [particles_group_names]
                except (json.JSONDecodeError, ValueError, TypeError):
                    particles_group_names = [particles_group_names]
            if not isinstance(particles_group_names, list):
                particles_group_names = [particles_group_names]

            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            refine_res_init = params.get("refine_res_init")
            if refine_res_init is None:
                refine_res_init = self._get_refinement_res_init()
            symmetry = self._get_refinement_symmetry()

            self.logger.info(
                f"Running non-uniform refinement: hetero={hetero_job_uid}, "
                f"particles={particles_group_names}, volume={volume_group_name}"
            )

            if len(particles_group_names) > 1:
                project = self.cryosparc_tools.cs.find_project(project_uid)
                workspace = project.find_workspace(workspace_uid)
                job_params = {
                    "refine_do_init_scale_est": True,
                    "refine_symmetry_do_align": True,
                    "refine_defocus_refine": True,
                    "refine_ctf_global_refine": True,
                }
                if symmetry and symmetry != "C1":
                    job_params["refine_symmetry"] = symmetry
                if refine_res_init is not None:
                    job_params["refine_res_init"] = float(refine_res_init)

                particle_connections = [(hetero_job_uid, group_name) for group_name in particles_group_names]
                connections = {
                    "particles": particle_connections,
                    "volume": (hetero_job_uid, volume_group_name),
                }
                job = workspace.create_job("nonuniform_refine_new", connections=connections, params=job_params)
                used_lane = self.cryosparc_tools._queue_job_with_lane_fallback(
                    job, log_prefix="No lane specified; using lane", logger=self.logger
                )
                refine_result = {
                    "success": True,
                    "job_uid": job.uid,
                    "job_type": "nonuniform_refine_new",
                    "lane": used_lane,
                }
            else:
                refine_params = {
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "particles_job_uid": hetero_job_uid,
                    "volume_job_uid": hetero_job_uid,
                    "symmetry": symmetry,
                    "refine_defocus_refine": True,
                    "refine_ctf_global_refine": True,
                    "particles_group_name": particles_group_names[0],
                    "volume_group_name": volume_group_name,
                    "wait_for_completion": False,
                    "timeout": self.config.job_management.default_timeout,
                    "check_interval": self.config.job_management.status_check_interval,
                }
                if refine_res_init is not None:
                    refine_params["refine_res_init"] = float(refine_res_init)
                refine_result = self.cryosparc_tools.nonuniform_refine_new(**refine_params)

            if not refine_result.get("success", False):
                error_msg = refine_result.get("error") or "Unknown error"
                return json.dumps({"success": False, "error": f"Non-uniform refinement failed: {error_msg}"})

            job_uid = refine_result.get("job_uid")
            result = {
                "success": True,
                "job_uid": job_uid,
                "job_type": "nonuniform_refine_new",
                "hetero_job_uid": hetero_job_uid,
                "particles_group_names": particles_group_names,
                "volume_group_name": volume_group_name,
                "status": "queued",
                "next_step": "wait_for_job then get_fsc_info to report final_resolution_angstroms",
                "refinement_result": refine_result,
            }
            tool_params = {
                "hetero_job_uid": hetero_job_uid,
                "particles_group_names": particles_group_names,
                "volume_group_name": volume_group_name,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
            }
            self._record_tool_execution("run_non_uniform_refinement", tool_params, result=result)
            return json.dumps(result)
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution(
                "run_non_uniform_refinement", params if "params" in locals() else {}, error=str(e)
            )
            return json.dumps(error_result)

    def _get_fsc_info_tool(self, tool_input: str) -> str:
        """Get FSC resolution from a completed non-uniform refinement job."""
        try:
            params = self._parse_tool_input(tool_input)
            refinement_job_uid = params.get("refinement_job_uid") or params.get("job_uid")
            if not refinement_job_uid:
                input_stripped = tool_input.strip().strip("\"'")
                if input_stripped.startswith("J") and len(input_stripped) <= 10:
                    refinement_job_uid = input_stripped

            if not refinement_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: refinement_job_uid (or pass job UID e.g. 'JXXX')",
                })

            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refinement_job_uid)
            if not fsc_info.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get FSC info: {fsc_info.get('error', 'Unknown error')}",
                })

            result = {
                "success": True,
                "refinement_job_uid": refinement_job_uid,
                "box_size": fsc_info.get("box_size"),
                "resolution_angstroms": fsc_info.get("resolution_angstroms"),
                "final_resolution_angstroms": fsc_info.get("resolution_angstroms"),
            }
            self._record_tool_execution(
                "get_fsc_info",
                {"refinement_job_uid": refinement_job_uid, "project_uid": project_uid},
                result=result,
            )
            return json.dumps(result)
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("get_fsc_info", params if "params" in locals() else {}, error=str(e))
            return json.dumps(error_result)

