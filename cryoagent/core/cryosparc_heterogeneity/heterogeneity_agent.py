"""ReAct-based heterogeneity analysis agent for CryoEM 3D reconstruction."""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..cryosparc_common_tools import CryoSPARCCommonTools
from ..base_react_agent import BaseReActAgent
from ...tools.cryosparc_tools import CryoSPARCTools
from ...tools.alignment_tools.compare_all_densities_tool import CompareAllDensitiesTool
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


class HeterogeneityAgent(BaseReActAgent):
    """ReAct-based agent for heterogeneity analysis in CryoEM 3D reconstruction."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the heterogeneity analysis agent.
        
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
        self.logger = logging.getLogger("HeterogeneityAgent")
    
    def _create_tools(self) -> List[Tool]:
        """Create heterogeneity analysis-specific tools."""
        from ..cryosparc_tool_registry import build_tools, AGENT_TOOL_SETS
        tools = build_tools(self, AGENT_TOOL_SETS["heterogeneity"])

        # Add compare_all_densities tool
        # Note: Contour levels are not provided by default - RMS will be calculated automatically
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
        """Load heterogeneity analysis stage configuration."""
        config_path = Path("configs/cryosparc/heterogeneity_config.json")
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
    
    def _should_use_nonuniform_refinement(self) -> bool:
        """Check if non-uniform refinement should be used."""
        return self.stage_workflow.get("heterogeneity_analysis", {}).get("use_nonuniform_refinement", True)
    
    def _get_refinement_res_init(self) -> Optional[float]:
        """Get initial lowpass resolution for refinement from config."""
        return self._get_stage_param("heterogeneity_analysis", "refine_res_init", None)
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Fetch a parameter from the stage workflow configuration."""
        if section == "heterogeneity_analysis":
            return self.stage_workflow.get("heterogeneity_analysis", {}).get(key, default)
        elif section == "density_comparison":
            density_config = self.stage_workflow.get("heterogeneity_analysis", {}).get("density_comparison", {})
            return density_config.get(key, default)
        elif section == "external_tools":
            external_config = self.stage_workflow.get("heterogeneity_analysis", {}).get("external_tools", {})
            return external_config.get(key, default)
        return default
    
    def _get_system_prompt_context(self) -> Dict[str, Any]:
        """Build template variables for cryosparc/heterogeneity/system.md."""
        initial_k_values = self._get_stage_param("heterogeneity_analysis", "initial_k_values", [3, 5])
        max_k = self._get_stage_param("heterogeneity_analysis", "max_k", 10)
        resolution_threshold = self._get_stage_param("heterogeneity_analysis", "resolution_threshold", 12.0)
        return {
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
            "initial_k_values": initial_k_values,
            "max_k": max_k,
            "resolution_threshold": resolution_threshold,
        }

    def _get_react_system_prompt(self) -> str:
        """Get the heterogeneity analysis-specific ReAct system prompt."""
        return self._compose_stage_system_prompt(
            "cryosparc/heterogeneity/system.md",
            self._get_system_prompt_context(),
        )
    
    def update_workflow_defaults(self, defaults: Dict[str, Any]) -> None:
        """Store workflow-level default parameters for later tool invocations."""
        if defaults:
            if not hasattr(self, "workflow_defaults") or self.workflow_defaults is None:
                self.workflow_defaults = {}
            self.workflow_defaults.update(defaults)
    
    # =================================================================
    # Tool Implementation Methods
    # =================================================================
    def _extract_density_maps_tool(self, tool_input: str) -> str:
        """
        Get the job directory containing density map files (*_volume.mrc) from a heterogeneous refinement job.
        Returns the job directory directly without copying files.
        
        Can accept either:
        - Just the job UID as a string (e.g., "JXXX")
        - JSON with hetero_job_uid parameter
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Support both direct job UID string and JSON parameter
            hetero_job_uid = params.get("hetero_job_uid") or params.get("job_uid")
            
            # If still not found, try to extract from input string directly
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
            
            # Get job output directory - use it directly without copying
            job_info = self.cryosparc_tools.get_job_output_directory(project_uid, hetero_job_uid)
            job_dir = Path(job_info["job_directory"])
            
            self.logger.info(f"📦 Using job directory directly for density maps: {job_dir}")
            
            # Find all *_volume.mrc files in the job directory
            volume_files = list(job_dir.glob("*_volume.mrc"))
            
            if not volume_files:
                # Try searching in subdirectories
                volume_files = list(job_dir.rglob("*_volume.mrc"))
            
            if not volume_files:
                return json.dumps({
                    "success": False,
                    "error": f"No *_volume.mrc files found in job directory: {job_dir}",
                    "job_directory": str(job_dir)
                })
            
            # Return job directory and list of map files (full paths)
            map_files = [str(vol_file) for vol_file in volume_files]
            self.logger.info(f"  Found {len(map_files)} density map(s) in job directory")
            
            result = {
                "success": True,
                "hetero_job_uid": hetero_job_uid,
                "output_folder": str(job_dir),  # Return job directory directly
                "num_maps_extracted": len(map_files),
                "map_files": map_files  # Full paths to maps in job directory
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
        
        Can accept either:
        - Just the job UID as a string (e.g., "JXXX")
        - JSON with job_uid parameter
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Support both direct job UID string and JSON parameter
            job_uid = params.get("job_uid") or params.get("hetero_job_uid") or params.get("refinement_job_uid")
            
            # If still not found, try to extract from input string directly
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
            
            # Get class resolutions from the heterogeneous refinement job
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
    
    def _run_non_uniform_refinement_tool(self, tool_input: str) -> str:
        """
        Run non-uniform homogeneous refinement for a specific group of particles.
        
        This tool refines a single group using:
        - Particles from one or more classes (particles_group_names)
        - Volume from the best-resolution class (volume_group_name)
        
        Can accept JSON with parameters:
        - hetero_job_uid: Heterogeneous refinement job UID
        - particles_group_names: List of particle group names (e.g., ["particles_class_0", "particles_class_1"])
        - volume_group_name: Volume group name (e.g., "volume_class_0")
        - Optional: project_uid, workspace_uid, refine_res_init
        """
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
            
            # Ensure particles_group_names is a list
            if isinstance(particles_group_names, str):
                # Try to parse as JSON array first
                try:
                    import json as json_module
                    parsed = json_module.loads(particles_group_names)
                    if isinstance(parsed, list):
                        particles_group_names = parsed
                    else:
                        # If parsed but not a list, treat as single group name
                        particles_group_names = [particles_group_names]
                except (json.JSONDecodeError, ValueError, TypeError):
                    # If not JSON, treat as single group name
                    particles_group_names = [particles_group_names]
            
            # Ensure it's a list after parsing
            if not isinstance(particles_group_names, list):
                particles_group_names = [particles_group_names] if particles_group_names else []
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get refinement parameters from config
            refine_res_init = params.get("refine_res_init")
            if refine_res_init is None:
                refine_res_init = self._get_refinement_res_init()
            
            symmetry = self._get_refinement_symmetry()

            # Raw CryoSPARC parameter passthrough (e.g. crg_min_res_A) the LLM may
            # supply to override defaults. Excludes keys this wrapper consumes.
            passthrough = self._extract_passthrough_params(
                params,
                consumed_keys=[
                    "hetero_job_uid", "particles_group_names", "particles_group_name",
                    "volume_group_name", "refine_res_init", "symmetry",
                ],
            )

            self.logger.info(f"🔧 Running non-uniform refinement for group:")
            self.logger.info(f"   Hetero job: {hetero_job_uid}")
            self.logger.info(f"   Particles groups: {particles_group_names}")
            self.logger.info(f"   Volume group: {volume_group_name}")
            
            # For multiple particle groups, we need to manually create the job with multiple connections
            # CryoSPARC's nonuniform_refine_new supports multiple particle connections similar to heterogeneous refinement
            if len(particles_group_names) > 1:
                self.logger.info(f"📦 Combining particles from {len(particles_group_names)} classes: {particles_group_names}")
                
                # Manually create the job with multiple particle connections
                try:
                    from cryosparc.tools import CryoSPARC
                    project = self.cryosparc_tools.cs.find_project(project_uid)
                    workspace = project.find_workspace(workspace_uid)
                    
                    # Prepare job parameters
                    job_params = {
                        "refine_do_init_scale_est": True,
                        "refine_symmetry_do_align": True,
                        "refine_defocus_refine": True,
                        "refine_ctf_global_refine": True
                    }
                    
                    if symmetry and symmetry != "C1":
                        job_params["refine_symmetry"] = symmetry
                    
                    if refine_res_init is not None:
                        job_params["refine_res_init"] = float(refine_res_init)
                    # LLM-supplied raw params take precedence over the defaults above.
                    if passthrough:
                        job_params.update(passthrough)

                    # Build connections with multiple particle groups
                    # For nonuniform_refine_new, the "particles" input can accept multiple connections
                    # Similar to how heterogeneous refinement handles multiple volumes
                    particle_connections = [(hetero_job_uid, group_name) for group_name in particles_group_names]
                    connections = {
                        "particles": particle_connections,  # List of (job_uid, group_name) tuples
                        "volume": (hetero_job_uid, volume_group_name)
                    }
                    
                    self.logger.info(f"   Connecting {len(particle_connections)} particle groups to refinement job")
                    self.logger.info(f"   Particle connections: {particle_connections}")
                    self.logger.info(f"   Volume connection: ({hetero_job_uid}, {volume_group_name})")
                    
                    # Create the job manually
                    job = workspace.create_job(
                        "nonuniform_refine_new",
                        connections=connections,
                        params=job_params
                    )
                    
                    # Queue the job
                    used_lane = self.cryosparc_tools._queue_job_with_lane_fallback(
                        job,
                        log_prefix="⚙️ No lane specified; using lane",
                        logger=self.logger,
                    )
                    
                    job_uid = job.uid
                    self.logger.info(f"✅ Non-uniform refinement job queued: {job_uid} (with {len(particles_group_names)} particle groups)")
                    
                    refine_result = {
                        "success": True,
                        "job_uid": job_uid,
                        "job_type": "nonuniform_refine_new",
                        "message": f"Non-uniform refinement job {job_uid} queued successfully with {len(particles_group_names)} particle groups",
                        "symmetry": symmetry,
                        "refinement_resolution": None,
                        "lane": used_lane,
                        "particles_group_names": particles_group_names,
                        "volume_group_name": volume_group_name
                    }
                    
                except Exception as e:
                    error_msg = f"Failed to create non-uniform refinement job with multiple particle groups: {str(e)}"
                    self.logger.error(f"❌ {error_msg}")
                    return json.dumps({
                        "success": False,
                        "error": error_msg,
                        "refinement_result": None
                    })
            else:
                # Single particle group - use the standard API
                particles_group_name = particles_group_names[0] if particles_group_names else None
                
                # Prepare refinement parameters
                refine_params = {
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "particles_job_uid": hetero_job_uid,
                    "volume_job_uid": hetero_job_uid,
                    "symmetry": symmetry,
                    "refine_defocus_refine": True,
                    "refine_ctf_global_refine": True,
                    "particles_group_name": particles_group_name,
                    "volume_group_name": volume_group_name,
                    "wait_for_completion": False,  # Don't wait - let the agent monitor
                    "timeout": self.config.job_management.default_timeout,
                    "check_interval": self.config.job_management.status_check_interval
                }
                
                if refine_res_init is not None:
                    refine_params["refine_res_init"] = float(refine_res_init)
                if passthrough:
                    refine_params["params"] = passthrough

                # Run non-uniform refinement using standard API
                refine_result = self.cryosparc_tools.nonuniform_refine_new(**refine_params)
            
            if not refine_result.get("success", False):
                error_msg = refine_result.get("error") or "Unknown error"
                self.logger.error(f"❌ Non-uniform refinement failed: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Non-uniform refinement failed: {error_msg}",
                    "refinement_result": refine_result
                })
            
            job_uid = refine_result.get("job_uid")
            self.logger.info(f"✅ Non-uniform refinement job queued: {job_uid}")
            
            # Extract particles_group_names and volume_group_name from refine_result if available
            result_particles_groups = refine_result.get("particles_group_names", particles_group_names)
            result_volume_group = refine_result.get("volume_group_name", volume_group_name)
            
            result = {
                "success": True,
                "job_uid": job_uid,
                "job_type": "nonuniform_refine_new",
                "hetero_job_uid": hetero_job_uid,
                "particles_group_names": result_particles_groups,
                "volume_group_name": result_volume_group,
                "status": "queued",
                "refinement_result": refine_result
            }
            
            # Record tool execution with proper parameters
            tool_params = {
                "hetero_job_uid": hetero_job_uid,
                "particles_group_names": result_particles_groups,
                "volume_group_name": result_volume_group,
                "project_uid": project_uid,
                "workspace_uid": workspace_uid
            }
            if refine_res_init is not None:
                tool_params["refine_res_init"] = refine_res_init
            
            self._record_tool_execution("run_non_uniform_refinement", tool_params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("run_non_uniform_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)


    def _ab_initio_tool(self, tool_input: str) -> str:
        """Execute ab initio reconstruction (single atomic job)."""
        try:
            params = self._parse_tool_input(tool_input)

            particles_job_uid = params.get("particles_job_uid") or params.get("job_uid")
            if not particles_job_uid and "__arg1" in params:
                arg_value = str(params.get("__arg1", "")).strip()
                if arg_value:
                    particles_job_uid = arg_value.split(",")[0].strip()
            if not particles_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: particles_job_uid or job_uid"
                })

            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)

            ab_initio_defaults = self.stage_workflow.get("ab_initio", {})
            num_classes = params.get("num_classes", ab_initio_defaults.get("num_classes", 1))
            initial_resolution = params.get("initial_resolution", ab_initio_defaults.get("initial_resolution", 20.0))
            final_resolution = params.get("final_resolution", ab_initio_defaults.get("final_resolution", 10.0))
            max_iterations = params.get("max_iterations", ab_initio_defaults.get("max_iterations", 50))
            # Heterogeneity analysis: do NOT enforce point-group symmetry during ab initio.
            # Symmetry is only applied later in the (NU) homogeneous refinement step.
            # Honor an explicit caller-provided symmetry, otherwise use C1.
            symmetry = params.get("symmetry") or "C1"
            params["symmetry"] = symmetry

            wait_for_completion = self._parse_bool_param(params.get("wait_for_completion"), False)
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))

            passthrough = self._extract_passthrough_params(
                params,
                consumed_keys=["particles_job_uid", "job_uid", "num_classes", "initial_resolution", "final_resolution", "max_iterations", "symmetry"],
            )

            result = self.cryosparc_tools.ab_initio_reconstruction(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particles_job_uid=particles_job_uid,
                num_classes=num_classes,
                initial_resolution=initial_resolution,
                final_resolution=final_resolution,
                max_iterations=max_iterations,
                symmetry=symmetry,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval,
                params=passthrough
            )

            self._record_tool_execution("ab_initio_reconstruction", params, result=result)
            return json.dumps(result)

        except Exception as e:
            self._record_tool_execution("ab_initio_reconstruction", params if 'params' in locals() else {}, error=str(e))
            return json.dumps({"success": False, "error": str(e)})

    def _heterogeneous_refinement_tool(self, tool_input: str) -> str:
        """Execute a single heterogeneous refinement job (atomic).

        Accepts two forms for supplying initial volumes:
        1. volume_from_job_uid (+ num_classes or volume_group_names): connect K
           distinct volume outputs of a single job.
        2. volume_job_uids: a list (or comma-separated string) of volume job UIDs.
        """
        try:
            params = self._parse_tool_input(tool_input)

            particles_job_uid = params.get("particles_job_uid")
            if not particles_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: particles_job_uid"
                })

            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)

            # Heterogeneity analysis: do NOT enforce point-group symmetry during
            # heterogeneous refinement. Symmetry is only applied later in the (NU)
            # homogeneous refinement step. Honor an explicit override, else C1.
            symmetry = params.get("symmetry") or "C1"

            volume_from_job_uid = params.get("volume_from_job_uid")
            volume_job_uids = params.get("volume_job_uids")

            wait_for_completion = self._parse_bool_param(params.get("wait_for_completion"), False)
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))

            consumed = [
                "particles_job_uid", "volume_job_uids", "num_classes", "symmetry",
                "volume_from_job_uid", "volume_group_names", "particles_group_name",
            ]
            passthrough = self._extract_passthrough_params(params, consumed_keys=consumed)

            if volume_from_job_uid:
                volume_group_names = params.get("volume_group_names")
                if isinstance(volume_group_names, str):
                    volume_group_names = [v.strip() for v in volume_group_names.split(",")]
                num_classes = params.get("num_classes")
                if num_classes is not None:
                    num_classes = int(num_classes)
                result = self.cryosparc_tools.heterogeneous_refinement(
                    project_uid=project_uid,
                    workspace_uid=workspace_uid,
                    particles_job_uid=particles_job_uid,
                    num_classes=num_classes,
                    symmetry=symmetry,
                    volume_from_job_uid=volume_from_job_uid,
                    volume_group_names=volume_group_names,
                    particles_group_name=params.get("particles_group_name"),
                    wait_for_completion=wait_for_completion,
                    timeout=timeout,
                    check_interval=check_interval,
                    params=passthrough
                )
            else:
                if not volume_job_uids:
                    return json.dumps({
                        "success": False,
                        "error": "Missing required parameter: provide volume_from_job_uid or volume_job_uids"
                    })
                if isinstance(volume_job_uids, str):
                    volume_job_uids = [v.strip() for v in volume_job_uids.split(",")]
                num_classes = params.get("num_classes", len(volume_job_uids))
                result = self.cryosparc_tools.heterogeneous_refinement(
                    project_uid=project_uid,
                    workspace_uid=workspace_uid,
                    particles_job_uid=particles_job_uid,
                    volume_job_uids=volume_job_uids,
                    num_classes=num_classes,
                    symmetry=symmetry,
                    wait_for_completion=wait_for_completion,
                    timeout=timeout,
                    check_interval=check_interval,
                    params=passthrough
                )

            self._record_tool_execution("heterogeneous_refinement", params, result=result)
            return json.dumps(result)

        except Exception as e:
            self._record_tool_execution("heterogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps({"success": False, "error": str(e)})

    def _get_fsc_info_tool(self, tool_input: str) -> str:
        """Get FSC resolution and box size from a refinement job.

        Can accept either:
        - Just the job UID as a string (e.g., "JXXX")
        - JSON with refinement_job_uid parameter
        """
        try:
            params = self._parse_tool_input(tool_input)

            refinement_job_uid = params.get("refinement_job_uid") or params.get("job_uid")

            if not refinement_job_uid:
                input_stripped = tool_input.strip().strip('"\'')
                if input_stripped.startswith("J") and len(input_stripped) <= 10:
                    refinement_job_uid = input_stripped
                elif "input" in params:
                    input_val = str(params["input"]).strip().strip('"\'')
                    if input_val.startswith("J") and len(input_val) <= 10:
                        refinement_job_uid = input_val

            if not refinement_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: refinement_job_uid. You can pass just the job UID (e.g., 'JXXX') or JSON with refinement_job_uid parameter."
                })

            project_uid = params.get("project_uid", self.config.workflow.project_uid)

            fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refinement_job_uid)

            if not fsc_info.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get FSC info: {fsc_info.get('error', 'Unknown error')}"
                })

            result = {
                "success": True,
                "refinement_job_uid": refinement_job_uid,
                "box_size": fsc_info.get("box_size"),
                "resolution_angstroms": fsc_info.get("resolution_angstroms")
            }

            self._record_tool_execution("get_fsc_info", {"refinement_job_uid": refinement_job_uid, "project_uid": project_uid}, result=result)
            return json.dumps(result)

        except Exception as e:
            self._record_tool_execution("get_fsc_info", params if 'params' in locals() else {}, error=str(e))
            return json.dumps({"success": False, "error": str(e)})
