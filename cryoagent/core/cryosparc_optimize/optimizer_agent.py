"""ReAct-based box size optimization agent for CryoEM 3D reconstruction."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..cryosparc_common_tools import CryoSPARCCommonTools
from ..base_react_agent import BaseReActAgent
from ...tools.cryosparc_tools import CryoSPARCTools
from ...config.config_loader import CryoAgentConfig
from ...prompts.prompt_loader import load_prompt


class OptimizerAgent(BaseReActAgent):
    """ReAct-based agent for optimizing box size/diameter in CryoEM 3D reconstruction."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the box size optimization agent.
        
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
        self.logger = logging.getLogger("OptimizerAgent")
    
    def _create_tools(self) -> List[Tool]:
        """Create optimization-specific tools."""
        from ..cryosparc_tool_registry import build_tools, AGENT_TOOL_SETS
        return build_tools(self, AGENT_TOOL_SETS["optimization"])
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load optimization stage configuration."""
        config_path = Path("configs/cryosparc/optimization_config.json")
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
        """Get symmetry from reconstruction_config.json for homogeneous refinement."""
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
        """Check if non-uniform refinement should be used instead of homogeneous refinement."""
        return self.stage_workflow.get("optimization", {}).get("use_nonuniform_refinement", True)
    
    def _get_refinement_res_init(self) -> Optional[float]:
        """Get initial lowpass resolution for refinement from optimization config."""
        return self._get_stage_param("optimization", "refine_res_init", None)
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Fetch a parameter from the stage workflow configuration."""
        return self.stage_workflow.get(section, {}).get(key, default)
    
    def _get_optimization_flags(self) -> Dict[str, Any]:
        """Read optimization config flags used by system and task prompts."""
        enable_box_size = self._get_stage_param("optimization", "enable_box_size_optimization", True)
        enable_hetero = self._get_stage_param("optimization", "enable_heterogeneous_refinement", False)
        enable_multi_round = self._get_stage_param("optimization", "enable_multi_round_3d_classification", False)
        max_hetero_iterations = self._get_stage_param("optimization", "heterogeneous_refinement_max_iterations", 3)
        multi_round_num_classes = self._get_stage_param("optimization", "multi_round_3d_classification_num_classes", 4)
        multi_round_max_rounds = self._get_stage_param("optimization", "multi_round_3d_classification_max_rounds", 5)
        multi_round_improvement_threshold = self._get_stage_param(
            "optimization", "multi_round_3d_classification_improvement_threshold", 0.1
        )
        optimization_types = []
        if enable_box_size:
            optimization_types.append("box size/diameter")
        if enable_hetero:
            optimization_types.append("heterogeneous refinement (K values)")
        if enable_multi_round:
            optimization_types.append("multi-round 3D classification")
        return {
            "enable_box_size": enable_box_size,
            "enable_hetero": enable_hetero,
            "enable_multi_round": enable_multi_round,
            "max_hetero_iterations": max_hetero_iterations,
            "multi_round_num_classes": multi_round_num_classes,
            "multi_round_max_rounds": multi_round_max_rounds,
            "multi_round_improvement_threshold": multi_round_improvement_threshold,
            "optimization_desc": " and ".join(optimization_types) if optimization_types else "parameters",
        }

    def _load_optimization_section(self, name: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """Load an optimization prompt section from cryosparc/optimization/sections/."""
        return load_prompt(f"cryosparc/optimization/sections/{name}.md", variables or {})

    def _get_system_prompt_context(self) -> Dict[str, Any]:
        """Build template variables for cryosparc/optimization/system.md."""
        flags = self._get_optimization_flags()
        enable_box_size = flags["enable_box_size"]
        enable_hetero = flags["enable_hetero"]
        enable_multi_round = flags["enable_multi_round"]

        priority_section = ""
        if enable_box_size and enable_hetero:
            priority_section = (
                "**IMPORTANT**: Both heterogeneous refinement and box size optimization are enabled. "
                "Complete heterogeneous refinement (K optimization) FIRST, then use the optimized refinement job "
                "for box size optimization.**"
            )
        elif not enable_box_size and enable_hetero:
            priority_section = (
                "**CRITICAL: Box size optimization is DISABLED. DO NOT use test_box_size tool. "
                "Proceed directly to heterogeneous refinement using the refinement_job_uid provided.**"
            )
        elif enable_box_size and not enable_hetero:
            priority_section = "**Heterogeneous refinement is DISABLED**. Proceed with box size optimization only.**"

        box_size_section = self._load_optimization_section("box_size") if enable_box_size else ""

        hetero_section = ""
        if enable_hetero:
            hetero_section = self._load_optimization_section(
                "hetero",
                {"max_hetero_iterations": flags["max_hetero_iterations"]},
            )

        multi_round_section = self._load_optimization_section("multi_round_base")
        if enable_multi_round:
            multi_round_section += "\n" + self._load_optimization_section(
                "multi_round_config",
                {
                    "multi_round_num_classes": flags["multi_round_num_classes"],
                    "multi_round_max_rounds": flags["multi_round_max_rounds"],
                    "multi_round_improvement_threshold": flags["multi_round_improvement_threshold"],
                },
            )

        combined_section = ""
        if enable_box_size or enable_hetero or enable_multi_round:
            combined_section = self._load_optimization_section("combined")

        return {
            "optimization_desc": flags["optimization_desc"],
            "priority_section": priority_section,
            "box_size_section": box_size_section,
            "hetero_section": hetero_section,
            "multi_round_section": multi_round_section,
            "combined_section": combined_section,
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
            "box_size_status": "ENABLED" if enable_box_size else "DISABLED",
            "hetero_status": "ENABLED" if enable_hetero else "DISABLED",
            "multi_round_status": "ENABLED" if enable_multi_round else "DISABLED",
        }

    def _get_react_system_prompt(self) -> str:
        """Get the optimization-specific ReAct system prompt."""
        return self._compose_stage_system_prompt(
            "cryosparc/optimization/system.md",
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
    
    def _get_fsc_info_tool(self, tool_input: str) -> str:
        """
        Get FSC resolution and box size from a refinement job.
        
        This tool retrieves the FSC resolution and box size information
        from a completed refinement job.
        
        Can accept either:
        - Just the job UID as a string (e.g., "JXXX")
        - JSON with refinement_job_uid parameter
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Support both direct job UID string and JSON parameter
            # _parse_tool_input converts "JXXX" to {"job_uid": "JXXX"}
            refinement_job_uid = params.get("refinement_job_uid") or params.get("job_uid")
            
            # If still not found, try to extract from input string directly
            if not refinement_job_uid:
                # Check if input is just a job UID (starts with J and is short)
                input_stripped = tool_input.strip().strip('"\'')
                if input_stripped.startswith("J") and len(input_stripped) <= 10:
                    refinement_job_uid = input_stripped
                # Also check if it's in the "input" field
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
            
            # Get FSC info from the refinement job
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
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("get_fsc_info", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    def _optimize_diameter_tool(self, tool_input: str) -> str:
        """
        [DEPRECATED] Optimize box size/diameter by testing different box sizes and comparing FSC resolutions.
        
        This tool is deprecated in favor of the agentic approach using test_box_size and get_fsc_info.
        The LLM should now use those tools iteratively to optimize box size.
        
        This tool implements an iterative optimization algorithm:
        1. After first homogeneous refinement, test with 10% less and 10% more box sizes
        2. Extract particles with new box sizes and run homogeneous refinement
        3. Compare three results (original, -10%, +10%)
        4. If middle is best, choose value in between
        5. If side is better, test another 10% more/less on that side
        6. Stop if >5 diameters tested or new box size equals original
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            refinement_job_uid = params.get("refinement_job_uid")
            particles_job_uid = params.get("particles_job_uid")
            micrographs_job_uid = params.get("micrographs_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            
            # Try to get from workflow defaults if not provided
            defaults = getattr(self, "workflow_defaults", {}) or {}
            if not particles_job_uid:
                # Try to get picking job (for re-extraction) from various possible keys
                particles_job_uid = (
                    defaults.get("particles_job_uid")  # Fallback to extracted particles
                    or defaults.get("selected_particles_job_uid")
                )
            if not micrographs_job_uid:
                micrographs_job_uid = (
                    defaults.get("micrographs_job_uid")
                    or defaults.get("micrograph_selection_job_uid")
                    or defaults.get("final_micrographs_job_uid")
                )
            if not volume_job_uid:
                volume_job_uid = (
                    defaults.get("volume_job_uid")
                    or defaults.get("last_volume_job_uid")
                    or defaults.get("ab_init_job_uid")
                    or defaults.get("homogeneous_reconstruction_job_uid")
                )
            
            if not all([refinement_job_uid, particles_job_uid, micrographs_job_uid, volume_job_uid]):
                missing = []
                if not refinement_job_uid:
                    missing.append("refinement_job_uid")
                if not particles_job_uid:
                    missing.append("particles_job_uid")
                if not micrographs_job_uid:
                    missing.append("micrographs_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}"
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get original FSC info
            fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refinement_job_uid)
            if not fsc_info.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get FSC info from refinement job: {fsc_info.get('error')}"
                })
            
            original_box_size = fsc_info["box_size"]
            original_resolution = fsc_info["resolution_angstroms"]
            
            # Track tested box sizes and their resolutions
            tested_data = [
                {"box_size": original_box_size, "resolution": original_resolution, "job_uid": refinement_job_uid}
            ]
            
            max_iterations = 5
            iteration = 0
            initial_three_tested = False
            
            while iteration < max_iterations:
                iteration += 1
                
                # First iteration: test all three box sizes (original, -10%, +10%)
                # Note: Original is already tested from initial refinement, so we only test -10% and +10%
                if not initial_three_tested:
                    box_size_less = int(original_box_size * 0.9)
                    box_size_more = int(original_box_size * 1.1)
                    
                    # Normalize to allowed box sizes
                    box_size_less = self._normalize_box_size(box_size_less)
                    box_size_more = self._normalize_box_size(box_size_more)
                    
                    # Skip if same as original (shouldn't happen, but check anyway)
                    if box_size_less == original_box_size:
                        box_size_less = None
                    if box_size_more == original_box_size:
                        box_size_more = None
                    
                    # Skip if already tested (shouldn't happen in first iteration, but check)
                    if box_size_less and box_size_less in [d["box_size"] for d in tested_data]:
                        box_size_less = None
                    if box_size_more and box_size_more in [d["box_size"] for d in tested_data]:
                        box_size_more = None
                    
                    initial_three_tested = True
                    self.logger.info(f"First iteration: Testing box sizes -10%={box_size_less}, +10%={box_size_more} (original={original_box_size} already tested)")
                else:
                    # Subsequent iterations: determine next box size based on best result
                    # Sort by box size to find positions
                    tested_data_sorted = sorted(tested_data, key=lambda x: x["box_size"])
                    
                    # Find best resolution (lower is better)
                    best_idx = min(range(len(tested_data_sorted)), key=lambda i: tested_data_sorted[i]["resolution"])
                    best_box_size = tested_data_sorted[best_idx]["box_size"]
                    best_resolution = tested_data_sorted[best_idx]["resolution"]
                    
                    # Determine next box size based on which is best
                    next_box_size = None
                    
                    # Find position of best in sorted list
                    if len(tested_data_sorted) >= 3:
                        # Check if best is in the middle
                        middle_idx = len(tested_data_sorted) // 2
                        if best_idx == middle_idx:
                            # Middle is best, choose value between middle and better side
                            # Compare left and right sides
                            if best_idx > 0 and best_idx < len(tested_data_sorted) - 1:
                                left_res = tested_data_sorted[best_idx - 1]["resolution"]
                                right_res = tested_data_sorted[best_idx + 1]["resolution"]
                                
                                if left_res < right_res:
                                    # Left is better, choose between left and middle
                                    next_box_size = int((tested_data_sorted[best_idx - 1]["box_size"] + best_box_size) / 2)
                                else:
                                    # Right is better, choose between middle and right
                                    next_box_size = int((best_box_size + tested_data_sorted[best_idx + 1]["box_size"]) / 2)
                        elif best_idx == 0:
                            # Leftmost (smallest) is best, test 10% less
                            next_box_size = int(best_box_size * 0.9)
                        elif best_idx == len(tested_data_sorted) - 1:
                            # Rightmost (largest) is best, test 10% more
                            next_box_size = int(best_box_size * 1.1)
                        else:
                            # Best is somewhere in between, move towards it
                            if best_idx < middle_idx:
                                # Best is on left side, test 10% less
                                next_box_size = int(best_box_size * 0.9)
                            else:
                                # Best is on right side, test 10% more
                                next_box_size = int(best_box_size * 1.1)
                    else:
                        # Less than 3 data points, continue in direction of best
                        if best_idx == 0:
                            next_box_size = int(best_box_size * 0.9)
                        else:
                            next_box_size = int(best_box_size * 1.1)
                    
                    if next_box_size:
                        next_box_size = self._normalize_box_size(next_box_size)
                        
                        # Check if already tested
                        if next_box_size in [d["box_size"] for d in tested_data]:
                            # Already tested, stop
                            break
                        
                        box_size_less = None
                        box_size_more = None
                        
                        # Set the appropriate variable based on direction
                        if next_box_size < best_box_size:
                            box_size_less = next_box_size
                        else:
                            box_size_more = next_box_size
                    else:
                        # No next box size determined, stop
                        break
                
                # If no box sizes to test, stop
                if not box_size_less and not box_size_more:
                    break
                
                # Test box_size_less
                if box_size_less:
                    # Check if already tested
                    if box_size_less in [d["box_size"] for d in tested_data]:
                        self.logger.info(f"Skipping box_size_less {box_size_less} - already tested")
                    else:
                        try:
                            self.logger.info(f"🔬 Testing box_size_less: {box_size_less}")
                            
                            # Step 1: Extract particles with new box size
                            # Use refinement job for particle coordinates (refined positions/orientations)
                            self.logger.info(f"📦 Step 1/3: Extracting particles with box_size {box_size_less}...")
                            extract_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": refinement_job_uid,  # Use refinement job for refined coordinates
                                "micrographs_job_uid": micrographs_job_uid,
                                "box_size_pix": box_size_less
                            }
                            self._record_tool_execution("extract_particles", extract_params)
                            extract_result = self.cryosparc_tools.extract_particles(
                                **extract_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("extract_particles", extract_params, result=extract_result)
                            
                            # Verify extraction completed successfully
                            extract_status = extract_result.get("status", "unknown")
                            if extract_status != "completed":
                                error_msg = extract_result.get("error") or f"Status: {extract_status}"
                                self.logger.error(f"❌ Particle extraction failed for box_size {box_size_less}: {error_msg}")
                                # Don't proceed to refinement if extraction failed
                                continue
                            
                            extract_job_uid = extract_result["job_uid"]
                            self.logger.info(f"✅ Step 1/3: Extraction completed for box_size {box_size_less}, job: {extract_job_uid}")
                            
                            # Step 2: Run refinement (non-uniform or homogeneous based on config)
                            use_nonuniform = self._should_use_nonuniform_refinement()
                            refinement_type = "non-uniform" if use_nonuniform else "homogeneous"
                            self.logger.info(f"🔧 Step 2/3: Starting {refinement_type} refinement for box_size {box_size_less}...")
                            # Get symmetry from reconstruction_config.json
                            symmetry = self._get_refinement_symmetry()
                            refine_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": extract_job_uid,
                                "volume_job_uid": volume_job_uid,
                                "symmetry": symmetry
                            }
                            # Add initial lowpass resolution if configured
                            refine_res_init = self._get_refinement_res_init()
                            if refine_res_init is not None:
                                refine_params["refine_res_init"] = float(refine_res_init)
                            tool_name = "nonuniform_refine_new" if use_nonuniform else "homogeneous_refinement"
                            self._record_tool_execution(tool_name, refine_params)
                            if use_nonuniform:
                                refine_result = self.cryosparc_tools.nonuniform_refine_new(
                                    **refine_params,
                                    wait_for_completion=True,
                                    timeout=self.config.job_management.default_timeout,
                                    check_interval=self.config.job_management.status_check_interval
                                )
                            else:
                                refine_result = self.cryosparc_tools.homogeneous_refinement(
                                    **refine_params,
                                    wait_for_completion=True,
                                    timeout=self.config.job_management.default_timeout,
                                    check_interval=self.config.job_management.status_check_interval
                                )
                            self._record_tool_execution(tool_name, refine_params, result=refine_result)
                            
                            # Verify refinement completed successfully
                            # refinement returns "success" and updates with status_result
                            if not refine_result.get("success", False):
                                error_msg = refine_result.get("error") or "Unknown error"
                                self.logger.error(f"❌ Refinement failed for box_size {box_size_less}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement failed
                                continue
                            
                            refine_status = refine_result.get("status", "unknown")
                            if refine_status != "completed":
                                error_msg = refine_result.get("error") or f"Status: {refine_status}"
                                self.logger.error(f"❌ Refinement did not complete for box_size {box_size_less}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement didn't complete
                                continue
                            
                            refine_job_uid = refine_result["job_uid"]
                            self.logger.info(f"✅ Step 2/3: Refinement completed for box_size {box_size_less}, job: {refine_job_uid}")
                            
                            # Step 3: Get FSC info and resolution
                            self.logger.info(f"📊 Step 3/3: Getting FSC resolution for box_size {box_size_less}...")
                            fsc_info_less = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
                            if fsc_info_less.get("success"):
                                resolution = fsc_info_less["resolution_angstroms"]
                                self.logger.info(f"✅ Step 3/3: Box size {box_size_less}: Resolution = {resolution} Å")
                                tested_data.append({
                                    "box_size": box_size_less,
                                    "resolution": resolution,
                                    "job_uid": refine_job_uid
                                })
                                self.logger.info(f"✅ Completed testing box_size {box_size_less}: Resolution = {resolution} Å")
                            else:
                                self.logger.error(f"❌ Failed to get FSC info for box_size {box_size_less}: {fsc_info_less.get('error')}")
                        except Exception as e:
                            # Continue with other tests even if one fails
                            self.logger.warning(f"⚠️ Failed to test box_size_less {box_size_less}: {e}", exc_info=True)
                            pass
                
                # Test box_size_more
                if box_size_more:
                    # Check if already tested
                    if box_size_more in [d["box_size"] for d in tested_data]:
                        self.logger.info(f"Skipping box_size_more {box_size_more} - already tested")
                    else:
                        try:
                            self.logger.info(f"🔬 Testing box_size_more: {box_size_more}")
                            
                            # Step 1: Extract particles with new box size
                            # Use refinement job for particle coordinates (refined positions/orientations)
                            self.logger.info(f"📦 Step 1/3: Extracting particles with box_size {box_size_more}...")
                            extract_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": refinement_job_uid,  # Use refinement job for refined coordinates
                                "micrographs_job_uid": micrographs_job_uid,
                                "box_size_pix": box_size_more
                            }
                            self._record_tool_execution("extract_particles", extract_params)
                            extract_result = self.cryosparc_tools.extract_particles(
                                **extract_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("extract_particles", extract_params, result=extract_result)
                            
                            # Verify extraction completed successfully
                            extract_status = extract_result.get("status", "unknown")
                            if extract_status != "completed":
                                error_msg = extract_result.get("error") or f"Status: {extract_status}"
                                self.logger.error(f"❌ Particle extraction failed for box_size {box_size_more}: {error_msg}")
                                # Don't proceed to refinement if extraction failed
                                continue
                            
                            extract_job_uid = extract_result["job_uid"]
                            self.logger.info(f"✅ Step 1/3: Extraction completed for box_size {box_size_more}, job: {extract_job_uid}")
                            
                            # Step 2: Run refinement (non-uniform or homogeneous based on config)
                            use_nonuniform = self._should_use_nonuniform_refinement()
                            refinement_type = "non-uniform" if use_nonuniform else "homogeneous"
                            self.logger.info(f"🔧 Step 2/3: Starting {refinement_type} refinement for box_size {box_size_more}...")
                            # Get symmetry from reconstruction_config.json
                            symmetry = self._get_refinement_symmetry()
                            refine_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": extract_job_uid,
                                "volume_job_uid": volume_job_uid,
                                "symmetry": symmetry
                            }
                            # Add initial lowpass resolution if configured
                            refine_res_init = self._get_refinement_res_init()
                            if refine_res_init is not None:
                                refine_params["refine_res_init"] = float(refine_res_init)
                            tool_name = "nonuniform_refine_new" if use_nonuniform else "homogeneous_refinement"
                            self._record_tool_execution(tool_name, refine_params)
                            if use_nonuniform:
                                refine_result = self.cryosparc_tools.nonuniform_refine_new(
                                    **refine_params,
                                    wait_for_completion=True,
                                    timeout=self.config.job_management.default_timeout,
                                    check_interval=self.config.job_management.status_check_interval
                                )
                            else:
                                refine_result = self.cryosparc_tools.homogeneous_refinement(
                                    **refine_params,
                                    wait_for_completion=True,
                                    timeout=self.config.job_management.default_timeout,
                                    check_interval=self.config.job_management.status_check_interval
                                )
                            self._record_tool_execution(tool_name, refine_params, result=refine_result)
                            
                            # Verify refinement completed successfully
                            # refinement returns "success" and updates with status_result
                            if not refine_result.get("success", False):
                                error_msg = refine_result.get("error") or "Unknown error"
                                self.logger.error(f"❌ Refinement failed for box_size {box_size_more}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement failed
                                continue
                            
                            refine_status = refine_result.get("status", "unknown")
                            if refine_status != "completed":
                                error_msg = refine_result.get("error") or f"Status: {refine_status}"
                                self.logger.error(f"❌ Refinement did not complete for box_size {box_size_more}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement didn't complete
                                continue
                            
                            refine_job_uid = refine_result["job_uid"]
                            self.logger.info(f"✅ Step 2/3: Refinement completed for box_size {box_size_more}, job: {refine_job_uid}")
                            
                            # Step 3: Get FSC info and resolution
                            self.logger.info(f"📊 Step 3/3: Getting FSC resolution for box_size {box_size_more}...")
                            fsc_info_more = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
                            if fsc_info_more.get("success"):
                                resolution = fsc_info_more["resolution_angstroms"]
                                self.logger.info(f"✅ Step 3/3: Box size {box_size_more}: Resolution = {resolution} Å")
                                tested_data.append({
                                    "box_size": box_size_more,
                                    "resolution": resolution,
                                    "job_uid": refine_job_uid
                                })
                                self.logger.info(f"✅ Completed testing box_size {box_size_more}: Resolution = {resolution} Å")
                            else:
                                self.logger.error(f"❌ Failed to get FSC info for box_size {box_size_more}: {fsc_info_more.get('error')}")
                        except Exception as e:
                            # Continue with other tests even if one fails
                            self.logger.warning(f"⚠️ Failed to test box_size_more {box_size_more}: {e}", exc_info=True)
                            pass
                
                # After testing, sort by box size for next iteration's analysis
                tested_data.sort(key=lambda x: x["box_size"])
            
            # Find final best result
            best_result = min(tested_data, key=lambda x: x["resolution"])
            
            result = {
                "success": True,
                "optimization_complete": True,
                "best_box_size": best_result["box_size"],
                "best_resolution_angstroms": best_result["resolution"],
                "best_job_uid": best_result["job_uid"],
                "tested_combinations": tested_data,
                "iterations": iteration
            }
            
            self._record_tool_execution("optimize_diameter", params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("optimize_diameter", params if 'params' in locals() else {}, error=str(e))
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
                elif "input" in params:
                    input_val = str(params["input"]).strip().strip('"\'')
                    if input_val.startswith("J") and len(input_val) <= 10:
                        job_uid = input_val
            
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
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about optimization workflow state."""
        try:
            reasoning = f"""
🤔 **Box Size Optimization Workflow Analysis**:

**Current State**: {input_str}

**Agentic Optimization Process**:
1. **Initial Assessment**: Use `get_fsc_info` to get baseline resolution from original refinement job
2. **First Round**: Test 10% less and 10% more box sizes using `test_box_size`
3. **Analysis**: Compare three results (original, -10%, +10%) and identify trends
4. **Iterative Refinement**: Based on analysis, decide next box size to test
5. **Continue**: Keep testing until optimal found or stopping condition reached

**Key Tools**:
- **get_fsc_info**: Get FSC resolution and box size from a refinement job
- **test_box_size**: Test a specific box size (extract + refine + get FSC)

**Parameters Needed**:
- **refinement_job_uid**: First refinement job (source of refined coordinates and baseline)
- **micrographs_job_uid**: Micrographs for re-extraction
- **volume_job_uid**: Initial volume for refinement

**Optimization Strategy**:
- Start with baseline from original refinement
- Test 10% less and 10% more initially
- Analyze trends: Which direction improves resolution?
- Test further in promising direction
- Stop after 5-7 tests or when optimal point is clear

**Decision Making**:
- Smaller resolution_angstroms value = better quality (e.g., 3.0 Å is better than 5.0 Å)
- If middle is best: Test between middle and better extreme
- If extreme is best: Test further in that direction
- Consider refinement_resolution parameter if needed

**Next Steps**:
- If not started: Get baseline FSC info, then test ±10%
- If testing: Analyze results and decide next box size
- If complete: Summarize best box size and resolution
"""
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": reasoning})
            return reasoning
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error in workflow reasoning: {str(e)}"

    # =================================================================
    # Atomic Tool Implementation Methods
    # =================================================================

    def _extract_particles_tool(self, tool_input: str) -> str:
        """Execute particle extraction with a specified box size."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(tool_input)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)

            box_size_pix = params.get("box_size_pix")
            if box_size_pix is None:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: box_size_pix"
                })

            micrographs_job_uid = params.get("micrographs_job_uid")
            if not micrographs_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: micrographs_job_uid"
                })

            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": params.get("particles_job_uid"),
                "micrographs_job_uid": micrographs_job_uid,
                "box_size_pix": int(box_size_pix),
                "wait_for_completion": self._parse_bool_param(params.get("wait_for_completion"), False),
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            passthrough = self._extract_passthrough_params(
                params,
                consumed_keys=["particles_job_uid", "micrographs_job_uid", "box_size_pix"],
            )
            if passthrough:
                used_params["params"] = passthrough

            result = self.cryosparc_tools.extract_particles(**used_params)
            self._record_tool_execution("extract_particles", used_params, result=result)
            return json.dumps(result)

        except Exception as e:
            context = used_params or params or {"raw_input": tool_input}
            self._record_tool_execution("extract_particles", context, error=str(e))
            return json.dumps({"success": False, "error": str(e)})

    def _ab_initio_tool(self, tool_input: str) -> str:
        """Execute ab initio reconstruction."""
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
            symmetry = params.get("symmetry")
            if not symmetry:
                symmetry = self._get_refinement_symmetry() or ab_initio_defaults.get("symmetry") or "C1"
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
        """Execute heterogeneous refinement.

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

            symmetry = params.get("symmetry") or self._get_refinement_symmetry()

            volume_from_job_uid = params.get("volume_from_job_uid")
            volume_job_uids = params.get("volume_job_uids") or params.get("volume_job_uid")

            wait_for_completion = self._parse_bool_param(params.get("wait_for_completion"), False)
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))

            consumed = [
                "particles_job_uid", "volume_job_uids", "volume_job_uid", "num_classes", "symmetry",
                "volume_from_job_uid", "volume_group_names", "particles_group_name",
            ]
            passthrough = self._extract_passthrough_params(params, consumed_keys=consumed)

            if volume_from_job_uid:
                # Single-job multi-volume form
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

    def _regroup_classes_tool(self, tool_input: str) -> str:
        """Regroup K classes from a heterogeneous refinement into superclasses."""
        try:
            params = self._parse_tool_input(tool_input)

            particles_job_uid = params.get("particles_job_uid") or params.get("hetero_job_uid") or params.get("job_uid")
            if not particles_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: particles_job_uid"
                })

            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            num_superclasses = int(params.get("num_superclasses", 2))
            job_title = params.get("job_title", "regroup_3D_new")

            wait_for_completion = self._parse_bool_param(params.get("wait_for_completion"), False)
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))

            regroup_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": particles_job_uid,
                "num_superclasses": num_superclasses,
                "job_title": job_title,
            }
            result = self.cryosparc_tools.regroup_classes(
                **regroup_params,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )

            self._record_tool_execution("regroup_classes", regroup_params, result=result)
            return json.dumps(result)

        except Exception as e:
            self._record_tool_execution("regroup_classes", params if 'params' in locals() else {}, error=str(e))
            return json.dumps({"success": False, "error": str(e)})

    def _get_regroup_superclass_info_tool(self, tool_input: str) -> str:
        """Get num_items per superclass from a regroup job."""
        try:
            params = self._parse_tool_input(tool_input)

            regroup_job_uid = params.get("regroup_job_uid") or params.get("job_uid")
            if not regroup_job_uid:
                input_stripped = tool_input.strip().strip('"\'')
                if input_stripped.startswith("J") and len(input_stripped) <= 10:
                    regroup_job_uid = input_stripped
            if not regroup_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: regroup_job_uid"
                })

            project_uid = params.get("project_uid", self.config.workflow.project_uid)

            result = self.cryosparc_tools.get_regroup_superclass_info(project_uid, regroup_job_uid)

            self._record_tool_execution("get_regroup_superclass_info", {"regroup_job_uid": regroup_job_uid, "project_uid": project_uid}, result=result)
            return json.dumps(result)

        except Exception as e:
            self._record_tool_execution("get_regroup_superclass_info", params if 'params' in locals() else {}, error=str(e))
            return json.dumps({"success": False, "error": str(e)})

    def _nonuniform_refinement_tool(self, tool_input: str) -> str:
        """Execute non-uniform refinement on a single 3D structure."""
        try:
            params = self._parse_tool_input(tool_input)

            particles_job_uid = params.get("particles_job_uid")
            volume_job_uid = params.get("volume_job_uid")
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
            symmetry = params.get("symmetry") or self._get_refinement_symmetry()
            refinement_resolution = params.get("refinement_resolution")

            refine_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": particles_job_uid,
                "volume_job_uid": volume_job_uid,
                "symmetry": symmetry,
            }
            _rr = self.cryosparc_tools._coerce_float(refinement_resolution)
            if _rr is not None:
                refine_params["refinement_resolution"] = _rr
            refine_res_init = params.get("refine_res_init", self._get_refinement_res_init())
            if refine_res_init is not None:
                refine_params["refine_res_init"] = float(refine_res_init)
            # Optional group-name kwargs passed through if provided
            if params.get("particles_group_name"):
                refine_params["particles_group_name"] = params.get("particles_group_name")
            if params.get("volume_group_name"):
                refine_params["volume_group_name"] = params.get("volume_group_name")

            # Raw CryoSPARC parameter passthrough (e.g. crg_min_res_A, refine_ctf_global_refine)
            passthrough = self._extract_passthrough_params(
                params,
                consumed_keys=[
                    "particles_job_uid", "volume_job_uid", "symmetry",
                    "refinement_resolution", "refine_res_init",
                    "particles_group_name", "volume_group_name",
                ],
            )
            if passthrough:
                refine_params["params"] = passthrough

            wait_for_completion = self._parse_bool_param(params.get("wait_for_completion"), False)
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))

            result = self.cryosparc_tools.nonuniform_refine_new(
                **refine_params,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )

            self._record_tool_execution("nonuniform_refine_new", refine_params, result=result)
            return json.dumps(result)

        except Exception as e:
            self._record_tool_execution("nonuniform_refine_new", params if 'params' in locals() else {}, error=str(e))
            return json.dumps({"success": False, "error": str(e)})

    def _homogeneous_refinement_tool(self, tool_input: str) -> str:
        """Execute homogeneous refinement on a single 3D structure."""
        try:
            params = self._parse_tool_input(tool_input)

            particles_job_uid = params.get("particles_job_uid")
            volume_job_uid = params.get("volume_job_uid") or params.get("job_uid")
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
            symmetry = params.get("symmetry") or self._get_refinement_symmetry()
            refinement_resolution = params.get("refinement_resolution")

            wait_for_completion = self._parse_bool_param(params.get("wait_for_completion"), False)
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))

            consumed = [
                "particles_job_uid", "volume_job_uid", "job_uid", "refinement_resolution",
                "symmetry", "particles_group_name", "volume_group_name", "refine_res_init",
            ]
            passthrough = self._extract_passthrough_params(params, consumed_keys=consumed)

            call_kwargs = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": particles_job_uid,
                "volume_job_uid": volume_job_uid,
                "symmetry": symmetry,
            }
            _rr = self.cryosparc_tools._coerce_float(refinement_resolution)
            if _rr is not None:
                call_kwargs["refinement_resolution"] = _rr
            refine_res_init = params.get("refine_res_init", self._get_refinement_res_init())
            if refine_res_init is not None:
                call_kwargs["refine_res_init"] = float(refine_res_init)
            if params.get("particles_group_name"):
                call_kwargs["particles_group_name"] = params.get("particles_group_name")
            if params.get("volume_group_name"):
                call_kwargs["volume_group_name"] = params.get("volume_group_name")

            result = self.cryosparc_tools.homogeneous_refinement(
                **call_kwargs,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval,
                params=passthrough
            )

            self._record_tool_execution("homogeneous_refinement", params, result=result)
            return json.dumps(result)

        except Exception as e:
            self._record_tool_execution("homogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps({"success": False, "error": str(e)})



