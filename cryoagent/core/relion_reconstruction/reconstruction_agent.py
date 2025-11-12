"""ReAct-based reconstruction agent for RELION CryoEM data processing."""

import json
import os
import glob
import logging
from typing import Dict, Any, List
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from typing import Optional
from pathlib import Path

from ..base_react_agent import BaseReActAgent
from .reconstruction_tools import ReconstructionTools
from ...config.config_loader import CryoAgentConfig, ConfigLoader
from ...tools.relion_tools import RELIONTools


class ReconstructionAgent(BaseReActAgent):
    """ReAct-based agent for RELION CryoEM reconstruction operations."""
    
    def __init__(
        self,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the reconstruction agent.
        
        Args:
            config: Complete configuration object
            llm: Language model for the agent
        """
        # Initialize RELION tools
        self.config_loader = ConfigLoader(
            config_path="configs/relion/reconstruction_config.json",
            master_config_path="configs/master_config.json"
        )
        self.relion_tools = RELIONTools(
            self.config_loader.get_relion_settings(),
            self.config_loader
        )
        
        # Enable backend execution for RELION tools
        self.relion_tools.enable_backend_execution(True)
        self.stage_config = self._load_stage_config()
        self.stage_workflow = self.stage_config.get("workflow", {})
        
        super().__init__(None, config, llm)  # No CryoSPARC tools needed for RELION
        # Initialize logger for this agent
        self.logger = logging.getLogger("RelionReconstructionAgent")
        # Cache microscope configuration for derived defaults (respecting microscope_config overrides)
        stage_defaults = self.stage_config.get("microscope_parameters", {})
        self.microscope_config = self._resolve_microscope_defaults(stage_defaults, update_cache=True)
        
        # Initialize workflow state tracking
        self.workflow_state = {
            "ab_initio_reconstruction": {
                "completed": False,
                "job_dir": None,
                "output_file": None,
                "initial_model": None,
                "data_star": None  # Store data STAR file for re-extraction
            },
            "particle_reextraction": {
                "completed": False,
                "job_dir": None,
                "output_file": None,
                "particles_star": None
            },
            "refinement_3d": {
                "completed": False,
                "job_dir": None,
                "output_file": None,
                "refined_map": None
            }
        }
        self.context_stage_outputs: Dict[Any, Any] = {}
    def set_context_stage_outputs(self, stage_outputs: Dict[Any, Any]) -> None:
        """Store upstream stage outputs for context-aware auto-detection."""
        self.context_stage_outputs = stage_outputs or {}

    
    def _parse_boolean_param(self, value: Any) -> bool:
        """Parse boolean parameter that might be string or boolean."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'on']
        else:
            return bool(value)
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load RELION reconstruction stage configuration."""
        config_path = Path(self.config_loader.config_path)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration from stage config."""
        return self.stage_workflow
    
    def _create_tools(self) -> List[Tool]:
        """Create reconstruction-specific tools."""
        return [
            ReconstructionTools.create_ab_initio_reconstruction_tool(self),
            ReconstructionTools.create_particle_reextraction_tool(self),
            ReconstructionTools.create_refinement_3d_tool(self),
            ReconstructionTools.create_check_job_status_tool(self),
            ReconstructionTools.create_wait_for_job_tool(self),
            ReconstructionTools.create_get_job_log_tool(self),
            ReconstructionTools.create_validate_inputs_tool(self),
            ReconstructionTools.create_reason_about_workflow_tool(self)
        ]
    
    def _get_react_system_prompt(self) -> str:
        """Get the reconstruction-specific ReAct system prompt."""
        return "You are a RELION CryoEM reconstruction assistant using the ReAct (Reasoning + Acting) framework. Follow the instructions provided in the workflow input."

    # Tool implementations
    def _ab_initio_reconstruction_tool(self, input_str: str) -> str:
        """Tool wrapper for ab initio 3D reconstruction using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get required parameters
            input_star = params.get("input_star")
            if not input_star:
                return "❌ Error: input_star parameter is required"
            
            workflow_config = self._get_workflow_config()
            ab_initio_config = workflow_config.get("ab_initio_reconstruction", {})

            particle_diameter_param = params.get("particle_diameter")
            scaled_diameter = self._get_scaled_particle_diameter(1.2)
            if scaled_diameter is not None:
                if particle_diameter_param is not None and abs(float(particle_diameter_param) - scaled_diameter) > 1e-6:
                    self.logger.info(
                        "Overriding provided particle_diameter=%s with microscope-derived value %.3f Å for ab initio reconstruction",
                        particle_diameter_param,
                        scaled_diameter,
                    )
                particle_diameter = float(scaled_diameter)
            elif particle_diameter_param is not None:
                particle_diameter = float(particle_diameter_param)
            else:
                particle_diameter = None
                if "particle_diameter" in ab_initio_config:
                    try:
                        particle_diameter = float(ab_initio_config["particle_diameter"])
                    except (TypeError, ValueError):
                        particle_diameter = None
            if particle_diameter is None:
                return "❌ Error: particle_diameter parameter is required"
            def _extract_symmetry_value(*candidates: Any) -> Optional[str]:
                for value in candidates:
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                    if value not in (None, ""):
                        return str(value)
                return None

            microscope_sym = self._get_microscope_parameter("symmetry")

            sym_value = _extract_symmetry_value(
                params.get("sym"),
                ab_initio_config.get("sym") if isinstance(ab_initio_config, dict) else None,
                microscope_sym,
            ) or "C1"

            align_sym = _extract_symmetry_value(
                params.get("align_sym"),
                params.get("sym"),
                ab_initio_config.get("align_sym") if isinstance(ab_initio_config, dict) else None,
                ab_initio_config.get("sym") if isinstance(ab_initio_config, dict) else None,
                microscope_sym,
            ) or sym_value
            
            # Ensure configs are dictionaries for subsequent lookups
            if not isinstance(ab_initio_config, dict):
                ab_initio_config = {}
            
            used_params = {
                "input_star": input_star,
                "output_dir": "InitialModel",
                "iter": int(params.get("iter") or ab_initio_config.get("iter", 200)),
                "K": int(params.get("K") or ab_initio_config.get("K", 1)),
                "sym": sym_value,
                "align_sym": align_sym,
                "particle_diameter": float(particle_diameter),
                "oversampling": int(params.get("oversampling") or ab_initio_config.get("oversampling", 1)),
                "healpix_order": int(params.get("healpix_order") or ab_initio_config.get("healpix_order", 1)),
                "offset_range": float(params.get("offset_range") or ab_initio_config.get("offset_range", 6.0)),
                "offset_step": float(params.get("offset_step") or ab_initio_config.get("offset_step", 2.0)),
                "tau2_fudge": float(params.get("tau2_fudge") or ab_initio_config.get("tau2_fudge", 4.0)),
                "pool": int(params.get("pool") or ab_initio_config.get("pool", 3)),
                "pad": int(params.get("pad") or ab_initio_config.get("pad", 1)),
                "j": int(params.get("j") or ab_initio_config.get("j", 4)),
                "gpu": params.get("gpu") if params.get("gpu") is not None else ab_initio_config.get("gpu"),
                "ctf": self._parse_boolean_param(params.get("ctf", ab_initio_config.get("ctf", True))),
                "flatten_solvent": self._parse_boolean_param(params.get("flatten_solvent", ab_initio_config.get("flatten_solvent", True))),
                "zero_mask": self._parse_boolean_param(params.get("zero_mask", ab_initio_config.get("zero_mask", True))),
                "dont_combine_weights_via_disc": self._parse_boolean_param(params.get("dont_combine_weights_via_disc", ab_initio_config.get("dont_combine_weights_via_disc", True))),
                "auto_sampling": self._parse_boolean_param(params.get("auto_sampling", ab_initio_config.get("auto_sampling", True))),
                "grad": self._parse_boolean_param(params.get("grad", ab_initio_config.get("grad", True))),
                "denovo_3dref": self._parse_boolean_param(params.get("denovo_3dref", ab_initio_config.get("denovo_3dref", True))),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "false")),
                "timeout": int(params.get("timeout", 86400)),
                "use_backend": self._parse_boolean_param(params.get("use_backend", "true")),
                "conda_env": params.get("conda_env", "relion-5.0")
            }

            result = self.relion_tools.ab_initio_reconstruction(**used_params)
            # Note: _record_tool_execution logs the raw dict for debugging, but the function returns a formatted string to the LLM
            self._record_tool_execution("ab_initio_reconstruction", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Store job_dir in workflow_state for tracking
            if job_dir_relative:
                self.workflow_state["ab_initio_reconstruction"]["job_dir"] = job_dir_relative
            
            # Check if wait_for_completion is requested and job is running
            wait_for_completion = used_params.get("wait_for_completion", False)
            job_status = result.get("status")
            
            # Check if job actually completed
            if job_status == "completed":
                # Update workflow state
                self.workflow_state["ab_initio_reconstruction"]["completed"] = True
                if result.get("initial_model"):
                    self.workflow_state["ab_initio_reconstruction"]["initial_model"] = result.get("initial_model")
                if result.get("output_file"):
                    self.workflow_state["ab_initio_reconstruction"]["output_file"] = result.get("output_file")
                # Store data STAR file path for re-extraction
                if output_dir_full:
                    # Try to find the data STAR file (typically run_itXXX_data.star)
                    data_star_pattern = os.path.join(output_dir_full, "run_it*_data.star")
                    data_star_files = glob.glob(data_star_pattern)
                    if data_star_files:
                        # Get the one with highest iteration number
                        data_star_files.sort(reverse=True)
                        self.workflow_state["ab_initio_reconstruction"]["data_star"] = data_star_files[0]
                return f"✅ Successfully completed ab initio reconstruction: {result.get('initial_model')}"
            elif job_status == "running":
                # Job started but not completed yet
                self.workflow_state["ab_initio_reconstruction"]["completed"] = False
                
                # Always instruct LLM to use wait_for_job tool to monitor completion
                # Extract job directory for wait_for_job tool
                job_dir_for_wait = output_dir_full if output_dir_full else result.get('output_dir')
                if job_dir_for_wait:
                    return f"🔄 Started ab initio reconstruction job (still running). " \
                           f"Job directory: {job_dir_for_wait}. " \
                           f"Initial model will be available at: {result.get('initial_model', 'N/A')}. " \
                           f"**IMPORTANT: You must use the 'wait_for_job' tool with job_dir='{job_dir_for_wait}' to wait for this job to complete before proceeding.** " \
                           f"This job can take hours to complete. After it completes, you can proceed to 3D refinement."
                else:
                    return f"🔄 Started ab initio reconstruction job (still running): {result.get('output_dir')}. " \
                           f"**IMPORTANT: Use the 'wait_for_job' tool to monitor job completion before proceeding.**"
            else:
                # Job failed or unknown status
                return f"❌ Ab initio reconstruction job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("ab_initio_reconstruction", context, error=str(e))
            return f"❌ Error running ab initio reconstruction: {str(e)}"
    
    def _refinement_3d_tool(self, input_str: str) -> str:
        """Tool wrapper for 3D refinement using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get required parameters
            input_star = params.get("input_star")
            if not input_star:
                # Try to get from particle re-extraction output if available
                input_star = self.workflow_state["particle_reextraction"].get("particles_star")
                if not input_star:
                    return "❌ Error: input_star parameter is required. You can use re-extracted particles from particle_reextraction step if available, or provide the original particles STAR file."
            
            ref_mrc = params.get("ref_mrc")
            if not ref_mrc:
                # Try to get from ab initio reconstruction output
                ref_mrc = self.workflow_state["ab_initio_reconstruction"].get("initial_model")
                if not ref_mrc:
                    return "❌ Error: ref_mrc parameter is required and no initial model from ab initio reconstruction found"
            
            refinement_config = self._get_workflow_config().get("refinement_3d", {})

            particle_diameter_param = params.get("particle_diameter")
            scaled_diameter = self._get_scaled_particle_diameter(1.2)
            if scaled_diameter is not None:
                if particle_diameter_param is not None and abs(float(particle_diameter_param) - scaled_diameter) > 1e-6:
                    self.logger.info(
                        "Overriding provided particle_diameter=%s with microscope-derived value %.3f Å for 3D refinement",
                        particle_diameter_param,
                        scaled_diameter,
                    )
                particle_diameter = float(scaled_diameter)
            elif particle_diameter_param is not None:
                particle_diameter = float(particle_diameter_param)
            else:
                particle_diameter = None
                if "particle_diameter" in refinement_config:
                    try:
                        particle_diameter = float(refinement_config["particle_diameter"])
                    except (TypeError, ValueError):
                        particle_diameter = None
            if particle_diameter is None:
                return "❌ Error: particle_diameter parameter is required"
            
            def _extract_symmetry_value(*candidates: Any) -> Optional[str]:
                for value in candidates:
                    if isinstance(value, str) and value.strip():
                        return value.strip()
                    if value not in (None, ""):
                        return str(value)
                return None

            sym = _extract_symmetry_value(
                params.get("sym"),
                refinement_config.get("sym") if isinstance(refinement_config, dict) else None,
                self._get_microscope_parameter("symmetry"),
            ) or "C1"
            
            used_params = {
                "input_star": input_star,
                "ref_mrc": ref_mrc,
                "output_dir": "Refine3D",
                "sym": sym,
                "particle_diameter": float(particle_diameter),
                "oversampling": int(params.get("oversampling") or refinement_config.get("oversampling", 1)),
                "healpix_order": int(params.get("healpix_order") or refinement_config.get("healpix_order", 2)),
                "auto_local_healpix_order": int(params.get("auto_local_healpix_order") or refinement_config.get("auto_local_healpix_order", 4)),
                "offset_range": float(params.get("offset_range") or refinement_config.get("offset_range", 5.0)),
                "offset_step": float(params.get("offset_step") or refinement_config.get("offset_step", 2.0)),
                "pool": int(params.get("pool") or refinement_config.get("pool", 3)),
                "pad": int(params.get("pad") or refinement_config.get("pad", 2)),
                "j": int(params.get("j") or refinement_config.get("j", 2)),
                "gpu": params.get("gpu") if params.get("gpu") is not None else refinement_config.get("gpu", ""),
                "ctf": self._parse_boolean_param(params.get("ctf", refinement_config.get("ctf", True))),
                "flatten_solvent": self._parse_boolean_param(params.get("flatten_solvent", refinement_config.get("flatten_solvent", True))),
                "zero_mask": self._parse_boolean_param(params.get("zero_mask", refinement_config.get("zero_mask", True))),
                "dont_combine_weights_via_disc": self._parse_boolean_param(params.get("dont_combine_weights_via_disc", refinement_config.get("dont_combine_weights_via_disc", True))),
                "auto_refine": self._parse_boolean_param(params.get("auto_refine", refinement_config.get("auto_refine", True))),
                "split_random_halves": self._parse_boolean_param(params.get("split_random_halves", refinement_config.get("split_random_halves", True))),
                "firstiter_cc": self._parse_boolean_param(params.get("firstiter_cc", refinement_config.get("firstiter_cc", True))),
                "trust_ref_size": self._parse_boolean_param(params.get("trust_ref_size", refinement_config.get("trust_ref_size", True))),
                "ini_high": float(params.get("ini_high") or refinement_config.get("ini_high", 60.0)),
                "low_resol_join_halves": float(params.get("low_resol_join_halves") or refinement_config.get("low_resol_join_halves", 40.0)),
                "norm": self._parse_boolean_param(params.get("norm", refinement_config.get("norm", True))),
                "scale": self._parse_boolean_param(params.get("scale", refinement_config.get("scale", True))),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "false")),
                "timeout": int(params.get("timeout", 86400)),
                "use_backend": self._parse_boolean_param(params.get("use_backend", "true")),
                "conda_env": params.get("conda_env", "relion-5.0")
            }

            result = self.relion_tools.refinement_3d(**used_params)
            self._record_tool_execution("refinement_3d", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Store job_dir in workflow_state for tracking
            if job_dir_relative:
                self.workflow_state["refinement_3d"]["job_dir"] = job_dir_relative
            
            # Check if wait_for_completion is requested and job is running
            wait_for_completion = used_params.get("wait_for_completion", False)
            job_status = result.get("status")
            
            # Check if job actually completed
            if job_status == "completed":
                # Update workflow state
                self.workflow_state["refinement_3d"]["completed"] = True
                if result.get("output_file"):
                    self.workflow_state["refinement_3d"]["output_file"] = result.get("output_file")
                return f"✅ Successfully completed 3D refinement: {result.get('output_dir')}"
            elif job_status == "running":
                # Job started but not completed yet
                self.workflow_state["refinement_3d"]["completed"] = False
                
                # Always instruct LLM to use wait_for_job tool to monitor completion
                job_dir_for_wait = output_dir_full if output_dir_full else result.get('output_dir')
                if job_dir_for_wait:
                    return f"🔄 Started 3D refinement job (still running). " \
                           f"Job directory: {job_dir_for_wait}. " \
                           f"**IMPORTANT: You must use the 'wait_for_job' tool with job_dir='{job_dir_for_wait}' to wait for this job to complete.** " \
                           f"This job can take hours to complete."
                else:
                    return f"🔄 Started 3D refinement job (still running): {result.get('output_dir')}. " \
                           f"**IMPORTANT: Use the 'wait_for_job' tool to monitor job completion.**"
            else:
                # Job failed or unknown status
                return f"❌ 3D refinement job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("refinement_3d", context, error=str(e))
            return f"❌ Error running 3D refinement: {str(e)}"
    
    def _resolve_micrographs_path(self, candidate: Optional[Any]) -> Optional[str]:
        """Resolve micrographs STAR candidate path relative to RELION directory."""
        if not candidate:
            return None

        if isinstance(candidate, Path):
            cand_path = candidate
        else:
            try:
                cand_path = Path(str(candidate)).expanduser()
            except Exception:
                return None

        # If absolute path, ensure it exists
        if cand_path.is_absolute() and cand_path.exists():
            return str(cand_path.resolve())

        relion_dir = Path(self.relion_tools.relion_dir)
        rel_path = relion_dir / cand_path
        if rel_path.exists():
            return str(rel_path.resolve())

        # Try resolving relative to current working directory
        try:
            resolved = cand_path.resolve()
            if resolved.exists():
                return str(resolved)
        except Exception:
            pass

        return None

    def _extract_micrographs_from_mapping(self, data: Any) -> Optional[str]:
        """Search nested mappings for a micrographs STAR candidate."""
        if hasattr(data, "stage_outputs"):
            return self._extract_micrographs_from_mapping(getattr(data, "stage_outputs"))

        if isinstance(data, dict):
            for key in ("selected_micrographs_star", "micrographs_star"):
                value = data.get(key)
                if isinstance(value, (str, Path)):
                    return str(value)

            micro_loc = data.get("micrograph_location")
            if isinstance(micro_loc, dict):
                for key in ("micrographs_star", "selected_micrographs_star", "path", "path_absolute"):
                    value = micro_loc.get(key)
                    if isinstance(value, (str, Path)):
                        return str(value)
            elif isinstance(micro_loc, str):
                if micro_loc.endswith(".star"):
                    return micro_loc

            for value in data.values():
                candidate = self._extract_micrographs_from_mapping(value)
                if candidate:
                    return candidate

        elif isinstance(data, list):
            for item in data:
                candidate = self._extract_micrographs_from_mapping(item)
                if candidate:
                    return candidate

        elif isinstance(data, (str, Path)):
            text = str(data)
            if text.endswith("micrographs.star") or "micrographs.star" in text:
                return text

        return None

    def _extract_transition_configs(self, data: Any) -> List[str]:
        """Collect transition config file paths from nested mappings."""
        configs: List[str] = []

        if hasattr(data, "stage_outputs"):
            configs.extend(self._extract_transition_configs(getattr(data, "stage_outputs")))
            return configs

        if isinstance(data, dict):
            for key in ("transition_config", "transition_config_outputs", "transition_config_transitions"):
                value = data.get(key)
                if isinstance(value, (str, Path)):
                    configs.append(str(value))
            for value in data.values():
                configs.extend(self._extract_transition_configs(value))
        elif isinstance(data, list):
            for item in data:
                configs.extend(self._extract_transition_configs(item))

        return configs

    def _find_micrographs_star(self) -> Optional[str]:
        """Helper method to find micrographs.star file from previous preprocessing stage."""
        # Prefer transition metadata before legacy locations
        transition_configs: List[str] = []
        if isinstance(self.context_stage_outputs, dict):
            for outputs in self.context_stage_outputs.values():
                transition_configs.extend(self._extract_transition_configs(outputs))

        for config_path in transition_configs:
            resolved_config = self._resolve_micrographs_path(config_path)
            if not resolved_config or not os.path.exists(resolved_config):
                continue
            try:
                with open(resolved_config, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
            except Exception:
                continue

            transit_section = config_data.get("transit_micrographs", {})
            candidate = (
                transit_section.get("micrographs_star")
                or config_data.get("micrographs_star")
                or config_data.get("selected_micrographs_star")
                or config_data.get("micrograph_location", {}).get("micrographs_star")
            )
            resolved = self._resolve_micrographs_path(candidate)
            if resolved:
                return resolved

        # Method 0: Inspect cached stage outputs provided by orchestrator (after transition)
        candidates: List[str] = []
        if isinstance(self.context_stage_outputs, dict):
            for outputs in self.context_stage_outputs.values():
                candidate = self._extract_micrographs_from_mapping(outputs)
                resolved = self._resolve_micrographs_path(candidate)
                if resolved:
                    candidates.append(resolved)

        for candidate in candidates:
            if candidate and "transit_cs" in candidate:
                return candidate
        if candidates:
            return candidates[0]

        # Method 0.5: Check transit micrograph directory created during transition
        relion_dir = Path(self.relion_tools.relion_dir)
        transit_micrographs = relion_dir / "transit_cs" / "Micrographs" / "micrographs.star"
        if transit_micrographs.exists():
            return str(transit_micrographs.resolve())

        # Method 1: Try to find from preprocessing results JSON file
        try:
            import glob
            outputs_dir = Path("outputs")
            if outputs_dir.exists():
                # Look for preprocessing results JSON files
                result_files = glob.glob(str(outputs_dir / "preprocessing_results_relion_*.json"))
                if result_files:
                    # Sort by modification time, get most recent
                    result_files.sort(key=os.path.getmtime, reverse=True)
                    with open(result_files[0], 'r') as f:
                        results = json.load(f)
                        selected_star = results.get("stage_outputs", {}).get("selected_micrographs_star")
                        if selected_star and os.path.exists(selected_star):
                            return selected_star
        except Exception as e:
            self.logger.debug(f"Could not find micrographs.star from results file: {e}")
        
        # Method 2: Try to find Select job directories in RELION directory
        try:
            relion_dir = self.relion_tools.relion_dir
            select_pattern = os.path.join(relion_dir, "Select", "job*")
            select_dirs = glob.glob(select_pattern)
            if select_dirs:
                # Sort by directory name (job number) and get most recent
                select_dirs.sort(reverse=True)
                for select_dir in select_dirs:
                    micrographs_star = os.path.join(select_dir, "micrographs.star")
                    if os.path.exists(micrographs_star):
                        return micrographs_star
        except Exception as e:
            self.logger.debug(f"Could not find micrographs.star in Select directories: {e}")
        
        return None
    
    def _particle_reextraction_tool(self, input_str: str) -> str:
        """Tool wrapper for particle re-extraction with original pixel size using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get required parameters
            reextract_data_star = params.get("reextract_data_star")
            if not reextract_data_star:
                # Try to get from ab initio reconstruction output
                reextract_data_star = self.workflow_state["ab_initio_reconstruction"].get("data_star")
                if reextract_data_star:
                    self.logger.info(f"Auto-detected reextract_data_star from ab initio: {reextract_data_star}")
                else:
                    return "❌ Error: reextract_data_star parameter is required and no data STAR file from ab initio reconstruction found. Please complete ab initio reconstruction first."
            
            micrographs_star = params.get("micrographs_star")
            if not micrographs_star:
                # Try to auto-detect from previous preprocessing stage
                micrographs_star = self._find_micrographs_star()
                if micrographs_star:
                    self.logger.info(f"Auto-detected micrographs_star from preprocessing: {micrographs_star}")
                else:
                    return "❌ Error: micrographs_star parameter is required. " \
                           "Path to micrographs.star from Select job (e.g., Select/jobXXX/micrographs.star). " \
                           "Could not auto-detect from previous preprocessing results. " \
                           "Please check outputs/preprocessing_results_relion_*.json or provide the path manually."
            
            # Get re-extraction config from JSON file
            reextraction_config = self._get_workflow_config().get("particle_reextraction", {})
            
            # Extract size is required for re-extraction
            extract_size = params.get("extract_size")
            if extract_size is None:
                extract_size = reextraction_config.get("extract_size", -1)
            if extract_size == -1 or extract_size is None:
                return "❌ Error: extract_size parameter is required for particle re-extraction (typically 440 or larger)"
            
            used_params = {
                "reextract_data_star": reextract_data_star,
                "micrographs_star": micrographs_star,
                "output_dir": "ReExtract",
                "extract_size": int(extract_size),
                "norm": self._parse_boolean_param(params.get("norm", reextraction_config.get("norm", True))),
                "bg_radius": float(params.get("bg_radius") or reextraction_config.get("bg_radius", -1)),
                "white_dust": float(params.get("white_dust") or reextraction_config.get("white_dust", -1)),
                "black_dust": float(params.get("black_dust") or reextraction_config.get("black_dust", -1)),
                "invert_contrast": self._parse_boolean_param(params.get("invert_contrast", reextraction_config.get("invert_contrast", False))),
                "only_do_unfinished": self._parse_boolean_param(params.get("only_do_unfinished", reextraction_config.get("only_do_unfinished", False))),
                "float16": self._parse_boolean_param(params.get("float16", reextraction_config.get("float16", True))),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "false")),
                "timeout": int(params.get("timeout", 86400)),
                "use_backend": self._parse_boolean_param(params.get("use_backend", "true")),
                "conda_env": params.get("conda_env", "relion-5.0")
            }

            result = self.relion_tools.reextract_particles_original_pixelsize(**used_params)
            self._record_tool_execution("particle_reextraction", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Store job_dir in workflow_state for tracking
            if job_dir_relative:
                self.workflow_state["particle_reextraction"]["job_dir"] = job_dir_relative
            
            # Check if wait_for_completion is requested and job is running
            wait_for_completion = used_params.get("wait_for_completion", False)
            job_status = result.get("status")
            
            # Check if job actually completed
            if job_status == "completed":
                # Update workflow state
                self.workflow_state["particle_reextraction"]["completed"] = True
                particles_star = os.path.join(output_dir_full, "particles.star")
                if os.path.exists(particles_star):
                    self.workflow_state["particle_reextraction"]["particles_star"] = particles_star
                if result.get("output_file"):
                    self.workflow_state["particle_reextraction"]["output_file"] = result.get("output_file")
                return f"✅ Successfully completed particle re-extraction: {particles_star}"
            elif job_status == "running":
                # Job started but not completed yet
                self.workflow_state["particle_reextraction"]["completed"] = False
                
                # Always instruct LLM to use wait_for_job tool to monitor completion
                job_dir_for_wait = output_dir_full if output_dir_full else result.get('output_dir')
                if job_dir_for_wait:
                    return f"🔄 Started particle re-extraction job (still running). " \
                           f"Job directory: {job_dir_for_wait}. " \
                           f"**IMPORTANT: You must use the 'wait_for_job' tool with job_dir='{job_dir_for_wait}' to wait for this job to complete before proceeding.** " \
                           f"After it completes, you can proceed to 3D refinement."
                else:
                    return f"🔄 Started particle re-extraction job (still running): {result.get('output_dir')}. " \
                           f"**IMPORTANT: Use the 'wait_for_job' tool to monitor job completion before proceeding.**"
            else:
                # Job failed or unknown status
                return f"❌ Particle re-extraction job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("particle_reextraction", context, error=str(e))
            return f"❌ Error running particle re-extraction: {str(e)}"
    
    def _check_job_status_tool(self, input_str: str) -> str:
        """Tool wrapper for checking RELION job status."""
        try:
            params = self._parse_tool_input(input_str)
            job_dir = params.get("job_dir")
            
            # Handle case where job directory is passed directly as input
            if not job_dir and "input" in params:
                job_dir = params["input"]
            
            if not job_dir:
                return "❌ Error: job_dir parameter is required"
            
            result = self.relion_tools.get_job_status(job_dir)
            self._record_tool_execution("check_job_status", {"job_dir": job_dir}, result=result)
            return str(result)
            
        except Exception as e:
            self._record_tool_execution("check_job_status", {"input": input_str}, error=str(e))
            return f"❌ Error checking job status: {str(e)}"
    
    def _wait_for_job_tool(self, input_str: str) -> str:
        """Tool wrapper for waiting for RELION job completion."""
        try:
            params = self._parse_tool_input(input_str)
            job_dir = params.get("job_dir")
            
            # Handle case where job directory is passed directly as input
            if not job_dir and "input" in params:
                job_dir = params["input"]
            
            # Get defaults from config (via relion_tools)
            default_timeout = getattr(self.relion_tools, '_backend_timeout', 86400)
            default_check_interval = getattr(self.relion_tools, '_backend_check_interval', 30)
            
            timeout = int(params.get("timeout", default_timeout))
            check_interval = int(params.get("check_interval", default_check_interval))
            
            if not job_dir:
                error_msg = "❌ Error: job_dir parameter is required"
                self._record_tool_execution("wait_for_job", params, error=error_msg)
                return error_msg
            
            # Convert to relative path for comparison with workflow_state
            job_dir_abs = job_dir
            if os.path.isabs(job_dir):
                relion_dir = self.relion_tools.relion_dir
                try:
                    job_dir_relative = os.path.relpath(job_dir, relion_dir)
                except ValueError:
                    job_dir_relative = job_dir
            else:
                job_dir_relative = job_dir
                job_dir_abs = os.path.join(self.relion_tools.relion_dir, job_dir)
            
            # Wait for job completion
            result = self.relion_tools.wait_for_job_completion(job_dir_relative, timeout, check_interval)
            self._record_tool_execution("wait_for_job", {"job_dir": job_dir, "timeout": timeout, "check_interval": check_interval}, result=result)
            
            # Update workflow state based on job type
            status = result.get("status")
            if status == "completed":
                # Try to determine which step this job belongs to
                if "InitialModel" in job_dir_relative:
                    self.workflow_state["ab_initio_reconstruction"]["completed"] = True
                    # Try to get initial model path
                    if job_dir_abs:
                        initial_model = os.path.join(job_dir_abs, "initial_model.mrc")
                        if os.path.exists(initial_model):
                            self.workflow_state["ab_initio_reconstruction"]["initial_model"] = initial_model
                        # Try to find data STAR file for re-extraction
                        data_star_pattern = os.path.join(job_dir_abs, "run_it*_data.star")
                        data_star_files = glob.glob(data_star_pattern)
                        if data_star_files:
                            data_star_files.sort(reverse=True)
                            self.workflow_state["ab_initio_reconstruction"]["data_star"] = data_star_files[0]
                elif "ReExtract" in job_dir_relative:
                    self.workflow_state["particle_reextraction"]["completed"] = True
                    if job_dir_abs:
                        particles_star = os.path.join(job_dir_abs, "particles.star")
                        if os.path.exists(particles_star):
                            self.workflow_state["particle_reextraction"]["particles_star"] = particles_star
                elif "Refine3D" in job_dir_relative:
                    self.workflow_state["refinement_3d"]["completed"] = True
            
            return f"✅ Job completed with status: {status}. Result: {result}"
            
        except Exception as e:
            self._record_tool_execution("wait_for_job", {"input": input_str}, error=str(e))
            return f"❌ Error waiting for job: {str(e)}"
    
    def _get_job_log_tool(self, input_str: str) -> str:
        """Tool wrapper for reading RELION job logs."""
        try:
            params = self._parse_tool_input(input_str)
            job_dir = params.get("job_dir")
            
            # Handle case where job directory is passed directly as input
            if not job_dir and "input" in params:
                job_dir = params["input"]
            
            if not job_dir:
                return "❌ Error: job_dir parameter is required"
            
            result = self.relion_tools.get_job_log(job_dir)
            self._record_tool_execution("get_job_log", {"job_dir": job_dir}, result=result)
            return str(result)
            
        except Exception as e:
            self._record_tool_execution("get_job_log", {"input": input_str}, error=str(e))
            return f"❌ Error reading job log: {str(e)}"
    
    def _validate_inputs_tool(self, input_str: str) -> str:
        """Tool wrapper for validating input files and parameters."""
        try:
            params = self._parse_tool_input(input_str)
            input_type = params.get("input_type")
            input_path = params.get("input_path")
            
            if not input_type or not input_path:
                return "❌ Error: input_type and input_path parameters are required"
            
            result = self.relion_tools.validate_inputs(input_type, input_path)
            self._record_tool_execution("validate_inputs", {"input_type": input_type, "input_path": input_path}, result=result)
            return result
            
        except Exception as e:
            self._record_tool_execution("validate_inputs", {"input": input_str}, error=str(e))
            return f"❌ Error validating inputs: {str(e)}"
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool wrapper for analyzing current workflow state and determining next steps."""
        try:
            analysis = "🤔 **RELION Reconstruction Workflow Analysis**:\n\n"
            
            for step, state in self.workflow_state.items():
                status = "✅ COMPLETED" if state["completed"] else "⏳ PENDING"
                analysis += f"**{step.replace('_', ' ').title()}**: {status}\n"
                if state["job_dir"]:
                    analysis += f"  - Job directory: {state['job_dir']}\n"
                if state.get("initial_model"):
                    analysis += f"  - Initial model: {state['initial_model']}\n"
                if state.get("output_file"):
                    analysis += f"  - Output file: {state['output_file']}\n"
                analysis += "\n"
            
            # Determine next step
            if not self.workflow_state["ab_initio_reconstruction"]["completed"]:
                analysis += "**Next Step**: Run ab_initio_reconstruction to create initial 3D model\n"
                analysis += "- Requires: input_star (particles STAR file), particle_diameter, sym\n"
                analysis += "- This step takes a long time (typically hours)\n"
                analysis += "- Use validate_inputs first to check particles STAR file\n"
            elif not self.workflow_state["particle_reextraction"]["completed"]:
                analysis += "**Next Step**: Run particle_reextraction to re-extract particles with original pixel size\n"
                analysis += f"- Re-extract data STAR: {self.workflow_state['ab_initio_reconstruction'].get('data_star', 'N/A')} (auto-detected from ab initio)\n"
                # Try to auto-detect micrographs_star
                micrographs_star = self._find_micrographs_star()
                if micrographs_star:
                    analysis += f"- Micrographs STAR: {micrographs_star} (auto-detected from preprocessing)\n"
                else:
                    analysis += "- Micrographs STAR: Not auto-detected (will try to find from preprocessing results or Select/jobXXX/micrographs.star)\n"
                analysis += "- Requires: reextract_data_star (auto-detected from ab initio), micrographs_star (auto-detected from preprocessing, or provide manually), extract_size (required, typically 440)\n"
                analysis += "- Optional: norm, bg_radius, invert_contrast, float16, etc.\n"
                analysis += "- This step re-extracts particles at full resolution for better refinement\n"
            elif not self.workflow_state["refinement_3d"]["completed"]:
                analysis += "**Next Step**: Run refinement_3d to refine the 3D structure\n"
                reextracted_star = self.workflow_state["particle_reextraction"].get("particles_star")
                if reextracted_star:
                    analysis += f"- Input particles: {reextracted_star} (re-extracted with original pixel size)\n"
                else:
                    analysis += f"- Input particles: (from previous step)\n"
                analysis += f"- Reference model: {self.workflow_state['ab_initio_reconstruction'].get('initial_model', 'N/A')}\n"
                analysis += "- Requires: input_star (use re-extracted particles), ref_mrc (or will use initial_model from ab initio), particle_diameter, sym\n"
                analysis += "- This step also takes a long time\n"
            else:
                analysis += "**All reconstruction steps completed!** ✅\n"
                analysis += "Ready for further analysis or refinement\n"
            
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": analysis})
            return analysis
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error analyzing workflow: {str(e)}"

