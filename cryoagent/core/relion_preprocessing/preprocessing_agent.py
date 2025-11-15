"""ReAct-based preprocessing agent for RELION CryoEM data processing."""

import json
import subprocess
import os
import time
import logging
from typing import Dict, Any, List
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from typing import Optional
from pathlib import Path

from ..base_react_agent import BaseReActAgent
from .preprocessing_tools import PreprocessingTools
from ...config.config_loader import CryoAgentConfig, ConfigLoader
from ...tools.relion_tools import RELIONTools
from ...tools.relion_parser_tools import RelionPreprocessingParser, WorkflowContext


class PreprocessingAgent(BaseReActAgent):
    """ReAct-based agent for RELION CryoEM preprocessing operations."""
    
    def __init__(
        self,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None,
        master_config_path: str = "configs/master_config.json"
    ):
        """
        Initialize the preprocessing agent.
        
        Args:
            config: Complete configuration object
            llm: Language model for the agent
        """
        # Initialize RELION tools
        self.config_loader = ConfigLoader(
            config_path="configs/relion/preprocessing_config.json",
            master_config_path=master_config_path
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
        self.logger = logging.getLogger("RelionPreprocessingAgent")
        # Load microscope configuration from stage config, applying microscope_config overrides when requested
        stage_defaults = self.stage_config.get("microscope_parameters", {})
        self.microscope_config = self._resolve_microscope_defaults(stage_defaults, update_cache=True)
        self.workflow_state = {
            "import_movies": {"completed": False, "job_dir": None, "output_file": None},
            "motion_correction": {"completed": False, "job_dir": None, "output_file": None},
            "ctf_estimation": {"completed": False, "job_dir": None, "output_file": None},
            "micrograph_selection": {"completed": False, "job_dir": None, "output_file": None}
        }
    
    def _parse_boolean_param(self, value: Any) -> bool:
        """Parse boolean parameter that might be string or boolean."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            return value.lower() in ['true', '1', 'yes', 'on']
        else:
            return bool(value)

    def _parse_int_param(self, value: Any, default: int = 0, param_name: str = "value") -> int:
        """Parse integer-like parameter."""
        if value is None or value == "":
            return default
        try:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped == "":
                    return default
                lowered = stripped.lower()
                if lowered in {'true', 'yes', 'on'}:
                    return 1
                if lowered in {'false', 'no', 'off'}:
                    return 0
                return int(float(stripped))
        except (TypeError, ValueError):
            self.logger.warning("Invalid %s value '%s'; defaulting to %s", param_name, value, default)
        return default
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load RELION preprocessing stage configuration."""
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
        """Create preprocessing-specific tools."""
        return [
            PreprocessingTools.create_import_movies_tool(self),
            PreprocessingTools.create_motion_correction_tool(self),
            PreprocessingTools.create_ctf_estimation_tool(self),
            PreprocessingTools.create_micrograph_selection_tool(self),
            PreprocessingTools.create_check_job_status_tool(self),
            PreprocessingTools.create_wait_for_job_tool(self),
            PreprocessingTools.create_get_job_log_tool(self),
            PreprocessingTools.create_validate_inputs_tool(self),
            PreprocessingTools.create_reason_about_workflow_tool(self)
        ]
    
    def _get_react_system_prompt(self) -> str:
        """Get the preprocessing-specific ReAct system prompt."""
        return "You are a RELION CryoEM preprocessing assistant using the ReAct (Reasoning + Acting) framework. Follow the instructions provided in the workflow input."

    # Tool implementations
    def _import_movies_tool(self, input_str: str) -> str:
        """Tool wrapper for importing movies using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get microscope parameters
            movies_path = params.get("movies_path", self.microscope_config.get("movies_path", "Micrographs/*.tif"))
            pixel_size = float(params.get("pixel_size", self.microscope_config.get("pixel_size", 0.6575)))
            voltage = float(params.get("voltage", self.microscope_config.get("voltage", 300.0)))
            cs_mm = float(params.get("cs_mm", self.microscope_config.get("cs_mm", 2.7)))
            q0 = float(params.get("q0", self.microscope_config.get("q0", 0.1)))
            beamtilt_x = float(params.get("beamtilt_x", self.microscope_config.get("beamtilt_x", 0.0)))
            beamtilt_y = float(params.get("beamtilt_y", self.microscope_config.get("beamtilt_y", 0.0)))
            
            used_params = {
                "movies_path": movies_path,
                "output_dir": "Import/job001",
                "optics_group_name": params.get("optics_group_name", "opticsGroup1"),
                "angpix": pixel_size,
                "voltage": voltage,
                "cs": cs_mm,
                "q0": q0,
                "beamtilt_x": beamtilt_x,
                "beamtilt_y": beamtilt_y,
                "output_file": "movies.star",
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "true")),
                "timeout": int(params.get("timeout", 600)),
                "use_backend": self._parse_boolean_param(params.get("use_backend", "true")),
                "conda_env": params.get("conda_env", "relion-5.0")
            }

            result = self.relion_tools.import_movies(**used_params)
            self._record_tool_execution("import_movies", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Store job_dir in workflow_state for tracking
            if job_dir_relative:
                self.workflow_state["import_movies"]["job_dir"] = job_dir_relative
            
            # Check if wait_for_completion is requested and job is running
            wait_for_completion = used_params.get("wait_for_completion", False)
            job_status = result.get("status")
            
            # Check if job actually completed
            if job_status == "completed":
                # Update workflow state
                self.workflow_state["import_movies"]["completed"] = True
                if result.get("output_file"):
                    self.workflow_state["import_movies"]["output_file"] = result.get("output_file")
                return f"✅ Successfully imported movies: {result.get('output_file')}"
            elif job_status == "running":
                # Job started but not completed yet
                self.workflow_state["import_movies"]["completed"] = False
                
                # If wait_for_completion=True, instruct LLM to call wait_for_job tool agentically
                if wait_for_completion and output_dir_full:
                    # Only pass job_dir - timeout and check_interval are handled by wait_for_job_tool from config
                    return f"🔄 Started import movies job (still running): {result.get('output_dir')}. " \
                           f"Since wait_for_completion=True, please use the 'wait_for_job' tool with job_dir: {output_dir_full} to monitor completion."
                else:
                    return f"🔄 Started import movies job (still running): {result.get('output_dir')}. Please wait for completion before proceeding."
            else:
                # Job failed or unknown status
                return f"❌ Import movies job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("import_movies", context, error=str(e))
            return f"❌ Error importing movies: {str(e)}"
    
    def _motion_correction_tool(self, input_str: str) -> str:
        """Tool wrapper for motion correction using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get input from previous step
            input_star = params.get("input_star", self.workflow_state["import_movies"]["output_file"])
            if not input_star:
                return "❌ Error: No movies.star file from import step. Run import_movies first."
            
            # Get gain reference - check both gain_ref_path and gain_ref parameter names
            # First check parameters, then microscope config
            gain_ref_path = params.get("gain_ref_path", 
                params.get("gain_ref", 
                    self.microscope_config.get("gain_ref_path")))
            
            # Get motion correction parameters from config JSON
            motion_correction_config = self._get_workflow_config().get("motion_correction", {})
            
            # Get motion correction method from config (prioritize config over explicit parameters)
            # Only use params if explicitly provided AND different from None/empty string
            use_motioncor2_param = params.get("use_motioncor2")
            if use_motioncor2_param is None or use_motioncor2_param == "":
                # Use config value if parameter not provided
                use_motioncor2 = self._parse_boolean_param(motion_correction_config.get("use_motioncor2", False))
            else:
                # Parameter was explicitly provided, use it
                use_motioncor2 = self._parse_boolean_param(use_motioncor2_param)
            
            use_own = not use_motioncor2  # If not using MotionCor2, use RELION's own implementation
            
            # Get MotionCor2 executable path from config if not provided
            motioncor2_exe = params.get("motioncor2_exe") or motion_correction_config.get("motioncor2_exe", None)

            # Determine gain reference rotation and flip values. These default to microscope configuration.
            gain_rot_value = params.get("gain_rot", None)
            if gain_rot_value is None or gain_rot_value == "":
                gain_rot_value = self.microscope_config.get(
                    "gain_rot",
                    self.microscope_config.get(
                        "gain_ref_rotation",
                        self.microscope_config.get("gain_ref_rotation_deg")
                    )
                )
            gain_rot = self._parse_int_param(gain_rot_value, default=0, param_name="gain_rot") % 4

            gain_flip_value = params.get("gain_flip", None)
            if gain_flip_value is None or gain_flip_value == "":
                gain_flip_value = self.microscope_config.get("gain_flip", self.microscope_config.get("gain_ref_flip"))
            gain_flip = self._parse_int_param(gain_flip_value, default=0, param_name="gain_flip")
            
            used_params = {
                "input_star": input_star,
                "output_dir": "MotionCorr/job002",
                "first_frame_sum": int(params.get("first_frame_sum") or motion_correction_config.get("first_frame_sum", 1)),
                "last_frame_sum": int(params.get("last_frame_sum") or motion_correction_config.get("last_frame_sum", -1)),
                "use_own": use_own,
                "use_motioncor2": use_motioncor2,
                "motioncor2_exe": motioncor2_exe,
                "num_threads": int(params.get("num_threads") or 4),
                "bin_factor": int(params.get("bin_factor") or motion_correction_config.get("bin_factor", 1)),
                "bfactor": float(params.get("bfactor") or motion_correction_config.get("bfactor", 150.0)),
                "dose_per_frame": float(params.get("dose_per_frame") or motion_correction_config.get("dose_per_frame", 1.39)),
                "preexposure": float(params.get("preexposure") or motion_correction_config.get("preexposure", 0.0)),
                "patch_x": int(params.get("patch_x") or motion_correction_config.get("patch_x", 1)),
                "patch_y": int(params.get("patch_y") or motion_correction_config.get("patch_y", 1)),
                "eer_grouping": int(params.get("eer_grouping") or motion_correction_config.get("eer_grouping", 32)),
                "gainref": gain_ref_path,
                "gain_rot": gain_rot,
                "gain_flip": gain_flip,
                "dose_weighting": self._parse_boolean_param(params.get("dose_weighting") or motion_correction_config.get("dose_weighting", True)),
                "grouping_for_ps": int(params.get("grouping_for_ps") or 3),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "true")),
                "timeout": int(params.get("timeout") or 1800),
                "use_backend": self._parse_boolean_param(params.get("use_backend", "true")),
                "conda_env": params.get("conda_env") or "relion-5.0"
            }

            result = self.relion_tools.motion_correction(**used_params)
            self._record_tool_execution("motion_correction", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Store job_dir in workflow_state for tracking
            if job_dir_relative:
                self.workflow_state["motion_correction"]["job_dir"] = job_dir_relative
            
            # Check if wait_for_completion is requested and job is running
            wait_for_completion = used_params.get("wait_for_completion", False)
            job_status = result.get("status")
            
            # Check if job actually completed
            if job_status == "completed":
                # Update workflow state
                self.workflow_state["motion_correction"]["completed"] = True
                if result.get("output_file"):
                    self.workflow_state["motion_correction"]["output_file"] = result.get("output_file")
                return f"✅ Successfully performed motion correction: {result.get('output_file')}"
            elif job_status == "running":
                # Job started but not completed yet
                self.workflow_state["motion_correction"]["completed"] = False
                
                # If wait_for_completion=True, instruct LLM to call wait_for_job tool agentically
                if wait_for_completion and output_dir_full:
                    # Only pass job_dir - timeout and check_interval are handled by wait_for_job_tool from config
                    return f"🔄 Started motion correction job (still running): {result.get('output_dir')}. " \
                           f"Since wait_for_completion=True, please use the 'wait_for_job' tool with job_dir: {output_dir_full} to monitor completion."
                else:
                    return f"🔄 Started motion correction job (still running): {result.get('output_dir')}. Please wait for completion before proceeding."
            else:
                # Job failed or unknown status
                return f"❌ Motion correction job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("motion_correction", context, error=str(e))
            return f"❌ Error during motion correction: {str(e)}"
    
    def _ctf_estimation_tool(self, input_str: str) -> str:
        """Tool wrapper for CTF estimation using RELION tools."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get input from previous step
            input_star = params.get("input_star", self.workflow_state["motion_correction"]["output_file"])
            if not input_star:
                return "❌ Error: No corrected_micrographs.star file from motion correction step. Run motion_correction first."
            
            # Get CTF parameters from config JSON
            ctf_config = self._get_workflow_config().get("ctf_estimation", {})
            
            used_params = {
                "input_star": input_star,
                "output_dir": "CtfFind/job003",
                "box_size": int(params.get("box_size", ctf_config.get("box_size", 512))),
                "res_min": float(params.get("res_min", ctf_config.get("res_min", 30.0))),
                "res_max": float(params.get("res_max", ctf_config.get("res_max", 5.0))),
                "df_min": float(params.get("df_min", ctf_config.get("df_min", 5000.0))),
                "df_max": float(params.get("df_max", ctf_config.get("df_max", 50000.0))),
                "fstep": float(params.get("fstep", ctf_config.get("fstep", 500.0))),
                "dast": float(params.get("dast", ctf_config.get("dast", 100.0))),
                "ctffind_exe": params.get("ctffind_exe", ctf_config.get("ctffind_exe", "/home/daoyi/tools/ctffind/ctffind_4_1_14/ctffind")),
                "ctf_win": int(params.get("ctf_win", ctf_config.get("ctf_win", -1))),
                "is_ctffind4": self._parse_boolean_param(params.get("is_ctffind4", str(ctf_config.get("is_ctffind4", True)))),
                "fast_search": self._parse_boolean_param(params.get("fast_search", str(ctf_config.get("fast_search", True)))),
                "only_do_unfinished": self._parse_boolean_param(params.get("only_do_unfinished", str(ctf_config.get("only_do_unfinished", True)))),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "true")),
                "timeout": int(params.get("timeout", 1800)),
                "use_backend": self._parse_boolean_param(params.get("use_backend", "true")),
                "conda_env": params.get("conda_env", "relion-5.0")
            }

            result = self.relion_tools.ctf_estimation(**used_params)
            self._record_tool_execution("ctf_estimation", used_params, result=result)
            
            # Extract relative job directory for tracking
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                relion_dir = self.relion_tools.relion_dir
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Store job_dir in workflow_state for tracking
            if job_dir_relative:
                self.workflow_state["ctf_estimation"]["job_dir"] = job_dir_relative
            
            # Check if wait_for_completion is requested and job is running
            wait_for_completion = used_params.get("wait_for_completion", False)
            job_status = result.get("status")
            
            # Check if job actually completed
            if job_status == "completed":
                # Update workflow state
                self.workflow_state["ctf_estimation"]["completed"] = True
                if result.get("output_file"):
                    self.workflow_state["ctf_estimation"]["output_file"] = result.get("output_file")
                return f"✅ Successfully estimated CTF parameters: {result.get('output_file')}"
            elif job_status == "running":
                # Job started but not completed yet
                self.workflow_state["ctf_estimation"]["completed"] = False
                
                # If wait_for_completion=True, instruct LLM to call wait_for_job tool agentically
                if wait_for_completion and output_dir_full:
                    # Only pass job_dir - timeout and check_interval are handled by wait_for_job_tool from config
                    return f"🔄 Started CTF estimation job (still running): {result.get('output_dir')}. " \
                           f"Since wait_for_completion=True, please use the 'wait_for_job' tool with job_dir: {output_dir_full} to monitor completion."
                else:
                    return f"🔄 Started CTF estimation job (still running): {result.get('output_dir')}. Please wait for completion before proceeding."
            else:
                # Job failed or unknown status
                return f"❌ CTF estimation job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
                
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("ctf_estimation", context, error=str(e))
            return f"❌ Error during CTF estimation: {str(e)}"
    
    def _micrograph_selection_tool(self, input_str: str) -> str:
        """Tool wrapper for micrograph selection using relion_star_handler."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get input from previous step
            input_star = params.get("input_star", self.workflow_state["ctf_estimation"]["output_file"])
            if not input_star:
                return "❌ Error: No CTF star file from CTF estimation step. Run ctf_estimation first."
            
            # Get micrograph selection parameters from config JSON
            selection_config = self._get_workflow_config().get("micrograph_selection", {})
            
            # Map min_resolution to minval and maxval (resolution is better when lower)            
            used_params = {
                "input_star": input_star,
                "output_dir": "Select",
                "select_field": params.get("select_field", selection_config.get("select_field", "rlnCtfMaxResolution")),
                "minval": float(params.get("minval", selection_config.get("minval"))),
                "maxval": float(params.get("maxval", selection_config.get("maxval"))),  # Use config min_resolution as maxval
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "true")),
                "timeout": int(params.get("timeout", 600)),
                "check_interval": int(params.get("check_interval", 30))
            }

            result = self.relion_tools.micrograph_selection(**used_params)
            self._record_tool_execution("micrograph_selection", used_params, result=result)
            
            # Extract relative job directory from the full output_dir path
            output_dir_full = result.get("output_dir")
            if output_dir_full:
                # Get the RELION directory from the tools
                relion_dir = self.relion_tools.relion_dir
                # Convert full path to relative job directory (e.g., "Select/job002")
                job_dir_relative = os.path.relpath(output_dir_full, relion_dir)
            else:
                job_dir_relative = None
            
            # Store job_dir in workflow_state for tracking
            if job_dir_relative:
                self.workflow_state["micrograph_selection"]["job_dir"] = job_dir_relative
            
            # Check if wait_for_completion is requested and job is running
            wait_for_completion = used_params.get("wait_for_completion", False)
            job_status = result.get("status")
            
            # Check if job actually completed
            if job_status == "completed":
                # Update workflow state
                self.workflow_state["micrograph_selection"]["completed"] = True
                if result.get("output_file"):
                    self.workflow_state["micrograph_selection"]["output_file"] = result.get("output_file")
                return f"✅ Successfully selected micrographs: {result.get('output_file')}"
            elif job_status == "running":
                # Job started but not completed yet
                self.workflow_state["micrograph_selection"]["completed"] = False
                
                # If wait_for_completion=True, instruct LLM to call wait_for_job tool agentically
                if wait_for_completion and output_dir_full:
                    # Only pass job_dir - timeout and check_interval are handled by wait_for_job_tool from config
                    return f"🔄 Started micrograph selection job (still running): {result.get('output_dir')}. " \
                           f"Since wait_for_completion=True, please use the 'wait_for_job' tool with job_dir: {output_dir_full} to monitor completion."
                else:
                    return f"🔄 Started micrograph selection job (still running): {result.get('output_dir')}. Please wait for completion before proceeding."
            else:
                # Job failed or unknown status
                return f"❌ Micrograph selection job has status: {job_status}. Error: {result.get('error', 'Unknown error')}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("micrograph_selection", context, error=str(e))
            return f"❌ Error during micrograph selection: {str(e)}"
    
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
            return result
            
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
            default_timeout = getattr(self.relion_tools, '_backend_timeout', 3600)
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
                    # If relpath fails (e.g., different drives on Windows), use absolute
                    job_dir_relative = job_dir
            else:
                job_dir_relative = job_dir
                job_dir_abs = os.path.join(self.relion_tools.relion_dir, job_dir)
            
            # Record tool execution BEFORE waiting (so it appears in log immediately)
            wait_params = {"job_dir": job_dir, "timeout": timeout, "check_interval": check_interval}
            self._record_tool_execution("wait_for_job", wait_params, result={"status": "monitoring_started", "job_dir": job_dir})
            
            # Now wait for job completion (this is blocking)
            result = self.relion_tools.wait_for_job_completion(job_dir_abs, timeout=timeout, check_interval=check_interval)
            
            # Update workflow_state when job completes
            if result.get("status") == "completed":
                # Find which step this job_dir belongs to
                matched_step = None
                for step_name, step_state in self.workflow_state.items():
                    stored_job_dir = step_state.get("job_dir")
                    if stored_job_dir:
                        # Compare relative paths
                        stored_abs = os.path.join(self.relion_tools.relion_dir, stored_job_dir) if not os.path.isabs(stored_job_dir) else stored_job_dir
                        if os.path.abspath(stored_abs) == os.path.abspath(job_dir_abs):
                            matched_step = step_name
                            break
                
                if matched_step:
                    # Update workflow_state for the matched step
                    self.workflow_state[matched_step]["completed"] = True
                    if "output_dir" in result and not self.workflow_state[matched_step].get("output_file"):
                        # Try to extract output_file from result if available
                        output_dir = result.get("output_dir")
                        if output_dir:
                            # For different step types, output files are in different locations
                            if "Import" in output_dir:
                                # Import jobs: output is movies.star
                                movies_star = os.path.join(output_dir, "movies.star")
                                if os.path.exists(movies_star):
                                    self.workflow_state[matched_step]["output_file"] = movies_star
                            elif "MotionCorr" in output_dir or "motion_correction" in output_dir.lower():
                                # Motion correction jobs: output is corrected_micrographs.star
                                corrected_star = os.path.join(output_dir, "corrected_micrographs.star")
                                if os.path.exists(corrected_star):
                                    self.workflow_state[matched_step]["output_file"] = corrected_star
                            elif "CtfFind" in output_dir or "ctf" in output_dir.lower():
                                # CTF jobs: output is micrographs_ctf.star
                                ctf_star = os.path.join(output_dir, "micrographs_ctf.star")
                                if os.path.exists(ctf_star):
                                    self.workflow_state[matched_step]["output_file"] = ctf_star
                            elif "Select" in output_dir:
                                # Selection jobs: output is micrographs.star
                                micrographs_star = os.path.join(output_dir, "micrographs.star")
                                if os.path.exists(micrographs_star):
                                    self.workflow_state[matched_step]["output_file"] = micrographs_star
                                else:
                                    # If micrographs.star doesn't exist yet, use output_dir as fallback
                                    self.workflow_state[matched_step]["output_file"] = output_dir
                    
                    self.logger.info(f"Updated workflow_state for {matched_step}: completed=True, job_dir={job_dir_relative}")
            
            # Record final result (this updates/replaces the "monitoring_started" entry in the tool_execution_log)
            # For the realtime log, it will create a new entry showing completion
            self._record_tool_execution("wait_for_job", wait_params, result=result)
            return result
            
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
            return result
            
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
            analysis = "🤔 **RELION Preprocessing Workflow Analysis**:\n\n"
            
            for step, state in self.workflow_state.items():
                status = "✅ COMPLETED" if state["completed"] else "⏳ PENDING"
                analysis += f"**{step.replace('_', ' ').title()}**: {status}\n"
                if state["job_dir"]:
                    analysis += f"  - Job directory: {state['job_dir']}\n"
                if state["output_file"]:
                    analysis += f"  - Output file: {state['output_file']}\n"
                analysis += "\n"
            
            # Determine next step
            if not self.workflow_state["import_movies"]["completed"]:
                analysis += "**Next Step**: Run import_movies to import movie files\n"
                analysis += "- Use validate_inputs first to check movie files\n"
                analysis += "- All microscope parameters are loaded from config\n"
            elif not self.workflow_state["motion_correction"]["completed"]:
                analysis += "**Next Step**: Run motion_correction to correct beam-induced motion\n"
                analysis += f"- Input: {self.workflow_state['import_movies']['output_file']}\n"
                analysis += "- Uses MotionCor2 for motion correction\n"
            elif not self.workflow_state["ctf_estimation"]["completed"]:
                analysis += "**Next Step**: Run ctf_estimation to estimate CTF parameters\n"
                analysis += f"- Input: {self.workflow_state['motion_correction']['output_file']}\n"
                analysis += "- Uses CTFfind for CTF estimation\n"
            elif not self.workflow_state["micrograph_selection"]["completed"]:
                analysis += "**Next Step**: Run micrograph_selection to filter micrographs\n"
                analysis += f"- Input: {self.workflow_state['ctf_estimation']['output_file']}\n"
                analysis += "- Filters based on CTF quality metrics\n"
                analysis += "- Completes immediately (no waiting needed)\n"
            else:
                analysis += "**All preprocessing steps completed!** ✅\n"
                analysis += "Ready for particle picking and 2D classification\n"
            
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": analysis})
            return analysis
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error analyzing workflow: {str(e)}"
    
    def process_workflow_results(self, results: List, context: WorkflowContext) -> Dict[str, Any]:
        """
        Process workflow results and extract stage outputs.
        
        Args:
            results: List of preprocessing workflow results
            context: Workflow context with project/workspace info
            
        Returns:
            Dictionary of stage outputs
        """
        parser = RelionPreprocessingParser(self.logger)
        return parser.process_workflow_results(results, context)
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the preprocessing workflow completed successfully.
        
        Args:
            stage_outputs: Dictionary of stage outputs to validate
            
        Returns:
            Dictionary with 'success' boolean and 'error' message if failed
        """
        parser = RelionPreprocessingParser(self.logger)
        return parser.validate_results(stage_outputs)
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """
        Save preprocessing results to a JSON file.
        
        Args:
            stage_outputs: Dictionary of stage outputs
            context: Workflow context
            success: Whether preprocessing was successful
            
        Returns:
            Path to the saved JSON file
        """
        parser = RelionPreprocessingParser(self.logger)
        return parser.save_results(stage_outputs, context, success)
    