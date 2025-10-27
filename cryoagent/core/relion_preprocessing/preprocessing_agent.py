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
        llm: Optional[BaseLanguageModel] = None
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
            master_config_path="configs/master_config.json"
        )
        self.relion_tools = RELIONTools(
            self.config_loader.get_relion_settings(),
            self.config_loader
        )
        
        super().__init__(None, config, llm)  # No CryoSPARC tools needed for RELION
        # Initialize logger for this agent
        self.logger = logging.getLogger("RelionPreprocessingAgent")
        # Load microscope configuration
        self.microscope_config = self._load_microscope_config()
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
    
    def _load_microscope_config(self) -> Dict[str, Any]:
        """Load microscope configuration from separate config file."""
        try:
            # Get the microscope config path from the workflow configuration
            microscope_config_path = getattr(self.config.workflow, 'microscope_config_path', 'configs/microscope_config.json')
            
            # If it's a relative path, make it relative to the project root
            if not Path(microscope_config_path).is_absolute():
                # Assume the config is relative to the project root
                microscope_config_path = Path.cwd() / microscope_config_path
            
            config_path = Path(microscope_config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Microscope configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                microscope_data = json.load(f)
            
            # Return the microscope parameters
            return microscope_data.get('microscope_parameters', {})
            
        except Exception as e:
            print(f"Warning: Could not load microscope configuration: {e}")
            # Return default values if loading fails
            return {
                "pixel_size": 0.6575,
                "voltage": 300.0,
                "cs_mm": 2.7,
                "q0": 0.1,
                "beamtilt_x": 0,
                "beamtilt_y": 0
            }
    
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
        # Safely get microscope config values, handling case where it might not be set yet
        microscope_config = getattr(self, 'microscope_config', {})
        
        return f"""You are a RELION CryoEM preprocessing assistant using the ReAct (Reasoning + Acting) framework. 
You specialize in the initial stages of cryoEM data processing using RELION: movie import, motion correction, CTF estimation, and micrograph selection.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Preprocessing Workflow Steps (in order):
1. **Import Movies**: Import raw movie files into RELION using relion_import
   - Required: None (all parameters loaded from microscope_config.json)
   - Optional: optics_group_name, continue_job
   - Note: All microscope parameters (movies_path, pixel_size, voltage, cs_mm, q0, beamtilt_x, beamtilt_y) are automatically loaded from microscope_config.json
   
2. **Motion Correction**: Correct beam-induced motion using relion_run_motioncorr with MotionCor2
   - Required: movies_star_file (from import_movies)
   - Optional: use_motioncor2, motioncor2_exe, gain_ref_path, first_frame_sum, last_frame_sum, use_own, num_threads, bin_factor, bfactor, dose_per_frame, preexposure, patch_x, patch_y, eer_grouping, gain_rot, gain_flip, dose_weighting, grouping_for_ps, wait_for_completion, timeout
   
3. **CTF Estimation**: Estimate Contrast Transfer Function parameters using relion_run_ctffind
   - Required: corrected_micrographs_star (from motion_correction)
   - Optional: box_size, res_min, res_max, df_min, df_max, fstep, dast, ctffind_exe, ctf_win, is_ctffind4, fast_search, only_do_unfinished
   
4. **Micrograph Selection**: Filter micrographs based on quality metrics
   - Required: ctf_star_file (from ctf_estimation)
   - Optional: min_resolution, quality_threshold

## ReAct Process:
For each step, you MUST follow this pattern:

**Thought**: [Your reasoning about what needs to be done, why, and how]
**Action**: [The specific tool to use with exact parameters]
**Observation**: [What happened as a result of the action]

## CRITICAL: Job Monitoring and Waiting
- After starting any job, you MUST wait for it to complete using wait_for_job
- Do NOT proceed to the next step until the current job is completed
- If a job fails, report the error and stop the workflow

## Tool Usage Guidelines:
- validate_inputs: Use JSON format {{"input_type": "movie_files", "input_path": "/path/to/files"}}
- import_movies: Start the import, then wait for completion
- motion_correction: Requires movies_star_file from completed import_movies job
- ctf_estimation: Requires corrected_micrographs_star from completed motion_correction job
- micrograph_selection: Requires ctf_star_file from completed ctf_estimation job
- check_job_status: Check status of a specific job directory
- wait_for_job: Wait for job completion
- reason_about_workflow: Analyze current preprocessing state and dependencies

## Job Directory Format (Alphabetical Order):
- Job directories follow alphabetical order: "Import/job001/", "MotionCorr/job002/", "CtfFind/job003/", "Select/job004/"
- For job-related tools (check_job_status, wait_for_job, get_job_log), you can pass the job directory directly (e.g., "Import/job001")
- For validate_inputs, use JSON format: {{"input_type": "movie_files", "input_path": "/path/to/files"}}

## Workflow Dependencies:
1. Import movies → Motion correction → CTF estimation → Micrograph selection
2. Each step must complete successfully before the next can begin
3. Always verify job completion before proceeding

## Current Configuration:
- Microscope Config: {getattr(self.config.workflow, 'microscope_config_path', 'configs/microscope_config.json')}
- Movies Path: {microscope_config.get('movies_path', 'N/A')}
- Pixel Size: {microscope_config.get('pixel_size', 'N/A')} Å
- Voltage: {microscope_config.get('voltage', 'N/A')} kV
- CS: {microscope_config.get('cs_mm', 'N/A')} mm
- Q0: {microscope_config.get('q0', 'N/A')}
- Beam tilt X: {microscope_config.get('beamtilt_x', 'N/A')}
- Beam tilt Y: {microscope_config.get('beamtilt_y', 'N/A')}

## Important Notes:
- Always validate inputs before starting jobs
- Use continue_job=true for import_movies to resume interrupted jobs
- Check job logs if jobs fail to understand the issue
- Follow the exact RELION command structure as shown in the examples
- Ensure all required executables (MotionCor2, ctffind) are available and properly configured

Remember: You are working with RELION, not CryoSPARC. Use the appropriate RELION commands and file formats.
"""

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
                "timeout": int(params.get("timeout", 600))
            }

            result = self.relion_tools.import_movies(**used_params)
            self._record_tool_execution("import_movies", used_params, result=result)
            
            # Update workflow state
            self.workflow_state["import_movies"]["completed"] = True
            self.workflow_state["import_movies"]["job_dir"] = result.get("output_dir")
            self.workflow_state["import_movies"]["output_file"] = result.get("output_file")
            
            return f"✅ Successfully imported movies: {result.get('output_file')}"
                
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
            # First check parameters, then workflow config, then microscope config
            gain_ref_path = params.get("gain_ref_path", 
                params.get("gain_ref", 
                    self.config.workflow.gain_ref or 
                        self.microscope_config.get("gain_ref_path")))
            
            # Get motion correction method from config or parameters
            # Check config first, then params
            config_use_motioncor2 = getattr(self.config.workflow, 'use_motioncor2', False)
            use_motioncor2 = self._parse_boolean_param(params.get("use_motioncor2", config_use_motioncor2))
            use_own = not use_motioncor2  # If not using MotionCor2, use RELION's own implementation
            
            # Get MotionCor2 executable path from config if not provided
            motioncor2_exe = params.get("motioncor2_exe", 
                getattr(self.config.workflow, 'motioncor2_exe', None))
            
            used_params = {
                "input_star": input_star,
                "output_dir": "MotionCorr/job002",
                "first_frame_sum": int(params.get("first_frame_sum", 1)),
                "last_frame_sum": int(params.get("last_frame_sum", -1)),
                "use_own": use_own,
                "use_motioncor2": use_motioncor2,
                "motioncor2_exe": motioncor2_exe,
                "num_threads": int(params.get("num_threads", 4)),
                "bin_factor": int(params.get("bin_factor", 1)),
                "bfactor": float(params.get("bfactor", 150.0)),
                "dose_per_frame": float(params.get("dose_per_frame", 1.39)),
                "preexposure": float(params.get("preexposure", 0.0)),
                "patch_x": int(params.get("patch_x", 1)),
                "patch_y": int(params.get("patch_y", 1)),
                "eer_grouping": int(params.get("eer_grouping", 32)),
                "gainref": gain_ref_path,
                "gain_rot": int(params.get("gain_rot", 0)),
                "gain_flip": int(params.get("gain_flip", 0)),
                "dose_weighting": self._parse_boolean_param(params.get("dose_weighting", "true")),
                "grouping_for_ps": int(params.get("grouping_for_ps", 3)),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "true")),
                "timeout": int(params.get("timeout", 1800))
            }

            result = self.relion_tools.motion_correction(**used_params)
            self._record_tool_execution("motion_correction", used_params, result=result)
            
            # Update workflow state
            self.workflow_state["motion_correction"]["completed"] = True
            self.workflow_state["motion_correction"]["job_dir"] = result.get("output_dir")
            self.workflow_state["motion_correction"]["output_file"] = result.get("output_file")
            
            return f"✅ Successfully performed motion correction: {result.get('output_file')}"
                
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
            
            used_params = {
                "input_star": input_star,
                "output_dir": "CtfFind/job003",
                "box_size": int(params.get("box_size", getattr(self.config.workflow, 'box_size', 512))),
                "res_min": float(params.get("res_min", getattr(self.config.workflow, 'ctf_min_res', 30.0))),
                "res_max": float(params.get("res_max", getattr(self.config.workflow, 'ctf_max_res', 5.0))),
                "df_min": float(params.get("df_min", getattr(self.config.workflow, 'df_min', 5000.0))),
                "df_max": float(params.get("df_max", getattr(self.config.workflow, 'df_max', 50000.0))),
                "fstep": float(params.get("fstep", getattr(self.config.workflow, 'fstep', 500.0))),
                "dast": float(params.get("dast", getattr(self.config.workflow, 'dast', 100.0))),
                "ctffind_exe": params.get("ctffind_exe", getattr(self.config.workflow, 'ctffind_exe', "/home/daoyi/tools/ctffind/ctffind_4_1_14/ctffind")),
                "ctf_win": int(params.get("ctf_win", getattr(self.config.workflow, 'ctf_win', -1))),
                "is_ctffind4": self._parse_boolean_param(params.get("is_ctffind4", str(getattr(self.config.workflow, 'is_ctffind4', True)))),
                "fast_search": self._parse_boolean_param(params.get("fast_search", str(getattr(self.config.workflow, 'fast_search', True)))),
                "only_do_unfinished": self._parse_boolean_param(params.get("only_do_unfinished", str(getattr(self.config.workflow, 'only_do_unfinished', True)))),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "true")),
                "timeout": int(params.get("timeout", 1800))
            }

            result = self.relion_tools.ctf_estimation(**used_params)
            self._record_tool_execution("ctf_estimation", used_params, result=result)
            
            # Update workflow state
            self.workflow_state["ctf_estimation"]["completed"] = True
            self.workflow_state["ctf_estimation"]["job_dir"] = result.get("output_dir")
            self.workflow_state["ctf_estimation"]["output_file"] = result.get("output_file")
            
            return f"✅ Successfully estimated CTF parameters: {result.get('output_file')}"
                
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("ctf_estimation", context, error=str(e))
            return f"❌ Error during CTF estimation: {str(e)}"
    
    def _micrograph_selection_tool(self, input_str: str) -> str:
        """Tool wrapper for micrograph selection."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            
            # Get input from previous step
            input_star = params.get("input_star", self.workflow_state["ctf_estimation"]["output_file"])
            if not input_star:
                return "❌ Error: No CTF star file from CTF estimation step. Run ctf_estimation first."
            
            used_params = {
                "input_star": input_star,
                "output_dir": "Select",
                "min_resolution": float(params.get("min_resolution", getattr(self.config.workflow, 'min_resolution', 5.0))),
                "quality_threshold": float(params.get("quality_threshold", getattr(self.config.workflow, 'quality_threshold', 0.8))),
                "wait_for_completion": self._parse_boolean_param(params.get("wait_for_completion", "true")),
                "timeout": int(params.get("timeout", 600))
            }

            result = self.relion_tools.micrograph_selection(**used_params)
            self._record_tool_execution("micrograph_selection", used_params, result=result)
            
            # Update workflow state
            self.workflow_state["micrograph_selection"]["completed"] = True
            self.workflow_state["micrograph_selection"]["job_dir"] = result.get("output_dir")
            self.workflow_state["micrograph_selection"]["output_file"] = result.get("output_file")
            
            return f"✅ Successfully selected micrographs: {result.get('output_file')}"
            
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
            
            timeout = int(params.get("timeout", 3600))
            check_interval = int(params.get("check_interval", 30))
            
            if not job_dir:
                return "❌ Error: job_dir parameter is required"
            
            result = self.relion_tools.wait_for_job_completion(job_dir, timeout, check_interval)
            self._record_tool_execution("wait_for_job", {"job_dir": job_dir, "timeout": timeout, "check_interval": check_interval}, result=result)
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
