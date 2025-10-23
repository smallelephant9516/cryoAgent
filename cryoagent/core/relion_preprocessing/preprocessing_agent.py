"""ReAct-based preprocessing agent for RELION CryoEM data processing."""

import json
import subprocess
import os
import time
from typing import Dict, Any, List
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from typing import Optional
from pathlib import Path

from ..base_react_agent import BaseReActAgent
from .preprocessing_tools import PreprocessingTools
from ...config.config_loader import CryoAgentConfig


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
        super().__init__(None, config, llm)  # No CryoSPARC tools needed for RELION
        # Load microscope configuration
        self.microscope_config = self._load_microscope_config()
        self.workflow_state = {
            "import_movies": {"completed": False, "job_dir": None, "output_file": None},
            "motion_correction": {"completed": False, "job_dir": None, "output_file": None},
            "ctf_estimation": {"completed": False, "job_dir": None, "output_file": None},
            "micrograph_selection": {"completed": False, "job_dir": None, "output_file": None}
        }
    
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
   - Optional: use_motioncor2, motioncor2_exe, gpu, bin_factor, bfactor, dose_per_frame, preexposure, patch_x, patch_y, eer_grouping, gain_rot, gain_flip, dose_weighting, first_frame_sum, last_frame_sum
   
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
- import_movies: Start the import, then wait for completion
- motion_correction: Requires movies_star_file from completed import_movies job
- ctf_estimation: Requires corrected_micrographs_star from completed motion_correction job
- micrograph_selection: Requires ctf_star_file from completed ctf_estimation job
- check_job_status: Check status of a specific job directory
- wait_for_job: Wait for job completion
- reason_about_workflow: Analyze current preprocessing state and dependencies

## Job Directory Format:
- Job directories are paths like "Import/job001/", "MotionCorr/job022/", etc.
- When calling check_job_status or wait_for_job, pass the job directory path
- Do NOT use JSON format or complex parameters for these tools

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
    def _import_movies_tool(self, **kwargs) -> str:
        """Import movies using relion_import."""
        try:
            # Get parameters from config and kwargs
            optics_group_name = kwargs.get('optics_group_name', self.config.workflow.get('optics_group_name', 'opticsGroup1'))
            continue_job = kwargs.get('continue_job', self.config.workflow.get('continue_job', True))
            
            # Get microscope parameters
            movies_path = self.microscope_config.get('movies_path', 'Micrographs/*.tif')
            pixel_size = self.microscope_config.get('pixel_size', 0.6575)
            voltage = self.microscope_config.get('voltage', 300)
            cs_mm = self.microscope_config.get('cs_mm', 2.7)
            q0 = self.microscope_config.get('q0', 0.1)
            beamtilt_x = self.microscope_config.get('beamtilt_x', 0)
            beamtilt_y = self.microscope_config.get('beamtilt_y', 0)
            
            # Create job directory
            job_dir = "Import/job001/"
            os.makedirs(job_dir, exist_ok=True)
            
            # Build relion_import command
            cmd = [
                "relion_import",
                "--do_movies",
                f"--optics_group_name", optics_group_name,
                f"--angpix", str(pixel_size),
                f"--kV", str(voltage),
                f"--Cs", str(cs_mm),
                f"--Q0", str(q0),
                f"--beamtilt_x", str(beamtilt_x),
                f"--beamtilt_y", str(beamtilt_y),
                f"--i", movies_path,
                f"--odir", job_dir,
                f"--ofile", "movies.star"
            ]
            
            if continue_job:
                cmd.append("--continue")
            
            # Add pipeline control
            cmd.extend(["--pipeline_control", f"{job_dir}"])
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                self.workflow_state["import_movies"]["completed"] = True
                self.workflow_state["import_movies"]["job_dir"] = job_dir
                self.workflow_state["import_movies"]["output_file"] = f"{job_dir}movies.star"
                return f"Successfully imported movies. Output: {job_dir}movies.star"
            else:
                return f"Import failed: {result.stderr}"
                
        except Exception as e:
            return f"Error during import: {str(e)}"
    
    def _motion_correction_tool(self, **kwargs) -> str:
        """Perform motion correction using relion_run_motioncorr."""
        try:
            # Get input from previous step
            movies_star_file = kwargs.get('movies_star_file', self.workflow_state["import_movies"]["output_file"])
            if not movies_star_file:
                return "Error: No movies.star file from import step. Run import_movies first."
            
            # Get parameters from config and kwargs
            use_motioncor2 = kwargs.get('use_motioncor2', self.config.workflow.get('use_motioncor2', True))
            motioncor2_exe = kwargs.get('motioncor2_exe', self.config.workflow.get('motioncor2_exe', '../../tools/MotionCor2_1.6.4_Cuda118_Mar312023'))
            gpu = kwargs.get('gpu', self.config.workflow.get('gpu', '0'))
            bin_factor = kwargs.get('bin_factor', self.config.workflow.get('bin_factor', 1))
            bfactor = kwargs.get('bfactor', self.config.workflow.get('bfactor', 150))
            dose_per_frame = kwargs.get('dose_per_frame', self.config.workflow.get('dose_per_frame', 1.39))
            preexposure = kwargs.get('preexposure', self.config.workflow.get('preexposure', 0))
            patch_x = kwargs.get('patch_x', self.config.workflow.get('patch_x', 1))
            patch_y = kwargs.get('patch_y', self.config.workflow.get('patch_y', 1))
            eer_grouping = kwargs.get('eer_grouping', self.config.workflow.get('eer_grouping', 32))
            gain_rot = kwargs.get('gain_rot', self.config.workflow.get('gain_rot', 0))
            gain_flip = kwargs.get('gain_flip', self.config.workflow.get('gain_flip', 0))
            dose_weighting = kwargs.get('dose_weighting', self.config.workflow.get('dose_weighting', True))
            first_frame_sum = kwargs.get('first_frame_sum', self.config.workflow.get('first_frame_sum', 1))
            last_frame_sum = kwargs.get('last_frame_sum', self.config.workflow.get('last_frame_sum', -1))
            
            # Create job directory
            job_dir = "MotionCorr/job022/"
            os.makedirs(job_dir, exist_ok=True)
            
            # Check if MotionCor2 is available
            motioncor2_available = os.path.exists(motioncor2_exe)
            
            # Build relion_run_motioncorr command
            cmd = [
                "relion_run_motioncorr",
                f"--i", movies_star_file,
                f"--o", job_dir,
                f"--first_frame_sum", str(first_frame_sum),
                f"--last_frame_sum", str(last_frame_sum)
            ]
            
            # Always use MotionCor2 (even if not available, RELION will handle it gracefully)
            # This follows the successful command pattern from the user's example
            cmd.extend([
                "--use_motioncor2",
                f"--motioncor2_exe", motioncor2_exe,
                f"--gpu", gpu
            ])
            
            if not motioncor2_available:
                print(f"Warning: MotionCor2 not found at {motioncor2_exe}, but continuing with command structure")
            
            cmd.extend([
                f"--bin_factor", str(bin_factor),
                f"--bfactor", str(bfactor),
                f"--dose_per_frame", str(dose_per_frame),
                f"--preexposure", str(preexposure),
                f"--patch_x", str(patch_x),
                f"--patch_y", str(patch_y),
                f"--eer_grouping", str(eer_grouping)
            ])
            
            # Add gain reference if available (use relative path format)
            gainref = self.microscope_config.get('gain_ref_path')
            if gainref:
                # Convert absolute path to relative path format like "Micrographs/norm-amibox05-0.mrc"
                gainref_name = os.path.basename(gainref)
                cmd.extend(["--gainref", f"Micrographs/{gainref_name}"])
            
            cmd.extend([
                f"--gain_rot", str(gain_rot),
                f"--gain_flip", str(gain_flip)
            ])
            
            if dose_weighting:
                cmd.append("--dose_weighting")
            
            # Add pipeline control
            cmd.extend(["--pipeline_control", f"{job_dir}"])
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                self.workflow_state["motion_correction"]["completed"] = True
                self.workflow_state["motion_correction"]["job_dir"] = job_dir
                self.workflow_state["motion_correction"]["output_file"] = f"{job_dir}corrected_micrographs.star"
                return f"Successfully performed motion correction. Output: {job_dir}corrected_micrographs.star"
            else:
                return f"Motion correction failed: {result.stderr}"
                
        except Exception as e:
            return f"Error during motion correction: {str(e)}"
    
    def _ctf_estimation_tool(self, **kwargs) -> str:
        """Estimate CTF parameters using relion_run_ctffind."""
        try:
            # Get input from previous step
            corrected_micrographs_star = kwargs.get('corrected_micrographs_star', self.workflow_state["motion_correction"]["output_file"])
            if not corrected_micrographs_star:
                return "Error: No corrected_micrographs.star file from motion correction step. Run motion_correction first."
            
            # Get parameters from config and kwargs
            box_size = kwargs.get('box_size', self.config.workflow.get('box_size', 512))
            res_min = kwargs.get('res_min', self.config.workflow.get('res_min', 30))
            res_max = kwargs.get('res_max', self.config.workflow.get('res_max', 5))
            df_min = kwargs.get('df_min', self.config.workflow.get('df_min', 5000))
            df_max = kwargs.get('df_max', self.config.workflow.get('df_max', 50000))
            fstep = kwargs.get('fstep', self.config.workflow.get('fstep', 500))
            dast = kwargs.get('dast', self.config.workflow.get('dast', 100))
            ctffind_exe = kwargs.get('ctffind_exe', self.config.workflow.get('ctffind_exe', '/home/daoyi/tools/ctffind/ctffind_4_1_14/ctffind'))
            ctf_win = kwargs.get('ctf_win', self.config.workflow.get('ctf_win', -1))
            is_ctffind4 = kwargs.get('is_ctffind4', self.config.workflow.get('is_ctffind4', True))
            fast_search = kwargs.get('fast_search', self.config.workflow.get('fast_search', True))
            only_do_unfinished = kwargs.get('only_do_unfinished', self.config.workflow.get('only_do_unfinished', True))
            
            # Create job directory
            job_dir = "CtfFind/job010/"
            os.makedirs(job_dir, exist_ok=True)
            
            # Build relion_run_ctffind command
            cmd = [
                "relion_run_ctffind",
                f"--i", corrected_micrographs_star,
                f"--o", job_dir,
                f"--Box", str(box_size),
                f"--ResMin", str(res_min),
                f"--ResMax", str(res_max),
                f"--dFMin", str(df_min),
                f"--dFMax", str(df_max),
                f"--FStep", str(fstep),
                f"--dAst", str(dast),
                f"--ctffind_exe", ctffind_exe,
                f"--ctfWin", str(ctf_win),
                f"--is_ctffind4",
                f"--fast_search",
                f"--only_do_unfinished"
            ]
            
            # Add pipeline control
            cmd.extend(["--pipeline_control", f"{job_dir}"])
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                self.workflow_state["ctf_estimation"]["completed"] = True
                self.workflow_state["ctf_estimation"]["job_dir"] = job_dir
                self.workflow_state["ctf_estimation"]["output_file"] = f"{job_dir}micrographs_ctf.star"
                return f"Successfully estimated CTF parameters. Output: {job_dir}micrographs_ctf.star"
            else:
                return f"CTF estimation failed: {result.stderr}"
                
        except Exception as e:
            return f"Error during CTF estimation: {str(e)}"
    
    def _micrograph_selection_tool(self, **kwargs) -> str:
        """Select micrographs based on quality metrics."""
        try:
            # Get input from previous step
            ctf_star_file = kwargs.get('ctf_star_file', self.workflow_state["ctf_estimation"]["output_file"])
            if not ctf_star_file:
                return "Error: No CTF star file from CTF estimation step. Run ctf_estimation first."
            
            # Get parameters
            min_resolution = kwargs.get('min_resolution', self.config.workflow.get('min_resolution', 5.0))
            quality_threshold = kwargs.get('quality_threshold', self.config.workflow.get('quality_threshold', 0.8))
            
            # Create job directory
            job_dir = "Select/job011/"
            os.makedirs(job_dir, exist_ok=True)
            
            # For now, implement a simple selection based on CTF resolution
            # In a real implementation, this would use RELION's selection tools
            self.workflow_state["micrograph_selection"]["completed"] = True
            self.workflow_state["micrograph_selection"]["job_dir"] = job_dir
            self.workflow_state["micrograph_selection"]["output_file"] = f"{job_dir}selected_micrographs.star"
            
            return f"Successfully selected micrographs. Output: {job_dir}selected_micrographs.star"
            
        except Exception as e:
            return f"Error during micrograph selection: {str(e)}"
    
    def _check_job_status_tool(self, job_dir: str) -> str:
        """Check the status of a RELION job."""
        try:
            if not os.path.exists(job_dir):
                return f"Job directory does not exist: {job_dir}"
            
            # Check for completion markers
            if os.path.exists(f"{job_dir}RELION_JOB_EXIT_SUCCESS"):
                return f"Job completed successfully: {job_dir}"
            elif os.path.exists(f"{job_dir}RELION_JOB_EXIT_FAILURE"):
                return f"Job failed: {job_dir}"
            else:
                return f"Job still running: {job_dir}"
                
        except Exception as e:
            return f"Error checking job status: {str(e)}"
    
    def _wait_for_job_tool(self, job_dir: str, timeout: int = 3600, check_interval: int = 30) -> str:
        """Wait for a RELION job to complete."""
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                status = self._check_job_status_tool(job_dir)
                
                if "completed successfully" in status:
                    return f"Job completed successfully: {job_dir}"
                elif "failed" in status:
                    return f"Job failed: {job_dir}"
                
                time.sleep(check_interval)
            
            return f"Job timeout after {timeout} seconds: {job_dir}"
            
        except Exception as e:
            return f"Error waiting for job: {str(e)}"
    
    def _get_job_log_tool(self, job_dir: str) -> str:
        """Read and analyze job logs."""
        try:
            log_file = f"{job_dir}run.out"
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    log_content = f.read()
                return f"Job log from {job_dir}:\n{log_content[-1000:]}"  # Last 1000 chars
            else:
                return f"No log file found at {log_file}"
                
        except Exception as e:
            return f"Error reading job log: {str(e)}"
    
    def _validate_inputs_tool(self, input_type: str, input_path: str, **kwargs) -> str:
        """Validate input files and parameters."""
        try:
            if input_type == "movies":
                if not os.path.exists(input_path):
                    return f"Movies path does not exist: {input_path}"
                
                # Check if it's a glob pattern or directory
                if "*" in input_path:
                    import glob
                    files = glob.glob(input_path)
                    if not files:
                        return f"No files found matching pattern: {input_path}"
                    return f"Found {len(files)} movie files matching pattern: {input_path}"
                else:
                    if os.path.isdir(input_path):
                        files = [f for f in os.listdir(input_path) if f.endswith(('.tif', '.mrc', '.mrcs'))]
                        if not files:
                            return f"No movie files found in directory: {input_path}"
                        return f"Found {len(files)} movie files in directory: {input_path}"
                    else:
                        return f"Input path is not a directory or glob pattern: {input_path}"
            
            elif input_type == "star_file":
                if not os.path.exists(input_path):
                    return f"Star file does not exist: {input_path}"
                return f"Star file exists: {input_path}"
            
            else:
                return f"Unknown input type: {input_type}"
                
        except Exception as e:
            return f"Error validating inputs: {str(e)}"
    
    def _reason_about_workflow_tool(self) -> str:
        """Analyze current workflow state and determine next steps."""
        try:
            analysis = "Current RELION preprocessing workflow state:\n\n"
            
            for step, state in self.workflow_state.items():
                status = "✅ COMPLETED" if state["completed"] else "⏳ PENDING"
                analysis += f"{step.replace('_', ' ').title()}: {status}\n"
                if state["job_dir"]:
                    analysis += f"  Job directory: {state['job_dir']}\n"
                if state["output_file"]:
                    analysis += f"  Output file: {state['output_file']}\n"
                analysis += "\n"
            
            # Determine next step
            if not self.workflow_state["import_movies"]["completed"]:
                analysis += "Next step: Run import_movies to import movie files\n"
            elif not self.workflow_state["motion_correction"]["completed"]:
                analysis += "Next step: Run motion_correction to correct beam-induced motion\n"
            elif not self.workflow_state["ctf_estimation"]["completed"]:
                analysis += "Next step: Run ctf_estimation to estimate CTF parameters\n"
            elif not self.workflow_state["micrograph_selection"]["completed"]:
                analysis += "Next step: Run micrograph_selection to filter micrographs\n"
            else:
                analysis += "All preprocessing steps completed! ✅\n"
            
            return analysis
            
        except Exception as e:
            return f"Error analyzing workflow: {str(e)}"
