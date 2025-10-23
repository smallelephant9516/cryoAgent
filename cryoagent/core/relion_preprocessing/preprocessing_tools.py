"""MCP tools for RELION preprocessing operations."""

from typing import Dict, Any
from langchain.tools import Tool


class PreprocessingTools:
    """Factory for creating preprocessing-specific MCP tools for RELION."""
    
    @staticmethod
    def create_import_movies_tool(agent) -> Tool:
        """Create tool for importing movies using relion_import."""
        return Tool(
            name="import_movies",
            description="Import movie files into RELION using relion_import. "
                       "Required parameters: None (all loaded from microscope_config.json). "
                       "Optional parameters: optics_group_name, continue_job. "
                       "All microscope parameters (movies_path, pixel_size, voltage, cs_mm, q0, beamtilt_x, beamtilt_y) are automatically loaded from microscope_config.json.",
            func=agent._import_movies_tool
        )
    
    @staticmethod
    def create_motion_correction_tool(agent) -> Tool:
        """Create tool for motion correction using relion_run_motioncorr."""
        return Tool(
            name="motion_correction",
            description="Perform motion correction using relion_run_motioncorr with MotionCor2. "
                       "Required parameters: movies_star_file. "
                       "Optional parameters: use_motioncor2, motioncor2_exe, gpu, bin_factor, bfactor, dose_per_frame, preexposure, patch_x, patch_y, eer_grouping, gain_rot, gain_flip, dose_weighting, first_frame_sum, last_frame_sum.",
            func=agent._motion_correction_tool
        )
    
    @staticmethod
    def create_ctf_estimation_tool(agent) -> Tool:
        """Create tool for CTF estimation using relion_run_ctffind."""
        return Tool(
            name="ctf_estimation",
            description="Estimate CTF parameters using relion_run_ctffind. "
                       "Required parameters: corrected_micrographs_star. "
                       "Optional parameters: box_size, res_min, res_max, df_min, df_max, fstep, dast, ctffind_exe, ctf_win, is_ctffind4, fast_search, only_do_unfinished.",
            func=agent._ctf_estimation_tool
        )
    
    @staticmethod
    def create_micrograph_selection_tool(agent) -> Tool:
        """Create tool for micrograph selection."""
        return Tool(
            name="micrograph_selection",
            description="Select micrographs with resolution better than specified threshold. "
                       "Required parameters: ctf_star_file. "
                       "Optional parameters: min_resolution, quality_threshold.",
            func=agent._micrograph_selection_tool
        )
    
    @staticmethod
    def create_reason_about_workflow_tool(agent) -> Tool:
        """Create tool for reasoning about preprocessing workflow."""
        return Tool(
            name="reason_about_workflow",
            description="Analyze the current preprocessing workflow state and determine next steps. "
                       "Use this to think through the workflow progression and identify dependencies.",
            func=agent._reason_about_workflow_tool
        )
    
    @staticmethod
    def create_check_job_status_tool(agent) -> Tool:
        """Create tool for checking RELION job status."""
        return Tool(
            name="check_job_status",
            description="Check the status of a RELION job by examining the job directory and log files. "
                       "Required parameters: job_dir.",
            func=agent._check_job_status_tool
        )
    
    @staticmethod
    def create_wait_for_job_tool(agent) -> Tool:
        """Create tool for waiting for job completion."""
        return Tool(
            name="wait_for_job",
            description="Wait for a RELION job to complete and return final status. "
                       "Required parameters: job_dir. "
                       "Optional parameters: timeout, check_interval.",
            func=agent._wait_for_job_tool
        )
    
    @staticmethod
    def create_get_job_log_tool(agent) -> Tool:
        """Create tool for reading job logs and analyzing errors."""
        return Tool(
            name="get_job_log",
            description="Read and analyze the log file of a RELION job to understand failures and get suggestions. "
                       "Required parameters: job_dir. "
                       "This tool helps diagnose why a job failed and provides suggestions for fixing the issues.",
            func=agent._get_job_log_tool
        )
    
    @staticmethod
    def create_validate_inputs_tool(agent) -> Tool:
        """Create tool for validating input files and parameters."""
        return Tool(
            name="validate_inputs",
            description="Validate input files and parameters before starting a RELION job. "
                       "Required parameters: input_type, input_path. "
                       "Optional parameters: expected_format, required_metadata.",
            func=agent._validate_inputs_tool
        )
