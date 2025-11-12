"""MCP tools for RELION preprocessing operations."""

from typing import Dict, Any
from langchain.tools import Tool


class PreprocessingTools:
    """Factory for creating preprocessing-specific MCP tools for RELION."""
    
    @staticmethod
    def create_import_movies_tool(agent) -> Tool:
        """Create tool for importing movies using RELION tools."""
        return Tool(
            name="import_movies",
            description="Import movie files into RELION using relion_import. "
                       "Required parameters: None (all loaded from microscope_config.json). "
                       "Optional parameters: movies_path, pixel_size, voltage, cs_mm, q0, beamtilt_x, beamtilt_y, optics_group_name, wait_for_completion, timeout. "
                       "All microscope parameters are automatically loaded from microscope_config.json if not provided.",
            func=agent._import_movies_tool
        )
    
    @staticmethod
    def create_motion_correction_tool(agent) -> Tool:
        """Create tool for motion correction using RELION tools."""
        return Tool(
            name="motion_correction",
            description="Perform motion correction using relion_run_motioncorr with MotionCor2. "
                       "Required parameters: input_star (from import_movies). "
                       "Optional parameters: gain_ref_path, first_frame_sum, last_frame_sum, use_own or use_motioncor2, motioncor2_exe, num_threads, bin_factor, bfactor, dose_per_frame, preexposure, patch_x, patch_y, eer_grouping, gain_rot, gain_flip, dose_weighting, grouping_for_ps, wait_for_completion, timeout. "
                       "Gain rotation/flip defaults are sourced directly from microscope_config.json.",
            func=agent._motion_correction_tool
        )
    
    @staticmethod
    def create_ctf_estimation_tool(agent) -> Tool:
        """Create tool for CTF estimation using RELION tools."""
        return Tool(
            name="ctf_estimation",
            description="Estimate CTF parameters using relion_run_ctffind. "
                       "Required parameters: input_star (from motion_correction). "
                       "Optional parameters: box_size, res_min, res_max, df_min, df_max, fstep, dast, ctffind_exe, ctf_win, is_ctffind4, fast_search, only_do_unfinished, wait_for_completion, timeout.",
            func=agent._ctf_estimation_tool
        )
    
    @staticmethod
    def create_micrograph_selection_tool(agent) -> Tool:
        """Create tool for micrograph selection using relion_star_handler."""
        return Tool(
            name="micrograph_selection",
            description="Select micrographs using relion_star_handler with filter criteria. "
                       "Required parameters: input_star (from ctf_estimation). "
                       "Optional parameters: select_field (default: rlnCtfMaxResolution), minval (default: 2.0), "
                       "maxval (default: 5.0), min_resolution, wait_for_completion, timeout, check_interval. "
                       "Example: Uses relion_star_handler --select rlnCtfMaxResolution --minval 2 --maxval 5 to filter micrographs.",
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
        """Create tool for checking RELION job status using RELION tools."""
        return Tool(
            name="check_job_status",
            description="Check the status of a RELION job by examining the job directory and log files. "
                       "Required parameters: job_dir. "
                       "Returns job status information including completion state and any errors.",
            func=agent._check_job_status_tool
        )
    
    @staticmethod
    def create_wait_for_job_tool(agent) -> Tool:
        """Create tool for waiting for job completion using RELION tools."""
        return Tool(
            name="wait_for_job",
            description="Wait for a RELION job to complete and return final status. "
                       "Required parameters: job_dir. "
                       "Optional parameters: timeout, check_interval. "
                       "Monitors job progress and returns when completed or failed.",
            func=agent._wait_for_job_tool
        )
    
    @staticmethod
    def create_get_job_log_tool(agent) -> Tool:
        """Create tool for reading job logs and analyzing errors using RELION tools."""
        return Tool(
            name="get_job_log",
            description="Read and analyze the log file of a RELION job to understand failures and get suggestions. "
                       "Required parameters: job_dir. "
                       "This tool helps diagnose why a job failed and provides suggestions for fixing the issues.",
            func=agent._get_job_log_tool
        )
    
    @staticmethod
    def create_validate_inputs_tool(agent) -> Tool:
        """Create tool for validating input files and parameters using RELION tools."""
        return Tool(
            name="validate_inputs",
            description="Validate input files and parameters before starting a RELION job. "
                       "Required parameters: input_type, input_path. "
                       "Optional parameters: expected_format, required_metadata. "
                       "Checks file existence, format, and accessibility for RELION processing.",
            func=agent._validate_inputs_tool
        )
    