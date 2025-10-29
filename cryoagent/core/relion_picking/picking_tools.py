"""MCP tools for RELION particle picking operations."""

from typing import Dict, Any
from langchain.tools import Tool


class PickingTools:
    """Factory for creating particle picking-specific MCP tools for RELION."""
    
    @staticmethod
    def create_blob_picker_tool(agent) -> Tool:
        """Create tool for blob picking using RELION tools."""
        return Tool(
            name="blob_picker",
            description="Perform blob picking using Laplacian-of-Gaussian filter with relion_autopick. "
                       "Required parameters: input_star (micrographs from micrograph_selection). "
                       "Optional parameters: particle_diameter, angpix, threshold, min_distance, "
                       "LoG, LoG_diam_min, LoG_diam_max, LoG_neighbour, LoG_adjust_threshold, "
                       "LoG_upper_threshold, LoG_use_ctf, gauss_max, write_fom_maps, only_do_unfinished, "
                       "wait_for_completion, timeout, use_backend, conda_env.",
            func=agent._blob_picker_tool
        )
    
    @staticmethod
    def create_particle_extraction_tool(agent) -> Tool:
        """Create tool for particle extraction using RELION tools."""
        return Tool(
            name="particle_extraction",
            description="Extract particles from micrographs using coordinate files with relion_preprocess. "
                       "Required parameters: input_star (micrographs), coord_suffix (e.g., '_autopick.star'). "
                       "Optional parameters: coord_dir, extract_size, norm, bg_radius, white_dust, black_dust, "
                       "invert_contrast, extract_bias_x, extract_bias_y, only_do_unfinished, "
                       "wait_for_completion, timeout, use_backend, conda_env.",
            func=agent._particle_extraction_tool
        )
    
    @staticmethod
    def create_classification_2d_tool(agent) -> Tool:
        """Create tool for 2D classification using RELION tools."""
        return Tool(
            name="classification_2d",
            description="Perform 2D classification of particles using relion_refine. "
                       "Required parameters: input_star (particles from particle_extraction). "
                       "Optional parameters: K (number of classes), iter (iterations), tau2_fudge, "
                       "particle_diameter, offset_range, offset_step, oversampling, healpix_order, "
                       "psi_step, skip_align, skip_rotate, ctf, norm, scale, pool, j, only_do_unfinished, "
                       "wait_for_completion, timeout, use_backend, conda_env.",
            func=agent._classification_2d_tool
        )
    
    @staticmethod
    def create_auto_2d_selection_tool(agent) -> Tool:
        """Create tool for automatic 2D class selection using RELION tools."""
        return Tool(
            name="auto_2d_selection",
            description="Automatically select good 2D classes and particles using relion_class_ranker. "
                       "Required parameters: input_opt (optimiser.star from classification_2d). "
                       "Optional parameters: min_score, max_score, select_min_nr_particles, "
                       "select_min_nr_classes, relative_thresholds, auto_select, fn_sel_parts, "
                       "fn_sel_classavgs, wait_for_completion, timeout.",
            func=agent._auto_2d_selection_tool
        )
    
    @staticmethod
    def create_reason_about_workflow_tool(agent) -> Tool:
        """Create tool for reasoning about particle picking workflow."""
        return Tool(
            name="reason_about_workflow",
            description="Analyze the current particle picking workflow state and determine next steps. "
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
