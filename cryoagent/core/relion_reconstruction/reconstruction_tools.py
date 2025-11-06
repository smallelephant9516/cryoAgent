"""MCP tools for RELION reconstruction operations."""

from typing import Dict, Any
from langchain.tools import Tool


class ReconstructionTools:
    """Factory for creating reconstruction-specific MCP tools for RELION."""
    
    @staticmethod
    def create_ab_initio_reconstruction_tool(agent) -> Tool:
        """Create tool for ab initio 3D reconstruction using RELION tools."""
        return Tool(
            name="ab_initio_reconstruction",
            description="Perform ab initio 3D reconstruction (de novo 3D refinement) from particles without a reference. "
                       "This runs relion_refine with --denovo_3dref flag, then aligns symmetry using relion_align_symmetry. "
                       "Required parameters: input_star, particle_diameter, sym. "
                       "Optional parameters: iter, K, oversampling, healpix_order, offset_range, offset_step, tau2_fudge, "
                       "pool, pad, j, gpu, ctf, flatten_solvent, zero_mask, dont_combine_weights_via_disc, auto_sampling, "
                       "grad, denovo_3dref, wait_for_completion, timeout, use_backend, conda_env.",
            func=agent._ab_initio_reconstruction_tool
        )
    
    @staticmethod
    def create_particle_reextraction_tool(agent) -> Tool:
        """Create tool for particle re-extraction with original pixel size using RELION tools."""
        return Tool(
            name="particle_reextraction",
            description="Re-extract particles from micrographs with original pixel size without scaling. "
                       "This is typically done after ab initio reconstruction to re-extract particles "
                       "at full resolution for refinement. Uses relion_preprocess with --reextract_data_star. "
                       "Required parameters: reextract_data_star (from ab initio, auto-detected if available), "
                       "micrographs_star (original micrographs STAR file). "
                       "Optional parameters: extract_size (default: -1, uses original size), norm, bg_radius, "
                       "white_dust, black_dust, invert_contrast, only_do_unfinished, wait_for_completion, "
                       "timeout, use_backend, conda_env.",
            func=agent._particle_reextraction_tool
        )
    
    @staticmethod
    def create_refinement_3d_tool(agent) -> Tool:
        """Create tool for 3D refinement using RELION tools."""
        return Tool(
            name="refinement_3d",
            description="Perform 3D refinement (auto-refinement) of particles using a reference map. "
                       "This runs relion_refine_mpi (or relion_refine) with --auto_refine flag to refine the 3D structure "
                       "with split random halves validation. "
                       "Required parameters: input_star (use re-extracted particles if available), ref_mrc, particle_diameter, sym. "
                       "Optional parameters: oversampling, healpix_order, auto_local_healpix_order, offset_range, offset_step, "
                       "pool, pad, j, gpu, ctf, flatten_solvent, zero_mask, dont_combine_weights_via_disc, auto_refine, "
                       "split_random_halves, firstiter_cc, trust_ref_size, ini_high, low_resol_join_halves, norm, scale, "
                       "wait_for_completion, timeout, use_backend, conda_env.",
            func=agent._refinement_3d_tool
        )
    
    @staticmethod
    def create_reason_about_workflow_tool(agent) -> Tool:
        """Create tool for reasoning about reconstruction workflow."""
        return Tool(
            name="reason_about_workflow",
            description="Analyze the current reconstruction workflow state and determine next steps. "
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

