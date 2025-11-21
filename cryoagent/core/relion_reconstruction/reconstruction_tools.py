"""MCP tools for RELION reconstruction operations."""

from typing import Dict, Any, Optional
from langchain.tools import Tool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


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
    def create_validate_inputs_tool(agent) -> StructuredTool:
        """Create tool for validating input files and parameters using RELION tools."""
        # Define the input schema for StructuredTool
        class ValidateInputsInput(BaseModel):
            input_type: str = Field(description="Type of input to validate (e.g., 'particles_star', 'star_file', 'mrc_file')")
            input_path: str = Field(description="Path to the input file to validate")
            expected_format: Optional[str] = Field(default=None, description="Optional expected format of the file")
            required_metadata: Optional[str] = Field(default=None, description="Optional required metadata fields")
        
        # Create a wrapper function that converts structured input to JSON string format
        def validate_inputs_wrapper(
            input_type: str,
            input_path: str,
            expected_format: Optional[str] = None,
            required_metadata: Optional[str] = None
        ) -> str:
            """Wrapper to convert structured input to JSON string format."""
            import json
            params = {
                "input_type": input_type,
                "input_path": input_path
            }
            if expected_format:
                params["expected_format"] = expected_format
            if required_metadata:
                params["required_metadata"] = required_metadata
            # Convert to JSON string and call the original function
            json_input = json.dumps(params)
            return agent._validate_inputs_tool(json_input)
        
        return StructuredTool.from_function(
            func=validate_inputs_wrapper,
            name="validate_inputs",
            description="Validate input files and parameters before starting a RELION job. "
                       "Required parameters: input_type, input_path. "
                       "Optional parameters: expected_format, required_metadata. "
                       "Checks file existence, format, and accessibility for RELION processing.",
            args_schema=ValidateInputsInput
        )
    
    @staticmethod
    def create_import_volumes_tool(agent) -> Tool:
        """Create tool for importing volumes (half maps) into CryoSPARC for FSC validation."""
        return Tool(
            name="import_volumes",
            description="Import two half maps (volumes) from RELION refinement into CryoSPARC for FSC validation. "
                       "This imports run_half1_class001_unfil.mrc as half map A and run_half2_class001_unfil.mrc as half map B. "
                       "Required parameters: half_map_a_path (path to run_half1_class001_unfil.mrc), "
                       "half_map_b_path (path to run_half2_class001_unfil.mrc). "
                       "Optional parameters: project_uid, workspace_uid (auto-detected from master_config.json if not provided), "
                       "pixel_size, wait_for_completion, timeout, check_interval. "
                       "Half maps are auto-detected from refinement_3d output if available.",
            func=agent._import_volumes_tool
        )
    
    @staticmethod
    def create_compute_fsc_validation_tool(agent) -> Tool:
        """Create tool for computing FSC validation using CryoSPARC validation tools."""
        return Tool(
            name="compute_fsc_validation",
            description="Compute FSC (Fourier Shell Correlation) between two half maps using CryoSPARC validation tools. "
                       "This calculates the resolution and FSC curve from the imported half maps. "
                       "Required parameters: volume_a_job_uid (job UID from import_volumes for half map A with volume_out_name='map_half_A'), "
                       "volume_b_job_uid (job UID from import_volumes for half map B with volume_out_name='map_half_B'). "
                       "Optional parameters: project_uid, workspace_uid (auto-detected from master_config.json if not provided), "
                       "wait_for_completion, timeout, check_interval. "
                       "The validation job connects using result_name ('map_half_A' and 'map_half_B') directly. "
                       "Job UIDs are auto-detected from import_volumes step if available. Returns FSC resolution and validation results.",
            func=agent._compute_fsc_validation_tool
        )

