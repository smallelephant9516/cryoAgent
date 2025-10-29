"""ReAct-based particle picking agent for RELION CryoEM data processing."""

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
from .picking_tools import PickingTools
from ...config.config_loader import CryoAgentConfig, ConfigLoader
from ...tools.relion_tools import RELIONTools
from ...tools.relion_parser_tools import RelionPreprocessingParser, WorkflowContext
from ...tools.cryosparc_tools import CryoSPARCTools


class PickingAgent(BaseReActAgent):
    """ReAct-based agent for RELION particle picking operations."""
    
    def __init__(
        self,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the particle picking agent.
        
        Args:
            config: Complete configuration object
            llm: Language model for the agent
        """
        # Initialize RELION tools
        self.config_loader = ConfigLoader(
            config_path="configs/master_config.json",
            master_config_path="configs/master_config.json"
        )
        self.relion_tools = RELIONTools(
            self.config_loader.get_relion_settings(),
            self.config_loader
        )
        
        # Enable backend execution for RELION tools
        self.relion_tools.enable_backend_execution(True)
        
        super().__init__(None, config, llm)  # No CryoSPARC tools needed for RELION
        # Initialize logger for this agent
        self.logger = logging.getLogger(__name__)
        
        # Initialize workflow
        from .picking_workflow import PickingWorkflow
        self.workflow = PickingWorkflow()
        
        self.logger.info("Particle Picking Agent initialized")
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for particle picking operations."""
        tools = []
        
        # Particle picking tools
        tools.append(PickingTools.create_blob_picker_tool(self))
        tools.append(PickingTools.create_particle_extraction_tool(self))
        tools.append(PickingTools.create_classification_2d_tool(self))
        tools.append(PickingTools.create_auto_2d_selection_tool(self))
        
        # Utility tools
        tools.append(PickingTools.create_reason_about_workflow_tool(self))
        tools.append(PickingTools.create_check_job_status_tool(self))
        tools.append(PickingTools.create_wait_for_job_tool(self))
        tools.append(PickingTools.create_get_job_log_tool(self))
        tools.append(PickingTools.create_validate_inputs_tool(self))
        
        return tools
    
    def _get_react_system_prompt(self) -> str:
        """Get the ReAct system prompt for particle picking operations."""
        return self._get_agent_instructions()
    
    def _get_agent_instructions(self) -> str:
        """Get specific instructions for the particle picking agent."""
        return """
You are a specialized RELION particle picking agent for cryoEM data processing. Your role is to:

1. **Blob Picking**: Use Laplacian-of-Gaussian (LoG) filtering to automatically detect particles in micrographs
2. **Particle Extraction**: Extract particle images from micrographs using coordinate files
3. **2D Classification**: Classify extracted particles into 2D classes to identify good particles
4. **Auto 2D Selection**: Automatically select the best 2D classes and particles for further processing

**Workflow Steps:**
1. Start with `blob_picker` using micrographs from micrograph selection
2. Run `particle_extraction` to extract particles using the coordinates from blob picking
3. Perform `classification_2d` to classify the extracted particles
4. Use `auto_2d_selection` to select the best classes and particles

**Key Parameters:**
- particle_diameter: Expected particle size in Angstroms (typically 100-300 Å)
- extract_size: Box size for particle extraction (typically 256-512 pixels)
- K: Number of 2D classes (typically 50-100)
- min_score: Minimum score for class selection (typically 0.5-0.7)

**Important Notes:**
- Always validate inputs before starting jobs
- Use backend execution for long-running jobs
- Monitor job progress and handle failures appropriately
- Check job logs if failures occur
- Ensure proper file paths and dependencies between steps

**Error Handling:**
- If a job fails, check the job log for specific error messages
- Validate input files exist and are accessible
- Use appropriate timeouts for different job types
- Consider adjusting parameters if jobs consistently fail

Remember to think step by step and use the available tools to complete the particle picking workflow successfully.
"""
    
    # Particle Picking Tool Methods
    def _blob_picker_tool(self, **kwargs) -> str:
        """Tool for blob picking using RELION tools."""
        try:
            # Get parameters with defaults
            input_star = kwargs.get('input_star')
            if not input_star:
                return "Error: input_star parameter is required"
            
            # Set default parameters
            params = {
                'output_dir': 'AutoPick/job005',
                'particle_diameter': kwargs.get('particle_diameter', 200.0),
                'angpix': kwargs.get('angpix', 1.0),
                'threshold': kwargs.get('threshold', 0.25),
                'min_distance': kwargs.get('min_distance', -1),
                'LoG': kwargs.get('LoG', True),
                'LoG_diam_min': kwargs.get('LoG_diam_min', 100.0),
                'LoG_diam_max': kwargs.get('LoG_diam_max', 300.0),
                'LoG_neighbour': kwargs.get('LoG_neighbour', 100.0),
                'LoG_adjust_threshold': kwargs.get('LoG_adjust_threshold', 0.0),
                'LoG_upper_threshold': kwargs.get('LoG_upper_threshold', 99999.0),
                'LoG_use_ctf': kwargs.get('LoG_use_ctf', False),
                'gauss_max': kwargs.get('gauss_max', 0.1),
                'write_fom_maps': kwargs.get('write_fom_maps', False),
                'only_do_unfinished': kwargs.get('only_do_unfinished', False),
                'wait_for_completion': kwargs.get('wait_for_completion', True),
                'timeout': kwargs.get('timeout', 3600),
                'use_backend': kwargs.get('use_backend', False),
                'conda_env': kwargs.get('conda_env', 'relion-5.0')
            }
            
            # Run blob picker
            result = self.relion_tools.blob_picker(input_star=input_star, **params)
            
            return f"Blob picker completed successfully. Output directory: {result['output_dir']}"
            
        except Exception as e:
            return f"Blob picker failed: {str(e)}"
    
    def _particle_extraction_tool(self, **kwargs) -> str:
        """Tool for particle extraction using RELION tools."""
        try:
            # Get parameters with defaults
            input_star = kwargs.get('input_star')
            if not input_star:
                return "Error: input_star parameter is required"
            
            # Set default parameters
            params = {
                'output_dir': 'Extract/job006',
                'coord_suffix': kwargs.get('coord_suffix', '_autopick.star'),
                'coord_dir': kwargs.get('coord_dir', 'ASINPUT'),
                'extract_size': kwargs.get('extract_size', 256),
                'norm': kwargs.get('norm', True),
                'bg_radius': kwargs.get('bg_radius', -1),
                'white_dust': kwargs.get('white_dust', -1),
                'black_dust': kwargs.get('black_dust', -1),
                'invert_contrast': kwargs.get('invert_contrast', False),
                'extract_bias_x': kwargs.get('extract_bias_x', 0.0),
                'extract_bias_y': kwargs.get('extract_bias_y', 0.0),
                'only_do_unfinished': kwargs.get('only_do_unfinished', False),
                'wait_for_completion': kwargs.get('wait_for_completion', True),
                'timeout': kwargs.get('timeout', 3600),
                'use_backend': kwargs.get('use_backend', False),
                'conda_env': kwargs.get('conda_env', 'relion-5.0')
            }
            
            # Run particle extraction
            result = self.relion_tools.particle_extraction(input_star=input_star, **params)
            
            return f"Particle extraction completed successfully. Output directory: {result['output_dir']}"
            
        except Exception as e:
            return f"Particle extraction failed: {str(e)}"
    
    def _classification_2d_tool(self, **kwargs) -> str:
        """Tool for 2D classification using RELION tools."""
        try:
            # Get parameters with defaults
            input_star = kwargs.get('input_star')
            if not input_star:
                return "Error: input_star parameter is required"
            
            # Set default parameters
            params = {
                'output_dir': 'Class2D',
                'K': kwargs.get('K', 50),
                'iter': kwargs.get('iter', 25),
                'tau2_fudge': kwargs.get('tau2_fudge', 2.0),
                'particle_diameter': kwargs.get('particle_diameter', 200.0),
                'offset_range': kwargs.get('offset_range', 6.0),
                'offset_step': kwargs.get('offset_step', 2.0),
                'oversampling': kwargs.get('oversampling', 1),
                'healpix_order': kwargs.get('healpix_order', 2),
                'psi_step': kwargs.get('psi_step', -1),
                'skip_align': kwargs.get('skip_align', False),
                'skip_rotate': kwargs.get('skip_rotate', False),
                'ctf': kwargs.get('ctf', True),
                'norm': kwargs.get('norm', True),
                'scale': kwargs.get('scale', True),
                'pool': kwargs.get('pool', 1),
                'j': kwargs.get('j', 1),
                'only_do_unfinished': kwargs.get('only_do_unfinished', False),
                'wait_for_completion': kwargs.get('wait_for_completion', True),
                'timeout': kwargs.get('timeout', 7200),
                'use_backend': kwargs.get('use_backend', False),
                'conda_env': kwargs.get('conda_env', 'relion-5.0')
            }
            
            # Run 2D classification
            result = self.relion_tools.classification_2d(input_star=input_star, **params)
            
            return f"2D classification completed successfully. Output directory: {result['output_dir']}"
            
        except Exception as e:
            return f"2D classification failed: {str(e)}"
    
    def _auto_2d_selection_tool(self, **kwargs) -> str:
        """Tool for automatic 2D class selection using RELION tools."""
        try:
            # Get parameters with defaults
            input_opt = kwargs.get('input_opt')
            if not input_opt:
                return "Error: input_opt parameter is required"
            
            # Set default parameters
            params = {
                'output_dir': 'Select',
                'min_score': kwargs.get('min_score', 0.5),
                'max_score': kwargs.get('max_score', 999.0),
                'select_min_nr_particles': kwargs.get('select_min_nr_particles', -1),
                'select_min_nr_classes': kwargs.get('select_min_nr_classes', -1),
                'relative_thresholds': kwargs.get('relative_thresholds', False),
                'auto_select': kwargs.get('auto_select', True),
                'fn_sel_parts': kwargs.get('fn_sel_parts', 'particles.star'),
                'fn_sel_classavgs': kwargs.get('fn_sel_classavgs', 'class_averages.star'),
                'wait_for_completion': kwargs.get('wait_for_completion', True),
                'timeout': kwargs.get('timeout', 1800)
            }
            
            # Run auto 2D selection
            result = self.relion_tools.auto_2d_selection(input_opt=input_opt, **params)
            
            return f"Auto 2D selection completed successfully. Output directory: {result['output_dir']}"
            
        except Exception as e:
            return f"Auto 2D selection failed: {str(e)}"
    
    # Utility tool methods (reuse from preprocessing agent)
    def _reason_about_workflow_tool(self, **kwargs) -> str:
        """Tool for reasoning about the particle picking workflow."""
        try:
            # Get current workflow state
            steps = self.workflow.get_workflow_steps()
            dependencies = self.workflow.get_workflow_dependencies()
            
            analysis = f"Particle Picking Workflow Analysis:\n"
            analysis += f"Total steps: {len(steps)}\n"
            analysis += f"Steps: {[step['step_name'] for step in steps]}\n"
            analysis += f"Dependencies: {dependencies}\n"
            
            # Add reasoning about next steps
            analysis += "\nWorkflow progression:\n"
            analysis += "1. blob_picker: Detect particles in micrographs using LoG filtering\n"
            analysis += "2. particle_extraction: Extract particle images using coordinates\n"
            analysis += "3. classification_2d: Classify particles into 2D classes\n"
            analysis += "4. auto_2d_selection: Select best classes and particles\n"
            
            return analysis
            
        except Exception as e:
            return f"Workflow analysis failed: {str(e)}"
    
    def _check_job_status_tool(self, **kwargs) -> str:
        """Tool for checking RELION job status."""
        try:
            job_dir = kwargs.get('job_dir')
            if not job_dir:
                return "Error: job_dir parameter is required"
            
            # Use RELION tools to check job status
            status = self.relion_tools.get_job_status(job_dir)
            
            return f"Job status: {status.get('status', 'unknown')}\nOutput directory: {status.get('output_dir', 'N/A')}"
            
        except Exception as e:
            return f"Job status check failed: {str(e)}"
    
    def _wait_for_job_tool(self, **kwargs) -> str:
        """Tool for waiting for job completion."""
        try:
            job_dir = kwargs.get('job_dir')
            if not job_dir:
                return "Error: job_dir parameter is required"
            
            timeout = kwargs.get('timeout', 3600)
            check_interval = kwargs.get('check_interval', 30)
            
            # Use RELION tools to wait for job
            result = self.relion_tools.wait_for_job_completion(
                job_dir, timeout=timeout, check_interval=check_interval
            )
            
            return f"Job completed with status: {result.get('status', 'unknown')}"
            
        except Exception as e:
            return f"Job wait failed: {str(e)}"
    
    def _get_job_log_tool(self, **kwargs) -> str:
        """Tool for reading job logs."""
        try:
            job_dir = kwargs.get('job_dir')
            if not job_dir:
                return "Error: job_dir parameter is required"
            
            # Use RELION tools to get job log
            log_result = self.relion_tools.get_job_log(job_dir)
            
            if log_result.get('success'):
                return f"Job log retrieved successfully. Error analysis: {log_result.get('error_analysis', {})}"
            else:
                return f"Failed to get job log: {log_result.get('error', 'Unknown error')}"
            
        except Exception as e:
            return f"Job log retrieval failed: {str(e)}"
    
    def _validate_inputs_tool(self, **kwargs) -> str:
        """Tool for validating inputs."""
        try:
            input_type = kwargs.get('input_type')
            input_path = kwargs.get('input_path')
            
            if not input_type or not input_path:
                return "Error: input_type and input_path parameters are required"
            
            # Use RELION tools to validate inputs
            result = self.relion_tools.validate_inputs(input_type, input_path)
            
            return result
            
        except Exception as e:
            return f"Input validation failed: {str(e)}"
    
    def run_particle_picking_workflow(
        self,
        input_micrographs_star: str,
        particle_diameter: float = 200.0,
        extract_size: int = 256,
        num_classes: int = 50,
        num_iterations: int = 25,
        min_score: float = 0.5,
        use_backend: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete particle picking workflow.
        
        Args:
            input_micrographs_star: Path to input micrographs STAR file
            particle_diameter: Diameter of particles in Angstroms
            extract_size: Size of particle box in pixels
            num_classes: Number of 2D classes
            num_iterations: Number of classification iterations
            min_score: Minimum score for class selection
            use_backend: Whether to use backend execution
            
        Returns:
            Dictionary containing workflow results
        """
        self.logger.info(f"Starting particle picking workflow with input: {input_micrographs_star}")
        
        # Create workflow context
        context = WorkflowContext(
            project_name="particle_picking",
            workflow_type="particle_picking",
            input_files={"micrographs_star": input_micrographs_star}
        )
        
        # Run the workflow using the agent
        results = self.run_workflow(
            f"Run the complete particle picking workflow with the following parameters: "
            f"input_micrographs_star={input_micrographs_star}, "
            f"particle_diameter={particle_diameter}, "
            f"extract_size={extract_size}, "
            f"num_classes={num_classes}, "
            f"num_iterations={num_iterations}, "
            f"min_score={min_score}, "
            f"use_backend={use_backend}",
            context
        )
        
        return results
