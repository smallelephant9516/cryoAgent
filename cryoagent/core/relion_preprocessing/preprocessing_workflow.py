"""ReAct-based RELION preprocessing workflow orchestrator."""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .preprocessing_agent import PreprocessingAgent
from ...config.config_loader import CryoAgentConfig


class PreprocessingStep(Enum):
    """Enumeration of preprocessing workflow steps."""
    IMPORT_MOVIES = "import_movies"
    MOTION_CORRECTION = "motion_correction"
    CTF_ESTIMATION = "ctf_estimation"
    MICROGRAPH_SELECTION = "micrograph_selection"


@dataclass
class PreprocessingResult:
    """Result of a preprocessing workflow execution."""
    step: PreprocessingStep
    success: bool
    job_dir: Optional[str] = None
    output_file: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None


class PreprocessingWorkflow:
    """ReAct-based orchestrator for RELION preprocessing workflows."""
    
    def __init__(self, agent: PreprocessingAgent, config: CryoAgentConfig):
        """
        Initialize the preprocessing workflow.
        
        Args:
            agent: Preprocessing agent instance
            config: Complete configuration object
        """
        self.agent = agent
        self.config = config
        self.results: List[PreprocessingResult] = []
        self.current_job_dirs: Dict[PreprocessingStep, str] = {}
        self.workflow_state: Dict[str, Any] = {}
    
    def run(self, conversation_id: Optional[str] = None) -> List[PreprocessingResult]:
        """
        Run the complete preprocessing workflow using ReAct approach.
        
        Args:
            conversation_id: Optional conversation identifier for memory control
            
        Returns:
            List of preprocessing results for each step
        """
        self.results = []
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "active_jobs": {},
            "workflow_status": "starting"
        }
        
        workflow_input = self._create_workflow_input()
        
        try:
            result = self.agent.run_react_workflow(workflow_input, conversation_id)
            self._parse_workflow_result(result)
            
        except Exception as e:
            error_result = PreprocessingResult(
                step=PreprocessingStep.IMPORT_MOVIES,
                success=False,
                error=f"Preprocessing workflow execution failed: {str(e)}",
                message="ReAct workflow failed to execute"
            )
            self.results.append(error_result)
        
        return self.results
    
    def _create_workflow_input(self) -> str:
        """Create the workflow input for the ReAct agent."""
        # Get microscope config from the agent
        microscope_config = getattr(self.agent, 'microscope_config', {})
        
        return f"""
Execute the complete RELION preprocessing workflow using the ReAct framework. Follow the Thought → Action → Observation pattern for each step.

## Workflow Steps (in order):

1. **Import Movies**: Import movie files using RELION tools
   - Movies path: {microscope_config.get('movies_path', 'Micrographs/*.tif')}
   - Pixel size: {microscope_config.get('pixel_size', 'N/A')} Å
   - Voltage: {microscope_config.get('voltage', 'N/A')} kV
   - CS: {microscope_config.get('cs_mm', 'N/A')} mm
   - Q0: {microscope_config.get('q0', 'N/A')}
   - Beam tilt X: {microscope_config.get('beamtilt_x', 'N/A')}
   - Beam tilt Y: {microscope_config.get('beamtilt_y', 'N/A')}
   - Optics group: {getattr(self.config.workflow, 'optics_group_name', 'opticsGroup1')}

2. **Motion Correction**: Correct motion using RELION tools with MotionCor2
   - Use MotionCor2: {getattr(self.config.workflow, 'use_motioncor2', False)}
   - MotionCor2 executable: {getattr(self.config.workflow, 'motioncor2_exe', '../../tools/MotionCor2_1.6.4_Cuda118_Mar312023')}
   - GPU: {getattr(self.config.workflow, 'gpu', '0')}
   - Bin factor: {getattr(self.config.workflow, 'motion_correction_binning', 1)}
   - B-factor: {getattr(self.config.workflow, 'bfactor', 150)}
   - Dose per frame: {getattr(self.config.workflow, 'dose_per_frame', 1.39)}
   - Dose weighting: {getattr(self.config.workflow, 'dose_weighting', True)}

3. **CTF Estimation**: Estimate CTF parameters using RELION tools
   - Box size: {getattr(self.config.workflow, 'box_size', 512)}
   - Resolution range: {getattr(self.config.workflow, 'ctf_min_res', 30)} - {getattr(self.config.workflow, 'ctf_max_res', 5)} Å
   - Defocus range: {getattr(self.config.workflow, 'df_min', 5000)} - {getattr(self.config.workflow, 'df_max', 50000)} Å
   - CTFfind executable: {getattr(self.config.workflow, 'ctffind_exe', '/home/daoyi/tools/ctffind/ctffind_4_1_14/ctffind')}
   - Fast search: {getattr(self.config.workflow, 'fast_search', True)}

4. **Micrograph Selection**: Select high-quality micrographs using RELION tools
   - Minimum resolution: {getattr(self.config.workflow, 'min_resolution', 5.0)} Å
   - Quality threshold: {getattr(self.config.workflow, 'quality_threshold', 0.8)}

## ReAct Process Requirements:
- **Thought**: Analyze what needs to be done and why
- **Action**: Execute the appropriate tool with correct parameters
- **Observation**: Analyze results and determine next steps

## Critical Workflow Rules:
- Execute steps in order: Import → Motion Correction → CTF Estimation → Selection
- Wait for each job to complete before starting the next
- Validate inputs before starting each step using validate_inputs
- Check job status and logs if any step fails
- Use wait_for_job to monitor job completion
- Use reason_about_workflow to analyze current state

## Expected Outputs:
- Import: Import/job001/movies.star
- Motion Correction: MotionCorr/job022/corrected_micrographs.star
- CTF Estimation: CtfFind/job010/micrographs_ctf.star
- Selection: Select/job011/selected_micrographs.star

## Tool Usage:
- Use validate_inputs to check movie files before import
- Use import_movies to start the import process
- Use wait_for_job to monitor job completion
- Use motion_correction with input from import_movies
- Use ctf_estimation with input from motion_correction
- Use micrograph_selection with input from ctf_estimation
- Use reason_about_workflow to analyze current state

Execute this workflow step by step using the ReAct framework, ensuring each job completes successfully before proceeding.
"""
    
    def _parse_workflow_result(self, result: str) -> None:
        """Parse the ReAct workflow result and create PreprocessingResult objects."""
        # This is a simplified parser - in a real implementation, you'd parse the agent's output
        # to extract specific results for each step
        
        # Get the RELION directory from the agent
        relion_dir = self.agent.relion_tools.relion_dir
        
        # Create results based on the agent's workflow state
        for step_name, step_state in self.agent.workflow_state.items():
            try:
                # Try to convert step_name to PreprocessingStep enum
                try:
                    step_enum = PreprocessingStep(step_name)
                except ValueError:
                    # If step_name doesn't match any enum value, skip it
                    continue
                
                # Convert relative job_dir to full path if it's a relative path
                job_dir = step_state.get("job_dir")
                if job_dir and not os.path.isabs(job_dir):
                    try:
                        job_dir = os.path.join(relion_dir, job_dir)
                    except Exception as e:
                        print(f"Warning: Could not convert job_dir to absolute path: {e}")
                
                output_file = step_state.get("output_file")
                
                result_obj = PreprocessingResult(
                    step=step_enum,
                    success=step_state.get("completed", False),
                    job_dir=job_dir,
                    output_file=output_file,
                    message=f"{step_name} {'completed' if step_state.get('completed', False) else 'pending'}"
                )
                
                self.results.append(result_obj)
                
            except Exception as e:
                print(f"Warning: Failed to parse workflow result for step {step_name}: {e}")
                continue
    
    def get_workflow_summary(self) -> str:
        """Get a summary of the workflow execution."""
        if not self.results:
            return "No workflow results available."
        
        summary = "RELION Preprocessing Workflow Summary:\n\n"
        
        for result in self.results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            summary += f"{result.step.value.replace('_', ' ').title()}: {status}\n"
            
            if result.job_dir:
                summary += f"  Job directory: {result.job_dir}\n"
            if result.output_file:
                summary += f"  Output file: {result.output_file}\n"
            if result.message:
                summary += f"  Message: {result.message}\n"
            if result.error:
                summary += f"  Error: {result.error}\n"
            
            summary += "\n"
        
        return summary
    
    def get_final_outputs(self) -> Dict[str, str]:
        """Get the final output files from the workflow."""
        outputs = {}
        
        for result in self.results:
            if result.success and result.output_file:
                outputs[result.step.value] = result.output_file
        
        return outputs
