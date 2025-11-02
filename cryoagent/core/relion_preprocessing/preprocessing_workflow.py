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
    
    def _get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration from JSON file (same method as agent)."""
        # Use the agent's method to get workflow config from JSON
        return self.agent._get_workflow_config()
    
    def _create_workflow_input(self) -> str:
        """Create the workflow input for the ReAct agent."""
        # Get microscope config from the agent
        microscope_config = getattr(self.agent, 'microscope_config', {})
        
        # Get workflow config from JSON file (not from config.workflow object)
        workflow_config = self._get_workflow_config()
        import_movies_config = workflow_config.get("import_movies", {})
        motion_correction_config = workflow_config.get("motion_correction", {})
        ctf_estimation_config = workflow_config.get("ctf_estimation", {})
        micrograph_selection_config = workflow_config.get("micrograph_selection", {})
        
        return f"""
Execute the complete RELION preprocessing workflow using the ReAct framework. Follow the Thought → Action → Observation pattern for each step.

## Workflow Steps (in order):

1. **Import Movies**: Import movie files using RELION tools
   - Movies path: {microscope_config.get('movies_path', 'Micrographs/*.tif')}
   - Pixel size: {microscope_config.get('pixel_size', 'N/A')} Å
   - Voltage: {microscope_config.get('voltage', 'N/A')} kV
   - CS: {microscope_config.get('cs_mm', 'N/A')} mm
   - Q0: {microscope_config.get('q0', 'N/A')}
   - Optics group: {import_movies_config.get('optics_group_name', 'opticsGroup1')}

2. **Motion Correction**: Correct motion using RELION tools
   - IMPORTANT: All motion correction parameters are automatically loaded from preprocessing_config.json
   - Config values: Use MotionCor2={motion_correction_config.get('use_motioncor2', False)}, bin_factor={motion_correction_config.get('bin_factor', 1)}, bfactor={motion_correction_config.get('bfactor', 150)}, dose_per_frame={motion_correction_config.get('dose_per_frame', 1.39)}, dose_weighting={motion_correction_config.get('dose_weighting', True)}
   - MotionCor2 executable: {motion_correction_config.get('motioncor2_exe', '../../tools/MotionCor2_1.6.4_Cuda118_Mar312023')} (if MotionCor2 is enabled)

3. **CTF Estimation**: Estimate CTF parameters using RELION tools
   - Box size: {ctf_estimation_config.get('box_size', 512)}
   - Resolution range: {ctf_estimation_config.get('res_min', 30)} - {ctf_estimation_config.get('res_max', 5)} Å
   - Defocus range: {ctf_estimation_config.get('df_min', 5000)} - {ctf_estimation_config.get('df_max', 50000)} Å
   - CTFfind executable: {ctf_estimation_config.get('ctffind_exe', '/home/daoyi/tools/ctffind/ctffind_4_1_14/ctffind')}
   - Fast search: {ctf_estimation_config.get('fast_search', True)}

4. **Micrograph Selection**: Select high-quality micrographs using RELION tools
   - Select field: {micrograph_selection_config.get('select_field', 'rlnCtfMaxResolution')}
   - Min value: {micrograph_selection_config.get('minval', 1.0)} Å
   - Max value: {micrograph_selection_config.get('maxval', 5.0)} Å

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
- Motion Correction: MotionCorr/job002/corrected_micrographs.star
- CTF Estimation: CtfFind/job003/micrographs_ctf.star
- Selection: Select/job004/selected_micrographs.star

## Tool Usage:
- Use validate_inputs to check movie files before import
- Use import_movies to start the import process
- Use wait_for_job to monitor job completion
- Use motion_correction with ONLY the required input_star parameter (e.g., {{"input_star": "Import/job001/movies.star"}}). DO NOT pass use_motioncor2, bin_factor, bfactor, dose_per_frame, or other optional parameters - they are automatically loaded from preprocessing_config.json
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
