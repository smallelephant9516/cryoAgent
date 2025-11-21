"""ReAct-based RELION reconstruction workflow orchestrator."""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .reconstruction_agent import ReconstructionAgent
from ...config.config_loader import CryoAgentConfig


class ReconstructionStep(Enum):
    """Enumeration of reconstruction workflow steps."""
    AB_INITIO_RECONSTRUCTION = "ab_initio_reconstruction"
    PARTICLE_REEXTRACTION = "particle_reextraction"
    REFINEMENT_3D = "refinement_3d"


@dataclass
class ReconstructionResult:
    """Result of a reconstruction workflow execution."""
    step: ReconstructionStep
    success: bool
    job_dir: Optional[str] = None
    output_file: Optional[str] = None
    initial_model: Optional[str] = None
    refined_map: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None


class ReconstructionWorkflow:
    """ReAct-based orchestrator for RELION reconstruction workflows."""
    
    def __init__(self, agent: ReconstructionAgent, config: CryoAgentConfig):
        """
        Initialize the reconstruction workflow.
        
        Args:
            agent: Reconstruction agent instance
            config: Complete configuration object
        """
        self.agent = agent
        self.config = config
        self.results: List[ReconstructionResult] = []
        self.current_job_dirs: Dict[ReconstructionStep, str] = {}
        self.workflow_state: Dict[str, Any] = {}
    
    def run(self, final_star_file: Optional[str] = None, conversation_id: Optional[str] = None) -> List[ReconstructionResult]:
        """
        Run the complete reconstruction workflow using ReAct approach.
        
        Args:
            final_star_file: Path to particles STAR file from particle picking stage
            conversation_id: Optional conversation identifier for memory control
            
        Returns:
            List of reconstruction results for each step
        """
        self.results = []
        self.workflow_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "active_jobs": {},
            "workflow_status": "starting"
        }
        
        if final_star_file:
            self.agent.set_initial_particles_star(final_star_file)

        workflow_input = self._create_workflow_input(final_star_file)
        
        try:
            result = self.agent.run_react_workflow(workflow_input, conversation_id)
            self._parse_workflow_result(result)
            
        except Exception as e:
            error_result = ReconstructionResult(
                step=ReconstructionStep.AB_INITIO_RECONSTRUCTION,
                success=False,
                error=f"Reconstruction workflow execution failed: {str(e)}",
                message="ReAct workflow failed to execute"
            )
            self.results.append(error_result)
        
        return self.results
    
    def _get_workflow_config(self) -> Dict[str, Any]:
        """Get workflow configuration from JSON file (same method as agent)."""
        # Use the agent's method to get workflow config from JSON
        return self.agent._get_workflow_config()
    
    def _create_workflow_input(self, final_star_file: Optional[str] = None) -> str:
        """Create the workflow input for the ReAct agent."""
        # Get workflow config from JSON file
        workflow_config = self._get_workflow_config()
        ab_initio_config = workflow_config.get("ab_initio_reconstruction", {})
        reextraction_config = workflow_config.get("particle_reextraction", {})
        refinement_config = workflow_config.get("refinement_3d", {})
        validation_config = workflow_config.get("validation", {})
        # Check both config value AND whether CryoSPARC tools are actually initialized
        cryosparc_fsc_config_enabled = validation_config.get("cryosparc_fsc", False)
        cryosparc_tools_available = self.agent.cryosparc_tools is not None
        cryosparc_fsc_enabled = cryosparc_fsc_config_enabled and cryosparc_tools_available
        
        # Build the input_star parameter
        input_star_info = ""
        if final_star_file:
            input_star_info = f"   - Input particles STAR file: {final_star_file}\n"
        
        return f"""
Execute the complete RELION reconstruction workflow using the ReAct framework. Follow the Thought → Action → Observation pattern for each step.

## Workflow Steps (in order):

1. **Ab Initio Reconstruction**: Create initial 3D model from particles without a reference
{input_star_info}   - Required parameters: input_star (particles STAR file), particle_diameter, sym
   - Iterations: {ab_initio_config.get('iter', 200)}
   - Number of classes (K): {ab_initio_config.get('K', 1)}
   - Symmetry: {ab_initio_config.get('sym', 'C1')}
   - Particle diameter: {ab_initio_config.get('particle_diameter', 200.0)} Å
   - Healpix order: {ab_initio_config.get('healpix_order', 1)}
   - Offset range: {ab_initio_config.get('offset_range', 6.0)} pixels
   - Tau2 fudge: {ab_initio_config.get('tau2_fudge', 4.0)}
   - GPU: {ab_initio_config.get('gpu', '0,1')}
   - This step performs de novo 3D refinement, then aligns symmetry

2. **Particle Re-extraction**: Re-extract particles from micrographs with original pixel size without scaling
   - Required parameters: reextract_data_star (auto-detected from ab initio), micrographs_star (auto-detected from preprocessing), extract_size
   - Extract size: {reextraction_config.get('extract_size', 440)} pixels (REQUIRED)
   - Norm: {reextraction_config.get('norm', True)}
   - Invert contrast: {reextraction_config.get('invert_contrast', True)}
   - Float16: {reextraction_config.get('float16', True)}
   - This step re-extracts particles at full resolution for better refinement
   - IMPORTANT: Both reextract_data_star and micrographs_star will be auto-detected if available:
     * reextract_data_star: Auto-detected from ab initio reconstruction output (InitialModel/jobXXX/run_it*_data.star)
     * micrographs_star: Auto-detected from preprocessing results (outputs/preprocessing_results_relion_*.json) or Select/jobXXX/micrographs.star
   - If auto-detection fails, you can provide micrographs_star manually (e.g., Select/jobXXX/micrographs.star)

3. **3D Refinement**: Refine the 3D structure using the initial model and re-extracted particles
   - Required parameters: input_star (particles STAR file), ref_mrc (initial model from ab initio), particle_diameter, sym
   - Symmetry: {refinement_config.get('sym', 'D7')}
   - Particle diameter: {refinement_config.get('particle_diameter', 260.0)} Å
   - Healpix order: {refinement_config.get('healpix_order', 2)}
   - Auto local healpix order: {refinement_config.get('auto_local_healpix_order', 4)}
   - Offset range: {refinement_config.get('offset_range', 5.0)} pixels
   - Initial high resolution: {refinement_config.get('ini_high', 60.0)} Å
   - Low resolution for joining halves: {refinement_config.get('low_resol_join_halves', 40.0)} Å
   - GPU: {refinement_config.get('gpu', '')} (empty string means use default GPU)
   - This step performs auto-refinement with split random halves validation
   - Outputs: run_class001.mrc (refined map), run_half1_class001_unfil.mrc (half map A), run_half2_class001_unfil.mrc (half map B)

{f"4. **CryoSPARC FSC Validation** (OPTIONAL): Import half maps and compute FSC using CryoSPARC validation tools" if cryosparc_fsc_enabled else ""}
{f"   - This step is available because CryoSPARC tools are initialized and ready" if cryosparc_fsc_enabled else ""}
{f"   - NOTE: CryoSPARC FSC validation is configured but tools are not available. Check CryoSPARC connection." if cryosparc_fsc_config_enabled and not cryosparc_fsc_enabled else ""}
{f"   - Required parameters: half_map_a_path (run_half1_class001_unfil.mrc), half_map_b_path (run_half2_class001_unfil.mrc)" if cryosparc_fsc_enabled else ""}
{f"   - Optional parameters: project_uid, workspace_uid (auto-detected from master_config.json workflow section if not provided)" if cryosparc_fsc_enabled else ""}
{f"   - Step 4a: Use import_volumes to import both half maps into CryoSPARC (auto-detected from refinement output)" if cryosparc_fsc_enabled else ""}
{f"   - Step 4b: Use compute_fsc_validation to calculate FSC between the two half maps" if cryosparc_fsc_enabled else ""}
{f"   - Returns FSC resolution and validation results" if cryosparc_fsc_enabled else ""}

## ReAct Process Requirements:
- **Thought**: Analyze what needs to be done and why
- **Action**: Execute the appropriate tool with correct parameters
- **Observation**: Analyze results and determine next steps

## Critical Workflow Rules:
- Execute steps in order: Ab Initio → Particle Re-extraction → 3D Refinement{f" → CryoSPARC FSC Validation" if cryosparc_fsc_enabled else ""}
- Wait for ab initio reconstruction to complete before starting re-extraction
- Wait for particle re-extraction to complete before starting refinement
- Wait for 3D refinement to complete before starting validation (if enabled)
- Particle re-extraction is REQUIRED and must be performed before 3D refinement
{f"- CryoSPARC FSC validation: {'AVAILABLE - CryoSPARC tools are initialized and ready to use' if cryosparc_fsc_enabled else 'NOT AVAILABLE - CryoSPARC is enabled in config but tools failed to initialize. Check CryoSPARC connection settings.'}" if cryosparc_fsc_config_enabled else ""}
- Validate inputs before starting each step using validate_inputs
- Check job status and logs if any step fails
- Use wait_for_job to monitor job completion (these jobs can take hours)
- Use reason_about_workflow to analyze current state
- Ab initio reconstruction will automatically use the initial_model from ab_initio_reconstruction if ref_mrc is not provided

## Expected Outputs:
- Ab Initio: InitialModel/jobXXX/initial_model.mrc
- Particle Re-extraction: ReExtract/jobXXX/particles.star (re-extracted particles with original pixel size)
- 3D Refinement: Refine3D/jobXXX/run_class001.mrc (refined map), run_half1_class001_unfil.mrc (half map A), run_half2_class001_unfil.mrc (half map B)
{f"- CryoSPARC FSC Validation: FSC resolution and validation results (if enabled)" if cryosparc_fsc_enabled else ""}

## Tool Usage:
- Use validate_inputs to check particles STAR file before ab initio
- Use ab_initio_reconstruction with input_star{f'="{final_star_file}"' if final_star_file else ''}, particle_diameter, and sym parameters
- Use wait_for_job to monitor ab initio completion (this takes a long time)
- Use particle_reextraction with reextract_data_star (auto-detected from ab initio), micrographs_star (from Select job), extract_size={reextraction_config.get('extract_size', 440)}, and other parameters
- Use wait_for_job to monitor re-extraction completion
- Use refinement_3d with input_star (use re-extracted particles from particle_reextraction), ref_mrc (from ab initio), particle_diameter, and sym parameters
- Use wait_for_job to monitor refinement completion
{f"- If CryoSPARC FSC validation is enabled, after refinement completes:" if cryosparc_fsc_enabled else ""}
{f"  * Use import_volumes with half_map_a_path and half_map_b_path (auto-detected from Refine3D output). project_uid and workspace_uid are optional (auto-detected from config)" if cryosparc_fsc_enabled else ""}
{f"  * Use compute_fsc_validation with volume_a_job_uid and volume_b_job_uid (auto-detected from import_volumes). project_uid and workspace_uid are optional (auto-detected from config)" if cryosparc_fsc_enabled else ""}
- Use reason_about_workflow to analyze current state

## Important Notes:
- Ab initio and refinement are computationally intensive and can take hours to complete
- Re-extraction is typically faster but still requires waiting for completion
- Particle re-extraction is REQUIRED before refinement - do not skip this step
- Use backend execution (use_backend=true) for long-running jobs
- Monitor jobs using check_job_status and wait_for_job tools
- The reextract_data_star from ab_initio_reconstruction will be automatically detected if available
- The initial_model from ab_initio_reconstruction will be automatically used as ref_mrc if not explicitly provided
- The re-extracted particles from particle_reextraction will be automatically used as input_star for refinement if available

Execute this workflow step by step using the ReAct framework, ensuring each job completes successfully before proceeding.
"""
    
    def _parse_workflow_result(self, result: str) -> None:
        """Parse the ReAct workflow result and create ReconstructionResult objects."""
        # Get the RELION directory from the agent
        relion_dir = self.agent.relion_tools.relion_dir
        
        # Create results based on the agent's workflow state
        for step_name, step_state in self.agent.workflow_state.items():
            try:
                # Try to convert step_name to ReconstructionStep enum
                try:
                    step_enum = ReconstructionStep(step_name)
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
                initial_model = step_state.get("initial_model")
                refined_map = step_state.get("refined_map")
                
                result_obj = ReconstructionResult(
                    step=step_enum,
                    success=step_state.get("completed", False),
                    job_dir=job_dir,
                    output_file=output_file,
                    initial_model=initial_model,
                    refined_map=refined_map,
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
        
        summary = "RELION Reconstruction Workflow Summary:\n\n"
        
        for result in self.results:
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            summary += f"{result.step.value.replace('_', ' ').title()}: {status}\n"
            
            if result.job_dir:
                summary += f"  Job directory: {result.job_dir}\n"
            if result.output_file:
                summary += f"  Output file: {result.output_file}\n"
            if result.initial_model:
                summary += f"  Initial model: {result.initial_model}\n"
            if result.refined_map:
                summary += f"  Refined map: {result.refined_map}\n"
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
            if result.success:
                if result.initial_model:
                    outputs["initial_model"] = result.initial_model
                if result.refined_map:
                    outputs["refined_map"] = result.refined_map
                if result.output_file:
                    outputs[result.step.value] = result.output_file
        
        return outputs

