"""ReAct-based RELION particle picking workflow orchestrator."""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .picking_agent import PickingAgent
from ...config.config_loader import CryoAgentConfig


class PickingStep(Enum):
    """Enumeration of particle picking workflow steps."""
    # Round 1
    BLOB_PICKER = "blob_picker"
    PARTICLE_EXTRACTION = "particle_extraction"
    CLASSIFICATION_2D = "classification_2d"
    AUTO_2D_SELECTION = "auto_2d_selection"
    # Round 2
    TEMPLATE_PICKER = "template_picker"
    PARTICLE_EXTRACTION_2 = "particle_extraction_2"
    CLASSIFICATION_2D_2 = "classification_2d_2"
    AUTO_2D_SELECTION_2 = "auto_2d_selection_2"


@dataclass
class PickingResult:
    """Result of a particle picking workflow execution."""
    step: PickingStep
    success: bool
    job_dir: Optional[str] = None
    output_file: Optional[str] = None
    message: str = ""
    error: Optional[str] = None
    reasoning: Optional[str] = None


class PickingWorkflow:
    """ReAct-based orchestrator for RELION particle picking workflows."""
    
    def __init__(self, agent: PickingAgent, config: CryoAgentConfig):
        """
        Initialize the particle picking workflow.
        
        Args:
            agent: Picking agent instance
            config: Complete configuration object
        """
        self.agent = agent
        self.config = config
        self.results: List[PickingResult] = []
        self.current_job_dirs: Dict[PickingStep, str] = {}
        self.workflow_state: Dict[str, Any] = {}
        self.selected_micrographs_star: Optional[str] = None
        self.original_micrographs_star: Optional[str] = None  # Original micrographs for template picking
    
    def run(self, selected_micrographs_star: str, conversation_id: Optional[str] = None) -> List[PickingResult]:
        """
        Run the complete particle picking workflow using ReAct approach.
        
        Args:
            selected_micrographs_star: Path to the selected micrographs STAR file from preprocessing
            conversation_id: Optional conversation identifier for memory control
            
        Returns:
            List of picking results for each step
        """
        self.selected_micrographs_star = selected_micrographs_star
        self.original_micrographs_star = selected_micrographs_star  # Use same file for template picking
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
            error_result = PickingResult(
                step=PickingStep.BLOB_PICKER,
                success=False,
                error=f"Particle picking workflow execution failed: {str(e)}",
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
        if not self.selected_micrographs_star:
            raise ValueError("selected_micrographs_star must be set before creating workflow input")
        
        # Get workflow config from JSON file
        workflow_config = self._get_workflow_config()
        blob_picker_config = workflow_config.get("blob_picker", {})
        extraction_config = workflow_config.get("particle_extraction", {})
        classification_config = workflow_config.get("classification_2d", {})
        selection_config = workflow_config.get("auto_2d_selection", {})
        template_config = workflow_config.get("template_picker", {})
        
        return f"""
Execute the complete RELION particle picking workflow using the ReAct framework. Follow the Thought → Action → Observation pattern for each step.

This is a TWO-ROUND workflow for improved particle picking quality.

## ROUND 1: Initial Picking

1. **Blob Picker**: Detect particles in micrographs using Laplacian-of-Gaussian (LoG) filtering
   - Input micrographs: {self.selected_micrographs_star}
   - IMPORTANT: All blob picking parameters are automatically loaded from particle_picking_config.json
   - Config values: LoG={blob_picker_config.get('LoG', True)}, LoG_diam_min={blob_picker_config.get('LoG_diam_min', 180.0)} Å, LoG_diam_max={blob_picker_config.get('LoG_diam_max', 360.0)} Å, min_distance={blob_picker_config.get('min_distance', -1)}
   - Do NOT pass optional parameters (LoG, LoG_diam_min, LoG_diam_max, etc.) unless you need to override the config values
   - Wait for job completion before proceeding

2. **Particle Extraction**: Extract particle images from micrographs using coordinate files
   - Input: Use the same micrographs STAR file: {self.selected_micrographs_star}
   - IMPORTANT: All extraction parameters are automatically loaded from particle_picking_config.json
   - Config values: extract_size={extraction_config.get('extract_size', 440)}, norm={extraction_config.get('norm', True)}, bg_radius={extraction_config.get('bg_radius', 48)}, invert_contrast={extraction_config.get('invert_contrast', True)}
   - Do NOT pass optional parameters (extract_size, norm, bg_radius, etc.) unless you need to override the config values
   - coord_list will automatically use blob_picker output directory
   - Wait for job completion before proceeding

3. **2D Classification**: Classify extracted particles into 2D classes
   - Input: Use particles STAR file from particle_extraction step (automatically detected)
   - IMPORTANT: All classification parameters are automatically loaded from particle_picking_config.json
   - Config values: K={classification_config.get('K', 50)}, iter={classification_config.get('iter', 25)}, particle_diameter={classification_config.get('particle_diameter', 260.0)} Å, angpix={classification_config.get('angpix', 1.0)}
   - Do NOT pass optional parameters (K, iter, particle_diameter, etc.) unless you need to override the config values
   - Wait for job completion before proceeding

4. **Auto 2D Selection (Round 1)**: Automatically select the best 2D classes and particles
   - Input: Use optimiser STAR file from classification_2d step (automatically detected)
   - IMPORTANT: All selection parameters are automatically loaded from particle_picking_config.json
   - Config values: min_score={selection_config.get('min_score', 0.05)}, auto_select={selection_config.get('auto_select', True)}
   - Do NOT pass optional parameters unless you need to override the config values
   - Outputs class_averages.star which will be used in round 2
   - Completes immediately (no waiting needed)

## ROUND 2: Template-Based Picking (Improved Quality)

5. **Template Picker**: Use class averages from round 1 as templates for improved picking
   - Input micrographs: {self.original_micrographs_star} (original micrographs)
   - Input ref_star: class_averages.star from round 1 auto_2d_selection (automatically detected)
   - IMPORTANT: All template picking parameters are automatically loaded from particle_picking_config.json
   - Config values: ang={template_config.get('ang', 5.0)}, lowpass={template_config.get('lowpass', 20.0)}, threshold={template_config.get('threshold', 0.05)}
   - Do NOT pass optional parameters unless you need to override the config values
   - Wait for job completion before proceeding

6. **Particle Extraction (Round 2)**: Extract particles using template picker coordinates
   - Input: Use the same micrographs STAR file: {self.selected_micrographs_star}
   - coord_list will automatically use template_picker output directory
   - Same extraction parameters as round 1 (from config)
   - Wait for job completion before proceeding

7. **2D Classification (Round 2)**: Classify extracted particles again
   - Input: Use particles STAR file from particle_extraction_2 step (automatically detected)
   - Same classification parameters as round 1 (from config)
   - Wait for job completion before proceeding

8. **Auto 2D Selection (Round 2)**: Select best classes with HIGHER threshold
   - Input: Use optimiser STAR file from classification_2d_2 step (automatically detected)
   - IMPORTANT: Use min_score=0.5 for better quality selection (higher than round 1's {selection_config.get('min_score', 0.05)})
   - All other selection parameters from config
   - This is the FINAL output

## ReAct Process Requirements:
- **Thought**: Analyze what needs to be done and why
- **Action**: Execute the appropriate tool with correct parameters
- **Observation**: Analyze results and determine next steps

## Critical Workflow Rules:
- Execute ALL 8 steps in order across both rounds
- Round 1: Blob Picker → Particle Extraction → 2D Classification → Auto 2D Selection
- Round 2: Template Picker → Particle Extraction → 2D Classification → Auto 2D Selection (higher threshold)
- Wait for each job to complete before starting the next
- Use wait_for_job to monitor job completion for all picking/extraction/classification steps
- auto_2d_selection runs synchronously and completes immediately - NO waiting needed
- For round 2 auto_2d_selection, use min_score=0.5 (higher than round 1's {selection_config.get('min_score', 0.05)}) for better quality
- Validate inputs before starting each step using validate_inputs if needed
- Check job status and logs if any step fails
- Use reason_about_workflow to analyze current state

## Expected Outputs:
Round 1:
- Blob Picker: AutoPick/job005/*_autopick.star files
- Particle Extraction: Particles/job006/particles.star
- 2D Classification: Class2D/job007/run_optimiser.star
- Auto 2D Selection: Select/job008/particles.star, class_averages.star

Round 2:
- Template Picker: AutoPick/job009/*_autopick.star files
- Particle Extraction: Particles/job010/particles.star
- 2D Classification: Class2D/job011/run_optimiser.star
- Auto 2D Selection: Select/job012/particles.star (FINAL OUTPUT)

## Tool Usage:
- Round 1: Use blob_picker with ONLY input_star. Parameters are auto-loaded from config.
- Round 1: Use particle_extraction with ONLY input_star. coord_list auto-detected from blob_picker.
- Round 1: Use classification_2d with ONLY input_star (auto-detected from extraction).
- Round 1: Use auto_2d_selection with ONLY input_opt (auto-detected from classification).
- Round 2: Use template_picker with input_star and ref_star (auto-detected from round 1).
- Round 2: Use particle_extraction with ONLY input_star. coord_list auto-detected from template_picker.
- Round 2: Use classification_2d with ONLY input_star (auto-detected from extraction_2).
- Round 2: Use auto_2d_selection with input_opt (auto-detected) and min_score=0.5 (higher threshold for final selection).
- Always use wait_for_job to monitor job completion before proceeding.
- Use reason_about_workflow to analyze current state and determine next step.

Execute this complete TWO-ROUND workflow step by step using the ReAct framework, ensuring each job completes successfully before proceeding.
"""
    
    def _parse_workflow_result(self, result: str) -> None:
        """Parse the ReAct workflow result and create PickingResult objects."""
        # Get the RELION directory from the agent
        relion_dir = self.agent.relion_tools.relion_dir
        
        # Create results based on the agent's workflow state
        for step_name, step_state in self.agent.workflow_state.items():
            try:
                # Try to convert step_name to PickingStep enum
                # Handle mapping between workflow_state keys and enum values
                step_name_mapping = {
                    "blob_picker": "blob_picker",
                    "particle_extraction": "particle_extraction",
                    "classification_2d": "classification_2d",
                    "auto_2d_selection": "auto_2d_selection",
                    "template_picker": "template_picker",
                    "particle_extraction_2": "particle_extraction_2",
                    "classification_2d_2": "classification_2d_2",
                    "auto_2d_selection_2": "auto_2d_selection_2"
                }
                
                mapped_step = step_name_mapping.get(step_name, step_name)
                try:
                    step_enum = PickingStep(mapped_step)
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
                
                result_obj = PickingResult(
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
        
        summary = "RELION Particle Picking Workflow Summary:\n\n"
        
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
