"""ReAct-based particle picking agent for CryoEM data processing."""

import logging
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..base_react_agent import BaseReActAgent
from .picking_tools import PickingTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...tools.cryosparc_parser_tools import CryoSPARCPickingParser, WorkflowContext
from ...config.config_loader import CryoAgentConfig


class PickingAgent(BaseReActAgent):
    """ReAct-based agent for CryoEM particle picking operations."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the particle picking agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        super().__init__(cryosparc_tools, config, llm)
        # Initialize logger for this agent
        self.logger = logging.getLogger("PickingAgent")
    
    def _create_tools(self) -> List[Tool]:
        """Create particle picking-specific tools."""
        return [
            PickingTools.create_blob_picker_tool(self),
            PickingTools.create_extract_particles_tool(self),
            PickingTools.create_class_2d_tool(self),
            PickingTools.create_select_2d_classes_tool(self),
            PickingTools.create_template_picker_tool(self),
            PickingTools.create_get_job_status_tool(self),
            PickingTools.create_wait_for_job_tool(self),
            PickingTools.create_get_job_log_tool(self),
            PickingTools.create_reason_about_workflow_tool(self)
        ]
    
    def _get_react_system_prompt(self) -> str:
        """Get the particle picking-specific ReAct system prompt."""
        return f"""You are a CryoEM particle picking assistant using the ReAct (Reasoning + Acting) framework. 
You specialize in detecting, extracting, and classifying particles from preprocessed micrographs.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Particle Picking Workflow (3 steps):
1. **Blob Picker GPU**: Detect particles using GPU-accelerated Gaussian blob detection
   - Required: micrographs_job_uid (from micrograph selection), particle_diameter
   - Optional: diameter_max (defaults to 2.0 * particle_diameter), project_uid, workspace_uid
   - The blob picker uses Gaussian blob detection to identify circular features
   - Particle diameter should be specified in Angstroms (this is the minimum diameter)
   - diameter_max specifies the maximum diameter to search for

2. **Particle Extraction**: Extract particles from micrographs based on picked coordinates
   - Required: particles_job_uid (from blob picker), box_size_angstroms
   - Box size determines the size of the extracted particle images in Angstroms
   - Typically set to ~1.5-2x the particle diameter to include sufficient context
   - Creates particle stacks for downstream processing

3. **2D Classification**: Group extracted particles into classes
   - Required: particles_job_uid (from extraction)
   - Optional: num_classes (default: 20)
   - Groups particles by similarity to identify different views and remove junk
   - Helps assess particle quality and data heterogeneity

## ReAct Process:
For each step, you MUST follow this pattern:

**Thought**: [Your reasoning about what needs to be done, why, and how]
**Action**: [The specific tool to use with exact parameters]
**Observation**: [What happened as a result of the action]

## CRITICAL: Job Monitoring and Waiting
- After starting ANY job (blob picker, extraction, or 2D classification), you MUST wait for it to complete
- Use wait_for_job with the job UID to wait for completion
- Do NOT proceed to the next step until the current job is completed
- If a job fails, report the error and stop the workflow

## Tool Usage Guidelines:
- blob_picker: Detect particles from micrographs using GPU-accelerated blob detection
  * Requires: micrographs_job_uid (from micrograph selection or CTF estimation)
  * Requires: particle_diameter (minimum diameter in Angstroms)
  * Optional: diameter_max (maximum diameter, default: 2.0 * particle_diameter)
  * Start the job, then wait for completion
  
- extract_particles: Extract particles based on picking coordinates
  * Requires: particles_job_uid (from blob picker job)
  * Requires: box_size_angstroms (extraction box size in Angstroms)
  * Box size should be ~1.5-2x particle diameter
  * Start the job, then wait for completion
  
- class_2d: Perform 2D classification on extracted particles
  * Requires: particles_job_uid (from extraction job)
  * Optional: num_classes (number of 2D classes, default: 20)
  * Start the job, then wait for completion
  
- get_job_status: Check status of a specific job (use job UID only, e.g., "J85")
- wait_for_job: Wait for job completion (use job UID only, e.g., "J85")
- reason_about_workflow: Analyze current picking state and parameters

## Job UID Format:
- Job UIDs are strings like "J85", "J86", etc.
- When calling get_job_status or wait_for_job, pass ONLY the job UID
- Do NOT use JSON format or complex parameters for these tools

## Particle Picking Parameters:
- **Particle Diameter**: The minimum expected diameter of particles in Angstroms
  * This is the most critical parameter for blob detection
  * Should match the actual size of your protein complex
  * Typical range: 50-500 Å depending on the particle
  * The blob picker will search for particles >= this diameter
  
- **Diameter Max**: Maximum diameter to search for
  * Default: 2.0 × particle_diameter
  * Defines the upper bound of the particle size range
  * Useful for detecting particles with size variation
  * Set to a larger value if particles vary significantly in size

## Workflow Dependencies:
1. Blob picker requires completed micrograph selection or CTF estimation job
2. Particle extraction requires completed blob picker job
3. 2D classification requires completed extraction job
4. Each step must complete successfully before the next can begin
5. Always verify job completion before proceeding to the next step

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}

## Example Workflow:
1. Reason about the particle size and extraction parameters
2. Execute blob_picker with appropriate diameter range
3. Wait for blob picker to complete
4. Execute extract_particles with appropriate box size
5. Wait for extraction to complete
6. Execute class_2d with desired number of classes
7. Wait for classification to complete
8. Observe the final results and report statistics

Remember: Always follow the Thought → Action → Observation pattern and WAIT for each job to complete!"""
    
    # Tool implementation methods
    def _blob_picker_tool(self, input_str: str) -> str:
        """Tool wrapper for blob picker particle detection."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get particle diameter from params or config
            particle_diameter = params.get("particle_diameter")
            if not particle_diameter:
                # Try to get from config if available
                particle_diameter = getattr(self.config.workflow, "particle_diameter", None)
            
            if not particle_diameter:
                return "❌ Error: particle_diameter parameter is required for blob picker"
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "micrographs_job_uid": params.get("micrographs_job_uid"),
                "particle_diameter": float(particle_diameter),
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }
            
            # Add optional diameter_max if provided (default is 2x diameter in blob_picker method)
            if params.get("diameter_max"):
                used_params["diameter_max"] = float(params.get("diameter_max"))

            result = self.cryosparc_tools.blob_picker(**used_params)
            self._record_tool_execution("blob_picker", used_params, result=result)
            
            diameter_range = f"{particle_diameter}-{used_params.get('diameter_max', particle_diameter*2.0)}"
            return f"✅ Successfully queued blob picker GPU job: {result['job_uid']} (diameter range: {diameter_range} Å)"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("blob_picker", context, error=str(e))
            return f"❌ Error starting blob picker: {str(e)}"
    
    def _extract_particles_tool(self, input_str: str) -> str:
        """Tool wrapper for particle extraction."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get box size from params or config
            box_size_pix = params.get("box_size_pix")
            if not box_size_pix:
                box_size_pix = getattr(self.config.workflow, "box_size_pix", None)
            
            if not box_size_pix:
                return "❌ Error: box_size_pix parameter is required for particle extraction"
            
            # Get micrographs_job_uid
            micrographs_job_uid = params.get("micrographs_job_uid")
            if not micrographs_job_uid:
                return "❌ Error: micrographs_job_uid parameter is required for particle extraction"
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": params.get("particles_job_uid"),
                "micrographs_job_uid": micrographs_job_uid,
                "box_size_pix": int(box_size_pix),
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            result = self.cryosparc_tools.extract_particles(**used_params)
            self._record_tool_execution("extract_particles", used_params, result=result)
            
            return f"✅ Successfully queued particle extraction job: {result['job_uid']} (box size: {box_size_pix} pixels)"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("extract_particles", context, error=str(e))
            return f"❌ Error starting particle extraction: {str(e)}"
    
    def _class_2d_tool(self, input_str: str) -> str:
        """Tool wrapper for 2D classification."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get num_classes from params or config (default 20)
            num_classes = params.get("num_classes")
            if not num_classes:
                num_classes = getattr(self.config.workflow, "num_classes", 20)
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": params.get("particles_job_uid"),
                "num_classes": int(num_classes),
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout * 2)),  # 2D classification takes longer
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            result = self.cryosparc_tools.class_2d(**used_params)
            self._record_tool_execution("class_2d", used_params, result=result)
            
            return f"✅ Successfully queued 2D classification job: {result['job_uid']} ({num_classes} classes)"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("class_2d", context, error=str(e))
            return f"❌ Error starting 2D classification: {str(e)}"
    
    def _select_2d_classes_tool(self, input_str: str) -> str:
        """Tool wrapper for 2D class selection."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Determine selection strategy and parameters
            selection_mode = (params.get("selection_mode") or params.get("selection_strategy") or "top_n").lower()

            top_n_classes = params.get("top_n_classes")
            if top_n_classes is None and selection_mode != "cryosift":
                top_n_classes = getattr(self.config.workflow, "top_n_classes", 5)

            cryosift_threshold = params.get("cryosift_threshold")
            cryosift_env = params.get("cryosift_env")
            cryosift_weights_path = params.get("cryosift_weights_path")
            cryosift_output_dir = params.get("cryosift_output_dir")
            cryosift_output_subdir = params.get("cryosift_output_subdir")
            cryosift_python = params.get("cryosift_python_executable")
            cryosift_fallback = params.get("cryosift_fallback_strategy")
            
            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "class_2d_job_uid": params.get("class_2d_job_uid"),
                "selection_mode": selection_mode,
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", 300)),
                "check_interval": int(params.get("check_interval", 10))
            }
            if top_n_classes is not None:
                try:
                    used_params["top_n_classes"] = int(top_n_classes)
                except (TypeError, ValueError):
                    used_params["top_n_classes"] = None

            cryosift_options: Dict[str, Any] = {}
            if cryosift_threshold is not None:
                try:
                    cryosift_options["threshold"] = float(cryosift_threshold)
                except (TypeError, ValueError):
                    pass
            if cryosift_env:
                cryosift_options["conda_env"] = cryosift_env
            if cryosift_weights_path:
                cryosift_options["weights_path"] = cryosift_weights_path
            if cryosift_output_dir:
                cryosift_options["output_dir"] = cryosift_output_dir
            if cryosift_output_subdir:
                cryosift_options["output_subdir"] = cryosift_output_subdir
            if cryosift_python:
                cryosift_options["python_executable"] = cryosift_python
            if cryosift_fallback:
                cryosift_options["fallback_strategy"] = cryosift_fallback

            if cryosift_options:
                used_params["cryosift_options"] = cryosift_options

            result = self.cryosparc_tools.select_2d_classes(**used_params)
            self._record_tool_execution("select_2d_classes", used_params, result=result)
            
            strategy = result.get("selection_metadata", {}).get("selection_mode", selection_mode)
            detail = ""
            if strategy == "top_n" and used_params.get("top_n_classes"):
                detail = f"selecting top {used_params['top_n_classes']} classes"
            elif strategy == "cryosift":
                selected = result.get("selected_template_indices") or []
                detail = f"CryoSift-selected classes {selected}" if selected else "CryoSift evaluation completed"

            detail_text = f" ({detail})" if detail else ""
            return f"✅ Successfully queued 2D class selection job: {result['job_uid']}{detail_text}"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("select_2d_classes", context, error=str(e))
            return f"❌ Error starting 2D class selection: {str(e)}"
    
    def _template_picker_tool(self, input_str: str) -> str:
        """Tool wrapper for template-based picking."""
        params: Dict[str, Any] = {}
        used_params: Dict[str, Any] = {}
        try:
            params = self._parse_tool_input(input_str)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get micrographs_job_uid
            micrographs_job_uid = params.get("micrographs_job_uid")
            if not micrographs_job_uid:
                return "❌ Error: micrographs_job_uid parameter is required for template picker"
            
            # Resolve template picker parameters (fall back to recorded blob picker values if needed)
            lowpass_resolution = params.get("lowpass_resolution") or getattr(self.config.workflow, "lowpass_resolution", 20.0)

            particle_diameter = params.get("particle_diameter") or getattr(self.config.workflow, "particle_diameter", None)
            angular_spacing_deg = params.get("angular_spacing_deg") or params.get("angle_search_range")
            blob_picker_job_uid = params.get("blob_picker_job_uid")

            if not particle_diameter or not blob_picker_job_uid:
                for entry in reversed(self.tool_execution_log):
                    if entry.get("tool") == "blob_picker" and isinstance(entry.get("result"), dict):
                        blob_picker_job_uid = blob_picker_job_uid or entry["result"].get("job_uid")
                        blob_params = entry["result"].get("params", {})
                        if not particle_diameter:
                            particle_diameter = blob_params.get("diameter") or blob_params.get("particle_diameter")
                        if blob_picker_job_uid and particle_diameter:
                            break

            used_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "micrographs_job_uid": micrographs_job_uid,
                "template_job_uid": params.get("template_job_uid"),
                "lowpass_resolution": float(lowpass_resolution),
                "particle_diameter": float(particle_diameter) if particle_diameter else None,
                "lowpass_micrograph": float(params.get("lowpass_micrograph")) if params.get("lowpass_micrograph") else None,
                "angular_spacing_deg": float(angular_spacing_deg) if angular_spacing_deg else None,
                "min_distance": float(params.get("min_distance")) if params.get("min_distance") else None,
                "use_ctf": params.get("use_ctf"),
                "blob_picker_job_uid": blob_picker_job_uid,
                "wait_for_completion": params.get("wait_for_completion", "false").lower() == "true",
                "timeout": int(params.get("timeout", self.config.job_management.default_timeout)),
                "check_interval": int(params.get("check_interval", self.config.job_management.status_check_interval))
            }

            result = self.cryosparc_tools.template_picker(**used_params)
            self._record_tool_execution("template_picker", used_params, result=result)
            
            return f"✅ Successfully queued template picker job: {result['job_uid']} (lowpass: {lowpass_resolution} Å)"
            
        except Exception as e:
            context = used_params or params or {"raw_input": input_str}
            self._record_tool_execution("template_picker", context, error=str(e))
            return f"❌ Error starting template picker: {str(e)}"
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about particle picking workflow state."""
        try:
            reasoning = f"""
🤔 **Particle Picking Workflow Analysis**:

**Current State**: {input_str}

**Complete Workflow Overview**:
1. **Blob Picker GPU**: Detect particles using GPU-accelerated Gaussian blob detection
2. **Particle Extraction**: Extract particle images based on picking coordinates
3. **2D Classification**: Group particles into classes for quality assessment

**Parameter Considerations**:
- **Particle Diameter (Min)**: Minimum expected size of the protein complex
  * Too small: May pick noise or fragment particles
  * Too large: May miss smaller particles
  * Should be determined from prior knowledge or initial screening
  * This is the lower bound of the search range for blob picker

- **Diameter Max**: Maximum expected particle size for blob picker
  * Default (2.0 × particle_diameter) works for most cases
  * Increase if particles have significant size variation
  * Defines the upper bound of the search range

- **Box Size (Angstroms)**: Size of extracted particle images
  * Typically 1.5-2x the particle diameter
  * Must be large enough to include the entire particle plus context
  * Affects downstream processing (classification, refinement)
  * Larger boxes provide more context but increase computational cost

- **Number of Classes**: Number of 2D classes for classification
  * Default: 20 classes
  * More classes: Better separation of views and junk, but slower
  * Fewer classes: Faster but less detailed classification
  * Typical range: 10-100 depending on dataset size and heterogeneity

**Workflow Dependencies**:
1. Blob picker requires completed micrograph job (CTF or selection)
2. Extraction requires completed blob picker job
3. 2D classification requires completed extraction job
4. Each step must complete before proceeding to the next

**Next Steps Analysis**:
- If no jobs running: Start with blob_picker
- If blob picker running/done: Proceed to extraction
- If extraction running/done: Proceed to 2D classification
- If 2D classification done: Workflow complete, analyze results

**Recommended Actions**:
- Always verify each job UID is valid and completed before using it
- Wait for each job to complete before starting the next
- Monitor job status for errors or warnings
- Use appropriate parameters based on particle characteristics
"""
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": reasoning})
            return reasoning
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error in workflow reasoning: {str(e)}"
    
    def process_workflow_results(self, results: List, context: WorkflowContext) -> Dict[str, Any]:
        """
        Process workflow results and extract stage outputs.
        
        Args:
            results: List of picking workflow results
            context: Workflow context with project/workspace info
            
        Returns:
            Dictionary of stage outputs
        """
        parser = CryoSPARCPickingParser(self.cryosparc_tools, self.logger)
        return parser.process_workflow_results(results, context)
    
    def validate_results(self, stage_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the picking workflow completed successfully.
        
        Args:
            stage_outputs: Dictionary of stage outputs to validate
            
        Returns:
            Dictionary with 'success' boolean and 'error' message if failed
        """
        parser = CryoSPARCPickingParser(self.cryosparc_tools, self.logger)
        return parser.validate_results(stage_outputs)
    
    def save_results(self, stage_outputs: Dict[str, Any], context: WorkflowContext, success: bool = True) -> str:
        """
        Save picking results to a JSON file.
        
        Args:
            stage_outputs: Dictionary of stage outputs
            context: Workflow context
            success: Whether picking was successful
            
        Returns:
            Path to the saved JSON file
        """
        parser = CryoSPARCPickingParser(self.cryosparc_tools, self.logger)
        return parser.save_results(stage_outputs, context, success)
