"""ReAct-based particle picking agent for CryoEM data processing."""

from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..base_react_agent import BaseReActAgent
from .picking_tools import PickingTools
from ...tools.cryosparc_tools import CryoSPARCTools
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
    
    def _create_tools(self) -> List[Tool]:
        """Create particle picking-specific tools."""
        return [
            PickingTools.create_blob_picker_tool(self),
            PickingTools.create_get_job_status_tool(self),
            PickingTools.create_wait_for_job_tool(self),
            PickingTools.create_reason_about_workflow_tool(self)
        ]
    
    def _get_react_system_prompt(self) -> str:
        """Get the particle picking-specific ReAct system prompt."""
        return f"""You are a CryoEM particle picking assistant using the ReAct (Reasoning + Acting) framework. 
You specialize in detecting and extracting particles from preprocessed micrographs using blob detection and other picking methods.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Particle Picking Workflow:
1. **Blob Picker GPU**: Detect particles using GPU-accelerated Gaussian blob detection
   - Required: micrographs_job_uid (from micrograph selection), particle_diameter
   - Optional: diameter_max (defaults to 2.0 * particle_diameter), project_uid, workspace_uid
   - The blob picker uses Gaussian blob detection to identify circular features
   - Particle diameter should be specified in Angstroms (this is the minimum diameter)
   - diameter_max specifies the maximum diameter to search for

## ReAct Process:
For each step, you MUST follow this pattern:

**Thought**: [Your reasoning about what needs to be done, why, and how]
**Action**: [The specific tool to use with exact parameters]
**Observation**: [What happened as a result of the action]

## CRITICAL: Job Monitoring and Waiting
- After starting the blob picker job, you MUST wait for it to complete using wait_for_job
- Do NOT proceed until the job is completed
- If a job fails, report the error and stop the workflow

## Tool Usage Guidelines:
- blob_picker: Detect particles from micrographs using GPU-accelerated blob detection
  * Requires: micrographs_job_uid (from micrograph selection or CTF estimation)
  * Requires: particle_diameter (minimum diameter in Angstroms)
  * Optional: diameter_max (maximum diameter, default: 2.0 * particle_diameter)
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
2. Input job must have "exposures" output containing curated micrographs
3. Always verify input job completion before starting particle picking
4. Wait for picker job to complete before analyzing results

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}

## Example Workflow:
1. Reason about the particle size and separation requirements
2. Execute blob_picker with appropriate parameters
3. Wait for the job to complete
4. Observe the results and report picking statistics

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
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about particle picking workflow state."""
        try:
            reasoning = f"""
🤔 **Particle Picking Workflow Analysis**:

**Current State**: {input_str}

**Workflow Overview**:
1. Blob picker GPU detects particles using GPU-accelerated Gaussian blob detection
2. Requires completed micrograph selection or CTF estimation job
3. Key parameters: particle_diameter (min) and diameter_max (max)

**Parameter Considerations**:
- **Particle Diameter (Min)**: Minimum expected size of the protein complex
  * Too small: May pick noise or fragment particles
  * Too large: May miss smaller particles
  * Should be determined from prior knowledge or initial screening
  * This is the lower bound of the search range

- **Diameter Max**: Maximum expected particle size
  * Default (2.0 × particle_diameter) works for most cases
  * Increase if particles have significant size variation
  * Defines the upper bound of the search range
  * Blob picker searches for particles between min and max diameters

**Next Steps Analysis**:
- If no picking job running: Start blob_picker with appropriate diameter range
- If blob picker is running: Wait for completion and analyze results
- If picking failed: Check parameters and input micrograph quality

**Recommended Actions**:
- Always verify input job UID is valid and completed
- Start with conservative particle diameter range
- Wait for job completion before proceeding
- Monitor job status for errors or warnings
- Review picked particles to validate diameter range
"""
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": reasoning})
            return reasoning
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error in workflow reasoning: {str(e)}"

