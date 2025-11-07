"""ReAct-based 3D reconstruction agent for CryoEM data processing."""

from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel
import json

from ..base_react_agent import BaseReActAgent
from .reconstruction_tools import ReconstructionTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...config.config_loader import CryoAgentConfig


class ReconstructionAgent(BaseReActAgent):
    """ReAct-based agent for CryoEM 3D reconstruction operations."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the 3D reconstruction agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        super().__init__(cryosparc_tools, config, llm)
    
    def _create_tools(self) -> List[Tool]:
        """Create 3D reconstruction-specific tools."""
        return [
            ReconstructionTools.create_ab_initio_tool(self),
            ReconstructionTools.create_homogeneous_reconstruction_tool(self),
            ReconstructionTools.create_homogeneous_refinement_tool(self),
            ReconstructionTools.create_heterogeneous_refinement_tool(self),
            ReconstructionTools.create_get_job_status_tool(self),
            ReconstructionTools.create_wait_for_job_tool(self),
            ReconstructionTools.create_get_job_log_tool(self),
            ReconstructionTools.create_reason_about_workflow_tool(self)
        ]
    
    def _get_react_system_prompt(self) -> str:
        """Get the 3D reconstruction-specific ReAct system prompt."""
        return f"""You are a CryoEM 3D reconstruction assistant using the ReAct (Reasoning + Acting) framework. 
You specialize in generating and refining 3D structures from 2D particle images.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## 3D Reconstruction Workflow Steps:

### Phase 1: Initial Model Generation
1. **Ab Initio Reconstruction**: Generate initial 3D model(s) from 2D particles
   - Required: particles_job_uid (from 2D class selection or extraction)
   - Optional: num_classes (number of 3D classes to generate, default: 1)
   - Optional: initial_resolution (starting resolution in Å, default: 20.0)
   - Optional: final_resolution (target resolution in Å, default: 10.0)
   - Optional: max_iterations (default: 50)
   - Optional: symmetry (e.g., C1, C2, D7, default: C1)
   - Generates de novo 3D structures without requiring a reference
   - Can generate multiple classes if structural heterogeneity is suspected
   - Uses stochastic gradient descent with branch and bound optimization

2. **Homogeneous Reconstruction**: Alternative method to generate 3D model from 2D particles
   - Required: particles_job_uid (from 2D class selection or extraction)
   - Optional: initial_resolution (starting resolution in Å, default: 20.0)
   - Optional: final_resolution (target resolution in Å, default: 8.0)
   - Optional: symmetry (e.g., C1, C2, D7, default: C1)
   - Often faster and more robust than ab initio for homogeneous datasets
   - Uses a different algorithm optimized for single structure reconstruction
   - Good alternative when ab initio struggles to converge

### Phase 2: Refinement (Optional)
2. **Homogeneous Refinement**: Refine a single 3D structure
   - Required: particles_job_uid, volume_job_uid (from ab initio)
   - Use when all particles represent the same structure
   - Improves resolution through iterative refinement
   
3. **Heterogeneous Refinement**: Refine multiple 3D classes simultaneously
   - Required: particles_job_uid, volume_job_uids (list of volumes from ab initio)
   - Use when structural heterogeneity is present
   - Classifies particles while refining structures

## ReAct Process:
For each step, you MUST follow this pattern:

**Thought**: [Your reasoning about what needs to be done, why, and how]
**Action**: [The specific tool to use with exact parameters]
**Observation**: [What happened as a result of the action]

## CRITICAL: Job Monitoring and Failure Recovery
- After starting ANY reconstruction job, you MUST wait for it to complete
- Use wait_for_job with the job UID to wait for completion
- Do NOT proceed to the next step until the current job is completed
- Ab initio reconstruction can take significant time (minutes to hours)

## ADAPTIVE RETRY MECHANISM:
**IMPORTANT**: Only start adaptive retry strategy when a job has FAILED, not when it has already completed successfully.

When a job FAILS (status = "failed"), you MUST implement an adaptive retry strategy with AT LEAST 3 attempts:

1. **FIRST check job status** using get_job_status tool to confirm the job has failed
2. **IMMEDIATELY read the job log** using get_job_log tool to understand the failure
3. **Analyze error patterns** and identify the root cause from CryoSPARC logs
4. **Implement adaptive retry strategy** with different parameter combinations:

   **ATTEMPT 1 (Default)**: Start with standard parameters
   
   **ATTEMPT 2 (CTF Issues)**: If CTF refinement fails:
   - refine_defocus_refine=false, refine_ctf_global_refine=false
   
   **ATTEMPT 3 (Resolution Issues)**: If resolution too aggressive:
   - refinement_resolution=15.0 (more conservative)
   
   **ATTEMPT 4 (Conservative)**: If convergence fails:
   - refinement_resolution=None (auto), symmetry=C1, no CTF refinement
   
   **ATTEMPT 5 (Alternative)**: If all refinement attempts fail:
   - Try homogeneous reconstruction instead of refinement
   
4. **Learn from each failure** and adapt parameters based on error analysis
5. **Document reasoning** for each parameter choice
6. **Continue until success** or all reasonable strategies exhausted

CRITICAL: You MUST try at least 3 different parameter combinations before giving up!

**DO NOT START RETRY STRATEGY IF:**
- Job status is "completed" (successful completion)
- Job status is "cancelled" (manually cancelled)
- Job is still "running" (wait for it to finish first)
- Job status is "queued" or "started" (wait for completion)

**ONLY START RETRY STRATEGY IF:**
- Job status is "failed" (actual failure requiring retry)

## Tool Usage Guidelines:

- **ab_initio_reconstruction**: Generate initial 3D model(s) de novo
  * Required: particles_job_uid (from 2D class selection or extraction)
  * Optional: num_classes (1 for homogeneous, 2-4 for heterogeneous)
  * Optional: initial_resolution (20.0 Å is typical starting point)
  * Optional: final_resolution (8-12 Å for initial models)
  * Optional: max_iterations (50 is usually sufficient)
  * Optional: symmetry (C1 for no symmetry, C2/D7 etc. if known)
  * Start the job, then wait for completion
  
- **homogeneous_reconstruction**: Generate single 3D model (alternative to ab initio)
  * Required: particles_job_uid (from 2D class selection or extraction)
  * Optional: initial_resolution (20.0 Å is typical starting point)
  * Optional: final_resolution (8.0 Å for initial models)
  * Optional: symmetry (C1 for no symmetry, C2/D7 etc. if known)
  * Often faster and more robust than ab initio for homogeneous datasets
  * Start the job, then wait for completion
  
- **homogeneous_refinement**: Refine single structure
  * Required: particles_job_uid, volume_job_uid (from ab initio)
  * Use after ab initio if only one good class emerges
  * Improves resolution and quality
  * Start the job, then wait for completion
  
- **heterogeneous_refinement**: Refine multiple structures
  * Required: particles_job_uid, volume_job_uids (from ab initio)
  * Use if multiple distinct structures are present
  * Simultaneously refines and classifies
  * Start the job, then wait for completion
  
- **get_job_status**: Check status of a specific job (use job UID only, e.g., "J113")
- **wait_for_job**: Wait for job completion (use job UID only, e.g., "J113")
- **reason_about_workflow**: Analyze current reconstruction state

## Job UID Format:
- Job UIDs are strings like "J113", "J114", etc.
- When calling get_job_status or wait_for_job, pass ONLY the job UID
- Do NOT use JSON format or complex parameters for these tools

## Ab Initio Parameters Guide:

**Number of Classes (num_classes)**:
- 1: Use when particles are homogeneous (all same structure)
- 2-3: Use when mild heterogeneity is suspected
- 3-4: Use when significant structural variation expected
- More classes = longer computation time

**Resolution Settings**:
- initial_resolution: Starting resolution (typically 20-30 Å)
- final_resolution: Target resolution for ab initio (typically 8-12 Å)
- Don't set final resolution too high initially (not < 8 Å)
- Better resolution comes from subsequent refinement

**Symmetry**:
- C1: No symmetry (safest default)
- CN: Cyclic symmetry (e.g., C2, C3, C5)
- DN: Dihedral symmetry (e.g., D2, D7)
- T, O, I: Tetrahedral, Octahedral, Icosahedral
- Only use if you know the symmetry - wrong symmetry can cause artifacts

**Iterations**:
- 50 iterations is typically sufficient for ab initio
- More iterations may help with difficult cases
- Monitor convergence in CryoSPARC

## Workflow Dependencies:
1. Ab initio requires completed 2D class selection or particle extraction job
2. Homogeneous refinement requires ab initio volume
3. Heterogeneous refinement requires multiple ab initio volumes
4. Each step must complete successfully before the next can begin
5. Always verify job completion before proceeding

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}

## Example Workflows:

**Simple Homogeneous Case (Ab Initio)**:
1. Run ab_initio_reconstruction with num_classes=1
2. Wait for completion
3. Optionally run homogeneous_refinement
4. Wait for completion

**Simple Homogeneous Case (Alternative)**:
1. Run homogeneous_reconstruction
2. Wait for completion
3. Optionally run homogeneous_refinement with the resulting volume
4. Wait for completion

**Heterogeneous Case**:
1. Run ab_initio_reconstruction with num_classes=3
2. Wait for completion
3. Analyze which classes are good
4. Run heterogeneous_refinement with good volumes
5. Wait for completion

Remember: Always follow the Thought → Action → Observation pattern and WAIT for each job to complete!
3D reconstruction jobs can take significant time, especially ab initio."""
    
    # =================================================================
    # Tool Implementation Methods
    # =================================================================
    
    def _ab_initio_tool(self, tool_input: str) -> str:
        """Execute ab initio reconstruction."""
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            # Support both "particles_job_uid" and "job_uid" for flexibility
            particles_job_uid = params.get("particles_job_uid") or params.get("job_uid")
            if not particles_job_uid and "__arg1" in params:
                arg_value = str(params.get("__arg1", "")).strip()
                if arg_value:
                    particles_job_uid = arg_value.split(",")[0].strip()
            if not particles_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: particles_job_uid or job_uid"
                })
            
            # Get project and workspace UIDs
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Extract optional parameters
            num_classes = params.get("num_classes", 1)
            initial_resolution = params.get("initial_resolution", 20.0)
            final_resolution = params.get("final_resolution", 10.0)
            max_iterations = params.get("max_iterations", 50)
            symmetry = params.get("symmetry", "C1")
            
            # Job control parameters
            wait_for_completion = params.get("wait_for_completion", "false").lower() == "true"
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            # Execute ab initio reconstruction
            result = self.cryosparc_tools.ab_initio_reconstruction(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particles_job_uid=particles_job_uid,
                num_classes=num_classes,
                initial_resolution=initial_resolution,
                final_resolution=final_resolution,
                max_iterations=max_iterations,
                symmetry=symmetry,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )
            
            # Log the tool execution
            self._record_tool_execution("ab_initio_reconstruction", params, result=result)
            
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("ab_initio_reconstruction", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _homogeneous_reconstruction_tool(self, tool_input: str) -> str:
        """Execute homogeneous reconstruction."""
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            # Support both "particles_job_uid" and "job_uid" for flexibility
            particles_job_uid = params.get("particles_job_uid") or params.get("job_uid")
            if not particles_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: particles_job_uid or job_uid"
                })
            
            # Get project and workspace UIDs
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Extract optional parameters
            initial_resolution = params.get("initial_resolution", 20.0)
            final_resolution = params.get("final_resolution", 8.0)
            symmetry = params.get("symmetry", "C1")
            
            # Job control parameters
            wait_for_completion = params.get("wait_for_completion", "false").lower() == "true"
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            # Execute homogeneous reconstruction
            result = self.cryosparc_tools.homogeneous_reconstruction(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particles_job_uid=particles_job_uid,
                initial_resolution=initial_resolution,
                final_resolution=final_resolution,
                symmetry=symmetry,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )
            
            # Log the tool execution
            self._record_tool_execution("homogeneous_reconstruction", params, result=result)
            
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("homogeneous_reconstruction", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _homogeneous_refinement_tool(self, tool_input: str) -> str:
        """Execute homogeneous refinement."""
        try:
            params = dict(self._parse_tool_input(tool_input))
            
            # Extract required parameters
            particles_job_uid = params.get("particles_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            arg_fallback = params.get("__arg1")
            if arg_fallback and (not particles_job_uid or not volume_job_uid):
                parsed_particles = None
                parsed_volume = None

                if isinstance(arg_fallback, dict):
                    parsed_particles = arg_fallback.get("particles_job_uid")
                    parsed_volume = arg_fallback.get("volume_job_uid")
                    params.update({k: v for k, v in arg_fallback.items() if k != "__arg1"})
                else:
                    fallback_str = str(arg_fallback).strip()
                    parsed_dict = None
                    if fallback_str.startswith("{") and fallback_str.endswith("}"):
                        try:
                            parsed_dict = json.loads(fallback_str)
                        except json.JSONDecodeError:
                            parsed_dict = None
                    if isinstance(parsed_dict, dict):
                        parsed_particles = parsed_dict.get("particles_job_uid")
                        parsed_volume = parsed_dict.get("volume_job_uid")
                        params.update({k: v for k, v in parsed_dict.items() if k != "__arg1"})
                    else:
                        tokens = [token.strip() for token in fallback_str.split(",") if token.strip()]
                        positional_values: List[str] = []
                        for token in tokens:
                            if "=" in token:
                                key, value = token.split("=", 1)
                                key = key.strip()
                                value = value.strip().strip('"')
                                if key == "particles_job_uid":
                                    parsed_particles = parsed_particles or value
                                elif key == "volume_job_uid":
                                    parsed_volume = parsed_volume or value
                                else:
                                    params.setdefault(key, value)
                            else:
                                positional_values.append(token)
                        if positional_values:
                            if not parsed_particles and positional_values:
                                parsed_particles = positional_values[0]
                            if not parsed_volume and len(positional_values) > 1:
                                parsed_volume = positional_values[1]

                particles_job_uid = particles_job_uid or parsed_particles
                volume_job_uid = volume_job_uid or parsed_volume
            
            if not particles_job_uid or not volume_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameters: particles_job_uid and volume_job_uid"
                })
            
            # Get project and workspace UIDs
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Extract optional parameters
            refinement_resolution = params.get("refinement_resolution", None)
            symmetry = params.get("symmetry", "C1")
            
            # Advanced refinement parameters
            refine_do_init_scale_est = params.get("refine_do_init_scale_est", "true").lower() == "true"
            refine_highpass_res = params.get("refine_highpass_res", None)
            refine_num_final_iterations = params.get("refine_num_final_iterations", None)
            refine_res_init = params.get("refine_res_init", None)
            refine_symmetry_do_align = params.get("refine_symmetry_do_align", "true").lower() == "true"
            
            # Job control parameters
            wait_for_completion = params.get("wait_for_completion", "false").lower() == "true"
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            # Execute homogeneous refinement
            result = self.cryosparc_tools.homogeneous_refinement(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particles_job_uid=particles_job_uid,
                volume_job_uid=volume_job_uid,
                refinement_resolution=refinement_resolution,
                symmetry=symmetry,
                refine_do_init_scale_est=refine_do_init_scale_est,
                refine_highpass_res=refine_highpass_res,
                refine_num_final_iterations=refine_num_final_iterations,
                refine_res_init=refine_res_init,
                refine_symmetry_do_align=refine_symmetry_do_align,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )
            
            # Log the tool execution
            self._record_tool_execution("homogeneous_refinement", params, result=result)
            
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("homogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _heterogeneous_refinement_tool(self, tool_input: str) -> str:
        """Execute heterogeneous refinement."""
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            particles_job_uid = params.get("particles_job_uid")
            volume_job_uids = params.get("volume_job_uids")
            
            if not particles_job_uid or not volume_job_uids:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameters: particles_job_uid and volume_job_uids"
                })
            
            # Ensure volume_job_uids is a list
            if isinstance(volume_job_uids, str):
                volume_job_uids = [v.strip() for v in volume_job_uids.split(",")]
            
            # Get project and workspace UIDs
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Extract optional parameters
            num_classes = params.get("num_classes", len(volume_job_uids))
            
            # Job control parameters
            wait_for_completion = params.get("wait_for_completion", "false").lower() == "true"
            timeout = int(params.get("timeout", self.config.job_management.default_timeout))
            check_interval = int(params.get("check_interval", self.config.job_management.status_check_interval))
            
            # Execute heterogeneous refinement
            result = self.cryosparc_tools.heterogeneous_refinement(
                project_uid=project_uid,
                workspace_uid=workspace_uid,
                particles_job_uid=particles_job_uid,
                volume_job_uids=volume_job_uids,
                num_classes=num_classes,
                wait_for_completion=wait_for_completion,
                timeout=timeout,
                check_interval=check_interval
            )
            
            # Log the tool execution
            self._record_tool_execution("heterogeneous_refinement", params, result=result)
            
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("heterogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about 3D reconstruction workflow state."""
        try:
            reasoning = f"""
🤔 **3D Reconstruction Workflow Analysis**:

**Current State**: {input_str}

**Workflow Dependencies**:
1. Initial Model Generation:
   - Ab Initio Reconstruction (requires particles from 2D class selection or extraction)
   - OR Homogeneous Reconstruction (alternative to ab initio for homogeneous datasets)
2. Optional: Homogeneous Refinement (requires volume + particles)
3. Optional: Heterogeneous Refinement (requires multiple volumes + particles)

**Ab Initio Reconstruction Parameters**:
- **num_classes**: 1 for homogeneous, 2-4 for heterogeneous datasets
- **initial_resolution**: Start at 20-30 Å for stability
- **final_resolution**: Target 8-12 Å for ab initio (don't go < 8 Å)
- **symmetry**: C1 (no symmetry) is safest default
- **max_iterations**: 50 is typical, 100 for difficult cases

**Homogeneous Reconstruction Parameters**:
- **initial_resolution**: Start at 20-30 Å for stability
- **final_resolution**: Target 8.0 Å (can be more aggressive than ab initio)
- **symmetry**: C1 (no symmetry) is safest default
- Often faster and more robust than ab initio for single structure datasets

**Next Steps Analysis**:
- If no reconstruction jobs are running: Choose between:
  * ab_initio_reconstruction (standard approach, supports multiple classes)
  * homogeneous_reconstruction (faster alternative for single structure)
- If reconstruction is running: Wait for completion, then assess results
- If initial model completed successfully: Decide on refinement strategy
  * Single good class → homogeneous refinement
  * Multiple distinct classes → heterogeneous refinement
  * Poor results → try alternative method or re-run with adjusted parameters

**Recommended Actions**:
- Always wait for ab initio to complete before refinement
- Check FSC curves in CryoSPARC to assess resolution
- Review 3D volumes visually before proceeding
- Use conservative parameters initially
- Save refinement for validated structures

**Common Issues**:
- Ab initio fails to converge: Increase initial_resolution, reduce num_classes
- Multiple classes look identical: Dataset may be homogeneous, use num_classes=1
- Low resolution: Expected for ab initio; refinement improves resolution
- Artifacts in volume: Check symmetry (try C1), verify particle quality
"""
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": reasoning})
            return reasoning
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error in workflow reasoning: {str(e)}"

