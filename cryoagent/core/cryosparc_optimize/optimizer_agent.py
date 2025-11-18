"""ReAct-based box size optimization agent for CryoEM 3D reconstruction."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain.tools import Tool
from langchain_core.language_models import BaseLanguageModel

from ..base_react_agent import BaseReActAgent
from .optimizer_tools import OptimizerTools
from ...tools.cryosparc_tools import CryoSPARCTools
from ...config.config_loader import CryoAgentConfig


class OptimizerAgent(BaseReActAgent):
    """ReAct-based agent for optimizing box size/diameter in CryoEM 3D reconstruction."""
    
    def __init__(
        self,
        cryosparc_tools: CryoSPARCTools,
        config: CryoAgentConfig,
        llm: Optional[BaseLanguageModel] = None
    ):
        """
        Initialize the box size optimization agent.
        
        Args:
            cryosparc_tools: CryoSPARC tools instance
            config: Complete configuration object
            llm: Language model for the agent
        """
        # Initialize stage_config BEFORE calling super().__init__() because
        # BaseReActAgent.__init__() calls _create_tools() which may access stage_config
        self.workflow_defaults: Dict[str, Any] = {}
        self.stage_config = self._load_stage_config()
        self.stage_workflow = self.stage_config.get("workflow", {})
        
        # Now call super().__init__() which will call _create_tools()
        super().__init__(cryosparc_tools, config, llm)
        
        # Initialize logger for this agent
        self.logger = logging.getLogger("OptimizerAgent")
    
    def _create_tools(self) -> List[Tool]:
        """Create optimization-specific tools."""
        return [
            OptimizerTools.create_test_box_size_tool(self),
            OptimizerTools.create_get_fsc_info_tool(self),
            OptimizerTools.create_get_job_status_tool(self),
            OptimizerTools.create_wait_for_job_tool(self),
            OptimizerTools.create_get_job_log_tool(self),
            OptimizerTools.create_reason_about_workflow_tool(self),
            OptimizerTools.create_get_hetero_class_resolutions_tool(self),
            OptimizerTools.create_test_heterogeneous_refinement_tool(self)
        ]
    
    def _load_stage_config(self) -> Dict[str, Any]:
        """Load optimization stage configuration."""
        config_path = Path("configs/cryosparc/optimization_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _load_reconstruction_config(self) -> Dict[str, Any]:
        """Load reconstruction stage configuration to get symmetry settings."""
        config_path = Path("configs/cryosparc/reconstruction_config.json")
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}
    
    def _get_refinement_symmetry(self) -> str:
        """Get symmetry from reconstruction_config.json for homogeneous refinement."""
        recon_config = self._load_reconstruction_config()
        
        # First try to get from workflow.refinement.symmetry (preferred)
        refinement_symmetry = recon_config.get("workflow", {}).get("refinement", {}).get("symmetry")
        if refinement_symmetry:
            return refinement_symmetry
        
        # Fall back to microscope_parameters.symmetry
        microscope_symmetry = recon_config.get("microscope_parameters", {}).get("symmetry")
        if microscope_symmetry:
            return microscope_symmetry
        
        # Default to C1 if not found
        return "C1"
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Fetch a parameter from the stage workflow configuration."""
        return self.stage_workflow.get(section, {}).get(key, default)
    
    def _get_react_system_prompt(self) -> str:
        """Get the optimization-specific ReAct system prompt."""
        # Check if heterogeneous refinement is enabled
        enable_box_size = self._get_stage_param("optimization", "enable_box_size_optimization", True)
        enable_hetero = self._get_stage_param("optimization", "enable_heterogeneous_refinement", False)
        max_hetero_iterations = self._get_stage_param("optimization", "heterogeneous_refinement_max_iterations", 3)
        
        # Determine what to optimize
        optimization_types = []
        if enable_box_size:
            optimization_types.append("box size/diameter")
        if enable_hetero:
            optimization_types.append("heterogeneous refinement (K values)")
        
        optimization_desc = " and ".join(optimization_types) if optimization_types else "parameters"
        
        # Build box size optimization section conditionally
        box_size_section = ""
        if enable_box_size:
            box_size_section = """
## Box Size Optimization Workflow:

**Purpose**: After the first round of 3D homogeneous refinement, optimize the box size to achieve the best resolution.

**Agentic Optimization Process**:
1. **Initial Assessment**: Get FSC resolution and box size from the original refinement job using `get_fsc_info`
2. **First Round Testing**: Test 10% less and 10% more box sizes using `test_box_size` tool
3. **REASONING REQUIRED**: After EACH test, you MUST actively reason about the results:
   - Compare box size results across all tested values so far
   - Identify trends: Which direction (larger/smaller box sizes) improves resolution?
   - Determine if the optimal point is clear or needs more testing
   - Calculate the next box size to test based on trends
4. **Decision Making**: Based on your reasoning, decide:
   - **Continue testing**: Choose which box size to test next based on trends
   - **Stop optimization**: If resolution plateaus, worsens, or you've found the optimal box size
5. **Iterative Process**: Repeat steps 3-4 until:
   - You find the optimal box size (clear best result)
   - Resolution plateaus or worsens (diminishing returns)
   - You reach a reasonable number of tests (5-7 different box sizes)
   - Further testing is unlikely to improve results
6. **Conclusion**: Summarize the best box size and resolution found, explaining why it was chosen

**Stopping Conditions**:
- You've tested 5-7 different box sizes
- The resolution improvement plateaus or starts getting worse
- You've found a clear optimal point
- The new box size to test would be the same as one already tested

## Tool Usage for Box Size Optimization:

- **get_fsc_info**: Get FSC resolution and box size from a refinement job
  * Required: refinement_job_uid
  * Returns: box_size (pixels), resolution_angstroms (FSC resolution)
  * Use this to get baseline information from the original refinement job

- **test_box_size**: Test a specific box size by extracting particles, running refinement, and getting FSC
  * Required: box_size_pix (box size in pixels), refinement_job_uid (source of refined coordinates),
    micrographs_job_uid, volume_job_uid
  * Optional: refinement_resolution (target resolution in Angstroms)
  * This tool: 1) Extracts particles with the box size using refined coordinates, 2) Runs refinement, 3) Returns FSC resolution
  * Note: Particle re-extraction uses coordinates from refinement_job_uid (refined positions/orientations)
  * **IMPORTANT**: Box sizes are automatically normalized to allowed values. If your requested box size is normalized, 
    the result will include `box_size_was_normalized: true` and `normalization_message` explaining the change.
    The `box_size_pix` in the result is the actual (normalized) box size that was used.
  * **WARNING: Only use this tool if box size optimization is enabled!**

## Optimization Strategy Guidelines:

**Initial Testing**:
- Always start by getting FSC info from the original refinement job
- Test 10% less (original * 0.9) and 10% more (original * 1.1) box sizes
- **Box Size Normalization**: Box sizes are automatically normalized to allowed CryoSPARC values (e.g., 16, 20, 24, 28, 32, ..., 2000).
  If you request a box size like 483, it may be normalized to 480 or 484 (the nearest allowed value).
  The tool result will indicate if normalization occurred via `box_size_was_normalized` and `normalization_message`.

**Trend Analysis**:
- **CRITICAL: Smaller resolution_angstroms value = BETTER quality** (e.g., 3.0 Å is BETTER than 5.0 Å)
- When comparing results, the box size with the SMALLEST resolution_angstroms value (lower numeric value) is the BEST
- Look for patterns: Does resolution improve with larger or smaller box sizes?
- Consider if the relationship is linear, quadratic, or has an optimal point
- Always identify the box size with the SMALLEST resolution_angstroms value as the best

**Next Steps Decision**:
- If middle box size is best: Test halfway between middle and the better extreme
- If smallest box size is best: Test 10% less than the smallest tested
- If largest box size is best: Test 10% more than the largest tested
- Consider testing refinement_resolution parameter if box size alone doesn't show clear improvement

**Example Reasoning Pattern** (use actual values from your test results, not these examples):
```
Thought: I have tested multiple box sizes. Let me analyze the results:
- Original box size: [resolution] Å
- Smaller box size (-10%): [resolution] Å  
- Larger box size (+10%): [resolution] Å

Analysis:
- Compare which box size gives the smallest resolution value
- Identify the trend: is resolution improving with larger or smaller box sizes?
- Assess if the improvement is significant or marginal

Decision: Based on the trend, decide:
- If larger box sizes improve resolution: test even larger box size
- If smaller box sizes improve resolution: test even smaller box size
- If middle is best: test between middle and better extreme
- Monitor if improvement is diminishing or resolution starts worsening

Action: test_box_size with appropriate box_size_pix value based on your analysis
```

## Example Workflow for Box Size Optimization:

1. Get refinement_job_uid, micrographs_job_uid, volume_job_uid from previous stages
2. Use `get_fsc_info` to get baseline resolution from refinement_job_uid
3. Calculate and test 10% less box size using `test_box_size`
4. Calculate and test 10% more box size using `test_box_size`
5. Analyze the three results (original, -10%, +10%)
6. Reason about trends and decide next box size to test
7. Continue testing and analyzing iteratively
8. Conclude with the best box size and resolution found
"""
        
        return f"""You are a CryoEM optimization assistant using the ReAct (Reasoning + Acting) framework. 
You specialize in optimizing {optimization_desc} for 3D reconstruction by testing different parameters and comparing FSC resolutions.

## ReAct Framework Rules:
1. **REASONING**: Always think through the problem step by step before taking action
2. **ACTING**: Execute specific tools based on your reasoning
3. **OBSERVING**: Analyze the results and update your understanding

## Optimization Workflow Priority:
{f'**IMPORTANT**: Both box size optimization and heterogeneous refinement are enabled. Complete box size optimization FIRST, then use the optimized box size refinement job for heterogeneous refinement.**' if enable_box_size and enable_hetero else ''}
{f'**CRITICAL: Box size optimization is DISABLED. DO NOT use test_box_size tool. Proceed directly to heterogeneous refinement using the refinement_job_uid provided.**' if not enable_box_size and enable_hetero else ''}
{f'**Heterogeneous refinement is DISABLED**. Proceed with box size optimization only.**' if enable_box_size and not enable_hetero else ''}
{box_size_section}
## General Tool Usage:

- **get_fsc_info**: Get FSC resolution and box size from a refinement job
  * Required: refinement_job_uid
  * Returns: box_size (pixels), resolution_angstroms (FSC resolution)
  * Use this to get baseline information from any refinement job

- **get_job_status**: Check status of a specific job (use job UID only, e.g., "J113")
- **wait_for_job**: Wait for job completion (use job UID only, e.g., "J113")
- **get_job_log**: Read and analyze job logs
- **reason_about_workflow**: Analyze current optimization state and think about next steps

## Job UID Format:
- Job UIDs are strings like "J113", "J114", etc.
- When calling get_job_status, wait_for_job, or get_fsc_info, you can pass ONLY the job UID (e.g., "J357")
- For other tools, use JSON format with parameter names

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}
- Box size optimization: {'ENABLED' if enable_box_size else 'DISABLED'}
- Heterogeneous refinement: {'ENABLED' if enable_hetero else 'DISABLED'}

## Heterogeneous Refinement Optimization Workflow:
{'' if not enable_hetero else f'''

**Purpose**: After box size optimization (if enabled), optimize the number of classes (K) in heterogeneous refinement to achieve the best resolution.

**Agentic Optimization Process**:
1. **Baseline**: Get FSC resolution from the final refinement job (K=1, which is homogeneous refinement)
2. **First Round Testing**: Test K=3 and K=5 using `test_heterogeneous_refinement` tool
3. **REASONING REQUIRED**: After EACH test, you MUST actively reason about the results:
   - Analyze the `class_comparison` data and `class_selection_reason` from each test
   - Compare resolution results across all K values tested so far
   - Identify trends: Which direction (more/fewer K) improves resolution?
   - Determine if the optimal point is clear or needs more testing
4. **Decision Making**: Based on your reasoning, decide:
   - **Continue testing**: Choose which K value to test next based on trends
   - **Stop optimization**: If resolution plateaus, worsens, or you've found the optimal K
5. **Iterative Process**: Repeat steps 3-4 until:
   - You find the optimal K value (clear best result)
   - Resolution plateaus or worsens (diminishing returns)
   - You reach max_iterations ({max_hetero_iterations})
   - Further testing is unlikely to improve results
6. **Conclusion**: Summarize the best K value and resolution found, explaining why it was chosen

**Stopping Conditions**:
- You've tested {max_hetero_iterations} different K values
- The resolution improvement plateaus or starts getting worse
- You've found a clear optimal point
- The new K value to test would be the same as one already tested

**Tool Usage for Heterogeneous Refinement**:

- **test_heterogeneous_refinement**: Test heterogeneous refinement with K classes
  * **Input format: JSON string** (e.g., `{{"k": 3, "refinement_job_uid": "J357"}}`)
  * Required parameters: k (number of classes, e.g., 3 or 5), refinement_job_uid (source of particles and volume, e.g., "J357")
  * This tool: 1) Repeats the volume from refinement_job_uid K times, 2) Runs heterogeneous refinement,
    3) Gets resolution for each class, 4) Selects best class (smallest resolution value, or HIGHEST fsc_loosemask_last if tied - higher FSC is better), 
    5) Runs homogeneous refinement on selected class, 6) Returns final FSC resolution
  * Returns: hetero_job_uid, best_class_id, best_class_resolution, class_selection_reason, class_comparison (all classes data), refine_job_uid, final_resolution_angstroms, and all_classes
  * **Important**: The tool automatically selects the best class using an algorithm, but you should REASON about the class_comparison and class_selection_reason to verify the selection makes sense and understand why it was chosen.
  * **Example**: Use JSON format: `{{"k": 3, "refinement_job_uid": "J357"}}` or `{{"k": 5, "refinement_job_uid": "J357"}}`

- **get_hetero_class_resolutions**: Get resolution for each class in a heterogeneous refinement job
  * Required: job_uid (heterogeneous refinement job UID)
  * Returns: classes (list with class_id, resolution_angstroms, fsc_loosemask_last), num_classes
  * Use this to analyze individual class resolutions if needed

- **get_fsc_info**: Get FSC resolution from any refinement job (works for both homogeneous and heterogeneous)
  * Required: refinement_job_uid
  * Returns: box_size, resolution_angstroms

**Heterogeneous Refinement Strategy Guidelines**:

**Initial Testing**:
- Always start by getting FSC info from the original refinement job (K=1 baseline)
- Test K=3 and K=5 in the first round
- Compare K=1, K=3, and K=5 to see the trend

**REASONING REQUIREMENT - After Each Test**:
**CRITICAL**: After calling `test_heterogeneous_refinement`, you MUST actively reason about:
1. **Class Selection Analysis**: Review the `class_comparison` data and `class_selection_reason` to understand which class was selected and why
   - Are all classes in the heterogeneous refinement similar in quality?
   - Was there a clear winner, or were classes close in resolution?
   - Does the selected class seem reasonable given all available data?

2. **K Value Comparison**: Compare the final resolution from this K value with previous K values tested
   - Which K value has given the best resolution so far?
   - Is there a clear trend (e.g., increasing K improves resolution, or vice versa)?

3. **Decision Making**: Based on your analysis, decide:
   - **Which K to test next** (if any): If there's a clear trend, test further in that direction. If optimal seems to be between tested values, test intermediate values.
   - **Whether to STOP**: Consider stopping if:
     * Resolution is getting worse with more K values
     * Resolution has plateaued (no improvement across multiple K values)
     * You've found a clear optimal K value
     * You've reached or approached max_iterations ({max_hetero_iterations})
     * Testing more K values would likely not improve results

**Trend Analysis Guidelines**:
- **CRITICAL: Smaller resolution_angstroms value = BETTER quality** (e.g., 3.0 Å is BETTER than 5.0 Å)
- When comparing results, the K value with the SMALLEST resolution_angstroms value (lower numeric value) is the BEST
- Look for patterns: Does resolution improve with larger or smaller K values?
- Consider if the relationship has an optimal point (sweet spot)

**Next Steps Decision Examples**:
- If K=1 is best: Test K=2 to see if slight heterogeneity helps. If K=2 is worse, consider stopping.
- If K=3 is best: Test K=4 to see if more classes help. Also consider testing K=2 to see if trend goes both ways.
- If K=5 is best: Test K=7 or K=8 to see if trend continues. Monitor if resolution improvement is diminishing.
- If middle K is best: Test values around it (e.g., if K=3 is best, test K=2 and K=4) to confirm it's truly optimal.
- If resolution is getting worse: Stop and use the best K found so far.
- If resolution plateaus: Stop if improvement is minimal (< 0.1 Å difference) and select the best K.

**Example Reasoning Pattern** (use actual values from your test results, not these examples):
```
Thought: I have tested multiple K values. Let me analyze the results:
- K=1 (original): [resolution] Å
- K=3: [resolution] Å (check class_comparison and class_selection_reason)
- K=5: [resolution] Å (check class_comparison and class_selection_reason)

Analysis:
- Compare resolutions across all tested K values - which has the smallest resolution value?
- Identify the trend: does increasing K improve or worsen resolution?
- Review class_comparison data: are classes similar or is there a clear winner?
- Check class_selection_reason: understand why each class was selected

Decision: Based on the trend and analysis:
- If larger K improves resolution: test higher K (e.g., K=7)
- If smaller K is better: test K=2 or stop
- If middle K is best: test around it (e.g., K=2, K=4)
- Monitor if improvement is diminishing

Action: test_heterogeneous_refinement with appropriate k value based on your analysis

[After getting result]
Thought: Compare new result with previous results. Has resolution improved, worsened, or plateaued?

Decision: Decide whether to:
- Continue testing if trend suggests improvement
- STOP if resolution worsens, plateaus, or optimal point is clear
```

**CRITICAL: When calling test_heterogeneous_refinement, you MUST use JSON format with Action Input!**
- Correct: Action Input: `{{"k": 3, "refinement_job_uid": "J357"}}`
- Wrong: test_heterogeneous_refinement(3, "J357") - this will fail!
- The tool requires a single JSON string input, not multiple arguments
'''}

## Combined Workflow:
{'' if not (enable_box_size and enable_hetero) else '''
**If both optimizations are enabled**:
1. **First**: Complete box size optimization (get best box size and refinement job)
2. **Then**: Use the best refinement job from box size optimization as the input for heterogeneous refinement
3. **Finally**: Report both optimizations' results

**Example Combined Flow**:
1. Optimize box size → Get best refinement job (e.g., J100)
2. Use J100 as refinement_job_uid for heterogeneous refinement
3. Optimize K values → Get best heterogeneous refinement result
4. Report: best_box_size, best_box_resolution, best_hetero_k, best_hetero_resolution
'''}

Remember: Always follow the Thought → Action → Observation pattern!
Think carefully about trends before deciding what to test next. Both optimizations can take significant time as each test requires running refinement jobs."""
    
    def update_workflow_defaults(self, defaults: Dict[str, Any]) -> None:
        """Store workflow-level default parameters for later tool invocations."""
        if defaults:
            if not hasattr(self, "workflow_defaults") or self.workflow_defaults is None:
                self.workflow_defaults = {}
            self.workflow_defaults.update(defaults)
    
    # =================================================================
    # Tool Implementation Methods
    # =================================================================
    
    def _get_fsc_info_tool(self, tool_input: str) -> str:
        """
        Get FSC resolution and box size from a refinement job.
        
        This tool retrieves the FSC resolution and box size information
        from a completed refinement job.
        
        Can accept either:
        - Just the job UID as a string (e.g., "JXXX")
        - JSON with refinement_job_uid parameter
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Support both direct job UID string and JSON parameter
            # _parse_tool_input converts "J357" to {"job_uid": "J357"}
            refinement_job_uid = params.get("refinement_job_uid") or params.get("job_uid")
            
            # If still not found, try to extract from input string directly
            if not refinement_job_uid:
                # Check if input is just a job UID (starts with J and is short)
                input_stripped = tool_input.strip().strip('"\'')
                if input_stripped.startswith("J") and len(input_stripped) <= 10:
                    refinement_job_uid = input_stripped
                # Also check if it's in the "input" field
                elif "input" in params:
                    input_val = str(params["input"]).strip().strip('"\'')
                    if input_val.startswith("J") and len(input_val) <= 10:
                        refinement_job_uid = input_val
            
            if not refinement_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: refinement_job_uid. You can pass just the job UID (e.g., 'J357') or JSON with refinement_job_uid parameter."
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            
            # Get FSC info from the refinement job
            fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refinement_job_uid)
            
            if not fsc_info.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get FSC info: {fsc_info.get('error', 'Unknown error')}"
                })
            
            result = {
                "success": True,
                "refinement_job_uid": refinement_job_uid,
                "box_size": fsc_info.get("box_size"),
                "resolution_angstroms": fsc_info.get("resolution_angstroms")
            }
            
            self._record_tool_execution("get_fsc_info", {"refinement_job_uid": refinement_job_uid, "project_uid": project_uid}, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("get_fsc_info", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _test_box_size_tool(self, tool_input: str) -> str:
        """
        Test a specific box size by extracting particles, running refinement, and getting FSC resolution.
        
        This tool:
        1. Extracts particles with the specified box size using refined coordinates from refinement_job_uid
        2. Runs homogeneous refinement
        3. Gets FSC resolution from the refinement result
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            box_size_pix = params.get("box_size_pix")
            refinement_job_uid = params.get("refinement_job_uid")
            micrographs_job_uid = params.get("micrographs_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            
            # Try to get from workflow defaults if not provided
            defaults = getattr(self, "workflow_defaults", {}) or {}
            if not micrographs_job_uid:
                micrographs_job_uid = (
                    defaults.get("micrographs_job_uid")
                    or defaults.get("micrograph_selection_job_uid")
                    or defaults.get("final_micrographs_job_uid")
                )
            if not volume_job_uid:
                volume_job_uid = (
                    defaults.get("volume_job_uid")
                    or defaults.get("last_volume_job_uid")
                    or defaults.get("ab_init_job_uid")
                    or defaults.get("homogeneous_reconstruction_job_uid")
                )
            
            if not all([box_size_pix, refinement_job_uid, micrographs_job_uid, volume_job_uid]):
                missing = []
                if not box_size_pix:
                    missing.append("box_size_pix")
                if not refinement_job_uid:
                    missing.append("refinement_job_uid")
                if not micrographs_job_uid:
                    missing.append("micrographs_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}"
                })
            
            # Normalize box size to allowed values
            original_box_size_pix = int(box_size_pix)
            box_size_pix = self._normalize_box_size(box_size_pix)
            
            # Warn if box size was changed by normalization
            if box_size_pix != original_box_size_pix:
                self.logger.warning(f"⚠️  Box size normalized: {original_box_size_pix} → {box_size_pix} (to nearest allowed value)")
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            refinement_resolution = params.get("refinement_resolution")
            
            self.logger.info(f"🔬 Testing box size: {box_size_pix} pixels" + (f" (normalized from {original_box_size_pix})" if box_size_pix != original_box_size_pix else ""))
            
            # Step 1: Extract particles with new box size
            self.logger.info(f"📦 Step 1/3: Extracting particles with box_size {box_size_pix}...")
            extract_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": refinement_job_uid,  # Use refinement job for refined coordinates
                "micrographs_job_uid": micrographs_job_uid,
                "box_size_pix": box_size_pix
            }
            self._record_tool_execution("extract_particles", extract_params)
            extract_result = self.cryosparc_tools.extract_particles(
                **extract_params,
                wait_for_completion=True,
                timeout=self.config.job_management.default_timeout,
                check_interval=self.config.job_management.status_check_interval
            )
            self._record_tool_execution("extract_particles", extract_params, result=extract_result)
            
            # Verify extraction completed successfully
            extract_status = extract_result.get("status", "unknown")
            if extract_status != "completed":
                error_msg = extract_result.get("error") or f"Status: {extract_status}"
                self.logger.error(f"❌ Particle extraction failed for box_size {box_size_pix}: {error_msg}")
                error_result = {
                    "success": False,
                    "error": f"Particle extraction failed: {error_msg}",
                    "box_size_pix": box_size_pix,  # Normalized box size
                    "box_size_was_normalized": box_size_pix != original_box_size_pix
                }
                if box_size_pix != original_box_size_pix:
                    error_result["requested_box_size_pix"] = original_box_size_pix
                    error_result["normalization_message"] = f"Box size {original_box_size_pix} was normalized to {box_size_pix}"
                return json.dumps(error_result)
            
            extract_job_uid = extract_result["job_uid"]
            self.logger.info(f"✅ Step 1/3: Extraction completed for box_size {box_size_pix}, job: {extract_job_uid}")
            
            # Step 2: Run homogeneous refinement
            self.logger.info(f"🔧 Step 2/3: Starting refinement for box_size {box_size_pix}...")
            symmetry = self._get_refinement_symmetry()
            refine_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": extract_job_uid,
                "volume_job_uid": volume_job_uid,
                "symmetry": symmetry
            }
            if refinement_resolution is not None:
                refine_params["refinement_resolution"] = float(refinement_resolution)
            
            self._record_tool_execution("homogeneous_refinement", refine_params)
            refine_result = self.cryosparc_tools.homogeneous_refinement(
                **refine_params,
                wait_for_completion=True,
                timeout=self.config.job_management.default_timeout,
                check_interval=self.config.job_management.status_check_interval
            )
            self._record_tool_execution("homogeneous_refinement", refine_params, result=refine_result)
            
            # Verify refinement completed successfully
            if not refine_result.get("success", False):
                error_msg = refine_result.get("error") or "Unknown error"
                self.logger.error(f"❌ Refinement failed for box_size {box_size_pix}: {error_msg}")
                error_result = {
                    "success": False,
                    "error": f"Refinement failed: {error_msg}",
                    "box_size_pix": box_size_pix,  # Normalized box size
                    "box_size_was_normalized": box_size_pix != original_box_size_pix
                }
                if box_size_pix != original_box_size_pix:
                    error_result["requested_box_size_pix"] = original_box_size_pix
                    error_result["normalization_message"] = f"Box size {original_box_size_pix} was normalized to {box_size_pix}"
                return json.dumps(error_result)
            
            refine_status = refine_result.get("status", "unknown")
            if refine_status != "completed":
                error_msg = refine_result.get("error") or f"Status: {refine_status}"
                self.logger.error(f"❌ Refinement did not complete for box_size {box_size_pix}: {error_msg}")
                error_result = {
                    "success": False,
                    "error": f"Refinement did not complete: {error_msg}",
                    "box_size_pix": box_size_pix,  # Normalized box size
                    "box_size_was_normalized": box_size_pix != original_box_size_pix
                }
                if box_size_pix != original_box_size_pix:
                    error_result["requested_box_size_pix"] = original_box_size_pix
                    error_result["normalization_message"] = f"Box size {original_box_size_pix} was normalized to {box_size_pix}"
                return json.dumps(error_result)
            
            refine_job_uid = refine_result["job_uid"]
            self.logger.info(f"✅ Step 2/3: Refinement completed for box_size {box_size_pix}, job: {refine_job_uid}")
            
            # Step 3: Get FSC info and resolution
            self.logger.info(f"📊 Step 3/3: Getting FSC resolution for box_size {box_size_pix}...")
            fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
            if fsc_info.get("success"):
                resolution = fsc_info["resolution_angstroms"]
                self.logger.info(f"✅ Step 3/3: Box size {box_size_pix}: Resolution = {resolution} Å")
                
                result = {
                    "success": True,
                    "box_size_pix": box_size_pix,  # Normalized (actual) box size used
                    "requested_box_size_pix": original_box_size_pix if box_size_pix != original_box_size_pix else None,  # Original requested if different
                    "box_size_was_normalized": box_size_pix != original_box_size_pix,
                    "refinement_job_uid": refine_job_uid,
                    "extract_job_uid": extract_job_uid,
                    "box_size": fsc_info.get("box_size", box_size_pix),
                    "resolution_angstroms": resolution
                }
                
                # Add warning message if normalized
                if box_size_pix != original_box_size_pix:
                    result["normalization_message"] = f"Box size {original_box_size_pix} was normalized to {box_size_pix} (nearest allowed value)"
                
                self._record_tool_execution("test_box_size", params, result=result)
                return json.dumps(result)
            else:
                error_msg = fsc_info.get("error", "Unknown error")
                self.logger.error(f"❌ Failed to get FSC info for box_size {box_size_pix}: {error_msg}")
                error_result = {
                    "success": False,
                    "error": f"Failed to get FSC info: {error_msg}",
                    "box_size_pix": box_size_pix,  # Normalized box size
                    "refinement_job_uid": refine_job_uid,
                    "box_size_was_normalized": box_size_pix != original_box_size_pix
                }
                if box_size_pix != original_box_size_pix:
                    error_result["requested_box_size_pix"] = original_box_size_pix
                    error_result["normalization_message"] = f"Box size {original_box_size_pix} was normalized to {box_size_pix}"
                return json.dumps(error_result)
                
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("test_box_size", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _optimize_diameter_tool(self, tool_input: str) -> str:
        """
        [DEPRECATED] Optimize box size/diameter by testing different box sizes and comparing FSC resolutions.
        
        This tool is deprecated in favor of the agentic approach using test_box_size and get_fsc_info.
        The LLM should now use those tools iteratively to optimize box size.
        
        This tool implements an iterative optimization algorithm:
        1. After first homogeneous refinement, test with 10% less and 10% more box sizes
        2. Extract particles with new box sizes and run homogeneous refinement
        3. Compare three results (original, -10%, +10%)
        4. If middle is best, choose value in between
        5. If side is better, test another 10% more/less on that side
        6. Stop if >5 diameters tested or new box size equals original
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            refinement_job_uid = params.get("refinement_job_uid")
            particles_job_uid = params.get("particles_job_uid")
            micrographs_job_uid = params.get("micrographs_job_uid")
            volume_job_uid = params.get("volume_job_uid")
            
            # Try to get from workflow defaults if not provided
            defaults = getattr(self, "workflow_defaults", {}) or {}
            if not particles_job_uid:
                # Try to get picking job (for re-extraction) from various possible keys
                particles_job_uid = (
                    defaults.get("particles_job_uid")  # Fallback to extracted particles
                    or defaults.get("selected_particles_job_uid")
                )
            if not micrographs_job_uid:
                micrographs_job_uid = (
                    defaults.get("micrographs_job_uid")
                    or defaults.get("micrograph_selection_job_uid")
                    or defaults.get("final_micrographs_job_uid")
                )
            if not volume_job_uid:
                volume_job_uid = (
                    defaults.get("volume_job_uid")
                    or defaults.get("last_volume_job_uid")
                    or defaults.get("ab_init_job_uid")
                    or defaults.get("homogeneous_reconstruction_job_uid")
                )
            
            if not all([refinement_job_uid, particles_job_uid, micrographs_job_uid, volume_job_uid]):
                missing = []
                if not refinement_job_uid:
                    missing.append("refinement_job_uid")
                if not particles_job_uid:
                    missing.append("particles_job_uid")
                if not micrographs_job_uid:
                    missing.append("micrographs_job_uid")
                if not volume_job_uid:
                    missing.append("volume_job_uid")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}"
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            # Get original FSC info
            fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refinement_job_uid)
            if not fsc_info.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get FSC info from refinement job: {fsc_info.get('error')}"
                })
            
            original_box_size = fsc_info["box_size"]
            original_resolution = fsc_info["resolution_angstroms"]
            
            # Track tested box sizes and their resolutions
            tested_data = [
                {"box_size": original_box_size, "resolution": original_resolution, "job_uid": refinement_job_uid}
            ]
            
            max_iterations = 5
            iteration = 0
            initial_three_tested = False
            
            while iteration < max_iterations:
                iteration += 1
                
                # First iteration: test all three box sizes (original, -10%, +10%)
                # Note: Original is already tested from initial refinement, so we only test -10% and +10%
                if not initial_three_tested:
                    box_size_less = int(original_box_size * 0.9)
                    box_size_more = int(original_box_size * 1.1)
                    
                    # Normalize to allowed box sizes
                    box_size_less = self._normalize_box_size(box_size_less)
                    box_size_more = self._normalize_box_size(box_size_more)
                    
                    # Skip if same as original (shouldn't happen, but check anyway)
                    if box_size_less == original_box_size:
                        box_size_less = None
                    if box_size_more == original_box_size:
                        box_size_more = None
                    
                    # Skip if already tested (shouldn't happen in first iteration, but check)
                    if box_size_less and box_size_less in [d["box_size"] for d in tested_data]:
                        box_size_less = None
                    if box_size_more and box_size_more in [d["box_size"] for d in tested_data]:
                        box_size_more = None
                    
                    initial_three_tested = True
                    self.logger.info(f"First iteration: Testing box sizes -10%={box_size_less}, +10%={box_size_more} (original={original_box_size} already tested)")
                else:
                    # Subsequent iterations: determine next box size based on best result
                    # Sort by box size to find positions
                    tested_data_sorted = sorted(tested_data, key=lambda x: x["box_size"])
                    
                    # Find best resolution (lower is better)
                    best_idx = min(range(len(tested_data_sorted)), key=lambda i: tested_data_sorted[i]["resolution"])
                    best_box_size = tested_data_sorted[best_idx]["box_size"]
                    best_resolution = tested_data_sorted[best_idx]["resolution"]
                    
                    # Determine next box size based on which is best
                    next_box_size = None
                    
                    # Find position of best in sorted list
                    if len(tested_data_sorted) >= 3:
                        # Check if best is in the middle
                        middle_idx = len(tested_data_sorted) // 2
                        if best_idx == middle_idx:
                            # Middle is best, choose value between middle and better side
                            # Compare left and right sides
                            if best_idx > 0 and best_idx < len(tested_data_sorted) - 1:
                                left_res = tested_data_sorted[best_idx - 1]["resolution"]
                                right_res = tested_data_sorted[best_idx + 1]["resolution"]
                                
                                if left_res < right_res:
                                    # Left is better, choose between left and middle
                                    next_box_size = int((tested_data_sorted[best_idx - 1]["box_size"] + best_box_size) / 2)
                                else:
                                    # Right is better, choose between middle and right
                                    next_box_size = int((best_box_size + tested_data_sorted[best_idx + 1]["box_size"]) / 2)
                        elif best_idx == 0:
                            # Leftmost (smallest) is best, test 10% less
                            next_box_size = int(best_box_size * 0.9)
                        elif best_idx == len(tested_data_sorted) - 1:
                            # Rightmost (largest) is best, test 10% more
                            next_box_size = int(best_box_size * 1.1)
                        else:
                            # Best is somewhere in between, move towards it
                            if best_idx < middle_idx:
                                # Best is on left side, test 10% less
                                next_box_size = int(best_box_size * 0.9)
                            else:
                                # Best is on right side, test 10% more
                                next_box_size = int(best_box_size * 1.1)
                    else:
                        # Less than 3 data points, continue in direction of best
                        if best_idx == 0:
                            next_box_size = int(best_box_size * 0.9)
                        else:
                            next_box_size = int(best_box_size * 1.1)
                    
                    if next_box_size:
                        next_box_size = self._normalize_box_size(next_box_size)
                        
                        # Check if already tested
                        if next_box_size in [d["box_size"] for d in tested_data]:
                            # Already tested, stop
                            break
                        
                        box_size_less = None
                        box_size_more = None
                        
                        # Set the appropriate variable based on direction
                        if next_box_size < best_box_size:
                            box_size_less = next_box_size
                        else:
                            box_size_more = next_box_size
                    else:
                        # No next box size determined, stop
                        break
                
                # If no box sizes to test, stop
                if not box_size_less and not box_size_more:
                    break
                
                # Test box_size_less
                if box_size_less:
                    # Check if already tested
                    if box_size_less in [d["box_size"] for d in tested_data]:
                        self.logger.info(f"Skipping box_size_less {box_size_less} - already tested")
                    else:
                        try:
                            self.logger.info(f"🔬 Testing box_size_less: {box_size_less}")
                            
                            # Step 1: Extract particles with new box size
                            # Use refinement job for particle coordinates (refined positions/orientations)
                            self.logger.info(f"📦 Step 1/3: Extracting particles with box_size {box_size_less}...")
                            extract_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": refinement_job_uid,  # Use refinement job for refined coordinates
                                "micrographs_job_uid": micrographs_job_uid,
                                "box_size_pix": box_size_less
                            }
                            self._record_tool_execution("extract_particles", extract_params)
                            extract_result = self.cryosparc_tools.extract_particles(
                                **extract_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("extract_particles", extract_params, result=extract_result)
                            
                            # Verify extraction completed successfully
                            extract_status = extract_result.get("status", "unknown")
                            if extract_status != "completed":
                                error_msg = extract_result.get("error") or f"Status: {extract_status}"
                                self.logger.error(f"❌ Particle extraction failed for box_size {box_size_less}: {error_msg}")
                                # Don't proceed to refinement if extraction failed
                                continue
                            
                            extract_job_uid = extract_result["job_uid"]
                            self.logger.info(f"✅ Step 1/3: Extraction completed for box_size {box_size_less}, job: {extract_job_uid}")
                            
                            # Step 2: Run homogeneous refinement (wait for completion)
                            self.logger.info(f"🔧 Step 2/3: Starting refinement for box_size {box_size_less}...")
                            # Get symmetry from reconstruction_config.json
                            symmetry = self._get_refinement_symmetry()
                            refine_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": extract_job_uid,
                                "volume_job_uid": volume_job_uid,
                                "symmetry": symmetry
                            }
                            self._record_tool_execution("homogeneous_refinement", refine_params)
                            refine_result = self.cryosparc_tools.homogeneous_refinement(
                                **refine_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("homogeneous_refinement", refine_params, result=refine_result)
                            
                            # Verify refinement completed successfully
                            # homogeneous_refinement returns "success" and updates with status_result
                            if not refine_result.get("success", False):
                                error_msg = refine_result.get("error") or "Unknown error"
                                self.logger.error(f"❌ Refinement failed for box_size {box_size_less}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement failed
                                continue
                            
                            refine_status = refine_result.get("status", "unknown")
                            if refine_status != "completed":
                                error_msg = refine_result.get("error") or f"Status: {refine_status}"
                                self.logger.error(f"❌ Refinement did not complete for box_size {box_size_less}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement didn't complete
                                continue
                            
                            refine_job_uid = refine_result["job_uid"]
                            self.logger.info(f"✅ Step 2/3: Refinement completed for box_size {box_size_less}, job: {refine_job_uid}")
                            
                            # Step 3: Get FSC info and resolution
                            self.logger.info(f"📊 Step 3/3: Getting FSC resolution for box_size {box_size_less}...")
                            fsc_info_less = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
                            if fsc_info_less.get("success"):
                                resolution = fsc_info_less["resolution_angstroms"]
                                self.logger.info(f"✅ Step 3/3: Box size {box_size_less}: Resolution = {resolution} Å")
                                tested_data.append({
                                    "box_size": box_size_less,
                                    "resolution": resolution,
                                    "job_uid": refine_job_uid
                                })
                                self.logger.info(f"✅ Completed testing box_size {box_size_less}: Resolution = {resolution} Å")
                            else:
                                self.logger.error(f"❌ Failed to get FSC info for box_size {box_size_less}: {fsc_info_less.get('error')}")
                        except Exception as e:
                            # Continue with other tests even if one fails
                            self.logger.warning(f"⚠️ Failed to test box_size_less {box_size_less}: {e}", exc_info=True)
                            pass
                
                # Test box_size_more
                if box_size_more:
                    # Check if already tested
                    if box_size_more in [d["box_size"] for d in tested_data]:
                        self.logger.info(f"Skipping box_size_more {box_size_more} - already tested")
                    else:
                        try:
                            self.logger.info(f"🔬 Testing box_size_more: {box_size_more}")
                            
                            # Step 1: Extract particles with new box size
                            # Use refinement job for particle coordinates (refined positions/orientations)
                            self.logger.info(f"📦 Step 1/3: Extracting particles with box_size {box_size_more}...")
                            extract_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": refinement_job_uid,  # Use refinement job for refined coordinates
                                "micrographs_job_uid": micrographs_job_uid,
                                "box_size_pix": box_size_more
                            }
                            self._record_tool_execution("extract_particles", extract_params)
                            extract_result = self.cryosparc_tools.extract_particles(
                                **extract_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("extract_particles", extract_params, result=extract_result)
                            
                            # Verify extraction completed successfully
                            extract_status = extract_result.get("status", "unknown")
                            if extract_status != "completed":
                                error_msg = extract_result.get("error") or f"Status: {extract_status}"
                                self.logger.error(f"❌ Particle extraction failed for box_size {box_size_more}: {error_msg}")
                                # Don't proceed to refinement if extraction failed
                                continue
                            
                            extract_job_uid = extract_result["job_uid"]
                            self.logger.info(f"✅ Step 1/3: Extraction completed for box_size {box_size_more}, job: {extract_job_uid}")
                            
                            # Step 2: Run homogeneous refinement (wait for completion)
                            self.logger.info(f"🔧 Step 2/3: Starting refinement for box_size {box_size_more}...")
                            # Get symmetry from reconstruction_config.json
                            symmetry = self._get_refinement_symmetry()
                            refine_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": extract_job_uid,
                                "volume_job_uid": volume_job_uid,
                                "symmetry": symmetry
                            }
                            self._record_tool_execution("homogeneous_refinement", refine_params)
                            refine_result = self.cryosparc_tools.homogeneous_refinement(
                                **refine_params,
                                wait_for_completion=True,
                                timeout=self.config.job_management.default_timeout,
                                check_interval=self.config.job_management.status_check_interval
                            )
                            self._record_tool_execution("homogeneous_refinement", refine_params, result=refine_result)
                            
                            # Verify refinement completed successfully
                            # homogeneous_refinement returns "success" and updates with status_result
                            if not refine_result.get("success", False):
                                error_msg = refine_result.get("error") or "Unknown error"
                                self.logger.error(f"❌ Refinement failed for box_size {box_size_more}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement failed
                                continue
                            
                            refine_status = refine_result.get("status", "unknown")
                            if refine_status != "completed":
                                error_msg = refine_result.get("error") or f"Status: {refine_status}"
                                self.logger.error(f"❌ Refinement did not complete for box_size {box_size_more}: {error_msg}")
                                # Don't proceed to FSC extraction if refinement didn't complete
                                continue
                            
                            refine_job_uid = refine_result["job_uid"]
                            self.logger.info(f"✅ Step 2/3: Refinement completed for box_size {box_size_more}, job: {refine_job_uid}")
                            
                            # Step 3: Get FSC info and resolution
                            self.logger.info(f"📊 Step 3/3: Getting FSC resolution for box_size {box_size_more}...")
                            fsc_info_more = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
                            if fsc_info_more.get("success"):
                                resolution = fsc_info_more["resolution_angstroms"]
                                self.logger.info(f"✅ Step 3/3: Box size {box_size_more}: Resolution = {resolution} Å")
                                tested_data.append({
                                    "box_size": box_size_more,
                                    "resolution": resolution,
                                    "job_uid": refine_job_uid
                                })
                                self.logger.info(f"✅ Completed testing box_size {box_size_more}: Resolution = {resolution} Å")
                            else:
                                self.logger.error(f"❌ Failed to get FSC info for box_size {box_size_more}: {fsc_info_more.get('error')}")
                        except Exception as e:
                            # Continue with other tests even if one fails
                            self.logger.warning(f"⚠️ Failed to test box_size_more {box_size_more}: {e}", exc_info=True)
                            pass
                
                # After testing, sort by box size for next iteration's analysis
                tested_data.sort(key=lambda x: x["box_size"])
            
            # Find final best result
            best_result = min(tested_data, key=lambda x: x["resolution"])
            
            result = {
                "success": True,
                "optimization_complete": True,
                "best_box_size": best_result["box_size"],
                "best_resolution_angstroms": best_result["resolution"],
                "best_job_uid": best_result["job_uid"],
                "tested_combinations": tested_data,
                "iterations": iteration
            }
            
            self._record_tool_execution("optimize_diameter", params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("optimize_diameter", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _get_hetero_class_resolutions_tool(self, tool_input: str) -> str:
        """
        Get resolution information for each class in a heterogeneous refinement job.
        
        Can accept either:
        - Just the job UID as a string (e.g., "JXXX")
        - JSON with job_uid parameter
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Support both direct job UID string and JSON parameter
            job_uid = params.get("job_uid") or params.get("hetero_job_uid") or params.get("refinement_job_uid")
            
            # If still not found, try to extract from input string directly
            if not job_uid:
                input_stripped = tool_input.strip().strip('"\'')
                if input_stripped.startswith("J") and len(input_stripped) <= 10:
                    job_uid = input_stripped
                elif "input" in params:
                    input_val = str(params["input"]).strip().strip('"\'')
                    if input_val.startswith("J") and len(input_val) <= 10:
                        job_uid = input_val
            
            if not job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: job_uid. You can pass just the job UID (e.g., 'J357') or JSON with job_uid parameter."
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            
            # Get class resolutions from the heterogeneous refinement job
            class_resolutions = self.cryosparc_tools.get_heterogeneous_refinement_class_resolutions(project_uid, job_uid)
            
            if not class_resolutions.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get class resolutions: {class_resolutions.get('error', 'Unknown error')}"
                })
            
            result = {
                "success": True,
                "job_uid": job_uid,
                "num_classes": class_resolutions.get("num_classes", 0),
                "classes": class_resolutions.get("classes", [])
            }
            
            self._record_tool_execution("get_hetero_class_resolutions", {"job_uid": job_uid, "project_uid": project_uid}, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("get_hetero_class_resolutions", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _test_heterogeneous_refinement_tool(self, tool_input: str) -> str:
        """
        Test heterogeneous refinement with K classes.
        
        This tool:
        1. Repeats the volume from refinement_job_uid K times as initial densities
        2. Runs heterogeneous refinement using particles from refinement_job_uid
        3. Gets resolution for each class
        4. Selects best class (smallest resolution value, or HIGHEST fsc_loosemask_last if tied - higher FSC is better)
        5. Runs homogeneous refinement on selected class particles
        6. Gets final FSC resolution
        """
        try:
            # Handle case where tool_input might be a tuple/list (from LangChain parsing issues)
            if isinstance(tool_input, (list, tuple)):
                # If it's passed as a list/tuple [k, job_uid], convert to dict
                if len(tool_input) >= 2:
                    params = {"k": tool_input[0], "refinement_job_uid": tool_input[1]}
                else:
                    return json.dumps({
                        "success": False,
                        "error": "Invalid input format. Expected JSON string like {'k': 3, 'refinement_job_uid': 'J357'} or list with [k, job_uid]"
                    })
            elif isinstance(tool_input, dict):
                # Already a dict, use directly
                params = tool_input
            else:
                # Parse string input
                params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            k = params.get("k") or params.get("num_classes")
            refinement_job_uid = params.get("refinement_job_uid")
            
            # Try to extract from list/tuple format if still not found
            if not k or not refinement_job_uid:
                if isinstance(tool_input, (list, tuple)) and len(tool_input) >= 2:
                    k = tool_input[0]
                    refinement_job_uid = tool_input[1]
            
            if not k or not refinement_job_uid:
                missing = []
                if not k:
                    missing.append("k (number of classes)")
                if not refinement_job_uid:
                    missing.append("refinement_job_uid")
                return json.dumps({
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing)}. Expected JSON format: {{'k': 3, 'refinement_job_uid': 'J357'}}"
                })
            
            k = int(k)
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            self.logger.info(f"🔬 Testing heterogeneous refinement with K={k}")
            
            # Step 1: Run heterogeneous refinement
            # For heterogeneous refinement, we need to repeat the same volume K times
            # The volume comes from refinement_job_uid
            volume_job_uids = [refinement_job_uid] * k  # Repeat the same volume K times
            
            # Get symmetry from reconstruction_config.json (workflow.refinement.symmetry)
            symmetry = self._get_refinement_symmetry()
            
            self.logger.info(f"📦 Step 1/5: Running heterogeneous refinement with K={k} (repeating volume {refinement_job_uid} {k} times) using symmetry {symmetry}...")
            hetero_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": refinement_job_uid,  # Use particles from refinement job
                "volume_job_uids": volume_job_uids,
                "num_classes": k,
                "symmetry": symmetry  # Use symmetry from reconstruction_config.json
            }
            self._record_tool_execution("heterogeneous_refinement", hetero_params)
            hetero_result = self.cryosparc_tools.heterogeneous_refinement(
                **hetero_params,
                wait_for_completion=True,
                timeout=self.config.job_management.default_timeout,
                check_interval=self.config.job_management.status_check_interval
            )
            self._record_tool_execution("heterogeneous_refinement", hetero_params, result=hetero_result)
            
            # Verify heterogeneous refinement completed successfully
            if not hetero_result.get("success", False):
                error_msg = hetero_result.get("error") or "Unknown error"
                self.logger.error(f"❌ Heterogeneous refinement failed for K={k}: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Heterogeneous refinement failed: {error_msg}",
                    "k": k
                })
            
            hetero_status = hetero_result.get("status", "unknown")
            if hetero_status != "completed":
                error_msg = hetero_result.get("error") or f"Status: {hetero_status}"
                self.logger.error(f"❌ Heterogeneous refinement did not complete for K={k}: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Heterogeneous refinement did not complete: {error_msg}",
                    "k": k
                })
            
            hetero_job_uid = hetero_result["job_uid"]
            self.logger.info(f"✅ Step 1/5: Heterogeneous refinement completed for K={k}, job: {hetero_job_uid}")
            
            # Step 2: Get class resolutions
            self.logger.info(f"📊 Step 2/5: Getting class resolutions for K={k}...")
            class_resolutions = self.cryosparc_tools.get_heterogeneous_refinement_class_resolutions(project_uid, hetero_job_uid)
            if not class_resolutions.get("success"):
                error_msg = class_resolutions.get("error", "Unknown error")
                self.logger.error(f"❌ Failed to get class resolutions for K={k}: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get class resolutions: {error_msg}",
                    "k": k,
                    "hetero_job_uid": hetero_job_uid
                })
            
            classes = class_resolutions.get("classes", [])
            if not classes:
                return json.dumps({
                    "success": False,
                    "error": "No classes found in heterogeneous refinement result",
                    "k": k,
                    "hetero_job_uid": hetero_job_uid
                })
            
            # Step 3: Select best class
            # Best = smallest resolution_angstroms value (lower is better), or if tied, HIGHEST fsc_loosemask_last (higher is better)
            # Use negative fsc_loosemask_last for tie-breaking so min() selects the highest value
            best_class = min(classes, key=lambda c: (
                c.get("resolution_angstroms", float('inf')),
                -(c.get("fsc_loosemask_last") if c.get("fsc_loosemask_last") is not None else -float('inf'))
            ))
            best_class_id = best_class["class_id"]
            best_class_resolution = best_class["resolution_angstroms"]
            best_fsc_last = best_class.get("fsc_loosemask_last")
            best_group_name = best_class["group_name"]
            
            # Build reasoning explanation for class selection
            class_comparison = []
            for cls in classes:
                cls_id = cls["class_id"]
                cls_res = cls["resolution_angstroms"]
                cls_fsc = cls.get("fsc_loosemask_last")
                class_comparison.append({
                    "class_id": cls_id,
                    "resolution_angstroms": cls_res,
                    "fsc_loosemask_last": cls_fsc
                })
            
            # Determine selection reason
            same_resolution_classes = [c for c in classes if c.get("resolution_angstroms") == best_class_resolution]
            if len(same_resolution_classes) > 1:
                selection_reason = f"Selected class {best_class_id} because it has the highest fsc_loosemask_last ({best_fsc_last}) among {len(same_resolution_classes)} classes with the same resolution ({best_class_resolution} Å)"
            else:
                selection_reason = f"Selected class {best_class_id} because it has the best (smallest) resolution ({best_class_resolution} Å)"
            
            self.logger.info(f"✅ Step 2/5: Found {len(classes)} classes. {selection_reason}")
            self.logger.info(f"📊 Class comparison: {class_comparison}")
            
            # Step 4: Run homogeneous refinement on selected class particles
            # The particles for the selected class come from the heterogeneous refinement job
            # The volume comes from the selected class in the heterogeneous refinement job
            self.logger.info(f"🔧 Step 3/5: Running homogeneous refinement on best class {best_class_id} particles...")
            symmetry = self._get_refinement_symmetry()
            
            # For homogeneous refinement from heterogeneous refinement, we need:
            # - particles_job_uid: the heterogeneous refinement job, with group_name = particles_class_X
            # - volume_job_uid: the heterogeneous refinement job, with group_name = volume_class_X
            particles_group_name = f"particles_class_{best_class_id}"
            volume_group_name = best_group_name  # Already set to volume_class_X
            
            refine_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": hetero_job_uid,
                "volume_job_uid": hetero_job_uid,
                "symmetry": symmetry,
                "particles_group_name": particles_group_name,  # Pass via kwargs
                "volume_group_name": volume_group_name  # Pass via kwargs
            }
            
            self._record_tool_execution("homogeneous_refinement", refine_params)
            refine_result = self.cryosparc_tools.homogeneous_refinement(
                **refine_params,
                wait_for_completion=True,
                timeout=self.config.job_management.default_timeout,
                check_interval=self.config.job_management.status_check_interval
            )
            self._record_tool_execution("homogeneous_refinement", refine_params, result=refine_result)
            
            # Verify refinement completed successfully
            if not refine_result.get("success", False):
                error_msg = refine_result.get("error") or "Unknown error"
                self.logger.error(f"❌ Homogeneous refinement failed for class {best_class_id}: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Homogeneous refinement failed: {error_msg}",
                    "k": k,
                    "hetero_job_uid": hetero_job_uid,
                    "best_class_id": best_class_id
                })
            
            refine_status = refine_result.get("status", "unknown")
            if refine_status != "completed":
                error_msg = refine_result.get("error") or f"Status: {refine_status}"
                self.logger.error(f"❌ Homogeneous refinement did not complete for class {best_class_id}: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Homogeneous refinement did not complete: {error_msg}",
                    "k": k,
                    "hetero_job_uid": hetero_job_uid,
                    "best_class_id": best_class_id
                })
            
            refine_job_uid = refine_result["job_uid"]
            self.logger.info(f"✅ Step 3/5: Homogeneous refinement completed for class {best_class_id}, job: {refine_job_uid}")
            
            # Step 5: Get final FSC resolution
            self.logger.info(f"📊 Step 4/5: Getting final FSC resolution for class {best_class_id}...")
            fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
            if not fsc_info.get("success"):
                error_msg = fsc_info.get("error", "Unknown error")
                self.logger.error(f"❌ Failed to get FSC info for class {best_class_id}: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get FSC info: {error_msg}",
                    "k": k,
                    "hetero_job_uid": hetero_job_uid,
                    "best_class_id": best_class_id,
                    "refine_job_uid": refine_job_uid
                })
            
            final_resolution = fsc_info["resolution_angstroms"]
            self.logger.info(f"✅ Step 4/5: Final resolution for K={k}, class {best_class_id}: {final_resolution} Å")
            
            result = {
                "success": True,
                "k": k,
                "hetero_job_uid": hetero_job_uid,
                "best_class_id": best_class_id,
                "best_class_resolution": best_class_resolution,
                "best_class_fsc_last": best_fsc_last,
                "class_selection_reason": selection_reason,
                "class_comparison": class_comparison,
                "refine_job_uid": refine_job_uid,
                "final_resolution_angstroms": final_resolution,
                "all_classes": classes
            }
            
            self._record_tool_execution("test_heterogeneous_refinement", params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("test_heterogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _reason_about_workflow_tool(self, input_str: str) -> str:
        """Tool for reasoning about optimization workflow state."""
        try:
            reasoning = f"""
🤔 **Box Size Optimization Workflow Analysis**:

**Current State**: {input_str}

**Agentic Optimization Process**:
1. **Initial Assessment**: Use `get_fsc_info` to get baseline resolution from original refinement job
2. **First Round**: Test 10% less and 10% more box sizes using `test_box_size`
3. **Analysis**: Compare three results (original, -10%, +10%) and identify trends
4. **Iterative Refinement**: Based on analysis, decide next box size to test
5. **Continue**: Keep testing until optimal found or stopping condition reached

**Key Tools**:
- **get_fsc_info**: Get FSC resolution and box size from a refinement job
- **test_box_size**: Test a specific box size (extract + refine + get FSC)

**Parameters Needed**:
- **refinement_job_uid**: First refinement job (source of refined coordinates and baseline)
- **micrographs_job_uid**: Micrographs for re-extraction
- **volume_job_uid**: Initial volume for refinement

**Optimization Strategy**:
- Start with baseline from original refinement
- Test 10% less and 10% more initially
- Analyze trends: Which direction improves resolution?
- Test further in promising direction
- Stop after 5-7 tests or when optimal point is clear

**Decision Making**:
- Smaller resolution_angstroms value = better quality (e.g., 3.0 Å is better than 5.0 Å)
- If middle is best: Test between middle and better extreme
- If extreme is best: Test further in that direction
- Consider refinement_resolution parameter if needed

**Next Steps**:
- If not started: Get baseline FSC info, then test ±10%
- If testing: Analyze results and decide next box size
- If complete: Summarize best box size and resolution
"""
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, result={"analysis": reasoning})
            return reasoning
            
        except Exception as e:
            self._record_tool_execution("reason_about_workflow", {"input": input_str}, error=str(e))
            return f"❌ Error in workflow reasoning: {str(e)}"


