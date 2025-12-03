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
            OptimizerTools.create_test_heterogeneous_refinement_tool(self),
            OptimizerTools.create_test_multi_round_3d_classification_tool(self)
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
    
    def _should_use_nonuniform_refinement(self) -> bool:
        """Check if non-uniform refinement should be used instead of homogeneous refinement."""
        return self.stage_workflow.get("optimization", {}).get("use_nonuniform_refinement", True)
    
    def _get_refinement_res_init(self) -> Optional[float]:
        """Get initial lowpass resolution for refinement from optimization config."""
        return self._get_stage_param("optimization", "refine_res_init", None)
    
    def _get_stage_param(self, section: str, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Fetch a parameter from the stage workflow configuration."""
        return self.stage_workflow.get(section, {}).get(key, default)
    
    def _get_react_system_prompt(self) -> str:
        """Get the optimization-specific ReAct system prompt."""
        # Check if heterogeneous refinement is enabled
        enable_box_size = self._get_stage_param("optimization", "enable_box_size_optimization", True)
        enable_hetero = self._get_stage_param("optimization", "enable_heterogeneous_refinement", False)
        max_hetero_iterations = self._get_stage_param("optimization", "heterogeneous_refinement_max_iterations", 3)
        enable_multi_round = self._get_stage_param("optimization", "enable_multi_round_3d_classification", False)
        multi_round_num_classes = self._get_stage_param("optimization", "multi_round_3d_classification_num_classes", 4)
        multi_round_max_rounds = self._get_stage_param("optimization", "multi_round_3d_classification_max_rounds", 5)
        multi_round_improvement_threshold = self._get_stage_param("optimization", "multi_round_3d_classification_improvement_threshold", 0.1)
        
        # Determine what to optimize
        optimization_types = []
        if enable_box_size:
            optimization_types.append("box size/diameter")
        if enable_hetero:
            optimization_types.append("heterogeneous refinement (K values)")
        if enable_multi_round:
            optimization_types.append("multi-round 3D classification")
        
        optimization_desc = " and ".join(optimization_types) if optimization_types else "parameters"
        
        # Build box size optimization section conditionally
        box_size_section = ""
        if enable_box_size:
            box_size_section = """
## Box Size Optimization Workflow:

**Purpose**: Optimize the box size to achieve the best resolution. This should be done AFTER heterogeneous refinement optimization (if heterogeneous refinement is enabled), otherwise after the first round of 3D homogeneous refinement.

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
{f'**IMPORTANT**: Both heterogeneous refinement and box size optimization are enabled. Complete heterogeneous refinement (K optimization) FIRST, then use the optimized refinement job for box size optimization.**' if enable_box_size and enable_hetero else ''}
{f'**CRITICAL: Box size optimization is DISABLED. DO NOT use test_box_size tool. Proceed directly to heterogeneous refinement using the refinement_job_uid provided.**' if not enable_box_size and enable_hetero else ''}
{f'**Heterogeneous refinement is DISABLED**. Proceed with box size optimization only.**' if enable_box_size and not enable_hetero else ''}
{box_size_section}
## General Tool Usage:

- **get_fsc_info**: Get FSC resolution and box size from a refinement job
  * Required: refinement_job_uid
  * Returns: box_size (pixels), resolution_angstroms (FSC resolution)
  * Use this to get baseline information from any refinement job

- **get_job_status**: Check status of a specific job (use job UID only, e.g., "JXXX")
- **wait_for_job**: Wait for job completion (use job UID only, e.g., "JXXX")
- **get_job_log**: Read and analyze job logs
- **reason_about_workflow**: Analyze current optimization state and think about next steps

## Job UID Format:
- Job UIDs are strings like "JXXX", "JYYY", etc.
- When calling get_job_status, wait_for_job, or get_fsc_info, you can pass ONLY the job UID (e.g., "JXXX")
- For other tools, use JSON format with parameter names

## Current Configuration:
- Project UID: {self.config.workflow.project_uid}
- Workspace UID: {self.config.workflow.workspace_uid}
- Box size optimization: {'ENABLED' if enable_box_size else 'DISABLED'}
- Heterogeneous refinement: {'ENABLED' if enable_hetero else 'DISABLED'}
- Multi-round 3D classification: {'ENABLED' if enable_multi_round else 'DISABLED'}

## Heterogeneous Refinement Optimization Workflow:
{'' if not enable_hetero else f'''

**Purpose**: Optimize the number of classes (K) in heterogeneous refinement to achieve the best resolution. This should be done BEFORE box size optimization (if box size optimization is enabled).

**Agentic Optimization Process**:
1. **Baseline**: Get FSC resolution from the final refinement job (K=1, which is homogeneous refinement)
2. **First Round Testing**: Test K=2 and K=3 using `test_heterogeneous_refinement` tool
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
  * **Input format: JSON string** (e.g., `{{"k": 3, "refinement_job_uid": "JXXX"}}`)
  * Required parameters: k (number of classes, e.g., 3 or 5), refinement_job_uid (source of particles and volume, e.g., "JXXX")
  * This tool: 1) Repeats the volume from refinement_job_uid K times, 2) Runs heterogeneous refinement,
    3) Gets resolution for each class, 4) Selects best class (smallest resolution value, or HIGHEST fsc_loosemask_last if tied - higher FSC is better), 
    5) Runs homogeneous refinement on selected class, 6) Returns final FSC resolution
  * Returns: hetero_job_uid, best_class_id, best_class_resolution, class_selection_reason, class_comparison (all classes data), refine_job_uid, final_resolution_angstroms, and all_classes
  * **Important**: The tool automatically selects the best class using an algorithm, but you should REASON about the class_comparison and class_selection_reason to verify the selection makes sense and understand why it was chosen.
  * **Example**: Use JSON format: `{{"k": 3, "refinement_job_uid": "JXXX"}}` or `{{"k": 5, "refinement_job_uid": "JXXX"}}`

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
- Test K=2 and K=3 in the first round
- Compare K=1, K=2, and K=3 to see the trend

**REASONING REQUIREMENT - After Each Test**:
**CRITICAL**: After calling `test_heterogeneous_refinement`, you MUST actively reason about:
1. **Class Selection Analysis**: Review the `class_comparison` data and `class_selection_reason` to understand which class was selected and why
   - Are all classes in the heterogeneous refinement similar in quality?
   - Was there a clear winner, or were classes close in resolution?
   - Does the selected class seem reasonable given all available data?

2. **K Value Comparison**: Compare the final resolution from this K value with previous K values tested
   - Which K value has given the best resolution so far?
   - Is there a clear trend (e.g., increasing K improves resolution, or vice versa or choose the value in between)?

3. **Decision Making**: Based on your analysis, decide:
   - **Which K to test next** (if any): If there's a clear trend, test further in that direction. If optimal seems to be between tested values, test intermediate values.
   - **Whether to STOP**: Consider stopping if:
     * Resolution is getting worse with more K values test
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
- Correct: Action Input: `{{"k": 3, "refinement_job_uid": "JXXX"}}`
- Wrong: test_heterogeneous_refinement(3, "JXXX") - this will fail!
- The tool requires a single JSON string input, not multiple arguments
'''}

## Multi-Round 3D Classification Optimization Workflow:

**Purpose**: Iteratively refine 3D structures using multi-round 3D classification to achieve the best resolution. This should be done FIRST, before heterogeneous refinement optimization (if enabled) and box size optimization (if enabled), using the initial homogeneous refinement result.

**Agentic Optimization Process**:
1. **Input**: Take volume and particles_job_uid from previous best homogeneous refinement
2. **3D Classification**: Run 3D classification (heterogeneous refinement) with 4 classes (default)
3. **Class Selection**: Select the best class based on resolution metric (lowest resolution is best)
4. **3D Refinement**: Run 3D refinement (homogeneous refinement) on the selected class using volume and particles from that class
5. **Resolution Check**: Check if resolution improved compared to previous round
6. **Decision Making**:
   - **If improved**: Continue the process using the refined result as input for the next round
   - **If plateau or worse**: Stop the process and return the best refinement job
7. **Iterative Process**: Repeat steps until:
   - Resolution plateaus or worsens
   - Maximum number of rounds is reached
   - Further rounds are unlikely to improve results

**Tool Usage for Multi-Round 3D Classification**:

- **test_multi_round_3d_classification**: Run multi-round 3D classification optimization
  * **Input format: JSON string** (e.g., `{{"refinement_job_uid": "JXXX", "num_classes": 4, "max_rounds": 5}}`)
  * Required parameters: refinement_job_uid (source of particles and volume from previous best homogeneous refinement, e.g., "JXXX")
  * Optional parameters: num_classes (number of classes for 3D classification, default: 4), max_rounds (maximum number of rounds, default: 5), improvement_threshold (minimum improvement in resolution in Å to continue, default: 0.1)
  * This tool automatically:
    1. Gets initial resolution from refinement_job_uid
    2. For each round: Runs 3D classification → Selects best class → Runs refinement → Checks improvement
    3. Stops when resolution plateaus/worsens or max_rounds reached
    4. Returns best_refinement_job_uid, best_resolution_angstroms, rounds_completed, and all_rounds_data
  * Returns: best_refinement_job_uid, best_resolution_angstroms, initial_resolution_angstroms, total_improvement, rounds_completed, all_rounds_data (detailed data for each round)
  * **Example**: Use JSON format: `{{"refinement_job_uid": "JXXX", "num_classes": 4, "max_rounds": 5}}`

**Multi-Round 3D Classification Strategy Guidelines**:

**When to Use**:
- FIRST, before heterogeneous refinement optimization (if enabled)
- FIRST, before box size optimization (if enabled)
- After the first round of 3D homogeneous refinement (use the initial refinement_job_uid)
- When you want to iteratively refine structures through multiple rounds of classification

**Parameters**:
- **num_classes**: Number of classes for 3D classification (default: 4). More classes may help identify better structures but take longer.
- **max_rounds**: Maximum number of rounds to run (default: 5). Each round includes classification and refinement.
- **improvement_threshold**: Minimum improvement in resolution (Å) to continue (default: 0.1). If improvement is less than this, the process may stop.

**Stopping Conditions**:
- Resolution plateaus or worsens (no improvement or worse resolution)
- Maximum number of rounds reached
- Improvement is below threshold for multiple consecutive rounds

**Understanding Results**:
- **best_refinement_job_uid**: The job UID of the best refinement result (use this for next stages)
- **best_resolution_angstroms**: The best resolution achieved (lower is better)
- **total_improvement**: Total improvement from initial to best resolution (positive means improvement)
- **rounds_completed**: Number of rounds that were completed
- **all_rounds_data**: Detailed information for each round including class selections and resolutions

**CRITICAL: When calling test_multi_round_3d_classification, you MUST use JSON format with Action Input!**
- Correct: Action Input: `{{"refinement_job_uid": "JXXX", "num_classes": 4, "max_rounds": 5}}`
- Wrong: test_multi_round_3d_classification("JXXX", 4, 5) - this will fail!
- The tool requires a single JSON string input, not multiple arguments
{'' if not enable_multi_round else f'''

**Multi-Round 3D Classification Configuration**:
- Number of classes per round: {multi_round_num_classes}
- Maximum rounds: {multi_round_max_rounds}
- Improvement threshold: {multi_round_improvement_threshold} Å
- **IMPORTANT**: Multi-round 3D classification is ENABLED. This should be done FIRST, before heterogeneous refinement (if enabled) and box size optimization (if enabled).
'''}

## Combined Workflow:
{'' if not (enable_box_size or enable_hetero or enable_multi_round) else '''
**Optimization Priority Order**:
1. **First**: Complete multi-round 3D classification (if enabled) - iteratively refine structures through multiple rounds
2. **Second**: Use the best refinement job from step 1 (or initial refinement if step 1 disabled) for heterogeneous refinement optimization (if enabled) - optimize K values, get best K and refinement job
3. **Third**: Use the best refinement job from step 2 (or step 1, or initial refinement) for box size optimization (if enabled)
4. **Finally**: Report all optimizations' results

**Example Combined Flow** (if all enabled):
1. Run multi-round 3D classification → Get best multi-round refinement result (e.g., JXXX)
2. Use JXXX as refinement_job_uid for heterogeneous refinement → Get best K and refinement job (e.g., JYYY)
3. Use JYYY as refinement_job_uid for box size optimization → Get best box size refinement result (e.g., JZZZ)
4. Report: best_multi_round_resolution, best_hetero_k, best_hetero_resolution, best_box_size, best_box_resolution

**If only multi-round 3D classification is enabled**:
- Use the refinement_job_uid provided directly for multi-round 3D classification
- Report: best_multi_round_refinement_job_uid, best_multi_round_resolution, rounds_completed
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
            # _parse_tool_input converts "JXXX" to {"job_uid": "JXXX"}
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
                    "error": "Missing required parameter: refinement_job_uid. You can pass just the job UID (e.g., 'JXXX') or JSON with refinement_job_uid parameter."
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
            
            # Step 2: Run refinement (non-uniform or homogeneous based on config)
            use_nonuniform = self._should_use_nonuniform_refinement()
            refinement_type = "non-uniform" if use_nonuniform else "homogeneous"
            self.logger.info(f"🔧 Step 2/3: Starting {refinement_type} refinement for box_size {box_size_pix}...")
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
            # Add initial lowpass resolution if configured
            refine_res_init = self._get_refinement_res_init()
            if refine_res_init is not None:
                refine_params["refine_res_init"] = float(refine_res_init)
            
            tool_name = "nonuniform_refine_new" if use_nonuniform else "homogeneous_refinement"
            self._record_tool_execution(tool_name, refine_params)
            if use_nonuniform:
                refine_result = self.cryosparc_tools.nonuniform_refine_new(
                    **refine_params,
                    wait_for_completion=True,
                    timeout=self.config.job_management.default_timeout,
                    check_interval=self.config.job_management.status_check_interval
                )
            else:
                refine_result = self.cryosparc_tools.homogeneous_refinement(
                    **refine_params,
                    wait_for_completion=True,
                    timeout=self.config.job_management.default_timeout,
                    check_interval=self.config.job_management.status_check_interval
                )
            self._record_tool_execution(tool_name, refine_params, result=refine_result)
            
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
                            
                            # Step 2: Run refinement (non-uniform or homogeneous based on config)
                            use_nonuniform = self._should_use_nonuniform_refinement()
                            refinement_type = "non-uniform" if use_nonuniform else "homogeneous"
                            self.logger.info(f"🔧 Step 2/3: Starting {refinement_type} refinement for box_size {box_size_less}...")
                            # Get symmetry from reconstruction_config.json
                            symmetry = self._get_refinement_symmetry()
                            refine_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": extract_job_uid,
                                "volume_job_uid": volume_job_uid,
                                "symmetry": symmetry
                            }
                            # Add initial lowpass resolution if configured
                            refine_res_init = self._get_refinement_res_init()
                            if refine_res_init is not None:
                                refine_params["refine_res_init"] = float(refine_res_init)
                            tool_name = "nonuniform_refine_new" if use_nonuniform else "homogeneous_refinement"
                            self._record_tool_execution(tool_name, refine_params)
                            if use_nonuniform:
                                refine_result = self.cryosparc_tools.nonuniform_refine_new(
                                    **refine_params,
                                    wait_for_completion=True,
                                    timeout=self.config.job_management.default_timeout,
                                    check_interval=self.config.job_management.status_check_interval
                                )
                            else:
                                refine_result = self.cryosparc_tools.homogeneous_refinement(
                                    **refine_params,
                                    wait_for_completion=True,
                                    timeout=self.config.job_management.default_timeout,
                                    check_interval=self.config.job_management.status_check_interval
                                )
                            self._record_tool_execution(tool_name, refine_params, result=refine_result)
                            
                            # Verify refinement completed successfully
                            # refinement returns "success" and updates with status_result
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
                            
                            # Step 2: Run refinement (non-uniform or homogeneous based on config)
                            use_nonuniform = self._should_use_nonuniform_refinement()
                            refinement_type = "non-uniform" if use_nonuniform else "homogeneous"
                            self.logger.info(f"🔧 Step 2/3: Starting {refinement_type} refinement for box_size {box_size_more}...")
                            # Get symmetry from reconstruction_config.json
                            symmetry = self._get_refinement_symmetry()
                            refine_params = {
                                "project_uid": project_uid,
                                "workspace_uid": workspace_uid,
                                "particles_job_uid": extract_job_uid,
                                "volume_job_uid": volume_job_uid,
                                "symmetry": symmetry
                            }
                            # Add initial lowpass resolution if configured
                            refine_res_init = self._get_refinement_res_init()
                            if refine_res_init is not None:
                                refine_params["refine_res_init"] = float(refine_res_init)
                            tool_name = "nonuniform_refine_new" if use_nonuniform else "homogeneous_refinement"
                            self._record_tool_execution(tool_name, refine_params)
                            if use_nonuniform:
                                refine_result = self.cryosparc_tools.nonuniform_refine_new(
                                    **refine_params,
                                    wait_for_completion=True,
                                    timeout=self.config.job_management.default_timeout,
                                    check_interval=self.config.job_management.status_check_interval
                                )
                            else:
                                refine_result = self.cryosparc_tools.homogeneous_refinement(
                                    **refine_params,
                                    wait_for_completion=True,
                                    timeout=self.config.job_management.default_timeout,
                                    check_interval=self.config.job_management.status_check_interval
                                )
                            self._record_tool_execution(tool_name, refine_params, result=refine_result)
                            
                            # Verify refinement completed successfully
                            # refinement returns "success" and updates with status_result
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
                    "error": "Missing required parameter: job_uid. You can pass just the job UID (e.g., 'JXXX') or JSON with job_uid parameter."
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
        3. Runs regroup to regroup K classes into 2 superclasses (job name: regroup_3D_new)
        4. Gets num_items for each superclass from regroup job.json
        5. Selects the superclass with more particles
        6. Runs homogeneous refinement on selected superclass particles and volumes
        7. Gets final FSC resolution
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
                        "error": "Invalid input format. Expected JSON string like {'k': 3, 'refinement_job_uid': 'JXXX'} or list with [k, job_uid]"
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
                    "error": f"Missing required parameters: {', '.join(missing)}. Expected JSON format: {{'k': 3, 'refinement_job_uid': 'JXXX'}}"
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
            
            self.logger.info(f"📦 Step 1/7: Running heterogeneous refinement with K={k} (repeating volume {refinement_job_uid} {k} times) using symmetry {symmetry}...")
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
            self.logger.info(f"✅ Step 1/7: Heterogeneous refinement completed for K={k}, job: {hetero_job_uid}")
            
            # Step 2: Run regroup to regroup K classes into 2 superclasses (or select best class if K=2)
            self.logger.info(f"🔄 Step 2/7: Running regroup to regroup {k} classes into 2 superclasses (job name: regroup_3D_new)...")
            regroup_params = {
                "project_uid": project_uid,
                "workspace_uid": workspace_uid,
                "particles_job_uid": hetero_job_uid,  # Use particles from heterogeneous refinement
                "num_superclasses": 2,
                "job_title": "regroup_3D_new"
            }
            self._record_tool_execution("regroup_classes", regroup_params)
            regroup_result = self.cryosparc_tools.regroup_classes(
                **regroup_params,
                wait_for_completion=True,
                timeout=self.config.job_management.default_timeout,
                check_interval=self.config.job_management.status_check_interval
            )
            self._record_tool_execution("regroup_classes", regroup_params, result=regroup_result)
            
            # Verify regroup completed successfully
            if not regroup_result.get("success", False):
                error_msg = regroup_result.get("error") or "Unknown error"
                self.logger.error(f"❌ Regroup failed for K={k}: {error_msg}")
                return json.dumps({
                    "success": False,
                    "error": f"Regroup failed: {error_msg}",
                    "k": k,
                    "hetero_job_uid": hetero_job_uid
                })
            
            # Check if K=2: regroup skipped and best class was selected
            regroup_job_type = regroup_result.get("job_type", "")
            regroup_job_uid = regroup_result.get("job_uid")
            
            if regroup_job_type == "class_selection" and regroup_job_uid is None:
                # K=2 case: Skip regroup, use selected class directly for homogeneous refinement
                self.logger.info(f"✅ Step 2/7: K=2 detected, skipped regroup and selected best class")
                
                selected_class = regroup_result.get("selected_class", {})
                if not selected_class:
                    return json.dumps({
                        "success": False,
                        "error": "Selected class information not found in regroup result",
                        "k": k,
                        "hetero_job_uid": hetero_job_uid
                    })
                
                best_class_id = selected_class.get("class_id")
                best_particles_group_name = selected_class.get("particles_group_name")
                best_volume_group_name = selected_class.get("volume_group_name")
                
                if not best_particles_group_name or not best_volume_group_name:
                    return json.dumps({
                        "success": False,
                        "error": "Particles or volume group name not found in selected class",
                        "k": k,
                        "hetero_job_uid": hetero_job_uid,
                        "selected_class": selected_class
                    })
                
                self.logger.info(f"✅ Selected best class: {best_particles_group_name} (class {best_class_id})")
                self.logger.info(f"   Using particles: {best_particles_group_name}, volume: {best_volume_group_name}")
                
                # Step 3: Run refinement directly on the selected class from heterogeneous refinement
                use_nonuniform = self._should_use_nonuniform_refinement()
                refinement_type = "non-uniform" if use_nonuniform else "homogeneous"
                self.logger.info(f"🔧 Step 3/4: Running {refinement_type} refinement on selected class {best_class_id}...")
                symmetry = self._get_refinement_symmetry()
                
                # For refinement, we use particles and volume from the heterogeneous refinement job
                refine_params = {
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "particles_job_uid": hetero_job_uid,  # Particles from heterogeneous refinement
                    "volume_job_uid": hetero_job_uid,  # Volume from heterogeneous refinement
                    "symmetry": symmetry,
                    "particles_group_name": best_particles_group_name,  # e.g., "particles_class_0"
                    "volume_group_name": best_volume_group_name  # e.g., "volume_class_0"
                }
                # Add initial lowpass resolution if configured
                refine_res_init = self._get_refinement_res_init()
                if refine_res_init is not None:
                    refine_params["refine_res_init"] = float(refine_res_init)
                tool_name = "nonuniform_refine_new" if use_nonuniform else "homogeneous_refinement"
            else:
                # K>2 case: Normal regroup flow
                regroup_status = regroup_result.get("status", "unknown")
                if regroup_status != "completed":
                    error_msg = regroup_result.get("error") or f"Status: {regroup_status}"
                    self.logger.error(f"❌ Regroup did not complete for K={k}: {error_msg}")
                    return json.dumps({
                        "success": False,
                        "error": f"Regroup did not complete: {error_msg}",
                        "k": k,
                        "hetero_job_uid": hetero_job_uid
                    })
                
                self.logger.info(f"✅ Step 2/7: Regroup completed, job: {regroup_job_uid}")
                
                # Step 3: Get superclass info (num_items for each superclass)
                self.logger.info(f"📊 Step 3/7: Getting superclass information from regroup job...")
                superclass_info = self.cryosparc_tools.get_regroup_superclass_info(project_uid, regroup_job_uid)
                if not superclass_info.get("success"):
                    error_msg = superclass_info.get("error", "Unknown error")
                    self.logger.error(f"❌ Failed to get superclass info: {error_msg}")
                    return json.dumps({
                        "success": False,
                        "error": f"Failed to get superclass info: {error_msg}",
                        "k": k,
                        "hetero_job_uid": hetero_job_uid,
                        "regroup_job_uid": regroup_job_uid
                    })
                
                superclasses = superclass_info.get("superclasses", [])
                if not superclasses:
                    return json.dumps({
                        "success": False,
                        "error": "No superclasses found in regroup result",
                        "k": k,
                        "hetero_job_uid": hetero_job_uid,
                        "regroup_job_uid": regroup_job_uid
                    })
                
                # Step 4: Select superclass with more particles
                best_superclass = max(superclasses, key=lambda s: s.get("num_items", 0))
                best_superclass_id = best_superclass["superclass_id"]
                best_superclass_num_items = best_superclass["num_items"]
                best_superclass_group_name = best_superclass["group_name"]
                
                # Build comparison for all superclasses
                superclass_comparison = []
                for sc in superclasses:
                    superclass_comparison.append({
                        "superclass_id": sc["superclass_id"],
                        "num_items": sc["num_items"],
                        "group_name": sc["group_name"]
                    })
                
                selection_reason = f"Selected superclass {best_superclass_id} because it has the most particles ({best_superclass_num_items})"
                
                self.logger.info(f"✅ Step 3/7: Found {len(superclasses)} superclasses. {selection_reason}")
                self.logger.info(f"📊 Superclass comparison: {superclass_comparison}")
                
                # Step 5: Get the volume for the selected superclass and run homogeneous refinement
                # The regroup job should output volumes for each superclass (volume_superclass_X)
                # We'll use the volume from the regroup job that corresponds to the selected superclass
                volume_group_name = f"volume_superclass_{best_superclass_id}"
                
                # Run refinement on selected superclass particles and volumes
                # Both particles and volumes come from the regroup job
                # - particles: particles_superclass_X from regroup job
                # - volume: volume_superclass_X from regroup job
                use_nonuniform = self._should_use_nonuniform_refinement()
                refinement_type = "non-uniform" if use_nonuniform else "homogeneous"
                self.logger.info(f"🔧 Step 5/7: Running {refinement_type} refinement on superclass {best_superclass_id} particles and volumes...")
                symmetry = self._get_refinement_symmetry()
                
                # For refinement, we need:
                # - particles_job_uid: the regroup job, with group_name = particles_superclass_X
                # - volume_job_uid: the regroup job, with group_name = volume_superclass_X
                particles_group_name = best_superclass_group_name  # particles_superclass_X
                
                refine_params = {
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "particles_job_uid": regroup_job_uid,  # Particles from regroup
                    "volume_job_uid": regroup_job_uid,  # Volume from regroup (volume_superclass_X)
                    "symmetry": symmetry,
                    "particles_group_name": particles_group_name,  # Pass via kwargs: particles_superclass_X
                    "volume_group_name": volume_group_name  # Pass via kwargs: volume_superclass_X
                }
                # Add initial lowpass resolution if configured
                refine_res_init = self._get_refinement_res_init()
                if refine_res_init is not None:
                    refine_params["refine_res_init"] = float(refine_res_init)
                best_class_id = best_superclass_id  # For consistency in logging
                # use_nonuniform and refinement_type already set above
                tool_name = "nonuniform_refine_new" if use_nonuniform else "homogeneous_refinement"
                # Log message already set above for K>2 case
            
            # Common refinement execution for both K=2 and K>2 cases
            self._record_tool_execution(tool_name, refine_params)
            if use_nonuniform:
                refine_result = self.cryosparc_tools.nonuniform_refine_new(
                    **refine_params,
                    wait_for_completion=True,
                    timeout=self.config.job_management.default_timeout,
                    check_interval=self.config.job_management.status_check_interval
                )
            else:
                refine_result = self.cryosparc_tools.homogeneous_refinement(
                    **refine_params,
                    wait_for_completion=True,
                    timeout=self.config.job_management.default_timeout,
                    check_interval=self.config.job_management.status_check_interval
                )
            self._record_tool_execution(tool_name, refine_params, result=refine_result)
            
            # Verify refinement completed successfully
            if not refine_result.get("success", False):
                error_msg = refine_result.get("error") or "Unknown error"
                self.logger.error(f"❌ {refinement_type.capitalize()} refinement failed for class {best_class_id}: {error_msg}")
                error_data = {
                    "success": False,
                    "error": f"{refinement_type.capitalize()} refinement failed: {error_msg}",
                    "k": k,
                    "hetero_job_uid": hetero_job_uid,
                    "best_class_id": best_class_id
                }
                if regroup_job_uid:
                    error_data["regroup_job_uid"] = regroup_job_uid
                return json.dumps(error_data)
            
            refine_status = refine_result.get("status", "unknown")
            if refine_status != "completed":
                error_msg = refine_result.get("error") or f"Status: {refine_status}"
                self.logger.error(f"❌ {refinement_type.capitalize()} refinement did not complete for class {best_class_id}: {error_msg}")
                error_data = {
                    "success": False,
                    "error": f"{refinement_type.capitalize()} refinement did not complete: {error_msg}",
                    "k": k,
                    "hetero_job_uid": hetero_job_uid,
                    "best_class_id": best_class_id
                }
                if regroup_job_uid:
                    error_data["regroup_job_uid"] = regroup_job_uid
                return json.dumps(error_data)
            
            refine_job_uid = refine_result["job_uid"]
            # Adjust step number based on whether regroup was skipped (K=2) or not (K>2)
            if regroup_job_type == "class_selection" and regroup_job_uid is None:
                step_num = "3/4"  # K=2: Step 1=hetero, Step 2=select class, Step 3=refine, Step 4=FSC
            else:
                step_num = "6/7"  # K>2: Normal flow
            self.logger.info(f"✅ Step {step_num}: {refinement_type.capitalize()} refinement completed for class {best_class_id}, job: {refine_job_uid}")
            
            # Get final FSC resolution
            if regroup_job_type == "class_selection" and regroup_job_uid is None:
                step_num = "4/4"
            else:
                step_num = "7/7"
            self.logger.info(f"📊 Step {step_num}: Getting final FSC resolution for class {best_class_id}...")
            fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
            if not fsc_info.get("success"):
                error_msg = fsc_info.get("error", "Unknown error")
                self.logger.error(f"❌ Failed to get FSC info for class {best_class_id}: {error_msg}")
                error_data = {
                    "success": False,
                    "error": f"Failed to get FSC info: {error_msg}",
                    "k": k,
                    "hetero_job_uid": hetero_job_uid,
                    "best_class_id": best_class_id,
                    "refine_job_uid": refine_job_uid
                }
                if regroup_job_uid:
                    error_data["regroup_job_uid"] = regroup_job_uid
                return json.dumps(error_data)
            
            final_resolution = fsc_info["resolution_angstroms"]
            # Adjust step number and message based on whether regroup was skipped
            if regroup_job_type == "class_selection" and regroup_job_uid is None:
                step_num = "4/4"
                self.logger.info(f"✅ Step {step_num}: Final resolution for K={k}, class {best_class_id}: {final_resolution} Å")
            else:
                step_num = "7/7"
                self.logger.info(f"✅ Step {step_num}: Final resolution for K={k}, class {best_class_id}: {final_resolution} Å")
            
            # Build result dictionary based on whether regroup was skipped (K=2) or not (K>2)
            result = {
                "success": True,
                "k": k,
                "hetero_job_uid": hetero_job_uid,
                "refine_job_uid": refine_job_uid,
                "final_resolution_angstroms": final_resolution,
                "best_class_id": best_class_id
            }
            
            if regroup_job_type == "class_selection" and regroup_job_uid is None:
                # K=2 case: regroup was skipped
                selected_class = regroup_result.get("selected_class", {})
                result.update({
                    "regroup_skipped": True,
                    "regroup_job_uid": None,
                    "selected_class": selected_class
                })
            else:
                # K>2 case: regroup was performed
                result.update({
                    "regroup_skipped": False,
                    "regroup_job_uid": regroup_job_uid,
                    "regroup_job_title": "regroup_3D_new",
                    "best_superclass_id": best_class_id,  # Same as best_class_id in this context
                    "best_superclass_num_items": best_superclass_num_items,
                    "superclass_selection_reason": selection_reason,
                    "superclass_comparison": superclass_comparison,
                    "all_superclasses": superclasses
                })
            
            self._record_tool_execution("test_heterogeneous_refinement", params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("test_heterogeneous_refinement", params if 'params' in locals() else {}, error=str(e))
            return json.dumps(error_result)
    
    def _test_multi_round_3d_classification_tool(self, tool_input: str) -> str:
        """
        Run multi-round 3D classification optimization.
        
        This tool iteratively:
        1. Runs 3D classification (heterogeneous refinement) with specified number of classes
        2. Selects best class based on resolution metric
        3. Runs 3D refinement (homogeneous refinement) on selected class
        4. Checks if resolution improved
        5. If improved, continues with refined result as input for next round
        6. If plateau or worse, stops and returns best refinement job
        """
        try:
            params = self._parse_tool_input(tool_input)
            
            # Extract required parameters
            refinement_job_uid = params.get("refinement_job_uid")
            
            # Get num_classes from config if not provided in params
            # Default to config value: multi_round_3d_classification_num_classes
            config_num_classes = self._get_stage_param("optimization", "multi_round_3d_classification_num_classes", 4)
            num_classes = int(params.get("num_classes", config_num_classes))
            
            # Get max_rounds from config if not provided in params
            config_max_rounds = self._get_stage_param("optimization", "multi_round_3d_classification_max_rounds", 5)
            max_rounds = int(params.get("max_rounds", config_max_rounds))
            
            # Get improvement_threshold from config if not provided in params
            config_improvement_threshold = self._get_stage_param("optimization", "multi_round_3d_classification_improvement_threshold", 0.1)
            improvement_threshold = float(params.get("improvement_threshold", config_improvement_threshold))
            
            if not refinement_job_uid:
                return json.dumps({
                    "success": False,
                    "error": "Missing required parameter: refinement_job_uid"
                })
            
            project_uid = params.get("project_uid", self.config.workflow.project_uid)
            workspace_uid = params.get("workspace_uid", self.config.workflow.workspace_uid)
            
            self.logger.info(f"🔄 Starting multi-round 3D classification optimization with {num_classes} classes, max {max_rounds} rounds")
            
            # Get initial resolution from the input refinement job
            self.logger.info(f"📊 Getting initial resolution from refinement job {refinement_job_uid}...")
            initial_fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refinement_job_uid)
            if not initial_fsc_info.get("success"):
                return json.dumps({
                    "success": False,
                    "error": f"Failed to get initial FSC info: {initial_fsc_info.get('error', 'Unknown error')}"
                })
            
            initial_resolution = initial_fsc_info["resolution_angstroms"]
            self.logger.info(f"✅ Initial resolution: {initial_resolution} Å")
            
            # Track best result and all rounds
            best_refinement_job_uid = refinement_job_uid
            best_resolution = initial_resolution
            current_refinement_job_uid = refinement_job_uid
            all_rounds_data = []
            
            # Track the particles source for each round
            # Round 1 uses initial refinement job, subsequent rounds use particles from the selected best class
            particles_source_job_uid = refinement_job_uid
            
            # Iterate for max_rounds
            for round_num in range(1, max_rounds + 1):
                self.logger.info(f"🔄 Round {round_num}/{max_rounds}: Starting 3D classification...")
                
                symmetry = self._get_refinement_symmetry()
                
                # Step 1: In EVERY round, run ab initio reconstruction with num_classes
                # Use particles from the selected best class that was refined with homogeneous refinement
                # Round 1: Use particles from initial refinement job
                # Round 2+: Use particles from previous round's homogeneous refinement (selected best class)
                if round_num > 1:
                    # For subsequent rounds, use particles from the previous round's homogeneous refinement
                    # This ensures we're using particles from the selected best class
                    particles_source_job_uid = current_refinement_job_uid
                    self.logger.info(f"📦 Round {round_num}: Using particles from previous round's selected best class (refinement job: {particles_source_job_uid})")
                else:
                    # Round 1: Use particles from initial refinement job
                    particles_source_job_uid = refinement_job_uid
                    self.logger.info(f"📦 Round {round_num}: Using particles from initial refinement job: {particles_source_job_uid}")
                
                self.logger.info(f"📦 Step 1/5 (Round {round_num}): Running ab initio reconstruction with {num_classes} classes...")
                
                # Get resolution parameters from workflow defaults (set from config)
                defaults = getattr(self, "workflow_defaults", {}) or {}
                initial_resolution = defaults.get("ab_initio_initial_resolution", 9.0)
                final_resolution = defaults.get("ab_initio_final_resolution", 7.0)
                
                ab_initio_params = {
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "particles_job_uid": particles_source_job_uid,
                    "num_classes": num_classes,
                    "symmetry": symmetry,
                    "initial_resolution": initial_resolution,
                    "final_resolution": final_resolution,
                    "max_iterations": 50
                }
                self._record_tool_execution("ab_initio_reconstruction", ab_initio_params)
                ab_initio_result = self.cryosparc_tools.ab_initio_reconstruction(
                    **ab_initio_params,
                    wait_for_completion=True,
                    timeout=self.config.job_management.default_timeout,
                    check_interval=self.config.job_management.status_check_interval
                )
                self._record_tool_execution("ab_initio_reconstruction", ab_initio_params, result=ab_initio_result)
                
                if not ab_initio_result.get("success", False):
                    error_msg = ab_initio_result.get("error") or "Unknown error"
                    self.logger.error(f"❌ Ab initio reconstruction failed in round {round_num}: {error_msg}")
                    break
                
                ab_initio_status = ab_initio_result.get("status", "unknown")
                if ab_initio_status != "completed":
                    error_msg = ab_initio_result.get("error") or f"Status: {ab_initio_status}"
                    self.logger.error(f"❌ Ab initio reconstruction did not complete in round {round_num}: {error_msg}")
                    break
                
                ab_initio_job_uid = ab_initio_result["job_uid"]
                self.logger.info(f"✅ Step 1/5: Ab initio reconstruction completed, job: {ab_initio_job_uid}")
                
                # Step 2: Run heterogeneous refinement using all ab initio volumes (0 to n-1)
                # For ab initio with multiple classes, volumes are in the same job with different group names
                # (volume_class_0, volume_class_1, ..., volume_class_{n-1})
                self.logger.info(f"📦 Step 2/5 (Round {round_num}): Running heterogeneous refinement with all ab initio classes (0 to {num_classes-1})...")
                
                # Create heterogeneous refinement with ab initio volumes
                # We need to manually create connections with different volume group names
                try:
                    from cryosparc.tools import CryoSPARC
                    project = self.cryosparc_tools.cs.find_project(project_uid)
                    workspace = project.find_workspace(workspace_uid)
                    
                    # Get particles from ab initio job (particles_all_classes)
                    # For ab initio with multiple classes, particles are in particles_all_classes group
                    particles_slot = self.cryosparc_tools._infer_particles_output_slot(project, ab_initio_job_uid)
                    # Ensure we use particles_all_classes if available (for multi-class ab initio)
                    try:
                        ab_initio_job = project.find_job(ab_initio_job_uid)
                        ab_initio_job.refresh()
                        ab_initio_doc = getattr(ab_initio_job, "doc", {}) or {}
                        ab_initio_outputs = ab_initio_doc.get("output_result_groups", []) or []
                        for group in ab_initio_outputs:
                            name = group.get("name") or ""
                            group_type = (group.get("type") or "").lower()
                            if "particle" in group_type and "all_classes" in name.lower():
                                particles_slot = name
                                break
                    except Exception:
                        pass  # Fall back to inferred slot
                    
                    # Create volume connections: same ab initio job, different volume group names (0 to n-1)
                    volume_connections = [
                        (ab_initio_job_uid, f"volume_class_{i}") 
                        for i in range(num_classes)
                    ]
                    
                    connections = {
                        "particles": (ab_initio_job_uid, particles_slot),
                        "volume": volume_connections
                    }
                    
                    self.logger.info(f"🔗 Connecting heterogeneous refinement:")
                    self.logger.info(f"   Particles: from {ab_initio_job_uid} (group: {particles_slot})")
                    self.logger.info(f"   Volumes: from {ab_initio_job_uid} (groups: {[f'volume_class_{i}' for i in range(num_classes)]})")
                    
                    # Do not impose symmetry in heterogeneous refinement (let it use default C1)
                    # Symmetry will be applied only in the homogeneous refinement step
                    job_params = {}
                    
                    hetero_job = workspace.create_job(
                        "hetero_refine",
                        connections=connections,
                        params=job_params
                    )
                    
                    # Queue the job
                    used_lane = None
                    try:
                        hetero_job.queue()
                    except Exception as queue_error:
                        message = str(queue_error)
                        if "Must specify a lane" in message:
                            try:
                                lanes = self.cryosparc_tools.cs.get_lanes()
                                if lanes:
                                    used_lane = lanes[0]["name"]
                                    self.logger.info(f"⚙️ No lane specified; using lane '{used_lane}'")
                                    hetero_job.queue(lane=used_lane)
                            except Exception:
                                raise queue_error
                        else:
                            raise queue_error
                    
                    hetero_job_uid = hetero_job.uid
                    self.logger.info(f"✅ Queued heterogeneous refinement job: {hetero_job_uid}")
                    
                    # Wait for completion
                    hetero_result = self.cryosparc_tools.wait_for_job_completion(
                        project_uid=project_uid,
                        job_uid=hetero_job_uid,
                        timeout=self.config.job_management.default_timeout,
                        check_interval=self.config.job_management.status_check_interval
                    )
                    
                    hetero_status = hetero_result.get("status", "unknown")
                    if hetero_status != "completed":
                        error_msg = hetero_result.get("error") or f"Status: {hetero_status}"
                        self.logger.error(f"❌ Heterogeneous refinement did not complete in round {round_num}: {error_msg}")
                        break
                    
                    # Create a result dict similar to heterogeneous_refinement method
                    hetero_result = {
                        "success": True,
                        "job_uid": hetero_job_uid,
                        "status": "completed"
                    }
                    
                    self.logger.info(f"✅ Step 2/5: Heterogeneous refinement completed, job: {hetero_job_uid}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to create heterogeneous refinement with ab initio volumes: {str(e)}")
                    hetero_result = {"success": False, "error": str(e)}
                    break
                
                # Check if heterogeneous refinement was successful (common check for both branches)
                if not hetero_result.get("success", False):
                    error_msg = hetero_result.get("error") or "Unknown error"
                    self.logger.error(f"❌ Heterogeneous refinement failed in round {round_num}: {error_msg}")
                    break
                
                hetero_status = hetero_result.get("status", "unknown")
                if hetero_status != "completed":
                    error_msg = hetero_result.get("error") or f"Status: {hetero_status}"
                    self.logger.error(f"❌ Heterogeneous refinement did not complete in round {round_num}: {error_msg}")
                    break
                
                # Ensure hetero_job_uid is set (it should be set in both branches, but add safety check)
                if "hetero_job_uid" not in locals():
                    hetero_job_uid = hetero_result.get("job_uid")
                    if not hetero_job_uid:
                        self.logger.error(f"❌ Could not get hetero_job_uid in round {round_num}")
                        break
                
                # Step 3: Get resolutions for all classes and select best class
                self.logger.info(f"📊 Step 3/5: Getting class resolutions and selecting best class...")
                class_resolutions = self.cryosparc_tools.get_heterogeneous_refinement_class_resolutions(
                    project_uid, hetero_job_uid
                )
                
                if not class_resolutions.get("success"):
                    error_msg = class_resolutions.get("error", "Unknown error")
                    self.logger.error(f"❌ Failed to get class resolutions in round {round_num}: {error_msg}")
                    break
                
                classes = class_resolutions.get("classes", [])
                if not classes:
                    self.logger.error(f"❌ No classes found in round {round_num}")
                    break
                
                # Select best class: lowest resolution is best
                # If resolution is the same (within tolerance), higher fsc_loosemask_last is better
                best_class = None
                best_class_resolution = float('inf')
                best_fsc_last = -1.0
                resolution_tolerance = 0.001
                
                for class_info in classes:
                    resolution = class_info.get("resolution_angstroms")
                    fsc_last = class_info.get("fsc_loosemask_last")
                    
                    if resolution is None:
                        continue
                    
                    if resolution < best_class_resolution - resolution_tolerance:
                        best_class = class_info
                        best_class_resolution = resolution
                        best_fsc_last = fsc_last if fsc_last is not None else -1.0
                    elif abs(resolution - best_class_resolution) <= resolution_tolerance:
                        if fsc_last is not None and fsc_last > best_fsc_last:
                            best_class = class_info
                            best_class_resolution = resolution
                            best_fsc_last = fsc_last
                
                if best_class is None:
                    self.logger.error(f"❌ Could not determine best class in round {round_num}")
                    break
                
                best_class_id = best_class.get("class_id")
                volume_group_name = best_class.get("group_name")  # e.g., "volume_class_0"
                best_particles_group_name = volume_group_name.replace("volume_class_", "particles_class_")
                
                self.logger.info(f"✅ Step 3/5: Selected best class {best_class_id} with resolution {best_class_resolution:.3f} Å")
                
                # Step 4: Run refinement on selected class (non-uniform or homogeneous based on config)
                use_nonuniform = self._should_use_nonuniform_refinement()
                refinement_type = "non-uniform" if use_nonuniform else "homogeneous"
                self.logger.info(f"🔧 Step 4/5: Running {refinement_type} refinement on selected class {best_class_id}...")
                refine_params = {
                    "project_uid": project_uid,
                    "workspace_uid": workspace_uid,
                    "particles_job_uid": hetero_job_uid,
                    "volume_job_uid": hetero_job_uid,
                    "symmetry": symmetry,
                    "particles_group_name": best_particles_group_name,
                    "volume_group_name": volume_group_name
                }
                # Add initial lowpass resolution if configured
                refine_res_init = self._get_refinement_res_init()
                if refine_res_init is not None:
                    refine_params["refine_res_init"] = float(refine_res_init)
                tool_name = "nonuniform_refine_new" if use_nonuniform else "homogeneous_refinement"
                self._record_tool_execution(tool_name, refine_params)
                if use_nonuniform:
                    refine_result = self.cryosparc_tools.nonuniform_refine_new(
                        **refine_params,
                        wait_for_completion=True,
                        timeout=self.config.job_management.default_timeout,
                        check_interval=self.config.job_management.status_check_interval
                    )
                else:
                    refine_result = self.cryosparc_tools.homogeneous_refinement(
                        **refine_params,
                        wait_for_completion=True,
                        timeout=self.config.job_management.default_timeout,
                        check_interval=self.config.job_management.status_check_interval
                    )
                self._record_tool_execution(tool_name, refine_params, result=refine_result)
                
                if not refine_result.get("success", False):
                    error_msg = refine_result.get("error") or "Unknown error"
                    self.logger.error(f"❌ {refinement_type.capitalize()} refinement failed in round {round_num}: {error_msg}")
                    break
                
                refine_status = refine_result.get("status", "unknown")
                if refine_status != "completed":
                    error_msg = refine_result.get("error") or f"Status: {refine_status}"
                    self.logger.error(f"❌ {refinement_type.capitalize()} refinement did not complete in round {round_num}: {error_msg}")
                    break
                
                refine_job_uid = refine_result["job_uid"]
                self.logger.info(f"✅ Step 4/5: {refinement_type.capitalize()} refinement completed, job: {refine_job_uid}")
                
                # Step 5: Get resolution from refinement
                self.logger.info(f"📊 Step 5/5: Getting final resolution from refinement...")
                fsc_info = self.cryosparc_tools.get_refinement_fsc_info(project_uid, refine_job_uid)
                if not fsc_info.get("success"):
                    error_msg = fsc_info.get("error", "Unknown error")
                    self.logger.error(f"❌ Failed to get FSC info in round {round_num}: {error_msg}")
                    break
                
                final_resolution = fsc_info["resolution_angstroms"]
                improvement = best_resolution - final_resolution  # Positive improvement means lower resolution (better)
                
                self.logger.info(f"✅ Step 5/5: Round {round_num} final resolution: {final_resolution:.3f} Å")
                self.logger.info(f"   Previous best: {best_resolution:.3f} Å, Improvement: {improvement:+.3f} Å")
                
                # Record round data
                round_data = {
                    "round": round_num,
                    "ab_initio_job_uid": ab_initio_job_uid,
                    "hetero_job_uid": hetero_job_uid,
                    "best_class_id": best_class_id,
                    "best_class_resolution": best_class_resolution,
                    "refine_job_uid": refine_job_uid,
                    "final_resolution": final_resolution,
                    "improvement": improvement,
                    "all_classes": classes
                }
                all_rounds_data.append(round_data)
                
                # Check if resolution improved
                if final_resolution < best_resolution:
                    # Improved: update best and continue
                    best_refinement_job_uid = refine_job_uid
                    best_resolution = final_resolution
                    # Update current_refinement_job_uid for next round's ab initio (particles from selected best class)
                    current_refinement_job_uid = refine_job_uid
                    
                    if improvement >= improvement_threshold:
                        self.logger.info(f"✅ Round {round_num}: Resolution improved by {improvement:.3f} Å (>= {improvement_threshold} Å threshold), continuing...")
                        self.logger.info(f"   Next round will use particles from this refinement job (selected best class)")
                    else:
                        self.logger.info(f"⚠️  Round {round_num}: Resolution improved by {improvement:.3f} Å (< {improvement_threshold} Å threshold), but still continuing...")
                        self.logger.info(f"   Next round will use particles from this refinement job (selected best class)")
                else:
                    # Plateau or worse: stop
                    if final_resolution >= best_resolution:
                        self.logger.info(f"🛑 Round {round_num}: Resolution did not improve (current: {final_resolution:.3f} Å, best: {best_resolution:.3f} Å). Stopping optimization.")
                        break
                    else:
                        # This shouldn't happen, but handle it
                        self.logger.warning(f"⚠️  Round {round_num}: Unexpected resolution comparison result")
                        break
            
            result = {
                "success": True,
                "best_refinement_job_uid": best_refinement_job_uid,
                "best_resolution_angstroms": best_resolution,
                "initial_resolution_angstroms": initial_resolution,
                "total_improvement": initial_resolution - best_resolution,
                "rounds_completed": len(all_rounds_data),
                "all_rounds_data": all_rounds_data
            }
            
            self.logger.info(f"✅ Multi-round 3D classification optimization completed:")
            self.logger.info(f"   Initial resolution: {initial_resolution:.3f} Å")
            self.logger.info(f"   Best resolution: {best_resolution:.3f} Å")
            self.logger.info(f"   Total improvement: {initial_resolution - best_resolution:.3f} Å")
            self.logger.info(f"   Rounds completed: {len(all_rounds_data)}")
            self.logger.info(f"   Best refinement job: {best_refinement_job_uid}")
            
            self._record_tool_execution("test_multi_round_3d_classification", params, result=result)
            return json.dumps(result)
            
        except Exception as e:
            error_result = {"success": False, "error": str(e)}
            self._record_tool_execution("test_multi_round_3d_classification", params if 'params' in locals() else {}, error=str(e))
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


