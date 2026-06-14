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
