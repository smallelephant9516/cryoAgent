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
   - You reach max_iterations ({{max_hetero_iterations}})
   - Further testing is unlikely to improve results
6. **Conclusion**: Summarize the best K value and resolution found, explaining why it was chosen

**Stopping Conditions**:
- You've tested {{max_hetero_iterations}} different K values
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
     * You've reached or approached max_iterations ({{max_hetero_iterations}})
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
