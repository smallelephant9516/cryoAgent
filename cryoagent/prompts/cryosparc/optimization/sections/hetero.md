**Purpose**: Optimize the number of classes (K) in heterogeneous refinement to achieve the best resolution. This should be done BEFORE box size optimization (if box size optimization is enabled).

**You drive this loop yourself using atomic tools.** There is no single "test a K value" tool — you run each step and reason between them.

### Recipe to test ONE K value (num_classes = K), given refinement_job_uid:
1. **heterogeneous_refinement** with `particles_job_uid=<refinement_job_uid>`, `volume_from_job_uid=<refinement_job_uid>`, `num_classes=K`. This uses the refinement job's consensus volume, automatically repeated K times as identical starting references. Then **wait_for_job** on the returned job UID until completed.
   * IMPORTANT: `volume_from_job_uid` should be the SAME consensus refinement job that provides the particles. The tool inspects that job's outputs and connects its single `volume` output K times (for refinement jobs) or its `volume_class_*` outputs (for ab-initio jobs) — you do NOT need to know which; just pass the refinement job UID and K. Do NOT substitute an unrelated ab-initio job's class volumes as the seeds.
2. **regroup_classes** with `particles_job_uid=<hetero_job_uid>`, `num_superclasses=2`. Then **wait_for_job** until completed.
3. (Only if K > 2) **get_regroup_superclass_info** with `regroup_job_uid=<regroup_job_uid>` and pick the superclass with the most `num_items`. Note its index `<best>`.
4. **Refine** the selected superclass. Use **nonuniform_refinement** when non-uniform refinement is preferred for this dataset, otherwise **homogeneous_refinement**, with `particles_job_uid=<regroup_job_uid>`, `volume_job_uid=<hetero_job_uid>`, `particles_group_name=particles_superclass_<best>`, `volume_group_name=volume_class_<best>`. Then **wait_for_job** until completed.
5. **get_fsc_info** with `refinement_job_uid=<refine_job_uid>` to read the achieved `resolution_angstroms` for K classes.

You may also call **get_hetero_class_resolutions** on the heterogeneous refinement job to inspect per-class resolutions and verify your selection makes sense.

Record (K → resolution) and move to the next candidate.

**Agentic Optimization Process**:
1. **Baseline**: Get FSC resolution from the final refinement job (K=1, which is homogeneous refinement)
2. **First Round Testing**: Test K=2 and K=3 by running the recipe above for each
3. **REASONING REQUIRED**: After EACH test, you MUST actively reason about the results:
   - Analyze the per-class data (from `get_hetero_class_resolutions` and the regroup superclass info) and your class/superclass selection from each test
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

- **heterogeneous_refinement**: Run heterogeneous refinement with K classes.
  * Required: particles_job_uid (source of particles), volume_from_job_uid (source of the starting volume, repeated K times), num_classes (K)
  * Returns: job_uid for the heterogeneous refinement job
- **regroup_classes**: Regroup the heterogeneous refinement classes into superclasses.
  * Required: particles_job_uid (the heterogeneous refinement job), num_superclasses (use 2)
  * Returns: job_uid for the regroup job
- **get_regroup_superclass_info**: Inspect superclasses produced by regroup_classes (use when K > 2 to pick the superclass with the most num_items).
  * Required: regroup_job_uid
  * Returns: superclasses with num_items for each
- **nonuniform_refinement** / **homogeneous_refinement**: Run final refinement on the selected superclass.
  * Required: particles_job_uid (the regroup job), volume_job_uid (the heterogeneous refinement job), particles_group_name (e.g. particles_superclass_<best>), volume_group_name (e.g. volume_class_<best>)
- **get_hetero_class_resolutions**: Get resolution for each class in a heterogeneous refinement job
  * Required: job_uid (heterogeneous refinement job UID)
  * Returns: classes (list with class_id, resolution_angstroms, fsc_loosemask_last), num_classes
  * Use this to analyze individual class resolutions and verify your selection
- **get_fsc_info**: Get FSC resolution from any refinement job (works for both homogeneous and heterogeneous)
  * Required: refinement_job_uid
  * Returns: box_size, resolution_angstroms
- **wait_for_job**: Always wait for each heterogeneous/regroup/refine job to complete before the next step.

**Heterogeneous Refinement Strategy Guidelines**:

**Initial Testing**:
- Always start by getting FSC info from the original refinement job (K=1 baseline)
- Test K=2 and K=3 in the first round
- Compare K=1, K=2, and K=3 to see the trend

**REASONING REQUIREMENT - After Each Test**:
**CRITICAL**: After running the recipe for a K value, you MUST actively reason about:
1. **Class Selection Analysis**: Review the per-class resolutions (`get_hetero_class_resolutions`) and the regroup superclass info to understand which class/superclass was selected and why
   - Are all classes in the heterogeneous refinement similar in quality?
   - Was there a clear winner, or were classes close in resolution?
   - Does the selected class/superclass seem reasonable given all available data?

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
- K=3: [resolution] Å (check per-class resolutions and superclass selection)
- K=5: [resolution] Å (check per-class resolutions and superclass selection)

Analysis:
- Compare resolutions across all tested K values - which has the smallest resolution value?
- Identify the trend: does increasing K improve or worsen resolution?
- Review per-class data: are classes similar or is there a clear winner?
- Understand why each class/superclass was selected

Decision: Based on the trend and analysis:
- If larger K improves resolution: test higher K (e.g., K=7)
- If smaller K is better: test K=2 or stop
- If middle K is best: test around it (e.g., K=2, K=4)
- Monitor if improvement is diminishing

Action: run the recipe (heterogeneous_refinement → wait_for_job → regroup_classes → wait_for_job → [get_regroup_superclass_info] → refine → wait_for_job → get_fsc_info) with the appropriate k value based on your analysis

[After getting result]
Thought: Compare new result with previous results. Has resolution improved, worsened, or plateaued?

Decision: Decide whether to:
- Continue testing if trend suggests improvement
- STOP if resolution worsens, plateaus, or optimal point is clear
```
