## Box Size Optimization Workflow:

**Purpose**: Optimize the box size to achieve the best resolution. This should be done AFTER heterogeneous refinement optimization (if heterogeneous refinement is enabled), otherwise after the first round of 3D homogeneous refinement.

**You drive this loop yourself using atomic tools.** There is no single "test a box size" tool — you run each step and reason between them.

### Recipe to test ONE box size (box_size_pix = N), given refinement_job_uid, micrographs_job_uid, volume_job_uid:
1. **extract_particles** with `particles_job_uid=<refinement_job_uid>` (refined coordinates), `micrographs_job_uid=<micrographs_job_uid>`, `box_size_pix=N`. Then **wait_for_job** on the returned job UID until completed.
2. **Refine** the re-extracted particles. Use **nonuniform_refinement** when non-uniform refinement is preferred for this dataset, otherwise **homogeneous_refinement**, with `particles_job_uid=<extract_job_uid>` and `volume_job_uid=<volume_job_uid>`. Then **wait_for_job** until completed.
3. **get_fsc_info** with `refinement_job_uid=<refine_job_uid>` to read the achieved `resolution_angstroms` for box size N.

Record (box size N → resolution) and move to the next candidate.

**Agentic Optimization Process**:
1. **Initial Assessment**: Get FSC resolution and box size from the original refinement job using `get_fsc_info`.
2. **First Round Testing**: Test 10% less and 10% more than the original box size by running the recipe above for each.
3. **REASONING REQUIRED**: After EACH test, you MUST actively reason about the results:
   - Compare box size results across all tested values so far
   - Identify trends: Which direction (larger/smaller box sizes) improves resolution?
   - Determine if the optimal point is clear or needs more testing
   - Calculate the next box size to test based on trends
4. **Decision Making**: Based on your reasoning, decide:
   - **Continue testing**: Choose which box size to test next based on trends (run the recipe again)
   - **Stop optimization**: If resolution plateaus, worsens, or you've found the optimal box size
5. **Iterative Process**: Repeat steps 3-4 until:
   - You find the optimal box size (clear best result)
   - Resolution plateaus or worsens (diminishing returns)
   - You reach a reasonable number of tests (5-7 different box sizes)
   - Further testing is unlikely to improve results
6. **Conclusion**: Summarize the best box size and resolution found, explaining why it was chosen.

**Stopping Conditions**:
- You've tested 5-7 different box sizes
- The resolution improvement plateaus or starts getting worse
- You've found a clear optimal point
- The new box size to test would be the same as one already tested

## Tool Usage for Box Size Optimization:

- **extract_particles**: Re-extract particles at a candidate box size using refined coordinates.
  * Required: particles_job_uid (the refinement job providing refined coords), micrographs_job_uid, box_size_pix
- **nonuniform_refinement** / **homogeneous_refinement**: Refine the re-extracted particles against the initial volume.
  * Required: particles_job_uid (the extract job), volume_job_uid
  * Optional: refinement_resolution, symmetry
- **get_fsc_info**: Get FSC resolution and box size from a refinement job.
  * Required: refinement_job_uid
  * Returns: box_size (pixels), resolution_angstroms (FSC resolution)
- **wait_for_job**: Always wait for each extract/refine job to complete before the next step.

**Box Size Normalization**: CryoSPARC only accepts certain box sizes (e.g., 16, 20, 24, ..., 2000). If you request a value like 483, pick the nearest allowed value (e.g., 480 or 484). When in doubt, you may call `describe_job_params("extract_particles")` to review parameters.

## Optimization Strategy Guidelines:

**Trend Analysis**:
- **CRITICAL: Smaller resolution_angstroms value = BETTER quality** (e.g., 3.0 Å is BETTER than 5.0 Å)
- When comparing results, the box size with the SMALLEST resolution_angstroms value is the BEST
- Look for patterns: Does resolution improve with larger or smaller box sizes?
- Consider if the relationship is linear or has an optimal point

**Next Steps Decision**:
- If middle box size is best: Test halfway between middle and the better extreme
- If smallest box size is best: Test 10% less than the smallest tested
- If largest box size is best: Test 10% more than the largest tested

**Example Reasoning Pattern** (use actual values from your test results, not these examples):
```
Thought: I have tested multiple box sizes:
- Original box size: [resolution] Å
- Smaller box size (-10%): [resolution] Å
- Larger box size (+10%): [resolution] Å
Analysis: which box size gives the smallest resolution value? Is resolution improving
with larger or smaller box sizes? Is the improvement significant or marginal?
Decision: based on the trend, pick the next box size to test (or stop).
Action: extract_particles(box_size_pix=...) → wait_for_job → refine → wait_for_job → get_fsc_info
```

