**CRITICAL: Check for Completed Work**:
- If you see a "COMPLETED WORK SUMMARY" section in the workflow input, READ IT CAREFULLY
- DO NOT re-run tasks that are marked as "COMPLETED" in the summary
- Use the best refinement job UIDs from completed tasks for the next step
- If multi-round 3D classification is completed, skip it and use the best_refinement_job_uid from the summary
- If heterogeneous refinement optimization is completed, skip it and use the best_refinement_job_uid from the summary
- If box size optimization has tests completed, continue from where it stopped (don't restart from the beginning)

**Optimization Priority Order**:
1. **First**: Complete multi-round 3D classification (if enabled) - iteratively refine structures through multiple rounds
   - **SKIP if already completed** - check the completed work summary
2. **Second**: Use the best refinement job from step 1 (or initial refinement if step 1 disabled) for heterogeneous refinement optimization (if enabled) - optimize K values, get best K and refinement job
   - **SKIP if already completed** - check the completed work summary
3. **Third**: Use the best refinement job from step 2 (or step 1, or initial refinement) for box size optimization (if enabled)
   - **Continue from where it stopped** if tests have already been run
4. **Finally**: Report all optimizations' results

**Example Combined Flow** (if all enabled):
1. Run multi-round 3D classification → Get best multi-round refinement result (e.g., JXXX)
2. Use JXXX as refinement_job_uid for heterogeneous refinement → Get best K and refinement job (e.g., JYYY)
3. Use JYYY as refinement_job_uid for box size optimization → Get best box size refinement result (e.g., JZZZ)
4. Report: best_multi_round_resolution, best_hetero_k, best_hetero_resolution, best_box_size, best_box_resolution

**If only multi-round 3D classification is enabled**:
- Use the refinement_job_uid provided directly for multi-round 3D classification
- Report: best_multi_round_refinement_job_uid, best_multi_round_resolution, rounds_completed
