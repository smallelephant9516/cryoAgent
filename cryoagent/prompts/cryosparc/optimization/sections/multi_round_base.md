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
