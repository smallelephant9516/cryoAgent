## Multi-Round 3D Classification Optimization Workflow:

**Purpose**: Iteratively refine 3D structures using multi-round 3D classification to achieve the best resolution. This should be done FIRST, before heterogeneous refinement optimization (if enabled) and box size optimization (if enabled), using the initial homogeneous refinement result.

**You drive this loop yourself using atomic tools.** There is no single "run multi-round classification" tool — you run each round's steps and track the best result across rounds.

### Recipe for ONE round, given a `particles_source` job (start with refinement_job_uid) and num_classes:
1. **ab_initio_reconstruction** with `particles_job_uid=<particles_source>`, `num_classes=<num_classes>`, prefer `symmetry=C1`. Then **wait_for_job** until completed.
2. **heterogeneous_refinement** with `particles_job_uid=<particles_source>`, `volume_from_job_uid=<ab_initio_job_uid>`, `num_classes=<num_classes>`. Then **wait_for_job** until completed.
3. **get_hetero_class_resolutions** with `job_uid=<hetero_job_uid>` and select the best class (smallest `resolution_angstroms`). Note its index `<best>`.
4. **Refine** the best class. Use **nonuniform_refinement** when non-uniform refinement is preferred for this dataset, otherwise **homogeneous_refinement**, with `particles_job_uid=<hetero_job_uid>`, `volume_job_uid=<hetero_job_uid>`, `particles_group_name=particles_class_<best>`, `volume_group_name=volume_class_<best>`. Then **wait_for_job** until completed.
5. **get_fsc_info** with `refinement_job_uid=<refine_job_uid>` to read this round's `resolution_angstroms`.

### Loop control (you track this yourself):
- Track `best_resolution` and `best_refine_job` across rounds (initialize best_resolution from `get_fsc_info` on the starting refinement_job_uid).
- After each round, compare the round's resolution to `best_resolution`:
  - **If it improved by >= improvement_threshold** (resolution got smaller by at least the threshold in Å): update best_resolution/best_refine_job, set `particles_source = <refine_job_uid>`, and continue to the next round.
  - **Otherwise**: STOP and report the best refine job found so far.
- Also STOP once you reach max_rounds rounds.

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

- **ab_initio_reconstruction**: Generate ab initio volumes to seed the heterogeneous refinement for a round.
  * Required: particles_job_uid (the current particles_source), num_classes
  * Returns: job_uid for the ab initio job
- **heterogeneous_refinement**: Run 3D classification for the round using the ab initio volumes.
  * Required: particles_job_uid (the current particles_source), volume_from_job_uid (the ab initio job), num_classes
  * Returns: job_uid for the heterogeneous refinement job
- **get_hetero_class_resolutions**: Get per-class resolutions to select the best class.
  * Required: job_uid (the heterogeneous refinement job)
  * Returns: classes (list with class_id, resolution_angstroms); pick the smallest resolution_angstroms
- **nonuniform_refinement** / **homogeneous_refinement**: Refine the selected best class.
  * Required: particles_job_uid (the heterogeneous refinement job), volume_job_uid (the heterogeneous refinement job), particles_group_name (e.g. particles_class_<best>), volume_group_name (e.g. volume_class_<best>)
- **get_fsc_info**: Read the round's resolution from the refinement job.
  * Required: refinement_job_uid
  * Returns: box_size, resolution_angstroms
- **wait_for_job**: Always wait for each ab initio / heterogeneous / refine job to complete before the next step.

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

**Understanding Results** (track these yourself across rounds):
- **best refinement job**: The job UID of the best refinement result (use this for next stages)
- **best resolution (Å)**: The best resolution achieved (lower is better)
- **total improvement**: Difference from initial to best resolution (positive means improvement)
- **rounds completed**: Number of rounds you ran
- **per-round data**: Track each round's class selection and resulting resolution as you go

**CRITICAL: Smaller resolution_angstroms value = BETTER quality** (e.g., 3.0 Å is BETTER than 5.0 Å). When selecting the best class within a round, pick the smallest resolution_angstroms. When deciding whether to continue, compare the round's resolution against your tracked best.
