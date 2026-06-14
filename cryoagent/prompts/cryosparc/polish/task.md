Execute the polish refinement workflow to achieve the best possible resolution:

**Project**: {{project_uid}} | **Workspace**: {{workspace_uid}}
**Symmetry**: {{symmetry}} (from reconstruction_config.json)

**Workflow Steps** (execute in order):

═══ STEP 1: Verify Inputs ═══

1. **Verify Inputs** - Check that optimization and preprocessing are complete
   - Tool: verify_inputs
   - No parameters required
   - This will read optimization_results_cryosparc_*.json to get best_job_uid
   - This will read preprocessing_results_cryosparc_*.json to get final_micrographs_job_uid
   - Wait for verification to complete

═══ STEP 2: Initial Homogeneous Refinement with CTF ═══

2. **Initial Homogeneous Refinement** - Refine with CTF refinement enabled
   - Tool: homogeneous_refinement
   - Parameters:
     * particles_job_uid=[best_job_uid from Step 1]
     * volume_job_uid=[best_job_uid from Step 1]
     * symmetry={{symmetry}} (from reconstruction_config.json)
     * refine_defocus_refine=true (enable local CTF refinement)
     * refine_ctf_global_refine=true (enable global CTF refinement)
   - Wait for completion and record job UID (J-A)

═══ STEP 3: Reference Motion Correction ═══

3. **Reference Motion Correction** - Correct particle motion using reference volume
   - Tool: reference_motion_correction
   - Parameters:
     * micrographs_job_uid=[final_micrographs_job_uid from Step 1]
     * particles_job_uid=[J-A from Step 2]
     * volume_job_uid=[J-A from Step 2]
   - Wait for completion and record job UID (J-B)
   - Output particles group name is "particles_0"

═══ STEP 4: Final Homogeneous Refinement with CTF ═══

4. **Final Homogeneous Refinement** - Final refinement with CTF refinement enabled
   - Tool: homogeneous_refinement
   - Parameters:
     * particles_job_uid=[J-B from Step 3]
     * particles_group_name="particles_0" (use particles_0 group from motion correction)
     * volume_job_uid=[J-A from Step 2] (use volume from initial refinement, NOT from motion correction)
     * symmetry={{symmetry}} (from reconstruction_config.json)
     * refine_defocus_refine=true (enable local CTF refinement)
     * refine_ctf_global_refine=true (enable global CTF refinement)
   - Wait for completion and record job UID (J-C)
   - J-C is the final best job UID

**Critical Instructions**:
- Execute ALL steps in order - do not skip any steps
- Each step MUST complete successfully before proceeding
- Always wait_for_job after each CryoSPARC job
- Track all job UIDs - each step depends on the previous step's output
- Polish refinement jobs can take significant time (minutes to hours)
- Use symmetry={{symmetry}} from reconstruction_config.json for all refinement steps

**Expected Outcome**:
- Final refined structure (J-C) with best resolution
- All CTF refinement enabled for optimal results
- Motion-corrected particles from reference-based correction

Begin by executing step 1 (verify_inputs) and proceed through all steps sequentially.
