Optimize heterogeneous refinement for 3D reconstruction.

I have completed homogeneous refinement with job UID: {{refinement_job_uid}}

IMPORTANT: Box size optimization is DISABLED. DO NOT re-extract particles or run box-size optimization.
Proceed directly to heterogeneous refinement optimization using the refinement_job_uid provided ({{refinement_job_uid}}).
To test each K value, drive the recipe yourself: heterogeneous_refinement → wait_for_job → regroup_classes → wait_for_job → (if K>2) get_regroup_superclass_info → (nonuniform_refinement or homogeneous_refinement) on the selected superclass → wait_for_job → get_fsc_info (see the Heterogeneous Refinement Workflow for the full recipe and stopping conditions).
