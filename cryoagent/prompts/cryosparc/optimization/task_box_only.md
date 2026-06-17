Optimize the box size for 3D reconstruction.

I have completed the first round of homogeneous refinement with job UID: {{refinement_job_uid}}
The initial volume is from job: {{volume_job_uid}}
Particles can be re-extracted from picking job: {{particles_job_uid}}
Micrographs are available from job: {{micrographs_job_uid}}

Please optimize the box size by driving the box-size optimization workflow yourself: for each candidate box size, run extract_particles → wait_for_job → (nonuniform_refinement or homogeneous_refinement) → wait_for_job → get_fsc_info, then reason about the trend and pick the next box size (see the Box Size Optimization Workflow for the full recipe and stopping conditions).
