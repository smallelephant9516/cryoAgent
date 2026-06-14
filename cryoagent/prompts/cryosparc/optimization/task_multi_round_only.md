Run multi-round 3D classification for 3D reconstruction.

I have completed homogeneous refinement with job UID: {{refinement_job_uid}}

IMPORTANT: Box size optimization and heterogeneous refinement are DISABLED. DO NOT use test_box_size or test_heterogeneous_refinement tools.
Proceed directly to multi-round 3D classification using the refinement_job_uid provided ({{refinement_job_uid}}).
Use the test_multi_round_3d_classification tool with:
- num_classes: {{multi_round_num_classes}}
- max_rounds: {{multi_round_max_rounds}}
- improvement_threshold: {{multi_round_improvement_threshold}}

After completing multi-round 3D classification, you will proceed to heterogeneous refinement (K optimization) and then box size optimization.
