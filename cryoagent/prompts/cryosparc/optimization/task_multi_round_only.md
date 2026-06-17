Run multi-round 3D classification for 3D reconstruction.

I have completed homogeneous refinement with job UID: {{refinement_job_uid}}

IMPORTANT: Box size optimization and heterogeneous refinement (K optimization) are DISABLED. DO NOT re-extract particles for box-size optimization, and DO NOT run a K-value optimization loop.
Proceed directly to multi-round 3D classification using the refinement_job_uid provided ({{refinement_job_uid}}).
Drive the multi-round loop yourself (see the Multi-Round 3D Classification Workflow for the full per-round recipe and loop control) with:
- num_classes: {{multi_round_num_classes}}
- max_rounds: {{multi_round_max_rounds}}
- improvement_threshold: {{multi_round_improvement_threshold}}
For each round: ab_initio_reconstruction → wait_for_job → heterogeneous_refinement → wait_for_job → get_hetero_class_resolutions (select best class) → (nonuniform_refinement or homogeneous_refinement) → wait_for_job → get_fsc_info. Continue while resolution improves by >= improvement_threshold and you are under max_rounds.

After completing multi-round 3D classification, you will proceed to heterogeneous refinement (K optimization) and then box size optimization.
