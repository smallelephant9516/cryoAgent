Optimize particle selection through iterative 2D classification and rescue excluded particles.

Input particles from picking: {{particles_job_uid}}

Workflow:
1. Step A: Run initial 2D classification and select good classes using CryoSift
2. Step B (Function 2 - Rescue): Classify excluded particles, select good classes
3. Step C (Function 1 - Iterative): Call class_2d ONCE to start iterative refinement until ≥{{threshold_pct}}% good particles (max {{max_rounds}} rounds)

For Step C: after Step B you will have two select_2d jobs. Call class_2d ONCE — the tool connects both jobs automatically. Do not call class_2d multiple times and do not merge manually.

Execute the complete workflow and return the final particles_job_uid.
