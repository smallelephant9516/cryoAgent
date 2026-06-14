Optimize particle selection through iterative 2D classification.

Input particles from picking: {{particles_job_uid}}

Workflow:
1. Step A: Run initial 2D classification and select good classes using CryoSift
2. Step C (Function 1 - Iterative): Iteratively refine until ≥{{threshold_pct}}% good particles (max {{max_rounds}} rounds)

Function 2 (Rescue) is disabled — skip Step B.

Execute the workflow and return the final particles_job_uid.
