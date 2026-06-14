Rescue good particles from the excluded set.

Input particles from picking: {{particles_job_uid}}

Workflow:
1. Step A: Run initial 2D classification and select good classes using CryoSift
2. Step B (Function 2 - Rescue): Classify excluded particles, select good classes, merge with Step A particles

Function 1 (Iterative) is disabled — return merged particles after Step B.

Execute the workflow and return the final particles_job_uid.
