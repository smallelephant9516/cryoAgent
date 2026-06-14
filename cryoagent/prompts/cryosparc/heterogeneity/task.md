Perform heterogeneity analysis to determine the true number of classes in the sample.

Homogeneous refinement job: {{refinement_job_uid}}
Initial volume job: {{volume_job_uid}}
Particles job: {{particles_job_uid}}
Micrographs job: {{micrographs_job_uid}}

Workflow:
1. Run ab initio + heterogeneous refinement combo for K={{initial_k_first}} and K={{initial_k_second}}
2. Extract density maps from each heterogeneous refinement job
3. Compare all density maps using compare_all_densities
4. For each K, determine true class count from clustering results
5. Filter clusters with resolution worse than {{resolution_threshold}} Å
6. If K={{initial_k_first}} and K={{initial_k_second}} show the same true class count, stop and refine
7. Otherwise try higher K values up to {{max_k}} until convergence
8. Select the hetero job with the most groups (tie-break: higher K), then refine each valid group

Return the analysis summary and refinement job UIDs.
