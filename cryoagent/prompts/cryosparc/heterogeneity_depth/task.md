Perform heterogeneity depth analysis on EACH starting cluster from the heterogeneity analysis results.

**MANDATORY FIRST STEP:** call `read_input_json` before any refinement job. Do not proceed until clusters are loaded from `heterogeneity_analysis_results_*.json`.

## Goal
Among structures that refine below {{resolution_threshold}} Å, determine whether further heterogeneity exists. Split good branches recursively; discard bad density clusters; terminate every branch with non-uniform refinement + FSC resolution.

## Resolution threshold: {{resolution_threshold}} Å
- GOOD class: resolution < {{resolution_threshold}} Å (lower is better)
- BAD class: resolution ≥ {{resolution_threshold}} Å
- PER CLUSTER / PER BRANCH filtering — a passing sibling does NOT rescue a failed cluster

## After each `run_heterogeneous_refinement` (auto-includes density comparison)
Use the returned fields directly — do NOT manually re-run compare unless re-analyzing an old job.

### Case A — ZERO good classes (`good_classes` empty or `fallback_non_uniform` present)
- Do NOT discard the branch
- `run_non_uniform_refinement` with:
  * `particles_group_names=["particles_all_classes"]` from the hetero job
  * `volume_group_name` = best-volume class among ALL classes (lowest resolution)
- `wait_for_job` → `get_fsc_info` → branch COMPLETE

### Case B — one or more good classes
1. Read KEPT vs FILTERED OUT clusters from `density_comparison`
2. **Throw away every BAD / FILTERED OUT density cluster** — no further hetero or refinement on them
3. Among KEPT clusters only:

**B1 — ONE KEPT cluster** → branch converged:
- `run_non_uniform_refinement` on that cluster's good `particles_class_X` + best `volume_class_X`
- `wait_for_job` → `get_fsc_info` → branch COMPLETE

**B2 — MULTIPLE KEPT clusters** → split and recurse:
- For EACH KEPT cluster only (skip discarded ones):
  * `run_heterogeneous_refinement` with K={{k_value}} on that cluster's good classes
  * Repeat Case A / B on the new hetero result

## Execution steps

1. **Read input JSON** → get all starting clusters from `final_refinement_jobs`
   - Each cluster: refinement_job_uid, particles_job_uid, volume_job_uid, particles_group_names, volume_group_name
   - Upstream refinement jobs use `particles_group_names=["particles"]`, `volume_group_name="volume"`
   - Process each starting cluster as an independent tree

2. **For each starting cluster:**
   a. `run_heterogeneous_refinement` with K={{k_value}} on cluster particles + volume (always hetero first — never skip to non-uniform)
   b. Apply Case A / B / B1 / B2 to the result
   c. Recurse on KEPT sub-branches until all branches terminate

3. **Record outputs for every terminated branch:**
   - Final refinement job UID
   - `final_resolution_angstroms` from `get_fsc_info`
   - Bad clusters discarded at every round (never refined further)

## Rules you must follow
- BAD density clusters after comparison → discard completely, even if another cluster in the same job passed
- Zero good classes → fallback non-uniform on all particles (branch is NOT abandoned)
- One KEPT good cluster → converged → non-uniform refinement (NOT homogeneous)
- Multiple KEPT good clusters → hetero on each good cluster separately
- Before each job, state which case (A, B1, B2) applies and which clusters you keep vs discard
