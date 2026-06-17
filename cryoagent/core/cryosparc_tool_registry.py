"""Unified CryoSPARC tool registry.

Single source of truth for every CryoSPARC LangChain tool the agents expose.
Each tool is declared once as a :class:`ToolSpec` (keyed by a unique spec id),
and each stage agent declares an ordered list of spec ids it offers. The same
tool *name* may have more than one spec when its description/behaviour is
intentionally stage-specific (e.g. ``class_2d`` in picking vs 2D-optimization),
so consolidating into this one module preserves existing behaviour exactly.

``build_tools(agent, AGENT_TOOL_SETS[stage])`` returns the LangChain ``Tool``
objects for a stage, bound to the agent's wrapper methods. A tool is only
realized when the agent implements the bound method, so an agent can be handed
any spec list and silently skip unsupported tools.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from langchain.tools import Tool


class ToolSpec:
    """One registered tool: unique id, exposed name, wrapper method, description."""

    __slots__ = ("spec_id", "name", "method", "description", "job_tool")

    def __init__(
        self,
        spec_id: str,
        name: str,
        method: str,
        description: str,
        *,
        job_tool: bool = False,
    ) -> None:
        self.spec_id = spec_id
        self.name = name
        self.method = method
        self.description = description
        self.job_tool = job_tool


# Placeholder; real specs registered below via _spec().
_SPECS: Dict[str, ToolSpec] = {}


def _spec(spec_id, name, method, description, *, job_tool=False):
    """Register and return a ToolSpec under a unique spec id."""
    s = ToolSpec(spec_id, name, method, description, job_tool=job_tool)
    _SPECS[spec_id] = s
    return s


# ---------------------------------------------------------------------------
# Shared diagnostic / introspection tools (identical wherever used).
# ---------------------------------------------------------------------------
_spec("get_job_status", "get_job_status", "_get_job_status_tool",
      "Check the status of a CryoSPARC job. Required parameters: job_uid.")
_spec("wait_for_job", "wait_for_job", "_wait_for_job_tool",
      "Wait for a job to complete and return final status. Required parameters: "
      "job_uid. Optional parameters: timeout.")
_spec("get_job_log", "get_job_log", "_get_job_log_tool",
      "Read and analyze the log file of a CryoSPARC job to understand failures "
      "and get suggestions. Required parameters: job_uid. Optional parameters: "
      "project_uid, workspace_uid. This tool helps diagnose why a job failed and "
      "provides suggestions for fixing the issues.")
_spec("get_job_log_common", "get_job_log", "_get_job_log_tool",
      "Read and analyze the log file of a CryoSPARC job to understand failures "
      "and get suggestions. Required parameters: job_uid. Optional parameters: "
      "project_uid, workspace_uid. Use after get_job_status confirms status = failed.")
_spec("search_cryosparc_forum", "search_cryosparc_forum", "_search_cryosparc_forum_tool",
      "Search https://discuss.cryosparc.com for troubleshooting threads related "
      "to a CryoSPARC error. Use after a job fails to find community solutions "
      "before retrying. Required: query (error keywords from the job log) and/or "
      "job_uid (failed job; log is parsed for search terms). Optional: max_results "
      "(default 5), project_uid, workspace_uid.")
_spec("describe_job_params", "describe_job_params", "_describe_job_params_tool",
      "Look up the full parameter specification (keys, types, defaults) for a "
      "CryoSPARC job type, so you can set ANY parameter via a job tool's 'params' "
      "dict. Required: job_type (a friendly name like 'motion_correction', "
      "'ctf_estimation', 'class_2d', 'ab_initio_reconstruction', or a raw "
      "CryoSPARC id like 'patch_motion_correction_multi'). Optional: include_hidden. "
      "Call this before submitting a job when you need a parameter that is not one "
      "of the tool's friendly named arguments.")

# Per-stage "reason about workflow" tools (intentionally distinct text).
_spec("reason_preprocessing", "reason_about_workflow", "_reason_about_workflow_tool",
      "Analyze the current preprocessing workflow state and determine next steps. "
      "Use this to think through the workflow progression and identify dependencies.")
_spec("reason_picking", "reason_about_workflow", "_reason_about_workflow_tool",
      "Analyze the current particle picking workflow state and determine next steps. "
      "Use this to think through particle detection parameters and job dependencies.")
_spec("reason_reconstruction", "reason_about_workflow", "_reason_about_workflow_tool",
      "Analyze the current 3D reconstruction workflow state and determine next steps. "
      "Use this to think through reconstruction parameters and job dependencies.")
_spec("reason_optimizer", "reason_about_workflow", "_reason_about_workflow_tool",
      "Analyze the current box size optimization workflow state and determine next "
      "steps. Use this to think through optimization parameters and job dependencies.")

# ---------------------------------------------------------------------------
# Preprocessing job tools.
# ---------------------------------------------------------------------------
_spec("import_movies", "import_movies", "_import_movies_tool",
      "Import movie files into CryoSPARC for processing. Required parameters: None "
      "(all loaded from microscope_config.json). Optional parameters: project_uid, "
      "workspace_uid, set_index, wait_for_completion, timeout, check_interval. When "
      "movies_path is a list, imports every path in one call (each paired with its "
      "gain_ref_path when provided). Use set_index to import only one configured set. "
      "All microscope parameters (movies_path, gain_ref_path, pixel_size, voltage, "
      "cs_mm, dose) are automatically loaded from microscope_config.json. Gain "
      "reference orientation (gainref_flip_x, gainref_flip_y, gainref_rotate_num) is "
      "auto-derived from the same configuration.", job_tool=True)
_spec("import_micrographs", "import_micrographs", "_import_micrographs_tool",
      "Import micrograph files directly into CryoSPARC (skips motion correction). "
      "Use this when you have already motion-corrected micrographs. Required "
      "parameters: None (all loaded from microscope_config.json). Optional parameters: "
      "project_uid, workspace_uid, wait_for_completion, timeout, check_interval. All "
      "microscope parameters (micrographs_path, pixel_size, voltage, cs_mm, dose) are "
      "automatically loaded from microscope_config.json. Note: When using "
      "import_micrographs, motion correction is NOT needed - proceed directly to CTF "
      "estimation.", job_tool=True)
_spec("motion_correction", "motion_correction", "_motion_correction_tool",
      "Perform motion correction on imported movies. Required parameters: "
      "movies_job_uid or movies_job_uids (comma-separated when multiple import jobs "
      "exist). When omitted, uses all import_movies job UIDs from the current session. "
      "Optional parameters: project_uid, workspace_uid, wait_for_completion, timeout, "
      "check_interval. NOTE: patch_motion_correction_multi has no 'binning'/'patch_size' "
      "params; to tune motion correction pass real CryoSPARC keys (e.g. res_max_align) "
      "via the params dict.", job_tool=True)
_spec("ctf_estimation", "ctf_estimation", "_ctf_estimation_tool",
      "Estimate CTF parameters for micrographs. Required parameters: "
      "micrographs_job_uid. Optional parameters: min_res, max_res, defocus_range, "
      "project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
      job_tool=True)
_spec("micrograph_selection", "micrograph_selection", "_micrograph_selection_tool",
      "Select micrographs with resolution better than specified threshold. Required "
      "parameters: ctf_job_uid. Optional parameters: min_resolution, project_uid, "
      "workspace_uid, wait_for_completion, timeout, check_interval.", job_tool=True)

# ---------------------------------------------------------------------------
# Picking job tools.
# ---------------------------------------------------------------------------
_spec("blob_picker", "blob_picker", "_blob_picker_tool",
      "Detect and pick particles from micrographs using GPU-accelerated blob "
      "detection. Required parameters: micrographs_job_uid, particle_diameter. "
      "Optional parameters: diameter_max (default: 2x particle_diameter), project_uid, "
      "workspace_uid, wait_for_completion, timeout, check_interval.", job_tool=True)
_spec("extract_particles", "extract_particles", "_extract_particles_tool",
      "Extract particles from micrographs using particle coordinates from picking. "
      "Required parameters: particles_job_uid (from blob picker), micrographs_job_uid "
      "(from CTF/selection), box_size_pix (box size in pixels). Optional parameters: "
      "project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
      job_tool=True)
_spec("class_2d_picking", "class_2d", "_class_2d_tool",
      "Perform 2D classification on extracted particles. Required parameters: "
      "particles_job_uid (from extraction). Optional parameters: num_classes, "
      "batchsize_per_class (defaults from particle_picking_config 2d_classification), "
      "project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
      job_tool=True)
_spec("select_2d_classes_picking", "select_2d_classes", "_select_2d_classes_tool",
      "Select 2D classes to use as templates. Required parameters: class_2d_job_uid "
      "(from 2D classification). Optional parameters: selection_mode (top_n or "
      "cryosift), top_n_classes (default: 5 when top_n), cryosift_threshold, "
      "cryosift_env, cryosift_weights_path, cryosift_output_dir, project_uid, "
      "workspace_uid, wait_for_completion, timeout, check_interval.", job_tool=True)
_spec("template_picker", "template_picker", "_template_picker_tool",
      "Template-based particle picking using 2D class averages as templates. More "
      "accurate than blob picker. Required parameters: micrographs_job_uid (from "
      "CTF/selection), template_job_uid (from select_2d_classes). Optional parameters: "
      "lowpass_resolution (default: 20.0 Å), project_uid, workspace_uid, "
      "wait_for_completion, timeout, check_interval.", job_tool=True)

# ---------------------------------------------------------------------------
# 2D optimization tools (class_2d / select_2d_classes have distinct text here).
# ---------------------------------------------------------------------------
_spec("class_2d_opt2d", "class_2d", "_class_2d_tool",
      "Run 2D classification on particles. Required parameters: particles_job_uid "
      "(e.g., 'J123') OR job_uid (when passing just 'J123'). Optional parameters: "
      "num_classes (default from config), particles_group_name (e.g., "
      "'particles_excluded' for excluded particles from select_2D job), project_uid, "
      "workspace_uid. Returns: job_uid and status. CRITICAL for Function 2 (Rescue): "
      "When running rescue workflow, you MUST: (1) Use the select_2D job_uid from Step "
      "A (e.g., 'J116'), (2) Specify particles_group_name='particles_excluded' to "
      "classify excluded particles. Example: class_2d with particles_job_uid='J116' "
      "and particles_group_name='particles_excluded'. DO NOT call class_2d on a "
      "select_2D job without particles_group_name - this will classify the wrong "
      "particles!", job_tool=True)
_spec("select_2d_classes_opt2d", "select_2d_classes", "_select_2d_classes_tool",
      "Select 2D classes using various selection modes. Required parameters: "
      "class_2d_job_uid (e.g., 'J123'). Optional parameters: selection_mode (default: "
      "'cryosift', options: 'cryosift', 'top_n', 'all'), cryosift_threshold, "
      "project_uid, workspace_uid. Selection modes: 'cryosift' (selects classes using "
      "CryoSift evaluation), 'top_n' (selects top N classes by particle count), 'all' "
      "(selects all classes). Returns: job_uid, selected_template_indices, "
      "selection_metadata, and status.", job_tool=True)
_spec("get_particle_count", "get_particle_count", "_get_particle_count_tool",
      "Get the number of particles in a particles job. Required parameters: "
      "particles_job_uid (e.g., 'J123'). Optional parameters: particles_group_name "
      "(default: 'particles'), project_uid. Returns: num_particles, "
      "particles_group_name, and success status.")
_spec("merge_particles", "merge_particles", "_merge_particles_tool",
      "Merge particles from multiple jobs into a single particles set. Required "
      "parameters: particles_job_uids (comma-separated list, e.g., 'J123,J124'). "
      "Optional parameters: project_uid, workspace_uid. Returns: merged job_uid and "
      "status.")

# ---------------------------------------------------------------------------
# Reconstruction tools.
# ---------------------------------------------------------------------------
_spec("ab_initio_reconstruction", "ab_initio_reconstruction", "_ab_initio_tool",
      "Generate initial 3D models from 2D particles using ab initio reconstruction. "
      "Required parameters: particles_job_uid (from 2D class selection or extraction). "
      "Optional parameters: num_classes (number of 3D classes, default: 1), "
      "initial_resolution (starting resolution in Å, default: 20.0), final_resolution "
      "(target resolution in Å, default: 10.0), max_iterations (default: 50), symmetry "
      "(default: C1), project_uid, workspace_uid, wait_for_completion, timeout, "
      "check_interval.", job_tool=True)
_spec("homogeneous_reconstruction", "homogeneous_reconstruction", "_homogeneous_reconstruction_tool",
      "Generate a 3D model from 2D particles using homogeneous reconstruction. This is "
      "an alternative to ab initio that's often faster and more robust for homogeneous "
      "datasets. Required parameters: particles_job_uid (from 2D class selection or "
      "extraction). Optional parameters: initial_resolution (starting resolution in Å, "
      "default: 20.0), final_resolution (target resolution in Å, default: 8.0), "
      "symmetry (default: C1), project_uid, workspace_uid, wait_for_completion, "
      "timeout, check_interval.", job_tool=True)
_spec("homogeneous_refinement_recon", "homogeneous_refinement", "_homogeneous_refinement_tool",
      "Refine a single 3D structure with all particles. Required parameters: "
      "particles_job_uid (from ORIGINAL input - Select 2D job or import particle job), "
      "volume_job_uid (from ab initio reconstruction job). CRITICAL: particles_job_uid "
      "and volume_job_uid must be DIFFERENT - particles from original input, volume "
      "from ab initio. Optional parameters: refinement_resolution (target resolution "
      "in Å as a NUMBER; omit it for automatic — do NOT pass 'auto'), symmetry, "
      "refine_do_init_scale_est (enable initial scale estimation), "
      "refine_highpass_res (high-pass filter resolution in Å), refine_num_final_iterations "
      "(number of final iterations), refine_res_init (initial resolution in Å), "
      "refine_symmetry_do_align (enable symmetry alignment), refine_defocus_refine "
      "(enable defocus refinement during CTF refinement, default: True), "
      "refine_ctf_global_refine (enable global CTF refinement, default: True), "
      "project_uid, workspace_uid, wait_for_completion, timeout, check_interval.",
      job_tool=True)
_spec("heterogeneous_refinement", "heterogeneous_refinement", "_heterogeneous_refinement_tool",
      "Simultaneously refine multiple 3D classes with particles. Required parameters: "
      "particles_job_uid, volume_job_uids (list of initial volumes). Optional "
      "parameters: num_classes (default: 3), project_uid, workspace_uid, "
      "wait_for_completion, timeout, check_interval.", job_tool=True)

# ---------------------------------------------------------------------------
# Box-size / 3D optimization analysis tools (composite test_* specs removed —
# the optimization stage now uses the atomic opt_* tools defined below).
# ---------------------------------------------------------------------------
_spec("get_fsc_info", "get_fsc_info", "_get_fsc_info_tool",
      "Get FSC resolution and box size information from a refinement job. You can pass "
      "just the job UID (e.g., 'JXXX') or JSON with refinement_job_uid parameter. "
      "Optional parameters: project_uid, workspace_uid. Returns: box_size (in pixels), "
      "resolution_angstroms (FSC resolution), and success status.")
_spec("get_hetero_class_resolutions_opt", "get_hetero_class_resolutions", "_get_hetero_class_resolutions_tool",
      "Get resolution information for each class in a heterogeneous refinement job. "
      "You can pass just the job UID (e.g., 'JXXX') or JSON with job_uid parameter. "
      "Returns a list of classes with resolution_angstroms and fsc_loosemask_last for "
      "each class. Optional parameters: project_uid, workspace_uid. Returns: classes "
      "(list with class_id, resolution_angstroms, fsc_loosemask_last), num_classes, "
      "and success status.")

# ---------------------------------------------------------------------------
# Heterogeneity tools (composite run_ab_initio_hetero_combo removed — the stage
# now uses atomic het_ab_initio + het_heterogeneous_refinement defined below).
# ---------------------------------------------------------------------------
_spec("extract_density_maps_hetero", "extract_density_maps", "_extract_density_maps_tool",
      "Get the job directory containing density map files (*_volume.mrc) from a "
      "heterogeneous refinement job. Returns the job directory directly without "
      "copying files (Docker-accessible). You can pass just the job UID (e.g., 'JXXX') "
      "or JSON with hetero_job_uid parameter. Optional parameters: project_uid. "
      "Returns: output_folder (job directory path), num_maps_extracted, map_files list "
      "(full paths), and success status.")
_spec("get_hetero_class_resolutions_hetero", "get_hetero_class_resolutions", "_get_hetero_class_resolutions_tool",
      "Get resolution information for each class in a heterogeneous refinement job. "
      "You can pass just the job UID (e.g., 'JXXX') or JSON with job_uid parameter. "
      "Returns a list of classes with resolution_angstroms and fsc_loosemask_last for "
      "each class. Optional parameters: project_uid, workspace_uid. Returns: classes "
      "(list with class_id, resolution_angstroms, fsc_loosemask_last), num_classes, "
      "and success status.")
_spec("run_non_uniform_refinement_hetero", "run_non_uniform_refinement", "_run_non_uniform_refinement_tool",
      "Run non-uniform homogeneous refinement for a specific group of particles. This "
      "tool refines a single group using particles from one or more classes and a "
      "volume from the best-resolution class. Required parameters: hetero_job_uid "
      "(heterogeneous refinement job UID), particles_group_names (list of particle "
      "group names like 'particles_class_0'), volume_group_name (volume group name "
      "like 'volume_class_0' from the class with best resolution). Optional "
      "parameters: project_uid, workspace_uid, refine_res_init (initial lowpass "
      "resolution in Angstroms). Returns: job_uid, job_type, and status.")


# ---------------------------------------------------------------------------
# Polish tools.
# ---------------------------------------------------------------------------
_spec("homogeneous_refinement_polish", "homogeneous_refinement", "_homogeneous_refinement_tool",
      "Refine a single 3D structure with local and global CTF refinement enabled. "
      "Required parameters: particles_job_uid, volume_job_uid. Optional parameters: "
      "refinement_resolution, symmetry, refine_defocus_refine (enable local CTF "
      "refinement, default: True), refine_ctf_global_refine (enable global CTF "
      "refinement, default: True), refine_do_init_scale_est, refine_highpass_res, "
      "refine_num_final_iterations, refine_res_init, refine_symmetry_do_align, "
      "particles_group_name, project_uid, workspace_uid, wait_for_completion, timeout, "
      "check_interval.", job_tool=True)
_spec("reference_motion_correction", "reference_motion_correction", "_reference_motion_correction_tool",
      "Run reference-based motion correction on particles using a reference volume. "
      "Required parameters: micrographs_job_uid, particles_job_uid, volume_job_uid. "
      "Optional parameters: All reference_motion_correction job parameters can be "
      "passed via kwargs. project_uid, workspace_uid, wait_for_completion, timeout, "
      "check_interval.", job_tool=True)
_spec("verify_inputs", "verify_inputs", "_verify_inputs_tool",
      "Verify that optimization and preprocessing stages are complete and read "
      "required job UIDs. This checks for optimization_results_cryosparc_*.json and "
      "preprocessing_results_cryosparc_*.json files. No parameters required.")

# Heterogeneity-depth specific specs (distinct descriptions from the plain
# heterogeneity stage).
_spec("read_input_json", "read_input_json", "_read_input_json_tool",
      "Read JSON file from either refinement job (reconstruction_results_*.json) or "
      "heterogeneity job (heterogeneity_analysis_results_*.json). If heterogeneity job "
      "JSON exists, it will be preferred. When reading from "
      "heterogeneity_analysis_results_*.json, returns all clusters from "
      "final_refinement_jobs with: num_clusters, clusters (array with "
      "refinement_job_uid, particles_job_uid, volume_job_uid, particles_group_name, "
      "volume_group_name for each cluster). When reading from "
      "reconstruction_results_*.json, returns: refinement_job_uid, particles_job_uid, "
      "volume_job_uid. Optional parameter: config_path (path to config file).")
_spec("extract_density_maps_depth", "extract_density_maps", "_extract_density_maps_tool",
      "Get the job directory containing density map files (*_volume.mrc) from a "
      "heterogeneous refinement job. Returns the job directory directly without "
      "copying files. You can pass just the job UID (e.g., 'JXXX') or JSON with "
      "hetero_job_uid parameter. Optional parameters: project_uid. Returns: "
      "output_folder (job directory path), num_maps_extracted, map_files list (full "
      "paths), and success status.")
_spec("run_homogeneous_refinement_depth", "run_homogeneous_refinement", "_run_homogeneous_refinement_tool",
      "Run homogeneous refinement using particles and volume from a job. Required "
      "parameters: particles_job_uid, volume_job_uid. Optional parameters: "
      "particles_group_name (e.g., 'particles_class_0' or 'particles_all_classes'), "
      "volume_group_name (e.g., 'volume_class_0'), refine_defocus_refine (enable "
      "defocus refinement during CTF refinement, default: True), "
      "refine_ctf_global_refine (enable global CTF refinement, default: True), "
      "project_uid, workspace_uid. Returns: job_uid, status.")
_spec("get_hetero_class_resolutions_depth", "get_hetero_class_resolutions", "_get_hetero_class_resolutions_tool",
      "Get resolution for each class and label GOOD vs BAD relative to the depth "
      "threshold. Normally auto-run inside run_heterogeneous_refinement — call manually "
      "only to re-analyze an older job UID. You can pass just the job UID (e.g., "
      "'JXXX') or JSON with job_uid parameter. Returns: good_classes, bad_classes, "
      "fallback_non_uniform (when zero good classes), resolution_threshold_angstroms, "
      "next_action, and per-class quality labels.")
_spec("run_non_uniform_refinement_depth", "run_non_uniform_refinement", "_run_non_uniform_refinement_tool",
      "Run non-uniform refinement to terminate a branch. Converged good cluster: "
      "hetero_job_uid + particles_group_names (good class(es)) + best volume. Zero good "
      "classes fallback: hetero_job_uid + particles_group_names=['particles_all_classes'] "
      "+ best volume_class_X. Required: hetero_job_uid, particles_group_names (list), "
      "volume_group_name. Optional: project_uid, workspace_uid, refine_res_init. After "
      "completion, call wait_for_job then get_fsc_info to report final resolution.")
_spec("get_fsc_info_depth", "get_fsc_info", "_get_fsc_info_tool",
      "Get FSC resolution and box size from a completed non-uniform refinement job. "
      "Pass job UID (e.g. 'JXXX') or JSON with refinement_job_uid. MUST be called after "
      "wait_for_job on the final non-uniform refinement job.")
_spec("compare_all_densities_depth", "compare_all_densities", "_compare_all_densities_tool",
      "Compare density maps in a folder and filter clusters by resolution. Normally "
      "auto-run inside run_heterogeneous_refinement — call manually only to re-analyze "
      "an older job UID. Required: folder (path to hetero job directory). Do NOT pass "
      "the full get_hetero_class_resolutions JSON — only folder + optional "
      "class_resolutions. KEPT clusters continue; FILTERED OUT (BAD) clusters must be "
      "thrown away with no further processing.")


# ---------------------------------------------------------------------------
# Atomic optimization tools (decomposed from the former composite test_* tools).
# These let the LLM drive box-size sweeps, hetero-K tests and multi-round 3D
# classification step by step (recipes live in the optimization task prompt).
# ---------------------------------------------------------------------------
_spec("opt_extract_particles", "extract_particles", "_extract_particles_tool",
      "Re-extract particles at a chosen box size using refined coordinates. Required: "
      "particles_job_uid (refined-coords source), micrographs_job_uid, box_size_pix. "
      "Used in box-size optimization to test a candidate box size before refinement.",
      job_tool=True)
_spec("opt_ab_initio", "ab_initio_reconstruction", "_ab_initio_tool",
      "Ab-initio 3D reconstruction. Required: particles_job_uid. Optional: num_classes "
      "(K), initial_resolution, final_resolution, symmetry. In multi-round 3D "
      "classification, run this with K classes to seed heterogeneous refinement.",
      job_tool=True)
_spec("opt_heterogeneous_refinement", "heterogeneous_refinement", "_heterogeneous_refinement_tool",
      "Heterogeneous (K-class) refinement. Supply volumes either as volume_job_uids "
      "(list/comma-string) OR volume_from_job_uid + num_classes (to use the K "
      "volume_class outputs of one ab-initio/refinement job). Required: "
      "particles_job_uid. Used in hetero-K tests and multi-round classification.",
      job_tool=True)
_spec("opt_regroup_classes", "regroup_classes", "_regroup_classes_tool",
      "Regroup the K classes of a heterogeneous refinement into fewer superclasses. "
      "Required: particles_job_uid (the hetero refinement job). Optional: "
      "num_superclasses (default 2). Returns the regroup job UID.", job_tool=True)
_spec("opt_get_regroup_superclass_info", "get_regroup_superclass_info", "_get_regroup_superclass_info_tool",
      "Read per-superclass particle counts (num_items) from a regroup job so you can "
      "pick the largest superclass. Required: regroup_job_uid.")
_spec("opt_nonuniform_refinement", "nonuniform_refinement", "_nonuniform_refinement_tool",
      "Non-uniform 3D refinement (higher resolution). Required: particles_job_uid, "
      "volume_job_uid. Optional: particles_group_name/volume_group_name (to refine a "
      "specific class/superclass), refine_res_init, symmetry. Used as the refinement "
      "step in optimization recipes when non-uniform refinement is preferred.",
      job_tool=True)
_spec("opt_homogeneous_refinement", "homogeneous_refinement", "_homogeneous_refinement_tool",
      "Homogeneous 3D refinement. Required: particles_job_uid, volume_job_uid. Optional: "
      "particles_group_name/volume_group_name, refinement_resolution, symmetry. Used as "
      "the refinement step in optimization recipes.", job_tool=True)


# ---------------------------------------------------------------------------
# Atomic heterogeneity tools (decomposed from run_ab_initio_hetero_combo).
# ---------------------------------------------------------------------------
_spec("het_ab_initio", "ab_initio_reconstruction", "_ab_initio_tool",
      "Ab-initio 3D reconstruction with K classes. Required: particles_job_uid. "
      "Optional: num_classes (K), symmetry, initial_resolution, final_resolution. "
      "Run this first, then heterogeneous_refinement using its K volume_class outputs.",
      job_tool=True)
_spec("het_heterogeneous_refinement", "heterogeneous_refinement", "_heterogeneous_refinement_tool",
      "Heterogeneous (K-class) refinement seeded from an ab-initio job's volumes. Pass "
      "volume_from_job_uid=<ab_initio_job> + num_classes=K (uses its volume_class_0..K-1 "
      "outputs), or volume_job_uids. Required: particles_job_uid. Then extract density "
      "maps, get class resolutions, and compare densities.", job_tool=True)
_spec("het_get_fsc_info", "get_fsc_info", "_get_fsc_info_tool",
      "Get FSC resolution and box size from a completed refinement job. Pass job UID or "
      "JSON with refinement_job_uid.")

# Atomic heterogeneity-depth tools (decomposed from the auto-analyzing
# run_heterogeneous_refinement composite).
_spec("depth_ab_initio", "ab_initio_reconstruction", "_ab_initio_tool",
      "Ab-initio 3D reconstruction with K classes. Required: particles_job_uid. "
      "Optional: num_classes (K), symmetry. Seeds heterogeneous_refinement.",
      job_tool=True)
_spec("depth_heterogeneous_refinement", "heterogeneous_refinement", "_heterogeneous_refinement_tool",
      "Heterogeneous (K-class) refinement — runs ONLY the hetero_refine job (no auto "
      "analysis). Pass volume_from_job_uid + num_classes, or volume_job_uids. Required: "
      "particles_job_uid. After it completes, call get_hetero_class_resolutions, "
      "extract_density_maps and compare_all_densities yourself.", job_tool=True)


# ---------------------------------------------------------------------------
# Per-stage ordered tool sets (spec ids). These reproduce each agent's existing
# tool list exactly, in order. ``compare_all_densities`` for the heterogeneity
# stages is omitted here because those agents construct it with config-bound
# scripts and append it themselves.
# ---------------------------------------------------------------------------
AGENT_TOOL_SETS: Dict[str, List[str]] = {
    "preprocessing": [
        "import_movies", "import_micrographs", "motion_correction", "ctf_estimation",
        "micrograph_selection", "get_job_status", "wait_for_job", "get_job_log",
        "search_cryosparc_forum", "describe_job_params", "reason_preprocessing",
    ],
    "particle_picking": [
        "blob_picker", "extract_particles", "class_2d_picking",
        "select_2d_classes_picking", "template_picker", "get_job_status",
        "wait_for_job", "get_job_log", "search_cryosparc_forum", "describe_job_params",
        "reason_picking",
    ],
    "optimization_2d": [
        "class_2d_opt2d", "select_2d_classes_opt2d", "get_particle_count",
        "merge_particles", "get_job_status", "wait_for_job", "get_job_log_common",
        "search_cryosparc_forum", "describe_job_params",
    ],
    "reconstruction": [
        "ab_initio_reconstruction", "homogeneous_refinement_recon",
        "heterogeneous_refinement", "get_job_status", "wait_for_job", "get_job_log",
        "search_cryosparc_forum", "describe_job_params", "reason_reconstruction",
    ],
    "optimization": [
        # Atomic action tools (LLM drives box-size sweep / hetero-K / multi-round
        # recipes via the optimization task prompt).
        "opt_extract_particles", "opt_ab_initio", "opt_heterogeneous_refinement",
        "opt_regroup_classes", "opt_get_regroup_superclass_info",
        "opt_nonuniform_refinement", "opt_homogeneous_refinement",
        # Analysis + diagnostics.
        "get_fsc_info", "get_hetero_class_resolutions_opt",
        "get_job_status", "wait_for_job", "get_job_log", "search_cryosparc_forum",
        "describe_job_params", "reason_optimizer",
    ],
    "heterogeneity": [
        # Atomic: ab-initio then K-class heterogeneous refinement (LLM drives the
        # combo + density-analysis recipe via the heterogeneity task prompt).
        "het_ab_initio", "het_heterogeneous_refinement",
        "extract_density_maps_hetero", "get_hetero_class_resolutions_hetero",
        "run_non_uniform_refinement_hetero", "het_get_fsc_info",
        "get_job_status", "wait_for_job", "get_job_log", "search_cryosparc_forum",
        "describe_job_params",
        # compare_all_densities appended by the agent (config-bound construction)
    ],
    "heterogeneity_depth": [
        # Atomic hetero refinement (no auto-analysis); LLM runs the analyze/branch
        # recipe itself via the heterogeneity_depth task prompt.
        "read_input_json", "depth_ab_initio", "depth_heterogeneous_refinement",
        "extract_density_maps_depth", "get_hetero_class_resolutions_depth",
        "run_homogeneous_refinement_depth", "run_non_uniform_refinement_depth",
        "get_fsc_info_depth", "get_job_status", "wait_for_job", "get_job_log",
        "search_cryosparc_forum", "describe_job_params",
        # compare_all_densities appended by the agent
    ],
    "polish": [
        "homogeneous_refinement_polish", "reference_motion_correction",
        "get_job_status", "wait_for_job", "get_job_log_common", "search_cryosparc_forum",
        "describe_job_params", "verify_inputs",
    ],
}


def build_tool(agent, spec_id: str) -> Optional[Tool]:
    """Build one plain Tool from a spec id, bound to the agent's wrapper method.

    Returns None when the spec id is unknown or the agent does not implement the
    bound method (so an agent can be handed any list and skip what it lacks).
    """
    spec = _SPECS.get(spec_id)
    if spec is None:
        return None
    func: Callable = getattr(agent, spec.method, None)
    if func is None or not callable(func):
        return None
    return Tool(name=spec.name, description=spec.description, func=func)


def build_tools(agent, spec_ids: List[str]) -> List[Tool]:
    """Build the ordered list of Tools for the given spec ids the agent supports."""
    tools: List[Tool] = []
    for spec_id in spec_ids:
        tool = build_tool(agent, spec_id)
        if tool is not None:
            tools.append(tool)
    return tools


def get_spec(spec_id: str) -> Optional[ToolSpec]:
    """Return the ToolSpec for a spec id (or None)."""
    return _SPECS.get(spec_id)





