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
      "Optional parameters: binning, patch_size, max_shift, project_uid, workspace_uid, "
      "wait_for_completion, timeout, check_interval.", job_tool=True)
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
      "in Å), symmetry, refine_do_init_scale_est (enable initial scale estimation), "
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
# Box-size / 3D optimization tools.
# ---------------------------------------------------------------------------
_spec("test_box_size", "test_box_size", "_test_box_size_tool",
      "Test a specific box size by extracting particles, running refinement, and "
      "getting FSC resolution. This tool: 1) Extracts particles with the specified "
      "box_size_pix using refined coordinates from refinement_job_uid, 2) Runs "
      "homogeneous refinement, 3) Gets FSC resolution. Required parameters: "
      "box_size_pix (box size in pixels to test), refinement_job_uid (source of "
      "refined particle coordinates). Optional parameters: micrographs_job_uid "
      "(micrographs for re-extraction), volume_job_uid (initial volume for "
      "refinement), refinement_resolution (target resolution in Angstroms), "
      "project_uid, workspace_uid. Returns: job_uid, box_size, resolution_angstroms, "
      "and status.")
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
_spec("test_heterogeneous_refinement", "test_heterogeneous_refinement", "_test_heterogeneous_refinement_tool",
      "Test heterogeneous refinement with K classes. This tool: 1) Repeats the volume "
      "from refinement_job_uid K times as initial densities, 2) Runs heterogeneous "
      "refinement using particles from refinement_job_uid, 3) Runs regroup to regroup "
      "K classes into 2 superclasses (job name: regroup_3D_new), 4) Gets num_items for "
      "each superclass from regroup job.json, 5) Selects the superclass with more "
      "particles, 6) Runs homogeneous refinement on selected superclass particles and "
      "volumes, 7) Gets final FSC resolution. Required parameters: k (number of "
      "classes, e.g., 3 or 5), refinement_job_uid (source of particles and volume, "
      "e.g., 'JXXX'). Optional parameters: project_uid, workspace_uid. Returns: "
      "hetero_job_uid, regroup_job_uid, best_superclass_id, best_superclass_num_items, "
      "refine_job_uid, final_resolution_angstroms, and status.")
_spec("test_multi_round_3d_classification", "test_multi_round_3d_classification", "_test_multi_round_3d_classification_tool",
      "Run multi-round 3D classification optimization. This tool iteratively: 1) Runs "
      "3D classification (heterogeneous refinement) with specified number of classes, "
      "2) Selects best class based on resolution metric, 3) Runs 3D refinement "
      "(homogeneous refinement) on selected class, 4) Checks if resolution improved, "
      "5) If improved, continues with refined result as input for next round, 6) If "
      "plateau or worse, stops and returns best refinement job. Required parameters: "
      "refinement_job_uid (source of particles and volume from previous best "
      "homogeneous refinement, e.g., 'JXXX'). Optional parameters: num_classes (number "
      "of classes for 3D classification, default: 4), max_rounds (maximum number of "
      "rounds, default: 5), improvement_threshold (minimum improvement in resolution "
      "in Å to continue, default: 0.1), project_uid, workspace_uid. Returns: "
      "best_refinement_job_uid, best_resolution_angstroms, rounds_completed, "
      "all_rounds_data, and status.")

# ---------------------------------------------------------------------------
# Heterogeneity tools.
# ---------------------------------------------------------------------------
_spec("run_ab_initio_hetero_combo", "run_ab_initio_hetero_combo", "_run_ab_initio_hetero_combo_tool",
      "Run ab initio reconstruction + heterogeneous refinement combo with K classes. "
      "This tool: 1) Runs ab initio reconstruction with K classes, 2) Runs "
      "heterogeneous refinement using the ab initio volumes, 3) Returns the "
      "heterogeneous refinement job UID. Required parameters: k (number of classes, "
      "e.g., 3 or 5), particles_job_uid (source of particles, e.g., 'JXXX'). Optional "
      "parameters: project_uid, workspace_uid. Returns: ab_initio_job_uid, "
      "hetero_job_uid, and status.")
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
_spec("compare_all_densities_hetero", "compare_all_densities", "_compare_all_densities_tool",
      "Compare all density maps and cluster them by structural similarity.")




