"""Unit tests for the generic CryoSPARC provider layer.

Covers the provider methods added to CryoSPARCTools:
  * describe_job_params additive metadata (description / enum choices /
    input_slot_groups)
  * describe_job_inputs / describe_job_outputs
  * run_cryosparc_job (generic guarded runner)
  * merge_particles (downsample_particles-based concatenation)
  * homogeneous_reconstruction (raw homo_reconstruct job type)
  * rebalance_orientations (curated rebalance_3D adapter)

All CryoSPARC interactions are mocked; no live connection is made and no real
jobs are created or queued. Fixtures mirror the real v4.7.1 job-spec structure
(sections -> contains -> JobSpec with params_base / input_slot_groups).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from cryoagent.tools.cryosparc_tools import CryoSPARCTools


# ---------------------------------------------------------------------------
# Spec fixtures (structure mirrors /tmp/cryosparc_v471_job_specs.json)
# ---------------------------------------------------------------------------

def _param(type_, value=None, title=None, hidden=False, desc="",
           enum_keys=None, enum_dict=None):
    details = {
        "title": title or type_,
        "type": type_,
        "value": value,
        "hidden": hidden,
        "desc": desc,
    }
    if enum_keys is not None:
        details["enum_keys"] = enum_keys
    if enum_dict is not None:
        details["enum_dict"] = enum_dict
    return details


def _slot(name, type_, optional=False):
    return {
        "name": name,
        "type": type_,
        "title": name,
        "description": "",
        "optional": optional,
    }


def _group(name, type_, count_min=0, count_max=1, slots=(), repeat_allowed=False,
           description="", title=None):
    return {
        "name": name,
        "type": type_,
        "title": title or name,
        "description": description,
        "count_min": count_min,
        "count_max": count_max,
        "repeat_allowed": repeat_allowed,
        "connections": [],
        "slots": list(slots),
    }


def _job_spec(name, title=None, input_slot_groups=(), params_base=None, description=""):
    return {
        "name": name,
        "title": title or name,
        "description": description,
        "input_slot_groups": list(input_slot_groups),
        "params_base": params_base or {},
    }


def _specs(*job_specs):
    return [{"name": "sec", "title": "Section", "description": "",
             "contains": list(job_specs)}]


INF = float("inf")

CLASS_2D_SPEC = _job_spec(
    "class_2D",
    title="2D Classification",
    description="Classify particles into 2D classes.",
    input_slot_groups=[
        _group(
            "particles", "particle", count_min=1, count_max=INF,
            slots=[_slot("blob", "particle.blob"),
                   _slot("alignments2D", "particle.alignments2D", optional=True)],
            description="Particles to classify",
        ),
    ],
    params_base={
        "class2D_K": _param("number", 50, title="Number of 2D classes",
                            desc="Number of classes to group particles into."),
        "class2D_mode": _param("enum", "on", title="Mode",
                               desc="Simple enum used to exercise enum validation.",
                               enum_keys=["on", "off"],
                               enum_dict={"on": 1, "off": 0}),
        "class2D_hidden": _param("number", 1, hidden=True),
    },
)

SHARPEN_SPEC = _job_spec(
    "sharpen",
    input_slot_groups=[
        _group("volume", "volume", count_min=1, count_max=1,
               slots=[_slot("map", "volume.blob")]),
        _group("mask", "mask", count_min=0, count_max=1,
               slots=[_slot("mask", "volume.blob")]),
    ],
    params_base={"sharp_bfactor": _param("number", 0)},
)

HOMO_RECONSTRUCT_SPEC = _job_spec(
    "homo_reconstruct",
    input_slot_groups=[
        _group("particles", "particle", count_min=1, count_max=INF,
               slots=[_slot("blob", "particle.blob"),
                      _slot("ctf", "particle.ctf"),
                      _slot("alignments3D", "particle.alignments3D")]),
        _group("mask", "mask", count_min=0, count_max=1,
               slots=[_slot("mask", "volume.blob")]),
    ],
    params_base={"refine_symmetry": _param("string", "C1")},
)

IMPORT_VOLUMES_SPEC = _job_spec("import_volumes", params_base={})
VALIDATION_SPEC = _job_spec("validation", params_base={})
IMPORT_PARTICLES_SPEC = _job_spec("import_particles", params_base={})

ALL_SPECS = _specs(
    CLASS_2D_SPEC, SHARPEN_SPEC, HOMO_RECONSTRUCT_SPEC,
    IMPORT_VOLUMES_SPEC, VALIDATION_SPEC, IMPORT_PARTICLES_SPEC,
)


def _make_tools(specs=None):
    """Build a CryoSPARCTools without any live connection."""
    tools = CryoSPARCTools.__new__(CryoSPARCTools)
    tools.settings = SimpleNamespace(lane=None, topaz_exec_path=None)
    tools.cs = Mock(name="cryosparc_client")
    tools._job_cache = {}
    if specs is not None:
        tools.cs.get_job_specs = Mock(return_value=specs)
    return tools


def _mock_job(uid, output_groups=None, job_type="job", status="completed"):
    """Mock Job whose doc carries the given (name, type[, num_items]) outputs."""
    job = Mock(name=f"job_{uid}")
    job.uid = uid
    groups = []
    for entry in (output_groups or []):
        name, type_ = entry[0], entry[1]
        num_items = entry[2] if len(entry) > 2 else 0
        groups.append({"name": name, "type": type_, "title": name,
                       "num_items": num_items})
    job.doc = {
        "type": job_type,
        "status": status,
        "title": job_type,
        "output_result_groups": groups,
    }
    return job


def _wire_project(tools, jobs_by_uid=None):
    """Wire tools.cs.find_project -> project with per-uid find_job lookups."""
    project = Mock(name="project")
    jobs_by_uid = jobs_by_uid or {}

    def _find_job(uid):
        if uid not in jobs_by_uid:
            raise RuntimeError(f"unknown job {uid}")
        return jobs_by_uid[uid]

    project.find_job.side_effect = _find_job
    workspace = Mock(name="workspace")
    project.find_workspace.return_value = workspace
    tools.cs.find_project.return_value = project
    return project, workspace


# ---------------------------------------------------------------------------
# describe_job_params additive metadata
# ---------------------------------------------------------------------------

class DescribeJobParamsMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tools = _make_tools(specs=ALL_SPECS)

    def test_adds_description_and_choices_without_breaking_existing_keys(self):
        result = self.tools.describe_job_params("class_2D")
        params = result["params"]

        k_param = params["class2D_K"]
        # Existing keys preserved.
        self.assertEqual(k_param["title"], "Number of 2D classes")
        self.assertEqual(k_param["type"], "number")
        self.assertEqual(k_param["default"], 50)
        # Additive description.
        self.assertEqual(k_param["description"],
                         "Number of classes to group particles into.")
        self.assertNotIn("choices", k_param)  # not an enum

        enum_param = params["class2D_mode"]
        self.assertEqual(enum_param["type"], "enum")
        self.assertEqual(enum_param["default"], "on")
        self.assertEqual(enum_param["choices"], ["on", "off"])
        self.assertEqual(enum_param["description"],
                         "Simple enum used to exercise enum validation.")

    def test_return_includes_input_slot_groups(self):
        result = self.tools.describe_job_params("class_2D")
        self.assertIn("input_slot_groups", result)
        groups = result["input_slot_groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["name"], "particles")
        self.assertTrue(groups[0]["required"])

    def test_existing_behavior_preserved(self):
        result = self.tools.describe_job_params("class_2d")  # alias resolves
        self.assertEqual(result["job_type"], "class_2D")
        self.assertEqual(result["requested_job_type"], "class_2d")
        self.assertNotIn("class2D_hidden", result["params"])
        with_hidden = self.tools.describe_job_params("class_2D", include_hidden=True)
        self.assertIn("class2D_hidden", with_hidden["params"])
        with self.assertRaises(ValueError):
            self.tools.describe_job_params("no_such_job")


# ---------------------------------------------------------------------------
# describe_job_inputs
# ---------------------------------------------------------------------------

class DescribeJobInputsTests(unittest.TestCase):
    def setUp(self):
        self.tools = _make_tools(specs=ALL_SPECS)

    def test_returns_validated_group_metadata(self):
        result = self.tools.describe_job_inputs("homo_reconstruct")
        self.assertEqual(result["job_type"], "homo_reconstruct")
        groups = {g["name"]: g for g in result["input_slot_groups"]}
        self.assertEqual(set(groups), {"particles", "mask"})

        particles = groups["particles"]
        self.assertEqual(particles["type"], "particle")
        self.assertEqual(particles["count_min"], 1)
        self.assertTrue(particles["required"])
        # Infinity is normalized to JSON-safe None + explicit flag.
        self.assertIsNone(particles["count_max"])
        self.assertTrue(particles["count_max_unlimited"])
        slot_names = [s["name"] for s in particles["slots"]]
        self.assertEqual(slot_names, ["blob", "ctf", "alignments3D"])

        mask = groups["mask"]
        self.assertFalse(mask["required"])
        self.assertEqual(mask["count_max"], 1)
        self.assertFalse(mask["count_max_unlimited"])

    def test_resolves_friendly_alias(self):
        result = self.tools.describe_job_inputs("class_2d")
        self.assertEqual(result["job_type"], "class_2D")
        self.assertEqual(result["requested_job_type"], "class_2d")

    def test_unknown_type_raises_with_available(self):
        with self.assertRaises(ValueError) as ctx:
            self.tools.describe_job_inputs("bogus_type")
        message = str(ctx.exception)
        self.assertIn("bogus_type", message)
        self.assertIn("class_2D", message)


# ---------------------------------------------------------------------------
# describe_job_outputs
# ---------------------------------------------------------------------------

class DescribeJobOutputsTests(unittest.TestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_returns_real_output_groups_after_refresh(self):
        job = _mock_job(
            "J10",
            output_groups=[("particles_selected", "particle", 1234),
                           ("particles_excluded", "particle", 55)],
            job_type="select_2D",
        )
        self.tools.cs.find_job.return_value = job

        result = self.tools.describe_job_outputs("J10", "P1")

        self.tools.cs.find_job.assert_called_once_with("P1", "J10")
        job.refresh.assert_called_once()
        self.assertEqual(result["job_uid"], "J10")
        self.assertEqual(result["project_uid"], "P1")
        self.assertEqual(result["job_type"], "select_2D")
        self.assertEqual(result["output_count"], 2)
        first = result["output_result_groups"][0]
        self.assertEqual(first["name"], "particles_selected")
        self.assertEqual(first["type"], "particle")
        self.assertEqual(first["num_items"], 1234)

    def test_failure_raises_runtime_error(self):
        self.tools.cs.find_job.side_effect = RuntimeError("no such job")
        with self.assertRaises(RuntimeError):
            self.tools.describe_job_outputs("J999", "P1")


# ---------------------------------------------------------------------------
# run_cryosparc_job
# ---------------------------------------------------------------------------

class RunCryoSPARCJobTests(unittest.TestCase):
    def setUp(self):
        self.tools = _make_tools(specs=ALL_SPECS)
        self.jobs = {
            "J1": _mock_job("J1", [("particles", "particle", 100)]),
            "J2": _mock_job("J2", [("particles_selected", "particle", 80),
                                   ("particles_excluded", "particle", 20)]),
            "J3": _mock_job("J3", [("volume", "volume", 1)]),
        }
        self.project, self.workspace = _wire_project(self.tools, self.jobs)
        created = _mock_job("J99")
        self.workspace.create_job.return_value = created
        self.created = created

    # -- job type resolution ------------------------------------------------

    def test_unknown_job_type_lists_available(self):
        result = self.tools.run_cryosparc_job("P1", "W1", "nonsense")
        self.assertFalse(result["success"])
        self.assertIn("nonsense", result["error"])
        self.assertIn("class_2D", result["available_job_types"])
        self.workspace.create_job.assert_not_called()

    def test_friendly_alias_resolves_and_queues(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2d",
            inputs=[{"input_slot": "particles", "job_uid": "J1",
                     "output_group": "particles"}],
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["job_type"], "class_2D")
        self.assertEqual(result["requested_job_type"], "class_2d")
        self.workspace.create_job.assert_called_once_with(
            "class_2D",
            connections={"particles": ("J1", "particles")},
            params={},
        )
        self.created.queue.assert_called_once()
        self.assertEqual(result["job_uid"], "J99")
        self.assertEqual(self.tools._job_cache["J99"],
                         {"project_uid": "P1", "workspace_uid": "W1"})

    # -- param validation ---------------------------------------------------

    def test_unknown_param_lists_available_params(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"input_slot": "particles", "job_uid": "J1",
                     "output_group": "particles"}],
            params={"bogus_param": 1},
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["unknown_params"], ["bogus_param"])
        self.assertIn("class2D_K", result["available_params"])
        self.workspace.create_job.assert_not_called()

    def test_params_forwarded_to_create_job(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"input_slot": "particles", "job_uid": "J1",
                     "output_group": "particles"}],
            params={"class2D_K": 100},
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(self.workspace.create_job.call_args.kwargs["params"],
                         {"class2D_K": 100})

    def test_enum_validation_accepts_keys_and_dict_values(self):
        base = dict(project_uid="P1", workspace_uid="W1", job_type="class_2D",
                    inputs=[{"input_slot": "particles", "job_uid": "J1",
                             "output_group": "particles"}])
        # enum_keys value passes.
        self.assertTrue(self.tools.run_cryosparc_job(
            params={"class2D_mode": "off"}, **base)["success"])
        # legitimate enum_dict value (0/1) is not rejected.
        self.assertTrue(self.tools.run_cryosparc_job(
            params={"class2D_mode": 1}, **base)["success"])

    def test_enum_validation_rejects_bad_value_with_choices(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"input_slot": "particles", "job_uid": "J1",
                     "output_group": "particles"}],
            params={"class2D_mode": "bogus"},
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["param"], "class2D_mode")
        self.assertIn("on", result["available_choices"])
        self.workspace.create_job.assert_not_called()

    # -- slot validation ----------------------------------------------------

    def test_unknown_slot_lists_available_slots(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"input_slot": "volumes", "job_uid": "J1",
                     "output_group": "particles"}],
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["unknown_slots"], ["volumes"])
        self.assertEqual(result["available_slots"], ["particles"])
        self.workspace.create_job.assert_not_called()

    def test_required_slot_count_enforced(self):
        result = self.tools.run_cryosparc_job("P1", "W1", "class_2D", inputs=[])
        self.assertFalse(result["success"])
        self.assertIn("requires at least 1", result["error"])
        self.assertEqual(result["slot"], "particles")
        self.workspace.create_job.assert_not_called()

    def test_count_max_enforced(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "sharpen",
            inputs=[
                {"input_slot": "volume", "job_uid": "J3", "output_group": "volume"},
                {"input_slot": "volume", "job_uid": "J3", "output_group": "volume"},
            ],
        )
        self.assertFalse(result["success"])
        self.assertIn("at most 1", result["error"])
        self.workspace.create_job.assert_not_called()

    # -- output group validation (never substitute) -------------------------

    def test_unknown_output_group_lists_available_groups(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"input_slot": "particles", "job_uid": "J2",
                     "output_group": "particles"}],  # J2 has no plain 'particles'
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["requested_output_group"], "particles")
        self.assertEqual(result["available_output_groups"],
                         ["particles_selected", "particles_excluded"])
        # The group name must NOT be silently substituted.
        self.workspace.create_job.assert_not_called()

    def test_uninspectable_source_job_is_an_error(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"input_slot": "particles", "job_uid": "J_missing",
                     "output_group": "particles"}],
        )
        self.assertFalse(result["success"])
        self.assertIn("J_missing", result["error"])
        self.workspace.create_job.assert_not_called()

    # -- connection shapes --------------------------------------------------

    def test_repeated_connections_become_list_of_tuples(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "homo_reconstruct",
            inputs=[
                {"input_slot": "particles", "job_uid": "J1", "output_group": "particles"},
                {"input_slot": "particles", "job_uid": "J2", "output_group": "particles_selected"},
            ],
        )
        self.assertTrue(result["success"], result)
        connections = self.workspace.create_job.call_args.kwargs["connections"]
        self.assertEqual(connections["particles"],
                         [("J1", "particles"), ("J2", "particles_selected")])

    def test_mapping_form_single_dict_and_tuple_entry(self):
        # single connection dict -> bare tuple
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs={"particles": {"job_uid": "J1", "output_group": "particles"}},
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(self.workspace.create_job.call_args.kwargs["connections"],
                         {"particles": ("J1", "particles")})

        self.workspace.create_job.reset_mock()
        # single tuple entry -> bare tuple
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs={"particles": ("J1", "particles")},
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(self.workspace.create_job.call_args.kwargs["connections"],
                         {"particles": ("J1", "particles")})

    def test_mapping_form_list_of_connections(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "homo_reconstruct",
            inputs={"particles": [
                {"job_uid": "J1", "output_group": "particles"},
                ("J2", "particles_selected"),
            ]},
        )
        self.assertTrue(result["success"], result)
        connections = self.workspace.create_job.call_args.kwargs["connections"]
        self.assertEqual(connections["particles"],
                         [("J1", "particles"), ("J2", "particles_selected")])

    def test_malformed_inputs_return_error(self):
        result = self.tools.run_cryosparc_job("P1", "W1", "class_2D", inputs="J1")
        self.assertFalse(result["success"])
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"job_uid": "J1", "output_group": "particles"}],  # no input_slot
        )
        self.assertFalse(result["success"])
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs={"particles": {"job_uid": "J1"}},  # no output_group
        )
        self.assertFalse(result["success"])
        self.workspace.create_job.assert_not_called()

    # -- exclusions ---------------------------------------------------------

    def test_irregular_jobs_rejected_with_curated_tool_pointer(self):
        expected = {
            "import_volumes": "import_volumes",
            "validation": "compute_fsc_validation",
            "import_particles": "import_particles_from_star",
        }
        for job_type, curated in expected.items():
            result = self.tools.run_cryosparc_job("P1", "W1", job_type)
            self.assertFalse(result["success"], job_type)
            self.assertEqual(result["use_tool"], curated, job_type)
            self.assertIn(curated, result["error"])
        self.workspace.create_job.assert_not_called()

    def test_destructive_live_administration_is_rejected(self):
        self.tools = _make_tools(specs=_specs(
            _job_spec("delete_live_datatype", params_base={}),
        ))
        result = self.tools.run_cryosparc_job("P1", "W1", "delete_live_datatype")
        self.assertFalse(result["success"])
        self.assertIn("destructive/administrative", result["error"])
        self.workspace.create_job.assert_not_called()

    # -- dry run ------------------------------------------------------------

    def test_dry_run_validates_but_never_creates(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "homo_reconstruct",
            inputs=[{"input_slot": "particles", "job_uid": "J1",
                     "output_group": "particles"}],
            params={"refine_symmetry": "C2"},
            dry_run=True,
        )
        self.assertTrue(result["success"], result)
        self.assertTrue(result["dry_run"])
        self.assertEqual(result["job_type"], "homo_reconstruct")
        self.assertEqual(result["connections"], {"particles": ("J1", "particles")})
        self.assertEqual(result["params"], {"refine_symmetry": "C2"})
        # Source groups were still inspected (read-only) ...
        self.jobs["J1"].refresh.assert_called()
        # ... but nothing was created or queued.
        self.project.find_workspace.assert_not_called()
        self.workspace.create_job.assert_not_called()
        self.created.queue.assert_not_called()
        self.assertEqual(self.tools._job_cache, {})

    def test_dry_run_still_enforces_validation(self):
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"input_slot": "particles", "job_uid": "J2",
                     "output_group": "particles"}],
            dry_run=True,
        )
        self.assertFalse(result["success"])
        self.workspace.create_job.assert_not_called()

    # -- queue / wait --------------------------------------------------------

    def test_queue_and_wait_with_mocks(self):
        self.tools.wait_for_job_completion = Mock(
            return_value={"status": "completed", "progress": 100})

        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"input_slot": "particles", "job_uid": "J1",
                     "output_group": "particles"}],
            wait_for_completion=True, timeout=10, check_interval=1,
        )

        self.assertTrue(result["success"], result)
        self.created.queue.assert_called_once_with(lane=None, hostname=None)
        self.tools.wait_for_job_completion.assert_called_once_with(
            project_uid="P1", job_uid="J99", workspace_uid="W1",
            timeout=10, check_interval=1)
        self.assertEqual(result["status"], "completed")

    def test_unexpected_exception_returns_error_dict(self):
        self.tools.cs.find_project.side_effect = RuntimeError("connection lost")
        result = self.tools.run_cryosparc_job(
            "P1", "W1", "class_2D",
            inputs=[{"input_slot": "particles", "job_uid": "J1",
                     "output_group": "particles"}],
        )
        self.assertFalse(result["success"])
        self.assertIn("connection lost", result["error"])


# ---------------------------------------------------------------------------
# Orientation diagnostics metrics
# ---------------------------------------------------------------------------

class OrientationDiagnosticsMetricsTests(unittest.TestCase):
    def test_existing_diagnostics_exposes_cfar_and_scf_star(self):
        tools = _make_tools()
        refinement = _mock_job("J1", job_type="nonuniform_refine_new")
        refinement.doc["children"] = ["J9"]
        diagnostics = _mock_job(
            "J9", [("volume_3DFSC", "volume")],
            job_type="orientation_diagnostics",
        )
        diagnostics.doc["output_result_groups"][0]["latest_summary_stats"] = {
            "cfar": 0.12,
            "scf_star": 0.55,
        }
        tools.cs.find_job.side_effect = lambda project_uid, uid: {
            "J1": refinement, "J9": diagnostics,
        }[uid]

        result = tools._find_orientation_diagnostics("P1", "J1")
        self.assertEqual(result, {
            "job_uid": "J9", "cfar": 0.12, "scf_star": 0.55,
        })
        self.assertIn("cause not determined", tools._cfar_label(0.12))
        self.assertNotIn("no real structure", tools._cfar_label(0.12))


# ---------------------------------------------------------------------------
# rebalance_orientations
# ---------------------------------------------------------------------------

class RebalanceOrientationsTests(unittest.TestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.tools.run_cryosparc_job = Mock(return_value={
            "success": True, "job_type": "rebalance_3D", "dry_run": True,
        })

    def test_curated_tool_uses_real_job_type_and_documented_defaults(self):
        result = self.tools.rebalance_orientations(
            "P1", "W1", "J1", particles_group_name="particles_selected",
            dry_run=True,
        )
        self.assertTrue(result["success"], result)
        kwargs = self.tools.run_cryosparc_job.call_args.kwargs
        self.assertEqual(kwargs["job_type"], "rebalance_3D")
        self.assertEqual(kwargs["inputs"], [{
            "input_slot": "particles",
            "job_uid": "J1",
            "output_group": "particles_selected",
        }])
        self.assertEqual(kwargs["params"], {
            "rebalance3D_N_bins": 192,
            "rebalance3D_reject_pct": 80.0,
            "rebalance3D_reject_by": "random",
        })
        self.assertIs(kwargs["dry_run"], True)

    def test_infers_particle_group_and_merges_controlled_overrides(self):
        self.tools._resolve_particles_slot = Mock(return_value="particles")
        self.tools.rebalance_orientations(
            "P1", "W1", "J1", num_orientation_bins=256,
            rebalance_percentile=75, exclusion_criterion="alignments3D/error",
            params={"rebalance3D_reject_pct": 65},
        )
        self.tools._resolve_particles_slot.assert_called_once()
        kwargs = self.tools.run_cryosparc_job.call_args.kwargs
        self.assertEqual(kwargs["inputs"][0]["output_group"], "particles")
        self.assertEqual(kwargs["params"], {
            "rebalance3D_N_bins": 256,
            "rebalance3D_reject_pct": 65,
            "rebalance3D_reject_by": "alignments3D/error",
        })

    def test_group_resolution_failure_never_submits(self):
        self.tools.cs.find_project.side_effect = RuntimeError("missing source")
        result = self.tools.rebalance_orientations("P1", "W1", "J404")
        self.assertFalse(result["success"])
        self.assertIn("missing source", result["error"])
        self.tools.run_cryosparc_job.assert_not_called()


# ---------------------------------------------------------------------------
# merge_particles
# ---------------------------------------------------------------------------

class MergeParticlesTests(unittest.TestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.jobs = {
            "J1": _mock_job("J1", [("particles", "particle", 100)]),
            "J2": _mock_job("J2", [("particles_selected", "particle", 80),
                                   ("particles_excluded", "particle", 20)]),
        }
        self.project, self.workspace = _wire_project(self.tools, self.jobs)
        self.workspace.create_job.return_value = _mock_job("J50")
        self.tools.wait_for_job_completion = Mock(
            return_value={"status": "completed"})

    def test_groups_all_sources_into_one_particles_input(self):
        result = self.tools.merge_particles("P1", "W1", ["J1", "J2"])
        self.assertTrue(result["success"], result)

        call = self.workspace.create_job.call_args
        self.assertEqual(call.args[0], "downsample_particles")
        # Both stacks on the single 'particles' input as a list of tuples;
        # J2 auto-resolved to its preferred particles_selected group.
        self.assertEqual(call.kwargs["connections"],
                         {"particles": [("J1", "particles"),
                                        ("J2", "particles_selected")]})

    def test_default_params_concatenate_without_cropping(self):
        self.tools.merge_particles("P1", "W1", ["J1", "J2"])
        params = self.workspace.create_job.call_args.kwargs["params"]
        self.assertEqual(params, {"box_size_pix": None, "bin_size_pix": None})

    def test_explicit_group_names_used_verbatim(self):
        result = self.tools.merge_particles(
            "P1", "W1", ["J1", "J2"],
            particles_group_names=["particles", "particles_selected"],
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(
            self.workspace.create_job.call_args.kwargs["connections"],
            {"particles": [("J1", "particles"), ("J2", "particles_selected")]})
        # Explicit names: no per-job slot resolution needed.
        self.jobs["J1"].refresh.assert_not_called()
        self.jobs["J2"].refresh.assert_not_called()

    def test_group_names_length_mismatch_is_an_error(self):
        result = self.tools.merge_particles(
            "P1", "W1", ["J1", "J2"], particles_group_names=["particles"])
        self.assertFalse(result["success"])
        self.assertIn("one entry per source", result["error"])
        self.workspace.create_job.assert_not_called()

    def test_no_source_jobs_is_an_error(self):
        result = self.tools.merge_particles("P1", "W1", [])
        self.assertFalse(result["success"])
        self.workspace.create_job.assert_not_called()

    def test_params_override_no_crop_defaults(self):
        self.tools.merge_particles("P1", "W1", ["J1", "J2"],
                                   params={"bin_size_pix": 128})
        params = self.workspace.create_job.call_args.kwargs["params"]
        self.assertEqual(params, {"box_size_pix": None, "bin_size_pix": 128})

    def test_queue_wait_and_cache(self):
        result = self.tools.merge_particles("P1", "W1", ["J1", "J2"])
        self.assertEqual(result["job_uid"], "J50")
        self.assertEqual(result["cryosparc_job_type"], "downsample_particles")
        self.assertEqual(result["merged_from"], ["J1", "J2"])
        self.assertEqual(result["status"], "completed")  # waited by default
        self.assertEqual(self.tools._job_cache["J50"],
                         {"project_uid": "P1", "workspace_uid": "W1"})
        self.tools.wait_for_job_completion.assert_called_once()


# ---------------------------------------------------------------------------
# homogeneous_reconstruction
# ---------------------------------------------------------------------------

class HomogeneousReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.jobs = {
            "J1": _mock_job("J1", [("particles", "particle", 100)]),
            "J5": _mock_job("J5", [("mask", "mask", 1)]),
        }
        self.project, self.workspace = _wire_project(self.tools, self.jobs)
        self.workspace.create_job.return_value = _mock_job("J60")

    def test_uses_raw_homo_reconstruct_job_type(self):
        result = self.tools.homogeneous_reconstruction(
            "P1", "W1", "J1", mask_job_uid="J5")
        self.assertTrue(result["success"], result)

        call = self.workspace.create_job.call_args
        self.assertEqual(call.args[0], "homo_reconstruct")
        self.assertEqual(call.kwargs["connections"],
                         {"particles": ("J1", "particles"),
                          "mask": ("J5", "mask")})
        self.assertEqual(result["job_type"], "homo_reconstruct")
        self.assertEqual(result["job_uid"], "J60")

    def test_mask_input_is_optional(self):
        result = self.tools.homogeneous_reconstruction("P1", "W1", "J1")
        self.assertTrue(result["success"], result)
        connections = self.workspace.create_job.call_args.kwargs["connections"]
        self.assertEqual(connections, {"particles": ("J1", "particles")})

    def test_docstring_clarifies_not_ab_initio(self):
        doc = CryoSPARCTools.homogeneous_reconstruction.__doc__
        self.assertIn("ALREADY", doc)
        self.assertIn("NOT ab-initio", doc)


if __name__ == "__main__":
    unittest.main()
