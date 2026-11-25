import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cryoagent.core.experiment_auditor import (
    ExperimentAuditor,
    attach_exact_run_params,
    audit_improvement_log,
    enrich_context_from_workflow_state,
    extract_improvement_context_from_log,
    final_report_from_log_text,
    harvest_log_evidence,
    parse_experiment_record,
)
from cryoagent.core.improvement_agent import ImprovementAgent


ROOT = Path(__file__).resolve().parents[1]
TRIAL_5 = ROOT / "outputs/llm_conversation_improvement_20261124_125725.log"
TRIAL_6 = ROOT / "outputs/llm_conversation_improvement_20261124_141042.log"
STATE = ROOT / "outputs/workflow_state.json"


class _Response:
    def __init__(self, content):
        self.content = content


class _LLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _Response(self.content)


def _run(params=None, inputs=None, job_type="nonuniform_refine_new"):
    return {
        "success": True,
        "job_type": job_type,
        "params": params if params is not None else {"refine_num_final_iterations": 0, "random_seed": 1},
        "inputs": inputs if inputs is not None else [
            {"slot": "particles", "job_uid": "J296", "group_name": "particles"},
            {"slot": "volume", "job_uid": "J303", "group_name": "volume"},
        ],
        "source": "test_exact",
    }


def _report(experiments, prose="No exhaustion claim is made; untested hypotheses remain future work.", branch=None):
    record = {
        "baseline": {"job": "J303", "resolution": 2.59, "cfar": 0.668, "particles": 227745},
        "experiments": experiments,
        "best_result": {"job": "J303", "resolution": 2.59, "cfar": 0.668},
        "branch_in_progress": branch,
    }
    return prose + "\n```json\n" + json.dumps(record) + "\n```"


def _experiment(uid="J361", resolution=2.65, cfar=0.641, **kwargs):
    value = {
        "hypothesis": "A terminal refinement tests one declared hypothesis.",
        "plan": ["nonuniform_refinement"],
        "result_job": uid,
        "resolution": resolution,
        "cfar": cfar,
        "particles": 227745,
        "verdict": "worse",
        "one_line_reason": "Measured result did not improve the baseline.",
    }
    value.update(kwargs)
    return value


class DeterministicAuditorTests(unittest.TestCase):
    def test_corrected_controlled_report_passes_and_negation_is_not_a_claim(self):
        context = {
            "baseline": {"job": "J303", "resolution": 2.59, "cfar": 0.668},
            "run_params": {
                "J303": _run(),
                "J361": _run({"refine_num_final_iterations": 10, "random_seed": 2}),
                "J363": _run({"refine_num_final_iterations": 10, "random_seed": 3}),
            },
            "job_metrics": {
                "J303": {"resolution": 2.59, "cfar": 0.668, "particles": 227745},
                "J361": {"resolution": 2.65, "cfar": 0.641, "particles": 227745},
                "J363": {"resolution": 2.66, "cfar": 0.640, "particles": 227745},
            },
        }
        text = _report([
            _experiment(),
            _experiment("J363", 2.66, 0.640),
        ], prose=(
            "No noise floor was established; J320 was not a same-set control. "
            "This was not a single-key comparison. Two valid experiments had no gain; stop condition met."
        ))
        result = ExperimentAuditor().audit(context, text)
        self.assertEqual(result["status"], "pass", result["findings"])

    def test_single_key_claim_reports_every_non_seed_difference(self):
        context = {
            "baseline": {"job": "J303"},
            "run_params": {
                "J303": _run(),
                "J361": _run({"a": 1, "b": 0, "random_seed": 1}),
                "J363": _run({"a": 2, "b": 1, "random_seed": 99}),
            },
            "job_metrics": {},
        }
        exp = _experiment("J363", one_line_reason="single-key versus control J361; only random_seed otherwise differs")
        result = ExperimentAuditor().audit(context, _report([_experiment(), exp], prose="No stopping claim."))
        finding = next(f for f in result["findings"] if f["check_id"] == "parameter_identity")
        self.assertEqual(finding["evidence"]["non_seed_param_diff"], {"a": 2, "b": 1})
        self.assertNotIn("random_seed", finding["evidence"]["non_seed_param_diff"])

    def test_noise_floor_requires_two_matched_reruns_and_same_inputs(self):
        other_input = _run(inputs=[{"slot": "particles", "job_uid": "J319", "group_name": "particles"}])
        context = {
            "baseline": {"job": "J303"},
            "run_params": {
                "J303": _run(),
                "J320": _run({"refine_num_final_iterations": 0, "random_seed": 2}, inputs=other_input["inputs"]),
            },
            "job_metrics": {"J303": {"resolution": 2.59}, "J320": {"resolution": 2.65}},
        }
        insufficient = _report([_experiment("J320", 2.65)], prose="J320 establishes a same-set noise floor.")
        result = ExperimentAuditor().audit(context, insufficient)
        finding = next(f for f in result["findings"] if f["check_id"] == "noise_floor")
        self.assertEqual(finding["severity"], "critical")
        self.assertEqual(finding["evidence"]["independent_matched_reruns"], 0)

        # Two additional reruns with identical non-seed params AND connections do
        # support a scatter estimate; the differently-connected J320 never counts.
        context["run_params"].update({
            "J400": _run({"refine_num_final_iterations": 0, "random_seed": 3}),
            "J401": _run({"refine_num_final_iterations": 0, "random_seed": 4}),
        })
        context["job_metrics"].update({"J400": {"resolution": 2.61}, "J401": {"resolution": 2.62}})
        result = ExperimentAuditor().audit(context, insufficient)
        self.assertNotIn("noise_floor", {f["check_id"] for f in result["findings"]})

    def test_explicit_union_acknowledged_as_invalid_does_not_block(self):
        superclass = _run(inputs=[{"slot": "particles", "job_uid": "J373", "group_name": "particles_superclass_1"}])
        context = {
            "baseline": {"job": "J303"},
            "run_params": {"J303": _run(), "J374": superclass},
            "job_metrics": {"J303": {"resolution": 2.59}, "J374": {"resolution": 2.71}},
        }
        bad = _experiment(
            "J374", 2.71,
            hypothesis="The union of particles_class_0 and particles_class_3 will help.",
            plan=["regroup_classes", "nonuniform_refinement"],
            one_line_reason="The superclass was treated as the clean union.",
        )
        result = ExperimentAuditor().audit(context, _report([bad], prose="No stopping claim."))
        self.assertIn("particle_plan", {f["check_id"] for f in result["findings"]})
        corrected = copy.deepcopy(bad)
        corrected["verdict"] = "invalid"
        corrected["one_line_reason"] = "J374 used a superclass instead of the planned union; it did not test the class union."
        branch = {"plan": ["nonuniform_refinement"], "next_job": "exact class union", "jobs_so_far": ["J370"]}
        result = ExperimentAuditor().audit(context, _report(
            [corrected], prose="The exact union remains untested; no stopping claim is made.", branch=branch,
        ))
        self.assertNotIn("particle_plan", {f["check_id"] for f in result["findings"]})

    def test_two_no_gain_is_optional_but_false_narrative_is_corrected(self):
        context = {
            "baseline": {"job": "J303", "resolution": 2.59, "cfar": 0.668},
            "run_params": {"J303": _run(), "J361": _run(), "J363": _run()},
            "job_metrics": {
                "J303": {"resolution": 2.59}, "J361": {"resolution": 2.50, "cfar": 0.68},
                "J363": {"resolution": 2.66, "cfar": 0.64},
            },
        }
        experiments = [
            _experiment(resolution=2.50, cfar=0.68, verdict="helped"),
            _experiment("J363", 2.66, 0.64, verdict="worse"),
        ]
        result = ExperimentAuditor().audit(context, _report(
            experiments, prose="Two consecutive valid experiments had no gain; stop condition met.",
        ))
        finding = next(f for f in result["findings"] if f["check_id"] == "stopping_consistency")
        self.assertEqual(finding["severity"], "critical")
        self.assertIn("wording correction only", finding["required_action"])
        self.assertIn("do not run new experiments", finding["required_action"])
        self.assertNotIn("premature_stop", {f["check_id"] for f in result["findings"]})

        # The same outcomes may be paused without any two-no-gain justification.
        result = ExperimentAuditor().audit(context, _report(
            experiments, prose="Pausing here; additional hypotheses remain untested.",
        ))
        self.assertEqual(result["status"], "pass", result["findings"])

    def test_true_no_gain_narrative_is_allowed_not_required(self):
        context = {
            "baseline": {"job": "J303", "resolution": 2.59, "cfar": 0.668},
            "run_params": {"J303": _run(), "J361": _run(), "J363": _run()},
            "job_metrics": {
                "J303": {"resolution": 2.59}, "J361": {"resolution": 2.65},
                "J363": {"resolution": 2.66},
            },
        }
        result = ExperimentAuditor().audit(context, _report([
            _experiment(), _experiment("J363", 2.66, 0.64),
        ], prose="Two consecutive terminal experiments had no gain."))
        self.assertEqual(result["status"], "pass", result["findings"])
        self.assertNotIn("stopping_consistency", {f["check_id"] for f in result["findings"]})

    def test_open_branch_pause_passes_but_false_exhaustion_is_wording_error(self):
        context = {
            "baseline": {"job": "J303", "resolution": 2.59, "cfar": 0.668},
            "run_params": {"J303": _run(), "J361": _run()},
            "job_metrics": {"J303": {"resolution": 2.59}, "J361": {"resolution": 2.65}},
        }
        branch = {"plan": ["nonuniform_refinement"], "next_job": "exact class union", "jobs_so_far": ["J361"]}
        result = ExperimentAuditor().audit(context, _report(
            [_experiment()], prose="Pausing; the exact-union branch remains open.", branch=branch,
        ))
        self.assertEqual(result["status"], "pass", result["findings"])

        result = ExperimentAuditor().audit(context, _report(
            [_experiment()], prose="The ranking is exhausted.", branch=branch,
        ))
        finding = next(f for f in result["findings"] if f["check_id"] == "stopping_consistency")
        self.assertEqual(finding["evidence"]["branch_in_progress"], branch)
        self.assertIn("wording correction only", finding["required_action"])

        invalid = _experiment("J368", None, None, verdict="invalid")
        result = ExperimentAuditor().audit(context, _report(
            [_experiment(), invalid], prose="Everything testable has been tried.",
        ))
        finding = next(f for f in result["findings"] if f["check_id"] == "stopping_consistency")
        self.assertEqual(finding["evidence"]["untested_hypothesis_jobs"], ["J368"])
        self.assertIn("do not run new experiments", finding["required_action"])

    def test_capability_retraction_is_not_a_stale_unavailable_claim(self):
        tools = SimpleNamespace(settings=SimpleNamespace(topaz_exec_path="/configured/topaz"))
        context = {
            "baseline": {"job": "J303"},
            "run_params": {"J303": _run(), "J361": _run()},
            "job_metrics": {"J303": {"resolution": 2.59}, "J361": {"resolution": 2.65}},
        }
        text = _report([_experiment()], prose=(
            "Topaz is configured but was not tested this round; the previous Topaz unavailable claim was stale. "
            "No stopping claim is made."
        ))
        result = ExperimentAuditor(cryosparc_tools=tools).audit(context, text)
        self.assertNotIn("capability_freshness", {f["check_id"] for f in result["findings"]})

    def test_measurement_and_helped_verdict_must_match_evidence(self):
        context = {
            "baseline": {"job": "J303", "resolution": 2.59, "cfar": 0.668},
            "run_params": {"J303": _run(), "J361": _run()},
            "job_metrics": {"J303": {"resolution": 2.59}, "J361": {"resolution": 2.65, "cfar": 0.64}},
        }
        result = ExperimentAuditor().audit(context, _report([
            _experiment(resolution=2.67, cfar=0.64, verdict="helped"),
        ], prose="No stopping claim."))
        ids = {f["check_id"] for f in result["findings"]}
        self.assertIn("measurement_consistency", ids)
        self.assertIn("verdict_consistency", ids)


class SemanticCriticTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "baseline": {"job": "J303"},
            "run_params": {"J303": _run(), "J361": _run()},
            "job_metrics": {"J303": {"resolution": 2.59}, "J361": {"resolution": 2.65}},
        }

    def test_contradicted_semantic_claim_blocks(self):
        llm = _LLM(json.dumps({"claims": [{
            "claim": "The result proves all discarded classes were junk.",
            "verdict": "contradicted", "material_accuracy": False,
            "rationale": "Class flexibility and particle count were not controlled.",
            "required_action": "Calibrate the diagnosis.",
        }], "summary": "Causal overclaim"}))
        result = ExperimentAuditor(critic_llm=llm).audit(self.context, _report([_experiment()]))
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["findings"][-1]["check_id"], "semantic_critic")
        self.assertFalse(result["findings"][-1]["evidence"]["material_accuracy"])
        self.assertIn("no new experiment is required", result["findings"][-1]["required_action"])
        self.assertEqual(len(llm.calls), 1)

    def test_material_causal_claim_blocks_but_legacy_stop_flag_is_ignored(self):
        llm = _LLM(json.dumps({"claims": [
            {
                "claim": "The result is N-limited, not junk-limited.",
                "verdict": "unsupported", "material_accuracy": True, "blocks_stop": False,
                "rationale": "Particle count and class composition are confounded.",
                "required_action": "Run twenty additional jobs.",
            },
            {
                "claim": "Every possible strategy is exhausted.",
                "verdict": "unsupported", "material_accuracy": False, "blocks_stop": True,
                "rationale": "Other hypotheses remain.", "required_action": "Keep running jobs.",
            },
        ], "summary": "Accuracy and completeness are different"}))
        result = ExperimentAuditor(critic_llm=llm).audit(self.context, _report([_experiment()]))
        semantic = [f for f in result["findings"] if f["check_id"] == "semantic_critic"]
        self.assertEqual([f["severity"] for f in semantic], ["critical", "warning"])
        self.assertIn("optional future work", semantic[0]["required_action"])
        self.assertNotIn("Run twenty", semantic[0]["required_action"])
        self.assertEqual(semantic[0]["evidence"]["critic_recommendation"], "Run twenty additional jobs.")
        self.assertIn("never requires new experiments", semantic[1]["required_action"])

    def test_nonmaterial_exhaustion_claim_is_only_a_warning(self):
        llm = _LLM(json.dumps({"claims": [{
            "claim": "All strategies are exhausted.", "verdict": "unsupported",
            "material_accuracy": False, "blocks_stop": True,
            "rationale": "The report lists open hypotheses.", "required_action": "Run more jobs.",
        }], "summary": "Completeness is not an accuracy failure"}))
        result = ExperimentAuditor(critic_llm=llm).audit(self.context, _report([_experiment()]))
        self.assertEqual(result["status"], "pass_with_warnings", result["findings"])
        self.assertEqual(result["critical_count"], 0)
        self.assertEqual(result["findings"][-1]["severity"], "warning")

    def test_malformed_critic_is_nonblocking(self):
        result = ExperimentAuditor(critic_llm=_LLM("not JSON")).audit(
            self.context, _report([_experiment()], prose="No stopping claim."),
        )
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["findings"][0]["severity"], "info")


class ImprovementSafetyCapTests(unittest.TestCase):
    def test_iteration_cap_is_a_large_configurable_runaway_guard(self):
        agent = ImprovementAgent.__new__(ImprovementAgent)
        agent.config = SimpleNamespace(agent=SimpleNamespace(
            improvement_max_iterations=None, max_iterations=50,
        ))
        self.assertEqual(agent._max_iterations(), 1000)
        agent.config.agent.improvement_max_iterations = 1200
        self.assertEqual(agent._max_iterations(), 1200)
        agent.config.agent.improvement_max_iterations = None
        agent.config.agent.max_iterations = 1500
        self.assertEqual(agent._max_iterations(), 1500)


class AuditGateTests(unittest.TestCase):
    def _agent(self, tmp, audit_result):
        agent = ImprovementAgent.__new__(ImprovementAgent)
        agent._parse_tool_input = json.loads
        agent.realtime_logger = SimpleNamespace(current_log_file=str(Path(tmp) / "llm_conversation_improvement_test.log"))
        Path(agent.realtime_logger.current_log_file).write_text("saved evidence")
        agent.config = SimpleNamespace(workflow=SimpleNamespace(project_uid="P8"))
        agent.cryosparc_tools = None
        agent.llm = None
        agent.outputs_dir = tmp
        agent.audit_critic_enabled = False
        agent.audit_max_attempts = 3
        agent.audit_attempts = 0
        agent.last_audit_report = None
        agent.last_audit_artifact = None
        return agent

    @patch("cryoagent.core.experiment_auditor.enrich_context_from_workflow_state", side_effect=lambda context, path: context)
    @patch("cryoagent.core.experiment_auditor.harvest_log_evidence", side_effect=lambda path, context: context)
    @patch("cryoagent.core.experiment_auditor.attach_exact_run_params", side_effect=lambda context, tools, project: context)
    @patch("cryoagent.core.experiment_auditor.extract_improvement_context_from_log", return_value={})
    def test_rejections_are_bounded_and_artifacts_saved(self, extract, attach, harvest, enrich):
        critical = {"check_id": "test", "severity": "critical", "claim": "bad", "evidence": {}, "required_action": "fix"}
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._agent(tmp, None)
            auditor = Mock()
            auditor.audit.return_value = {"status": "fail", "findings": [critical], "critical_count": 1, "warning_count": 0}
            with patch("cryoagent.core.experiment_auditor.ExperimentAuditor", return_value=auditor):
                first = json.loads(agent._submit_final_report_tool(json.dumps({"report": "complete report"})))
                second = json.loads(agent._submit_final_report_tool(json.dumps({"report": "complete report"})))
                third = json.loads(agent._submit_final_report_tool(json.dumps({"report": "complete report"})))
            self.assertEqual(first["status"], "fail")
            self.assertEqual(first["attempts_remaining"], 2)
            self.assertEqual(second["attempts_remaining"], 1)
            self.assertEqual(third["status"], "fail_unresolved")
            self.assertEqual(third["attempts_remaining"], 0)
            self.assertTrue(Path(third["artifact"]).exists())
            self.assertEqual(agent.last_audit_report["status"], "fail_unresolved")

    def test_improvement_toolset_contains_required_gate(self):
        agent = ImprovementAgent.__new__(ImprovementAgent)
        agent.audit_enabled = True
        agent._method_providers = {}
        tools = ImprovementAgent._create_tools(agent)
        self.assertIn("submit_final_report", {tool.name for tool in tools})


@unittest.skipUnless(TRIAL_5.exists() and TRIAL_6.exists() and STATE.exists(), "real replay fixtures unavailable")
class RealLogReplayTests(unittest.TestCase):
    def setUp(self):
        self.tools = SimpleNamespace(settings=SimpleNamespace(topaz_exec_path="/configured/topaz"))
        self.state_before = STATE.read_bytes()

    def tearDown(self):
        self.assertEqual(STATE.read_bytes(), self.state_before, "audit must not mutate the baseline/ratchet state")

    def _audit(self, path):
        return audit_improvement_log(path, cryosparc_tools=self.tools, state_path=STATE)

    def test_trial_5_rejects_known_false_controls(self):
        result = self._audit(TRIAL_5)
        critical = [f for f in result["findings"] if f["severity"] == "critical"]
        ids = {f["check_id"] for f in critical}
        self.assertEqual(ids, {"parameter_identity", "noise_floor", "capability_freshness"})
        parameter = next(f for f in critical if f["check_id"] == "parameter_identity")
        self.assertEqual(parameter["evidence"]["result"], "J363")
        self.assertEqual(set(parameter["evidence"]["non_seed_param_diff"]), {
            "refine_relax_symmetry", "refine_noise_init_from_imgs",
        })
        noise = next(f for f in critical if f["check_id"] == "noise_floor")
        self.assertNotIn("J320", noise["evidence"]["matched_jobs"])

    def test_trial_6_rejects_j374_but_not_j371_particle_inputs(self):
        result = self._audit(TRIAL_6)
        critical = [f for f in result["findings"] if f["severity"] == "critical"]
        ids = {f["check_id"] for f in critical}
        self.assertEqual(ids, {"particle_plan", "noise_floor", "capability_freshness", "stopping_consistency"})
        stopping = next(f for f in critical if f["check_id"] == "stopping_consistency")
        self.assertEqual(stopping["evidence"]["untested_hypothesis_jobs"], ["J368"])
        self.assertIn("wording correction only", stopping["required_action"])
        particle = next(f for f in critical if f["check_id"] == "particle_plan")
        self.assertEqual(particle["evidence"]["result"], "J374")
        self.assertEqual(particle["evidence"]["actual_particle_groups"], ["particles_superclass_1"])
        self.assertFalse(any(f["check_id"] == "parameter_identity" for f in critical))

    def test_corrected_trial_reports_can_pass_without_changing_measurements(self):
        report5 = final_report_from_log_text(TRIAL_5.read_text(encoding="utf-8", errors="ignore"))
        record5 = parse_experiment_record(report5)
        record5["experiments"][0]["one_line_reason"] = "Ten final iterations did not improve the baseline."
        record5["experiments"][1]["one_line_reason"] = (
            "Two non-seed keys changed versus J361; the result is worse and does not isolate symmetry."
        )
        corrected5 = "Two valid combined-parameter experiments had no gain; stop condition met.\n```json\n" + json.dumps(record5) + "\n```"
        context5 = extract_improvement_context_from_log(TRIAL_5)
        attach_exact_run_params(context5, None, None)
        harvest_log_evidence(TRIAL_5, context5)
        enrich_context_from_workflow_state(context5, STATE)
        result5 = ExperimentAuditor(cryosparc_tools=self.tools).audit(context5, corrected5)
        self.assertEqual(result5["status"], "pass", result5["findings"])

        report6 = final_report_from_log_text(TRIAL_6.read_text(encoding="utf-8", errors="ignore"))
        record6 = parse_experiment_record(report6)
        j374 = next(exp for exp in record6["experiments"] if exp["result_job"] == "J374")
        j374["verdict"] = "invalid"
        j374["one_line_reason"] = "J374 used a superclass instead of the planned union; it did not test the exact class union."
        record6["branch_in_progress"] = {
            "plan": ["nonuniform_refinement with particles_class_0 and particles_class_3"],
            "next_job": "terminal exact-union refinement", "jobs_so_far": ["J370"],
        }
        corrected6 = "The exact class union remains untested; no stopping claim is made.\n```json\n" + json.dumps(record6) + "\n```"
        context6 = extract_improvement_context_from_log(TRIAL_6)
        attach_exact_run_params(context6, None, None)
        harvest_log_evidence(TRIAL_6, context6)
        enrich_context_from_workflow_state(context6, STATE)
        result6 = ExperimentAuditor(cryosparc_tools=self.tools).audit(context6, corrected6)
        self.assertEqual(result6["status"], "pass", result6["findings"])


if __name__ == "__main__":
    unittest.main()
