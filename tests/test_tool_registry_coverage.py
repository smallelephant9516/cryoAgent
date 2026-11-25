import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from cryoagent.core.base_react_agent import BaseReActAgent
from cryoagent.core import cryosparc_tool_registry as reg
from cryoagent.core.cryosparc_2Doptimize.optimizer_2d_agent import Optimizer2DAgent
from cryoagent.core.cryosparc_heterogeneity.heterogeneity_agent import HeterogeneityAgent
from cryoagent.core.cryosparc_heterogeneity_depth.heterogeneity_depth_agent import HeterogeneityDepthAgent
from cryoagent.core.cryosparc_optimize.optimizer_agent import OptimizerAgent
from cryoagent.core.cryosparc_picking.picking_agent import PickingAgent
from cryoagent.core.cryosparc_polish.polish_agent import PolishAgent
from cryoagent.core.cryosparc_preprocessing.preprocessing_agent import PreprocessingAgent
from cryoagent.core.cryosparc_reconstruction.reconstruction_agent import ReconstructionAgent
from cryoagent.core.full_dynamic_agent import FullDynamicAgent
from cryoagent.core.improvement_agent import ImprovementAgent
from cryoagent.tools.cryosparc_tools import CryoSPARCTools


STAGE_CLASSES = {
    "preprocessing": PreprocessingAgent,
    "particle_picking": PickingAgent,
    "optimization_2d": Optimizer2DAgent,
    "reconstruction": ReconstructionAgent,
    "optimization": OptimizerAgent,
    "heterogeneity": HeterogeneityAgent,
    "heterogeneity_depth": HeterogeneityDepthAgent,
    "polish": PolishAgent,
}


def bare_cross_stage_agent(cls=ImprovementAgent, all_stages=True):
    agent = cls.__new__(cls)
    agent._stage_agents = (
        {name: cls_.__new__(cls_) for name, cls_ in STAGE_CLASSES.items()}
        if all_stages else {}
    )
    agent.audit_enabled = False
    agent._method_providers = {}
    agent._collect_tool_methods()
    return agent


class RegistryIntegrityTests(unittest.TestCase):
    def test_stage_sets_reference_known_unique_specs(self):
        for stage, spec_ids in reg.AGENT_TOOL_SETS.items():
            self.assertEqual(len(spec_ids), len(set(spec_ids)), stage)
            for spec_id in spec_ids:
                self.assertIn(spec_id, reg._SPECS, f"{stage}:{spec_id}")

    def test_every_spec_has_an_implementation(self):
        provider_classes = [BaseReActAgent, *STAGE_CLASSES.values()]
        for spec in reg._SPECS.values():
            self.assertTrue(
                any(callable(getattr(cls, spec.method, None)) for cls in provider_classes),
                f"{spec.spec_id} has no provider for {spec.method}",
            )

    def test_every_stage_spec_binds_on_its_own_agent(self):
        for stage, spec_ids in reg.AGENT_TOOL_SETS.items():
            agent = STAGE_CLASSES[stage].__new__(STAGE_CLASSES[stage])
            tools = reg.build_tools(agent, spec_ids)
            self.assertEqual(len(tools), len(spec_ids), stage)

    def test_previously_missing_provider_methods_exist(self):
        for method in (
            "merge_particles", "homogeneous_reconstruction",
            "import_particles_from_star", "import_volumes", "compute_fsc_validation",
            "run_cryosparc_job", "describe_job_inputs", "describe_job_outputs",
            "rebalance_orientations",
        ):
            self.assertTrue(callable(getattr(CryoSPARCTools, method, None)), method)


class CrossStageExposureTests(unittest.TestCase):
    def test_all_providers_expose_every_curated_name(self):
        agent = bare_cross_stage_agent()
        tools = agent._create_tools()
        names = [tool.name for tool in tools]
        expected = list(dict.fromkeys(spec.name for spec in reg._SPECS.values()))
        expected.append("consult_cryosparc_guide")
        self.assertEqual(names, expected)
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("homogeneous_reconstruction", names)
        self.assertIn("merge_particles", names)
        self.assertIn("run_cryosparc_job", names)
        self.assertIn("describe_job_inputs", names)
        self.assertIn("describe_job_outputs", names)

    def test_full_dynamic_uses_the_same_complete_assembly(self):
        agent = bare_cross_stage_agent(FullDynamicAgent)
        names = [tool.name for tool in agent._create_tools()]
        self.assertIn("run_cryosparc_job", names)
        self.assertIn("import_particles_from_star", names)
        self.assertIn("compute_fsc_validation", names)

    def test_missing_providers_are_reported(self):
        agent = bare_cross_stage_agent(all_stages=False)
        with self.assertLogs("ImprovementAgent", level="WARNING") as captured:
            agent._create_tools()
        message = "\n".join(captured.output)
        self.assertIn("import_movies", message)
        self.assertIn("no provider bound", message)


class NewWrapperContractTests(unittest.TestCase):
    def setUp(self):
        self.agent = ImprovementAgent.__new__(ImprovementAgent)
        self.agent.config = SimpleNamespace(
            workflow=SimpleNamespace(project_uid="P1", workspace_uid="W1")
        )
        self.agent.cryosparc_tools = Mock()
        self.agent._record_tool_execution = Mock()
        self.agent.cryosparc_tools.homogeneous_reconstruction.return_value = {"success": True}
        self.agent.cryosparc_tools.run_cryosparc_job.return_value = {"success": True}
        self.agent.cryosparc_tools.import_particles_from_star.return_value = {"success": True}
        self.agent.cryosparc_tools.rebalance_orientations.return_value = {"success": True}
        self.agent.cryosparc_tools.get_orientation_diagnostics.return_value = {
            "success": True, "job_uid": "J20", "cfar": 0.12,
            "cfar_label": "below 0.5", "scf_star": 0.55, "computed": False,
        }

    def test_homogeneous_reconstruction_calls_verified_provider(self):
        result = self.agent._homogeneous_reconstruction_tool(json.dumps({
            "particles_job_uid": "J10", "mask_job_uid": "J11",
            "params": {"refine_symmetry": "C2"},
        }))
        self.assertTrue(json.loads(result)["success"])
        kwargs = self.agent.cryosparc_tools.homogeneous_reconstruction.call_args.kwargs
        self.assertEqual(kwargs["particles_job_uid"], "J10")
        self.assertEqual(kwargs["mask_job_uid"], "J11")
        self.assertEqual(kwargs["params"], {"refine_symmetry": "C2"})

    def test_generic_runner_parses_dry_run_boolean(self):
        self.agent._run_cryosparc_job_tool(json.dumps({
            "job_type": "reslog", "dry_run": "false",
            "inputs": [{"input_slot": "particles", "job_uid": "J10",
                        "output_group": "particles"}],
        }))
        kwargs = self.agent.cryosparc_tools.run_cryosparc_job.call_args.kwargs
        self.assertIs(kwargs["dry_run"], False)
        self.assertEqual(kwargs["inputs"][0]["output_group"], "particles")

    def test_star_import_uses_provider_kwargs_not_nested_job_params(self):
        self.agent._import_particles_from_star_tool(json.dumps({
            "star_path": "/tmp/particles.star", "data_sign": "positive",
        }))
        kwargs = self.agent.cryosparc_tools.import_particles_from_star.call_args.kwargs
        self.assertEqual(kwargs["star_path"], "/tmp/particles.star")
        self.assertEqual(kwargs["data_sign"], "positive")
        self.assertNotIn("params", kwargs)

    def test_rebalance_wrapper_forwards_reasoning_controls_and_dry_run(self):
        result = self.agent._rebalance_orientations_tool(json.dumps({
            "particles_job_uid": "J10",
            "particles_group_name": "particles",
            "num_orientation_bins": 256,
            "rebalance_percentile": 75,
            "exclusion_criterion": "alignments3D/error",
            "dry_run": "true",
        }))
        self.assertTrue(json.loads(result)["success"])
        kwargs = self.agent.cryosparc_tools.rebalance_orientations.call_args.kwargs
        self.assertEqual(kwargs["particles_job_uid"], "J10")
        self.assertEqual(kwargs["particles_group_name"], "particles")
        self.assertEqual(kwargs["num_orientation_bins"], 256)
        self.assertEqual(kwargs["rebalance_percentile"], 75)
        self.assertEqual(kwargs["exclusion_criterion"], "alignments3D/error")
        self.assertIs(kwargs["dry_run"], True)

    def test_rebalance_description_explains_tradeoff_without_forcing_it(self):
        description = reg._SPECS["rebalance_orientations"].description
        self.assertIn("cFAR alone", description)
        self.assertIn("cannot create views", description)
        self.assertIn("MATCHED terminal refinement", description)
        self.assertNotIn("cFAR <", description)

    def test_orientation_tool_returns_cfar_and_scf_star_for_reasoning(self):
        result = self.agent._get_orientation_diagnostics_tool(json.dumps({
            "refinement_job_uid": "J10", "run_if_missing": False,
        }))
        self.assertIn("cFAR for J10: 0.120", result)
        self.assertIn("SCF*: 0.550", result)
        self.assertIn("neither metric alone", result)


if __name__ == "__main__":
    unittest.main()
