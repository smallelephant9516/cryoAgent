"""Opt-in dynamic improvement agent (Phase: data-driven core #3/#4).

After a guided run completes, the user may request further improvement with the
``--improve`` flag. This agent is a single ReAct loop that:

* reads the blackboard (all stages' real result metrics) to find what limits the
  current best result and where the ROOT CAUSE lies (possibly an upstream stage);
* has the FULL atomic toolset across all stages (motion/CTF/pick/extract/2D/
  ab-initio/refine/hetero/regroup/nonuniform/diagnostics) plus describe_job_results,
  get_orientation_diagnostics, and consult_cryosparc_guide;
* REUSES prior good jobs rather than blindly re-running expensive steps — it
  decides per situation whether to re-run a whole step or just fire atomic tools;
* keeps improving until neither FSC resolution nor cFAR improves significantly
  (resolution gain < ~0.02 Å is not meaningful), then stops.

It reuses the already-initialized stage agents' bound tool methods (so the ~30
atomic wrappers are not reimplemented) and the configured LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain.tools import Tool

from .base_react_agent import BaseReActAgent
from . import cryosparc_tool_registry as reg
from ..tools.flexible_tool import make_flexible_tool


logger = logging.getLogger("ImprovementAgent")


# Resolution gain (Å) below which an improvement is not considered significant.
DEFAULT_RES_EPSILON = 0.02


class ImprovementAgent(BaseReActAgent):
    """A cross-stage, atomic-tool ReAct agent for post-run result improvement.

    Unlike the stage agents, this does NOT override _create_tools to a fixed
    per-stage set; it assembles the full atomic toolset by delegating to the
    bound tool methods of the already-initialized stage agents.
    """

    def __init__(self, stage_agents: Dict[str, Any], cryosparc_tools, config,
                 workflow_state=None, llm=None, res_epsilon: float = DEFAULT_RES_EPSILON):
        self.stage_name = "improvement"
        self.workflow_type = "cryoem"
        self._stage_agents = stage_agents or {}
        self.workflow_state = workflow_state
        self.res_epsilon = res_epsilon
        # Collect, once, a mapping tool-method-name -> bound method from any stage
        # agent that implements it (modular agents hold the real wrappers).
        self._method_providers: Dict[str, Any] = {}
        self._collect_tool_methods()
        super().__init__(cryosparc_tools=cryosparc_tools, config=config, llm=llm)

    def _max_iterations(self) -> int:
        """The improvement loop runs baseline + up to ~5 jobs, each with verify
        steps (wait_for_job, describe_job_results, get_orientation_diagnostics),
        so it needs many more iterations than a single-stage agent.

        Prefer an explicit ``agent.improvement_max_iterations`` when set; otherwise
        use a high floor over the (stage-tuned) ``agent.max_iterations``.
        """
        explicit = getattr(self.config.agent, "improvement_max_iterations", None)
        if explicit and int(explicit) > 0:
            return int(explicit)
        return max(30, int(getattr(self.config.agent, "max_iterations", 10) or 10))

    def _modular_agents(self) -> List[Any]:
        agents = []
        for sa in self._stage_agents.values():
            ma = getattr(sa, "modular_agent", None)
            agents.append(ma if ma is not None else sa)
        return agents

    def _collect_tool_methods(self) -> None:
        """Find a bound implementation for each atomic tool method across stage agents."""
        # All distinct (name, method) pairs the registry knows about.
        wanted = {}
        for spec in reg._SPECS.values():
            wanted.setdefault(spec.name, spec.method)
        for name, method in wanted.items():
            for agent in self._modular_agents():
                fn = getattr(agent, method, None)
                if callable(fn):
                    # Prefer the first agent that implements it. For tools whose
                    # behavior is stage-specific (class_2d, select_2d_classes,
                    # homogeneous_refinement), the reconstruction/picking variants
                    # are equivalent for improvement purposes.
                    self._method_providers[method] = fn
                    break

    def _create_tools(self) -> List[Tool]:
        """Assemble the full deduplicated atomic toolset from collected methods."""
        tools: List[Tool] = []
        seen_names = set()
        # One spec per distinct tool name (registry may have several spec_ids per name).
        for spec in reg._SPECS.values():
            if spec.name in seen_names:
                continue
            fn = self._method_providers.get(spec.method)
            # Diagnostic/introspection tools live on BaseReActAgent (self).
            if fn is None:
                fn = getattr(self, spec.method, None)
            if callable(fn):
                tools.append(make_flexible_tool(spec.name, spec.description, fn))
                seen_names.add(spec.name)
        # consult_cryosparc_guide (added in #5) — include if implemented on self.
        guide = getattr(self, "_consult_cryosparc_guide_tool", None)
        if callable(guide) and "consult_cryosparc_guide" not in seen_names:
            tools.append(make_flexible_tool(
                "consult_cryosparc_guide",
                (
                    "Consult the official CryoSPARC guide AND its tutorial/case-study "
                    "library (worked problem-solving examples: preferred orientation, "
                    "pseudosymmetry, membrane proteins, 3D classification, 3DVA, CTF "
                    "refinement, end-to-end GPCR/ferritin, etc.). Modes: pass "
                    "question=<problem> to auto-match the best job page + tutorial and "
                    "get a list of related tutorials; pass slug=<slug> to read a "
                    "specific tutorial; pass list_tutorials=true to browse the whole "
                    "library. Advisor only."
                ),
                guide,
            ))
        return tools

    def _get_react_system_prompt(self) -> str:
        return self._compose_stage_system_prompt(
            "shared/improvement-agent.md",
            self._get_improvement_context(),
        )

    def _get_improvement_context(self) -> Dict[str, Any]:
        bb = self.workflow_state.summary_for_planner() if self.workflow_state else "(no blackboard)"
        best = self.workflow_state.best_resolution() if self.workflow_state else None
        best_s = (
            f"{best['job_uid']} at {best['resolution_angstroms']:.2f} Å"
            if best else "unknown"
        )
        return {
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
            "blackboard": bb,
            "best_result": best_s,
            "res_epsilon": self.res_epsilon,
        }
