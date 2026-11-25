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
* keeps testing worthwhile, non-duplicate hypotheses until the goal is achieved,
  no worthwhile hypothesis remains, the safety cap is reached, or the user stops;
  an honestly documented open-branch pause is also a valid outcome.

It reuses the already-initialized stage agents' bound tool methods (so the ~30
atomic wrappers are not reimplemented) and the configured LLM.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.tools import Tool

from .base_react_agent import BaseReActAgent
from . import cryosparc_tool_registry as reg
from ..tools.flexible_tool import make_flexible_tool


logger = logging.getLogger("ImprovementAgent")


# Runaway safety ceiling for improvement reasoning/tool steps, not a job budget.
DEFAULT_IMPROVEMENT_MAX_ITERATIONS = 1000


# Resolution gain (Å) below which an improvement is not considered significant
# (reporting threshold).
DEFAULT_RES_EPSILON = 0.02
# Resolution gain (Å) required to REPLACE the baseline on a same-particle-set
# refinement — same-set FSC differences below this are mostly noise/overfitting.
DEFAULT_RATCHET_EPSILON = 0.05


class ImprovementAgent(BaseReActAgent):
    """A cross-stage, atomic-tool ReAct agent for post-run result improvement.

    Unlike the stage agents, this does NOT override _create_tools to a fixed
    per-stage set; it assembles the full atomic toolset by delegating to the
    bound tool methods of the already-initialized stage agents.
    """

    def __init__(self, stage_agents: Dict[str, Any], cryosparc_tools, config,
                 workflow_state=None, llm=None, res_epsilon: float = DEFAULT_RES_EPSILON,
                 ratchet_epsilon: float = DEFAULT_RATCHET_EPSILON):
        self.stage_name = "improvement"
        self.workflow_type = "cryoem"
        # Audit state must exist before BaseReActAgent.__init__ calls _create_tools().
        agent_settings = config.agent
        self.audit_enabled = bool(getattr(agent_settings, "improvement_audit_enabled", True))
        self.audit_critic_enabled = bool(getattr(agent_settings, "improvement_audit_critic_enabled", True))
        self.audit_max_attempts = max(1, int(getattr(agent_settings, "improvement_audit_max_attempts", 3)))
        self.audit_attempts = 0
        self.last_audit_report: Optional[Dict[str, Any]] = None
        self.last_audit_artifact: Optional[str] = None
        self._stage_agents = stage_agents or {}
        self.workflow_state = workflow_state
        self.res_epsilon = res_epsilon
        self.ratchet_epsilon = ratchet_epsilon
        # Collect, once, a mapping tool-method-name -> bound method from any stage
        # agent that implements it (modular agents hold the real wrappers).
        self._method_providers: Dict[str, Any] = {}
        self._collect_tool_methods()
        super().__init__(cryosparc_tools=cryosparc_tools, config=config, llm=llm)

    def _max_iterations(self) -> int:
        """Return the configurable runaway guard for the improvement loop.

        This is a safety ceiling, not an experiment budget: the agent should
        finish because the goal is achieved or no worthwhile hypothesis remains,
        not because it spent a predetermined number of jobs or trials.
        """
        explicit = getattr(self.config.agent, "improvement_max_iterations", None)
        if explicit and int(explicit) > 0:
            return int(explicit)
        return max(
            DEFAULT_IMPROVEMENT_MAX_ITERATIONS,
            int(getattr(self.config.agent, "max_iterations", 10) or 10),
        )

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
        missing_specs = []
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
            else:
                missing_specs.append(f"{spec.spec_id}->{spec.method}")
        if missing_specs:
            logger.warning(
                "CryoSPARC tools unavailable because no provider bound: %s",
                ", ".join(missing_specs),
            )
        # consult_cryosparc_guide (added in #5) — include if implemented on self.
        guide = getattr(self, "_consult_cryosparc_guide_tool", None)
        if callable(guide) and "consult_cryosparc_guide" not in seen_names:
            tools.append(make_flexible_tool(
                "consult_cryosparc_guide",
                (
                    "Search the COMPLETE official CryoSPARC guide AND the CryoSPARC "
                    "forum. The advisor turns long questions into focused searches, "
                    "uses BM25 over a cached whole-guide index, and returns guide and "
                    "community evidence separately. Modes: question=<problem> for "
                    "whole-guide + forum triage; slug=<slug-or-guide-path> to deep-read "
                    "any guide page; forum_url=<Discourse-URL> to deep-read a full "
                    "discussion; list_tutorials=true browses the curated case-study "
                    "library. Deep reads return a condensed, actionable digest grounded "
                    "in your question and recent attempts. Advisor only."
                ),
                guide,
            ))
        if self.audit_enabled and "submit_final_report" not in seen_names:
            tools.append(make_flexible_tool(
                "submit_final_report",
                (
                    "REQUIRED final scientific accuracy audit. Call with report=<complete final report, "
                    "including its one fenced-json experiment record>. The auditor verifies "
                    "actual params_spec, particle input connections, metrics, controlled-comparison "
                    "and noise-floor claims, capability freshness, and that stopping or exhaustion "
                    "narratives match the record. A separate adversarial critic checks causal overclaims. "
                    "Untested hypotheses and documented open branches are valid, not defects. "
                    "If success=false with status=fail, correct the listed findings by rewording, "
                    "retracting claims, or retrieving existing evidence, then resubmit the COMPLETE "
                    "report. The audit never requires new experiments. If status=fail_unresolved, "
                    "stop rewriting and return the best corrected report with unresolved findings "
                    "explicitly labeled; do not claim the audit passed. Only a pass or "
                    "pass_with_warnings response authorizes the final answer."
                ),
                self._submit_final_report_tool,
            ))
        return tools

    def _submit_final_report_tool(self, input_str: str) -> str:
        """Mid-loop completion gate; never changes the measured baseline ratchet."""
        from .experiment_auditor import (
            ExperimentAuditor, attach_exact_run_params,
            enrich_context_from_workflow_state, extract_improvement_context_from_log,
            harvest_log_evidence, write_audit_artifact,
        )

        try:
            submitted = self._parse_tool_input(input_str)
            report_text = submitted.get("report") or submitted.get("input") or input_str
            if not isinstance(report_text, str) or not report_text.strip():
                return json.dumps({
                    "success": False, "status": "fail",
                    "instructions": "Provide report=<complete final report including fenced-json record>.",
                })
            log_path = getattr(self.realtime_logger, "current_log_file", None)
            if not log_path or not Path(log_path).exists():
                raise RuntimeError("The live improvement conversation log is unavailable for evidence harvesting.")
            path = Path(log_path)
            context = extract_improvement_context_from_log(path)
            attach_exact_run_params(
                context, self.cryosparc_tools, self.config.workflow.project_uid,
            )
            harvest_log_evidence(path, context)
            state_path = Path(getattr(self, "outputs_dir", "outputs")) / "workflow_state.json"
            enrich_context_from_workflow_state(context, state_path)
            report = ExperimentAuditor(
                cryosparc_tools=self.cryosparc_tools,
                project_uid=self.config.workflow.project_uid,
                critic_llm=self.llm if self.audit_critic_enabled else None,
            ).audit(context, report_text)
            self.audit_attempts += 1
            report["attempt"] = self.audit_attempts
            report["max_attempts"] = self.audit_max_attempts
            critical = [f for f in report.get("findings", []) if f.get("severity") == "critical"]
            if critical and self.audit_attempts >= self.audit_max_attempts:
                report["status"] = "fail_unresolved"
                report["unresolved_critical_findings"] = critical
            artifact = write_audit_artifact(
                getattr(self, "outputs_dir", "outputs"), path.name, report,
                attempt=self.audit_attempts,
            )
            report["artifact"] = str(artifact)
            self.last_audit_report = report
            self.last_audit_artifact = str(artifact)
            response = {
                "success": not critical,
                "status": report["status"],
                "critical_findings": critical,
                "warnings": [f for f in report.get("findings", []) if f.get("severity") == "warning"],
                "attempts_remaining": max(0, self.audit_max_attempts - self.audit_attempts),
                "artifact": str(artifact),
            }
            if critical and report["status"] != "fail_unresolved":
                response["instructions"] = (
                    "Address every critical finding using actual existing job evidence, then resubmit "
                    "the corrected COMPLETE report. Reword or retract unsupported claims; the audit "
                    "never requires new experiments. Do not merely repeat the same conclusion."
                )
            elif report["status"] == "fail_unresolved":
                response["instructions"] = (
                    "The report-rewrite attempt budget is exhausted. Return the best corrected report now, "
                    "explicitly labeling these findings unresolved. Do not claim scientific audit approval "
                    "and do not run new experiments merely to satisfy the audit."
                )
            else:
                response["instructions"] = "Scientific audit accepted. You may now return this report as the final answer."
            return json.dumps(response, default=str)
        except Exception as exc:
            # Infrastructure failures must not masquerade as scientific approval.
            return json.dumps({
                "success": False, "status": "fail",
                "error": str(exc),
                "instructions": "Retry submit_final_report with the complete report. If this persists, finish with an explicit unaudited-report warning.",
            }, default=str)
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
            "ratchet_epsilon": self.ratchet_epsilon,
        }
