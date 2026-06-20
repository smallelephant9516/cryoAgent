"""Workflow blackboard: a persistent, accumulating record of what each stage
produced, including the REAL result metrics (resolution, cFAR, particle/class
counts) read from CryoSPARC.

This is the shared observation memory for the data-driven workflow:

* During a guided run, the orchestrator records each finished stage here.
* The opt-in improvement agent reads it to find what limits the result and where
  the root cause lies (possibly an upstream stage), and to reuse prior good jobs
  rather than blindly re-running expensive steps.

Isolation is preserved: the blackboard holds only structured result metrics and
job UIDs (JSON), never a stage's internal conversation/context.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


logger = logging.getLogger("WorkflowState")

# Output-key conventions used across stages to point at the "main" job a stage
# produced. Checked in order; the first present wins.
_PRIMARY_JOB_KEYS = (
    "final_refinement_job_uid",
    "best_refinement_job_uid",
    "best_job_uid",
    "final_volume_job_uid",
    "homogeneous_refinement_job_uid",
    "final_particles_job_uid",
    "final_selection_job_uid",
    "selected_particles_job_uid",
    "micrograph_selection_job_uid",
    "final_micrographs_job_uid",
    "job_uid",
)

# Tools that are infrastructure/diagnostics, not workflow decisions.
_NON_ACTION_TOOLS = frozenset({
    "wait_for_job", "get_job_status", "get_job_log", "get_job_log_common",
    "describe_job_params", "describe_job_results", "get_orientation_diagnostics",
    "get_fsc_info", "get_particle_count", "search_cryosparc_forum",
    "reason_about_workflow", "verify_inputs", "read_input_json",
    "get_hetero_class_resolutions", "get_regroup_superclass_info",
    "consult_cryosparc_guide",
})

# Per-tool params worth surfacing as a "decision" (kept short on purpose).
_SALIENT_PARAMS = (
    "num_classes", "num_superclasses", "symmetry", "particle_diameter",
    "refinement_resolution", "initial_resolution", "final_resolution",
    "num_superclasses", "box_size",
)


def summarize_decisions(tool_execution_log: Optional[List[Dict[str, Any]]],
                        max_items: int = 8) -> List[str]:
    """Derive a short list of the action choices a stage actually made, from its
    tool-execution log: each action tool that produced a job, with its salient
    params. This is the concrete 'what was done' for dynamic/improvement mode."""
    decisions: List[str] = []
    for entry in tool_execution_log or []:
        tool = entry.get("tool")
        if not tool or tool in _NON_ACTION_TOOLS or entry.get("error"):
            continue
        result = entry.get("result")
        job_uid = result.get("job_uid") if isinstance(result, dict) else None
        params = entry.get("params") or {}
        salient = {k: params[k] for k in _SALIENT_PARAMS if k in params and params[k] is not None}
        bits = ", ".join(f"{k}={v}" for k, v in salient.items())
        label = f"{tool}({bits})" if bits else tool
        if job_uid:
            label += f" -> {job_uid}"
        decisions.append(label)
    # De-dup consecutive repeats, cap length.
    deduped: List[str] = []
    for d in decisions:
        if not deduped or deduped[-1] != d:
            deduped.append(d)
    return deduped[:max_items]


# Action tools whose output is a refined 3D volume (candidates for "best result"
# when recording an improvement run). Matched against the tool name AND the
# returned job_type, so both friendly names and raw CryoSPARC job types qualify.
_REFINEMENT_TOOL_HINTS = (
    "refine", "refinement", "reconstruct", "abinit", "ab_initio",
)


def refinement_job_uids_from_log(tool_execution_log: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Extract, in order, the job UIDs of refinement/reconstruction jobs an agent
    created (from its tool-execution log). These are the candidates whose
    resolution can be compared to find the improvement run's best result.

    Skips diagnostics/infrastructure tools and non-volume-producing actions
    (picking, extraction, 2D, curation), and de-dups while preserving order.
    """
    uids: List[str] = []
    for entry in tool_execution_log or []:
        tool = (entry.get("tool") or "").lower()
        if not tool or tool in _NON_ACTION_TOOLS or entry.get("error"):
            continue
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        job_uid = result.get("job_uid")
        if not isinstance(job_uid, str) or not job_uid.strip():
            continue
        job_type = (result.get("job_type") or "").lower()
        haystack = f"{tool} {job_type}"
        if any(h in haystack for h in _REFINEMENT_TOOL_HINTS):
            u = job_uid.strip()
            if u not in uids:
                uids.append(u)
    return uids


class StageRecord:
    """One stage's entry on the blackboard."""

    def __init__(
        self,
        stage: str,
        success: bool,
        stage_outputs: Optional[Dict[str, Any]] = None,
        primary_job_uid: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        assessment: Optional[str] = None,
        goal: Optional[str] = None,
        decisions: Optional[List[str]] = None,
        reasoning_summary: Optional[str] = None,
    ):
        self.stage = stage
        self.success = success
        self.stage_outputs = stage_outputs or {}
        self.primary_job_uid = primary_job_uid
        self.metrics = metrics or {}
        self.assessment = assessment
        # Narrative context (for dynamic/improvement mode to understand intent):
        self.goal = goal                          # what this stage tried to achieve
        self.decisions = decisions or []          # key param/branch choices made
        self.reasoning_summary = reasoning_summary  # one-line self-assessment
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "success": self.success,
            "primary_job_uid": self.primary_job_uid,
            "metrics": self.metrics,
            "assessment": self.assessment,
            "goal": self.goal,
            "decisions": self.decisions,
            "reasoning_summary": self.reasoning_summary,
            "stage_outputs": self.stage_outputs,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageRecord":
        rec = cls(
            stage=d.get("stage"),
            success=d.get("success", False),
            stage_outputs=d.get("stage_outputs") or {},
            primary_job_uid=d.get("primary_job_uid"),
            metrics=d.get("metrics") or {},
            assessment=d.get("assessment"),
            goal=d.get("goal"),
            decisions=d.get("decisions") or [],
            reasoning_summary=d.get("reasoning_summary"),
        )
        rec.timestamp = d.get("timestamp", time.time())
        return rec


class WorkflowState:
    """Persistent blackboard of stage records + result metrics."""

    def __init__(self, outputs_dir: str = "outputs", project_uid: Optional[str] = None,
                 workspace_uid: Optional[str] = None):
        self.outputs_dir = Path(outputs_dir)
        self.project_uid = project_uid
        self.workspace_uid = workspace_uid
        self.records: List[StageRecord] = []
        self.path = self.outputs_dir / "workflow_state.json"

    # ------------------------------------------------------------------
    # Primary-job + metric extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _pick_primary_job_uid(stage_outputs: Dict[str, Any]) -> Optional[str]:
        """Choose the 'main' job UID a stage produced, by key convention."""
        if not isinstance(stage_outputs, dict):
            return None
        for key in _PRIMARY_JOB_KEYS:
            v = stage_outputs.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def record_stage(
        self,
        stage: str,
        success: bool,
        stage_outputs: Optional[Dict[str, Any]] = None,
        cryosparc_tools=None,
        assessment: Optional[str] = None,
        goal: Optional[str] = None,
        decisions: Optional[List[str]] = None,
        reasoning_summary: Optional[str] = None,
    ) -> StageRecord:
        """
        Add (or replace) a stage record, auto-reading cheap result metrics.

        Metrics are read read-only via describe_job_results on the stage's primary
        job: resolution, particle/class counts, and cFAR ONLY if it already exists
        (no orientation_diagnostics is triggered here). Never raises — metric read
        failures are recorded as a note.

        goal/decisions/reasoning_summary capture WHY the stage did what it did, so
        dynamic/improvement mode can reason about intent, not just the numbers.
        """
        stage_outputs = stage_outputs or {}
        primary = self._pick_primary_job_uid(stage_outputs)
        metrics: Dict[str, Any] = {}

        if primary and cryosparc_tools is not None and success:
            try:
                res = cryosparc_tools.describe_job_results(primary, project_uid=self.project_uid)
                if res.get("success"):
                    for k in ("job_type", "resolution_angstroms", "box_size",
                              "symmetry", "num_particles", "num_classes", "cfar",
                              "cfar_label"):
                        if res.get(k) is not None:
                            metrics[k] = res[k]
                    if res.get("classes"):
                        metrics["classes"] = [
                            {
                                "class_id": c.get("class_id"),
                                "resolution_angstroms": c.get("resolution_angstroms"),
                                "num_particles": c.get("num_particles"),
                                "particle_fraction": c.get("particle_fraction"),
                                "cfar": c.get("cfar"),
                            }
                            for c in res["classes"]
                        ]
                else:
                    metrics["note"] = res.get("error", "metrics unavailable")
            except Exception as e:
                logger.warning("Failed to read metrics for %s (%s): %s", stage, primary, e)
                metrics["note"] = f"metric read error: {e}"

        # Fallback: if no resolution was obtained from describe_job_results but the
        # stage already computed one in its outputs (e.g. polish writes
        # `final_resolution`), trust that rather than recording None.
        if metrics.get("resolution_angstroms") is None:
            for k in ("final_resolution", "resolution_angstroms"):
                v = stage_outputs.get(k)
                if isinstance(v, (int, float)):
                    metrics["resolution_angstroms"] = float(v)
                    break

        # Replace any existing record for this stage (latest wins), else append.
        rec = StageRecord(stage, success, stage_outputs, primary, metrics, assessment,
                          goal=goal, decisions=decisions, reasoning_summary=reasoning_summary)
        self.records = [r for r in self.records if r.stage != stage] + [rec]
        self.save()
        return rec

    def record_improvement(
        self,
        cryosparc_tools,
        candidate_job_uids: List[str],
        assessment: Optional[str] = None,
        decisions: Optional[List[str]] = None,
    ) -> Optional[StageRecord]:
        """Record the best refinement the improvement agent produced.

        Reads `describe_job_results` for each candidate job UID (jobs the agent
        created during --improve), keeps those with a numeric resolution, picks
        the best (lowest resolution; tie-break on higher cFAR), and records it as
        an 'improvement' stage via the normal record_stage path. Returns the
        StageRecord, or None when no candidate yields a resolution (nothing is
        recorded in that case). Never raises.
        """
        best_uid = None
        best_res = None
        best_cfar = None
        for uid in candidate_job_uids or []:
            if not isinstance(uid, str) or not uid.strip():
                continue
            try:
                res = cryosparc_tools.describe_job_results(uid.strip(), project_uid=self.project_uid)
            except Exception as e:
                logger.warning("record_improvement: describe_job_results failed for %s: %s", uid, e)
                continue
            if not res.get("success"):
                continue
            r = res.get("resolution_angstroms")
            if not isinstance(r, (int, float)):
                continue
            c = res.get("cfar") if isinstance(res.get("cfar"), (int, float)) else None
            better = (
                best_res is None
                or r < best_res
                or (r == best_res and c is not None and (best_cfar is None or c > best_cfar))
            )
            if better:
                best_uid, best_res, best_cfar = uid.strip(), r, c

        if best_uid is None:
            logger.info("record_improvement: no candidate job produced a resolution; nothing recorded.")
            return None

        return self.record_stage(
            stage="improvement",
            success=True,
            stage_outputs={"final_refinement_job_uid": best_uid},
            cryosparc_tools=cryosparc_tools,
            assessment=assessment,
            decisions=decisions,
        )

    def set_assessment(self, stage: str, assessment: str) -> None:
        for r in self.records:
            if r.stage == stage:
                r.assessment = assessment
                self.save()
                return

    # ------------------------------------------------------------------
    # Queries for the improvement agent
    # ------------------------------------------------------------------
    def get(self, stage: str) -> Optional[StageRecord]:
        for r in self.records:
            if r.stage == stage:
                return r
        return None

    def best_resolution(self) -> Optional[Dict[str, Any]]:
        """Return {stage, job_uid, resolution_angstroms} for the best (lowest) res seen."""
        best = None
        for r in self.records:
            res = r.metrics.get("resolution_angstroms")
            if isinstance(res, (int, float)):
                if best is None or res < best["resolution_angstroms"]:
                    best = {
                        "stage": r.stage,
                        "job_uid": r.primary_job_uid,
                        "resolution_angstroms": res,
                        "cfar": r.metrics.get("cfar"),
                    }
        return best

    def summary_for_planner(self) -> str:
        """Compact human/LLM-readable digest of all stage metrics (JSON only)."""
        lines = []
        for r in self.records:
            status = "ok" if r.success else "FAILED"
            m = r.metrics
            bits = []
            if m.get("resolution_angstroms") is not None:
                bits.append(f"res={m['resolution_angstroms']:.2f}Å")
            if m.get("cfar") is not None:
                bits.append(f"cFAR={m['cfar']:.3f}")
            if m.get("num_particles") is not None:
                bits.append(f"particles={m['num_particles']}")
            if m.get("num_classes") is not None:
                bits.append(f"classes={m['num_classes']}")
            metric_s = ", ".join(bits) if bits else (m.get("note") or "no metrics")
            job_s = f" [{r.primary_job_uid}]" if r.primary_job_uid else ""
            lines.append(f"- {r.stage} ({status}){job_s}: {metric_s}")
            if r.goal:
                lines.append(f"    goal: {r.goal}")
            if r.decisions:
                lines.append(f"    decisions: {'; '.join(r.decisions)}")
            if r.reasoning_summary:
                lines.append(f"    summary: {r.reasoning_summary}")
            if r.assessment:
                lines.append(f"    assessment: {r.assessment}")
        return "\n".join(lines) if lines else "(blackboard empty)"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_uid": self.project_uid,
            "workspace_uid": self.workspace_uid,
            "records": [r.to_dict() for r in self.records],
        }

    def save(self) -> None:
        try:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to persist workflow_state.json: %s", e)

    @classmethod
    def load(cls, outputs_dir: str = "outputs") -> Optional["WorkflowState"]:
        path = Path(outputs_dir) / "workflow_state.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            st = cls(outputs_dir=outputs_dir,
                     project_uid=data.get("project_uid"),
                     workspace_uid=data.get("workspace_uid"))
            st.records = [StageRecord.from_dict(d) for d in data.get("records", [])]
            return st
        except Exception as e:
            logger.warning("Failed to load workflow_state.json: %s", e)
            return None
