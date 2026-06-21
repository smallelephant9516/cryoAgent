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
import re
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


# Canonical stage name -> result-JSON filename prefix (in pipeline order). Used to
# rebuild the blackboard from outputs/ artifacts when workflow_state.json is gone.
_STAGE_RESULT_PREFIXES = [
    ("preprocessing", "preprocessing"),
    ("particle_picking", "particle_picking"),
    ("optimization_2d", "2d_optimization"),
    ("reconstruction", "reconstruction"),
    ("optimization", "optimization"),
    ("polish", "polish"),
]


def _flatten_stage_outputs(d: Dict[str, Any]) -> Dict[str, Any]:
    """Lift one level of nested dicts (e.g. reconstruction's `outputs.*` and
    `job_uids.*`) to the top so the primary-job key resolves. Top-level keys win
    over nested ones; non-conflicting nested string/number values are promoted."""
    if not isinstance(d, dict):
        return {}
    flat = dict(d)
    for nest_key in ("outputs", "job_uids"):
        nested = d.get(nest_key)
        if isinstance(nested, dict):
            for k, v in nested.items():
                if k not in flat and isinstance(v, (str, int, float)):
                    flat[k] = v
    return flat


_FULL_DYNAMIC_LOG_GLOB = "llm_conversation_full_dynamic_*.log"

# Tools whose job UIDs are candidates for the "best" refinement when parsing logs.
_REFINEMENT_TOOL_NAMES = frozenset({
    "nonuniform_refinement", "homogeneous_refinement", "ab_initio_reconstruction",
    "local_refinement", "ctf_refine_global", "ctf_refine_local",
    "class_3d", "sharpen", "deepemhancer",
})


def has_full_dynamic_log(outputs_dir: str = "outputs") -> bool:
    """True when ``outputs_dir`` contains a full_dynamic conversation log."""
    return find_latest_full_dynamic_log(outputs_dir) is not None


def find_latest_full_dynamic_log(outputs_dir: str = "outputs") -> Optional[Path]:
    """Return the newest ``llm_conversation_full_dynamic_*.log`` under *outputs_dir*."""
    out = Path(outputs_dir)
    if not out.exists():
        return None
    cands = sorted(out.glob(_FULL_DYNAMIC_LOG_GLOB),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def has_guided_result_artifacts(outputs_dir: str = "outputs") -> bool:
    """True when per-stage guided result JSON files exist under *outputs_dir*."""
    out = Path(outputs_dir)
    if not out.exists():
        return False
    for _, prefix in _STAGE_RESULT_PREFIXES:
        if list(out.glob(f"{prefix}_results_cryosparc_*.json")):
            return True
    return False


def parse_full_dynamic_log(text: str) -> Dict[str, Any]:
    """Extract pipeline context from a ``full_dynamic`` conversation log.

    Returns a dict with keys: success, goal, decisions, best_job_uid, metrics,
    reasoning_summary, assessment, log_path (optional, set by caller).
    Never raises.
    """
    result: Dict[str, Any] = {
        "success": False,
        "goal": None,
        "decisions": [],
        "best_job_uid": None,
        "metrics": {},
        "reasoning_summary": None,
        "assessment": None,
    }
    if not text:
        return result

    # Goal from the opening system message.
    goal_m = re.search(
        r'SYSTEM MESSAGE:.*?Metadata:\s*\{.*?"workflow_input":\s*"(.+?)"',
        text, flags=re.S,
    )
    if goal_m:
        try:
            result["goal"] = json.loads(f'"{goal_m.group(1)}"')
        except Exception:
            result["goal"] = goal_m.group(1).replace("\\n", "\n").replace('\\"', '"')

    # Run outcome.
    ended_m = re.search(r"CONVERSATION ENDED:\s*\nSuccess:\s*(True|False)", text)
    if ended_m:
        result["success"] = ended_m.group(1).strip().lower() == "true"

    # Prefer structured TOOL EXECUTION blocks when present.
    tool_entries = _parse_conversation_log_text(text)
    decisions_from_tools = summarize_decisions(tool_entries, max_items=32)

    # Pair metadata next_tool with job UIDs mentioned in assistant prose.
    decisions_from_prose: List[str] = []
    pending_tool: Optional[str] = None
    assistant_blocks = re.findall(
        r"\[[^\]]+\] ASSISTANT RESPONSE:\n(.*?)(?=\n-{50}|\Z)",
        text, flags=re.S,
    )
    for block in assistant_blocks:
        meta_m = re.search(r'"next_tool":\s*"([^"]+)"', block)
        job_uids = re.findall(r"\b(J\d+)\b", block)
        if pending_tool and pending_tool not in _NON_ACTION_TOOLS and job_uids:
            uid = job_uids[0]
            label = f"{pending_tool} -> {uid}"
            if not decisions_from_prose or decisions_from_prose[-1] != label:
                decisions_from_prose.append(label)
        if meta_m:
            pending_tool = meta_m.group(1).strip()

    # Summary table rows: | N | `tool` | **Jxxx** | ...
    table_decisions: List[str] = []
    for row_m in re.finditer(
        r"\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*\*\*(J\d+)\*\*",
        text,
    ):
        tool, uid = row_m.group(1).strip(), row_m.group(2).strip()
        if tool not in _NON_ACTION_TOOLS:
            table_decisions.append(f"{tool} -> {uid}")

    # Merge: table is most reliable when present, then prose, then TOOL EXECUTION.
    if table_decisions:
        result["decisions"] = table_decisions
    elif decisions_from_prose:
        result["decisions"] = decisions_from_prose
    else:
        result["decisions"] = decisions_from_tools

    # Final-summary fields (successful runs).
    best_uid_m = re.search(
        r"(?:refinement|density|volume|map)\s+job\s+UID\*\*:\s*\*\*(J\d+)\*\*",
        text, flags=re.I,
    )
    if best_uid_m:
        result["best_job_uid"] = best_uid_m.group(1)
    else:
        # Table row explicitly marked as best result.
        best_row = re.search(
            r"\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*\*\*(J\d+)\*\*[^|\n]*best result",
            text, flags=re.I,
        )
        if best_row:
            result["best_job_uid"] = best_row.group(2)

    if not result["best_job_uid"]:
        # Prefer the best refinement-like tool in the decision list (not CTF/diagnostics).
        _density_tools = (
            "nonuniform_refinement", "homogeneous_refinement",
            "ab_initio_reconstruction", "local_refinement",
        )
        for d in reversed(result["decisions"]):
            tool_part = d.split("->")[0].strip().split("(")[0].strip()
            if tool_part in _density_tools:
                uid_m = re.search(r"(J\d+)\s*$", d)
                if uid_m:
                    result["best_job_uid"] = uid_m.group(1)
                    break
        if not result["best_job_uid"]:
            for d in reversed(result["decisions"]):
                tool_part = d.split("->")[0].strip().split("(")[0].strip()
                if tool_part in _REFINEMENT_TOOL_NAMES or any(
                    h in tool_part for h in _REFINEMENT_TOOL_HINTS
                ):
                    uid_m = re.search(r"(J\d+)\s*$", d)
                    if uid_m:
                        result["best_job_uid"] = uid_m.group(1)
                        break

    metrics: Dict[str, Any] = {}
    res_m = re.search(
        r"Resolution\*\*:\s*\*\*([\d.]+)\s*Å|resolution of \*\*([\d.]+)\s*Å\*\*"
        r"|improved to \*\*([\d.]+)\s*Å\*\*",
        text, flags=re.I,
    )
    if res_m:
        val = next(g for g in res_m.groups() if g)
        metrics["resolution_angstroms"] = float(val)

    cfar_m = re.search(
        r"cFAR\*\*:\s*([\d.]+)|cFAR\s*(?:=|of)\s*([\d.]+)",
        text, flags=re.I,
    )
    if cfar_m:
        val = cfar_m.group(1) or cfar_m.group(2)
        metrics["cfar"] = float(val)

    parts_m = re.search(
        r"Particles used\*\*:\s*\*\*([\d,]+)\*\*|([\d,]+)\s+clean particles",
        text, flags=re.I,
    )
    if parts_m:
        raw = (parts_m.group(1) or parts_m.group(2)).replace(",", "")
        metrics["num_particles"] = int(raw)

    sym_m = re.search(r"Symmetry\*\*:\s*\*\*(\w+)\*\*", text, flags=re.I)
    if sym_m:
        metrics["symmetry"] = sym_m.group(1)

    result["metrics"] = metrics

    # Last assistant block before CONVERSATION ENDED (or last block overall).
    if assistant_blocks:
        last = assistant_blocks[-1].split("Metadata:")[0].strip()
        # Drop trailing markdown headers for a compact one-liner.
        summary_lines = [
            ln.strip() for ln in last.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if summary_lines:
            result["reasoning_summary"] = " ".join(summary_lines)[:500]

    assess_m = re.search(
        r"## Remaining Limitations / Recommended Next Steps\n(.*?)(?=\nMetadata:|\n-{50}|\Z)",
        text, flags=re.S,
    )
    if assess_m:
        result["assessment"] = assess_m.group(1).strip()[:800]

    return result


def _parse_conversation_log_text(text: str) -> List[Dict[str, Any]]:
    """Like ``_parse_conversation_log`` but accepts raw log text."""
    entries: List[Dict[str, Any]] = []
    chunks = re.split(r"TOOL EXECUTION:\s*", text)
    for chunk in chunks[1:]:
        tool = chunk.splitlines()[0].strip() if chunk.strip() else ""
        if not tool:
            continue
        entry: Dict[str, Any] = {"tool": tool}
        m = re.search(r"Arguments:\s*(\{.*?\})\s*(?:Result:|-{5,}|\Z)", chunk, flags=re.S)
        if m:
            try:
                entry["params"] = json.loads(m.group(1))
            except Exception:
                entry["params"] = {}
        rm = re.search(
            r"Result:\s*\{.*?['\"]job_uid['\"]\s*:\s*['\"](J\d+)['\"]", chunk, flags=re.S,
        )
        if rm:
            entry["result"] = {"job_uid": rm.group(1)}
        entries.append(entry)
    return entries


def _parse_conversation_log(path: Path) -> List[Dict[str, Any]]:
    """Best-effort reconstruct a tool-execution log from an llm_conversation log.

    Scans `TOOL EXECUTION: <tool>` followed by an `Arguments: { ... }` JSON block
    (and an optional `Result: {...}`), returning [{tool, params, result?}]. Used to
    rebuild the `decisions` narrative. Never raises — returns what it can parse.
    """
    entries: List[Dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return entries
    return _parse_conversation_log_text(text)


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

    @classmethod
    def reconstruct_from_outputs(
        cls,
        outputs_dir: str = "outputs",
        cryosparc_tools=None,
        project_uid: Optional[str] = None,
        workspace_uid: Optional[str] = None,
        stage_goals: Optional[Dict[str, str]] = None,
    ) -> Optional["WorkflowState"]:
        """Rebuild the blackboard from a finished run's artifacts in outputs/ when
        workflow_state.json is absent.

        Sources: per-stage ``<prefix>_results_cryosparc_*.json`` (the stage_outputs
        a run wrote — latest per stage), and the matching
        ``llm_conversation_<stage>_*.log`` (for the `decisions` narrative).
        Each stage is fed through ``record_stage`` so it reuses the SAME live
        metric read (resolution/cFAR) + persistence as a normal recording.

        Returns the populated WorkflowState (already saved), or None if no per-stage
        result JSON was found at all.
        """
        out = Path(outputs_dir)
        if not out.exists():
            return None
        stage_goals = stage_goals or {}

        # Resolve project/workspace from the summary report or any result JSON.
        if not (project_uid and workspace_uid):
            for sj in sorted(out.glob("workflow_summary_report_*.json"),
                             key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    meta = json.loads(sj.read_text(encoding="utf-8")).get("workflow_metadata", {})
                    project_uid = project_uid or meta.get("project_uid")
                    workspace_uid = workspace_uid or meta.get("workspace_uid")
                    break
                except Exception:
                    continue

        def _latest(prefix: str) -> Optional[Path]:
            cands = sorted(out.glob(f"{prefix}_results_cryosparc_*.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0] if cands else None

        def _latest_log(stage: str) -> Optional[Path]:
            cands = sorted(out.glob(f"llm_conversation_{stage}_*.log"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            return cands[0] if cands else None

        ws = cls(outputs_dir=outputs_dir, project_uid=project_uid, workspace_uid=workspace_uid)
        found = 0
        for stage, prefix in _STAGE_RESULT_PREFIXES:
            rj = _latest(prefix)
            if rj is None:
                continue
            try:
                raw = json.loads(rj.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning("reconstruct: could not read %s: %s", rj.name, e)
                continue
            found += 1
            stage_outputs = _flatten_stage_outputs(raw)
            # Backfill project/workspace from the result file if still unknown.
            if ws.project_uid is None:
                ws.project_uid = stage_outputs.get("project_uid")
            if ws.workspace_uid is None:
                ws.workspace_uid = stage_outputs.get("workspace_uid")
            success = str(raw.get("status", "")).lower() in ("completed", "success", "")
            decisions = None
            log = _latest_log(stage)
            if log is not None:
                decisions = summarize_decisions(_parse_conversation_log(log))
            ws.record_stage(
                stage=stage,
                success=success,
                stage_outputs=stage_outputs,
                cryosparc_tools=cryosparc_tools,
                goal=stage_goals.get(stage),
                decisions=decisions or None,
            )

        if found == 0:
            return None
        logger.info("Reconstructed blackboard from %d stage result file(s).", found)
        return ws

    @classmethod
    def reconstruct_from_full_dynamic_log(
        cls,
        outputs_dir: str = "outputs",
        cryosparc_tools=None,
        project_uid: Optional[str] = None,
        workspace_uid: Optional[str] = None,
        log_path: Optional[Path] = None,
    ) -> Optional["WorkflowState"]:
        """Rebuild the blackboard from a ``full_dynamic`` conversation log ONLY.

        Used by ``--improve`` when the prior run was ``--mode full_dynamic`` and
        no per-stage result JSON exists. Parses tool usage and final metrics from
        ``llm_conversation_full_dynamic_*.log`` so the improvement agent can
        understand what was already tried and reuse prior job UIDs.

        Returns an in-memory WorkflowState (does not require workflow_state.json).
        """
        log_file = log_path or find_latest_full_dynamic_log(outputs_dir)
        if log_file is None:
            return None
        try:
            text = log_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            logger.warning("reconstruct_from_full_dynamic_log: could not read %s: %s",
                           log_file, e)
            return None

        parsed = parse_full_dynamic_log(text)
        ws = cls(outputs_dir=outputs_dir, project_uid=project_uid, workspace_uid=workspace_uid)

        best_uid = parsed.get("best_job_uid")
        stage_outputs: Dict[str, Any] = {}
        if best_uid:
            stage_outputs["final_refinement_job_uid"] = best_uid
        log_metrics = parsed.get("metrics") or {}
        if log_metrics.get("resolution_angstroms") is not None:
            stage_outputs["resolution_angstroms"] = log_metrics["resolution_angstroms"]

        metrics = dict(log_metrics)
        # Optionally refresh / fill metrics from CryoSPARC when reachable.
        if best_uid and cryosparc_tools is not None:
            try:
                res = cryosparc_tools.describe_job_results(
                    best_uid, project_uid=ws.project_uid,
                )
                if res.get("success"):
                    for k in ("job_type", "resolution_angstroms", "box_size",
                              "symmetry", "num_particles", "num_classes", "cfar",
                              "cfar_label"):
                        if res.get(k) is not None:
                            metrics[k] = res[k]
            except Exception as e:
                logger.debug("describe_job_results for %s failed: %s", best_uid, e)

        rec = StageRecord(
            stage="full_dynamic",
            success=bool(parsed.get("success")),
            stage_outputs=stage_outputs,
            primary_job_uid=best_uid,
            metrics=metrics,
            assessment=parsed.get("assessment"),
            goal=parsed.get("goal"),
            decisions=parsed.get("decisions") or [],
            reasoning_summary=parsed.get("reasoning_summary"),
        )
        ws.records = [rec]
        logger.info(
            "Reconstructed blackboard from full_dynamic log %s (%d decisions, best=%s).",
            log_file.name, len(rec.decisions), best_uid,
        )
        return ws

