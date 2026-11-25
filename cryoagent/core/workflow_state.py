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
import ast
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
    "get_job_run_params",
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


def refinement_job_uids_from_log_file(log_path: Path) -> List[str]:
    """Fallback: extract refinement job UIDs from a conversation log file when
    structured TOOL EXECUTION blocks are missing or incomplete.

    Scans assistant responses for job UID mentions (J123, J456) in the context of
    refinement/reconstruction prose (e.g., "After (J94)", "New (J117)", "J125 @ 2.18 Å"),
    excluding job UIDs that appear only in reasoning about baseline/prior jobs.

    Returns UIDs in appearance order, de-duped. This is a best-effort fallback when
    tool execution logging was broken — prefer refinement_job_uids_from_log when
    structured logs are available.
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    uids: List[str] = []
    seen = set()

    # Pattern 1: Comparison tables showing "After (J94)" or "New (J117)" or "After local CTF (J125)"
    # These strongly indicate newly created jobs
    for match in re.finditer(
        r'(?:After|New|created|completed|result|refinement)[^()\[]*(?:\(|\[|:\s*)([Jj]\d+)',
        text, flags=re.IGNORECASE
    ):
        uid = match.group(1).upper()
        if uid not in seen:
            seen.add(uid)
            uids.append(uid)

    # Pattern 2: Job UIDs in resolution tables (| J94 | 2.17 Å |)
    # Captures jobs mentioned in result comparisons
    for match in re.finditer(
        r'\|\s*([Jj]\d+)\s*\|\s*[\d.]+\s*Å',
        text
    ):
        uid = match.group(1).upper()
        if uid not in seen:
            seen.add(uid)
            uids.append(uid)

    # Pattern 3: "J125 @ 2.18 Å" style mentions (completed jobs with metrics)
    for match in re.finditer(
        r'([Jj]\d+)\s*@\s*[\d.]+\s*Å',
        text
    ):
        uid = match.group(1).upper()
        if uid not in seen:
            seen.add(uid)
            uids.append(uid)

    # Pattern 4: Prose mentioning tool completion ("CTF refine global completed")
    # followed by job mentions in next ~500 chars
    blocks = re.findall(
        r'(?:completed|finished|created|running).*?([Jj]\d+)',
        text, flags=re.IGNORECASE
    )
    for uid_match in blocks:
        uid = uid_match.upper()
        if uid not in seen:
            seen.add(uid)
            uids.append(uid)

    # Now filter to only refinement-related jobs by checking surrounding context
    # A job is refinement-related if it appears within ~1000 chars of refinement keywords
    refinement_keywords = _REFINEMENT_TOOL_HINTS + (
        "ctf", "local ctf", "global ctf", "polish", "nonuniform",
    )
    refinement_uids = []
    for uid in uids:
        # Find all occurrences of this UID in text
        for match in re.finditer(rf'\b{uid}\b', text):
            pos = match.start()
            # Check ±1000 char window around this mention
            window_start = max(0, pos - 1000)
            window_end = min(len(text), pos + 1000)
            window = text[window_start:window_end].lower()
            # If any refinement keyword appears in this window, it's a refinement job
            if any(kw in window for kw in refinement_keywords):
                if uid not in refinement_uids:
                    refinement_uids.append(uid)
                break

    return refinement_uids


# Outcome classification thresholds (Å / cFAR). Match the improvement-agent prompt.
_RES_EPS = 0.02
_CFAR_EPS = 0.02
_CFAR_WORSE = -0.015

# Baseline-ratchet thresholds: how much a challenger must improve to REPLACE the
# incumbent best. Stricter than the reporting epsilon because same-particle-set
# FSC differences of ±0.02–0.05 Å are mostly noise / overfitting.
RATCHET_RES_EPS = 0.05    # resolution gain (Å) that replaces the baseline outright
_RATCHET_CFAR_TRADE = 0.05  # cFAR gain needed to replace when resolution ~unchanged
_RATCHET_RES_NOISE = 0.005  # resolution slack allowed in a cFAR-driven replacement


def challenger_beats_incumbent(
    ch_res: Optional[float],
    ch_cfar: Optional[float],
    inc_res: Optional[float],
    inc_cfar: Optional[float],
) -> bool:
    """Decide whether a challenger refinement replaces the incumbent baseline.

    Rules (matching the improvement-agent prompt):
    * resolution gain >= RATCHET_RES_EPS (0.05 Å) wins, unless cFAR collapses
      to no-structure (<0.1) from an acceptable incumbent;
    * clearly worse resolution (>0.02 Å) loses;
    * within noise, a cFAR gain >= 0.05 with resolution no worse than 0.005 Å
      wins (cFAR-driven replacement);
    * anything else keeps the incumbent.
    """
    if not isinstance(ch_res, (int, float)):
        return False
    if not isinstance(inc_res, (int, float)):
        return True
    d_res = inc_res - ch_res  # positive = challenger better
    if d_res >= RATCHET_RES_EPS:
        if isinstance(ch_cfar, (int, float)) and ch_cfar < 0.1 and (
            not isinstance(inc_cfar, (int, float)) or inc_cfar >= 0.15
        ):
            return False
        return True
    if d_res <= -_RES_EPS:
        return False
    if (
        d_res >= -_RATCHET_RES_NOISE
        and isinstance(ch_cfar, (int, float))
        and isinstance(inc_cfar, (int, float))
        and (ch_cfar - inc_cfar) >= _RATCHET_CFAR_TRADE
    ):
        return True
    return False


def _classify_attempt_outcome(
    metrics: Optional[Dict[str, Any]],
    verdict: str = "",
    invalid: bool = False,
) -> str:
    """Classify an attempt as helped / worse / no_difference / invalid."""
    if invalid:
        return "invalid"
    verdict_l = (verdict or "").lower()
    metrics = metrics or {}
    delta_res = metrics.get("delta_resolution")
    delta_cfar = metrics.get("delta_cfar")

    has_check = ("✓" in (verdict or "")) or ("gain" in verdict_l and "no" not in verdict_l)
    has_cross = ("✗" in (verdict or "")) or ("worse" in verdict_l)

    if isinstance(delta_res, (int, float)):
        delta_res = round(float(delta_res), 2)
    if isinstance(delta_cfar, (int, float)):
        delta_cfar = round(float(delta_cfar), 3)

    if delta_res is None and delta_cfar is None:
        if has_cross:
            return "worse"
        return "helped" if has_check else "no_difference"
    if isinstance(delta_res, (int, float)) and delta_res < -_RES_EPS:
        return "worse"
    if isinstance(delta_cfar, (int, float)) and delta_cfar <= _CFAR_WORSE:
        return "worse"
    if isinstance(delta_res, (int, float)) and delta_res >= _RES_EPS:
        if not has_cross:
            return "helped"
    if isinstance(delta_cfar, (int, float)) and delta_cfar >= _CFAR_EPS and not has_cross:
        if not isinstance(delta_res, (int, float)) or delta_res > -_RES_EPS:
            return "helped"
    return "no_difference"


def _dedup_append_strategy(strategies: List[Dict[str, Any]], strategy: Dict[str, Any]) -> None:
    """Append or replace a strategy keyed by result_job (later, richer rows win)."""
    job = (strategy.get("result_job") or "").upper()
    if not job:
        strategies.append(strategy)
        return
    for i, existing in enumerate(strategies):
        if (existing.get("result_job") or "").upper() == job:
            existing_metrics = existing.get("metrics") or {}
            new_metrics = strategy.get("metrics") or {}
            if new_metrics.get("result_resolution") is not None or strategy.get("outcome_class"):
                merged = dict(existing)
                merged.update(strategy)
                if (not strategy.get("hypothesis") or strategy.get("hypothesis") == "Unknown hypothesis") and existing.get("hypothesis"):
                    merged["hypothesis"] = existing["hypothesis"]
                if existing_metrics and not new_metrics.get("result_cfar"):
                    merged["metrics"] = {**existing_metrics, **new_metrics}
                strategies[i] = merged
            return
    strategies.append(strategy)


def _parse_job_action_better_table(text: str, context: Dict[str, Any]) -> None:
    """Parse '| Job | Action | Resolution | cFAR | ... | Better? |' summary tables."""
    header = re.search(
        r'\|\s*Job\s*\|\s*Action\s*\|\s*Resolution\s*\|\s*cFAR\s*\|',
        text, flags=re.IGNORECASE,
    )
    if not header:
        return
    table_text = text[header.end():]
    end = re.search(r'\n(?=\n[^|\s]|\nCONVERSATION ENDED)', table_text)
    if end:
        table_text = table_text[:end.start()]

    row_re = re.compile(
        r'\|\s*\*?\*?([Jj]\d+)\*?\*?\s*\|\s*(.*?)\s*\|\s*\*?\*?([\d.]+)\s*Å\*?\*?\s*'
        r'\|\s*\*?\*?([\d.]+)\*?\*?\s*\|\s*([^|]*?)\s*\|\s*(.*?)\s*\|'
    )
    baseline = context.get("baseline") or {}
    baseline_res = baseline.get("resolution")
    baseline_cfar = baseline.get("cfar")
    helped_cfar = baseline_cfar

    for row in row_re.finditer(table_text):
        job = row.group(1).upper()
        action = re.sub(r'\*+', '', row.group(2)).strip()
        result_res = float(row.group(3))
        result_cfar = float(row.group(4))
        verdict = re.sub(r'\*+', '', row.group(6)).strip()

        if re.search(r'baseline', action, re.IGNORECASE) or verdict in ("—", "-", "–", ""):
            if not baseline:
                context["baseline"] = {"job": job, "resolution": result_res, "cfar": result_cfar}
                baseline_res = result_res
                baseline_cfar = result_cfar
                helped_cfar = result_cfar
            continue

        delta_res = (baseline_res - result_res) if isinstance(baseline_res, (int, float)) else None
        metrics = {
            "baseline_resolution": baseline_res,
            "result_resolution": result_res,
            "delta_resolution": delta_res,
            "baseline_cfar": baseline_cfar,
            "result_cfar": result_cfar,
            "delta_cfar": (result_cfar - baseline_cfar) if isinstance(baseline_cfar, (int, float)) else None,
        }
        outcome_class = _classify_attempt_outcome(metrics, verdict=verdict)
        if outcome_class == "no_difference" and isinstance(helped_cfar, (int, float)):
            if result_cfar - helped_cfar <= _CFAR_WORSE:
                outcome_class = "worse"
                metrics["delta_cfar"] = result_cfar - helped_cfar

        if outcome_class == "helped" and isinstance(result_cfar, (int, float)):
            helped_cfar = result_cfar if helped_cfar is None else max(helped_cfar, result_cfar)

        outcome_parts = []
        if delta_res is not None:
            outcome_parts.append(
                f"Resolution: {baseline_res:.2f} → {result_res:.2f} Å ({delta_res:+.2f} Å)"
            )
        else:
            outcome_parts.append(f"Resolution: {result_res:.2f} Å")
        if isinstance(baseline_cfar, (int, float)):
            outcome_parts.append(
                f"cFAR: {baseline_cfar:.3f} → {result_cfar:.3f} "
                f"({(result_cfar - baseline_cfar):+.3f})"
            )
        else:
            outcome_parts.append(f"cFAR: {result_cfar:.3f}")

        _dedup_append_strategy(context["strategies_tried"], {
            "hypothesis": action[:200],
            "approach": action[:200],
            "actions": [],
            "result_job": job,
            "outcome": " | ".join(outcome_parts),
            "conclusion": verdict[:150],
            "success": outcome_class == "helped",
            "outcome_class": outcome_class,
            "metrics": metrics,
        })
        if job not in context["job_uids"]:
            context["job_uids"].append(job)


def _parse_invalid_experiments(text: str, context: Dict[str, Any]) -> None:
    """Mark jobs whose hypothesis was never actually tested."""
    class_collapse = re.search(
        r'class\s*0[^\n]{0,80}all\s+([\d,]+)\s*particles|'
        r'all\s+([\d,]+)\s*particles[^\n]{0,80}class\s*0|'
        r'no effective filtering',
        text, flags=re.IGNORECASE,
    )
    if class_collapse:
        job_match = re.search(
            r'([Jj]\d+)[^\n]{0,40}class_3d|class_3d[^\n]{0,80}([Jj]\d+)|'
            r'3D classification[^\n]{0,40}([Jj]\d+)|([Jj]\d+)[^\n]{0,60}3D class',
            text, flags=re.IGNORECASE,
        )
        result_job = None
        if job_match:
            result_job = next((g.upper() for g in job_match.groups() if g), None)
        if not result_job:
            nearby = re.search(r'what particle groups\s+([Jj]\d+)', text, flags=re.IGNORECASE)
            if nearby:
                result_job = nearby.group(1).upper()
        result_job = result_job or "class_3d"
        _dedup_append_strategy(context["strategies_tried"], {
            "hypothesis": "3D classification to check heterogeneity",
            "approach": "class_3d",
            "actions": ["class_3d"],
            "result_job": result_job,
            "outcome": "All particles in one class — no effective split",
            "conclusion": "INVALID: hypothesis not tested (no volume / class collapse)",
            "success": False,
            "outcome_class": "invalid",
            "metrics": {},
        })
        if result_job not in context["job_uids"] and str(result_job).startswith("J"):
            context["job_uids"].append(result_job)

    if re.search(r'template_picker|template picker', text, flags=re.IGNORECASE) and \
            re.search(r'GPU timeout|gpu issues persist', text, flags=re.IGNORECASE):
        job_match = re.search(
            r'template_picker[^\n]{0,80}([Jj]\d+)|([Jj]\d+)[^\n]{0,40}template.pick',
            text, flags=re.IGNORECASE,
        )
        groups = job_match.groups() if job_match else ()
        result_job = next((g.upper() for g in groups if g), None) or "template_picker"
        _dedup_append_strategy(context["strategies_tried"], {
            "hypothesis": "Volume-template re-picking to recover views",
            "approach": "create_templates + template_picker",
            "actions": ["create_templates", "template_picker"],
            "result_job": result_job,
            "outcome": "Job never completed (GPU timeout)",
            "conclusion": "INVALID: hypothesis not tested",
            "success": False,
            "outcome_class": "invalid",
            "metrics": {},
        })


def _parse_best_result_prose(text: str, context: Dict[str, Any]) -> None:
    """Prefer the narrative 'Best result: J223, 2.74 Å, cFAR=0.503'."""
    match = re.search(
        r'\*\*[Bb]est result\*\*[:\s]*\*?\*?([Jj]\d+)\*?\*?[^\d]{0,80}'
        r'([\d.]+)\s*Å[^\d]{0,80}cFAR\s*=?\s*([\d.]+)',
        text, flags=re.IGNORECASE,
    )
    if not match:
        # '**Best result: J303 — 2.59 Å, cFAR 0.668, ...**' (job inside the bold)
        match = re.search(
            r'\*\*[Bb]est result[:\s]+([Jj]\d+)\s*[—–\-:,]\s*([\d.]+)\s*Å'
            r'[^\d]{0,40}cFAR\s*=?\s*([\d.]+)',
            text, flags=re.IGNORECASE,
        )
    if not match:
        match = re.search(
            r'###?\s*Best result\s*\n+\s*\*\*([Jj]\d+)\*\*:\s*\*\*([\d.]+)\s*Å'
            r'[^\n]*cFAR\s*=?\s*([\d.]+)',
            text, flags=re.IGNORECASE,
        )
    if match:
        context["best_result"] = {
            "job": match.group(1).upper(),
            "resolution": float(match.group(2)),
            "cfar": float(match.group(3)),
        }


def _parse_experiment_record_json(text: str, context: Dict[str, Any]) -> None:
    """Parse the agent-emitted ```json experiment record (the structured,
    machine-readable summary the improvement prompt requires).

    Expected shape:
        {"baseline": {"job": "J303", "resolution": 2.59, "cfar": 0.668},
         "experiments": [{"hypothesis": ..., "plan": [...], "result_job": "J330",
                          "resolution": 2.55, "cfar": 0.67, "particles": 227745,
                          "verdict": "helped|worse|no_difference|invalid",
                          "one_line_reason": ...}],
         "best_result": {"job": ..., "resolution": ..., "cfar": ...},
         "branch_in_progress": {...} | null}
    """
    for m in re.finditer(r"```json\s*(.*?)```", text, flags=re.DOTALL):
        try:
            data = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("experiments"), list):
            continue

        baseline = data.get("baseline") or context.get("baseline") or {}
        if data.get("baseline") and not context.get("baseline"):
            context["baseline"] = dict(data["baseline"])
        b_res = baseline.get("resolution")
        b_cfar = baseline.get("cfar")

        for exp in data["experiments"]:
            if not isinstance(exp, dict):
                continue
            job = (exp.get("result_job") or "").upper()
            if not job:
                continue
            res = exp.get("resolution")
            cfar = exp.get("cfar")
            metrics = {
                "baseline_resolution": b_res,
                "result_resolution": res,
                "delta_resolution": (b_res - res) if isinstance(b_res, (int, float)) and isinstance(res, (int, float)) else None,
                "baseline_cfar": b_cfar,
                "result_cfar": cfar,
                "delta_cfar": (cfar - b_cfar) if isinstance(b_cfar, (int, float)) and isinstance(cfar, (int, float)) else None,
            }
            verdict = (exp.get("verdict") or "").strip()
            if verdict not in ("helped", "worse", "no_difference", "invalid"):
                verdict = _classify_attempt_outcome(metrics, verdict=verdict)
            plan = exp.get("plan") or exp.get("actions") or []
            outcome = exp.get("outcome") or ""
            if not outcome and isinstance(res, (int, float)):
                outcome = f"Resolution: {b_res} → {res} Å" if isinstance(b_res, (int, float)) else f"Resolution: {res} Å"
                if isinstance(cfar, (int, float)):
                    outcome += f" | cFAR: {b_cfar} → {cfar}" if isinstance(b_cfar, (int, float)) else f" | cFAR: {cfar}"
            _dedup_append_strategy(context["strategies_tried"], {
                "hypothesis": (exp.get("hypothesis") or "")[:200],
                "approach": exp.get("approach") or (" → ".join(plan) if plan else ""),
                "actions": plan,
                "result_job": job,
                "outcome": outcome,
                "conclusion": (exp.get("one_line_reason") or verdict)[:150],
                "success": verdict == "helped",
                "outcome_class": verdict,
                "metrics": metrics,
            })
            if job not in context["job_uids"]:
                context["job_uids"].append(job)

        best = data.get("best_result") or {}
        if best.get("job"):
            context["best_result"] = {
                "job": str(best["job"]).upper(),
                "resolution": best.get("resolution"),
                "cfar": best.get("cfar"),
            }
        if data.get("branch_in_progress"):
            context["branch_in_progress"] = data["branch_in_progress"]


def _iter_tool_execution_blocks(text: str):
    """Yield (tool_name, arguments_dict, result_text) for each TOOL EXECUTION
    block in a conversation log. Blocks are separated by dashed lines; arguments
    are a JSON object whose 'input' string holds a Python-dict repr."""
    for block in re.split(r"\n-{50,}\n", text):
        m = re.search(
            r"TOOL EXECUTION: (\w+)\nArguments: (.*?)\nResult: (.*)\Z",
            block, flags=re.DOTALL,
        )
        if not m:
            continue
        tool, args_text, result = m.group(1), m.group(2), m.group(3)
        args: Dict[str, Any] = {}
        try:
            outer = json.loads(args_text)
            inner = outer.get("input") if isinstance(outer, dict) else None
            if isinstance(inner, str):
                try:
                    inner = ast.literal_eval(inner)
                except (ValueError, SyntaxError):
                    inner = {}
            if isinstance(inner, dict):
                args = inner.get("tool_input") if isinstance(inner.get("tool_input"), dict) else inner
        except (json.JSONDecodeError, ValueError):
            pass
        yield tool, args or {}, result


def _strategies_from_tool_executions(text: str, context: Dict[str, Any]) -> None:
    """Deterministic backstop: reconstruct experiments from the structured TOOL
    EXECUTION log instead of the LLM's prose tables.

    Every action-tool job creation is captured (with its exact requested args);
    every describe_job_results / get_orientation_diagnostics call provides real
    measurements. A created job with a measured resolution becomes a strategy
    whose approach is the action-tool chain since the previous measured
    refinement (plan-level experiments). Entries are GAP-MERGED into
    context["strategies_tried"] — prose/JSON rows keep their richer hypotheses,
    this parser only fills missing jobs, metrics, actions and params.
    """
    created: Dict[str, Dict[str, Any]] = {}
    measurements: Dict[str, Dict[str, Any]] = {}
    current_chain: List[str] = []
    terminal_chains: Dict[str, List[str]] = {}

    for tool, args, result in _iter_tool_execution_blocks(text):
        if tool == "describe_job_results":
            jm = re.search(r"Results for (J\d+) \(([^,)]+)", result)
            if not jm:
                continue
            uid = jm.group(1).upper()
            meas = measurements.setdefault(uid, {})
            meas["job_type"] = jm.group(2)
            rm = re.search(r"resolution: ([\d.]+) Å", result)
            pm = re.search(r"particles: (\d+)", result)
            cm = re.search(r"cFAR: ([\d.]+)", result)
            if rm:
                meas["resolution"] = float(rm.group(1))
            if pm:
                meas["particles"] = int(pm.group(1))
            if cm:
                meas["cfar"] = float(cm.group(1))
            # A measured refinement of a job created in this log closes one
            # plan-level experiment: the next action tool starts a new chain.
            if uid in created and meas.get("resolution") is not None and uid not in terminal_chains:
                terminal_chains[uid] = list(current_chain)
                current_chain = []
            continue
        if tool == "get_orientation_diagnostics":
            jm = re.search(r"cFAR for (J\d+)[^:]*: ([\d.]+)", result)
            if jm:
                meas = measurements.setdefault(jm.group(1).upper(), {})
                meas.setdefault("cfar", float(jm.group(2)))
            continue
        if tool in _NON_ACTION_TOOLS:
            continue
        # Action tool: capture the created job + exact requested arguments.
        current_chain.append(tool)
        uid = None
        try:
            rj = json.loads(result)
            if isinstance(rj, dict) and rj.get("success") and rj.get("job_uid"):
                uid = str(rj["job_uid"]).upper()
        except (json.JSONDecodeError, ValueError):
            pass
        if uid is None and ("✅" in result or '"success": true' in result):
            jm = re.search(r"\b(J\d+)\b", result)
            if jm:
                uid = jm.group(1).upper()
        if uid:
            created[uid] = {"tool": tool, "args": args}
            req = context.setdefault("requested_params_by_job", {})
            if uid not in req:
                req[uid] = {"tool": tool, "arguments": args}
            if uid not in context["job_uids"]:
                context["job_uids"].append(uid)

    if not created or not measurements:
        return

    # Baseline: prefer the prose/JSON baseline; else the first measured job NOT
    # created during this round.
    baseline = context.get("baseline") or {}
    if not baseline.get("job"):
        for uid, meas in measurements.items():
            if uid not in created and meas.get("resolution") is not None:
                baseline = {"job": uid, "resolution": meas["resolution"], "cfar": meas.get("cfar")}
                context["baseline"] = baseline
                break
    b_res = baseline.get("resolution")
    b_cfar = baseline.get("cfar")

    existing = {
        (s.get("result_job") or "").upper(): s
        for s in context["strategies_tried"]
    }
    for uid, chain in terminal_chains.items():
        meas = measurements.get(uid) or {}
        res = meas.get("resolution")
        if res is None:
            continue
        cfar = meas.get("cfar")
        metrics = {
            "baseline_resolution": b_res,
            "result_resolution": res,
            "delta_resolution": (b_res - res) if isinstance(b_res, (int, float)) else None,
            "baseline_cfar": b_cfar,
            "result_cfar": cfar,
            "delta_cfar": (cfar - b_cfar) if isinstance(b_cfar, (int, float)) and isinstance(cfar, (int, float)) else None,
        }
        approach = " → ".join(chain)
        ex = existing.get(uid)
        if ex is None:
            context["strategies_tried"].append({
                "hypothesis": "",
                "approach": approach,
                "actions": chain,
                "result_job": uid,
                "outcome": (
                    f"Resolution: {b_res} → {res} Å" if isinstance(b_res, (int, float))
                    else f"Resolution: {res} Å"
                ) + (f" | cFAR: {b_cfar} → {cfar}" if isinstance(b_cfar, (int, float)) and isinstance(cfar, (int, float)) else ""),
                "conclusion": "",
                "success": False,
                "metrics": metrics,
            })
        else:
            if not ex.get("approach"):
                ex["approach"] = approach
            if not ex.get("actions"):
                ex["actions"] = chain
            em = ex.setdefault("metrics", {})
            for k, v in metrics.items():
                if em.get(k) is None and v is not None:
                    em[k] = v


def _extract_final_report_fallback(text: str, max_chars: int = 4000) -> str:
    """Slice Summary of all actions / Final Report when structured parse is empty."""
    start = re.search(r'##\s*Summary of all actions|##\s*Final Report', text, flags=re.IGNORECASE)
    if not start:
        return ""
    chunk = text[start.start():]
    end = re.search(r'\nCONVERSATION ENDED:', chunk)
    if end:
        chunk = chunk[:end.start()]
    return chunk[:max_chars].strip()


def _extract_requested_params_by_job(text: str) -> Dict[str, Dict[str, Any]]:
    """Map job UID -> full TOOL EXECUTION arguments (agent-requested, possibly incomplete)."""
    by_job: Dict[str, Dict[str, Any]] = {}
    blocks = re.split(r'\[[^\]]+\] TOOL EXECUTION: (.+?)\n', text)
    for i in range(1, len(blocks) - 1, 2):
        if i + 1 >= len(blocks):
            break
        tool_name = blocks[i].strip()
        body = blocks[i + 1]
        args_match = re.search(r'Arguments:\s*(.+?)(?=\nResult:|\n-{50}|\Z)', body, re.DOTALL)
        result_match = re.search(r'Result:\s*(.+?)(?=\n\[|\n-{50}|\Z)', body, re.DOTALL)
        arguments: Any = {}
        if args_match:
            raw = args_match.group(1).strip()
            try:
                json_match = re.search(r'\{.*\}', raw, re.DOTALL)
                arguments = json.loads(json_match.group(0)) if json_match else json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                arguments = {"raw": raw}
        job_uid = None
        if result_match:
            uid_m = re.search(r'"job_uid"\s*:\s*"([Jj]\d+)"', result_match.group(1))
            if uid_m:
                job_uid = uid_m.group(1).upper()
        if job_uid:
            by_job[job_uid] = {"tool": tool_name, "arguments": arguments}
    return by_job


def params_spec_from_job_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the full key→value params map from a CryoSPARC job document.

    params_spec overrides params_base. Used by get_job_run_params and tests.
    """
    params: Dict[str, Any] = {}
    for source in (doc.get("params_base") or {}, doc.get("params_spec") or {}):
        if not isinstance(source, dict):
            continue
        for key, details in source.items():
            if isinstance(details, dict) and "value" in details:
                params[key] = details.get("value")
            else:
                params[key] = details
    return params


def attach_exact_run_params(
    context: Dict[str, Any],
    cryosparc_tools: Any,
    project_uid: Optional[str],
) -> Dict[str, Any]:
    """Fill each strategy (and baseline) with exact CryoSPARC params_spec + inputs.

    On find_job failure, uses TOOL EXECUTION arguments labeled
    ``source: agent_request_incomplete``. Never infers flags from prose.
    """
    requested = context.get("requested_params_by_job") or {}
    jobs: List[str] = list(context.get("job_uids") or [])
    baseline_job = (context.get("baseline") or {}).get("job")
    best_job = (context.get("best_result") or {}).get("job")
    for extra in (baseline_job, best_job):
        if extra and extra not in jobs:
            jobs.append(extra)

    run_params: Dict[str, Any] = {}
    for uid in jobs:
        if not uid or not str(uid).startswith("J"):
            continue
        loaded = None
        if cryosparc_tools is not None and project_uid:
            try:
                loaded = cryosparc_tools.get_job_run_params(uid, project_uid=project_uid)
            except Exception as e:
                logger.warning("get_job_run_params failed for %s: %s", uid, e)
                loaded = None
        if isinstance(loaded, dict) and loaded.get("success"):
            run_params[uid] = loaded
        elif uid in requested:
            run_params[uid] = {
                "success": False,
                "job_uid": uid,
                "source": "agent_request_incomplete",
                "params": requested[uid].get("arguments") or {},
                "inputs": [],
                "tool": requested[uid].get("tool"),
            }
        else:
            run_params[uid] = {
                "success": False,
                "job_uid": uid,
                "source": "unavailable",
                "params": {},
                "inputs": [],
            }

    context["run_params"] = run_params
    for strat in context.get("strategies_tried") or []:
        uid = strat.get("result_job")
        if uid in run_params:
            strat["params"] = run_params[uid].get("params") or {}
            strat["inputs"] = run_params[uid].get("inputs") or []
            strat["params_source"] = run_params[uid].get("source")
            if run_params[uid].get("job_type"):
                strat["job_type"] = run_params[uid]["job_type"]
    return context


def _format_inputs_line(inputs: List[Dict[str, Any]]) -> str:
    if not inputs:
        return "(no input connections recorded)"
    parts = []
    for inp in inputs:
        slot = inp.get("slot") or inp.get("type") or "input"
        uid = inp.get("job_uid") or "?"
        group = inp.get("group_name")
        parts.append(f"{slot}={uid}" + (f"/{group}" if group else ""))
    return ", ".join(parts)


def _params_diff(reference: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    """Keys whose value differs from *reference* (exact params_spec, not a salient subset)."""
    diff: Dict[str, Any] = {}
    keys = set(reference or {}) | set(other or {})
    for key in sorted(keys):
        rv = (reference or {}).get(key, "<absent>")
        ov = (other or {}).get(key, "<absent>")
        if rv != ov:
            diff[key] = ov
    return diff


def _format_params_block(params: Dict[str, Any]) -> List[str]:
    if not params:
        return ["  (no params_spec available)"]
    lines = []
    for key in sorted(params.keys()):
        val = params[key]
        if isinstance(val, (dict, list)):
            try:
                val_s = json.dumps(val, default=str)
            except TypeError:
                val_s = str(val)
        else:
            val_s = str(val)
        lines.append(f"  {key}: {val_s}")
    return lines


def format_improvement_evidence_brief(context: Dict[str, Any]) -> str:
    """Five-bucket evidence brief for the next --improve task/prompt."""
    strategies = context.get("strategies_tried") or []
    baseline = context.get("baseline") or {}
    best = context.get("best_result") or {}
    run_params = context.get("run_params") or {}

    if not strategies and not context.get("fallback_excerpt"):
        return ""

    buckets = {"helped": [], "worse": [], "no_difference": [], "invalid": []}
    for strat in strategies:
        cls = strat.get("outcome_class") or _classify_attempt_outcome(
            strat.get("metrics"), verdict=strat.get("conclusion") or "",
        )
        if cls not in buckets:
            cls = "no_difference"
        buckets[cls].append(strat)

    lines: List[str] = []
    lines.append("## Prior improvement evidence (you must reason from this)")
    lines.append("")
    best_job = best.get("job")
    best_res = best.get("resolution")
    best_cfar = best.get("cfar")
    if best_job:
        from_s = ""
        if baseline.get("job"):
            from_s = (
                f" (from {baseline.get('job')} {baseline.get('resolution')} Å / "
                f"{baseline.get('cfar')})"
            )
        lines.append(f"Best so far: {best_job} @ {best_res} Å, cFAR {best_cfar}{from_s}")
    elif baseline.get("job"):
        lines.append(
            f"Baseline: {baseline.get('job')} @ {baseline.get('resolution')} Å, "
            f"cFAR {baseline.get('cfar')}"
        )
    if context.get("branch_in_progress"):
        lines.append(
            "Branch in progress (resume it; do not restart from scratch): "
            + json.dumps(context["branch_in_progress"], default=str)
        )
    lines.append("")

    def _bucket_lines(title: str, items: List[Dict[str, Any]]) -> None:
        lines.append(title)
        if not items:
            lines.append("- (none)")
            return
        for strat in items:
            job = strat.get("result_job", "?")
            approach = strat.get("approach") or strat.get("hypothesis") or ""
            outcome = strat.get("outcome") or ""
            lines.append(f"- {job}: {approach} | {outcome}")
            if strat.get("conclusion"):
                lines.append(f"  prior conclusion: {strat['conclusion']}")
            lines.append("  (exact params: see bucket 5)")

    _bucket_lines("1. Helped:", buckets["helped"])
    lines.append("")
    _bucket_lines("2. Made worse:", buckets["worse"])
    lines.append("")
    _bucket_lines("3. No difference:", buckets["no_difference"])
    lines.append("")
    _bucket_lines("4. Invalid / incomplete (hypothesis NOT tested):", buckets["invalid"])
    lines.append("")
    lines.append("5. Parameters (exact CryoSPARC params_spec + inputs):")
    lines.append("   Full dump of the best/helped job; other jobs list only keys that differ.")

    param_jobs = []
    seen = set()
    for extra in (best.get("job"), baseline.get("job")):
        if extra and extra not in seen:
            param_jobs.append(extra)
            seen.add(extra)
    for strat in strategies:
        uid = strat.get("result_job")
        if uid and uid not in seen and str(uid).startswith("J"):
            param_jobs.append(uid)
            seen.add(uid)

    ref_uid = best.get("job") or (buckets["helped"][0].get("result_job") if buckets["helped"] else None)
    ref_params = ((run_params.get(ref_uid) or {}).get("params") if ref_uid else None) or {}

    if not param_jobs:
        lines.append("- (no jobs)")
    for uid in param_jobs:
        block = run_params.get(uid) or {}
        source = block.get("source") or "unavailable"
        job_type = block.get("job_type") or ""
        params = block.get("params") or {}
        header = f"- {uid} ({job_type}, source={source})" if job_type else f"- {uid} (source={source})"
        lines.append(header)
        lines.append(f"  inputs: {_format_inputs_line(block.get('inputs') or [])}")
        dump_full = (uid == ref_uid) or not ref_params
        if dump_full:
            for pline in _format_params_block(params):
                lines.append(pline)
        else:
            diff = _params_diff(ref_params, params)
            lines.append(f"  vs {ref_uid} ({len(diff)} keys differ):")
            if not diff:
                lines.append("  (identical params_spec)")
            else:
                for pline in _format_params_block(diff):
                    lines.append(pline)

    if not strategies and context.get("fallback_excerpt"):
        lines.append("")
        lines.append("Structured parse was empty; raw Final Report / summary follows:")
        lines.append(context["fallback_excerpt"])

    lines.append("")
    lines.append(
        "Required: write a causal reading of all five buckets (cite exact param "
        "keys/values, not job-type nicknames), THEN consult_cryosparc_guide with a "
        "question that states what helped, what hurt, what did nothing, which exact "
        "params were used, and why. The next action must follow from that reading. "
        "Same job type + same params_spec = the same experiment; do not rerun it."
    )
    return "\n".join(lines)


def merge_improvement_contexts(contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine parsed improvement logs into one five-bucket context.

    Later logs overwrite ``best_result``. Strategies with the same ``result_job``
    are merged (richer later rows win).
    """
    merged: Dict[str, Any] = {
        "job_uids": [],
        "baseline": {},
        "best_result": {},
        "strategies_tried": [],
        "run_params": {},
        "requested_params_by_job": {},
        "fallback_excerpt": "",
        "branch_in_progress": None,
    }
    for ctx in contexts or []:
        if not ctx:
            continue
        if ctx.get("branch_in_progress"):
            merged["branch_in_progress"] = ctx["branch_in_progress"]
        for uid in ctx.get("job_uids") or []:
            if uid not in merged["job_uids"]:
                merged["job_uids"].append(uid)
        if ctx.get("baseline") and not merged["baseline"]:
            merged["baseline"] = dict(ctx["baseline"])
        br = ctx.get("best_result") or {}
        if br.get("job"):
            merged["best_result"] = dict(br)
        for strat in ctx.get("strategies_tried") or []:
            _dedup_append_strategy(merged["strategies_tried"], strat)
        merged["run_params"].update(ctx.get("run_params") or {})
        merged["requested_params_by_job"].update(ctx.get("requested_params_by_job") or {})
        if ctx.get("fallback_excerpt") and not merged["fallback_excerpt"]:
            merged["fallback_excerpt"] = ctx["fallback_excerpt"]
    return merged


def extract_improvement_context_from_log(log_path: Path) -> Dict[str, Any]:
    """Extract rich context from an improvement log: reasoning, hypotheses, outcomes.

    Parses assistant responses to capture what strategies were tried, why, and what
    the results were. This allows new improvement runs to see not just which jobs
    were created, but the full narrative of what was attempted and why it succeeded
    or failed.

    Returns:
        {
            "job_uids": ["J94", "J117", "J125"],
            "baseline": {"job": "J57", "resolution": 2.18, "cfar": 0.662},
            "strategies_tried": [
                {
                    "hypothesis": "CTF aberrations limiting resolution",
                    "approach": "Global CTF refinement then re-refine",
                    "actions": ["ctf_refine_global", "nonuniform_refinement"],
                    "result_job": "J94",
                    "outcome": "2.18 → 2.17 Å (0.01 Å gain)",
                    "conclusion": "Below 0.02 Å threshold - not meaningful",
                    "success": False
                },
                ...
            ]
        }
    """
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"job_uids": [], "strategies_tried": []}

    context: Dict[str, Any] = {
        "job_uids": [],
        "baseline": {},
        "strategies_tried": []
    }

    # Extract baseline information (usually in "STEP 0" or early in log)
    # Try format 1: inline "Baseline: J57 2.18 Å ... cFAR: 0.662"
    baseline_match = re.search(
        r'\*\*[Bb]aseline[:\s]+([Jj]\d+)[^\d]*([\d.]+)\s*Å.*?cFAR[:\s]*([\d.]+)',
        text, flags=re.DOTALL
    )
    if baseline_match:
        context["baseline"] = {
            "job": baseline_match.group(1).upper(),
            "resolution": float(baseline_match.group(2)),
            "cfar": float(baseline_match.group(3))
        }
    else:
        # Try format 2: bullet points with separate lines
        # **Baseline established:**
        # - **Job:** J238 ...
        # - **Resolution:** 2.51 Å
        # - **cFAR:** 0.687
        baseline_section = re.search(
            r'\*\*[Bb]aseline\s+established[:\s]*\*\*\s*(.*?)(?:\n\n|Let me reason)',
            text, flags=re.DOTALL
        )
        if baseline_section:
            baseline_text = baseline_section.group(1)
            job_match = re.search(r'[Jj]ob[:\s]*\*\*[:\s]*([Jj]\d+)', baseline_text)
            res_match = re.search(r'[Rr]esolution[:\s]*\*\*[:\s]*([\d.]+)\s*Å', baseline_text)
            cfar_match = re.search(r'cFAR[:\s]*\*\*[:\s]*([\d.]+)', baseline_text)

            if job_match and res_match:
                context["baseline"] = {
                    "job": job_match.group(1).upper(),
                    "resolution": float(res_match.group(1)),
                    "cfar": float(cfar_match.group(1)) if cfar_match else None
                }

    # Split into assistant response blocks to parse strategies
    assistant_blocks = re.findall(
        r'\[([^\]]+)\] ASSISTANT RESPONSE:\n(.*?)(?=\n-{50}|\Z)',
        text, flags=re.S
    )

    current_hypothesis = None
    current_approach = None
    current_actions = []

    for timestamp, block in assistant_blocks:
        # Look for hypothesis statements (markdown bold format)
        hyp_patterns = [
            r'\*\*[Hh]ypothesis(?:\s*#?\s*\d+)?\*\*[:\s]+(.+?)(?=\n\n|\n[A-Z#*]|\Z)',
            r'\*\*[Nn]ew hypothesis\*\*[:\s]+(.+?)(?=\n\n|\n[A-Z#*]|\Z)',
            r'[Rr]ank\s+\d+\s*→\s*([^—]+)—',
        ]
        for pattern in hyp_patterns:
            match = re.search(pattern, block, flags=re.DOTALL)
            if match:
                current_hypothesis = match.group(1).strip()[:200]  # Cap at 200 chars
                break

        # Look for approach/strategy descriptions
        approach_match = re.search(
            r'(?:[Aa]pproach|[Ss]trategy|[Pp]lan)[:\s]+(.+?)(?=\n\n|\n[A-Z#*]|\Z)',
            block, flags=re.DOTALL
        )
        if approach_match:
            current_approach = approach_match.group(1).strip()[:200]

        # Extract actions (tool mentions with backticks)
        tool_mentions = re.findall(
            r'`(\w+(?:_\w+)*)`',
            block
        )
        if tool_mentions:
            current_actions.extend([t for t in tool_mentions if t not in current_actions])

        # Look for comparison/evaluation tables with outcomes
        # First find the table header row to extract job UIDs
        table_match = re.search(
            r'\*\*[Cc]omparison[^:]*:\*\*\s*\n\s*'
            r'\|\s*[Mm]etric\s*\|([^\|]+)\|([^\|]+)\|',
            block, flags=re.DOTALL
        )

        if table_match:
            # Extract job UIDs from header columns
            col2_header = table_match.group(1).strip()  # "Baseline (J57)"
            col3_header = table_match.group(2).strip()  # "New (J117)" or "After local CTF (J125)"

            baseline_job_match = re.search(r'([Jj]\d+)', col2_header)
            result_job_match = re.search(r'([Jj]\d+)', col3_header)

            if baseline_job_match and result_job_match:
                baseline_job = baseline_job_match.group(1).upper()
                result_job = result_job_match.group(1).upper()

                # Now extract resolution values from the Resolution row (comes after header)
                res_row_match = re.search(
                    r'\|\s*\*\*[Rr]esolution\*\*\s*\|\s*([\d.]+)\s*Å\s*\|\s*([\d.]+)\s*Å',
                    block[table_match.end():], flags=re.DOTALL
                )

                if res_row_match:
                    baseline_res = float(res_row_match.group(1))
                    result_res = float(res_row_match.group(2))
                    delta_res = baseline_res - result_res

                    # Set context baseline from first comparison table if not already set
                    if not context.get("baseline"):
                        context["baseline"] = {
                            "job": baseline_job,
                            "resolution": baseline_res,
                            "cfar": None  # Will be filled if cFAR row is found
                        }

                    # Also extract cFAR values if present
                    cfar_row_match = re.search(
                        r'\|\s*\*\*cFAR\*\*\s*\|\s*([\d.]+)[^\|]*\|\s*([\d.]+)',
                        block[table_match.end():], flags=re.DOTALL | re.IGNORECASE
                    )
                    baseline_cfar = None
                    result_cfar = None
                    delta_cfar = None
                    if cfar_row_match:
                        baseline_cfar = float(cfar_row_match.group(1))
                        result_cfar = float(cfar_row_match.group(2))
                        delta_cfar = result_cfar - baseline_cfar

                        # Update context baseline cFAR if this is the first table
                        if context.get("baseline") and context["baseline"].get("cfar") is None:
                            context["baseline"]["cfar"] = baseline_cfar

                    # Build outcome string with both metrics
                    outcome_parts = [f"Resolution: {baseline_res:.2f} → {result_res:.2f} Å ({delta_res:+.2f} Å)"]
                    if baseline_cfar is not None and result_cfar is not None:
                        outcome_parts.append(f"cFAR: {baseline_cfar:.3f} → {result_cfar:.3f} ({delta_cfar:+.3f})")
                    outcome = " | ".join(outcome_parts)

                    # Extract conclusion from the text after the table
                    conclusion_match = re.search(
                        r'(?:meaningful|significant|improvement|gain|worse|better|no gain|below threshold)[^\n]{0,200}',
                        block[table_match.end() + res_row_match.end():], flags=re.IGNORECASE
                    )
                    conclusion = conclusion_match.group(0).strip() if conclusion_match else ""

                    # Determine success based on BOTH resolution and cFAR
                    success = False
                    # Resolution improved significantly
                    if abs(delta_res) >= 0.02:
                        success = True
                    # But cFAR degradation is a dealbreaker
                    if delta_cfar is not None and delta_cfar < -0.1:  # cFAR dropped >0.1
                        success = False
                    # Explicit failure keywords
                    if any(kw in conclusion.lower() for kw in ['not meaningful', 'below threshold', 'worse', 'no gain', 'meaningfully worse']):
                        success = False

                    # Record this strategy
                    if current_hypothesis or current_actions:
                        strategy = {
                            "hypothesis": current_hypothesis or "Unknown hypothesis",
                            "approach": current_approach or "See actions",
                            "actions": list(set(current_actions[-10:])),  # Last 10 unique actions
                            "result_job": result_job,
                            "outcome": outcome,
                            "conclusion": conclusion[:150] if conclusion else "No explicit conclusion",
                            "success": success,
                            "metrics": {
                                "baseline_resolution": baseline_res,
                                "result_resolution": result_res,
                                "delta_resolution": delta_res,
                                "baseline_cfar": baseline_cfar,
                                "result_cfar": result_cfar,
                                "delta_cfar": delta_cfar,
                            }
                        }
                        context["strategies_tried"].append(strategy)
                        if result_job not in context["job_uids"]:
                            context["job_uids"].append(result_job)

                        # Reset for next strategy
                        current_hypothesis = None
                        current_approach = None
                        current_actions = []

        # === NEW: Parse "What I Changed and Why" summary table ===
        # This table has format:
        # | # | Hypothesis | Action | Result | Verdict |
        # | 1 | ... | J301: ... | **2.78 Å** | ✗ Worse |
        summary_table_match = re.search(
            r'###?\s*What I Changed and Why\s*\n+\s*\|\s*#\s*\|\s*Hypothesis\s*\|\s*Action\s*\|\s*Result\s*\|\s*Verdict\s*\|',
            block, flags=re.IGNORECASE
        )

        if summary_table_match:
            # Find all table rows after the header
            table_start = summary_table_match.end()
            # Skip the separator line (|---|---|...)
            table_text = block[table_start:]

            # Match each data row
            row_pattern = r'\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*\*\*?([\d.]+)\s*Å\*\*?\s*\|\s*([^|]+?)\s*\|'

            for row_match in re.finditer(row_pattern, table_text):
                row_num = row_match.group(1)
                hypothesis = row_match.group(2).strip()
                action = row_match.group(3).strip()
                result_res = float(row_match.group(4))
                verdict = row_match.group(5).strip()

                # Extract job UID from action column (e.g., "J301: nonuniform_refinement...")
                job_match = re.search(r'([Jj]\d+)', action)
                result_job = job_match.group(1).upper() if job_match else f"Unknown_{row_num}"

                # Determine success from verdict
                success = '✓' in verdict or 'gain' in verdict.lower()
                if '✗' in verdict or 'worse' in verdict.lower() or 'same' in verdict.lower():
                    success = False

                # Extract baseline resolution if available (from context or assume it's from earlier)
                baseline_res = context.get("baseline", {}).get("resolution", None)

                # Build outcome string
                if baseline_res:
                    delta_res = baseline_res - result_res
                    outcome = f"Resolution: {baseline_res:.2f} → {result_res:.2f} Å ({delta_res:+.2f} Å)"
                else:
                    outcome = f"Resolution: {result_res:.2f} Å"

                # Add strategy to context
                strategy = {
                    "hypothesis": hypothesis[:200],
                    "approach": action[:200],
                    "actions": [],  # Could parse from action text if needed
                    "result_job": result_job,
                    "outcome": outcome,
                    "conclusion": verdict[:150],
                    "success": success,
                    "metrics": {
                        "baseline_resolution": baseline_res,
                        "result_resolution": result_res,
                        "delta_resolution": baseline_res - result_res if baseline_res else None,
                        "baseline_cfar": None,
                        "result_cfar": None,
                        "delta_cfar": None,
                    }
                }
                context["strategies_tried"].append(strategy)
                if result_job not in context["job_uids"]:
                    context["job_uids"].append(result_job)

        # === NEW: Parse "Action | Hypothesis | Result" summary table ===
        # This table has format:
        # | Action | Hypothesis | Result |
        # | **1. Strategy** (J508→J509→J510) | reasoning | **2.73→2.67 Å** ✅ ... |
        action_summary_match = re.search(
            r'\|\s*Action\s*\|\s*Hypothesis\s*\|\s*Result\s*\|',
            block, flags=re.IGNORECASE
        )

        if action_summary_match:
            table_start = action_summary_match.end()
            table_text = block[table_start:]

            # Match each data row: | **N. Action** (jobs) | hypothesis | **baseline→result Å** status |
            row_pattern = r'\|\s*\*\*(\d+)\.\s*([^(]+)\*\*\s*\(([^)]+)\)\s*\|\s*([^|]+?)\s*\|\s*\*\*?([\d.]+)→([\d.]+)\s*Å\*\*?[^|]*?(✅|❌)'

            for row_match in re.finditer(row_pattern, table_text):
                row_num = row_match.group(1)
                strategy_name = row_match.group(2).strip()
                job_chain = row_match.group(3).strip()
                hypothesis = row_match.group(4).strip()
                baseline_res = float(row_match.group(5))
                result_res = float(row_match.group(6))
                status = row_match.group(7).strip()

                # Extract final job UID from chain (e.g., "J508→J509→J510" → "J510")
                job_uids = re.findall(r'[Jj]\d+', job_chain)
                result_job = job_uids[-1].upper() if job_uids else f"Unknown_{row_num}"

                # Determine success from status
                success = (status == '✅')
                delta_res = baseline_res - result_res

                # Extract additional metrics from the result cell if present
                result_cell = table_text[row_match.start():row_match.end() + 200]
                cfar_match = re.search(r'cFAR\s*([\d.]+)→([\d.]+)', result_cell)
                baseline_cfar = float(cfar_match.group(1)) if cfar_match else None
                result_cfar = float(cfar_match.group(2)) if cfar_match else None

                # Build outcome string
                outcome = f"Resolution: {baseline_res:.2f} → {result_res:.2f} Å ({delta_res:+.2f} Å)"
                if baseline_cfar and result_cfar:
                    outcome += f" | cFAR: {baseline_cfar:.3f} → {result_cfar:.3f}"

                # Extract conclusion from result cell
                conclusion_match = re.search(r'(Meaningful gain|Below.*?threshold|Worse|Failed)[^|]{0,100}', result_cell, re.IGNORECASE)
                conclusion = conclusion_match.group(0).strip() if conclusion_match else ("Success" if success else "No improvement")

                # Add strategy to context
                strategy = {
                    "hypothesis": hypothesis[:200],
                    "approach": f"{strategy_name} ({job_chain})",
                    "actions": [],
                    "result_job": result_job,
                    "outcome": outcome,
                    "conclusion": conclusion[:150],
                    "success": success,
                    "metrics": {
                        "baseline_resolution": baseline_res,
                        "result_resolution": result_res,
                        "delta_resolution": delta_res,
                        "baseline_cfar": baseline_cfar,
                        "result_cfar": result_cfar,
                        "delta_cfar": (result_cfar - baseline_cfar) if (baseline_cfar and result_cfar) else None,
                    }
                }
                context["strategies_tried"].append(strategy)
                if result_job not in context["job_uids"]:
                    context["job_uids"].append(result_job)

                # Set baseline from first row if not already set
                if not context.get("baseline") and row_num == "1":
                    context["baseline"] = {
                        "resolution": baseline_res,
                        "cfar": baseline_cfar
                    }

    # Whole-log parsers (cover table formats the per-block regexes miss).
    _parse_job_action_better_table(text, context)
    _parse_invalid_experiments(text, context)
    _parse_best_result_prose(text, context)
    # Structured experiment record (agent-emitted JSON) + deterministic
    # reconstruction from the TOOL EXECUTION log as a backstop, so the five
    # buckets stay populated even when prose-table parsing fails.
    _parse_experiment_record_json(text, context)
    _strategies_from_tool_executions(text, context)

    for strat in context["strategies_tried"]:
        if not strat.get("outcome_class"):
            strat["outcome_class"] = _classify_attempt_outcome(
                strat.get("metrics"),
                verdict=strat.get("conclusion") or "",
            )
            strat["success"] = strat["outcome_class"] == "helped"

    context["requested_params_by_job"] = _extract_requested_params_by_job(text)
    if not context["strategies_tried"]:
        context["fallback_excerpt"] = _extract_final_report_fallback(text)

    # Also extract job UIDs using the existing function for completeness
    all_job_uids = refinement_job_uids_from_log_file(log_path)
    for uid in all_job_uids:
        if uid not in context["job_uids"]:
            context["job_uids"].append(uid)

    return context


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

    # run_status values for live visualizer monitoring
    STATUS_IDLE = "idle"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    def __init__(self, outputs_dir: str = "outputs", project_uid: Optional[str] = None,
                 workspace_uid: Optional[str] = None):
        self.outputs_dir = Path(outputs_dir)
        self.project_uid = project_uid
        self.workspace_uid = workspace_uid
        self.records: List[StageRecord] = []
        self.path = self.outputs_dir / "workflow_state.json"
        self.current_stage: Optional[str] = None
        self.run_status: str = self.STATUS_IDLE

    # ------------------------------------------------------------------
    # Live progress markers (for visualizer Real-time mode)
    # ------------------------------------------------------------------
    def begin_stage(self, stage: str) -> None:
        """Mark a stage as currently running and persist for live monitors."""
        self.current_stage = stage
        self.run_status = self.STATUS_RUNNING
        self.save()

    def finish_run(self, success: bool = True) -> None:
        """Clear the in-progress marker and set a terminal run_status."""
        self.current_stage = None
        self.run_status = self.STATUS_COMPLETED if success else self.STATUS_FAILED
        self.save()

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

        Special case: for stage="improvement", this APPENDS a new record (numbered
        improvement_1, improvement_2, ...) rather than replacing, so multiple
        --improve rounds accumulate their full history.
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

                    # For preprocessing stages, extract micrograph count from output_groups
                    if stage == "preprocessing" and res.get("output_groups"):
                        for group in res["output_groups"]:
                            if isinstance(group, dict) and group.get("name") == "exposures_accepted":
                                metrics["num_micrographs"] = group.get("num_items")
                                break
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

        # For "improvement", append with an auto-incremented suffix (improvement_1, _2, ...)
        # so multiple --improve rounds accumulate. For all other stages, replace (latest wins).
        begun_stage = stage
        if stage == "improvement":
            existing_improvement_nums = [
                int(r.stage.split("_")[-1])
                for r in self.records
                if r.stage.startswith("improvement_") and r.stage.split("_")[-1].isdigit()
            ]
            next_num = max(existing_improvement_nums, default=0) + 1
            stage = f"improvement_{next_num}"
        else:
            # Replace any existing record for this non-improvement stage.
            self.records = [r for r in self.records if r.stage != stage]

        rec = StageRecord(stage, success, stage_outputs, primary, metrics, assessment,
                          goal=goal, decisions=decisions, reasoning_summary=reasoning_summary)
        self.records.append(rec)
        if self.current_stage in (begun_stage, stage):
            self.current_stage = None
        self.save()
        return rec

    def record_improvement(
        self,
        cryosparc_tools,
        candidate_job_uids: List[str],
        assessment: Optional[str] = None,
        decisions: Optional[List[str]] = None,
        strategies_tried: Optional[List[Dict[str, Any]]] = None,
        baseline: Optional[Dict[str, Any]] = None,
        preferred_job_uid: Optional[str] = None,
        source_log: Optional[str] = None,
    ) -> Optional[StageRecord]:
        """Record the best refinement the improvement agent produced.

        Reads `describe_job_results` for each candidate job UID (jobs the agent
        created during --improve), keeps those with a numeric resolution, picks
        the best (lowest resolution; tie-break on higher cFAR), and records it as
        an 'improvement' stage via the normal record_stage path. Returns the
        StageRecord, or None when no candidate yields a resolution (nothing is
        recorded in that case). Never raises.

        Args:
            cryosparc_tools: CryoSPARC tools instance for querying job results
            candidate_job_uids: Job UIDs created during this improvement run
            assessment: Optional one-line summary
            decisions: Optional list of decision strings (legacy format)
            strategies_tried: Optional list of strategy dicts with hypothesis/outcome
            baseline: Optional baseline job info {"job": "J57", "resolution": 2.18}
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

            # Tournament: resolution-primary with cFAR guard (same rule as the
            # baseline ratchet), so the round's best is picked consistently with
            # best_resolution().
            if best_uid is None or challenger_beats_incumbent(r, c, best_res, best_cfar):
                best_uid, best_res, best_cfar = uid.strip(), r, c

        if preferred_job_uid:
            pref = preferred_job_uid.strip()
            try:
                pref_res = cryosparc_tools.describe_job_results(pref, project_uid=self.project_uid)
                if pref_res.get("success") and isinstance(pref_res.get("resolution_angstroms"), (int, float)):
                    best_uid = pref
                    best_res = pref_res.get("resolution_angstroms")
                    best_cfar = pref_res.get("cfar") if isinstance(pref_res.get("cfar"), (int, float)) else None
            except Exception as e:
                logger.warning("record_improvement: preferred job %s failed: %s", pref, e)

        if best_uid is None:
            logger.info("record_improvement: no candidate job produced a resolution; nothing recorded.")
            return None

        # Ratchet guard: the round's best must BEAT the current global baseline
        # to displace it. Otherwise record the round with the incumbent as the
        # primary job (history + strategies preserved, baseline unchanged).
        incumbent = self.best_resolution()
        baseline_stands = bool(
            incumbent and incumbent.get("job_uid") == best_uid
        )
        if incumbent and incumbent.get("job_uid") and best_uid != incumbent["job_uid"]:
            if not challenger_beats_incumbent(
                best_res, best_cfar,
                incumbent.get("resolution_angstroms"), incumbent.get("cfar"),
            ):
                baseline_stands = True
                best_uid = incumbent["job_uid"]
                best_res = incumbent.get("resolution_angstroms")
                best_cfar = incumbent.get("cfar")
                assessment = (
                    f"{assessment} | baseline stands: no candidate beat "
                    f"{incumbent['job_uid']} ({incumbent.get('resolution_angstroms')} Å)"
                    if assessment else
                    f"baseline stands: no candidate beat {incumbent['job_uid']}"
                )

        # Build rich decisions narrative from strategies_tried if available
        if strategies_tried and isinstance(strategies_tried, list):
            decisions = self._format_strategies_for_decisions(strategies_tried, baseline)

        outputs = {
            "final_refinement_job_uid": best_uid,
            "strategies_tried": strategies_tried or [],
            "baseline": baseline or {},
            "baseline_stands": baseline_stands,
        }
        if source_log:
            outputs["source_log"] = source_log

        return self.record_stage(
            stage="improvement",
            success=True,
            stage_outputs=outputs,
            cryosparc_tools=cryosparc_tools,
            assessment=assessment,
            decisions=decisions,
        )

    def _format_strategies_for_decisions(
        self,
        strategies: List[Dict[str, Any]],
        baseline: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Format strategy dicts into human-readable decision lines for the blackboard."""
        lines = []
        if baseline:
            lines.append(f"Baseline: {baseline.get('job', '?')} @ {baseline.get('resolution', '?')} Å, cFAR {baseline.get('cfar', '?')}")

        for i, strat in enumerate(strategies, 1):
            outcome_class = strat.get("outcome_class") or ("helped" if strat.get("success") else "no_difference")
            icon = {
                "helped": "✓",
                "worse": "✗ worse",
                "invalid": "✗ invalid",
                "no_difference": "≈",
            }.get(outcome_class, "✗")
            hypothesis = strat.get("hypothesis", "Unknown")[:80]
            outcome = strat.get("outcome", "No outcome recorded")
            conclusion = strat.get("conclusion", "")[:100]

            lines.append(f"{i}. [{outcome_class}] {icon} {hypothesis}")
            lines.append(f"   → {outcome}")
            if conclusion:
                lines.append(f"   → {conclusion}")

            # Show why it failed if we have metric details
            metrics = strat.get("metrics", {})
            if not strat.get("success") and metrics:
                reasons = []
                delta_res = metrics.get("delta_resolution")
                delta_cfar = metrics.get("delta_cfar")
                if delta_res is not None and abs(delta_res) < 0.02:
                    reasons.append(f"Δres {delta_res:+.2f} Å < 0.02 threshold")
                if delta_cfar is not None and delta_cfar < -0.1:
                    reasons.append(f"cFAR degraded {delta_cfar:+.3f}")
                if reasons:
                    lines.append(f"   → Why failed: {', '.join(reasons)}")

        return lines

    def find_improvement_record_for_log(self, log_name: str) -> Optional[StageRecord]:
        """Find an existing improvement record that came from this conversation log."""
        if not log_name:
            return None
        for r in self.records:
            if not str(r.stage).startswith("improvement"):
                continue
            if (r.stage_outputs or {}).get("source_log") == log_name:
                return r
            if log_name in (r.assessment or ""):
                return r
        return None

    def update_improvement_record(
        self,
        rec: StageRecord,
        cryosparc_tools=None,
        strategies_tried: Optional[List[Dict[str, Any]]] = None,
        baseline: Optional[Dict[str, Any]] = None,
        preferred_job_uid: Optional[str] = None,
        source_log: Optional[str] = None,
        assessment: Optional[str] = None,
    ) -> StageRecord:
        """Refresh a thin improvement blackboard row from a re-parsed log."""
        strategies_tried = strategies_tried or []
        baseline = baseline or {}
        outputs = dict(rec.stage_outputs or {})
        outputs["strategies_tried"] = strategies_tried
        outputs["baseline"] = baseline
        if source_log:
            outputs["source_log"] = source_log

        if preferred_job_uid:
            # Ratchet guard on retrospective refreshes: the log's preferred job
            # must beat the incumbent (excluding this record) to displace it.
            incumbent = self.best_resolution(exclude=rec)
            outputs["baseline_stands"] = bool(
                incumbent and incumbent.get("job_uid") == preferred_job_uid.strip()
            )
            if incumbent and incumbent.get("job_uid") and preferred_job_uid.strip() != incumbent["job_uid"]:
                pref = preferred_job_uid.strip()
                ch_res = ch_cfar = None
                if cryosparc_tools is not None:
                    try:
                        r = cryosparc_tools.describe_job_results(pref, project_uid=self.project_uid)
                        if r.get("success"):
                            ch_res = r.get("resolution_angstroms")
                            ch_cfar = r.get("cfar") if isinstance(r.get("cfar"), (int, float)) else None
                    except Exception:
                        pass
                if ch_res is None:
                    for strat in strategies_tried:
                        if (strat.get("result_job") or "").upper() == pref.upper():
                            m = strat.get("metrics") or {}
                            ch_res = m.get("result_resolution")
                            ch_cfar = m.get("result_cfar")
                            break
                if not challenger_beats_incumbent(
                    ch_res, ch_cfar,
                    incumbent.get("resolution_angstroms"), incumbent.get("cfar"),
                ):
                    preferred_job_uid = incumbent["job_uid"]
                    outputs["baseline_stands"] = True
                    assessment = f"{assessment or ''} | baseline stands (ratchet)".strip(" |")

        if preferred_job_uid:
            pref = preferred_job_uid.strip()
            outputs["final_refinement_job_uid"] = pref
            rec.primary_job_uid = pref
            loaded_metrics = False
            if cryosparc_tools is not None:
                try:
                    res = cryosparc_tools.describe_job_results(pref, project_uid=self.project_uid)
                    if res.get("success"):
                        for k in ("job_type", "resolution_angstroms", "box_size",
                                  "symmetry", "num_particles", "cfar", "cfar_label"):
                            if res.get(k) is not None:
                                rec.metrics[k] = res[k]
                        loaded_metrics = True
                except Exception as e:
                    logger.warning(
                        "update_improvement_record: describe_job_results failed for %s: %s",
                        pref, e,
                    )
            if not loaded_metrics:
                for strat in strategies_tried:
                    if (strat.get("result_job") or "").upper() == pref.upper():
                        m = strat.get("metrics") or {}
                        if m.get("result_resolution") is not None:
                            rec.metrics["resolution_angstroms"] = m["result_resolution"]
                        if m.get("result_cfar") is not None:
                            rec.metrics["cfar"] = m["result_cfar"]
                        break

        rec.stage_outputs = outputs
        if strategies_tried:
            rec.decisions = self._format_strategies_for_decisions(strategies_tried, baseline)
        if assessment:
            rec.assessment = assessment
        self.save()
        return rec

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

    def best_resolution(self, exclude: Optional["StageRecord"] = None) -> Optional[Dict[str, Any]]:
        """Return {stage, job_uid, resolution_angstroms, cfar} for the best overall result.

        Resolution-primary with a cFAR sanity floor: among records whose cFAR is
        acceptable (or unknown), pick the lowest resolution; break near-ties
        (<0.005 Å) by higher cFAR. Records with cFAR < 0.15 (no real structure /
        severe preferred orientation) are only considered when nothing better
        exists. This keeps the baseline a true ratchet — a slightly-worse-
        resolution job with marginally higher cFAR can no longer displace the
        incumbent (previously possible with the 70/30 composite score).
        """
        candidates = [
            r for r in self.records
            if r is not exclude and isinstance(r.metrics.get("resolution_angstroms"), (int, float))
        ]
        if not candidates:
            return None

        def _cfar_ok(r: "StageRecord") -> bool:
            c = r.metrics.get("cfar")
            return not isinstance(c, (int, float)) or c >= 0.15

        pool = [r for r in candidates if _cfar_ok(r)] or candidates
        best_rec = min(
            pool,
            key=lambda r: (
                round(r.metrics["resolution_angstroms"], 3),
                -(r.metrics.get("cfar") if isinstance(r.metrics.get("cfar"), (int, float)) else 0.0),
            ),
        )
        res = best_rec.metrics["resolution_angstroms"]
        cfar = best_rec.metrics.get("cfar")
        res_score = max(0, min(1, (5.0 - res) / (5.0 - 1.5)))
        cfar_score = cfar if isinstance(cfar, (int, float)) else 0.5
        return {
            "stage": best_rec.stage,
            "job_uid": best_rec.primary_job_uid,
            "resolution_angstroms": res,
            "cfar": cfar,
            "composite_score": (0.7 * res_score) + (0.3 * cfar_score),
        }

    def summary_for_planner(self) -> str:
        """Compact human/LLM-readable digest of all stage metrics (JSON only).

        Shows all improvement rounds (improvement_1, improvement_2, ...) so the agent
        can see the full improvement history and avoid repeating failed approaches.
        """
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
            "current_stage": self.current_stage,
            "run_status": self.run_status,
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
            st.current_stage = data.get("current_stage")
            st.run_status = data.get("run_status") or cls.STATUS_IDLE
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

