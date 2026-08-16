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
            r'\*\*[Hh]ypothesis\*\*[:\s]+(.+?)(?=\n\n|\n[A-Z#*]|\Z)',
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
        best_score = None

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

            # Composite scoring: lower resolution is better, higher cFAR is better
            # Normalize both to a 0-1 scale where higher is better
            # Resolution: invert and scale (assume 1.5-5.0 Å range)
            res_score = max(0, min(1, (5.0 - r) / (5.0 - 1.5)))  # Lower res → higher score

            # cFAR: already 0-1 scale, higher is better (but can be None)
            # Use 0.5 as neutral default if cFAR is missing
            cfar_score = c if c is not None else 0.5

            # Weighted composite: resolution is primary (70%), cFAR is important (30%)
            # A job with good resolution but terrible cFAR will score poorly
            composite_score = (0.7 * res_score) + (0.3 * cfar_score)

            better = (
                best_score is None
                or composite_score > best_score
            )
            if better:
                best_uid, best_res, best_cfar, best_score = uid.strip(), r, c, composite_score

        if best_uid is None:
            logger.info("record_improvement: no candidate job produced a resolution; nothing recorded.")
            return None

        # Build rich decisions narrative from strategies_tried if available
        if strategies_tried and isinstance(strategies_tried, list):
            decisions = self._format_strategies_for_decisions(strategies_tried, baseline)

        return self.record_stage(
            stage="improvement",
            success=True,
            stage_outputs={
                "final_refinement_job_uid": best_uid,
                "strategies_tried": strategies_tried or [],
                "baseline": baseline or {}
            },
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
            success_icon = "✓" if strat.get("success") else "✗"
            hypothesis = strat.get("hypothesis", "Unknown")[:80]
            outcome = strat.get("outcome", "No outcome recorded")
            conclusion = strat.get("conclusion", "")[:100]

            lines.append(f"{i}. {success_icon} {hypothesis}")
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
        """Return {stage, job_uid, resolution_angstroms, cfar} for the best overall result.

        Uses composite scoring (70% resolution, 30% cFAR) to pick the truly best result,
        not just the lowest resolution. Scans all records including all improvement rounds.
        """
        best = None
        best_score = None
        for r in self.records:
            res = r.metrics.get("resolution_angstroms")
            cfar = r.metrics.get("cfar")
            if isinstance(res, (int, float)):
                # Composite scoring: resolution primary (70%), cFAR important (30%)
                res_score = max(0, min(1, (5.0 - res) / (5.0 - 1.5)))  # Lower res → higher score
                cfar_score = cfar if isinstance(cfar, (int, float)) else 0.5  # Default 0.5 if missing
                composite_score = (0.7 * res_score) + (0.3 * cfar_score)

                if best_score is None or composite_score > best_score:
                    best = {
                        "stage": r.stage,
                        "job_uid": r.primary_job_uid,
                        "resolution_angstroms": res,
                        "cfar": cfar,
                        "composite_score": composite_score,
                    }
                    best_score = composite_score
        return best

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

