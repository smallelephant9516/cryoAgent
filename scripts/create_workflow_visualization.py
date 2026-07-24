#!/usr/bin/env python3
"""
Unified script to create workflow_state.json and vis_report.json from workflow output folders.

This script combines functionality from:
- create_workflow_state.py (basic parsing)
- create_workflow_state_v2.py (improved parsing with LLM log analysis)
- enrich_vis_report_with_cryosparc.py (live CryoSPARC metric enrichment)
- cryoagent/core/workflow_state.py (live system patterns)

Features:
- Parses stage result JSON files
- Extracts decisions from LLM conversation logs
- Optionally enriches with live CryoSPARC metrics
- Can use LLM to generate summary descriptions
- Creates both workflow_state.json and vis_report.json
- Reads CryoSPARC license and LLM config from master_config.json

Usage:
    # Basic: create from result files only (no CryoSPARC connection)
    python scripts/create_workflow_visualization.py /path/to/outputs/trial_3

    # With CryoSPARC enrichment (reads license from master_config.json)
    python scripts/create_workflow_visualization.py --enrich /path/to/outputs/trial_3

    # With LLM-generated summaries (reads API keys from master_config.json)
    python scripts/create_workflow_visualization.py --llm-summary /path/to/outputs/trial_3

    # Full enrichment (CryoSPARC + LLM)
    python scripts/create_workflow_visualization.py --enrich --llm-summary /path/to/outputs/trial_3

    # Batch process multiple workflows
    python scripts/create_workflow_visualization.py /path/to/outputs/dynamic_mode

    # Force recreate existing files
    python scripts/create_workflow_visualization.py --force /path/to/outputs/trial_3

Configuration:
    All configuration is read from configs/master_config.json:
    - cryosparc.license_id - CryoSPARC license (can use ${LICENSE_ID} env var)
    - agent.models.* - LLM API keys and endpoints
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path to import cryoagent modules
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONFIGURATION LOADING ===

def load_master_config() -> Dict[str, Any]:
    """Load master_config.json with environment variable substitution."""
    config_path = Path(__file__).parent.parent / "configs" / "master_config.json"
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Substitute environment variables recursively
        config = substitute_env_vars(config)
        return config
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        return {}


def substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute ${VAR_NAME} with environment variables."""
    if isinstance(obj, dict):
        return {k: substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        return os.environ.get(var_name, obj)
    else:
        return obj


def get_llm_client(config: Dict[str, Any]):
    """Initialize LLM client from config. Returns None if not available."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed. Install with: pip install openai")
        return None

    agent_config = config.get("agent", {})
    provider = agent_config.get("provider", "deepseek")
    models_config = agent_config.get("models", {})
    provider_config = models_config.get(provider, {})

    api_key = provider_config.get("api_key", "")
    base_url = provider_config.get("base_url", "")
    model_name = provider_config.get("model_name", "")

    # Check if environment variables are still placeholders
    if not api_key or api_key.startswith("${"):
        logger.warning(f"API key not set for {provider}. Set the appropriate environment variable.")
        return None

    if not base_url or not model_name:
        logger.warning(f"Incomplete {provider} configuration")
        return None

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return {
            "client": client,
            "model": model_name,
            "temperature": provider_config.get("temperature", 0.1),
            "provider": provider
        }
    except Exception as e:
        logger.warning(f"Failed to initialize LLM client: {e}")
        return None


def generate_workflow_summary_with_llm(llm_config: Dict[str, Any], workflow_state: Dict[str, Any]) -> Optional[str]:
    """Use LLM to generate a human-readable summary of the workflow."""
    if not llm_config:
        return None

    try:
        client = llm_config["client"]
        model = llm_config["model"]

        # Prepare workflow data for LLM
        records = workflow_state.get("records", [])

        summary_parts = []
        for record in records:
            stage = record.get("stage", "")
            metrics = record.get("metrics", {})
            decisions = record.get("decisions", [])

            stage_info = f"Stage: {stage}"
            if metrics:
                metric_strs = [f"{k}={v}" for k, v in metrics.items()]
                stage_info += f"\n  Metrics: {', '.join(metric_strs)}"
            if decisions:
                stage_info += f"\n  Decisions: {'; '.join(decisions[:3])}"

            summary_parts.append(stage_info)

        workflow_summary = "\n\n".join(summary_parts)

        prompt = f"""Analyze this cryo-EM workflow and provide a concise 2-3 sentence summary of what was accomplished:

{workflow_summary}

Focus on: final resolution achieved, key optimizations made, and overall workflow success."""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_config.get("temperature", 0.1),
            max_tokens=200
        )

        summary = response.choices[0].message.content.strip()
        logger.info(f"  ✓ Generated LLM summary using {llm_config['provider']}")
        return summary

    except Exception as e:
        logger.warning(f"Failed to generate LLM summary: {e}")
        return None


def extract_improvement_strategies_with_llm(llm_config: Dict[str, Any], log_path: Path) -> Dict[str, Any]:
    """Use LLM to extract improvement strategies from improvement log in a flexible way."""
    if not llm_config or not log_path.exists():
        return {"baseline": {}, "strategies_tried": []}

    try:
        # Read the log file
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")

        # Truncate if too long (keep first 20k and last 20k characters)
        if len(log_text) > 50000:
            log_text = log_text[:20000] + "\n\n... [middle truncated] ...\n\n" + log_text[-20000:]

        client = llm_config["client"]
        model = llm_config["model"]

        prompt = """Extract improvement strategies from this cryo-EM improvement workflow log.

Find:
1. Baseline: the starting job, resolution, and cFAR
2. All strategies tried: each hypothesis tested with its result

Return ONLY valid JSON in this exact format (no markdown, no code fences):
{
  "baseline": {
    "job": "J443",
    "resolution": 2.73,
    "cfar": 0.779
  },
  "strategies_tried": [
    {
      "hypothesis": "CTF aberrations limiting resolution",
      "approach": "CTF Refine Global + Local then Nonuniform Refinement",
      "job_chain": "J508→J509→J510",
      "result_job": "J510",
      "baseline_resolution": 2.73,
      "result_resolution": 2.67,
      "baseline_cfar": 0.779,
      "result_cfar": 0.783,
      "success": true,
      "conclusion": "Meaningful gain"
    }
  ]
}

CRITICAL RULES:
- Return ONLY the JSON object, no other text
- Do NOT wrap in markdown code fences
- Escape all quotes inside string values using backslash
- success=true if resolution improved meaningfully (≥0.02 Å improvement)
- result_job is the FINAL job in the chain that produced the result
- Keep hypothesis and approach under 150 characters each
- Extract ALL strategies tried, even failed ones

Log content:
""" + log_text

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a cryo-EM workflow analyzer. Extract structured data from logs. Return ONLY valid JSON, no markdown formatting, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=4000
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown code fences if present
        result_text = re.sub(r'^```(?:json)?\s*\n?', '', result_text)
        result_text = re.sub(r'\s*\n?```\s*$', '', result_text)

        # Try to parse JSON - if it fails, try to clean it up
        try:
            extracted = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON (attempt 1): {e}")
            # Try to find JSON object in the text
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
                try:
                    extracted = json.loads(result_text)
                    logger.info("  ✓ Recovered JSON after cleanup")
                except json.JSONDecodeError as e2:
                    logger.warning(f"LLM returned invalid JSON (attempt 2): {e2}")
                    return {"baseline": {}, "strategies_tried": []}
            else:
                return {"baseline": {}, "strategies_tried": []}

        # Transform to match expected format
        context = {
            "baseline": extracted.get("baseline", {}),
            "strategies_tried": [],
            "job_uids": []
        }

        for strat in extracted.get("strategies_tried", []):
            baseline_res = strat.get("baseline_resolution")
            result_res = strat.get("result_resolution")
            baseline_cfar = strat.get("baseline_cfar")
            result_cfar = strat.get("result_cfar")

            delta_res = baseline_res - result_res if (baseline_res and result_res) else None
            delta_cfar = result_cfar - baseline_cfar if (baseline_cfar and result_cfar) else None

            strategy = {
                "hypothesis": strat.get("hypothesis", "")[:200],
                "approach": strat.get("approach", "")[:200],
                "actions": [],
                "result_job": strat.get("result_job", ""),
                "outcome": f"Resolution: {baseline_res:.2f} → {result_res:.2f} Å ({delta_res:+.2f} Å)" if delta_res else "",
                "conclusion": strat.get("conclusion", "")[:150],
                "success": strat.get("success", False),
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

            job = strat.get("result_job", "")
            if job and job not in context["job_uids"]:
                context["job_uids"].append(job)

        logger.info(f"  ✓ Extracted {len(context['strategies_tried'])} strategies using LLM")
        return context

    except Exception as e:
        logger.warning(f"Failed to extract strategies with LLM: {e}")
        return {"baseline": {}, "strategies_tried": []}


def extract_optimization_iterations_with_llm(llm_config: Dict[str, Any], log_path: Path) -> List[Dict[str, Any]]:
    """Use LLM to extract optimization iterations from optimization log."""
    if not llm_config or not log_path.exists():
        return []

    try:
        # Read the log file
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")

        # Truncate if too long (keep first 30k and last 30k characters for optimization)
        if len(log_text) > 80000:
            log_text = log_text[:30000] + "\n\n... [middle truncated] ...\n\n" + log_text[-30000:]

        client = llm_config["client"]
        model = llm_config["model"]

        prompt = """Extract optimization iterations from this cryo-EM workflow log following STRICT format rules.

The workflow has 4 phases in order:
1. Baseline (iteration 0) - initial reconstruction
2. 3D classification rounds (iterations 1+) - multi-round ab_initio + hetero refinement
3. Heterogeneous refinement K-value tests (iterations N+) - testing K=2, K=3, K=4
4. Box size optimization (iterations M+) - testing different box sizes

Return ONLY valid JSON (no markdown, no code fences):
{
  "iterations": [
    {
      "iteration": 0,
      "phase": "reconstruction_baseline",
      "job_uid": "J422",
      "job_type": "homo_abinit",
      "description": "Baseline from reconstruction",
      "box_size": 324,
      "num_particles": 5989,
      "note": "Baseline from reconstruction stage"
    },
    {
      "iteration": 1,
      "phase": "3d_classification",
      "job_uid": "J427",
      "job_type": "nonuniform_refine_new",
      "description": "3D classification - Round 1",
      "box_size": 324,
      "resolution": 2.80,
      "num_particles": 5500,
      "round": 1
    },
    {
      "iteration": 2,
      "phase": "3d_classification",
      "job_uid": "J430",
      "job_type": "nonuniform_refine_new",
      "description": "3D classification - Round 2",
      "box_size": 324,
      "resolution": 2.73,
      "num_particles": 5467,
      "round": 2
    },
    {
      "iteration": 3,
      "phase": "heterogeneous_refinement",
      "job_uid": "J432",
      "job_type": "nonuniform_refine_new",
      "description": "Heterogeneous refinement K=2",
      "box_size": 324,
      "resolution": 2.67,
      "k": 2
    },
    {
      "iteration": 4,
      "phase": "heterogeneous_refinement",
      "job_uid": "J434",
      "job_type": "nonuniform_refine_new",
      "description": "Heterogeneous refinement K=3",
      "box_size": 324,
      "resolution": 2.69,
      "k": 3
    },
    {
      "iteration": 5,
      "phase": "box_size_optimization",
      "job_uid": "J437",
      "job_type": "nonuniform_refine_new",
      "description": "Box size optimization: 292px",
      "box_size": 292,
      "resolution": 2.79
    }
  ]
}

CRITICAL RULES:
1. Phase sequence MUST be: "reconstruction_baseline" → "3d_classification" → "heterogeneous_refinement" → "box_size_optimization"
2. Each 3D classification ROUND creates ONE iteration with the FINAL NU refinement result (e.g., J425→J426→J427 = one iteration with J427's resolution)
3. Each heterogeneous K-value test creates ONE iteration (e.g., K=2 hetero J431 → NU J432 = one iteration with J432's resolution and k=2)
4. Each box size test creates ONE iteration with that box_size
5. job_uid is the FINAL NU refinement job, resolution from that NU job's get_fsc_info result
6. "3d_classification" iterations get "round" field (1, 2, 3...)
7. "heterogeneous_refinement" iterations get "k" field (2, 3, 4...)
8. "box_size_optimization" iterations get "box_size" field
9. Baseline (iteration 0) comes from reconstruction stage, may not have resolution
10. Return ONLY JSON, no explanations

Log content:
""" + log_text

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a cryo-EM workflow analyzer. Extract structured iteration data from optimization logs. Return ONLY valid JSON, no markdown formatting, no explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=6000
        )

        result_text = response.choices[0].message.content.strip()

        # Remove markdown code fences if present
        result_text = re.sub(r'^```(?:json)?\s*\n?', '', result_text)
        result_text = re.sub(r'\s*\n?```\s*$', '', result_text)

        # Try to parse JSON - if it fails, try to clean it up
        try:
            extracted = json.loads(result_text)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON (attempt 1): {e}")
            # Try to find JSON object in the text
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(0)
                try:
                    extracted = json.loads(result_text)
                    logger.info("  ✓ Recovered JSON after cleanup")
                except json.JSONDecodeError as e2:
                    logger.warning(f"LLM returned invalid JSON (attempt 2): {e2}")
                    return []
            else:
                return []

        iterations = extracted.get("iterations", [])
        logger.info(f"  ✓ Extracted {len(iterations)} optimization iterations using LLM")
        return iterations

    except Exception as e:
        logger.warning(f"Failed to extract optimization iterations with LLM: {e}")
        return []


# === CONSTANTS (from workflow_state.py) ===

# Borrowed from workflow_state.py - keys checked in order to find primary job
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

# Tools that are infrastructure/diagnostics, not workflow decisions
_NON_ACTION_TOOLS = frozenset({
    "wait_for_job", "get_job_status", "get_job_log", "get_job_log_common",
    "describe_job_params", "describe_job_results", "get_orientation_diagnostics",
    "get_fsc_info", "get_particle_count", "search_cryosparc_forum",
    "reason_about_workflow", "verify_inputs", "read_input_json",
    "get_hetero_class_resolutions", "get_regroup_superclass_info",
    "consult_cryosparc_guide",
})

# Per-tool params worth surfacing as a "decision"
_SALIENT_PARAMS = (
    "num_classes", "num_superclasses", "symmetry", "particle_diameter",
    "refinement_resolution", "initial_resolution", "final_resolution",
    "box_size",
)

# Canonical stage name -> result-JSON filename prefix (in pipeline order)
_STAGE_RESULT_PREFIXES = [
    ("preprocessing", "preprocessing"),
    ("particle_picking", "particle_picking"),
    ("optimization_2d", "2d_optimization"),
    ("reconstruction", "reconstruction"),
    ("optimization", "optimization"),
    ("polish", "polish"),
]


# === UTILITY FUNCTIONS ===

def pick_primary_job_uid(stage_outputs: Dict[str, Any]) -> Optional[str]:
    """Choose the 'main' job UID a stage produced, by key convention."""
    if not isinstance(stage_outputs, dict):
        return None
    for key in _PRIMARY_JOB_KEYS:
        v = stage_outputs.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def flatten_stage_outputs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Lift nested dicts (e.g. reconstruction's outputs.* and job_uids.*) to top level."""
    if not isinstance(data, dict):
        return {}
    flat = dict(data)
    for nest_key in ("outputs", "job_uids"):
        nested = data.get(nest_key)
        if isinstance(nested, dict):
            for k, v in nested.items():
                if k not in flat and isinstance(v, (str, int, float)):
                    flat[k] = v
    return flat


# === LLM LOG PARSING ===

def parse_tool_execution_log(text: str) -> List[Dict[str, Any]]:
    """Extract tool execution entries from conversation log text."""
    entries: List[Dict[str, Any]] = []
    chunks = re.split(r"TOOL EXECUTION:\s*", text)

    for chunk in chunks[1:]:
        tool = chunk.splitlines()[0].strip() if chunk.strip() else ""
        if not tool:
            continue

        entry: Dict[str, Any] = {"tool": tool}

        # Extract Arguments JSON
        m = re.search(r"Arguments:\s*(\{.*?\})\s*(?:Result:|-{5,}|\Z)", chunk, flags=re.S)
        if m:
            try:
                entry["params"] = json.loads(m.group(1))
            except Exception:
                entry["params"] = {}

        # Extract job_uid from Result
        rm = re.search(
            r"Result:\s*\{.*?['\"]job_uid['\"]\s*:\s*['\"](J\d+)['\"]", chunk, flags=re.S,
        )
        if rm:
            entry["result"] = {"job_uid": rm.group(1)}

        entries.append(entry)

    return entries


def summarize_decisions_from_log(tool_log: List[Dict[str, Any]], max_items: int = 8) -> List[str]:
    """Derive a short list of action choices from tool execution log."""
    decisions: List[str] = []

    for entry in tool_log:
        tool = entry.get("tool")
        if not tool or tool in _NON_ACTION_TOOLS or entry.get("error"):
            continue

        result = entry.get("result")
        job_uid = result.get("job_uid") if isinstance(result, dict) else None
        params = entry.get("params") or {}

        # Extract salient parameters
        salient = {k: params[k] for k in _SALIENT_PARAMS if k in params and params[k] is not None}
        bits = ", ".join(f"{k}={v}" for k, v in salient.items())
        label = f"{tool}({bits})" if bits else tool

        if job_uid:
            label += f" -> {job_uid}"

        decisions.append(label)

    # De-dup consecutive repeats, cap length
    deduped: List[str] = []
    for d in decisions:
        if not deduped or deduped[-1] != d:
            deduped.append(d)

    return deduped[:max_items]


def parse_conversation_log(log_path: Path) -> Dict[str, Any]:
    """Parse LLM conversation log to extract goal, decisions, reasoning, assessment."""
    result = {
        "goal": None,
        "decisions": [],
        "reasoning_summary": None,
        "assessment": None,
    }

    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return result

    # Extract goal from system prompt or first assistant response
    goal_m = re.search(r"## Goal[:\n]+(.*?)(?=\n##|\n-{50}|\Z)", text, flags=re.S | re.I)
    if goal_m:
        result["goal"] = goal_m.group(1).strip()[:300]

    # Extract decisions from tool execution log
    tool_log = parse_tool_execution_log(text)
    if tool_log:
        result["decisions"] = summarize_decisions_from_log(tool_log)

    # Extract reasoning summary from last assistant block
    assistant_blocks = re.findall(
        r"Assistant:(.*?)(?=User:|TOOL EXECUTION:|CONVERSATION ENDED|\Z)",
        text, flags=re.S
    )
    if assistant_blocks:
        last = assistant_blocks[-1].split("Metadata:")[0].strip()
        summary_lines = [
            ln.strip() for ln in last.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if summary_lines:
            result["reasoning_summary"] = " ".join(summary_lines)[:500]

    # Extract assessment
    assess_m = re.search(
        r"## Remaining Limitations / Recommended Next Steps\n(.*?)(?=\nMetadata:|\n-{50}|\Z)",
        text, flags=re.S,
    )
    if assess_m:
        result["assessment"] = assess_m.group(1).strip()[:800]

    return result


# === CRYOSPARC ENRICHMENT ===

def load_cryosparc_config(master_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Load CryoSPARC config from master_config.json and environment."""
    if master_config is None:
        config_path = Path(__file__).parent.parent / "configs" / "master_config.json"
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return None

        try:
            with open(config_path) as f:
                master_config = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read config: {e}")
            return None

    cs_config = master_config.get("cryosparc", {})

    # Get license ID from environment or config
    license_id = os.environ.get("CRYOSPARC_LICENSE_ID", cs_config.get("license_id", ""))
    if license_id.startswith("${"):
        license_id = os.environ.get("CRYOSPARC_LICENSE_ID", "")

    if not license_id:
        logger.warning("CRYOSPARC_LICENSE_ID not set in config or environment")
        return None

    return {
        "license": license_id,
        "host": cs_config.get("host", "localhost"),
        "base_port": cs_config.get("base_port", 39000),
        "email": cs_config.get("username", ""),
        "password": cs_config.get("password", "")
    }


def connect_to_cryosparc(config: Dict[str, Any]):
    """Connect to CryoSPARC instance. Returns None if cryosparc-tools not available."""
    try:
        from cryosparc.tools import CryoSPARC
    except ImportError:
        logger.warning("cryosparc-tools not installed. Skipping enrichment.")
        logger.info("Install with: pip install cryosparc-tools")
        return None

    try:
        cs = CryoSPARC(**config)
        logger.info(f"✓ Connected to CryoSPARC at {config['host']}:{config['base_port']}")
        return cs
    except Exception as e:
        logger.warning(f"Failed to connect to CryoSPARC: {e}")
        return None


def extract_metrics_from_cryosparc(cs, project_uid: str, job_uid: str) -> Dict[str, Any]:
    """Query CryoSPARC for job metrics (mimics workflow_state.py approach)."""
    metrics = {}

    if not job_uid or not project_uid or not cs:
        return metrics

    try:
        job = cs.find_job(project_uid, job_uid)
        if not job:
            logger.debug(f"Job {job_uid} not found in project {project_uid}")
            return metrics

        job.refresh()
        doc = getattr(job, "doc", {}) or {}
        job_type = doc.get("type") or doc.get("job_type")
        output_groups = doc.get("output_result_groups", []) or []

        if job_type:
            metrics["job_type"] = job_type

        # Extract particle count
        for g in output_groups:
            if isinstance(g, dict) and g.get("type") == "particle":
                name = g.get("name") or ""
                if name in ("particles", "particles_all_classes", "particles_selected"):
                    metrics["num_particles"] = g.get("num_items")
                    break

        # Extract micrograph count for preprocessing jobs
        for g in output_groups:
            if isinstance(g, dict) and g.get("type") in ("exposure", "micrograph"):
                name = g.get("name") or ""
                if name in ("exposures_accepted", "micrographs"):
                    metrics["num_micrographs"] = g.get("num_items")
                    break

        # Extract resolution from latest_summary_stats (for refinement jobs)
        jt = (job_type or "").lower()
        is_refine = "refine" in jt or "abinit" in jt

        if is_refine:
            for g in output_groups:
                if isinstance(g, dict) and g.get("latest_summary_stats"):
                    stats = g["latest_summary_stats"]

                    # Look for resolution in fsc_info first (preferred location)
                    fsc_info = stats.get("fsc_info", {})
                    if isinstance(fsc_info, dict):
                        # Try radwn_tightmask_A first (most commonly used), then others
                        for key in ["radwn_tightmask_A", "radwn_noisesub_A", "radwn_final_A"]:
                            if key in fsc_info:
                                try:
                                    metrics["resolution_angstroms"] = float(fsc_info[key])
                                    break
                                except (TypeError, ValueError):
                                    continue

                        # Fallback: any radwn_*_A key
                        if "resolution_angstroms" not in metrics:
                            for key, val in fsc_info.items():
                                if key.startswith("radwn_") and key.endswith("_A"):
                                    try:
                                        metrics["resolution_angstroms"] = float(val)
                                        break
                                    except (TypeError, ValueError):
                                        continue

                    # Fallback: Look for resolution in stats root (old format)
                    if "resolution_angstroms" not in metrics:
                        for key, val in stats.items():
                            if key.startswith("radwn_") and key.endswith("_A"):
                                try:
                                    metrics["resolution_angstroms"] = float(val)
                                    break
                                except (TypeError, ValueError):
                                    continue

                    # Get box size if available
                    for cont in (stats, stats.get("fsc_info", {}), stats.get("fsc_info_autotight", {})):
                        if isinstance(cont, dict) and cont.get("N"):
                            metrics["box_size"] = int(cont.get("N"))
                            break

                    if "resolution_angstroms" in metrics:
                        break

        # Extract symmetry from params_spec
        params_spec = doc.get("params_spec", {}) or {}
        for sk in ("refine_symmetry", "abinit_symmetry", "multirefine_symmetry"):
            if sk in params_spec and isinstance(params_spec[sk], dict):
                metrics["symmetry"] = params_spec[sk].get("value")
                break

        return metrics

    except Exception as e:
        logger.debug(f"Could not extract metrics from job {job_uid}: {e}")
        return metrics


# === WORKFLOW PARSING ===

def discover_workflow_folders(base_dir: Path) -> List[Path]:
    """Discover all workflow folders that contain stage result files."""
    folders = []

    # Check if base_dir itself has result files
    result_files = list(base_dir.glob("*_results_*.json"))
    if result_files:
        folders.append(base_dir)
        return folders

    # Otherwise search subdirectories
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            result_files = list(subdir.glob("*_results_*.json"))
            if result_files:
                folders.append(subdir)

    return folders


def extract_project_workspace_from_results(workflow_dir: Path) -> Tuple[str, str]:
    """Extract project_uid and workspace_uid from stage result files."""
    result_files = list(workflow_dir.glob("*_results_*.json"))

    for result_file in result_files:
        try:
            with open(result_file) as f:
                data = json.load(f)
                project_uid = data.get("project_uid", "")
                workspace_uid = data.get("workspace_uid", "")
                if project_uid and workspace_uid:
                    return project_uid, workspace_uid
        except Exception as e:
            logger.debug(f"Could not read {result_file.name}: {e}")
            continue

    return "unknown", workflow_dir.name


def extract_metrics_from_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from result data using various field name conventions."""
    metrics = {}

    # Resolution fields (various naming conventions)
    for key in ("best_resolution_angstroms", "final_resolution", "resolution", "resolution_angstroms"):
        if key in data and isinstance(data[key], (int, float)):
            metrics["resolution_angstroms"] = float(data[key])
            break

    # Box size
    for key in ("best_box_size", "box_size"):
        if key in data and isinstance(data[key], (int, float)):
            metrics["box_size"] = int(data[key])
            break

    # Particle count
    for key in ("num_particles", "total_particles", "final_good_particles_count"):
        if key in data and isinstance(data[key], int):
            metrics["num_particles"] = data[key]
            break

    # Other common metrics
    if "symmetry" in data:
        metrics["symmetry"] = data["symmetry"]
    if "cfar" in data:
        metrics["cfar"] = data["cfar"]
    if "num_classes" in data:
        metrics["num_classes"] = data["num_classes"]
    if "job_type" in data:
        metrics["job_type"] = data["job_type"]
    if "num_micrographs" in data:
        metrics["num_micrographs"] = data["num_micrographs"]

    return metrics


def parse_stage_results(workflow_dir: Path, cs_connection=None, project_uid: str = "", llm_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Parse all *_results_*.json files to build stage records."""
    result_files = list(workflow_dir.glob("*_results_*.json"))

    # Group by stage name
    stages_data = {}

    for result_file in result_files:
        # Extract stage name from filename: {stage}_results_*.json
        match = re.match(r"(.+?)_results_", result_file.name)
        if not match:
            continue

        stage_name = match.group(1)

        try:
            with open(result_file) as f:
                data = json.load(f)

            # Use the most recent file per stage
            if stage_name not in stages_data or result_file.stat().st_mtime > stages_data[stage_name]["mtime"]:
                stages_data[stage_name] = {
                    "data": data,
                    "mtime": result_file.stat().st_mtime,
                    "file": result_file
                }
        except Exception as e:
            logger.warning(f"Could not parse {result_file.name}: {e}")
            continue

    # Convert to stage records, sorted by timestamp (chronological order)
    records = []

    # Try to sort by timestamp in data, fall back to mtime
    def get_sort_key(item):
        stage_name, stage_info = item
        data = stage_info["data"]
        ts = data.get("timestamp")
        if isinstance(ts, (int, float)):
            return ts
        if isinstance(ts, str):
            try:
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                return dt.timestamp()
            except:
                pass
        return stage_info["mtime"]

    sorted_stages = sorted(stages_data.items(), key=get_sort_key)

    for stage_name, stage_info in sorted_stages:
        data = stage_info["data"]
        stage_outputs = flatten_stage_outputs(data.get("stage_outputs", data))

        # Extract primary job UID using convention
        primary_job_uid = pick_primary_job_uid(stage_outputs)
        if not primary_job_uid:
            primary_job_uid = data.get("primary_job_uid", data.get("best_job_uid", ""))

        # Extract metrics from result file
        metrics = extract_metrics_from_data(data)

        # Enrich with live CryoSPARC metrics if connection available
        if cs_connection and primary_job_uid and project_uid:
            cs_metrics = extract_metrics_from_cryosparc(cs_connection, project_uid, primary_job_uid)
            # For optimization/improvement stages, prefer file metrics (historical values)
            # For other stages, prefer live CryoSPARC metrics (more accurate)
            if stage_name in ('optimization', 'improvement', 'polish') or stage_name.startswith('improvement_'):
                # File metrics take precedence, fill in missing with CryoSPARC
                metrics = {**cs_metrics, **metrics}
            else:
                # Live metrics take precedence over file metrics
                metrics = {**metrics, **cs_metrics}

        # Special handling for reconstruction stage - extract resolution if missing
        if stage_name in ('reconstruction', '3d_reconstruction') and 'resolution_angstroms' not in metrics:
            if cs_connection and primary_job_uid and project_uid:
                cs_metrics = extract_metrics_from_cryosparc(cs_connection, project_uid, primary_job_uid)
                if 'resolution_angstroms' in cs_metrics:
                    metrics['resolution_angstroms'] = cs_metrics['resolution_angstroms']

        # Parse conversation log if available
        log_data = {}
        log_files = list(workflow_dir.glob(f"llm_conversation_{stage_name}_*.log"))
        if log_files:
            latest_log = max(log_files, key=lambda p: p.stat().st_mtime)
            log_data = parse_conversation_log(latest_log)

        # Build timestamp
        ts = data.get("timestamp")
        if isinstance(ts, str):
            try:
                dt = datetime.strptime(ts, "%Y%m%d_%H%M%S")
                timestamp = dt.timestamp()
            except:
                timestamp = stage_info["mtime"]
        elif isinstance(ts, (int, float)):
            timestamp = ts
        else:
            timestamp = stage_info["mtime"]

        record = {
            "stage": stage_name,
            "success": data.get("status") == "completed" or data.get("success", True),
            "primary_job_uid": primary_job_uid,
            "metrics": metrics,
            "goal": log_data.get("goal") or data.get("goal", ""),
            "decisions": log_data.get("decisions") or data.get("decisions", []),
            "reasoning_summary": log_data.get("reasoning_summary") or data.get("reasoning_summary", ""),
            "assessment": log_data.get("assessment") or data.get("assessment", ""),
            "stage_outputs": stage_outputs,
            "timestamp": timestamp,
            "execution_time": stage_outputs.get("execution_time", 0)
        }

        records.append(record)

    # === IMPROVEMENT STAGE PARSING ===
    # Look for improvement logs that don't have corresponding result files
    improvement_logs = list(workflow_dir.glob("llm_conversation_improvement_*.log"))

    # Check if we already have improvement stages from result files
    existing_improvement_stages = [r["stage"] for r in records if r["stage"].startswith("improvement")]

    for log_path in improvement_logs:
        # If we already have improvement records, skip this log
        if existing_improvement_stages:
            logger.info(f"Skipping {log_path.name} - improvement stages already exist from result files")
            continue

        # Try to extract rich context from the log
        # First try regex-based parser, then fall back to LLM if it finds nothing
        from cryoagent.core.workflow_state import extract_improvement_context_from_log

        try:
            context = extract_improvement_context_from_log(log_path)
            candidates = context.get("job_uids", [])
            strategies = context.get("strategies_tried", [])
            baseline = context.get("baseline", {})

            # If regex parser found nothing and we have LLM access, try LLM extraction
            if not strategies and llm_config:
                logger.info(f"📋 Regex parser found no strategies, trying LLM extraction for {log_path.name}")
                context = extract_improvement_strategies_with_llm(llm_config, log_path)
                candidates = context.get("job_uids", [])
                strategies = context.get("strategies_tried", [])
                baseline = context.get("baseline", {})

            if not candidates:
                logger.info(f"No job UIDs found in {log_path.name}, skipping")
                continue

            logger.info(f"📋 Found improvement log with {len(candidates)} job(s) and {len(strategies)} strategy/ies")

            # Show what strategies were tried
            for i, strat in enumerate(strategies[:3], 1):
                success_icon = "✓" if strat.get("success") else "✗"
                hyp = strat.get("hypothesis", "Unknown")[:60]
                logger.info(f"   {success_icon} {hyp}...")
            if len(strategies) > 3:
                logger.info(f"   ... and {len(strategies) - 3} more")

            # Pick the best job as primary (last candidate, which is usually the final result)
            primary_job_uid = candidates[-1] if candidates else ""

            # Extract metrics: prefer recorded strategy results over live CryoSPARC
            # This ensures we use the resolution that was recorded at the time
            metrics = {}

            # First, try to get the best resolution from strategies
            best_resolution = None
            if strategies:
                # Find the best (lowest) resolution among all strategies
                all_resolutions = []
                for strat in strategies:
                    if "metrics" in strat and "result_resolution" in strat["metrics"]:
                        res = strat["metrics"]["result_resolution"]
                        if isinstance(res, (int, float)) and res > 0:
                            all_resolutions.append(res)

                if all_resolutions:
                    best_resolution = min(all_resolutions)
                    metrics["resolution_angstroms"] = best_resolution

            # If no resolution from strategies, use baseline
            if not best_resolution and baseline:
                baseline_res = baseline.get("resolution")
                if isinstance(baseline_res, (int, float)) and baseline_res > 0:
                    metrics["resolution_angstroms"] = baseline_res

            # Get other metrics from CryoSPARC (particles, box size, etc.) but not resolution
            if cs_connection and primary_job_uid and project_uid:
                cs_metrics = extract_metrics_from_cryosparc(cs_connection, project_uid, primary_job_uid)
                # Only take non-resolution metrics from CryoSPARC
                cs_metrics.pop("resolution_angstroms", None)
                metrics.update(cs_metrics)

            # Set job type
            if "job_type" not in metrics:
                metrics["job_type"] = "improvement_iteration"

            # Build decisions list from strategies
            decisions = []
            for i, strat in enumerate(strategies, 1):
                success_icon = "✓" if strat.get("success") else "✗"
                hyp = strat.get("hypothesis", "Unknown")[:90]
                outcome = strat.get("outcome", "")
                conclusion = strat.get("conclusion", "")

                decision = f"{i}. {success_icon} {hyp}"
                if outcome:
                    decision += f"\n   → {outcome}"
                if conclusion:
                    decision += f"\n   → {conclusion}"
                if not strat.get("success"):
                    reason = "Why failed: " + (strat.get("conclusion", "unknown")[:80])
                    decision += f"\n   → {reason}"

                decisions.append(decision)

            # Determine stage number (improvement_1, improvement_2, etc.)
            improvement_count = len([r for r in records if r["stage"].startswith("improvement")])
            stage_name = f"improvement_{improvement_count + 1}"

            # Determine if the improvement workflow completed successfully
            # Success = workflow completed (not necessarily found improvements)
            # Check if log indicates completion (has phase="final" or stopped naturally)
            workflow_completed = True  # Default to True if we parsed strategies
            try:
                with open(log_path) as f:
                    log_content = f.read()
                    # If agent was interrupted/errored before completion, mark as failed
                    if "Error:" in log_content or "Exception:" in log_content:
                        # Check if it's at the end (real failure) vs during execution (recovered)
                        last_500 = log_content[-500:]
                        if "Error:" in last_500 or "Exception:" in last_500:
                            workflow_completed = False
            except:
                pass

            # Build the record
            improvement_record = {
                "stage": stage_name,
                "success": workflow_completed,  # True if workflow completed, regardless of finding improvements
                "primary_job_uid": primary_job_uid,
                "metrics": metrics,
                "assessment": f"retrospective recording from {log_path.name}",
                "goal": None,
                "decisions": decisions,
                "reasoning_summary": None,
                "stage_outputs": {
                    "final_refinement_job_uid": primary_job_uid,
                    "strategies_tried": strategies,
                    "baseline": baseline
                },
                "timestamp": log_path.stat().st_mtime,
                "execution_time": 0
            }

            records.append(improvement_record)
            logger.info(f"✅ Created {stage_name} record from log: {primary_job_uid}")

        except Exception as e:
            logger.warning(f"Failed to parse improvement log {log_path.name}: {e}")
            continue

    return records


# === FILE CREATION ===

def create_workflow_state(workflow_dir: Path, cs_connection=None, llm_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create workflow_state.json structure from folder contents."""
    project_uid, workspace_uid = extract_project_workspace_from_results(workflow_dir)
    records = parse_stage_results(workflow_dir, cs_connection, project_uid, llm_config)

    return {
        "project_uid": project_uid,
        "workspace_uid": workspace_uid,
        "records": records
    }


def transform_optimization_detailed_results(
    stage_outputs: Dict[str, Any],
    workflow_dir: Path,
    cs_connection=None,
    project_uid: str = "",
    llm_config: Optional[Dict[str, Any]] = None,
    all_records: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Transform optimization stage data to match target format.

    Strategy: Try regex parsing from tested_combinations first.
    If fewer than 3 iterations found, fall back to LLM extraction from log.

    Args:
        all_records: All stage records to find reconstruction baseline resolution
    """
    # First pass: use tested_combinations from stage_outputs (regex parsing)
    tested_combinations = stage_outputs.get("tested_combinations", [])
    regex_iteration_count = len(tested_combinations)

    # Check if we need LLM fallback:
    # 1. Fewer than 3 iterations found, OR
    # 2. Optimization log exists and contains heterogeneous refinement jobs not in tested_combinations
    needs_llm_fallback = regex_iteration_count < 3
    llm_iterations = []

    # Check for missing heterogeneous refinement context
    if not needs_llm_fallback and llm_config:
        optimization_logs = list(workflow_dir.glob("llm_conversation_optimization_*.log"))
        if optimization_logs:
            latest_log = max(optimization_logs, key=lambda p: p.stat().st_mtime)
            try:
                log_text = latest_log.read_text(encoding="utf-8", errors="ignore")
                # Count heterogeneous refinement jobs in log
                hetero_jobs = set(re.findall(r"'job_uid':\s*'(J\d+)'.*?'job_type':\s*'hetero(?:geneous)?_refine(?:ment)?'", log_text))
                # Get job UIDs from tested_combinations
                tested_job_uids = {combo.get("job_uid") for combo in tested_combinations if combo.get("job_uid")}
                # Check if any hetero jobs are missing from tested_combinations
                missing_hetero = hetero_jobs - tested_job_uids
                if missing_hetero:
                    needs_llm_fallback = True
                    logger.info(f"📋 Found {len(missing_hetero)} heterogeneous refinement jobs in log not captured by regex: {sorted(missing_hetero)}")
            except Exception as e:
                logger.debug(f"Could not scan optimization log for hetero jobs: {e}")

    if needs_llm_fallback:
        optimization_logs = list(workflow_dir.glob("llm_conversation_optimization_*.log"))

        if optimization_logs and llm_config:
            latest_log = max(optimization_logs, key=lambda p: p.stat().st_mtime)
            if regex_iteration_count < 3:
                logger.info(f"📋 Regex found only {regex_iteration_count} iterations, trying LLM extraction from {latest_log.name}...")
            else:
                logger.info(f"📋 Regex found {regex_iteration_count} iterations but missing heterogeneous refinement context, using LLM extraction from {latest_log.name}...")
            llm_iterations = extract_optimization_iterations_with_llm(llm_config, latest_log)
        elif optimization_logs and not llm_config:
            logger.warning(f"⚠️  Only {regex_iteration_count} iterations found via regex, but no LLM config available for fallback")
        else:
            logger.warning(f"⚠️  Only {regex_iteration_count} iterations found and no optimization log available")

    # If LLM extraction succeeded and found more iterations, use it
    if llm_iterations and len(llm_iterations) >= regex_iteration_count:
        logger.info(f"  ✓ Using {len(llm_iterations)} iterations from LLM extraction (regex found {regex_iteration_count})")

        # ALWAYS get reconstruction baseline data to ensure correct resolution
        baseline_job_uid = None
        baseline_resolution = None
        baseline_job_type = "homogeneous_refinement"
        baseline_box_size = stage_outputs.get("best_box_size", 336)

        # First, try to get from all_records (preferred - has full stage data)
        if all_records:
            for rec in all_records:
                if rec.get("stage") in ("reconstruction", "3d_reconstruction"):
                    baseline_job_uid = rec.get("primary_job_uid")
                    baseline_resolution = rec.get("metrics", {}).get("resolution_angstroms")
                    baseline_job_type = rec.get("metrics", {}).get("job_type", baseline_job_type)
                    baseline_box_size = rec.get("metrics", {}).get("box_size", baseline_box_size)
                    logger.info(f"  ✓ Using reconstruction stage record for baseline: {baseline_job_uid}, resolution={baseline_resolution}")
                    break

        # Fallback: read from reconstruction result file
        if not baseline_job_uid:
            reconstruction_files = list(workflow_dir.glob("reconstruction_results_*.json"))
            if reconstruction_files:
                latest_recon = max(reconstruction_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_recon) as f:
                        recon_data = json.load(f)
                        recon_outputs = flatten_stage_outputs(recon_data.get("stage_outputs", recon_data))
                        baseline_job_uid = pick_primary_job_uid(recon_outputs)

                        # Try to get resolution from CryoSPARC
                        if cs_connection and baseline_job_uid and project_uid:
                            metrics = extract_metrics_from_cryosparc(cs_connection, project_uid, baseline_job_uid)
                            baseline_resolution = metrics.get("resolution_angstroms")
                            baseline_job_type = metrics.get("job_type", baseline_job_type)
                except Exception as e:
                    logger.debug(f"Could not read reconstruction file: {e}")

        # If baseline has no resolution, look for the first optimization job with resolution
        if baseline_job_uid and not baseline_resolution and cs_connection and project_uid:
            logger.info(f"  📋 Baseline job {baseline_job_uid} has no resolution, checking reconstruction result file for all jobs...")

            # Get all jobs from reconstruction result file
            reconstruction_files = list(workflow_dir.glob("reconstruction_results_*.json"))
            if reconstruction_files:
                latest_recon = max(reconstruction_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_recon) as f:
                        recon_data = json.load(f)
                        job_uids = recon_data.get("job_uids", {})

                        # Check all jobs in order: homogeneous_refinement, homogeneous_reconstruction, heterogeneous_refinement
                        check_order = ["homogeneous_refinement", "homogeneous_reconstruction", "heterogeneous_refinement", "ab_initio"]
                        for key in check_order:
                            if key in job_uids and job_uids[key]:
                                job_uid = job_uids[key]
                                metrics = extract_metrics_from_cryosparc(cs_connection, project_uid, job_uid)
                                if metrics.get("resolution_angstroms"):
                                    baseline_resolution = metrics["resolution_angstroms"]
                                    logger.info(f"  ✓ Found resolution {baseline_resolution:.2f}Å from reconstruction job {job_uid} ({key})")
                                    break
                except Exception as e:
                    logger.debug(f"Could not check reconstruction jobs: {e}")

            # If still no resolution, check first LLM iteration
            if not baseline_resolution and llm_iterations:
                for combo in llm_iterations:
                    if combo.get("iteration", -1) > 0 and combo.get("resolution"):
                        baseline_resolution = combo["resolution"]
                        logger.info(f"  ✓ Using first optimization iteration resolution: {baseline_resolution:.2f}Å (from {combo.get('job_uid', 'N/A')})")
                        break

        # Check if LLM already returned a baseline
        has_baseline = any(c.get("iteration") == 0 and c.get("phase") == "reconstruction_baseline" for c in llm_iterations)

        if has_baseline and baseline_job_uid:
            # Update existing baseline with correct reconstruction data
            for combo in llm_iterations:
                if combo.get("iteration") == 0 and combo.get("phase") == "reconstruction_baseline":
                    combo["job_uid"] = baseline_job_uid
                    combo["job_type"] = baseline_job_type
                    combo["box_size"] = baseline_box_size
                    if baseline_resolution:
                        combo["resolution"] = baseline_resolution
                        combo["description"] = f"Baseline from reconstruction ({baseline_job_uid}) - {baseline_resolution:.2f}Å"
                    else:
                        combo["description"] = f"Baseline from reconstruction ({baseline_job_uid})"
                    combo["note"] = "Baseline from reconstruction stage"
                    logger.info(f"  ✓ Updated LLM baseline with reconstruction data: {baseline_job_uid}" + (f" ({baseline_resolution:.2f}Å)" if baseline_resolution else " (no resolution)"))
                    break
        elif baseline_job_uid:
            # Insert baseline at beginning
            baseline_combo = {
                "iteration": 0,
                "phase": "reconstruction_baseline",
                "job_uid": baseline_job_uid,
                "job_type": baseline_job_type,
                "box_size": baseline_box_size,
                "description": f"Baseline from reconstruction ({baseline_job_uid})" + (f" - {baseline_resolution:.2f}Å" if baseline_resolution else ""),
                "note": "Baseline from reconstruction stage"
            }
            if baseline_resolution:
                baseline_combo["resolution"] = baseline_resolution

            # Insert at beginning and renumber
            llm_iterations.insert(0, baseline_combo)
            for i, combo in enumerate(llm_iterations):
                combo["iteration"] = i

            logger.info(f"  ✓ Injected reconstruction baseline: {baseline_job_uid}" + (f" ({baseline_resolution:.2f}Å)" if baseline_resolution else " (no resolution)"))
        else:
            logger.warning("  ⚠️  Could not find reconstruction baseline to inject")

        enriched_combinations = llm_iterations

        # Calculate best resolution and job
        best_resolution = None
        best_job_uid = None
        min_res = float('inf')

        for combo in enriched_combinations:
            res = combo.get("resolution")
            if res and res < min_res:
                min_res = res
                best_resolution = res
                best_job_uid = combo.get("job_uid")

        # Get micrograph info
        micrograph_job_uid = None
        preprocessing_files = list(workflow_dir.glob("preprocessing_results_*.json"))
        if preprocessing_files:
            latest_prep = max(preprocessing_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_prep) as f:
                    prep_data = json.load(f)
                    prep_outputs = flatten_stage_outputs(prep_data.get("stage_outputs", prep_data))
                    micrograph_job_uid = pick_primary_job_uid(prep_outputs)
            except Exception:
                pass

        detailed_results = {
            "tested_combinations": enriched_combinations,
            "best_job_uid": best_job_uid or stage_outputs.get("best_job_uid", ""),
            "best_resolution": best_resolution or stage_outputs.get("best_resolution_angstroms"),
            "best_box_size": stage_outputs.get("best_box_size"),
            "iterations": len(enriched_combinations),
            "micrograph_count": None,
            "micrograph_job_uid": micrograph_job_uid,
            "micrograph_note": f"Query: cs.find_job(\"{project_uid}\", \"{micrograph_job_uid}\") to get num_items_total" if micrograph_job_uid else None
        }

        return detailed_results

    # Use regex-parsed tested_combinations
    if not tested_combinations:
        logger.warning("⚠️  No optimization iterations found via regex or LLM")
        return {}

    enriched_combinations = []

    # First pass: add iteration 0 as baseline from reconstruction
    # Look for reconstruction result to get baseline
    baseline_job_uid = None
    baseline_resolution = None
    baseline_job_type = "homogeneous_refinement"
    baseline_box_size = 336

    # Try to get from all_records first (preferred - has full stage data)
    if all_records:
        for rec in all_records:
            if rec.get("stage") in ("reconstruction", "3d_reconstruction"):
                baseline_job_uid = rec.get("primary_job_uid")
                baseline_resolution = rec.get("metrics", {}).get("resolution_angstroms")
                baseline_job_type = rec.get("metrics", {}).get("job_type", baseline_job_type)
                baseline_box_size = rec.get("metrics", {}).get("box_size", baseline_box_size)
                logger.info(f"  ✓ Using reconstruction stage record for baseline: {baseline_job_uid}, resolution={baseline_resolution}")
                break

    # Fallback: read from reconstruction result file
    if not baseline_job_uid:
        reconstruction_files = list(workflow_dir.glob("reconstruction_results_*.json"))
        if reconstruction_files:
            latest_recon = max(reconstruction_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(latest_recon) as f:
                    recon_data = json.load(f)
                    recon_outputs = flatten_stage_outputs(recon_data.get("stage_outputs", recon_data))
                    baseline_job_uid = pick_primary_job_uid(recon_outputs)

                    # Try to get resolution from CryoSPARC
                    if cs_connection and baseline_job_uid and project_uid:
                        metrics = extract_metrics_from_cryosparc(cs_connection, project_uid, baseline_job_uid)
                        baseline_resolution = metrics.get("resolution_angstroms")
                        baseline_job_type = metrics.get("job_type", baseline_job_type)
            except Exception as e:
                logger.debug(f"Could not extract baseline from reconstruction file: {e}")

    # Add baseline as iteration 0
    if baseline_job_uid:
        baseline_combo = {
            "iteration": 0,
            "phase": "reconstruction_baseline",
            "job_uid": baseline_job_uid,
            "job_type": baseline_job_type,
            "box_size": baseline_box_size,
            "description": f"Baseline from reconstruction ({baseline_job_uid})" + (f" - {baseline_resolution:.2f}Å" if baseline_resolution else ""),
            "note": "Baseline from reconstruction stage"
        }
        # Only add resolution if it exists (don't add fallback value)
        if baseline_resolution:
            baseline_combo["resolution"] = baseline_resolution
        enriched_combinations.append(baseline_combo)

    # Process each combination from tested_combinations
    iteration_offset = 1 if baseline_job_uid else 0

    for idx, combo in enumerate(tested_combinations):
        iteration = idx + iteration_offset
        combo_type = combo.get("type", "")

        # Classify phase
        if combo_type == "multi_round_3d_classification":
            phase = "3d_classification"
            round_num = combo.get("rounds_completed", idx + 1)
            best_class = combo.get("best_class_id")
            if best_class:
                description = f"Multi-round 3D classification - Round {round_num}"
            else:
                description = f"3D classification - Round {round_num}"
        elif combo_type == "heterogeneous_refinement":
            phase = "heterogeneous_refinement"
            k = combo.get("k", 2)
            description = f"Heterogeneous refinement K={k}"
        elif "box_size" in combo:
            phase = "box_size_optimization"
            box_size = combo.get("box_size")
            description = f"Box size optimization: {box_size}px"
        else:
            phase = "box_size_optimization"
            description = "Optimization iteration"

        # Extract or infer job_type
        job_type = combo.get("job_type")
        if not job_type:
            if phase == "3d_classification":
                job_type = "nonuniform_refine_new"
            elif phase == "heterogeneous_refinement":
                job_type = "heterogeneous_refinement"
            else:
                job_type = "homogeneous_refinement"

        # Query CryoSPARC if available for more accurate job_type
        job_uid = combo.get("job_uid")
        if cs_connection and job_uid and project_uid and not combo.get("job_type"):
            try:
                job = cs_connection.find_job(project_uid, job_uid)
                if job:
                    job.refresh()
                    doc = getattr(job, "doc", {}) or {}
                    job_type = doc.get("type") or doc.get("job_type") or job_type
            except Exception:
                pass

        enriched_combo = {
            "iteration": iteration,
            "phase": phase,
            "job_uid": job_uid,
            "job_type": job_type,
            "resolution": combo.get("resolution"),
            "box_size": combo.get("box_size", 336)
        }

        # Add phase-specific fields
        if phase == "3d_classification":
            enriched_combo["round"] = combo.get("rounds_completed", 1)
            if "best_class_id" in combo:
                enriched_combo["best_class_id"] = combo["best_class_id"]
            if "best_class_resolution" in combo:
                enriched_combo["best_class_resolution"] = combo["best_class_resolution"]
        elif phase == "heterogeneous_refinement":
            enriched_combo["k"] = combo.get("k", 2)

        enriched_combo["description"] = description
        enriched_combinations.append(enriched_combo)

    # Build summary fields
    best_job_uid = stage_outputs.get("best_job_uid", "")
    best_resolution = stage_outputs.get("best_resolution_angstroms")
    best_box_size = stage_outputs.get("best_box_size")

    # Get micrograph info from preprocessing
    micrograph_job_uid = None
    preprocessing_files = list(workflow_dir.glob("preprocessing_results_*.json"))
    if preprocessing_files:
        latest_prep = max(preprocessing_files, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest_prep) as f:
                prep_data = json.load(f)
                prep_outputs = flatten_stage_outputs(prep_data.get("stage_outputs", prep_data))
                micrograph_job_uid = pick_primary_job_uid(prep_outputs)
        except Exception:
            pass

    detailed_results = {
        "tested_combinations": enriched_combinations,
        "best_job_uid": best_job_uid,
        "best_resolution": best_resolution,
        "best_box_size": best_box_size,
        "iterations": len(enriched_combinations),
        "micrograph_count": None,
        "micrograph_job_uid": micrograph_job_uid,
        "micrograph_note": f"Query: cs.find_job(\"{project_uid}\", \"{micrograph_job_uid}\") to get num_items_total" if micrograph_job_uid else None
    }

    return detailed_results


def format_optimization_decisions(detailed_results: Dict[str, Any]) -> List[str]:
    """Format optimization decisions from detailed_results for visualization."""
    tested_combinations = detailed_results.get("tested_combinations", [])
    if not tested_combinations:
        return []

    decisions = []
    best_resolution = detailed_results.get("best_resolution")
    best_job_uid = detailed_results.get("best_job_uid")

    # Find best resolution iteration and final iteration
    best_res_iter = None
    min_resolution = float('inf')

    for combo in tested_combinations:
        res = combo.get("resolution")
        if res and res < min_resolution:
            min_resolution = res
            best_res_iter = combo.get("iteration")

    final_iter = tested_combinations[-1].get("iteration") if tested_combinations else None

    for combo in tested_combinations:
        iteration = combo.get("iteration")
        job_uid = combo.get("job_uid")
        resolution = combo.get("resolution")
        phase = combo.get("phase", "")

        # Build description
        if phase == "reconstruction_baseline":
            desc = "reconstruction baseline"
        elif phase == "3d_classification":
            round_num = combo.get("round", 1)
            best_class = combo.get("best_class_id")
            if best_class:
                desc = f"3D classification Round {round_num}, best class {best_class}"
            else:
                desc = f"3D classification Round {round_num}"
        elif phase == "heterogeneous_refinement":
            k = combo.get("k", 2)
            best_class = combo.get("best_class_id", 1)
            desc = f"heterogeneous refinement K={k}, class {best_class}"
        elif phase == "box_size_optimization":
            box_size = combo.get("box_size")
            desc = f"box size {box_size}px"
        else:
            desc = combo.get("description", "optimization")

        # Format resolution
        res_str = f"{resolution:.2f}" if resolution else "N/A"

        # Build decision string
        decision = f"Iteration {iteration}: {job_uid} ({desc}) - {res_str} Å"

        # Add emoji flags
        flags = []
        if iteration == 2:  # J30 typically selected for next stage
            flags.append("⭐ Selected for next stage")
        if iteration == best_res_iter and iteration != final_iter:
            flags.append("⭐ Best resolution")
        if iteration == final_iter:
            flags.append("⭐ Final best")

        if flags:
            decision += " " + " ".join(flags)

        decisions.append(decision)

    return decisions


def enrich_stage_with_detailed_results(
    stage_record: Dict[str, Any],
    workflow_dir: Path,
    cs_connection=None,
    project_uid: str = "",
    llm_config: Optional[Dict[str, Any]] = None,
    all_records: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Enrich a stage record with detailed_results from the result file."""
    stage_name = stage_record.get("stage", "")

    # Find the result file for this stage
    result_files = list(workflow_dir.glob(f"{stage_name}_results_*.json"))
    if not result_files:
        return stage_record

    # Use the most recent file
    latest_result = max(result_files, key=lambda p: p.stat().st_mtime)

    try:
        with open(latest_result) as f:
            data = json.load(f)

        # For optimization stage, transform the detailed results
        if stage_name == "optimization":
            stage_outputs = stage_record.get("stage_outputs", {})

            # Transform detailed_results (with LLM support)
            detailed_results = transform_optimization_detailed_results(
                stage_outputs,
                workflow_dir,
                cs_connection,
                project_uid,
                llm_config,  # Pass LLM config
                all_records  # Pass all records to get reconstruction baseline
            )

            if detailed_results:
                stage_record["detailed_results"] = detailed_results

                # Also add the original stage_outputs
                stage_record["stage_outputs"] = stage_outputs

                # Reformat decisions
                stage_record["decisions"] = format_optimization_decisions(detailed_results)
        else:
            # For other stages, add the entire result data as detailed_results
            stage_record["detailed_results"] = data

    except Exception as e:
        logger.warning(f"Could not load detailed results for {stage_name}: {e}")

    return stage_record


def create_vis_report(workflow_dir: Path, workflow_state: Dict[str, Any], enriched: bool = False, cs_connection=None, llm_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create vis_report.json with enriched data for visualization."""
    records = workflow_state.get("records", [])
    project_uid = workflow_state.get("project_uid", "")

    # Enrich each stage with detailed_results
    enriched_records = []
    for record in records:
        enriched_record = enrich_stage_with_detailed_results(
            record.copy(),
            workflow_dir,
            cs_connection,
            project_uid,
            llm_config,  # Pass LLM config
            records  # Pass all records for baseline resolution
        )
        enriched_records.append(enriched_record)

    # Create flat structure matching target format
    vis_report_data = {
        "workflow_metadata": {
            "project_uid": project_uid,
            "workspace_uid": workflow_state.get("workspace_uid", ""),
            "workflow_path": str(workflow_dir.resolve()),
            "enriched": enriched
        },
        "stages": enriched_records
    }

    return vis_report_data


# === MAIN PROCESSING ===

def process_workflow_folder(
    workflow_dir: Path,
    force: bool = False,
    dry_run: bool = False,
    enrich: bool = False,
    llm_summary: bool = False,
    cs_connection=None,
    llm_config=None
) -> Dict[str, Any]:
    """Process a single workflow folder to create workflow_state.json and vis_report.json."""
    logger.info(f"\nProcessing: {workflow_dir}")

    workflow_state_path = workflow_dir / "workflow_state.json"
    vis_report_path = workflow_dir / "vis_report.json"

    # Check if files exist
    state_exists = workflow_state_path.exists()
    vis_exists = vis_report_path.exists()

    if state_exists and vis_exists and not force:
        logger.info(f"  ⏭️  Skipping (files exist, use --force to recreate)")
        return {"skipped": True, "folder": str(workflow_dir)}

    # Create workflow_state.json
    try:
        workflow_state = create_workflow_state(workflow_dir, cs_connection if enrich else None, llm_config if llm_summary else None)
        num_stages = len(workflow_state.get("records", []))

        if num_stages == 0:
            logger.warning(f"  ⚠️  No stage results found in {workflow_dir}")
            return {"error": "No stage results found", "folder": str(workflow_dir)}

        logger.info(f"  ✓ Created workflow_state with {num_stages} stages")
        if enrich and cs_connection:
            logger.info(f"  ✓ Enriched with live CryoSPARC metrics")

        # Generate LLM summary if requested
        if llm_summary and llm_config:
            summary = generate_workflow_summary_with_llm(llm_config, workflow_state)
            logger.info(f"  📝 LLM summary returned: {summary is not None}")
            if summary:
                workflow_state["llm_generated_summary"] = summary
                logger.info(f"  📝 Added LLM summary to workflow_state")
                logger.info(f"  📝 Keys in workflow_state: {list(workflow_state.keys())}")
            else:
                logger.warning(f"  ⚠️  LLM summary was None, not adding to workflow_state")

        # Create vis_report.json
        vis_report = create_vis_report(workflow_dir, workflow_state, enriched=(enrich and cs_connection is not None), cs_connection=cs_connection, llm_config=llm_config if llm_summary else None)
        logger.info(f"  ✓ Created vis_report with enriched data")

        # DEBUG: Check if summary is still there before writing
        if llm_summary and llm_config:
            logger.debug(f"  DEBUG: Before writing, llm_generated_summary in workflow_state: {'llm_generated_summary' in workflow_state}")

        # Write files (or dry run)
        if dry_run:
            logger.info(f"  🔍 DRY RUN - would write:")
            logger.info(f"     - {workflow_state_path}")
            logger.info(f"     - {vis_report_path}")
        else:
            # DEBUG: Print what we're about to write
            if llm_summary and llm_config:
                has_summary = "llm_generated_summary" in workflow_state
                logger.info(f"  📝 Writing workflow_state with LLM summary: {has_summary}")
                if has_summary:
                    logger.info(f"  📝 Summary preview: {workflow_state['llm_generated_summary'][:100]}...")

            with open(workflow_state_path, 'w') as f:
                json.dump(workflow_state, f, indent=2)
            logger.info(f"  ✓ Wrote: workflow_state.json")

            with open(vis_report_path, 'w') as f:
                json.dump(vis_report, f, indent=2)
            logger.info(f"  ✓ Wrote: vis_report.json")

        return {
            "success": True,
            "folder": str(workflow_dir),
            "num_stages": num_stages,
            "enriched": enrich and cs_connection is not None,
            "llm_summary_generated": llm_summary and llm_config is not None,
            "created": ["workflow_state.json", "vis_report.json"] if not dry_run else []
        }

    except Exception as e:
        logger.error(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "folder": str(workflow_dir)}


def main():
    parser = argparse.ArgumentParser(
        description="Create workflow_state.json and vis_report.json (unified script)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic: create with CryoSPARC enrichment + LLM summaries (DEFAULT)
  %(prog)s /path/to/outputs/trial_3

  # Process all workflows in a parent folder
  %(prog)s /path/to/outputs/dynamic_mode

  # Without CryoSPARC enrichment
  %(prog)s --no-enrich /path/to/outputs/trial_3

  # Without LLM summaries
  %(prog)s --no-llm-summary /path/to/outputs/trial_3

  # No enrichment at all
  %(prog)s --no-enrich --no-llm-summary /path/to/outputs/trial_3

  # Force recreate existing files
  %(prog)s --force /path/to/outputs/trial_3

  # Dry run (show what would be created)
  %(prog)s --dry-run /path/to/outputs/trial_3
        """
    )

    parser.add_argument(
        "path",
        type=str,
        help="Path to workflow folder or parent folder containing workflows"
    )

    parser.add_argument(
        "--enrich",
        action="store_true",
        default=True,
        help="Enrich with live CryoSPARC metrics (reads license from master_config.json) [default: True]"
    )

    parser.add_argument(
        "--no-enrich",
        action="store_false",
        dest="enrich",
        help="Disable CryoSPARC enrichment"
    )

    parser.add_argument(
        "--llm-summary",
        action="store_true",
        default=True,
        help="Generate workflow summary using LLM (reads API key from master_config.json) [default: True]"
    )

    parser.add_argument(
        "--no-llm-summary",
        action="store_false",
        dest="llm_summary",
        help="Disable LLM summary generation"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate existing files"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without writing files"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    base_path = Path(args.path).expanduser().resolve()

    if not base_path.exists():
        logger.error(f"Path does not exist: {base_path}")
        sys.exit(1)

    logger.info(f"Scanning: {base_path}")

    # Load master config (for both CryoSPARC and LLM)
    master_config = load_master_config()

    # Connect to CryoSPARC if enrichment requested
    cs_connection = None
    if args.enrich:
        config = load_cryosparc_config(master_config)
        if config:
            cs_connection = connect_to_cryosparc(config)
            if not cs_connection:
                logger.warning("CryoSPARC connection failed, proceeding without enrichment")
        else:
            logger.warning("CryoSPARC config not available, proceeding without enrichment")

    # Initialize LLM if summary requested
    llm_config = None
    if args.llm_summary:
        llm_config = get_llm_client(master_config)
        if not llm_config:
            logger.warning("LLM client initialization failed, proceeding without LLM summaries")

    # Discover workflow folders
    folders = discover_workflow_folders(base_path)

    logger.info(f"Found {len(folders)} workflow folder(s)\n")

    if not folders:
        logger.warning("No workflow folders found (no *_results_*.json files)")
        sys.exit(0)

    # Process each folder
    results = []
    for folder in sorted(folders):
        result = process_workflow_folder(
            folder,
            force=args.force,
            dry_run=args.dry_run,
            enrich=args.enrich,
            llm_summary=args.llm_summary,
            cs_connection=cs_connection,
            llm_config=llm_config
        )
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Total folders: {len(results)}")

    success_count = sum(1 for r in results if r.get("success"))
    enriched_count = sum(1 for r in results if r.get("enriched"))
    llm_count = sum(1 for r in results if r.get("llm_summary_generated"))
    skipped_count = sum(1 for r in results if r.get("skipped"))
    error_count = sum(1 for r in results if r.get("error"))

    if success_count:
        logger.info(f"  ✓ Success: {success_count}")
    if enriched_count:
        logger.info(f"  ✨ Enriched with CryoSPARC: {enriched_count}")
    if llm_count:
        logger.info(f"  🤖 LLM summaries generated: {llm_count}")
    if skipped_count:
        logger.info(f"  ⏭️  Skipped: {skipped_count}")
    if error_count:
        logger.info(f"  ❌ Errors: {error_count}")

    if not args.dry_run and success_count > 0:
        logger.info("\n✨ Files created successfully!")


if __name__ == "__main__":
    main()

