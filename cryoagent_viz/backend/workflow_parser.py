"""
Parse CryoAgent vis_report.json / workflow_state.json for workflow visualization.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _normalize_metadata(workflow_state: Dict[str, Any]) -> Dict[str, Any]:
    """Build metadata from either nested metadata or root project/workspace fields."""
    metadata = dict(workflow_state.get("metadata") or {})
    if not metadata.get("project_uid") and workflow_state.get("project_uid"):
        metadata["project_uid"] = workflow_state["project_uid"]
    if not metadata.get("workspace_uid") and workflow_state.get("workspace_uid"):
        metadata["workspace_uid"] = workflow_state["workspace_uid"]
    return metadata


def _attach_stage_results(workflow_dir: Path, records: List[Dict[str, Any]]) -> None:
    """Attach detailed_results from stage result JSON files when present."""
    for i, record in enumerate(records):
        if record.get("detailed_results"):
            continue
        stage_name = record.get("stage", "")

        stage_result_path = workflow_dir / f"stage_{i}" / "result.json"
        if stage_result_path.exists():
            with open(stage_result_path) as f:
                record["detailed_results"] = json.load(f)
            continue

        result_files = list(workflow_dir.glob(f"{stage_name}_results_*.json"))
        if result_files:
            latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
            with open(latest_result) as f:
                record["detailed_results"] = json.load(f)


def _parse_summary_artifacts(workflow_dir: Path) -> tuple:
    """Return (summary_text, timeline) from summary report files if present."""
    summary_json_files = list(workflow_dir.glob("workflow_summary_report_*.json"))
    summary_text_files = list(workflow_dir.glob("*_summary_report.txt")) + \
                         list(workflow_dir.glob("*_summary_report.md"))

    summary_text = None
    timeline = None

    if summary_json_files:
        latest_summary = max(summary_json_files, key=lambda p: p.stat().st_mtime)
        with open(latest_summary) as f:
            summary_data = json.load(f)
            if "timeline" in summary_data:
                timeline = summary_data["timeline"]
            summary_text = json.dumps(summary_data, indent=2)
    elif summary_text_files:
        latest_summary = max(summary_text_files, key=lambda p: p.stat().st_mtime)
        with open(latest_summary) as f:
            summary_text = f.read()
        timeline = parse_timeline_from_summary(summary_text)

    return summary_text, timeline


def parse_from_workflow_state(workflow_dir: Path) -> Dict[str, Any]:
    """
    Build visualization data from live agent-written workflow_state.json.

    Normalizes root project_uid/workspace_uid into metadata and exposes
    current_stage / run_status for Real-time monitoring.
    """
    workflow_state_path = workflow_dir / "workflow_state.json"
    if not workflow_state_path.exists():
        raise FileNotFoundError(
            f"No workflow_state.json found in {workflow_dir}."
        )

    with open(workflow_state_path) as f:
        workflow_state = json.load(f)

    # Live agent format uses "records"; enriched/post-processed may use "stages".
    records = workflow_state.get("records") or workflow_state.get("stages") or []
    if isinstance(records, list):
        records = [dict(r) for r in records]
    else:
        records = []

    _attach_stage_results(workflow_dir, records)
    summary_text, timeline = _parse_summary_artifacts(workflow_dir)
    metadata = _normalize_metadata(workflow_state)

    mtime = _file_mtime(workflow_state_path)
    last_updated = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    return {
        "workflow_state": {
            "metadata": metadata,
            "stages": records,
            "current_stage": workflow_state.get("current_stage"),
            "run_status": workflow_state.get("run_status") or "idle",
        },
        "summary": {
            "workflow_timeline": timeline,
            "summary_text": summary_text,
        } if (timeline or summary_text) else None,
        "output_dir": str(workflow_dir),
        "last_updated": last_updated,
        "source": "workflow_state",
    }


def _parse_from_vis_report(workflow_dir: Path, vis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize vis_report.json into the WorkflowData shape."""
    if "stages" in vis_data and "workflow_state" not in vis_data:
        ws = {
            "metadata": vis_data.get("workflow_metadata", {}),
            "stages": vis_data["stages"],
            "current_stage": vis_data.get("current_stage"),
            "run_status": vis_data.get("run_status") or "completed",
        }
        return {
            "workflow_state": ws,
            "summary": vis_data.get("summary", None),
            "output_dir": str(workflow_dir),
            "last_updated": vis_data.get("last_updated", ""),
            "source": "vis_report",
        }

    # Already wrapped — ensure progress fields exist if present at root
    result = dict(vis_data)
    ws = result.get("workflow_state") or {}
    if isinstance(ws, dict):
        ws = dict(ws)
        if "current_stage" not in ws:
            ws["current_stage"] = vis_data.get("current_stage")
        if "run_status" not in ws:
            ws["run_status"] = vis_data.get("run_status") or "completed"
        # Prefer nested metadata; fall back to workflow_metadata
        if not ws.get("metadata") and vis_data.get("workflow_metadata"):
            ws["metadata"] = vis_data["workflow_metadata"]
        result["workflow_state"] = ws
    result.setdefault("output_dir", str(workflow_dir))
    result["source"] = "vis_report"
    return result


def parse_workflow(workflow_dir: Path, prefer_live: bool = False) -> Dict[str, Any]:
    """
    Parse a workflow directory and return visualization data.

    By default prefers vis_report.json (enriched finished view) when present.
    When prefer_live=True (Real-time WebSocket), prefer workflow_state.json if it
    exists and is newer than vis_report.json (or if vis_report is absent).
    """
    workflow_dir = Path(workflow_dir)
    vis_report_path = workflow_dir / "vis_report.json"
    workflow_state_path = workflow_dir / "workflow_state.json"

    vis_mtime = _file_mtime(vis_report_path) if vis_report_path.exists() else 0.0
    state_mtime = _file_mtime(workflow_state_path) if workflow_state_path.exists() else 0.0

    use_live = False
    if prefer_live and workflow_state_path.exists():
        # Live path: use workflow_state when fresher or when no vis_report
        use_live = (not vis_report_path.exists()) or (state_mtime >= vis_mtime)
    elif not vis_report_path.exists() and workflow_state_path.exists():
        use_live = True

    if use_live:
        return parse_from_workflow_state(workflow_dir)

    if vis_report_path.exists():
        with open(vis_report_path) as f:
            vis_data = json.load(f)
        return _parse_from_vis_report(workflow_dir, vis_data)

    if workflow_state_path.exists():
        return parse_from_workflow_state(workflow_dir)

    raise FileNotFoundError(
        f"No vis_report.json or workflow_state.json found in {workflow_dir}. "
        f"Run scripts/create_workflow_visualization.py to generate these files."
    )


def parse_timeline_from_summary(summary_text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Parse the timeline section from summary report.

    Example format:
    Workflow Timeline:
    - preprocessing: 25.3 minutes
    - particle_picking: 10.2 minutes
    """
    timeline = []
    in_timeline = False

    for line in summary_text.split('\n'):
        if 'Workflow Timeline' in line or 'Timeline:' in line:
            in_timeline = True
            continue

        if in_timeline:
            # Stop at empty line or next section
            if not line.strip() or line.startswith('=='):
                break

            # Parse lines like "- preprocessing: 25.3 minutes"
            if line.strip().startswith('-'):
                parts = line.strip()[1:].split(':', 1)
                if len(parts) == 2:
                    stage_name = parts[0].strip()
                    duration_str = parts[1].strip()

                    # Extract minutes
                    try:
                        if 'minute' in duration_str:
                            minutes = float(duration_str.split()[0])
                            timeline.append({
                                "stage": stage_name,
                                "duration_minutes": minutes
                            })
                    except (ValueError, IndexError):
                        pass

    return timeline if timeline else None


def _read_run_status(workflow_dir: Path) -> Optional[str]:
    state_path = workflow_dir / "workflow_state.json"
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            data = json.load(f)
        return data.get("run_status")
    except (OSError, json.JSONDecodeError):
        return None


def _workflow_display_name(workflow_dir: Path) -> Dict[str, str]:
    """
    Build display name as parent/leaf (e.g. dynamic_mode/10181, relion_tutorial/trial_3).
    """
    project = workflow_dir.parent.name
    trial = workflow_dir.name
    item = {
        "name": f"{project}/{trial}",
        "path": str(workflow_dir),
        "project": project,
        "trial": trial,
    }
    run_status = _read_run_status(workflow_dir)
    if run_status:
        item["run_status"] = run_status
    return item


def discover_workflows(base_dir: Path) -> List[Dict[str, str]]:
    """
    Discover all workflow directories under base_dir.

    A valid workflow directory contains workflow_state.json.

    Display names use parent/leaf so workflows stay distinguishable across modes
    (e.g. dynamic_mode/10181 instead of 10181).

    Returns list of {name, path, project, trial, run_status?}
    """
    workflows = []
    seen_paths = set()

    # Include base_dir if it is itself a workflow, then also search nested dirs
    # (a mode folder may have both its own state and nested project/trial workflows).
    candidates = []
    if (base_dir / "workflow_state.json").exists():
        candidates.append(base_dir)

    for workflow_state_path in base_dir.rglob("workflow_state.json"):
        candidates.append(workflow_state_path.parent)

    for workflow_dir in candidates:
        resolved = str(workflow_dir.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)

        workflows.append(_workflow_display_name(workflow_dir))

    # Stable order by display name
    workflows.sort(key=lambda w: w["name"])
    return workflows


def watched_mtimes(workflow_dir: Path) -> Dict[str, float]:
    """Return mtimes of files that should trigger a live refresh."""
    workflow_dir = Path(workflow_dir)
    mtimes: Dict[str, float] = {}
    for name in ("workflow_state.json", "vis_report.json"):
        path = workflow_dir / name
        if path.exists():
            mtimes[name] = _file_mtime(path)
    for path in workflow_dir.glob("*_results_*.json"):
        mtimes[path.name] = _file_mtime(path)
    return mtimes
