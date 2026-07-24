"""
Parse CryoAgent vis_report.json for workflow visualization.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_workflow(workflow_dir: Path) -> Dict[str, Any]:
    """
    Parse a workflow directory and return visualization data.

    Loads vis_report.json which contains all enriched data needed for visualization.
    Falls back to building from workflow_state.json if vis_report.json doesn't exist.
    """
    # Check if vis_report.json exists (preferred)
    vis_report_path = workflow_dir / "vis_report.json"
    if vis_report_path.exists():
        with open(vis_report_path) as f:
            vis_data = json.load(f)

            # If vis_report has 'stages' at root level, wrap it in workflow_state
            if 'stages' in vis_data and 'workflow_state' not in vis_data:
                return {
                    "workflow_state": {
                        "metadata": vis_data.get("workflow_metadata", {}),
                        "stages": vis_data["stages"]
                    },
                    "summary": vis_data.get("summary", None),
                    "output_dir": str(workflow_dir),
                    "last_updated": vis_data.get("last_updated", "")
                }

            # Otherwise return as-is
            return vis_data

    # Fallback: build from workflow_state.json if it exists
    workflow_state_path = workflow_dir / "workflow_state.json"
    if not workflow_state_path.exists():
        raise FileNotFoundError(
            f"No vis_report.json or workflow_state.json found in {workflow_dir}. "
            f"Run scripts/create_workflow_state.py to generate these files."
        )

    with open(workflow_state_path) as f:
        workflow_state = json.load(f)

    # Build minimal vis_report from workflow_state
    records = workflow_state.get("records", [])
    for i, record in enumerate(records):
        stage_name = record.get("stage", "")

        # Look for stage_N/result.json (old structure)
        stage_result_path = workflow_dir / f"stage_{i}" / "result.json"
        if stage_result_path.exists():
            with open(stage_result_path) as f:
                stage_result = json.load(f)
                record["detailed_results"] = stage_result
        else:
            # Look for {stage_name}_results_*.json in root directory (new structure)
            result_files = list(workflow_dir.glob(f"{stage_name}_results_*.json"))
            if result_files:
                latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
                with open(latest_result) as f:
                    stage_result = json.load(f)
                    record["detailed_results"] = stage_result

    # Parse summary report for timeline (try JSON first, then text)
    summary_json_files = list(workflow_dir.glob("workflow_summary_report_*.json"))
    summary_text_files = list(workflow_dir.glob("*_summary_report.txt")) + \
                         list(workflow_dir.glob("*_summary_report.md"))

    summary_text = None
    timeline = None

    if summary_json_files:
        latest_summary = max(summary_json_files, key=lambda p: p.stat().st_mtime)
        with open(latest_summary) as f:
            summary_data = json.load(f)
            # Extract timeline if present in JSON
            if "timeline" in summary_data:
                timeline = summary_data["timeline"]
            summary_text = json.dumps(summary_data, indent=2)
    elif summary_text_files:
        latest_summary = max(summary_text_files, key=lambda p: p.stat().st_mtime)
        with open(latest_summary) as f:
            summary_text = f.read()
        timeline = parse_timeline_from_summary(summary_text)

    return {
        "workflow_state": {
            "metadata": workflow_state.get("metadata", {}),
            "stages": records
        },
        "summary": {
            "workflow_timeline": timeline,
            "summary_text": summary_text
        }
    }


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


def _workflow_display_name(workflow_dir: Path) -> Dict[str, str]:
    """
    Build display name as parent/leaf (e.g. dynamic_mode/10181, relion_tutorial/trial_3).
    """
    project = workflow_dir.parent.name
    trial = workflow_dir.name
    return {
        "name": f"{project}/{trial}",
        "path": str(workflow_dir),
        "project": project,
        "trial": trial,
    }


def discover_workflows(base_dir: Path) -> List[Dict[str, str]]:
    """
    Discover all workflow directories under base_dir.

    A valid workflow directory contains workflow_state.json.

    Display names use parent/leaf so workflows stay distinguishable across modes
    (e.g. dynamic_mode/10181 instead of 10181).

    Returns list of {name, path, project, trial}
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
