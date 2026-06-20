#!/usr/bin/env python3
"""
Read CryoAgent workflow performance from the blackboard and/or live CryoSPARC jobs.

Uses the same metrics pipeline as WorkflowState (describe_job_results) and calls
consult_cryosparc_guide (same tool the improvement agent uses) to ground
diagnosis in the official CryoSPARC guide.

By default, after printing the blackboard the script auto-calls
consult_cryosparc_guide with a context-aware question derived from your metrics.

Examples:
  python scripts/read_workflow_performance.py
  python scripts/read_workflow_performance.py --no-guide
  python scripts/read_workflow_performance.py --guide-demo
  python scripts/read_workflow_performance.py --state outputs/workflow_state.json
  python scripts/read_workflow_performance.py --job-uid J443
  python scripts/read_workflow_performance.py --refresh
  python scripts/read_workflow_performance.py --guide-slug tutorial-orientation-diagnostics
  python scripts/read_workflow_performance.py --guide-question "how to interpret cFAR"
  python scripts/read_workflow_performance.py --verbose-guide
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cryoagent.config.config_loader import ConfigLoader
from cryoagent.core.workflow_state import StageRecord, WorkflowState
from cryoagent.tools.cryosparc_guide_tools import consult_cryosparc_guide
from cryoagent.tools.cryosparc_tools import CryoSPARCTools

DEFAULT_GUIDE_QUESTION = (
    "How should I interpret resolution, FSC, and cFAR / orientation diagnostics "
    "when assessing reconstruction quality?"
)

DEMO_GUIDE_SLUG = "tutorial-orientation-diagnostics"


def _load_config(config_path: str):
    loader = ConfigLoader(config_path, session_config_path="configs/session.json")
    return loader.load_config()


def _state_from_dict(data: Dict[str, Any], outputs_dir: str) -> WorkflowState:
    state = WorkflowState(
        outputs_dir=outputs_dir,
        project_uid=data.get("project_uid"),
        workspace_uid=data.get("workspace_uid"),
    )
    state.records = [StageRecord.from_dict(r) for r in data.get("records", [])]
    return state


def _load_state(state_path: Optional[Path], outputs_dir: str) -> WorkflowState:
    if state_path is not None:
        if not state_path.is_file():
            raise FileNotFoundError(f"Workflow state file not found: {state_path}")
        with open(state_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return _state_from_dict(data, str(state_path.parent))

    loaded = WorkflowState.load(outputs_dir=outputs_dir)
    if loaded is None:
        raise FileNotFoundError(
            f"No workflow_state.json found under {outputs_dir!r}. "
            "Run a workflow first or pass --state."
        )
    return loaded


def format_job_results(res: Dict[str, Any]) -> str:
    """Human-readable summary matching the agent describe_job_results tool."""
    if not res.get("success"):
        return f"❌ {res.get('error', 'Unknown error')}"

    lines = [
        f"📊 Results for {res['job_uid']} ({res.get('job_type')}, status={res.get('status')}):"
    ]
    if res.get("resolution_angstroms") is not None:
        lines.append(f"  - resolution: {res['resolution_angstroms']:.2f} Å (lower is better)")
    if res.get("box_size") is not None:
        lines.append(f"  - box size: {res['box_size']} px")
    if res.get("symmetry"):
        lines.append(f"  - symmetry: {res['symmetry']}")
    if res.get("num_particles") is not None:
        lines.append(f"  - particles: {res['num_particles']}")
    if res.get("cfar") is not None:
        lines.append(f"  - cFAR: {res['cfar']:.3f} ({res.get('cfar_label')}) (higher is better)")
    elif res.get("cfar_note"):
        lines.append(f"  - cFAR: not computed — {res['cfar_note']}")
    if res.get("classes"):
        lines.append(
            f"  - {res.get('num_classes')} classes (per-class resolution / particle share / cFAR):"
        )
        for cls in res["classes"]:
            frac = cls.get("particle_fraction")
            frac_s = f"{frac * 100:.1f}%" if isinstance(frac, (int, float)) else "?"
            rr = cls.get("resolution_angstroms")
            rr_s = f"{rr:.2f} Å" if isinstance(rr, (int, float)) else "?"
            cf = cls.get("cfar")
            cf_s = (
                f", cFAR {cf:.3f} ({cls.get('cfar_label')})"
                if isinstance(cf, (int, float))
                else ""
            )
            lines.append(
                f"      class {cls.get('class_id')}: {rr_s}, "
                f"{cls.get('num_particles')} particles ({frac_s}){cf_s}"
            )
        if res.get("cfar_note"):
            lines.append(f"  - note: {res['cfar_note']}")
    return "\n".join(lines)


def format_guide_response(res: Dict[str, Any]) -> str:
    """Match ImprovementAgent / BaseReActAgent._consult_cryosparc_guide_tool output."""
    if res.get("tutorials"):
        lines = [f"📚 CryoSPARC tutorial library ({res.get('message')}):"]
        for entry in res["tutorials"]:
            lines.append(f"- [{entry['slug']}] {entry['title']} — {entry['when']}")
        return "\n".join(lines)

    if not res.get("success"):
        return f"ℹ️ {res.get('message')}"

    lines = [f"📖 CryoSPARC guide ({res.get('message')}):"]
    for page in res.get("pages", []):
        lines.append(f"\n— {page['url']}\n{page['excerpt']}")
    related = res.get("related_tutorials") or []
    if related:
        lines.append("\nRelated tutorials/case-studies (fetch with slug=<slug>):")
        for entry in related:
            lines.append(f"- [{entry['slug']}] {entry['title']} — {entry['when']}")
    return "\n".join(lines)


def _effective_best_result(state: WorkflowState) -> Optional[Dict[str, Any]]:
    """Best resolution from blackboard metrics, with polish stage_outputs fallback."""
    best = state.best_resolution()
    polish = state.get("polish")
    if polish and isinstance(polish.stage_outputs, dict):
        final_res = polish.stage_outputs.get("final_resolution")
        best_job = polish.stage_outputs.get("best_job_uid")
        if isinstance(final_res, (int, float)):
            if best is None or final_res < best["resolution_angstroms"]:
                return {
                    "stage": "polish",
                    "job_uid": best_job,
                    "resolution_angstroms": float(final_res),
                    "cfar": polish.metrics.get("cfar"),
                }
    return best


def build_guide_question_from_state(state: WorkflowState) -> str:
    """Build the kind of question the improvement agent asks in STEP 1."""
    best = _effective_best_result(state)
    parts: List[str] = []

    if best:
        res = best["resolution_angstroms"]
        job = best.get("job_uid") or "unknown"
        stage = best.get("stage", "unknown")
        cfar = best.get("cfar")
        parts.append(
            f"Current best reconstruction is {res:.2f} Å from {stage} job {job}."
        )
        if isinstance(cfar, (int, float)):
            if cfar < 0.1:
                parts.append(f"cFAR is {cfar:.3f} (very poor — severe preferred orientation).")
            elif cfar < 0.15:
                parts.append(f"cFAR is {cfar:.3f} (poor orientation sampling).")
            elif cfar < 0.5:
                parts.append(f"cFAR is {cfar:.3f} (acceptable but not ideal).")
            else:
                parts.append(f"cFAR is {cfar:.3f} (good orientation sampling).")
        else:
            parts.append("cFAR has not been computed yet.")
    else:
        parts.append("No refinement resolution recorded on the blackboard yet.")

    for record in state.records:
        cfar = record.metrics.get("cfar")
        if isinstance(cfar, (int, float)) and cfar < 0.15:
            parts.append(
                f"Stage {record.stage} reported low cFAR ({cfar:.3f}) on {record.primary_job_uid}."
            )
            break

    opt = state.get("optimization")
    if opt and opt.stage_outputs.get("tested_combinations"):
        parts.append(
            "Box-size optimization already tested several extraction sizes."
        )

    parts.append(
        "What are the most likely limiting factors and which CryoSPARC workflows "
        "or jobs should I try next to improve resolution or cFAR?"
    )
    return " ".join(parts)


def call_consult_cryosparc_guide(
    *,
    question: str = "",
    slug: Optional[str] = None,
    list_tutorials: bool = False,
    max_pages: int = 2,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Call consult_cryosparc_guide and optionally print request/response details."""
    request = {
        "question": question,
        "slug": slug,
        "list_tutorials": list_tutorials,
        "max_pages": max_pages,
    }
    if verbose:
        print("\n🔧 consult_cryosparc_guide request")
        print("-" * 50)
        print(json.dumps(request, indent=2))

    res = consult_cryosparc_guide(
        question,
        max_pages=max_pages,
        slug=slug,
        list_tutorials=list_tutorials,
    )

    if verbose:
        summary = {
            "success": res.get("success"),
            "message": res.get("message"),
            "page_count": len(res.get("pages") or []),
            "tutorial_count": len(res.get("tutorials") or []),
            "related_tutorial_count": len(res.get("related_tutorials") or []),
        }
        if res.get("pages"):
            summary["page_urls"] = [p.get("url") for p in res["pages"]]
        if res.get("related_tutorials"):
            summary["related_slugs"] = [t.get("slug") for t in res["related_tutorials"]]
        print("\n🔧 consult_cryosparc_guide response (summary)")
        print("-" * 50)
        print(json.dumps(summary, indent=2))

    return res


def print_guide_consult(
    *,
    label: str,
    question: str = "",
    slug: Optional[str] = None,
    list_tutorials: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Print one consult_cryosparc_guide call (improvement-agent-style)."""
    print(f"\n📖 CryoSPARC guide — {label}")
    print("=" * 50)
    if list_tutorials:
        print("Mode: list_tutorials=true (browse catalog)")
    elif slug:
        print(f"Mode: slug={slug!r}" + (f", question={question!r}" if question else ""))
    else:
        print(f"Mode: auto-match question={question!r}")

    res = call_consult_cryosparc_guide(
        question=question,
        slug=slug,
        list_tutorials=list_tutorials,
        verbose=verbose,
    )
    print()
    print(format_guide_response(res))
    return res


def run_guide_demo(state: Optional[WorkflowState], verbose: bool = False) -> None:
    """Walk through all three consult_cryosparc_guide modes the improvement agent uses."""
    print("\n🧪 consult_cryosparc_guide — demo (improvement-agent modes)")
    print("=" * 50)

    print_guide_consult(
        label="Mode 1 — browse tutorial library",
        list_tutorials=True,
        verbose=verbose,
    )

    question = (
        build_guide_question_from_state(state)
        if state is not None
        else "preferred orientation low cFAR how to improve resolution"
    )
    auto_res = print_guide_consult(
        label="Mode 2 — auto-match from workflow question",
        question=question,
        verbose=verbose,
    )

    slug = DEMO_GUIDE_SLUG
    related = auto_res.get("related_tutorials") or []
    if related:
        slug = related[0]["slug"]
        print(f"\n(hint: using first related slug from Mode 2: {slug!r})")

    print_guide_consult(
        label="Mode 3 — fetch specific tutorial by slug",
        slug=slug,
        question=question,
        verbose=verbose,
    )


def collect_live_metrics(
    state: WorkflowState,
    cryosparc_tools: CryoSPARCTools,
    job_uids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Refresh describe_job_results for stage primary jobs or explicit UIDs."""
    project_uid = state.project_uid
    targets: Dict[str, str] = {}

    if job_uids:
        for uid in job_uids:
            targets[uid] = uid
    else:
        for record in state.records:
            if record.primary_job_uid:
                targets[record.primary_job_uid] = record.stage

    results: Dict[str, Any] = {"project_uid": project_uid, "jobs": {}}
    for uid, label in targets.items():
        res = cryosparc_tools.describe_job_results(uid, project_uid=project_uid)
        results["jobs"][uid] = {"stage": label if label != uid else None, **res}
    return results


def build_report(
    state: WorkflowState,
    *,
    live: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "project_uid": state.project_uid,
        "workspace_uid": state.workspace_uid,
        "best_result": state.best_resolution(),
        "blackboard_summary": state.summary_for_planner(),
        "records": [record.to_dict() for record in state.records],
        "live_metrics": live,
    }


def print_blackboard_summary(state: WorkflowState) -> None:
    print("📋 Workflow performance (blackboard)")
    print("=" * 50)
    if state.project_uid or state.workspace_uid:
        print(f"Project: {state.project_uid or '?'}  Workspace: {state.workspace_uid or '?'}")
    print()
    print(state.summary_for_planner())
    best = _effective_best_result(state)
    if best:
        cfar = best.get("cfar")
        cfar_s = f", cFAR={cfar:.3f}" if isinstance(cfar, (int, float)) else ""
        job_s = best.get("job_uid") or "?"
        print()
        print(
            f"🏆 Best resolution: {best['resolution_angstroms']:.2f} Å "
            f"({best['stage']}, {job_s}{cfar_s})"
        )


def print_live_metrics(live: Dict[str, Any]) -> None:
    print("\n🔄 Live CryoSPARC metrics")
    print("=" * 50)
    for uid, payload in live.get("jobs", {}).items():
        stage = payload.get("stage")
        header = f"{uid}" + (f" [{stage}]" if stage else "")
        print(f"\n{header}")
        print(format_job_results(payload))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read CryoAgent workflow performance metrics and optional CryoSPARC guide context.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        help="Path to workflow_state.json (default: outputs/workflow_state.json).",
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Outputs directory when --state is not set (default: outputs).",
    )
    parser.add_argument(
        "--config",
        default="configs/master_config.json",
        help="Master config path for CryoSPARC access (default: configs/master_config.json).",
    )
    parser.add_argument(
        "--job-uid",
        action="append",
        dest="job_uids",
        metavar="UID",
        help="Describe one or more CryoSPARC jobs live (repeatable).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch describe_job_results for every stage primary job via CryoSPARC.",
    )
    parser.add_argument(
        "--no-guide",
        action="store_true",
        help="Skip the default consult_cryosparc_guide call after the blackboard.",
    )
    parser.add_argument(
        "--guide-demo",
        action="store_true",
        help="Demo all three consult_cryosparc_guide modes (list / auto / slug).",
    )
    parser.add_argument(
        "--guide-question",
        default="",
        help="Custom question for consult_cryosparc_guide (overrides auto question).",
    )
    parser.add_argument(
        "--guide-slug",
        help="Fetch a specific CryoSPARC guide tutorial/page by slug.",
    )
    parser.add_argument(
        "--list-guide-tutorials",
        action="store_true",
        help="List curated CryoSPARC guide tutorials/case studies.",
    )
    parser.add_argument(
        "--verbose-guide",
        action="store_true",
        help="Print consult_cryosparc_guide request/response summaries (JSON).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    need_cryosparc = bool(args.job_uids or args.refresh)
    cryosparc_tools = None
    config = None

    if need_cryosparc:
        try:
            config = _load_config(args.config)
            cryosparc_tools = CryoSPARCTools(config.cryosparc)
        except Exception as exc:
            print(f"❌ Could not connect to CryoSPARC: {exc}")
            print("   Omit --refresh / --job-uid to read the saved blackboard only.")
            return 1

    state: Optional[WorkflowState] = None
    if not args.job_uids or args.refresh or not args.json:
        try:
            state = _load_state(args.state, args.outputs_dir)
        except FileNotFoundError as exc:
            if args.job_uids and not args.refresh:
                state = None
            else:
                print(f"❌ {exc}")
                return 1

    live: Optional[Dict[str, Any]] = None
    if args.job_uids and cryosparc_tools:
        project_uid = (
            (state.project_uid if state else None)
            or getattr(getattr(config, "workflow", None), "project_uid", None)
        )
        if not project_uid:
            print("❌ project_uid is required for --job-uid (set it in workflow_state or config).")
            return 1
        if state is None:
            state = WorkflowState(outputs_dir=args.outputs_dir, project_uid=project_uid)
        live = collect_live_metrics(state, cryosparc_tools, job_uids=args.job_uids)
    elif args.refresh and cryosparc_tools and state is not None:
        live = collect_live_metrics(state, cryosparc_tools)

    if args.guide_demo:
        if state is None:
            try:
                state = _load_state(args.state, args.outputs_dir)
            except FileNotFoundError:
                state = None
        run_guide_demo(state, verbose=args.verbose_guide)
        return 0

    guide_question = args.guide_question
    if not guide_question and state is not None and not args.list_guide_tutorials and not args.guide_slug:
        guide_question = build_guide_question_from_state(state)
    elif not guide_question and (args.list_guide_tutorials or args.guide_slug):
        guide_question = ""
    elif not guide_question:
        guide_question = DEFAULT_GUIDE_QUESTION

    run_default_guide = not args.no_guide and not args.guide_slug and not args.list_guide_tutorials
    run_explicit_guide = bool(args.guide_slug or args.list_guide_tutorials or args.guide_question)

    if args.json:
        payload: Dict[str, Any] = {}
        if state is not None:
            payload = build_report(state, live=live)
            payload["guide_question"] = guide_question
        elif live is not None:
            payload = {"live_metrics": live}
        if run_default_guide or run_explicit_guide:
            payload["guide"] = call_consult_cryosparc_guide(
                question=guide_question,
                slug=args.guide_slug,
                list_tutorials=args.list_guide_tutorials,
                verbose=False,
            )
        print(json.dumps(payload, indent=2, default=str))
        return 0

    if state is not None:
        print_blackboard_summary(state)
    if live is not None:
        print_live_metrics(live)
    elif args.job_uids and cryosparc_tools is None:
        print("❌ --job-uid requires a working CryoSPARC connection.")
        return 1

    if run_explicit_guide:
        print_guide_consult(
            label="explicit consult_cryosparc_guide",
            question=guide_question,
            slug=args.guide_slug,
            list_tutorials=args.list_guide_tutorials,
            verbose=args.verbose_guide,
        )
    elif run_default_guide and state is not None:
        print_guide_consult(
            label="improvement-style consult (auto question from blackboard)",
            question=guide_question,
            verbose=args.verbose_guide,
        )

    if state is None and live is None and not run_explicit_guide:
        print("❌ Nothing to show. Pass --state, --job-uid, --refresh, or --guide-demo.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
