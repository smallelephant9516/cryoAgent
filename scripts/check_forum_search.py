#!/usr/bin/env python3
"""
Test CryoSPARC Discuss search used by the reconstruction agent.

Examples:
  python check_forum_search.py
  python check_forum_search.py --query "ab initio failed to converge"
  python check_forum_search.py --job-uid J200
  python check_forum_search.py --log-file /path/to/job.log
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cryoagent.config.config_loader import ConfigLoader
from cryoagent.core.base_react_agent import BaseReActAgent
from cryoagent.tools.cryosparc_forum_tools import (
    derive_forum_conclusions,
    extract_search_queries_from_log,
    format_forum_conclusions,
    format_forum_search_response,
    search_cryosparc_forum,
    search_cryosparc_forum_multi,
)
from cryoagent.tools.cryosparc_tools import CryoSPARCTools


DEMO_QUERIES = [
    "CUDA_ERROR_OUT_OF_MEMORY",
    "ab initio failed to converge",
    "homogeneous refinement CTF",
]


class _MinimalConfig:
    class _Workflow:
        project_uid = "P1"
        workspace_uid = "W1"

    workflow = _Workflow()


class _ForumSearchProbe(BaseReActAgent):
    """Minimal agent shell to exercise _search_cryosparc_forum_tool."""

    def _create_tools(self):
        return []

    def _get_react_system_prompt(self) -> str:
        return "forum search probe"


def _load_config(config_path: str):
    loader = ConfigLoader(config_path, session_config_path="configs/session.json")
    return loader.load_config()


def _print_search_results(
    payload: dict,
    *,
    show_conclusions: bool = True,
    log_analysis: dict | None = None,
) -> None:
    print(format_forum_search_response(payload))
    if show_conclusions:
        conclusions = derive_forum_conclusions(payload, log_analysis=log_analysis)
        print(format_forum_conclusions(conclusions))


def test_direct_search(query: str, max_results: int, show_conclusions: bool = True) -> bool:
    print(f"\n🔍 Direct search: {query!r}")
    print("-" * 50)
    start = time.time()
    payload = search_cryosparc_forum(query, max_results=max_results)
    elapsed = time.time() - start

    if not payload.get("success"):
        print(f"❌ Search failed: {payload.get('error', 'unknown error')}")
        return False

    print(f"✅ {payload.get('result_count', 0)} result(s) in {elapsed:.2f}s")
    _print_search_results(payload, show_conclusions=show_conclusions)
    return payload.get("result_count", 0) > 0


def test_log_extraction(
    log_path: Path,
    max_results: int,
    show_conclusions: bool = True,
) -> bool:
    print(f"\n📄 Log extraction + search: {log_path}")
    print("-" * 50)
    if not log_path.is_file():
        print(f"❌ Log file not found: {log_path}")
        return False

    log_content = log_path.read_text(encoding="utf-8", errors="ignore")
    queries = extract_search_queries_from_log(log_content)
    if not queries:
        print("❌ No search queries could be extracted from the log.")
        return False

    print(f"✅ Extracted queries: {queries}")
    payload = search_cryosparc_forum_multi(queries, max_total_results=max_results)
    log_analysis = {
        "has_errors": True,
        "critical_errors": log_content.splitlines()[-20:],
    }
    _print_search_results(
        payload,
        show_conclusions=show_conclusions,
        log_analysis=log_analysis,
    )
    return payload.get("success", False)


def test_agent_tool_with_query(
    query: str,
    max_results: int,
    show_conclusions: bool = True,
) -> bool:
    print(f"\n🤖 Agent tool (query path): {query!r}")
    print("-" * 50)
    probe = _ForumSearchProbe.__new__(_ForumSearchProbe)
    probe.cryosparc_tools = None
    probe.config = _MinimalConfig()
    probe.tool_execution_log = []
    probe.enable_conversation_logging = False
    probe.realtime_logger = None

    import json

    output = probe._search_cryosparc_forum_tool(
        json.dumps({"query": query, "max_results": max_results})
    )
    if output.startswith("❌"):
        print(output)
        return False

    print(output)
    if show_conclusions:
        logged_entry = next(
            (e for e in probe.tool_execution_log if e.get("tool") == "search_cryosparc_forum"),
            None,
        )
        if logged_entry and logged_entry.get("result"):
            conclusions = derive_forum_conclusions(logged_entry["result"])
            print(format_forum_conclusions(conclusions))
    logged = any(e.get("tool") == "search_cryosparc_forum" for e in probe.tool_execution_log)
    print(f"\n✅ Tool execution logged: {logged}")
    return "discuss.cryosparc.com/t/" in output


def test_agent_tool_with_job(
    job_uid: str,
    config_path: str,
    max_results: int,
    show_conclusions: bool = True,
) -> bool:
    print(f"\n🤖 Agent tool (job_uid path): {job_uid}")
    print("-" * 50)

    try:
        config = _load_config(config_path)
        cryosparc_tools = CryoSPARCTools(config.cryosparc)
    except Exception as exc:
        print(f"❌ Could not connect to CryoSPARC: {exc}")
        print("   Use --query or --log-file to test forum search without CryoSPARC.")
        return False

    probe = _ForumSearchProbe.__new__(_ForumSearchProbe)
    probe.cryosparc_tools = cryosparc_tools
    probe.config = config
    probe.tool_execution_log = []
    probe.enable_conversation_logging = False
    probe.realtime_logger = None

    import json

    output = probe._search_cryosparc_forum_tool(
        json.dumps(
            {
                "job_uid": job_uid,
                "max_results": max_results,
                "project_uid": config.workflow.project_uid,
                "workspace_uid": config.workflow.workspace_uid,
            }
        )
    )
    if output.startswith("❌"):
        print(output)
        return False

    print(output)
    if show_conclusions:
        logged_entry = next(
            (e for e in probe.tool_execution_log if e.get("tool") == "search_cryosparc_forum"),
            None,
        )
        if logged_entry and logged_entry.get("result"):
            conclusions = derive_forum_conclusions(logged_entry["result"])
            print(format_forum_conclusions(conclusions))
    return "discuss.cryosparc.com/t/" in output


def run_demo(max_results: int, show_conclusions: bool = True) -> int:
    print("🧪 CryoSPARC Forum Search — demo mode")
    print("=" * 50)
    passed = 0
    for query in DEMO_QUERIES:
        if test_direct_search(query, max_results, show_conclusions=show_conclusions):
            passed += 1
    print("\n" + "=" * 50)
    print(f"Demo summary: {passed}/{len(DEMO_QUERIES)} queries returned results")
    return 0 if passed == len(DEMO_QUERIES) else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test search_cryosparc_forum (CryoSPARC Discuss integration).",
    )
    parser.add_argument(
        "--query",
        "-q",
        help="Search the forum with this query string.",
    )
    parser.add_argument(
        "--job-uid",
        help="Read job log from CryoSPARC and search using extracted error terms.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Extract search terms from a local job.log file, then search the forum.",
    )
    parser.add_argument(
        "--config",
        default="configs/master_config.json",
        help="Master config path (for --job-uid). Default: configs/master_config.json",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum forum topics to return (default: 5).",
    )
    parser.add_argument(
        "--agent-tool",
        action="store_true",
        help="Also run through BaseReActAgent._search_cryosparc_forum_tool.",
    )
    parser.add_argument(
        "--no-conclusions",
        action="store_true",
        help="Skip the heuristic conclusions section after search results.",
    )
    args = parser.parse_args()
    show_conclusions = not args.no_conclusions

    if args.job_uid:
        ok = test_agent_tool_with_job(
            args.job_uid, args.config, args.max_results, show_conclusions=show_conclusions
        )
        return 0 if ok else 1

    if args.log_file:
        ok = test_log_extraction(
            args.log_file, args.max_results, show_conclusions=show_conclusions
        )
        return 0 if ok else 1

    if args.query:
        ok = test_direct_search(
            args.query, args.max_results, show_conclusions=show_conclusions
        )
        if args.agent_tool:
            ok = test_agent_tool_with_query(
                args.query, args.max_results, show_conclusions=show_conclusions
            ) and ok
        return 0 if ok else 1

    return run_demo(args.max_results, show_conclusions=show_conclusions)


if __name__ == "__main__":
    sys.exit(main())
