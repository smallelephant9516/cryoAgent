#!/usr/bin/env python3
"""Replay the improvement scientific auditor on saved conversation logs.

By default this is CPU-only: it uses parameters and measurements already saved in
the log and workflow_state.json, performs no CryoSPARC calls, and queues no jobs.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryoagent.config.config_loader import CryoSPARCSettings
from cryoagent.core.experiment_auditor import (
    audit_improvement_log,
    final_report_from_log_text,
    parse_experiment_record,
)


def _offline_capability_tools(config_path: Path):
    """Expose current capability settings without opening a CryoSPARC connection."""
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        settings = CryoSPARCSettings(**(data.get("cryosparc") or {}))
    except (OSError, ValueError, TypeError):
        settings = CryoSPARCSettings.from_env()
    return SimpleNamespace(settings=settings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--state", type=Path, default=Path("outputs/workflow_state.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/master_config.json"),
                        help="Current capability config (read locally; no connection is opened)")
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print complete audit JSON")
    args = parser.parse_args()

    exit_code = 0
    for log_path in args.logs:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        original = final_report_from_log_text(text)
        record = parse_experiment_record(original)
        report = audit_improvement_log(
            log_path,
            cryosparc_tools=_offline_capability_tools(args.config),
            outputs_dir=None if args.no_artifact else args.outputs_dir,
            state_path=args.state,
        )
        print(f"\n=== {log_path.name} ===")
        best = record.get("best_result") or {}
        print(f"Before: no audit; final report claimed best={best.get('job')} "
              f"resolution={best.get('resolution')} cFAR={best.get('cfar')}")
        print(f"After:  {report['status']} — {len([f for f in report['findings'] if f['severity'] == 'critical'])} "
              f"critical finding(s), {len([f for f in report['findings'] if f['severity'] == 'warning'])} warning(s)")
        for finding in report["findings"]:
            if finding["severity"] not in ("critical", "warning"):
                continue
            print(f"[{finding['severity'].upper()}] {finding['check_id']}: {finding['claim']}")
            print(f"  Required: {finding['required_action']}")
        if report.get("artifact"):
            print(f"Artifact: {report['artifact']}")
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        if report["status"] == "fail":
            exit_code = 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
