#!/usr/bin/env python3
"""Test script to monitor a CryoSPARC import movies job from queue to completion."""

import sys
import time
from pathlib import Path

# Make repository modules importable when running as a script
sys.path.insert(0, str(Path(__file__).parent))

from cryoagent.tools.cryosparc_tools import CryoSPARCTools
from cryoagent.config.config_loader import ConfigLoader


def monitor_import_job(
    cryosparc_tools: CryoSPARCTools,
    project_uid: str,
    workspace_uid: str,
    job_uid: str,
    timeout: int,
    poll_interval: int
):
    """Poll CryoSPARC for import job status updates until it finishes or the timeout is hit."""
    deadline = time.time() + timeout
    print(f"Monitoring import job {job_uid} (timeout={timeout}s, poll_interval={poll_interval}s)")

    while True:
        status = cryosparc_tools.get_job_status(
            job_uid,
            project_uid=project_uid,
            workspace_uid=workspace_uid
        )
        progress = status.get("progress")
        message = status.get("message")
        line_parts = [f"status={status['status']}"]

        if isinstance(progress, (int, float)):
            line_parts.append(f"progress={progress}%")
        if message:
            line_parts.append(f"message={message}")

        print("Job update:" , ", ".join(line_parts))

        if status["status"] in ("completed", "failed", "cancelled"):
            return status
        if time.time() > deadline:
            raise TimeoutError(f"Import job {job_uid} did not finish within {timeout} seconds")

        time.sleep(poll_interval)


def test_import_movies_with_monitoring() -> bool:
    """Queue an import movies job and monitor it until completion."""
    try:
        print("Starting CryoSPARC import movies monitoring test")

        config_loader = ConfigLoader("config.json")
        config = config_loader.load_config()
        cryosparc_tools = CryoSPARCTools(config.cryosparc)

        print("Loaded workflow parameters:")
        print(f"  Project UID: {config.workflow.project_uid}")
        print(f"  Workspace UID: {config.workflow.workspace_uid}")
        print(f"  Movies path: {config.workflow.movies_path}")
        if config.workflow.gain_ref_path:
            print(f"  Gain reference path: {config.workflow.gain_ref_path}")

        import_result = cryosparc_tools.import_movies(
            project_uid=config.workflow.project_uid,
            workspace_uid=config.workflow.workspace_uid,
            movies_path=config.workflow.movies_path,
            gain_ref_path=config.workflow.gain_ref_path,
            pixel_size=config.workflow.pixel_size,
            voltage=config.workflow.voltage,
            cs_mm=config.workflow.cs_mm,
            dose=config.workflow.dose,
            wait_for_completion=False
        )

        job_uid = import_result["job_uid"]
        print(f"Queued import movies job: {job_uid}")

        timeout = config.job_management.default_timeout
        poll_interval = config.job_management.status_check_interval
        final_status = monitor_import_job(
            cryosparc_tools,
            config.workflow.project_uid,
            config.workflow.workspace_uid,
            job_uid,
            timeout,
            poll_interval
        )

        print("Import job finished with status:", final_status["status"])
        if final_status.get("progress") is not None:
            print(f"Final reported progress: {final_status['progress']}%")
        if final_status.get("message"):
            print(f"Final message: {final_status['message']}")

        success = final_status["status"] == "completed"
        if success:
            print("Import movies job completed successfully")
        else:
            print("Import movies job did not complete successfully")
        return success

    except Exception as exc:
        print(f"Import movies monitoring test failed: {exc}")
        return False


if __name__ == "__main__":
    RESULT = test_import_movies_with_monitoring()
    sys.exit(0 if RESULT else 1)
