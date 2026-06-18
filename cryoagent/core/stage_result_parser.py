"""Shared parser for stage-workflow execution logs.

Several stage workflows (preprocessing, particle picking, reconstruction,
polish) all derive per-step success the same way: index the agent's tool
execution log, then for each expected step decide success/failure through the
identical ladder:

    no record  ->  tool error  ->  no job_uid  ->  completion not confirmed  ->  status

This module holds that ladder ONCE so the four workflows can't drift apart
(the reconstruction/polish "wait_for_job-absent" bug had to be fixed in two
places precisely because the logic was duplicated).

The other stage workflows (optimization, 2D optimization, heterogeneity,
heterogeneity_depth) derive results from metric tools (get_fsc_info) in a loop
rather than per-step, so they intentionally do NOT use this parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StepOutcome:
    """Semantic result of checking one workflow step against the execution log.

    Workflows map this onto their own ``*Result`` dataclass. ``message`` and
    ``error`` are human-facing log text; ``success``/``job_uid``/``status`` carry
    the logic-bearing fields.
    """
    success: bool
    job_uid: Optional[str] = None
    job_uids: List[str] = field(default_factory=list)
    status: Optional[str] = None
    statuses: List[Optional[str]] = field(default_factory=list)
    error: Optional[str] = None
    message: str = ""
    reason_code: str = ""  # no_record | tool_error | no_job_uid | not_confirmed | ok


def index_execution_log(
    execution_log: List[Dict[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """Build (tool_entries, waits) from an agent tool-execution log.

    ``tool_entries`` maps tool name -> ordered list of its invocations.
    ``waits`` maps job_uid -> the result of the wait_for_job call for it.
    """
    tool_entries: Dict[str, List[Dict[str, Any]]] = {}
    waits: Dict[str, Dict[str, Any]] = {}
    for entry in execution_log or []:
        tool_name = entry.get("tool")
        tool_entries.setdefault(tool_name, []).append(entry)
        if tool_name == "wait_for_job" and entry.get("result"):
            job_uid = entry.get("params", {}).get("job_uid")
            if job_uid:
                waits[job_uid] = entry["result"]
    return tool_entries, waits


def _status_for(
    job_uid: str,
    waits: Dict[str, Dict[str, Any]],
    result_payload: Any,
    allow_action_status_fallback: bool,
) -> Optional[str]:
    """Completion status for a job: prefer wait_for_job, else the action tool's
    own reported status (action tools block until done and return final status)."""
    wait_info = waits.get(job_uid)
    if wait_info:
        return wait_info.get("status")
    if allow_action_status_fallback and isinstance(result_payload, dict):
        return result_payload.get("status")
    return None


def outcome_for_record(
    record: Dict[str, Any],
    waits: Dict[str, Dict[str, Any]],
    *,
    synchronous: bool = False,
) -> StepOutcome:
    """Run the completion ladder on a single already-selected invocation record.

    Ladder: tool error -> no job_uid -> completion not confirmed -> status.
    Used by check_step (after it selects the record) and directly by workflows
    that do their own record selection (e.g. polish's first/last-success logic).
    """
    error_message = record.get("error")
    result_payload = record.get("result", {})
    job_uid = result_payload.get("job_uid") if isinstance(result_payload, dict) else None

    job_uids: List[str] = []
    if isinstance(result_payload, dict):
        if result_payload.get("job_uids"):
            job_uids = list(result_payload["job_uids"])
        elif job_uid:
            job_uids = [job_uid]

    if error_message:
        return StepOutcome(False, job_uid=job_uid, reason_code="tool_error",
                           error=error_message,
                           message="Tool execution reported an error")

    if not job_uids:
        return StepOutcome(False, job_uid=job_uid, reason_code="no_job_uid",
                           error="Tool did not return a job UID",
                           message="Unable to confirm CryoSPARC job submission")

    statuses: List[Optional[str]] = []
    for uid in job_uids:
        status = _status_for(uid, waits, result_payload,
                             allow_action_status_fallback=True)
        if status is None and synchronous:
            status = "completed"
        statuses.append(status)

    unconfirmed = [u for u, s in zip(job_uids, statuses) if s is None]
    if unconfirmed:
        return StepOutcome(False, job_uid=job_uid, job_uids=job_uids,
                           reason_code="not_confirmed",
                           error="Job completion was not confirmed",
                           message=f"No completion status for: {', '.join(unconfirmed)}")

    success = all(s == "completed" for s in statuses)
    if len(job_uids) == 1:
        msg = (f"CryoSPARC job {job_uids[0]} completed successfully" if success
               else f"CryoSPARC job {job_uids[0]} finished with status '{statuses[0]}'")
    else:
        msg = (f"CryoSPARC jobs {', '.join(job_uids)} completed successfully" if success
               else f"CryoSPARC jobs finished with statuses: {', '.join(str(s) for s in statuses)}")
    error = None if success else f"Job statuses: {', '.join(str(s) for s in statuses)}"

    return StepOutcome(success, job_uid=job_uid, job_uids=job_uids,
                       status=statuses[0], statuses=statuses,
                       reason_code="ok" if success else "not_completed",
                       error=error, message=msg)


def check_step(
    tool_entries: Dict[str, List[Dict[str, Any]]],
    waits: Dict[str, Dict[str, Any]],
    tool_name: str,
    *,
    record_index: int = -1,
    synchronous: bool = False,
) -> StepOutcome:
    """Evaluate one expected step against the indexed execution log.

    Args:
        tool_name: the action tool whose invocation realizes this step.
        record_index: which invocation to inspect. -1 (default) = the latest;
            picking uses positional indices because it calls some tools several
            times in one stage.
        synchronous: True for tools that block internally and never need a
            separate wait_for_job (e.g. select_2d_classes). With no wait entry,
            a returned job_uid is treated as completed.

    Returns a StepOutcome; the ladder lives in this module only.
    """
    records = tool_entries.get(tool_name, [])

    if record_index < 0:
        if not records:
            return StepOutcome(False, reason_code="no_record",
                               error=f"{tool_name} was never executed",
                               message="No tool invocation recorded")
    else:
        if len(records) <= record_index:
            return StepOutcome(False, reason_code="no_record",
                               error=f"{tool_name} invocation #{record_index + 1} was never executed",
                               message="No tool invocation recorded")
    record = records[record_index]
    return outcome_for_record(record, waits, synchronous=synchronous)
