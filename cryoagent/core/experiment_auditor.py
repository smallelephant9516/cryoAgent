"""Scientific audit for improvement-mode experiment reports.

The auditor deliberately separates deterministic evidence checks from LLM-based
criticism. Python verifies actual job parameters, input connections, measured
metrics and plan completion; a separate critic call reviews causal language.
Contradicted claims and material scientific overclaims block completion. Search
completeness is not an accuracy requirement, and no finding requires new jobs.
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from .workflow_state import (
    _iter_tool_execution_blocks,
    _params_diff,
    attach_exact_run_params,
    challenger_beats_incumbent,
    extract_improvement_context_from_log,
)


AUDIT_VERSION = 2
_NOISE_RERUNS_REQUIRED = 2  # in addition to the baseline


def _finding(check_id: str, severity: str, claim: str, evidence: Dict[str, Any],
             required_action: str) -> Dict[str, Any]:
    return {
        "check_id": check_id,
        "severity": severity,
        "claim": claim[:500],
        "evidence": evidence,
        "required_action": required_action,
    }


def parse_experiment_record(report_text: str) -> Dict[str, Any]:
    """Return the last valid fenced experiment record in a submitted report."""
    result: Dict[str, Any] = {}
    for match in re.finditer(r"```json\s*(.*?)```", report_text or "", flags=re.S | re.I):
        try:
            value = json.loads(match.group(1))
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and isinstance(value.get("experiments"), list):
            result = value
    return result


def final_report_from_log_text(text: str) -> str:
    """Isolate the final assistant report, excluding prior-evidence prompts."""
    blocks = re.findall(
        r"\[[^\]]+\] ASSISTANT RESPONSE:\n(.*?)(?=\n-{50}|\Z)",
        text or "", flags=re.S,
    )
    for block in reversed(blocks):
        if parse_experiment_record(block):
            return block.strip()
    return blocks[-1].strip() if blocks else (text or "")


def _decode_result(result: str) -> Any:
    try:
        return json.loads(result)
    except (TypeError, ValueError):
        try:
            return ast.literal_eval(result)
        except (ValueError, SyntaxError):
            return None


def harvest_log_evidence(log_path: Path, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Harvest exact run parameters and measurements already present in a log.

    This makes historical audits deterministic and offline. It never queues jobs.
    """
    context = context if context is not None else extract_improvement_context_from_log(log_path)
    run_params = context.setdefault("run_params", {})
    metrics = context.setdefault("job_metrics", {})
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return context

    for tool, _args, result in _iter_tool_execution_blocks(text):
        if tool == "get_job_run_params":
            payload = _decode_result(result)
            if isinstance(payload, dict) and payload.get("success") and payload.get("job_uid"):
                uid = str(payload["job_uid"]).upper()
                payload = dict(payload)
                payload["source"] = "log_params_spec"
                run_params[uid] = payload
        elif tool == "describe_job_results":
            match = re.search(r"Results for (J\d+) \(([^,)]+)", result)
            if not match:
                continue
            uid = match.group(1).upper()
            item = metrics.setdefault(uid, {})
            item["job_type"] = match.group(2)
            for key, pattern, converter in (
                ("resolution", r"resolution: ([\d.]+) Å", float),
                ("particles", r"particles: (\d+)", int),
                ("cfar", r"cFAR: ([\d.]+)", float),
            ):
                found = re.search(pattern, result)
                if found:
                    item[key] = converter(found.group(1))
        elif tool == "get_orientation_diagnostics":
            match = re.search(r"cFAR for (J\d+)[^:]*: ([\d.]+)", result)
            if match:
                metrics.setdefault(match.group(1).upper(), {})["cfar"] = float(match.group(2))
    return context


def enrich_context_from_workflow_state(
    context: Dict[str, Any], state_path: Path, through_log: Optional[str] = None,
) -> Dict[str, Any]:
    """Fill saved evidence without network calls or future-record leakage.

    Timestamped improvement log names sort chronologically. Historical replays
    pass ``through_log`` so later rounds cannot retroactively validate claims.
    """
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return context
    records = list(state.get("records") or [])
    if through_log:
        records = [
            record for record in records
            if not (record.get("stage_outputs") or {}).get("source_log")
            or (record.get("stage_outputs") or {}).get("source_log") <= through_log
        ]
    run_params = context.setdefault("run_params", {})
    metrics = context.setdefault("job_metrics", {})
    for record in records:
        if not str(record.get("stage", "")).startswith("improvement_"):
            continue
        primary = record.get("primary_job_uid")
        if primary:
            saved_metrics = record.get("metrics") or {}
            item = metrics.setdefault(primary, {})
            for source_key, target_key in (
                ("resolution_angstroms", "resolution"),
                ("num_particles", "particles"),
                ("cfar", "cfar"),
                ("job_type", "job_type"),
            ):
                if saved_metrics.get(source_key) is not None:
                    item[target_key] = saved_metrics[source_key]
        for strategy in (record.get("stage_outputs") or {}).get("strategies_tried") or []:
            uid = strategy.get("result_job")
            if not uid:
                continue
            if strategy.get("params") and (
                uid not in run_params
                or run_params[uid].get("source") in ("agent_request_incomplete", "unavailable")
            ):
                run_params[uid] = {
                    "success": True,
                    "job_uid": uid,
                    "job_type": strategy.get("job_type"),
                    "params": strategy.get("params") or {},
                    "inputs": strategy.get("inputs") or [],
                    "source": "workflow_state",
                }
            values = strategy.get("metrics") or {}
            item = metrics.setdefault(uid, {})
            if values.get("result_resolution") is not None:
                item["resolution"] = values["result_resolution"]
            if values.get("result_cfar") is not None:
                item["cfar"] = values["result_cfar"]
            if strategy.get("job_type"):
                item["job_type"] = strategy["job_type"]
    return context


def _input_signature(run: Dict[str, Any], particles_only: bool = False) -> Tuple[Tuple[str, ...], ...]:
    signature = []
    for item in run.get("inputs") or []:
        slot = item.get("slot") or item.get("type") or "input"
        if particles_only and slot != "particles":
            continue
        signature.append((str(slot), str(item.get("job_uid") or ""),
                          str(item.get("group_name") or "")))
    return tuple(sorted(signature))


def _without_seed(params: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in (params or {}).items() if key != "random_seed"}


def _record_experiment_text(experiment: Dict[str, Any]) -> str:
    return "\n".join([
        str(experiment.get("hypothesis") or ""),
        "\n".join(str(step) for step in experiment.get("plan") or []),
        str(experiment.get("one_line_reason") or ""),
    ])


def _has_affirmative_claim(text: str, pattern: str) -> bool:
    """Match a claim while tolerating explicit retractions/nearby negation."""
    for match in re.finditer(pattern, text or "", re.I):
        before = (text or "")[max(0, match.start() - 70):match.start()]
        after = (text or "")[match.end():match.end() + 45]
        if re.search(
            r"(?:\bno\b|\bnot\b|\bwithout\b|\bcannot\b|\bcan't\b|"
            r"\bunsupported\b|\bunproven\b|\bretract(?:ed|ing)?\b)"
            r"(?:\s+[\w'-]+){0,4}\s*$",
            before, re.I,
        ):
            continue
        if re.match(
            r"\s*(?:\w+\s+){0,3}(?:is\s+|was\s+|remains\s+)?"
            r"(?:unsupported|unproven|retracted|not\s+established)\b",
            after, re.I,
        ):
            continue
        return True
    return False


class ExperimentAuditor:
    """Audit one submitted report against actual CryoSPARC job evidence."""

    def __init__(self, cryosparc_tools=None, project_uid: Optional[str] = None,
                 critic_llm=None):
        self.cryosparc_tools = cryosparc_tools
        self.project_uid = project_uid
        self.critic_llm = critic_llm
        self._param_cache: Dict[str, Dict[str, Any]] = {}
        self._metric_cache: Dict[str, Dict[str, Any]] = {}
        self.context: Dict[str, Any] = {}
        self.record: Dict[str, Any] = {}
        self.report_text = ""

    def _job_uids(self) -> List[str]:
        uids = set(re.findall(r"\bJ\d+\b", self.report_text.upper()))
        uids.update(self.context.get("job_uids") or [])
        uids.update((self.context.get("run_params") or {}).keys())
        uids.update((self.context.get("job_metrics") or {}).keys())
        for exp in self.record.get("experiments") or []:
            if exp.get("result_job"):
                uids.add(str(exp["result_job"]).upper())
        for section in ("baseline", "best_result"):
            value = self.record.get(section) or self.context.get(section) or {}
            if value.get("job"):
                uids.add(str(value["job"]).upper())
        return sorted(uid for uid in uids if re.fullmatch(r"J\d+", uid))

    def _params(self, uid: str) -> Dict[str, Any]:
        if uid in self._param_cache:
            return self._param_cache[uid]
        existing = (self.context.get("run_params") or {}).get(uid) or {}
        if existing.get("params") and existing.get("source") not in (
            "agent_request_incomplete", "unavailable",
        ):
            self._param_cache[uid] = existing
            return existing
        loaded = None
        if self.cryosparc_tools is not None and self.project_uid:
            try:
                loaded = self.cryosparc_tools.get_job_run_params(uid, project_uid=self.project_uid)
            except Exception:
                loaded = None
        if isinstance(loaded, dict) and loaded.get("success"):
            existing = loaded
        self._param_cache[uid] = existing or {"params": {}, "inputs": [], "source": "unavailable"}
        return self._param_cache[uid]

    def _metrics(self, uid: str) -> Dict[str, Any]:
        if uid in self._metric_cache:
            return self._metric_cache[uid]
        item = dict((self.context.get("job_metrics") or {}).get(uid) or {})
        baseline = self.record.get("baseline") or self.context.get("baseline") or {}
        if str(baseline.get("job") or "").upper() == uid:
            for source, target in (("resolution", "resolution"), ("cfar", "cfar"),
                                   ("particles", "particles")):
                if baseline.get(source) is not None:
                    item.setdefault(target, baseline[source])
        for exp in self.record.get("experiments") or []:
            if str(exp.get("result_job") or "").upper() == uid:
                for source, target in (("resolution", "resolution"), ("cfar", "cfar"),
                                       ("particles", "particles")):
                    if exp.get(source) is not None:
                        item.setdefault(target, exp[source])
        if self.cryosparc_tools is not None and self.project_uid and not item.get("resolution"):
            try:
                loaded = self.cryosparc_tools.describe_job_results(uid, project_uid=self.project_uid)
            except Exception:
                loaded = None
            if isinstance(loaded, dict) and loaded.get("success"):
                for source, target in (
                    ("resolution_angstroms", "resolution"),
                    ("num_particles", "particles"),
                    ("cfar", "cfar"),
                    ("job_type", "job_type"),
                ):
                    if loaded.get(source) is not None:
                        item[target] = loaded[source]
        self._metric_cache[uid] = item
        return item

    def _prime_caches(self) -> None:
        for uid in self._job_uids():
            self._params(uid)
            self._metrics(uid)

    def _check_parameter_claims(self) -> List[Dict[str, Any]]:
        findings = []
        baseline_job = str((self.record.get("baseline") or self.context.get("baseline") or {}).get("job") or "").upper()
        for exp in self.record.get("experiments") or []:
            uid = str(exp.get("result_job") or "").upper()
            if not uid:
                continue
            text = _record_experiment_text(exp)
            single = _has_affirmative_claim(text, r"single[- ]key|only\s+[\w_]+\s+(?:was\s+)?changed")
            identical = _has_affirmative_claim(text, r"exact\s+params|identical[- ]params|same\s+params_spec")
            if not single and not identical:
                continue
            control_match = re.search(r"(?:control|versus|vs\.?)\s*(?:job\s*)?(J\d+)", text, re.I)
            control = control_match.group(1).upper() if control_match else baseline_job
            if not control or control == uid:
                continue
            actual = self._params(uid)
            reference = self._params(control)
            if not actual.get("params") or not reference.get("params"):
                findings.append(_finding(
                    "parameter_identity", "warning",
                    f"{uid} was described as a controlled comparison with {control}.",
                    {"jobs": [control, uid], "reason": "exact params_spec unavailable"},
                    "Retrieve both jobs with get_job_run_params before making a controlled-comparison claim.",
                ))
                continue
            diff = _params_diff(_without_seed(reference.get("params") or {}),
                                _without_seed(actual.get("params") or {}))
            invalid = (single and len(diff) != 1) or (identical and bool(diff))
            if invalid:
                findings.append(_finding(
                    "parameter_identity", "critical",
                    f"{uid} was described as {'single-key' if single else 'identical-parameter'} versus {control}.",
                    {"control": control, "result": uid,
                     "non_seed_param_diff": diff, "random_seed_excluded": True},
                    "Correct the causal attribution and record every changed key. Use a properly controlled rerun before attributing the effect to one parameter.",
                ))
        return findings

    def _check_noise_floor(self) -> List[Dict[str, Any]]:
        if not _has_affirmative_claim(
            self.report_text,
            r"noise[- ]floor|identical[- ]param(?:eter)?\s+reruns?[^.\n]*(?:scatter|span)",
        ):
            return []
        baseline_job = str((self.record.get("baseline") or self.context.get("baseline") or {}).get("job") or "").upper()
        if not baseline_job:
            return [_finding("noise_floor", "warning", "A noise floor was claimed without an identified baseline.", {},
                             "Identify the baseline and matched reruns, or retract the noise-floor claim.")]
        baseline = self._params(baseline_job)
        baseline_signature = _input_signature(baseline)
        if not baseline.get("params") or not baseline_signature:
            return [_finding("noise_floor", "warning", f"A noise floor was claimed for {baseline_job}.",
                             {"baseline": baseline_job, "reason": "baseline params or inputs unavailable"},
                             "Verify matched params and input connections before claiming a noise floor.")]
        baseline_type = baseline.get("job_type") or self._metrics(baseline_job).get("job_type")
        matched = []
        for uid in self._job_uids():
            run = self._params(uid)
            if not run.get("params") or _input_signature(run) != baseline_signature:
                continue
            if baseline_type and (run.get("job_type") or self._metrics(uid).get("job_type")) != baseline_type:
                continue
            if _params_diff(_without_seed(baseline.get("params") or {}), _without_seed(run.get("params") or {})):
                continue
            if self._metrics(uid).get("resolution") is not None:
                matched.append(uid)
        independent_reruns = [uid for uid in matched if uid != baseline_job]
        if len(independent_reruns) >= _NOISE_RERUNS_REQUIRED:
            return []
        return [_finding(
            "noise_floor", "critical",
            f"The report claims a reproducible same-set noise floor around {baseline_job}.",
            {"baseline": baseline_job, "matched_jobs": matched,
             "independent_matched_reruns": len(independent_reruns),
             "required_independent_reruns": _NOISE_RERUNS_REQUIRED,
             "requirement": "same job type, all input connections, and all params except random_seed"},
            "Retract the noise-floor claim or run at least two genuinely matched independent reruns. Do not label different particle sets or parameter changes as seed controls.",
        )]

    def _check_particle_plan(self) -> List[Dict[str, Any]]:
        findings = []
        for exp in self.record.get("experiments") or []:
            uid = str(exp.get("result_job") or "").upper()
            if not uid:
                continue
            text = _record_experiment_text(exp)
            run = self._params(uid)
            actual_groups = [str(item.get("group_name") or "") for item in run.get("inputs") or []
                             if (item.get("slot") or item.get("type")) == "particles"]
            explicit_groups = sorted(set(re.findall(r"particles_class_\d+", text)))
            missing = [group for group in explicit_groups if group not in actual_groups]
            union_claim = bool(re.search(r"\bunion\b|both\s+(?:clean|good)\s+(?:3D\s+)?classes", text, re.I))
            regroup_used = any("regroup" in str(step).lower() for step in exp.get("plan") or [])
            regroup_mismatch = union_claim and regroup_used and (
                not actual_groups or any("superclass" in group for group in actual_groups)
            )
            if missing or regroup_mismatch:
                acknowledged_invalid = (
                    exp.get("verdict") == "invalid"
                    and bool(re.search(
                        r"did\s+not|does\s+not|not\s+(?:an?\s+)?(?:exact\s+)?(?:class\s+)?union|"
                        r"mismatch|superclass.{0,50}(?:instead|rather)",
                        str(exp.get("one_line_reason") or ""), re.I,
                    ))
                )
                if acknowledged_invalid:
                    continue
                expected = sorted(set(re.findall(r"\b\d{1,3}(?:,\d{3})+\b", self.report_text)))
                findings.append(_finding(
                    "particle_plan", "critical",
                    f"{uid} is reported as testing an explicit particle-class union/selection.",
                    {"result": uid, "declared_groups": explicit_groups,
                     "actual_particle_groups": actual_groups,
                     "measured_particles": self._metrics(uid).get("particles"),
                     "submitted_particles": exp.get("particles"),
                     "explicit_counts_elsewhere_in_report": expected[:10],
                     "regroup_is_not_an_explicit_union": regroup_mismatch},
                    "Run the decisive terminal refinement with the intended particle groups via particles_group_names, or retract this hypothesis verdict. A regroup superclass is not proof of an exact class union.",
                ))
        return findings

    def _check_capabilities(self) -> List[Dict[str, Any]]:
        settings = getattr(self.cryosparc_tools, "settings", None)
        topaz_path = getattr(settings, "topaz_exec_path", None)
        if not topaz_path:
            return []
        normalized = re.sub(r"[*_`()]", " ", self.report_text)
        pattern = (
            r"\btopaz\s+(?:(?:is|was|remains|still)\s+)?(?:not\s+installed|unavailable)|"
            r"\btopaz\s+has\s+no\s+executable|"
            r"(?:re-?)?install\s+topaz\b"
        )
        unavailable = None
        for match in re.finditer(pattern, normalized, re.I):
            before = normalized[max(0, match.start() - 60):match.start()]
            after = normalized[match.end():match.end() + 35]
            if re.search(r"(?:previous|historical|stale|incorrect|old|retracted)\s+(?:\w+\s+){0,3}$", before, re.I):
                continue
            if re.match(r"\s+claim\s+(?:was|is)\s+(?:wrong|stale|incorrect|retracted)", after, re.I):
                continue
            if re.search(r"(?:\bno\b|\bnot\b|\bwithout\b)(?:\s+[\w'-]+){0,3}\s*$", before, re.I):
                continue
            unavailable = match
            break
        if not unavailable:
            return []
        return [_finding(
            "capability_freshness", "critical",
            "The report says Topaz is unavailable or still needs installation.",
            {"current_topaz_exec_path": topaz_path,
             "matched_text": unavailable.group(0)[:250]},
            "Re-verify current worker capability or state only that this round did not test Topaz. Do not reuse a historical unavailable claim after configuration changed.",
        )]

    def _check_measurements(self) -> List[Dict[str, Any]]:
        findings = []
        baseline = self.record.get("baseline") or self.context.get("baseline") or {}
        for exp in self.record.get("experiments") or []:
            uid = str(exp.get("result_job") or "").upper()
            if not uid:
                continue
            measured = self._metrics(uid)
            for key in ("resolution", "cfar", "particles"):
                submitted = exp.get(key)
                actual = measured.get(key)
                if submitted is None or actual is None:
                    continue
                tolerance = 0.015 if key == "resolution" else (0.002 if key == "cfar" else 0)
                if abs(float(submitted) - float(actual)) > tolerance:
                    findings.append(_finding(
                        "measurement_consistency", "critical",
                        f"{uid} has an incorrect {key} value in the experiment record.",
                        {"job": uid, "submitted": submitted, "measured": actual},
                        "Replace the submitted value with describe_job_results' measured value.",
                    ))
            if exp.get("verdict") == "helped" and not challenger_beats_incumbent(
                exp.get("resolution"), exp.get("cfar"),
                baseline.get("resolution"), baseline.get("cfar"),
            ):
                findings.append(_finding(
                    "verdict_consistency", "critical",
                    f"{uid} is marked helped but does not satisfy the baseline-ratchet rule.",
                    {"job": uid, "baseline": baseline,
                     "challenger": {"resolution": exp.get("resolution"), "cfar": exp.get("cfar")}},
                    "Change the verdict and keep the incumbent, or provide the missing validated measurement.",
                ))
        return findings

    def _check_stopping_consistency(self) -> List[Dict[str, Any]]:
        """Check stopping prose for factual contradictions, not completeness.

        Pausing with open hypotheses is valid. Only an affirmative narrative that
        contradicts the submitted experiment record requires a wording fix.
        """
        experiments = self.record.get("experiments") or []
        measured = [
            exp for exp in experiments
            if exp.get("verdict") in ("helped", "worse", "no_difference")
            and exp.get("resolution") is not None
        ]
        untested_jobs = [
            str(exp.get("result_job") or "").upper()
            for exp in experiments if exp.get("verdict") == "invalid"
        ]
        branch = self.record.get("branch_in_progress")
        findings = []

        two_no_gain = _has_affirmative_claim(
            self.report_text,
            r"(?:second|two|2)\s+consecutive|"
            r"(?:two|2)(?:\s+[\w-]+){0,5}\s+experiments?\s+"
            r"(?:with|had|showed|produced)\s+no(?:\s+meaningful)?\s+gain|"
            r"no(?:\s+meaningful)?\s+gain\s+in\s+the\s+last\s+(?:two|2)",
        )
        two_recent_no_gain = len(measured) >= 2 and all(
            exp.get("verdict") in ("worse", "no_difference")
            for exp in measured[-2:]
        )
        if two_no_gain and not two_recent_no_gain:
            findings.append(_finding(
                "stopping_consistency", "critical",
                "The report's consecutive-no-gain narrative contradicts the experiment record.",
                {"last_terminal_results": [
                    {"job": exp.get("result_job"), "verdict": exp.get("verdict")}
                    for exp in measured[-2:]
                ]},
                "Correct the stopping narrative to state the actual outcomes or remove "
                "the no-gain count. This is a wording correction only; do not run new "
                "experiments to satisfy this finding.",
            ))

        exhaustion = _has_affirmative_claim(
            self.report_text,
            r"ranking\s+(?:is\s+)?exhausted|"
            r"\b(?:all|every)\s+(?:ranked\s+)?"
            r"(?:candidates|factors|branches|hypotheses)\s+"
            r"(?:are\s+|were\s+|have\s+been\s+|has\s+been\s+)?"
            r"(?:tried|tested|exhausted|complete|completed)\b|"
            r"\bno\s+(?:remaining|untried|untested)\s+"
            r"(?:candidates|hypotheses|branches)\b|"
            r"\bnothing\s+left\s+to\s+(?:try|test)\b|"
            r"\beverything\s+(?:testable\s+)?has\s+been\s+(?:tried|tested)\b",
        )
        if exhaustion and (branch or untested_jobs):
            findings.append(_finding(
                "stopping_consistency", "critical",
                "The report declares the hypotheses exhausted while the record documents untested work.",
                {"untested_hypothesis_jobs": untested_jobs,
                 "branch_in_progress": branch},
                "Name the open or untested work and frame stopping as a pause, or remove "
                "the exhaustion claim. This is a wording correction only; do not run new "
                "experiments to satisfy this finding.",
            ))
        return findings

    def _evidence_table(self) -> str:
        baseline_job = str((self.record.get("baseline") or self.context.get("baseline") or {}).get("job") or "").upper()
        baseline_params = self._params(baseline_job).get("params") or {} if baseline_job else {}
        lines = []
        for uid in self._job_uids():
            run = self._params(uid)
            metrics = self._metrics(uid)
            if not run.get("params") and not metrics:
                continue
            diff = _without_seed(_params_diff(baseline_params, run.get("params") or {})) if baseline_params else {}
            lines.append(json.dumps({
                "job": uid, "metrics": metrics,
                "inputs": run.get("inputs") or [],
                "non_seed_diff_from_baseline": diff,
                "params_source": run.get("source"),
            }, default=str))
        return "\n".join(lines)

    def _semantic_critic(self, deterministic: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        if self.critic_llm is None:
            return [], None
        system = (
            "You are a skeptical cryo-EM experiment reviewer auditing SCIENTIFIC "
            "ACCURACY, not research completeness. Separate observations from causal "
            "interpretations and consider alternative explanations and confounders. "
            "Use ONLY the supplied evidence; do not invent metrics or jobs. Challenge "
            "claims that misrepresent controls, noise floors, junk, particle-number "
            "limits, symmetry, tool capabilities, or causal conclusions. Untested "
            "hypotheses, open branches, and pausing are legitimate outcomes, not "
            "defects. Never require or recommend that new experiments are necessary "
            "to pass this audit: every required_action must be satisfiable by "
            "correcting, calibrating, or retracting wording using existing evidence. "
            "For a material scientific claim you may mention an optional future "
            "discriminating test, but retraction must always be sufficient. Set "
            "material_accuracy=true only when a claim concerns the scientific meaning "
            "of the data, controls, capabilities, or causal diagnosis. Set it false "
            "for process claims about stopping, pausing, rankings, completeness, "
            "exhaustion, or future work; the existence of untested work itself is not "
            "a finding. Verdicts: contradicted means the evidence refutes the claim; "
            "unsupported means it goes beyond the evidence; supported means the "
            "evidence adequately supports it. Return ONLY one JSON object: "
            "{\"claims\":[{\"claim\":str,\"verdict\":\"supported|contradicted|"
            "unsupported\",\"material_accuracy\":bool,\"rationale\":str,"
            "\"required_action\":str}],\"summary\":str}. Return at most five "
            "important claims."
        )
        human = (
            f"SUBMITTED REPORT:\n{self.report_text[:18000]}\n\n"
            f"DETERMINISTIC JOB EVIDENCE:\n{self._evidence_table()[:24000]}\n\n"
            f"DETERMINISTIC FINDINGS:\n{json.dumps(deterministic, default=str)[:12000]}"
        )
        try:
            response = self.critic_llm.invoke([
                SystemMessage(content=system), HumanMessage(content=human),
            ])
            content = getattr(response, "content", None) or str(response)
            start, end = content.find("{"), content.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("critic returned no JSON object")
            payload = json.loads(content[start:end + 1])
            if not isinstance(payload.get("claims"), list):
                raise ValueError("critic JSON has no claims list")
        except Exception as exc:
            return [_finding(
                "semantic_critic", "info", "The semantic critic could not complete.",
                {"error": str(exc)}, "No action required; deterministic checks remain authoritative.",
            )], None
        findings = []
        for claim in payload.get("claims", [])[:5]:
            if not isinstance(claim, dict) or claim.get("verdict") == "supported":
                continue
            verdict = str(claim.get("verdict") or "unsupported")
            material = bool(claim.get("material_accuracy"))
            critical = verdict == "contradicted" or (verdict == "unsupported" and material)
            if verdict == "contradicted":
                action = (
                    "Correct or retract this claim so it matches the cited evidence. "
                    "Use existing job evidence; no new experiment is required."
                )
            elif material:
                action = (
                    "Calibrate or retract this scientific claim so it matches the evidence. "
                    "A discriminating test is optional future work, never required for this audit."
                )
            else:
                action = (
                    "Calibrate the process wording to match the record. This never "
                    "requires new experiments."
                )
            findings.append(_finding(
                "semantic_critic", "critical" if critical else "warning",
                str(claim.get("claim") or "Unsupported causal claim"),
                {"verdict": verdict, "material_accuracy": material,
                 "rationale": str(claim.get("rationale") or ""),
                 "critic_recommendation": str(claim.get("required_action") or "")},
                action,
            ))
        return findings, payload

    def audit(self, context: Dict[str, Any], report_text: str) -> Dict[str, Any]:
        self.context = context or {}
        self.report_text = report_text or ""
        self.record = parse_experiment_record(self.report_text)
        findings = []
        if not self.record:
            findings.append(_finding(
                "record_format", "critical", "No valid fenced-json experiment record was submitted.", {},
                "Submit the complete report with one valid fenced-json experiment record.",
            ))
            return {"status": "fail", "findings": findings, "critic": None,
                    "audit_version": AUDIT_VERSION, "created_at": time.time()}
        self._prime_caches()
        findings.extend(self._check_parameter_claims())
        findings.extend(self._check_noise_floor())
        findings.extend(self._check_particle_plan())
        findings.extend(self._check_capabilities())
        findings.extend(self._check_measurements())
        findings.extend(self._check_stopping_consistency())
        semantic, critic = self._semantic_critic(findings)
        findings.extend(semantic)
        critical = [item for item in findings if item["severity"] == "critical"]
        warnings = [item for item in findings if item["severity"] == "warning"]
        return {
            "status": "fail" if critical else ("pass_with_warnings" if warnings else "pass"),
            "findings": findings,
            "critical_count": len(critical),
            "warning_count": len(warnings),
            "critic": critic,
            "audit_version": AUDIT_VERSION,
            "created_at": time.time(),
        }


def write_audit_artifact(outputs_dir, source_name: str, report: Dict[str, Any],
                         attempt: Optional[int] = None) -> Path:
    """Persist an audit attempt without changing workflow-state/ratchet data."""
    output = Path(outputs_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name or "improvement").stem
    stem = stem.replace("llm_conversation_improvement_", "improvement_audit_")
    if not stem.startswith("improvement_audit_"):
        stem = f"improvement_audit_{stem}"
    suffix = f"_attempt_{attempt}" if attempt is not None else "_posthoc"
    path = output / f"{stem}{suffix}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def audit_improvement_log(log_path, cryosparc_tools=None, project_uid: Optional[str] = None,
                          critic_llm=None, outputs_dir=None, state_path=None) -> Dict[str, Any]:
    """Offline/live-reverified audit of a completed improvement conversation."""
    path = Path(log_path)
    context = extract_improvement_context_from_log(path)
    attach_exact_run_params(context, cryosparc_tools, project_uid)
    harvest_log_evidence(path, context)
    if state_path is not None:
        enrich_context_from_workflow_state(context, Path(state_path), through_log=path.name)
    report_text = final_report_from_log_text(path.read_text(encoding="utf-8", errors="ignore"))
    report = ExperimentAuditor(
        cryosparc_tools=cryosparc_tools, project_uid=project_uid, critic_llm=critic_llm,
    ).audit(context, report_text)
    if outputs_dir is not None:
        artifact = write_audit_artifact(outputs_dir, path.name, report)
        report["artifact"] = str(artifact)
    return report
