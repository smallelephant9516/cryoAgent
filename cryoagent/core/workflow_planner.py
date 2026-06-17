"""LLM-driven workflow planner for dynamic, isolated-stage orchestration.

Phase 2D of the cryoagent refactor. Two responsibilities:

1. ``WorkflowPlanner.choose_next_stage`` — in **dynamic** mode, decide which stage
   to run next (or finish) given the goal and the JSON outputs of completed stages.
   Stages stay isolated: the planner only sees each completed stage's output JSON
   summary, never the stages' internal conversation/context.

2. ``WorkflowPlanner.replan_after_failure`` — in **guided** mode, when a stage
   fails, decide whether to retry it, skip it (must be explicit), or abort, within
   a retry cap. The detailed in-stage forum-informed retry lives in the stage agent
   prompts (shared/failure-recovery-forum.md); this is the cross-stage decision.

The planner reuses the configured LLM (same provider/model as the stage agents).
It is deliberately small: it returns a structured decision the orchestrator acts on.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from .llm_factory import LLMFactory


logger = logging.getLogger("WorkflowPlanner")


# The canonical CryoSPARC stage order, used as the "normal" happy path the planner
# is seeded with. The planner may deviate (skip/repeat/branch) but must justify it.
DEFAULT_STAGE_ORDER: List[str] = [
    "preprocessing",
    "particle_picking",
    "optimization_2d",
    "reconstruction",
    "optimization",
    "heterogeneity",
    "heterogeneity_depth",
    "polish",
]


class PlannerDecision:
    """Structured decision returned by the planner."""

    __slots__ = ("action", "stage", "reason", "raw")

    def __init__(self, action: str, stage: Optional[str], reason: str, raw: str = ""):
        # action ∈ {"run", "finish", "retry", "skip", "abort"}
        self.action = action
        self.stage = stage
        self.reason = reason
        self.raw = raw

    def __repr__(self) -> str:
        return f"PlannerDecision(action={self.action!r}, stage={self.stage!r}, reason={self.reason!r})"


class WorkflowPlanner:
    """Decides cross-stage workflow actions using the configured LLM."""

    def __init__(self, config, available_stages: List[str], goal: Optional[str] = None):
        """
        Args:
            config: A CryoAgentConfig (provides agent.provider + model config).
            available_stages: Stage names that are actually initialized/runnable.
            goal: Optional high-level goal statement for the run.
        """
        self.config = config
        self.available_stages = list(available_stages)
        self.goal = goal or (
            "Process this cryo-EM dataset to the highest-resolution 3D reconstruction "
            "that the data supports, following the standard pipeline unless results "
            "indicate a different path."
        )
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            model_config = self.config.agent.get_current_model_config()
            self._llm = LLMFactory.create_llm(model_config, self.config.agent.provider)
        return self._llm

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _format_completed(self, completed: List[Dict[str, Any]]) -> str:
        """Render completed stages + their JSON output summaries (isolation: JSON only)."""
        if not completed:
            return "(none yet — this is the first stage)"
        lines = []
        for entry in completed:
            name = entry.get("stage")
            success = entry.get("success")
            status = "✅ success" if success else "❌ failed"
            outputs = entry.get("stage_outputs") or {}
            # Only surface the JSON output keys/values, not internal context.
            summary = json.dumps(outputs, default=str)
            if len(summary) > 1500:
                summary = summary[:1500] + " …(truncated)"
            lines.append(f"- {name}: {status}; outputs={summary}")
        return "\n".join(lines)

    def _decision_from_text(self, text: str, valid_stages: List[str]) -> PlannerDecision:
        """Parse the LLM's JSON decision, tolerating fenced/loose output."""
        raw = text.strip()
        payload = raw
        # Strip code fences if present.
        if "```" in payload:
            parts = payload.split("```")
            for part in parts:
                p = part.strip()
                if p.startswith("{") or p.startswith("json"):
                    payload = p[4:].strip() if p.startswith("json") else p
                    break
        try:
            start = payload.index("{")
            end = payload.rindex("}") + 1
            data = json.loads(payload[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Planner returned unparseable decision; defaulting to finish. Raw: %s", raw[:300])
            return PlannerDecision("finish", None, "Planner output could not be parsed.", raw)

        action = str(data.get("action", "")).lower().strip()
        stage = data.get("stage")
        reason = str(data.get("reason", "")).strip()
        if stage is not None:
            stage = str(stage).strip()
        if action == "run" and stage not in valid_stages:
            logger.warning("Planner chose invalid stage %r; finishing instead.", stage)
            return PlannerDecision("finish", None, f"Planner chose unavailable stage {stage!r}.", raw)
        if action not in ("run", "finish"):
            return PlannerDecision("finish", None, reason or "Planner did not choose a runnable stage.", raw)
        return PlannerDecision(action, stage, reason, raw)

    # ------------------------------------------------------------------
    # Dynamic mode: choose the next stage from JSON outputs
    # ------------------------------------------------------------------
    def choose_next_stage(
        self,
        completed: List[Dict[str, Any]],
        remaining_runnable: List[str],
    ) -> PlannerDecision:
        """
        Decide the next stage to run (action="run", stage=<name>) or stop (action="finish").

        Args:
            completed: Ordered list of {stage, success, stage_outputs} for finished stages.
            remaining_runnable: Stage names still available to run.
        """
        if not remaining_runnable:
            return PlannerDecision("finish", None, "No runnable stages remain.")

        system = (
            "You are the workflow planner for a cryo-EM processing pipeline. Each stage "
            "runs in isolation and communicates ONLY through its output JSON (shown to you). "
            "Decide the single next stage to run, or finish.\n\n"
            f"Standard happy-path order: {' → '.join(DEFAULT_STAGE_ORDER)}.\n"
            "Follow this order by default, but you MAY deviate (skip, repeat, or reorder) "
            "when the JSON results justify it. RULES:\n"
            "1. Never skip a stage silently — if you skip one in the standard order, say so "
            "explicitly in 'reason'.\n"
            "2. Only choose a stage from the AVAILABLE list.\n"
            "3. A stage may be repeated if its result was poor and re-running with the "
            "downstream feedback could help.\n"
            "4. Finish when the goal is met or no further stage would add value.\n\n"
            "Respond with ONLY a JSON object: "
            '{\"action\": \"run\"|\"finish\", \"stage\": \"<stage name or null>\", \"reason\": \"<one or two sentences>\"}'
        )
        human = (
            f"GOAL: {self.goal}\n\n"
            f"COMPLETED STAGES (with output JSON):\n{self._format_completed(completed)}\n\n"
            f"AVAILABLE STAGES TO RUN NEXT: {remaining_runnable}\n\n"
            "What is the next action?"
        )
        try:
            resp = self._get_llm().invoke([SystemMessage(content=system), HumanMessage(content=human)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            return self._decision_from_text(text, remaining_runnable)
        except Exception as e:
            logger.error("Planner LLM call failed (%s); falling back to next stage in standard order.", e)
            # Deterministic fallback: first remaining stage in canonical order.
            for stage in DEFAULT_STAGE_ORDER:
                if stage in remaining_runnable:
                    return PlannerDecision("run", stage, f"Fallback to standard order (planner error: {e}).")
            return PlannerDecision("run", remaining_runnable[0], f"Fallback (planner error: {e}).")

    # ------------------------------------------------------------------
    # Guided mode: decide what to do when a stage fails
    # ------------------------------------------------------------------
    def replan_after_failure(
        self,
        failed_stage: str,
        error: str,
        attempt: int,
        max_attempts: int,
        is_critical: bool,
    ) -> PlannerDecision:
        """
        Decide how to handle a failed stage in guided mode.

        Returns a PlannerDecision with action ∈ {"retry", "skip", "abort"}:
        - retry: re-run the same stage (only allowed while attempt < max_attempts).
        - skip: continue to the next stage WITHOUT this one (must be explicit; not
          allowed for critical stages like preprocessing).
        - abort: stop the whole workflow.

        Guardrails enforced here (not left to the LLM):
        - Once attempt >= max_attempts, retry is no longer offered.
        - A critical stage may never be skipped — only retried (if attempts remain) or aborted.
        """
        # Hard guardrail: exhausted retries.
        retries_left = attempt < max_attempts
        if not retries_left and is_critical:
            return PlannerDecision(
                "abort", failed_stage,
                f"Critical stage '{failed_stage}' failed and retry cap ({max_attempts}) reached.",
            )

        system = (
            "You are the workflow planner for a cryo-EM pipeline. A stage just FAILED. "
            "Decide the cross-stage recovery action. (The stage itself already attempted "
            "forum-informed parameter retries internally; this is the higher-level call.)\n\n"
            "Allowed actions:\n"
            "- retry: run the same stage again from scratch (only if retries remain).\n"
            "- skip: proceed to the next stage without this one. NOT allowed for critical "
            "stages. Skipping must be a deliberate, justified choice.\n"
            "- abort: stop the entire workflow.\n\n"
            "Respond with ONLY JSON: "
            '{\"action\": \"retry\"|\"skip\"|\"abort\", \"reason\": \"<one or two sentences>\"}'
        )
        human = (
            f"FAILED STAGE: {failed_stage}\n"
            f"CRITICAL STAGE (cannot be skipped): {is_critical}\n"
            f"ATTEMPT: {attempt} of {max_attempts} (retries remaining: {retries_left})\n"
            f"ERROR: {error}\n\n"
            "What should the workflow do?"
        )
        try:
            resp = self._get_llm().invoke([SystemMessage(content=system), HumanMessage(content=human)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            data = self._parse_recovery(text)
            action = data.get("action", "abort")
            reason = data.get("reason", "")
            # Enforce guardrails regardless of what the LLM said.
            if action == "retry" and not retries_left:
                action = "skip" if not is_critical else "abort"
                reason = f"Retry cap reached; downgraded to {action}. " + reason
            if action == "skip" and is_critical:
                action = "abort"
                reason = f"Cannot skip critical stage '{failed_stage}'; aborting. " + reason
            if action not in ("retry", "skip", "abort"):
                action = "abort"
            return PlannerDecision(action, failed_stage, reason.strip() or "(no reason given)", text)
        except Exception as e:
            logger.error("Recovery planner failed (%s); aborting conservatively.", e)
            return PlannerDecision("abort", failed_stage, f"Recovery planner error: {e}")

    def _parse_recovery(self, text: str) -> Dict[str, str]:
        raw = text.strip()
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            return json.loads(raw[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Recovery decision unparseable; aborting. Raw: %s", raw[:200])
            return {"action": "abort", "reason": "Unparseable planner output."}

