"""Turn natural-language requests into structured plan edits using the LLM.

The interactive GUI lets the user type things like:

    "change the binning factor of motion correction to 2"
    "skip the polish stage and turn on heterogeneity analysis"
    "use a higher max resolution of 3.5 for ctf estimation"

:class:`PlanIntentParser` sends the current :class:`~cryoagent.interactive.plan_model.Plan`
plus the user's text to the configured LLM (same provider/model as the stage
agents) and asks for a strict JSON list of edits. Each edit is validated against
the real plan so the model can only touch stages/steps/params that actually
exist; anything it invents is flagged instead of silently applied.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from ..config.config_loader import ConfigLoader
from ..core.llm_factory import LLMFactory
from .plan_model import Plan

logger = logging.getLogger("PlanIntentParser")

VALID_OPS = {"set_param", "enable_stage", "disable_stage", "reorder"}


@dataclass
class PlanEdit:
    """A single structured edit proposed by the LLM (pre-confirmation)."""

    op: str
    stage: Optional[str] = None
    step: Optional[str] = None
    param: Optional[str] = None
    value: Any = None
    # New stage order (list of stage names) for op == "reorder".
    order: Optional[List[str]] = None
    summary: str = ""
    # Populated by validation; empty when the edit is applicable.
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "stage": self.stage,
            "step": self.step,
            "param": self.param,
            "value": self.value,
            "order": self.order,
            "summary": self.summary,
            "error": self.error,
        }


@dataclass
class IntentResult:
    """Result of parsing one user message."""

    edits: List[PlanEdit] = field(default_factory=list)
    reply: str = ""
    raw: str = ""

    @property
    def valid_edits(self) -> List[PlanEdit]:
        return [e for e in self.edits if e.is_valid]

    @property
    def invalid_edits(self) -> List[PlanEdit]:
        return [e for e in self.edits if not e.is_valid]


class PlanIntentParser:
    """Parses NL into structured, validated plan edits via the configured LLM."""

    def __init__(self, master_config_path: str):
        """
        Args:
            master_config_path: Path to the master config (its ``agent`` section
                selects the provider/model, mirroring the stage agents).
        """
        self.master_config_path = master_config_path
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            config = ConfigLoader(
                config_path=self.master_config_path,
                master_config_path=self.master_config_path,
            ).load_config()
            model_config = config.agent.get_current_model_config()
            self._llm = LLMFactory.create_llm(model_config, config.agent.provider)
        return self._llm

    # ------------------------------------------------------------------
    # Prompting
    # ------------------------------------------------------------------
    def _system_prompt(self) -> str:
        return (
            "You are the plan editor for a cryo-EM processing pipeline. The user "
            "describes, in natural language, how they want to change the current "
            "PLAN (which stages run, in what order, and the parameters of each "
            "step). Translate their request into a list of structured edits.\n\n"
            "You may ONLY reference stages, steps, and parameters that exist in the "
            "PLAN shown to you. Never invent new ones. If the user asks for something "
            "that has no matching parameter/stage, do not fabricate an edit; instead "
            "explain the problem in the 'reply' field.\n\n"
            "Allowed edit operations:\n"
            "- set_param: change a step parameter. Fields: stage, step, param, value.\n"
            "- enable_stage: turn a stage on. Fields: stage.\n"
            "- disable_stage: turn a stage off. Fields: stage.\n"
            "- reorder: set a new stage execution order. Fields: order (full list of "
            "stage names in the new order).\n\n"
            "For set_param, 'value' MUST be a JSON value of the same kind as the "
            "current value (number stays number, list stays list, etc.).\n\n"
            "Respond with ONLY a JSON object of the form:\n"
            "{\n"
            '  "reply": "<one or two sentences summarizing what you changed or why not>",\n'
            '  "edits": [\n'
            '    {"op": "set_param", "stage": "preprocessing", "step": "motion_correction", '
            '"param": "binning", "value": 2, "summary": "Set motion correction binning to 2"}\n'
            "  ]\n"
            "}\n"
            "If no valid edit can be made, return an empty 'edits' list and explain in 'reply'."
        )

    def _human_prompt(self, plan: Plan, user_text: str) -> str:
        return (
            f"CURRENT PLAN (JSON):\n{plan.to_prompt_text()}\n\n"
            f"USER REQUEST:\n{user_text}\n\n"
            "Return the JSON object with 'reply' and 'edits'."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def parse(self, user_text: str, plan: Plan) -> IntentResult:
        """Parse a user message into validated edits against ``plan``."""
        try:
            resp = self._get_llm().invoke(
                [
                    SystemMessage(content=self._system_prompt()),
                    HumanMessage(content=self._human_prompt(plan, user_text)),
                ]
            )
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:  # pragma: no cover - network/LLM failure
            logger.error("Intent parser LLM call failed: %s", exc)
            return IntentResult(edits=[], reply=f"Could not reach the planning model: {exc}", raw="")

        return self._result_from_text(text, plan)

    # ------------------------------------------------------------------
    # Parsing + validation
    # ------------------------------------------------------------------
    def _result_from_text(self, text: str, plan: Plan) -> IntentResult:
        data = self._extract_json_object(text)
        if data is None:
            return IntentResult(
                edits=[],
                reply="I couldn't interpret that into a concrete plan change. Could you rephrase?",
                raw=text,
            )

        reply = str(data.get("reply", "")).strip()
        edits_raw = data.get("edits", [])
        if not isinstance(edits_raw, list):
            edits_raw = []

        edits: List[PlanEdit] = []
        for item in edits_raw:
            if not isinstance(item, dict):
                continue
            edit = PlanEdit(
                op=str(item.get("op", "")).strip(),
                stage=_as_optional_str(item.get("stage")),
                step=_as_optional_str(item.get("step")),
                param=_as_optional_str(item.get("param")),
                value=item.get("value"),
                order=item.get("order") if isinstance(item.get("order"), list) else None,
                summary=str(item.get("summary", "")).strip(),
            )
            self._validate_edit(edit, plan)
            edits.append(edit)

        return IntentResult(edits=edits, reply=reply, raw=text)

    def _validate_edit(self, edit: PlanEdit, plan: Plan) -> None:
        """Annotate ``edit.error`` if it does not map onto the real plan."""
        if edit.op not in VALID_OPS:
            edit.error = f"Unknown operation '{edit.op}'."
            return

        if edit.op in ("enable_stage", "disable_stage"):
            if not edit.stage or plan.get_stage(edit.stage) is None:
                edit.error = f"Stage '{edit.stage}' does not exist in the plan."
            return

        if edit.op == "reorder":
            if not edit.order:
                edit.error = "Reorder requires a non-empty 'order' list."
                return
            existing = {s.name for s in plan.stages}
            requested = set(edit.order)
            if requested != existing:
                missing = existing - requested
                extra = requested - existing
                problems = []
                if missing:
                    problems.append(f"missing stages {sorted(missing)}")
                if extra:
                    problems.append(f"unknown stages {sorted(extra)}")
                edit.error = "Reorder must list every stage exactly once (" + "; ".join(problems) + ")."
            return

        # op == "set_param"
        stage = plan.get_stage(edit.stage) if edit.stage else None
        if stage is None:
            edit.error = f"Stage '{edit.stage}' does not exist in the plan."
            return
        if not edit.step:
            edit.error = "set_param requires a 'step'."
            return
        step = stage.get_step(edit.step)
        if step is None:
            edit.error = f"Step '{edit.step}' does not exist in stage '{edit.stage}'."
            return
        if not edit.param:
            edit.error = "set_param requires a 'param'."
            return
        param = step.get_param(edit.param)
        if param is None:
            edit.error = (
                f"Parameter '{edit.param}' does not exist in {edit.stage}.{edit.step}."
            )
            return
        # Coerce the value to the current parameter's kind where it's unambiguous.
        coerced, coerce_error = _coerce_value(edit.value, param.value)
        if coerce_error:
            edit.error = coerce_error
            return
        edit.value = coerced

    @staticmethod
    def _extract_json_object(text: str) -> Optional[dict]:
        """Tolerantly extract the first JSON object from possibly fenced text."""
        raw = (text or "").strip()
        payload = raw
        if "```" in payload:
            parts = payload.split("```")
            for part in parts:
                p = part.strip()
                if p.startswith("json"):
                    payload = p[4:].strip()
                    break
                if p.startswith("{"):
                    payload = p
                    break
        try:
            start = payload.index("{")
            end = payload.rindex("}") + 1
            return json.loads(payload[start:end])
        except (ValueError, json.JSONDecodeError):
            logger.warning("Intent parser returned unparseable output: %s", raw[:300])
            return None


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_value(new_value: Any, current_value: Any) -> tuple[Any, Optional[str]]:
    """Best-effort coercion of ``new_value`` to ``current_value``'s kind.

    Returns ``(coerced_value, error)``. ``error`` is non-empty when the value is
    incompatible with the existing parameter type.
    """
    # Unknown/None baseline: accept as-is.
    if current_value is None:
        return new_value, None

    if isinstance(current_value, bool):
        if isinstance(new_value, bool):
            return new_value, None
        if isinstance(new_value, str) and new_value.lower() in ("true", "false"):
            return new_value.lower() == "true", None
        return None, f"Expected a boolean value, got {new_value!r}."

    if isinstance(current_value, int) and not isinstance(current_value, bool):
        if isinstance(new_value, bool):
            return None, f"Expected an integer, got boolean {new_value!r}."
        try:
            return int(new_value), None
        except (TypeError, ValueError):
            return None, f"Expected an integer value, got {new_value!r}."

    if isinstance(current_value, float):
        try:
            return float(new_value), None
        except (TypeError, ValueError):
            return None, f"Expected a number, got {new_value!r}."

    if isinstance(current_value, list):
        if isinstance(new_value, list):
            return new_value, None
        return None, f"Expected a list value, got {new_value!r}."

    # Strings and everything else: stringify.
    return str(new_value), None
