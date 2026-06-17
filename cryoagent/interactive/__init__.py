"""Interactive agentic mode for CryoAgent.

This package adds a second way to drive the pipeline: instead of editing JSON
configs by hand and launching the CLI, the user chats in natural language with
an agent that builds an editable *plan* (which stages run + their per-step
parameters), interprets the user's intent into structured edits, confirms each
change, and then runs the existing orchestrator unchanged.

Modules:
- ``plan_model``: build a structured :class:`Plan` from the config files.
- ``plan_intent_parser``: turn a natural-language request into structured
  :class:`PlanEdit` objects using the configured LLM.
- ``plan_applier``: validate + write confirmed edits to a working copy of the
  configs that the orchestrator then runs against.
"""

from .plan_model import (
    Plan,
    PlanStage,
    PlanStep,
    PlanParam,
    build_plan,
)
from .plan_intent_parser import PlanEdit, PlanIntentParser
from .plan_applier import PlanApplier, create_working_configs

__all__ = [
    "Plan",
    "PlanStage",
    "PlanStep",
    "PlanParam",
    "build_plan",
    "PlanEdit",
    "PlanIntentParser",
    "PlanApplier",
    "create_working_configs",
]
