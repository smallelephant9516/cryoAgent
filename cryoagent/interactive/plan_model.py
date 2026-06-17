"""Structured plan model built from CryoAgent's JSON configuration files.

The "plan" is not a new concept in CryoAgent: it already lives implicitly in the
config files. This module reads that scattered state into a single structured
object so the interactive GUI can render it, the intent parser can reason about
it, and the applier can write edits back deterministically.

Sources:
- ``<configs>/session.json`` -> ``master_workflow.stages`` gives the ordered
  list of stages, whether each is enabled, and its ``agent_group``.
- ``<configs>/<agent_group>/<stage>_config.json`` gives the per-step parameters
  (under ``workflow.<step>``) and step descriptions (under
  ``react_workflow.steps``).
- ``<configs>/microscope_config.json`` gives the acquisition parameters.

Each editable parameter records exactly which file and JSON path it came from,
so :mod:`cryoagent.interactive.plan_applier` can write it back without guessing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("PlanModel")

SESSION_FILENAME = "session.json"
MICROSCOPE_FILENAME = "microscope_config.json"

# Parameter keys we never expose as editable (free-text annotations / helpers).
_SKIP_PARAM_SUFFIXES = ("_description",)


@dataclass
class PlanParam:
    """A single editable parameter within a stage/step."""

    name: str
    value: Any
    # File (relative to the configs dir) the value lives in.
    config_file: str
    # Nested JSON path within that file, e.g. ["workflow", "motion_correction", "binning"].
    json_path: List[str]
    description: str = ""

    @property
    def value_type(self) -> str:
        if isinstance(self.value, bool):
            return "bool"
        if isinstance(self.value, int):
            return "int"
        if isinstance(self.value, float):
            return "float"
        if isinstance(self.value, list):
            return "list"
        if self.value is None:
            return "null"
        return "str"


@dataclass
class PlanStep:
    """A fine-grained step within a stage (e.g. motion_correction)."""

    name: str
    description: str = ""
    params: List[PlanParam] = field(default_factory=list)

    def get_param(self, name: str) -> Optional[PlanParam]:
        for param in self.params:
            if param.name == name:
                return param
        return None


@dataclass
class PlanStage:
    """A coarse pipeline stage (e.g. preprocessing)."""

    name: str
    description: str
    enabled: bool
    agent_group: str
    agent_class: str
    order: int
    steps: List[PlanStep] = field(default_factory=list)
    # Path to the stage config file (relative to the configs dir), if it exists.
    config_file: Optional[str] = None

    def get_step(self, name: str) -> Optional[PlanStep]:
        for step in self.steps:
            if step.name == name:
                return step
        return None


@dataclass
class Plan:
    """The full editable plan derived from the config directory."""

    configs_dir: Path
    stages: List[PlanStage] = field(default_factory=list)
    # Microscope acquisition parameters, exposed as a pseudo-stage's params.
    microscope_params: List[PlanParam] = field(default_factory=list)

    def get_stage(self, name: str) -> Optional[PlanStage]:
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    @property
    def enabled_stages(self) -> List[PlanStage]:
        return [s for s in self.stages if s.enabled]

    # ------------------------------------------------------------------
    # Serialization for the LLM intent parser
    # ------------------------------------------------------------------
    def to_prompt_dict(self) -> Dict[str, Any]:
        """A compact, LLM-friendly view of the plan (names + current values)."""
        stages_view = []
        for stage in self.stages:
            steps_view = []
            for step in stage.steps:
                steps_view.append(
                    {
                        "step": step.name,
                        "description": step.description,
                        "params": {p.name: p.value for p in step.params},
                    }
                )
            stages_view.append(
                {
                    "stage": stage.name,
                    "enabled": stage.enabled,
                    "order": stage.order,
                    "agent_group": stage.agent_group,
                    "description": stage.description,
                    "steps": steps_view,
                }
            )
        return {
            "stage_order": [s.name for s in self.stages],
            "stages": stages_view,
            "microscope_params": {p.name: p.value for p in self.microscope_params},
        }

    def to_prompt_text(self) -> str:
        return json.dumps(self.to_prompt_dict(), indent=2, default=str)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.debug("Config file not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        return None


def _is_editable_param(key: str) -> bool:
    return not any(key.endswith(suffix) for suffix in _SKIP_PARAM_SUFFIXES)


def _build_steps_for_stage(
    stage_config: Dict[str, Any], stage_config_rel: str
) -> List[PlanStep]:
    """Build steps + editable params from a stage config's ``workflow`` section."""
    workflow = stage_config.get("workflow")
    if not isinstance(workflow, dict):
        return []

    # Descriptions + canonical ordering come from react_workflow.steps when present.
    descriptions: Dict[str, str] = {}
    ordered_names: List[str] = []
    react_steps = stage_config.get("react_workflow", {}).get("steps", [])
    if isinstance(react_steps, list):
        for entry in react_steps:
            if isinstance(entry, dict) and entry.get("name"):
                name = entry["name"]
                ordered_names.append(name)
                descriptions[name] = entry.get("description", "")

    # Include any workflow step keys not declared in react_workflow.steps, preserving
    # the order they appear in the workflow dict.
    for name in workflow.keys():
        if name not in ordered_names:
            ordered_names.append(name)

    steps: List[PlanStep] = []
    for step_name in ordered_names:
        step_params_raw = workflow.get(step_name)
        if not isinstance(step_params_raw, dict):
            # react_workflow may list a step (e.g. import_micrographs) that has no
            # editable params in the workflow section; skip it from the editable view.
            continue
        params: List[PlanParam] = []
        for key, value in step_params_raw.items():
            if not _is_editable_param(key):
                continue
            params.append(
                PlanParam(
                    name=key,
                    value=value,
                    config_file=stage_config_rel,
                    json_path=["workflow", step_name, key],
                    description=step_params_raw.get(f"{key}_description", ""),
                )
            )
        steps.append(
            PlanStep(
                name=step_name,
                description=descriptions.get(step_name, ""),
                params=params,
            )
        )
    return steps


def _build_microscope_params(configs_dir: Path) -> List[PlanParam]:
    data = _load_json(configs_dir / MICROSCOPE_FILENAME)
    if not data:
        return []
    micro = data.get("microscope_parameters")
    if not isinstance(micro, dict):
        return []
    params: List[PlanParam] = []
    for key, value in micro.items():
        if not _is_editable_param(key):
            continue
        params.append(
            PlanParam(
                name=key,
                value=value,
                config_file=MICROSCOPE_FILENAME,
                json_path=["microscope_parameters", key],
            )
        )
    return params


def build_plan(configs_dir: Path | str) -> Plan:
    """Build a :class:`Plan` from a configs directory.

    Args:
        configs_dir: Directory containing ``session.json``, ``microscope_config.json``
            and the ``<agent_group>/`` stage-config subdirectories.
    """
    configs_dir = Path(configs_dir)
    plan = Plan(configs_dir=configs_dir)

    session = _load_json(configs_dir / SESSION_FILENAME)
    if not session:
        logger.warning("No %s found in %s; plan will be empty.", SESSION_FILENAME, configs_dir)
        plan.microscope_params = _build_microscope_params(configs_dir)
        return plan

    stages_raw = session.get("master_workflow", {}).get("stages", [])
    for order, stage_entry in enumerate(stages_raw):
        if not isinstance(stage_entry, dict):
            continue
        name = stage_entry.get("name")
        if not name:
            continue
        agent_group = stage_entry.get("agent_group", "cryosparc")
        stage_config_rel = f"{agent_group}/{name}_config.json"
        stage_config_path = configs_dir / agent_group / f"{name}_config.json"
        stage_config = _load_json(stage_config_path)

        steps: List[PlanStep] = []
        config_file: Optional[str] = None
        if stage_config is not None:
            config_file = stage_config_rel
            steps = _build_steps_for_stage(stage_config, stage_config_rel)

        plan.stages.append(
            PlanStage(
                name=name,
                description=stage_entry.get("description", ""),
                enabled=bool(stage_entry.get("enabled", False)),
                agent_group=agent_group,
                agent_class=stage_entry.get("agent_class", ""),
                order=order,
                steps=steps,
                config_file=config_file,
            )
        )

    plan.microscope_params = _build_microscope_params(configs_dir)
    return plan
