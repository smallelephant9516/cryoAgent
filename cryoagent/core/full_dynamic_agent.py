"""Fully-dynamic, from-scratch single-agent mode (`--mode full_dynamic`).

Unlike the guided/dynamic stage pipeline, this mode runs ONE ReAct agent that
is given almost nothing to go on:

* the acquisition parameters from ``microscope_config.json`` (pixel size,
  voltage, Cs, dose, particle diameter, symmetry) and the input data path(s)
  (``movies_path`` / ``micrographs_path``);
* the CryoSPARC session ids (``project_uid`` / ``workspace_uid``) taken from the
  ``workflow`` block of ``session.json``.

It is given NO predefined stage order, NO stage ``*_config.json`` tuned
parameters, NO blackboard, and it does NOT reuse any multi-stage workflow
prompt. It must decide every action — from import all the way to a refined 3D
density — purely from each tool's own description and the official CryoSPARC
guide (``consult_cryosparc_guide``).

It reuses :class:`ImprovementAgent`'s tool-assembly machinery (which collects
the full deduplicated atomic toolset from the already-initialized stage agents
plus ``consult_cryosparc_guide``), so the ~30 atomic wrappers are not
reimplemented here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .improvement_agent import ImprovementAgent


logger = logging.getLogger("FullDynamicAgent")


# Microscope-config keys surfaced to the agent, in display order.
_MICROSCOPE_SUMMARY_KEYS = (
    "pixel_size",
    "voltage",
    "cs_mm",
    "dose",
    "particle_diameter",
    "symmetry",
    "movies_path",
    "micrographs_path",
    "gain_ref_path",
    "gain_rot",
    "gain_flip",
)


class FullDynamicAgent(ImprovementAgent):
    """Single from-scratch ReAct agent for the ``full_dynamic`` mode.

    Inherits the full-toolset assembly (``_collect_tool_methods`` /
    ``_create_tools``) from :class:`ImprovementAgent` but starts with no prior
    results: no blackboard, no stage configs, no workflow recipe.
    """

    def __init__(self, stage_agents: Dict[str, Any], cryosparc_tools, config, llm=None):
        # No blackboard / workflow_state: this agent builds the pipeline from zero.
        super().__init__(
            stage_agents=stage_agents,
            cryosparc_tools=cryosparc_tools,
            config=config,
            workflow_state=None,
            llm=llm,
        )
        self.stage_name = "full_dynamic"

    def _max_iterations(self) -> int:
        """The whole pipeline (import -> motion/CTF -> pick -> extract -> 2D ->
        ab-initio -> refine), with verify steps, runs in a single loop, so this
        needs a high iteration ceiling.

        Prefer an explicit ``agent.full_dynamic_max_iterations`` when set;
        otherwise use a high floor over the configured ``agent.max_iterations``.
        """
        explicit = getattr(self.config.agent, "full_dynamic_max_iterations", None)
        if explicit and int(explicit) > 0:
            return int(explicit)
        return max(60, int(getattr(self.config.agent, "max_iterations", 10) or 10))

    def _get_react_system_prompt(self) -> str:
        # include_failure_recovery=False: the only guidance this agent gets is
        # tool descriptions + the CryoSPARC guide — no extra shared prompts.
        return self._compose_stage_system_prompt(
            "shared/full-dynamic-agent.md",
            self._get_full_dynamic_context(),
            include_failure_recovery=False,
        )

    def _get_full_dynamic_context(self) -> Dict[str, Any]:
        return {
            "project_uid": self.config.workflow.project_uid,
            "workspace_uid": self.config.workflow.workspace_uid,
            "microscope_summary": self._build_microscope_summary(),
        }

    def _build_microscope_summary(self) -> str:
        """Read microscope_config.json and render a plain key: value summary.

        Reads the raw ``microscope_parameters`` directly (independent of the
        ``overwrite`` flag) so the agent always sees the acquisition params and
        input path(s) it is allowed to know about.
        """
        params = self._load_microscope_parameters()
        if not params:
            return "(microscope_config.json not found or empty)"

        lines = []
        for key in _MICROSCOPE_SUMMARY_KEYS:
            if key in params and params[key] not in (None, ""):
                lines.append(f"- {key}: {params[key]}")
        # Include any extra acquisition keys not in the canonical list.
        for key, value in params.items():
            if key in _MICROSCOPE_SUMMARY_KEYS:
                continue
            if value in (None, ""):
                continue
            lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "(no microscope parameters set)"

    def _load_microscope_parameters(self) -> Dict[str, Any]:
        config_path = Path(self.config.workflow.microscope_config_path)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            with open(config_path, "r", encoding="utf-8") as fp:
                data = json.load(fp) or {}
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read microscope_config.json at %s: %s", config_path, exc)
            return {}
        params = data.get("microscope_parameters")
        return params if isinstance(params, dict) else {}
