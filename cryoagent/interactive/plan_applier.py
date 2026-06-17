"""Apply confirmed plan edits to a working copy of the config directory.

To stay non-destructive, the interactive mode never edits the canonical
``configs/`` directory directly. Instead :func:`create_working_configs` clones
the relevant config files into a per-run directory (e.g.
``outputs/<run_id>/configs/``) following the same self-contained layout used by
``run_batch_datasets.py``. The orchestrator is then pointed at that working
copy's ``master_config.json``.

:class:`PlanApplier` writes confirmed :class:`~cryoagent.interactive.plan_intent_parser.PlanEdit`
objects into that working copy:
- ``set_param`` -> nested write into the relevant stage / microscope config file.
- ``enable_stage`` / ``disable_stage`` -> flip ``enabled`` in ``session.json``.
- ``reorder`` -> reorder ``master_workflow.stages`` in ``session.json``.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .plan_intent_parser import PlanEdit
from .plan_model import SESSION_FILENAME

logger = logging.getLogger("PlanApplier")


def create_working_configs(source_configs_dir: Path | str, working_dir: Path | str) -> Path:
    """Clone the config directory into ``working_dir`` and return the new configs path.

    Args:
        source_configs_dir: The canonical ``configs/`` directory to copy from.
        working_dir: Destination directory for this run; a ``configs`` subfolder
            is created inside it.

    Returns:
        Path to the working ``configs`` directory (contains ``master_config.json``,
        ``session.json``, ``microscope_config.json`` and stage-config subdirs).
    """
    source_configs_dir = Path(source_configs_dir)
    working_dir = Path(working_dir)
    working_configs = working_dir / "configs"

    if working_configs.exists():
        shutil.rmtree(working_configs)
    working_configs.mkdir(parents=True, exist_ok=True)

    for item in source_configs_dir.iterdir():
        if item.name.startswith("."):
            continue
        dest = working_configs / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    logger.info("Created working config copy at %s", working_configs)
    return working_configs


class PlanApplier:
    """Writes confirmed plan edits into a working configs directory."""

    def __init__(self, configs_dir: Path | str):
        self.configs_dir = Path(configs_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(self, edits: Sequence[PlanEdit]) -> List[str]:
        """Apply a sequence of (already validated) edits.

        Invalid edits are skipped. Returns a list of human-readable messages
        describing what was applied (or why an edit was skipped).
        """
        messages: List[str] = []
        for edit in edits:
            if not edit.is_valid:
                messages.append(f"Skipped: {edit.summary or edit.op} ({edit.error})")
                continue
            try:
                messages.append(self._apply_one(edit))
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to apply edit %s", edit.to_dict())
                messages.append(f"Failed: {edit.summary or edit.op} ({exc})")
        return messages

    # ------------------------------------------------------------------
    # Per-op handlers
    # ------------------------------------------------------------------
    def _apply_one(self, edit: PlanEdit) -> str:
        if edit.op == "set_param":
            return self._apply_set_param(edit)
        if edit.op in ("enable_stage", "disable_stage"):
            return self._apply_toggle_stage(edit)
        if edit.op == "reorder":
            return self._apply_reorder(edit)
        return f"Skipped unknown operation '{edit.op}'."

    def _apply_set_param(self, edit: PlanEdit) -> str:
        # Resolve the file + json path from the live plan so we write to the
        # correct location regardless of which config file the param lives in.
        from .plan_model import build_plan

        plan = build_plan(self.configs_dir)
        stage = plan.get_stage(edit.stage) if edit.stage else None
        if stage is None:
            raise ValueError(f"Stage '{edit.stage}' not found.")
        step = stage.get_step(edit.step) if edit.step else None
        if step is None:
            raise ValueError(f"Step '{edit.step}' not found in '{edit.stage}'.")
        param = step.get_param(edit.param) if edit.param else None
        if param is None:
            raise ValueError(f"Param '{edit.param}' not found in '{edit.stage}.{edit.step}'.")

        target_path = self.configs_dir / param.config_file
        data = _load_json(target_path)
        if data is None:
            raise FileNotFoundError(f"Config file not found: {target_path}")

        old_value = _get_nested(data, param.json_path)
        _set_nested(data, param.json_path, edit.value)
        _save_json(target_path, data)
        return (
            f"{edit.stage}.{edit.step}.{edit.param}: {old_value!r} -> {edit.value!r}"
        )

    def _apply_toggle_stage(self, edit: PlanEdit) -> str:
        enabled = edit.op == "enable_stage"
        session_path = self.configs_dir / SESSION_FILENAME
        data = _load_json(session_path)
        if data is None:
            raise FileNotFoundError(f"{SESSION_FILENAME} not found at {session_path}")

        stages = data.get("master_workflow", {}).get("stages", [])
        for stage_entry in stages:
            if isinstance(stage_entry, dict) and stage_entry.get("name") == edit.stage:
                stage_entry["enabled"] = enabled
                _save_json(session_path, data)
                return f"Stage '{edit.stage}' {'enabled' if enabled else 'disabled'}."
        raise ValueError(f"Stage '{edit.stage}' not found in {SESSION_FILENAME}.")

    def _apply_reorder(self, edit: PlanEdit) -> str:
        session_path = self.configs_dir / SESSION_FILENAME
        data = _load_json(session_path)
        if data is None:
            raise FileNotFoundError(f"{SESSION_FILENAME} not found at {session_path}")

        master_workflow = data.get("master_workflow", {})
        stages = master_workflow.get("stages", [])
        by_name: Dict[str, Any] = {
            s.get("name"): s for s in stages if isinstance(s, dict) and s.get("name")
        }
        if set(edit.order or []) != set(by_name.keys()):
            raise ValueError("Reorder list must contain exactly the existing stages.")

        master_workflow["stages"] = [by_name[name] for name in edit.order]
        data["master_workflow"] = master_workflow
        _save_json(session_path, data)
        return f"Reordered stages to: {' -> '.join(edit.order)}."


# ----------------------------------------------------------------------
# JSON helpers (nested get/set mirror microscope_override_updater patterns)
# ----------------------------------------------------------------------
def _load_json(path: Path) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        return None


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _get_nested(data: Dict[str, Any], path: Sequence[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _set_nested(data: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    current = data
    for key in path[:-1]:
        nxt = current.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            current[key] = nxt
        current = nxt
    current[path[-1]] = value
