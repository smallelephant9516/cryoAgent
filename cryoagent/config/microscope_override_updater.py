"""
Utility functions to propagate microscope_config.json overrides into stage configs.

This module centralises the logic for applying derived microscope parameters (e.g.
LoG diameter ranges, extraction box sizes, symmetry overrides) so that the master
orchestrator can ensure all stage configuration files stay in sync whenever the
user toggles overwrite=true in configs/microscope_config.json.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

LOGGER = logging.getLogger(__name__)


class _UnsetType:
    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return "<UNSET>"


_UNSET = _UnsetType()
MicroscopeValue = Union[str, float, int, None, _UnsetType]


@dataclass(frozen=True)
class MicroscopeParameters:
    """Container for microscope override parameters."""

    movies_path: MicroscopeValue = _UNSET
    gain_ref_path: MicroscopeValue = _UNSET
    gain_rot: MicroscopeValue = _UNSET
    gain_flip: MicroscopeValue = _UNSET
    pixel_size: MicroscopeValue = _UNSET
    voltage: MicroscopeValue = _UNSET
    cs_mm: MicroscopeValue = _UNSET
    dose: MicroscopeValue = _UNSET
    particle_diameter: MicroscopeValue = _UNSET
    symmetry: MicroscopeValue = _UNSET

    @property
    def is_complete(self) -> bool:
        """Return True if the critical parameters are present."""
        pixel_size = _value_or_none(self.pixel_size)
        diameter = _value_or_none(self.particle_diameter)
        return pixel_size is not None and diameter is not None


def apply_microscope_overrides_if_enabled(base_path: Optional[Path] = None) -> None:
    """
    Apply microscope overrides to all relevant stage configuration files when requested.

    Args:
        base_path: Optional base directory of the project workspace. Defaults to CWD.
    """
    base_dir = Path(base_path).resolve() if base_path else Path.cwd()
    microscope_config_path = base_dir / "configs" / "microscope_config.json"

    microscope_data = _load_json(microscope_config_path)
    if not microscope_data:
        return

    if not microscope_data.get("overwrite", False):
        LOGGER.debug("Microscope overrides disabled (overwrite flag is false).")
        return

    overrides = _parse_microscope_parameters(microscope_data.get("microscope_parameters", {}))

    # Stage configuration files to update (CryoSPARC + RELION).
    stage_config_paths = {
        "relion_preprocessing": base_dir / "configs" / "relion" / "preprocessing_config.json",
        "cryosparc_preprocessing": base_dir / "configs" / "cryosparc" / "preprocessing_config.json",
        "relion_picking": base_dir / "configs" / "relion" / "particle_picking_config.json",
        "cryosparc_picking": base_dir / "configs" / "cryosparc" / "particle_picking_config.json",
        "relion_reconstruction": base_dir / "configs" / "relion" / "reconstruction_config.json",
        "cryosparc_reconstruction": base_dir / "configs" / "cryosparc" / "reconstruction_config.json",
    }

    for stage_name, cfg_path in stage_config_paths.items():
        try:
            _apply_overrides_to_stage(stage_name, cfg_path, overrides)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to apply microscope overrides to %s: %s", cfg_path, exc)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON from disk, returning None if the file is missing or invalid."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        LOGGER.debug("Configuration file not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        LOGGER.error("Invalid JSON in %s: %s", path, exc)
        return None


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    """Persist JSON data back to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def _parse_microscope_parameters(raw: Dict[str, Any]) -> MicroscopeParameters:
    """Parse microscope parameter overrides into a structured dataclass."""
    return MicroscopeParameters(
        movies_path=_maybe_cast(raw, "movies_path", _safe_str),
        gain_ref_path=_maybe_cast(raw, "gain_ref_path", _safe_str),
        gain_rot=_maybe_cast(raw, "gain_rot", _safe_int),
        gain_flip=_maybe_cast(raw, "gain_flip", _safe_int),
        pixel_size=_maybe_cast(raw, "pixel_size", _safe_float),
        voltage=_maybe_cast(raw, "voltage", _safe_float),
        cs_mm=_maybe_cast(raw, "cs_mm", _safe_float),
        dose=_maybe_cast(raw, "dose", _safe_float),
        particle_diameter=_maybe_cast(raw, "particle_diameter", _safe_float),
        symmetry=_maybe_cast(raw, "symmetry", _safe_str),
    )


def _maybe_cast(raw: Dict[str, Any], key: str, caster) -> MicroscopeValue:
    if key not in raw:
        return _UNSET
    return caster(raw.get(key)) if caster else raw.get(key)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _apply_overrides_to_stage(stage_name: str, cfg_path: Path, overrides: MicroscopeParameters) -> None:
    """Dispatch override logic per stage type."""
    config_data = _load_json(cfg_path)
    if config_data is None:
        return

    updated = False
    if stage_name == "relion_preprocessing":
        updated = _update_relion_preprocessing(config_data, overrides)
    elif stage_name == "cryosparc_preprocessing":
        updated = _update_cryosparc_preprocessing(config_data, overrides)
    elif stage_name == "relion_picking":
        updated = _update_relion_picking(config_data, overrides)
    elif stage_name == "cryosparc_picking":
        updated = _update_cryosparc_picking(config_data, overrides)
    elif stage_name == "relion_reconstruction":
        updated = _update_relion_reconstruction(config_data, overrides)
    elif stage_name == "cryosparc_reconstruction":
        updated = _update_cryosparc_reconstruction(config_data, overrides)

    if updated:
        _save_json(cfg_path, config_data)
        LOGGER.info("Applied microscope overrides to %s", cfg_path)


def _update_relion_preprocessing(config_data: Dict[str, Any], overrides: MicroscopeParameters) -> bool:
    paths_to_update = [
        (["microscope_parameters", "movies_path"], overrides.movies_path),
        (["microscope_parameters", "gain_ref_path"], overrides.gain_ref_path),
        (["microscope_parameters", "pixel_size"], overrides.pixel_size),
        (["microscope_parameters", "voltage"], overrides.voltage),
        (["microscope_parameters", "cs_mm"], overrides.cs_mm),
        (["microscope_parameters", "dose"], overrides.dose),
        (["microscope_parameters", "gain_rot"], overrides.gain_rot),
        (["microscope_parameters", "gain_flip"], overrides.gain_flip),
        (["microscope_parameters", "particle_diameter"], overrides.particle_diameter),
        (["microscope_parameters", "symmetry"], overrides.symmetry),
        (["workflow", "import_movies", "movies_path"], overrides.movies_path),
        (["workflow", "import_movies", "gain_ref_path"], overrides.gain_ref_path),
        (["workflow", "import_movies", "pixel_size"], overrides.pixel_size),
        (["workflow", "import_movies", "voltage"], overrides.voltage),
        (["workflow", "import_movies", "cs_mm"], overrides.cs_mm),
        (["workflow", "import_movies", "dose"], overrides.dose),
        (["workflow", "import_movies", "gain_rot"], overrides.gain_rot),
        (["workflow", "import_movies", "gain_flip"], overrides.gain_flip),
    ]
    return _bulk_update(config_data, paths_to_update)


def _update_cryosparc_preprocessing(config_data: Dict[str, Any], overrides: MicroscopeParameters) -> bool:
    paths_to_update = [
        (["microscope_parameters", "movies_path"], overrides.movies_path),
        (["microscope_parameters", "gain_ref_path"], overrides.gain_ref_path),
        (["microscope_parameters", "pixel_size"], overrides.pixel_size),
        (["microscope_parameters", "voltage"], overrides.voltage),
        (["microscope_parameters", "cs_mm"], overrides.cs_mm),
        (["microscope_parameters", "dose"], overrides.dose),
        (["microscope_parameters", "gain_rot"], overrides.gain_rot),
        (["microscope_parameters", "gain_flip"], overrides.gain_flip),
        (["microscope_parameters", "particle_diameter"], overrides.particle_diameter),
        (["microscope_parameters", "symmetry"], overrides.symmetry),
        (["workflow", "import_movies", "movies_path"], overrides.movies_path),
        (["workflow", "import_movies", "gain_ref_path"], overrides.gain_ref_path),
        (["workflow", "import_movies", "pixel_size"], overrides.pixel_size),
        (["workflow", "import_movies", "voltage"], overrides.voltage),
        (["workflow", "import_movies", "cs_mm"], overrides.cs_mm),
        (["workflow", "import_movies", "dose"], overrides.dose),
        (["workflow", "import_movies", "gain_rot"], overrides.gain_rot),
        (["workflow", "import_movies", "gain_flip"], overrides.gain_flip),
    ]
    return _bulk_update(config_data, paths_to_update)


def _update_relion_picking(config_data: Dict[str, Any], overrides: MicroscopeParameters) -> bool:
    updates_made = False
    paths_to_update = [
        (["microscope_parameters", "pixel_size"], overrides.pixel_size),
        (["microscope_parameters", "particle_diameter"], overrides.particle_diameter),
    ]
    updates_made |= _bulk_update(config_data, paths_to_update)

    diameter = _value_or_none(overrides.particle_diameter)
    if diameter is not None:
        log_min = _round_sig(diameter * 0.7)
        log_max = _round_sig(diameter * 1.3)
        class_diameter = _scale_value(diameter, 1.2)
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "blob_picker", "LoG_diam_min"], log_min),
                (["workflow", "blob_picker", "LoG_diam_max"], log_max),
                (["workflow", "classification_2d", "particle_diameter"], class_diameter),
                (["workflow", "classification_2d2", "particle_diameter"], class_diameter),
            ],
        )

    if overrides.is_complete:
        pixel_size = _value_or_none(overrides.pixel_size)
        box_size = _compute_nearest_box_size(diameter, pixel_size)
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "particle_extraction", "extract_size"], box_size),
                (["workflow", "particle_extraction2", "extract_size"], box_size),
            ],
        )

    return updates_made


def _update_cryosparc_picking(config_data: Dict[str, Any], overrides: MicroscopeParameters) -> bool:
    updates_made = False
    paths_to_update = [
        (["microscope_parameters", "pixel_size"], overrides.pixel_size),
        (["microscope_parameters", "particle_diameter"], overrides.particle_diameter),
        (["microscope_parameters", "dose"], overrides.dose),
        (["microscope_parameters", "voltage"], overrides.voltage),
        (["microscope_parameters", "cs_mm"], overrides.cs_mm),
        (["microscope_parameters", "symmetry"], overrides.symmetry),
        (["microscope_parameters", "movies_path"], overrides.movies_path),
        (["microscope_parameters", "gain_ref_path"], overrides.gain_ref_path),
        (["microscope_parameters", "gain_rot"], overrides.gain_rot),
        (["microscope_parameters", "gain_flip"], overrides.gain_flip),
    ]
    updates_made |= _bulk_update(config_data, paths_to_update)

    particle_diameter = _value_or_none(overrides.particle_diameter)
    if particle_diameter is not None:
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "blob_picker", "particle_diameter"], _round_sig(particle_diameter)),
            ],
        )

    if overrides.is_complete:
        pixel_size = _value_or_none(overrides.pixel_size)
        box_size = _compute_nearest_box_size(particle_diameter, pixel_size)
        if box_size is not None:
            desc = (
                f"Extraction box size in pixels. Calculated as "
                f"{particle_diameter}/{pixel_size} + 125 ≈ {box_size}."
            )
            updates_made |= _bulk_update(
                config_data,
                [
                    (["workflow", "particle_extraction", "box_size_pix"], box_size),
                    (["workflow", "particle_extraction", "box_size_pix_description"], desc),
                ],
            )

    return updates_made


def _update_relion_reconstruction(config_data: Dict[str, Any], overrides: MicroscopeParameters) -> bool:
    updates_made = False
    paths_to_update = [
        (["microscope_parameters", "particle_diameter"], overrides.particle_diameter),
        (["microscope_parameters", "symmetry"], overrides.symmetry),
    ]
    updates_made |= _bulk_update(config_data, paths_to_update)

    symmetry = _value_or_none(overrides.symmetry)
    if symmetry is not None:
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "ab_initio_reconstruction", "align_sym"], symmetry),
                (["workflow", "refinement_3d", "sym"], symmetry),
            ],
        )

    particle_diameter = _value_or_none(overrides.particle_diameter)
    if particle_diameter is not None:
        scaled_diameter = _scale_value(particle_diameter, 1.2)
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "ab_initio_reconstruction", "particle_diameter"], scaled_diameter),
                (["workflow", "refinement_3d", "particle_diameter"], scaled_diameter),
            ],
        )

    if overrides.is_complete:
        pixel_size = _value_or_none(overrides.pixel_size)
        box_size = _compute_nearest_box_size(particle_diameter, pixel_size)
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "particle_reextraction", "extract_size"], box_size),
            ],
        )

    return updates_made


def _update_cryosparc_reconstruction(config_data: Dict[str, Any], overrides: MicroscopeParameters) -> bool:
    updates_made = False
    paths_to_update = [
        (["microscope_parameters", "particle_diameter"], overrides.particle_diameter),
        (["microscope_parameters", "symmetry"], overrides.symmetry),
        (["microscope_parameters", "pixel_size"], overrides.pixel_size),
        (["microscope_parameters", "voltage"], overrides.voltage),
        (["microscope_parameters", "cs_mm"], overrides.cs_mm),
        (["microscope_parameters", "dose"], overrides.dose),
        (["microscope_parameters", "movies_path"], overrides.movies_path),
        (["microscope_parameters", "gain_ref_path"], overrides.gain_ref_path),
        (["microscope_parameters", "gain_rot"], overrides.gain_rot),
        (["microscope_parameters", "gain_flip"], overrides.gain_flip),
    ]
    updates_made |= _bulk_update(config_data, paths_to_update)

    # Ab initio always stays C1; only propagate known point-group symmetry into
    # refinement. CryoSPARC recommends C1 for ab initio unless forced otherwise.
    updates_made |= _bulk_update(
        config_data,
        [
            (["workflow", "ab_initio", "symmetry"], "C1"),
        ],
    )
    symmetry = _value_or_none(overrides.symmetry)
    if symmetry is not None:
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "refinement", "symmetry"], symmetry),
            ],
        )

    particle_diameter = _value_or_none(overrides.particle_diameter)
    if particle_diameter is not None:
        scaled_diameter = _scale_value(particle_diameter, 1.2)
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "ab_initio", "particle_diameter"], scaled_diameter),
                (["workflow", "refinement", "particle_diameter"], scaled_diameter),
            ],
        )

    return updates_made


def _bulk_update(config_data: Dict[str, Any], path_value_pairs: Iterable[tuple[List[str], Any]]) -> bool:
    """Apply multiple nested updates and return True if any changes were made."""
    updated = False
    for path, value in path_value_pairs:
        if value is _UNSET:
            continue
        if _update_nested_value(config_data, path, value):
            updated = True
    return updated


def _update_nested_value(data: Dict[str, Any], path: Sequence[str], value: Any) -> bool:
    """Set a nested dictionary value, returning True if the value changed."""
    current = data
    for key in path[:-1]:
        if not isinstance(current, dict):
            return False
        current = current.setdefault(key, {})
    final_key = path[-1]
    if not isinstance(current, dict):
        return False
    if current.get(final_key) == value:
        return False
    current[final_key] = value
    return True


def _compute_nearest_box_size(diameter: Optional[float], pixel_size: Optional[float]) -> Optional[int]:
    if not diameter or not pixel_size:
        return None
    raw_size = (diameter / pixel_size) + 125.0
    try:
        from ..core.base_react_agent import BaseReActAgent

        allowed_sizes: Sequence[int] = getattr(BaseReActAgent, "_ALLOWED_BOX_SIZES", ())
    except Exception:  # pragma: no cover - defensive fallback
        allowed_sizes = ()

    if not allowed_sizes:
        return int(round(raw_size))

    return min(allowed_sizes, key=lambda candidate: abs(candidate - raw_size))


def _round_sig(value: float, digits: int = 3) -> float:
    """Round a float to a sensible precision while preserving integers."""
    rounded = round(value, digits)
    if rounded.is_integer():
        return int(rounded)
    return rounded


def _scale_value(value: Optional[float], factor: float) -> Optional[float]:
    """Scale a microscope parameter (e.g., diameter) by a given factor with rounding."""
    if value is None:
        return None
    return _round_sig(value * factor)


def _value_or_none(value: MicroscopeValue) -> Optional[Union[str, float, int]]:
    if value is _UNSET:
        return None
    return value


def apply_cryosift_overrides_if_enabled(base_path: Optional[Path] = None) -> None:
    """
    Apply CryoSift configuration from master_config.json to relevant stage configuration files.
    
    This propagates cryosift_weights_path and cryosift_env from master_config.json
    to optimization_2d_config.json and particle_picking_config.json.
    
    Args:
        base_path: Optional base directory of the project workspace. Defaults to CWD.
    """
    base_dir = Path(base_path).resolve() if base_path else Path.cwd()
    master_config_path = base_dir / "configs" / "master_config.json"
    
    master_config = _load_json(master_config_path)
    if not master_config:
        return
    
    # Check if cryosift section exists
    cryosift_config = master_config.get("cryosift", {})
    if not cryosift_config:
        LOGGER.debug("CryoSift configuration not found in master_config.json.")
        return
    
    # Extract CryoSift parameters
    cryosift_weights_path = cryosift_config.get("cryosift_weights_path")
    cryosift_env = cryosift_config.get("cryosift_env")
    
    # If neither is set, nothing to propagate
    if not cryosift_weights_path and not cryosift_env:
        LOGGER.debug("No CryoSift parameters to propagate from master_config.json.")
        return
    
    # Stage configuration files to update
    stage_config_paths = {
        "optimization_2d": base_dir / "configs" / "cryosparc" / "optimization_2d_config.json",
        "particle_picking": base_dir / "configs" / "cryosparc" / "particle_picking_config.json",
    }
    
    for stage_name, cfg_path in stage_config_paths.items():
        try:
            _apply_cryosift_overrides_to_stage(stage_name, cfg_path, cryosift_weights_path, cryosift_env)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Failed to apply CryoSift overrides to %s: %s", cfg_path, exc)


def _apply_cryosift_overrides_to_stage(
    stage_name: str,
    cfg_path: Path,
    cryosift_weights_path: Optional[str],
    cryosift_env: Optional[str]
) -> None:
    """Apply CryoSift overrides to a specific stage configuration file."""
    config_data = _load_json(cfg_path)
    if config_data is None:
        return
    
    updated = False
    if stage_name == "optimization_2d":
        updated = _update_optimization_2d_cryosift(config_data, cryosift_weights_path, cryosift_env)
    elif stage_name == "particle_picking":
        updated = _update_particle_picking_cryosift(config_data, cryosift_weights_path, cryosift_env)
    
    if updated:
        _save_json(cfg_path, config_data)
        LOGGER.info("Applied CryoSift overrides to %s", cfg_path)


def _update_optimization_2d_cryosift(
    config_data: Dict[str, Any],
    cryosift_weights_path: Optional[str],
    cryosift_env: Optional[str]
) -> bool:
    """Update CryoSift settings in optimization_2d_config.json."""
    paths_to_update = []
    
    if cryosift_weights_path is not None:
        paths_to_update.append(
            (["workflow", "2d_optimization", "select_2d_classes", "cryosift_weights_path"], cryosift_weights_path)
        )
    
    if cryosift_env is not None:
        paths_to_update.append(
            (["workflow", "2d_optimization", "select_2d_classes", "cryosift_env"], cryosift_env)
        )
    
    return _bulk_update(config_data, paths_to_update)


def _update_particle_picking_cryosift(
    config_data: Dict[str, Any],
    cryosift_weights_path: Optional[str],
    cryosift_env: Optional[str]
) -> bool:
    """Update CryoSift settings in particle_picking_config.json."""
    paths_to_update = []
    
    if cryosift_weights_path is not None:
        paths_to_update.append(
            (["workflow", "select_2d_classes", "cryosift_weights_path"], cryosift_weights_path)
        )
    
    if cryosift_env is not None:
        paths_to_update.append(
            (["workflow", "select_2d_classes", "cryosift_env"], cryosift_env)
        )
    
    return _bulk_update(config_data, paths_to_update)


