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
from typing import Any, Dict, Iterable, List, Optional, Sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MicroscopeParameters:
    """Container for microscope override parameters."""

    movies_path: Optional[str] = None
    gain_ref_path: Optional[str] = None
    gain_rot: Optional[int] = None
    gain_flip: Optional[int] = None
    pixel_size: Optional[float] = None
    voltage: Optional[float] = None
    cs_mm: Optional[float] = None
    dose: Optional[float] = None
    particle_diameter: Optional[float] = None
    symmetry: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        """Return True if the critical parameters are present."""
        return self.pixel_size is not None and self.particle_diameter is not None


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
        movies_path=raw.get("movies_path"),
        gain_ref_path=raw.get("gain_ref_path"),
        gain_rot=_safe_int(raw.get("gain_rot")),
        gain_flip=_safe_int(raw.get("gain_flip")),
        pixel_size=_safe_float(raw.get("pixel_size")),
        voltage=_safe_float(raw.get("voltage")),
        cs_mm=_safe_float(raw.get("cs_mm")),
        dose=_safe_float(raw.get("dose")),
        particle_diameter=_safe_float(raw.get("particle_diameter")),
        symmetry=_safe_str(raw.get("symmetry")),
    )


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

    diameter = overrides.particle_diameter
    if diameter is not None:
        log_min = _round_sig(diameter * 0.7)
        log_max = _round_sig(diameter * 1.3)
        class_diameter = _round_sig(diameter * 1.2)
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
        box_size = _compute_nearest_box_size(overrides.particle_diameter, overrides.pixel_size)
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

    if overrides.particle_diameter is not None:
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "blob_picker", "particle_diameter"], _round_sig(overrides.particle_diameter)),
            ],
        )

    if overrides.is_complete:
        box_size = _compute_nearest_box_size(overrides.particle_diameter, overrides.pixel_size)
        if box_size is not None:
            desc = (
                f"Extraction box size in pixels. Calculated as "
                f"{overrides.particle_diameter}/{overrides.pixel_size} + 125 ≈ {box_size}."
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

    symmetry = overrides.symmetry
    if symmetry is not None:
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "ab_initio_reconstruction", "align_sym"], symmetry),
                (["workflow", "refinement_3d", "sym"], symmetry),
            ],
        )

    if overrides.particle_diameter is not None:
        scaled_diameter = _round_sig(overrides.particle_diameter * 1.2)
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "ab_initio_reconstruction", "particle_diameter"], scaled_diameter),
                (["workflow", "refinement_3d", "particle_diameter"], scaled_diameter),
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

    symmetry = overrides.symmetry
    if symmetry is not None:
        updates_made |= _bulk_update(
            config_data,
            [
                (["workflow", "ab_initio", "symmetry"], symmetry),
                (["workflow", "refinement", "symmetry"], symmetry),
            ],
        )

    if overrides.particle_diameter is not None:
        scaled_diameter = _round_sig(overrides.particle_diameter * 1.2)
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
        if value is None:
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


