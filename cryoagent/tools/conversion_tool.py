"""Unified conversion helpers bridging CryoSPARC, RELION, and Parquet outputs."""

from __future__ import annotations

import math
import os
import uuid
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants & Shared Helpers (from previous parquet_export module)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0.0"


def _safe_numeric(series: Optional[pd.Series], *, dtype=float) -> Optional[pd.Series]:
    if series is None:
        return None
    try:
        return pd.to_numeric(series, errors="coerce").astype(dtype)
    except Exception:  # pragma: no cover - fallback path
        return pd.to_numeric(series, errors="coerce")


def _basename(series: pd.Series) -> pd.Series:
    return series.astype(str).apply(lambda p: os.path.basename(str(p)))


def _deg_from_rad(series: Optional[pd.Series]) -> Optional[pd.Series]:
    if series is None:
        return None
    return pd.to_numeric(series, errors="coerce") * (180.0 / np.pi)


def _ensure_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: Dict[str, int] = {}
    new_cols: List[str] = []
    for col in df.columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}__dup{seen[col]}")
    if new_cols != list(df.columns):
        df = df.copy()
        df.columns = new_cols
    return df


# ---------------------------------------------------------------------------
# CryoSPARC -> RELION conversion utilities (from previous file_conversion_tools)
# ---------------------------------------------------------------------------


class FileConversionTools:
    """Utility functions to convert CryoSPARC metadata into RELION STAR files."""

    def __init__(self) -> None:
        pass

    def read_cs_file(self, cs_path: Union[str, Path]) -> pd.DataFrame:
        cs_path = Path(cs_path)
        if not cs_path.exists():
            raise FileNotFoundError(f"CryoSPARC file not found: {cs_path}")

        structured = np.load(cs_path, allow_pickle=True)
        if structured.dtype.names is None:
            raise ValueError("Unsupported CryoSPARC file: missing named fields")

        data: Dict[str, Any] = {}
        for name in structured.dtype.names:
            column = structured[name]
            if isinstance(column, np.ndarray) and column.ndim > 1:
                data[name] = list(column)
            else:
                data[name] = column
        df = pd.DataFrame(data)
        return df

    def convert_cs_to_star(
        self,
        cs_path: Union[str, Path],
        star_path: Union[str, Path],
        passthrough_path: Optional[Union[str, Path]] = None,
        job_directory: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Convert CryoSparc .cs file to Relion STAR file.
        
        Automatically detects if the file is a particle file or exposure/micrograph file
        and uses the appropriate conversion method.
        
        Args:
            cs_path: Path to CryoSparc .cs file
            star_path: Path to output Relion STAR file
            passthrough_path: Optional path to passthrough file
            job_directory: Optional job directory for resolving relative micrograph paths
        """
        df_raw = self.read_cs_file(cs_path)
        
        # Check if this is an exposure/micrograph file (has micrograph_blob/path)
        # or a particle file (has blob/path)
        is_exposure = (
            "micrograph_blob/path" in df_raw.columns or
            "micrograph_blob_non_dw/path" in df_raw.columns
        )
        
        if is_exposure:
            # Use micrograph conversion method
            return self.convert_cs_micrographs_to_star(cs_path, star_path, passthrough_path, job_directory)
        
        # Original particle conversion logic
        if passthrough_path is None:
            cs_path_obj = Path(cs_path)
            passthrough_candidates = [
                cs_path_obj.parent / "J57_passthrough_particles.cs",
                cs_path_obj.parent / f"{cs_path_obj.stem}_passthrough_particles.cs",
                cs_path_obj.parent / f"{cs_path_obj.stem}_passthrough.cs",
            ]
            for candidate in passthrough_candidates:
                if candidate.exists():
                    passthrough_path = candidate
                    break

        if passthrough_path and Path(passthrough_path).exists():
            try:
                df_passthrough = self.read_cs_file(passthrough_path)
                print(f"Loading location information from: {passthrough_path}")
                df_raw = self._merge_location_data(df_raw, df_passthrough)
            except Exception as exc:  # pragma: no cover - logging path
                print(f"Warning: Could not load passthrough file {passthrough_path}: {exc}")

        particles, optics = self._build_relion_tables(df_raw)
        particles.attrs["optics"] = optics
        star_path = Path(star_path)
        star_path.parent.mkdir(parents=True, exist_ok=True)
        self._dataframe_to_star(particles, star_path, format="v3")
        return star_path

    def convert_cs_micrographs_to_star(
        self,
        cs_path: Union[str, Path],
        star_path: Union[str, Path],
        passthrough_path: Optional[Union[str, Path]] = None,
        job_directory: Optional[Union[str, Path]] = None,
        binning_factor: Optional[float] = None,
    ) -> Path:
        """Convert CryoSparc exposure/micrograph .cs file to Relion micrographs STAR file."""
        df_exposure = self.read_cs_file(cs_path)
        
        # Find passthrough file if not provided
        if passthrough_path is None:
            cs_path_obj = Path(cs_path)
            passthrough_candidates = [
                cs_path_obj.parent / f"{cs_path_obj.stem}_passthrough.cs",
                cs_path_obj.parent / f"{cs_path_obj.stem}_passthrough_exposures_accepted.cs",
                cs_path_obj.parent / "J57_passthrough_exposures_accepted.cs",
            ]
            for candidate in passthrough_candidates:
                if candidate.exists():
                    passthrough_path = candidate
                    break
        
        # Load passthrough data which contains CTF info and micrograph paths
        df_passthrough = None
        if passthrough_path and Path(passthrough_path).exists():
            try:
                df_passthrough = self.read_cs_file(passthrough_path)
                print(f"Loading passthrough data from: {passthrough_path}")
            except Exception as exc:
                print(f"Warning: Could not load passthrough file {passthrough_path}: {exc}")
        
        # Use passthrough data if available (has more complete info), otherwise use exposure data
        df_raw = df_passthrough if df_passthrough is not None else df_exposure
        
        # Get micrograph paths - try different field names
        micrograph_path_field = None
        for field in ["micrograph_blob_non_dw/path", "micrograph_blob/path", "micrograph_thumbnail_blob_1x/micrograph_path"]:
            if field in df_raw.columns:
                micrograph_path_field = field
                break
        
        if micrograph_path_field is None:
            raise ValueError("Could not find micrograph path field in CryoSparc data")
        
        micrograph_paths = df_raw[micrograph_path_field].apply(self._decode_bytes)
        n_micrographs = len(df_raw)
        
        # Convert paths to absolute paths
        # CryoSparc paths are typically relative to the job directory or project directory
        cs_path_obj = Path(cs_path)
        base_dir = Path(job_directory) if job_directory else cs_path_obj.parent
        
        def resolve_micrograph_path(path_str: str) -> str:
            """Resolve micrograph path to absolute path."""
            path = Path(path_str)
            # If already absolute, return as is
            if path.is_absolute():
                return str(path)
            # Try relative to job directory first
            full_path = base_dir / path
            if full_path.exists():
                return str(full_path.resolve())
            # Try relative to parent (project directory)
            full_path = base_dir.parent / path
            if full_path.exists():
                return str(full_path.resolve())
            # If not found, return absolute path based on job directory
            return str((base_dir / path).resolve())
        
        micrograph_paths_absolute = micrograph_paths.apply(resolve_micrograph_path)
        
        # Build micrographs data
        micrograph_data: Dict[str, Any] = {
            "rlnMicrographName": micrograph_paths_absolute.tolist(),
        }
        
        # Pixel size
        psize_field = None
        for field in ["micrograph_blob_non_dw/psize_A", "micrograph_blob/psize_A", "micrograph_thumbnail_blob_1x/psize_A"]:
            if field in df_raw.columns:
                psize_field = field
                break
        
        if psize_field:
            pixel_size = pd.to_numeric(df_raw[psize_field], errors="coerce")
            micrograph_data["rlnMicrographOriginalPixelSize"] = pixel_size
        else:
            micrograph_data["rlnMicrographOriginalPixelSize"] = pd.Series([np.nan] * n_micrographs)
        
        # Get binning factor from motion correction
        # The binning factor affects the pixel size: if binning is 1, pixel size is unchanged
        # If binning > 1, the effective pixel size increases (pixel size = original / binning_factor)
        
        # Use provided binning_factor if available, otherwise default to 1.0
        if binning_factor is None:
            binning_factor = 1.0
        
        # Calculate rlnMicrographPixelSize
        # If binning factor is 1, it equals rlnMicrographOriginalPixelSize
        # Otherwise, it's rlnMicrographOriginalPixelSize / binning_factor
        original_pixel_size = micrograph_data["rlnMicrographOriginalPixelSize"]
        if binning_factor == 1.0:
            micrograph_data["rlnMicrographPixelSize"] = original_pixel_size.copy()
        else:
            micrograph_data["rlnMicrographPixelSize"] = original_pixel_size / binning_factor
        
        # CTF parameters
        defocus_u = df_raw.get("ctf/df1_A")
        defocus_v = df_raw.get("ctf/df2_A")
        micrograph_data["rlnDefocusU"] = pd.to_numeric(defocus_u, errors="coerce") if defocus_u is not None else pd.Series([np.nan] * n_micrographs)
        micrograph_data["rlnDefocusV"] = pd.to_numeric(defocus_v, errors="coerce") if defocus_v is not None else pd.Series([np.nan] * n_micrographs)
        
        defocus_angle_series = df_raw.get("ctf/df_angle_rad")
        if defocus_angle_series is not None:
            micrograph_data["rlnDefocusAngle"] = np.degrees(pd.to_numeric(defocus_angle_series, errors="coerce"))
        else:
            micrograph_data["rlnDefocusAngle"] = pd.Series([0.0] * n_micrographs)
        
        phase_shift_series = df_raw.get("ctf/phase_shift_rad")
        if phase_shift_series is not None:
            micrograph_data["rlnPhaseShift"] = np.degrees(pd.to_numeric(phase_shift_series, errors="coerce"))
        else:
            micrograph_data["rlnPhaseShift"] = pd.Series([0.0] * n_micrographs)
        
        # Microscope parameters
        voltage_series = df_raw.get("ctf/accel_kv")
        if voltage_series is None:
            voltage_series = df_raw.get("mscope_params/accel_kv")
        
        cs_series = df_raw.get("ctf/cs_mm")
        if cs_series is None:
            cs_series = df_raw.get("mscope_params/cs_mm")
        
        amp_series = df_raw.get("ctf/amp_contrast")
        
        micrograph_data["rlnVoltage"] = pd.to_numeric(voltage_series, errors="coerce") if voltage_series is not None else pd.Series([np.nan] * n_micrographs)
        micrograph_data["rlnSphericalAberration"] = pd.to_numeric(cs_series, errors="coerce") if cs_series is not None else pd.Series([np.nan] * n_micrographs)
        micrograph_data["rlnAmplitudeContrast"] = pd.to_numeric(amp_series, errors="coerce") if amp_series is not None else pd.Series([np.nan] * n_micrographs)
        
        # Optics group
        optics_group_series = df_raw.get("ctf/exp_group_id")
        if optics_group_series is None:
            optics_group_series = df_raw.get("mscope_params/exp_group_id")
        if optics_group_series is None:
            optics_group_series = pd.Series(np.ones(n_micrographs, dtype=int))
        else:
            optics_group_series = pd.to_numeric(optics_group_series, errors="coerce").fillna(1).astype(int)
        micrograph_data["rlnOpticsGroup"] = optics_group_series
        
        # CTF quality metrics
        ctf_fit_to_A = df_raw.get("ctf/ctf_fit_to_A")
        if ctf_fit_to_A is None:
            ctf_fit_to_A = df_raw.get("ctf_stats/ctf_fit_to_A")
        if ctf_fit_to_A is not None:
            micrograph_data["rlnCtfMaxResolution"] = pd.to_numeric(ctf_fit_to_A, errors="coerce")
        else:
            micrograph_data["rlnCtfMaxResolution"] = pd.Series([np.nan] * n_micrographs)
        
        ctf_fom = df_raw.get("ctf/fig_of_merit_gctf")
        if ctf_fom is None:
            ctf_fom = df_raw.get("ctf/cross_corr_ctffind4")
        if ctf_fom is not None:
            micrograph_data["rlnCtfFigureOfMerit"] = pd.to_numeric(ctf_fom, errors="coerce")
        else:
            micrograph_data["rlnCtfFigureOfMerit"] = pd.Series([np.nan] * n_micrographs)
        
        # Create DataFrame
        micrographs_df = pd.DataFrame(micrograph_data)
        print(f"Loaded {len(micrographs_df)} micrographs")
        
        # Create optics table
        optics_records: List[Dict[str, Any]] = []
        for group in sorted(optics_group_series.unique()):
            mask = optics_group_series == group
            optics_records.append({
                "rlnOpticsGroupName": f"opticsGroup{group}",
                "rlnOpticsGroup": int(group),
                "rlnMicrographOriginalPixelSize": float(self._first_valid(micrographs_df.loc[mask, "rlnMicrographOriginalPixelSize"], np.nan)),
                "rlnMicrographPixelSize": float(self._first_valid(micrographs_df.loc[mask, "rlnMicrographPixelSize"], np.nan)),
                "rlnVoltage": float(self._first_valid(micrographs_df.loc[mask, "rlnVoltage"], np.nan)),
                "rlnSphericalAberration": float(self._first_valid(micrographs_df.loc[mask, "rlnSphericalAberration"], np.nan)),
                "rlnAmplitudeContrast": float(self._first_valid(micrographs_df.loc[mask, "rlnAmplitudeContrast"], np.nan)),
            })
        
        optics_df = pd.DataFrame(optics_records)
        micrographs_df.attrs["optics"] = optics_df
        
        # Write STAR file with micrographs data type
        star_path = Path(star_path)
        star_path.parent.mkdir(parents=True, exist_ok=True)
        self._dataframe_to_star(micrographs_df, star_path, format="v3", data_type="micrographs")
        
        return star_path
    
    def _merge_location_data(self, df_main: pd.DataFrame, df_passthrough: pd.DataFrame) -> pd.DataFrame:
        df_merged = df_main.copy()
        # Placeholder for potential merges (kept for future extension)
        return df_merged

    def _build_relion_tables(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        n_particles = len(df_raw)
        image_paths = df_raw.get("blob/path")
        if image_paths is None:
            raise ValueError("CryoSPARC metadata is missing 'blob/path'")
        image_paths = image_paths.apply(self._decode_bytes)

        image_indices = df_raw.get("blob/idx")
        if image_indices is None:
            image_indices = pd.Series(np.arange(n_particles))
        else:
            image_indices = pd.to_numeric(image_indices, errors="coerce").fillna(0).astype(int)

        image_names = [f"{i:08d}@{path}" for i, path in zip(image_indices + 1, image_paths)]
        micrograph_names = [PurePath(path).name for path in image_paths]

        pixel_series = df_raw.get("blob/psize_A")
        pixel_size = pd.to_numeric(pixel_series, errors="coerce") if pixel_series is not None else pd.Series([np.nan] * n_particles)

        autopick_fom: Optional[pd.Series] = None
        for field_name in [
            "pick_stats/ncc_score",
            "pick_stats/power",
            "alignments3D/cross_cor",
            "alignments2D/cross_cor",
        ]:
            if field_name in df_raw:
                autopick_fom = pd.to_numeric(df_raw.get(field_name), errors="coerce")
                break
        if autopick_fom is None:
            autopick_fom = pd.Series([1.0] * n_particles)

        particle_data: Dict[str, Any] = {
            "rlnImageName": image_names,
            "rlnMicrographName": micrograph_names,
            "rlnImagePixelSize": pixel_size,
            "rlnMicrographOriginalPixelSize": pixel_size,
            "rlnAutopickFigureOfMerit": autopick_fom,
        }

        class_series = None
        for field in ("alignments3D/class", "alignments2D/class", "pick_stats/template_idx"):
            if field in df_raw:
                class_series = df_raw.get(field)
                break
        if class_series is None:
            particle_data["rlnClassNumber"] = pd.Series(np.ones(n_particles, dtype=int))
        else:
            class_values = pd.to_numeric(class_series, errors="coerce").fillna(0)
            particle_data["rlnClassNumber"] = class_values.round().astype(int) + 1

        pose_series = None
        for field_name in ["alignments3D/pose", "alignments3D_multi/pose"]:
            if field_name in df_raw:
                pose_series = df_raw.get(field_name)
                break

        if pose_series is not None:
            def extract_pose_angles(x):
                if isinstance(x, (list, tuple, np.ndarray)):
                    if len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray)) and len(x[0]) >= 3:
                        return x[0][:3]
                    if len(x) >= 3:
                        return x[:3]
                return [0, 0, 0]

            pose_data = pose_series.apply(extract_pose_angles)
            particle_data["rlnAngleRot"] = pose_data.apply(lambda x: np.degrees(x[0]) if len(x) >= 1 else 0)
            particle_data["rlnAngleTilt"] = pose_data.apply(lambda x: np.degrees(x[1]) if len(x) >= 2 else 0)
            particle_data["rlnAnglePsi"] = pose_data.apply(lambda x: np.degrees(x[2]) if len(x) >= 3 else 0)
        else:
            angle_series = df_raw.get("pick_stats/angle_rad")
            if angle_series is not None:
                angle_rad = pd.to_numeric(angle_series, errors="coerce")
                particle_data["rlnAnglePsi"] = np.degrees(angle_rad)
            else:
                particle_data["rlnAnglePsi"] = 0.0
            particle_data["rlnAngleRot"] = 0.0
            particle_data["rlnAngleTilt"] = 0.0

        defocus_u = df_raw.get("ctf/df1_A")
        defocus_v = df_raw.get("ctf/df2_A")
        particle_data["rlnDefocusU"] = pd.to_numeric(defocus_u, errors="coerce") if defocus_u is not None else np.nan
        particle_data["rlnDefocusV"] = pd.to_numeric(defocus_v, errors="coerce") if defocus_v is not None else np.nan

        defocus_angle_series = df_raw.get("ctf/df_angle_rad")
        if defocus_angle_series is not None:
            defocus_angle = pd.to_numeric(defocus_angle_series, errors="coerce")
            particle_data["rlnDefocusAngle"] = np.degrees(defocus_angle)

        phase_shift_series = df_raw.get("ctf/phase_shift_rad")
        if phase_shift_series is not None:
            phase_shift = pd.to_numeric(phase_shift_series, errors="coerce")
            particle_data["rlnPhaseShift"] = np.degrees(phase_shift)

        ctf_bfactor = df_raw.get("ctf/bfactor")
        if ctf_bfactor is not None:
            particle_data["rlnCtfBfactor"] = pd.to_numeric(ctf_bfactor, errors="coerce")
        ctf_scale = df_raw.get("ctf/scale")
        if ctf_scale is not None:
            particle_data["rlnCtfFigureOfMerit"] = pd.to_numeric(ctf_scale, errors="coerce")

        center_x_frac = df_raw.get("location/center_x_frac")
        center_y_frac = df_raw.get("location/center_y_frac")
        micrograph_shape = df_raw.get("location/micrograph_shape")
        micrograph_psize = df_raw.get("location/micrograph_psize_A")

        if center_x_frac is not None and center_y_frac is not None and micrograph_shape is not None and micrograph_psize is not None:
            center_x_pix = center_x_frac * micrograph_shape.apply(lambda x: x[0] if len(x) > 0 else 0)
            center_y_pix = center_y_frac * micrograph_shape.apply(lambda x: x[1] if len(x) > 0 else 0)
            center_x_angst = center_x_pix * micrograph_psize
            center_y_angst = center_y_pix * micrograph_psize
            particle_data["rlnCoordinateX"] = center_x_pix
            particle_data["rlnCoordinateY"] = center_y_pix
            particle_data["rlnCoordinateXAngst"] = center_x_angst
            particle_data["rlnCoordinateYAngst"] = center_y_angst
        else:
            shift_series = df_raw.get("ctf/shift_A")
            if shift_series is not None:
                particle_data["rlnCoordinateX"] = shift_series.apply(lambda val: self._extract_shift_component(val, axis=0))
                particle_data["rlnCoordinateY"] = shift_series.apply(lambda val: self._extract_shift_component(val, axis=1))
                particle_data["rlnCoordinateXAngst"] = particle_data["rlnCoordinateX"]
                particle_data["rlnCoordinateYAngst"] = particle_data["rlnCoordinateY"]
            else:
                particle_data["rlnCoordinateX"] = 0.0
                particle_data["rlnCoordinateY"] = 0.0
                particle_data["rlnCoordinateXAngst"] = 0.0
                particle_data["rlnCoordinateYAngst"] = 0.0

        shift_series = df_raw.get("ctf/shift_A")
        if shift_series is not None:
            particle_data["rlnOriginX"] = shift_series.apply(lambda val: self._extract_shift_component(val, axis=0))
            particle_data["rlnOriginY"] = shift_series.apply(lambda val: self._extract_shift_component(val, axis=1))
            particle_data["rlnOriginXAngst"] = particle_data["rlnOriginX"]
            particle_data["rlnOriginYAngst"] = particle_data["rlnOriginY"]
        else:
            particle_data["rlnOriginX"] = 0.0
            particle_data["rlnOriginY"] = 0.0
            particle_data["rlnOriginXAngst"] = 0.0
            particle_data["rlnOriginYAngst"] = 0.0

        optics_group_series = df_raw.get("ctf/exp_group_id")
        if optics_group_series is None:
            optics_group_series = pd.Series(np.ones(n_particles, dtype=int))
        else:
            optics_group_series = pd.to_numeric(optics_group_series, errors="coerce").fillna(1).astype(int)
        particle_data["rlnOpticsGroup"] = optics_group_series
        particle_data["rlnOpticsGroupName"] = optics_group_series.apply(lambda g: f"opticsGroup{g}")
        particle_data["rlnGroupNumber"] = optics_group_series

        particles = pd.DataFrame(particle_data)
        print(len(particles), "particles is being loaded")

        optics_records: List[Dict[str, Any]] = []
        for group in sorted(optics_group_series.unique()):
            mask = optics_group_series == group
            accel_series = df_raw.get("ctf/accel_kv")
            accel_series = pd.to_numeric(accel_series, errors="coerce") if accel_series is not None else None
            cs_series = df_raw.get("ctf/cs_mm")
            cs_series = pd.to_numeric(cs_series, errors="coerce") if cs_series is not None else None
            amp_series = df_raw.get("ctf/amp_contrast")
            amp_series = pd.to_numeric(amp_series, errors="coerce") if amp_series is not None else None
            shape_series = df_raw.get("blob/shape")
            if shape_series is not None:
                image_size = shape_series.apply(lambda val: self._extract_shape_dim(val, axis=0))
            else:
                image_size = None
            optics_records.append(
                {
                    "rlnOpticsGroupName": f"opticsGroup{group}",
                    "rlnOpticsGroup": int(group),
                    "rlnMicrographOriginalPixelSize": float(self._first_valid(particles.loc[mask, "rlnMicrographOriginalPixelSize"], np.nan)),
                    "rlnVoltage": float(self._first_valid(accel_series, np.nan)),
                    "rlnSphericalAberration": float(self._first_valid(cs_series, np.nan)),
                    "rlnAmplitudeContrast": float(self._first_valid(amp_series, np.nan)),
                    "rlnImagePixelSize": float(self._first_valid(particles.loc[mask, "rlnImagePixelSize"], np.nan)),
                    "rlnImageSize": float(self._first_valid(image_size, np.nan)),
                    "rlnImageDimensionality": 2,
                    "rlnCtfDataAreCtfPremultiplied": 0,
                }
            )

        optics = pd.DataFrame(optics_records)
        return particles, optics

    def _dataframe_to_star(
        self,
        data: pd.DataFrame,
        star_file: Union[str, Path],
        format: str = "v3",
        data_type: Optional[str] = None,
    ) -> None:
        """Write DataFrame to Relion STAR file.
        
        Args:
            data: DataFrame with data to write
            star_file: Output STAR file path
            format: STAR file format version (default: "v3")
            data_type: Type of data block - "particles" or "micrographs". 
                      If None, will auto-detect from columns.
        """
        data = data.copy()
        optics = data.attrs.get("optics")
        
        # Auto-detect data type if not specified
        if data_type is None:
            # Check if this looks like micrograph data (has rlnMicrographName but not rlnImageName)
            has_micrograph_name = "rlnMicrographName" in data.columns
            has_image_name = "rlnImageName" in data.columns
            if has_micrograph_name and not has_image_name:
                data_type = "micrographs"
            else:
                data_type = "particles"
        
        with open(star_file, "w", encoding="utf-8") as handle:
            handle.write("# version 30001\n\n")
            if optics is not None and len(optics) > 0:
                handle.write("data_optics\n\n")
                handle.write("loop_\n")
                optics_columns = [c for c in optics.columns if c.startswith("rln")]
                for idx, col in enumerate(optics_columns, start=1):
                    handle.write(f"_{col} #{idx}\n")
                for _, row in optics.iterrows():
                    values = [self._format_star_value(row[col]) for col in optics_columns]
                    handle.write(" ".join(values) + "\n")
                handle.write("\n")

            handle.write(f"data_{data_type}\n\n")
            handle.write("loop_\n")
            data_columns = [c for c in data.columns if c.startswith("rln")]
            for idx, col in enumerate(data_columns, start=1):
                handle.write(f"_{col} #{idx}\n")
            for _, row in data.iterrows():
                values = [self._format_star_value(row[col]) for col in data_columns]
                handle.write(" ".join(values) + "\n")

    @staticmethod
    def _decode_bytes(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, np.bytes_):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    @staticmethod
    def _extract_shape_dim(shape: Any, axis: int = 0) -> float:
        if isinstance(shape, np.ndarray):
            seq = shape.tolist()
        elif isinstance(shape, (list, tuple)):
            seq = list(shape)
        else:
            return np.nan
        if 0 <= axis < len(seq):
            try:
                return float(seq[axis])
            except (TypeError, ValueError):
                return np.nan
        return np.nan

    @staticmethod
    def _extract_shift_component(shift: Any, axis: int = 0) -> float:
        if isinstance(shift, np.ndarray):
            seq = shift.tolist()
        elif isinstance(shift, (list, tuple)):
            seq = list(shift)
        else:
            return 0.0
        if 0 <= axis < len(seq):
            try:
                return float(seq[axis])
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    @staticmethod
    def _format_star_value(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "?"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _first_valid(series: Optional[pd.Series], default: Any) -> Any:
        if series is None:
            return default
        valid = series.dropna() if hasattr(series, "dropna") else []
        if isinstance(valid, pd.Series) and not valid.empty:
            return valid.iloc[0]
        return default


# ---------------------------------------------------------------------------
# RELION STAR reader and unified schema exporters (from previous parquet_export)
# ---------------------------------------------------------------------------


def read_star_particles(star_path: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    with open(star_path, "r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    particles_df: Optional[pd.DataFrame] = None
    optics_df: Optional[pd.DataFrame] = None
    current_block = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("data_"):
            if "optics" in line:
                current_block = "optics"
            elif "particles" in line:
                current_block = "particles"
            else:
                current_block = None
            i += 1
            continue
        if line.startswith("loop_"):
            i += 1
            df, i = _parse_star_loop(lines, i)
            if current_block == "optics":
                optics_df = df
            elif current_block == "particles":
                particles_df = df
            continue
        i += 1

    if particles_df is None:
        raise ValueError(f"No data_particles block found in STAR file: {star_path}")
    return particles_df, optics_df


def _parse_star_loop(lines: List[str], start_idx: int) -> Tuple[pd.DataFrame, int]:
    headers: List[str] = []
    data: List[List[str]] = []
    i = start_idx
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("_"):
            headers.append(line.split()[0][1:])
            i += 1
            continue
        break
    while i < len(lines):
        line = lines[i].strip()
        if (not line) or line.startswith("data_") or line.startswith("loop_"):
            break
        parts = line.split()
        if len(parts) < len(headers):
            parts += [""] * (len(headers) - len(parts))
        elif len(parts) > len(headers):
            parts = parts[: len(headers)]
        data.append(parts)
        i += 1
    df = pd.DataFrame(data, columns=headers)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df, i


def unify_particles_schema_from_cs(
    df_cs: pd.DataFrame,
    *,
    dataset_id: str,
    stage: str,
    agent: str,
    backend: str,
    session_id: str,
    job_uid: str,
    prefix_raw: str = "cs__",
) -> pd.DataFrame:
    src = df_cs.copy()
    src.columns = [f"{prefix_raw}{c}" for c in src.columns]

    img_path = src.get(f"{prefix_raw}blob/path").astype(str)
    img_idx = _safe_numeric(src.get(f"{prefix_raw}blob/idx"), dtype=int)
    psize = _safe_numeric(src.get(f"{prefix_raw}blob/psize_A"))

    cx_frac = src.get(f"{prefix_raw}location/center_x_frac")
    cy_frac = src.get(f"{prefix_raw}location/center_y_frac")
    mshape = src.get(f"{prefix_raw}location/micrograph_shape")
    mpsize = _safe_numeric(src.get(f"{prefix_raw}location/micrograph_psize_A"))

    if cx_frac is not None and cy_frac is not None and mshape is not None:
        width = mshape.apply(lambda s: (s[0] if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 0 else np.nan))
        height = mshape.apply(lambda s: (s[1] if isinstance(s, (list, tuple, np.ndarray)) and len(s) > 1 else np.nan))
        coord_x_pix = _safe_numeric(cx_frac) * width
        coord_y_pix = _safe_numeric(cy_frac) * height
        micrograph_pixel_size_A = mpsize
    else:
        coord_x_pix = None
        coord_y_pix = None
        micrograph_pixel_size_A = psize

    angle_psi_deg = _deg_from_rad(src.get(f"{prefix_raw}pick_stats/angle_rad"))

    def _class_num() -> Optional[pd.Series]:
        for field in ("alignments3D/class", "alignments2D/class", "pick_stats/template_idx"):
            s = src.get(f"{prefix_raw}{field}")
            if s is not None:
                v = pd.to_numeric(s, errors="coerce").fillna(0)
                return (v.round().astype(int) + 1)
        return None

    micrograph_uid = src.get(f"{prefix_raw}location/micrograph_uid")

    out = pd.DataFrame({
        "common__dataset_id": dataset_id,
        "common__stage": stage,
        "common__agent": agent,
        "common__backend": backend,
        "common__session_id": session_id,
        "common__job_uid": job_uid,
        "common__schema_version": SCHEMA_VERSION,
        "common__micrograph_path": img_path,
        "common__micrograph_basename": _basename(img_path) if img_path is not None else None,
        "common__micrograph_id": (micrograph_uid if micrograph_uid is not None else (_basename(img_path) if img_path is not None else None)),
        "common__image_name": img_path,
        "common__image_idx": img_idx,
        "common__coord_x_pix": coord_x_pix,
        "common__coord_y_pix": coord_y_pix,
        "common__origin_x_pix": 0.0,
        "common__origin_y_pix": 0.0,
        "common__pixel_size_A": psize,
        "common__micrograph_pixel_size_A": micrograph_pixel_size_A,
        "common__pick_score": _safe_numeric(src.get(f"{prefix_raw}pick_stats/ncc_score")),
        "common__class_id": _class_num(),
        "common__angle_rot_deg": 0.0,
        "common__angle_tilt_deg": 0.0,
        "common__angle_psi_deg": angle_psi_deg,
        "common__defocus_u_A": _safe_numeric(src.get(f"{prefix_raw}ctf/df1_A")),
        "common__defocus_v_A": _safe_numeric(src.get(f"{prefix_raw}ctf/df2_A")),
        "common__defocus_angle_deg": _deg_from_rad(src.get(f"{prefix_raw}ctf/df_angle_rad")),
        "common__phase_shift_deg": _deg_from_rad(src.get(f"{prefix_raw}ctf/phase_shift_rad")),
        "common__ctf_bfactor": _safe_numeric(src.get(f"{prefix_raw}ctf/bfactor")),
        "common__ctf_fom": _safe_numeric(src.get(f"{prefix_raw}ctf/scale")),
    })

    coord_x_safe = pd.to_numeric(out["common__coord_x_pix"], errors="coerce").fillna(0.0)
    coord_y_safe = pd.to_numeric(out["common__coord_y_pix"], errors="coerce").fillna(0.0)
    base = (
        out["common__micrograph_basename"].fillna("unknown").astype(str)
        + ":" + out["common__image_idx"].fillna(0).astype(int).astype(str)
        + ":" + coord_x_safe.round(2).astype(str)
        + "," + coord_y_safe.round(2).astype(str)
    )
    out["common__particle_uid"] = base.apply(lambda s: str(uuid.uuid5(uuid.NAMESPACE_URL, s)))

    unified = pd.concat([out, src], axis=1)
    return _ensure_unique_columns(unified)


def unify_particles_schema_from_relion(
    star_df: pd.DataFrame,
    *,
    dataset_id: str,
    stage: str,
    agent: str,
    backend: str,
    session_id: str,
    job_uid: str,
    prefix_raw: str = "rln__",
) -> pd.DataFrame:
    src = star_df.copy()
    src.columns = [f"{prefix_raw}{c}" for c in src.columns]

    image_name = src.get(f"{prefix_raw}rlnImageName").astype(str)

    def split_image_name(s: str) -> Tuple[Optional[int], str]:
        try:
            idx, path = s.split("@", 1)
            return int(idx), path
        except Exception:
            return None, s

    parts = image_name.apply(split_image_name)
    image_idx = parts.apply(lambda t: t[0] if t[0] is not None else 0)
    img_path = parts.apply(lambda t: t[1])

    out = pd.DataFrame({
        "common__dataset_id": dataset_id,
        "common__stage": stage,
        "common__agent": agent,
        "common__backend": backend,
        "common__session_id": session_id,
        "common__job_uid": job_uid,
        "common__schema_version": SCHEMA_VERSION,
        "common__micrograph_path": src.get(f"{prefix_raw}rlnMicrographName", img_path),
        "common__micrograph_basename": _basename(src.get(f"{prefix_raw}rlnMicrographName", img_path)),
        "common__micrograph_id": _basename(src.get(f"{prefix_raw}rlnMicrographName", img_path)),
        "common__image_name": image_name,
        "common__image_idx": image_idx,
        "common__coord_x_pix": _safe_numeric(src.get(f"{prefix_raw}rlnCoordinateX")),
        "common__coord_y_pix": _safe_numeric(src.get(f"{prefix_raw}rlnCoordinateY")),
        "common__origin_x_pix": _safe_numeric(src.get(f"{prefix_raw}rlnOriginX")),
        "common__origin_y_pix": _safe_numeric(src.get(f"{prefix_raw}rlnOriginY")),
        "common__pixel_size_A": _safe_numeric(src.get(f"{prefix_raw}rlnImagePixelSize")),
        "common__micrograph_pixel_size_A": _safe_numeric(src.get(f"{prefix_raw}rlnMicrographOriginalPixelSize")),
        "common__pick_score": _safe_numeric(src.get(f"{prefix_raw}rlnAutopickFigureOfMerit")),
        "common__class_id": _safe_numeric(src.get(f"{prefix_raw}rlnClassNumber"), dtype=int),
        "common__angle_rot_deg": _safe_numeric(src.get(f"{prefix_raw}rlnAngleRot")),
        "common__angle_tilt_deg": _safe_numeric(src.get(f"{prefix_raw}rlnAngleTilt")),
        "common__angle_psi_deg": _safe_numeric(src.get(f"{prefix_raw}rlnAnglePsi")),
        "common__defocus_u_A": _safe_numeric(src.get(f"{prefix_raw}rlnDefocusU")),
        "common__defocus_v_A": _safe_numeric(src.get(f"{prefix_raw}rlnDefocusV")),
        "common__defocus_angle_deg": _safe_numeric(src.get(f"{prefix_raw}rlnDefocusAngle")),
        "common__phase_shift_deg": _safe_numeric(src.get(f"{prefix_raw}rlnPhaseShift")),
        "common__ctf_bfactor": _safe_numeric(src.get(f"{prefix_raw}rlnCtfBfactor")),
        "common__ctf_fom": _safe_numeric(src.get(f"{prefix_raw}rlnCtfFigureOfMerit")),
    })

    coord_x_safe = pd.to_numeric(out["common__coord_x_pix"], errors="coerce").fillna(0.0)
    coord_y_safe = pd.to_numeric(out["common__coord_y_pix"], errors="coerce").fillna(0.0)
    base = (
        out["common__micrograph_basename"].fillna("unknown").astype(str)
        + ":" + out["common__image_idx"].fillna(0).astype(int).astype(str)
        + ":" + coord_x_safe.round(2).astype(str)
        + "," + coord_y_safe.round(2).astype(str)
    )
    out["common__particle_uid"] = base.apply(lambda s: str(uuid.uuid5(uuid.NAMESPACE_URL, s)))

    unified = pd.concat([out, src], axis=1)
    return _ensure_unique_columns(unified)


def unify_preprocessing_schema_from_cs(
    df_cs: pd.DataFrame,
    *,
    dataset_id: str,
    agent: str,
    backend: str,
    session_id: str,
    job_uid: str,
    prefix_raw: str = "cs__",
) -> pd.DataFrame:
    src = df_cs.copy()
    src.columns = [f"{prefix_raw}{c}" for c in src.columns]

    path_series = (
        src.get(f"{prefix_raw}location/micrograph_path")
        or src.get(f"{prefix_raw}micrograph/path")
        or src.get(f"{prefix_raw}blob/path")
    )
    micrograph_uid = src.get(f"{prefix_raw}location/micrograph_uid")

    micrograph_path = path_series.astype(str) if path_series is not None else None
    micrograph_basename = _basename(micrograph_path) if micrograph_path is not None else None
    micrograph_id = micrograph_uid if micrograph_uid is not None else micrograph_basename

    out = pd.DataFrame({
        "common__dataset_id": dataset_id,
        "common__stage": "preprocessing",
        "common__agent": agent,
        "common__backend": backend,
        "common__session_id": session_id,
        "common__job_uid": job_uid,
        "common__schema_version": SCHEMA_VERSION,
        "common__micrograph_path": micrograph_path,
        "common__micrograph_basename": micrograph_basename,
        "common__micrograph_id": micrograph_id,
        "common__defocus_u_A": _safe_numeric(src.get(f"{prefix_raw}ctf/df1_A")),
        "common__defocus_v_A": _safe_numeric(src.get(f"{prefix_raw}ctf/df2_A")),
        "common__defocus_angle_deg": _deg_from_rad(src.get(f"{prefix_raw}ctf/df_angle_rad")),
        "common__phase_shift_deg": _deg_from_rad(src.get(f"{prefix_raw}ctf/phase_shift_rad")),
        "common__pixel_size_A": _safe_numeric(src.get(f"{prefix_raw}location/micrograph_psize_A") or src.get(f"{prefix_raw}blob/psize_A")),
        "common__voltage_kv": _safe_numeric(src.get(f"{prefix_raw}ctf/accel_kv")),
        "common__cs_mm": _safe_numeric(src.get(f"{prefix_raw}ctf/cs_mm")),
        "common__amplitude_contrast": _safe_numeric(src.get(f"{prefix_raw}ctf/amp_contrast")),
    })

    if micrograph_id is not None:
        key = "common__micrograph_id"
    elif micrograph_path is not None:
        key = "common__micrograph_path"
    else:
        key = None
    unified = pd.concat([out, src], axis=1)
    unified = _ensure_unique_columns(unified)
    if key:
        unified = unified.sort_index().groupby(key, dropna=False).first().reset_index()
    return unified


def unify_preprocessing_schema_from_relion(
    star_df: pd.DataFrame,
    *,
    dataset_id: str,
    agent: str,
    backend: str,
    session_id: str,
    job_uid: str,
    prefix_raw: str = "rln__",
) -> pd.DataFrame:
    src = star_df.copy()
    src.columns = [f"{prefix_raw}{c}" for c in src.columns]

    micrograph_path = src.get(f"{prefix_raw}rlnMicrographName").astype(str)
    micrograph_basename = _basename(micrograph_path)
    micrograph_id = micrograph_basename

    out = pd.DataFrame({
        "common__dataset_id": dataset_id,
        "common__stage": "preprocessing",
        "common__agent": agent,
        "common__backend": backend,
        "common__session_id": session_id,
        "common__job_uid": job_uid,
        "common__schema_version": SCHEMA_VERSION,
        "common__micrograph_path": micrograph_path,
        "common__micrograph_basename": micrograph_basename,
        "common__micrograph_id": micrograph_id,
        "common__defocus_u_A": _safe_numeric(src.get(f"{prefix_raw}rlnDefocusU")),
        "common__defocus_v_A": _safe_numeric(src.get(f"{prefix_raw}rlnDefocusV")),
        "common__defocus_angle_deg": _safe_numeric(src.get(f"{prefix_raw}rlnDefocusAngle")),
        "common__phase_shift_deg": _safe_numeric(src.get(f"{prefix_raw}rlnPhaseShift")),
        "common__pixel_size_A": _safe_numeric(src.get(f"{prefix_raw}rlnMicrographOriginalPixelSize")),
        "common__voltage_kv": _safe_numeric(src.get(f"{prefix_raw}rlnVoltage")),
        "common__cs_mm": _safe_numeric(src.get(f"{prefix_raw}rlnSphericalAberration")),
        "common__amplitude_contrast": _safe_numeric(src.get(f"{prefix_raw}rlnAmplitudeContrast")),
    })

    unified = pd.concat([out, src], axis=1)
    unified = _ensure_unique_columns(unified)
    unified = unified.sort_index().groupby("common__micrograph_id", dropna=False).first().reset_index()
    return unified


def validate_stage_required(df: pd.DataFrame, stage: str) -> None:
    stage = str(stage).lower()
    if stage == "preprocessing":
        required = ["common__micrograph_id", "common__defocus_u_A", "common__defocus_v_A"]
    elif stage == "particle_picking":
        required = ["common__coord_x_pix", "common__coord_y_pix"]
    elif stage == "reconstruction":
        required = [
            "common__angle_rot_deg",
            "common__angle_tilt_deg",
            "common__angle_psi_deg",
            "common__class_id",
        ]
    else:
        required = []
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns for stage '{stage}': {missing}")


def write_parquet_partition(
    df: pd.DataFrame,
    root: str,
    *,
    stage: str,
    agent: str,
    session_id: str,
    dataset_id: str,
    filename: str = "part-00000.parquet",
    compression: str = "snappy",
) -> str:
    part_dir = os.path.join(
        root,
        f"stage={stage}",
        f"dataset_id={dataset_id}",
        f"agent={agent}",
        f"session_id={session_id}",
    )
    os.makedirs(part_dir, exist_ok=True)
    path = os.path.join(part_dir, filename)
    df.to_parquet(path, compression=compression, index=False)
    return path


# ---------------------------------------------------------------------------
# High-level Conversion Tool
# ---------------------------------------------------------------------------


class ConversionTool:
    """High-level adapter that wraps conversions between CryoSPARC, RELION, and Parquet."""

    def __init__(self) -> None:
        self._file_converter = FileConversionTools()

    def cryosparc_to_relion_star(
        self,
        cs_path: str | Path,
        star_path: str | Path,
        passthrough_path: Optional[str | Path] = None,
    ) -> Path:
        return self._file_converter.convert_cs_to_star(cs_path, star_path, passthrough_path)

    def cryosparc_to_parquet(
        self,
        cs_path: str | Path,
        parquet_root: str | Path,
        *,
        dataset_id: str,
        stage: str,
        session_id: str,
        job_uid: str,
        agent: str = "cryosparc",
        backend: str = "CryoSPARC",
        validate: bool = True,
        filename: str = "part-00000.parquet",
        compression: str = "snappy",
    ) -> Tuple[pd.DataFrame, str]:
        df_cs = self._file_converter.read_cs_file(cs_path)
        unified = self._unify_cs_dataframe(
            df_cs,
            stage=stage,
            dataset_id=dataset_id,
            agent=agent,
            backend=backend,
            session_id=session_id,
            job_uid=job_uid,
        )
        if validate:
            validate_stage_required(unified, stage)
        parquet_path = write_parquet_partition(
            unified,
            str(parquet_root),
            stage=stage,
            agent=agent,
            session_id=session_id,
            dataset_id=dataset_id,
            filename=filename,
            compression=compression,
        )
        return unified, parquet_path

    def relion_star_to_parquet(
        self,
        star_path: str | Path,
        parquet_root: str | Path,
        *,
        dataset_id: str,
        stage: str,
        session_id: str,
        job_uid: str,
        agent: str = "relion",
        backend: str = "RELION",
        validate: bool = True,
        filename: str = "part-00000.parquet",
        compression: str = "snappy",
    ) -> Tuple[pd.DataFrame, str]:
        particles_df, _ = read_star_particles(str(star_path))
        unified = self._unify_relion_dataframe(
            particles_df,
            stage=stage,
            dataset_id=dataset_id,
            agent=agent,
            backend=backend,
            session_id=session_id,
            job_uid=job_uid,
        )
        if validate:
            validate_stage_required(unified, stage)
        parquet_path = write_parquet_partition(
            unified,
            str(parquet_root),
            stage=stage,
            agent=agent,
            session_id=session_id,
            dataset_id=dataset_id,
            filename=filename,
            compression=compression,
        )
        return unified, parquet_path

    def _unify_cs_dataframe(
        self,
        df: pd.DataFrame,
        *,
        stage: str,
        dataset_id: str,
        agent: str,
        backend: str,
        session_id: str,
        job_uid: str,
    ) -> pd.DataFrame:
        stage_normalized = stage.lower()
        if stage_normalized == "preprocessing":
            return unify_preprocessing_schema_from_cs(
                df,
                dataset_id=dataset_id,
                agent=agent,
                backend=backend,
                session_id=session_id,
                job_uid=job_uid,
            )
        return unify_particles_schema_from_cs(
            df,
            dataset_id=dataset_id,
            stage=stage_normalized,
            agent=agent,
            backend=backend,
            session_id=session_id,
            job_uid=job_uid,
        )

    def _unify_relion_dataframe(
        self,
        df: pd.DataFrame,
        *,
        stage: str,
        dataset_id: str,
        agent: str,
        backend: str,
        session_id: str,
        job_uid: str,
    ) -> pd.DataFrame:
        stage_normalized = stage.lower()
        if stage_normalized == "preprocessing":
            return unify_preprocessing_schema_from_relion(
                df,
                dataset_id=dataset_id,
                agent=agent,
                backend=backend,
                session_id=session_id,
                job_uid=job_uid,
            )
        return unify_particles_schema_from_relion(
            df,
            dataset_id=dataset_id,
            stage=stage_normalized,
            agent=agent,
            backend=backend,
            session_id=session_id,
            job_uid=job_uid,
        )


__all__ = [
    "SCHEMA_VERSION",
    "FileConversionTools",
    "ConversionTool",
    "read_star_particles",
    "unify_particles_schema_from_cs",
    "unify_particles_schema_from_relion",
    "unify_preprocessing_schema_from_cs",
    "unify_preprocessing_schema_from_relion",
    "validate_stage_required",
    "write_parquet_partition",
]

