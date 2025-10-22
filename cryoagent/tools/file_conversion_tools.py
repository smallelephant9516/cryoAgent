"""CryoSPARC to RELION conversion helpers (inspired by helicon.images2star)."""

from __future__ import annotations

import math
from pathlib import Path, PurePath
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class FileConversionTools:
    """Utility functions to convert CryoSPARC metadata into RELION STAR files."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def read_cs_file(self, cs_path: Union[str, Path]) -> pd.DataFrame:
        """Load a CryoSPARC .cs structured array into a pandas DataFrame."""
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
    ) -> Path:
        """Convert the CryoSPARC metadata file to a RELION STAR file."""
        df_raw = self.read_cs_file(cs_path)
        particles, optics = self._build_relion_tables(df_raw)
        particles.attrs["optics"] = optics
        star_path = Path(star_path)
        star_path.parent.mkdir(parents=True, exist_ok=True)
        self._dataframe_to_star(particles, star_path, format="v3")
        return star_path

    # ------------------------------------------------------------------
    # Conversion helpers (adapted from helicon)
    # ------------------------------------------------------------------
    def _build_relion_tables(self, df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        n_particles = len(df_raw)
        indices = pd.Series(np.arange(1, n_particles + 1, dtype=np.int64))

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
        particle_data: Dict[str, Any] = {
            "rlnImageName": image_names,
            "rlnMicrographName": micrograph_names,
            "rlnImagePixelSize": pixel_size,
            "rlnMicrographOriginalPixelSize": pixel_size,
            "rlnAutopickFigureOfMerit": pd.to_numeric(
                df_raw.get("pick_stats/ncc_score"), errors="coerce"
            ),
        }

        class_series = None
        for field in (
            "alignments3D/class",
            "alignments2D/class",
            "pick_stats/template_idx",
        ):
            if field in df_raw:
                class_series = df_raw.get(field)
                break

        if class_series is None:
            particle_data["rlnClassNumber"] = pd.Series(np.ones(n_particles, dtype=int))
        else:
            class_values = pd.to_numeric(class_series, errors="coerce").fillna(0)
            particle_data["rlnClassNumber"] = class_values.round().astype(int) + 1

        angle_series = df_raw.get("pick_stats/angle_rad")
        if angle_series is not None:
            angle_rad = pd.to_numeric(angle_series, errors="coerce")
            particle_data["rlnAnglePsi"] = np.degrees(angle_rad)

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

        shift_series = df_raw.get("ctf/shift_A")
        if shift_series is not None:
            particle_data["rlnOriginX"] = shift_series.apply(
                lambda val: self._extract_shift_component(val, axis=0)
            )
            particle_data["rlnOriginY"] = shift_series.apply(
                lambda val: self._extract_shift_component(val, axis=1)
            )
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
                    "rlnMicrographOriginalPixelSize": float(
                        self._first_valid(particles.loc[mask, "rlnMicrographOriginalPixelSize"], np.nan)
                    ),
                    "rlnVoltage": float(self._first_valid(accel_series, np.nan)),
                    "rlnSphericalAberration": float(self._first_valid(cs_series, np.nan)),
                    "rlnAmplitudeContrast": float(self._first_valid(amp_series, np.nan)),
                    "rlnImagePixelSize": float(
                        self._first_valid(particles.loc[mask, "rlnImagePixelSize"], np.nan)
                    ),
                    "rlnImageSize": float(self._first_valid(image_size, np.nan)),
                    "rlnImageDimensionality": 2,
                    "rlnCtfDataAreCtfPremultiplied": 0,
                }
            )

        optics = pd.DataFrame(optics_records)
        return particles, optics

    # ------------------------------------------------------------------
    # STAR writer (based on helicon.lib.io.dataframe2star)
    # ------------------------------------------------------------------
    def _dataframe_to_star(
        self,
        data: pd.DataFrame,
        star_file: Union[str, Path],
        format: str = "v3",
    ) -> None:
        data = data.copy()
        optics = data.attrs.get("optics")
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

            handle.write("data_particles\n\n")
            handle.write("loop_\n")
            particle_columns = [c for c in data.columns if c.startswith("rln")]
            for idx, col in enumerate(particle_columns, start=1):
                handle.write(f"_{col} #{idx}\n")
            for _, row in data.iterrows():
                values = [self._format_star_value(row[col]) for col in particle_columns]
                handle.write(" ".join(values) + "\n")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
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
        valid = series.dropna()
        if valid.empty:
            return default
        return valid.iloc[0]


__all__ = ["FileConversionTools"]
