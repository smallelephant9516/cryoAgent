#!/usr/bin/env python3
"""Minimal tests for the CryoSPARC -> RELION conversion helpers."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from cryoagent.tools.file_conversion_tools import FileConversionTools


def build_mock_cs_file(destination: Path) -> None:
    dtype = np.dtype(
        [
            ("blob/path", "S80"),
            ("blob/idx", "<i4"),
            ("blob/psize_A", "<f4"),
            ("blob/shape", "<i4", (2,)),
            ("ctf/exp_group_id", "<i4"),
            ("ctf/accel_kv", "<f4"),
            ("ctf/cs_mm", "<f4"),
            ("ctf/amp_contrast", "<f4"),
            ("ctf/df1_A", "<f4"),
            ("ctf/df2_A", "<f4"),
            ("ctf/df_angle_rad", "<f4"),
            ("ctf/phase_shift_rad", "<f4"),
            ("ctf/scale", "<f4"),
            ("ctf/bfactor", "<f4"),
            ("ctf/shift_A", "<f4", (2,)),
            ("pick_stats/ncc_score", "<f4"),
            ("pick_stats/template_idx", "<i4"),
            ("pick_stats/angle_rad", "<f4"),
        ]
    )

    records = np.zeros(3, dtype=dtype)
    paths = [
        b"J000/particles/particle_000001.mrcs",
        b"J000/particles/particle_000002.mrcs",
        b"J000/particles/particle_000003.mrcs",
    ]
    records["blob/path"] = paths
    records["blob/idx"] = np.array([0, 1, 2], dtype="<i4")
    records["blob/psize_A"] = 2.26
    records["blob/shape"] = np.array([[128, 128]] * 3, dtype="<i4")
    records["ctf/exp_group_id"] = np.array([1, 1, 1], dtype="<i4")
    records["ctf/accel_kv"] = 300.0
    records["ctf/cs_mm"] = 2.7
    records["ctf/amp_contrast"] = 0.1
    records["ctf/df1_A"] = np.array([12000.0, 12100.0, 12200.0], dtype="<f4")
    records["ctf/df2_A"] = np.array([11800.0, 11900.0, 12000.0], dtype="<f4")
    records["ctf/df_angle_rad"] = np.array([0.1, 0.2, 0.3], dtype="<f4")
    records["ctf/phase_shift_rad"] = np.array([0.0, 0.01, 0.02], dtype="<f4")
    records["ctf/scale"] = 0.95
    records["ctf/bfactor"] = 150.0
    records["ctf/shift_A"] = np.array([[0.0, 0.0], [1.0, -1.0], [0.5, 2.0]], dtype="<f4")
    records["pick_stats/ncc_score"] = np.array([0.8, 0.75, 0.72], dtype="<f4")
    records["pick_stats/template_idx"] = np.array([0, 1, 2], dtype="<i4")
    records["pick_stats/angle_rad"] = np.array([0.0, 0.5, 1.0], dtype="<f4")

    with destination.open("wb") as handle:
        np.save(handle, records)


def run_conversion() -> None:
    tool = FileConversionTools()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cs_path = tmp_path / "mock_particles.cs"
        star_path = tmp_path / "particles.star"
        build_mock_cs_file(cs_path)
        tool.convert_cs_to_star(cs_path, star_path)
        contents = star_path.read_text(encoding="utf-8")

    assert "data_optics" in contents
    assert "data_particles" in contents
    assert "opticsGroup1" in contents
    assert "00000001@J000/particles/particle_000001.mrcs" in contents
    assert "rlnAutopickFigureOfMerit" in contents


def main() -> None:
    parser = argparse.ArgumentParser(description="CryoSPARC to RELION converter")
    parser.add_argument("cs_path", nargs="?", help="Input CryoSPARC .cs file")
    parser.add_argument("star_path", nargs="?", help="Output RELION .star file")
    parser.add_argument("--self-test", action="store_true", help="Run internal conversion test")
    args = parser.parse_args()

    if args.self_test or (args.cs_path is None and args.star_path is None):
        run_conversion()
        print("Conversion test completed successfully")
        return

    if args.cs_path is None or args.star_path is None:
        parser.error("Provide both <cs_path> and <star_path> or use --self-test")

    tool = FileConversionTools()
    tool.convert_cs_to_star(args.cs_path, args.star_path)
    print(f"Wrote STAR file to {args.star_path}")


if __name__ == "__main__":
    main()
