"""Test coverage for cross-agent conversion and interoperability utilities."""

from __future__ import annotations

import os
import sys
import types
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
# ---------------------------------------------------------------------------
# Provide a stub implementation for cryosparc.tools so CryoSPARCTools can load
# without a live CryoSPARC installation.
# ---------------------------------------------------------------------------


class _FakeJob:
    def __init__(self, job_type: str, params: Dict[str, Any]):
        self.job_type = job_type
        self.params = params
        self.uid = f"JTEST-{uuid.uuid4().hex[:8]}"
        self.queued = False

    def queue(self, lane: str | None = None, hostname: str | None = None) -> None:
        self.queued = True


class _FakeWorkspace:
    def __init__(self) -> None:
        self.created_jobs: List[_FakeJob] = []

    def create_job(self, job_type: str, *, params: Dict[str, Any], connections: Dict[str, Any] | None = None) -> _FakeJob:
        job = _FakeJob(job_type, params)
        self.created_jobs.append(job)
        return job


class _FakeProject:
    def __init__(self) -> None:
        self.workspace = _FakeWorkspace()

    def find_workspace(self, workspace_uid: str) -> _FakeWorkspace:
        return self.workspace


class _FakeCryoSPARC:
    def __init__(self, **kwargs: Any) -> None:
        self.connection_kwargs = kwargs
        self.project = _FakeProject()

    def test_connection(self) -> bool:
        return True

    def find_project(self, project_uid: str) -> _FakeProject:
        return self.project

    # Optional helper for CryoSPARCTools when queueing without lane
    def get_lanes(self) -> List[Dict[str, str]]:
        return [{"name": "lane0"}]


_fake_module = types.ModuleType("cryosparc.tools")
_fake_module.CryoSPARC = _FakeCryoSPARC
sys.modules.setdefault("cryosparc.tools", _fake_module)


from cryoagent.tools import (  # noqa: E402  (import after stub setup)
    ConversionTool,
    CryoSPARCTools,
    read_star_particles,
)
from cryoagent.config.config_loader import CryoSPARCSettings  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_dummy_cs_file(path: Path, *, n_particles: int = 5) -> Path:
    """Create a synthetic CryoSPARC `.cs` file for testing."""

    dtype = [
        ("blob/path", "U256"),
        ("blob/idx", np.int64),
        ("blob/psize_A", np.float64),
        ("pick_stats/ncc_score", np.float64),
        ("ctf/df1_A", np.float64),
        ("ctf/df2_A", np.float64),
        ("ctf/df_angle_rad", np.float64),
        ("ctf/phase_shift_rad", np.float64),
        ("ctf/accel_kv", np.float64),
        ("ctf/cs_mm", np.float64),
        ("ctf/amp_contrast", np.float64),
        ("pick_stats/angle_rad", np.float64),
        ("alignments2D/class", np.int64),
        ("location/center_x_frac", np.float64),
        ("location/center_y_frac", np.float64),
        ("location/micrograph_shape", object),
        ("location/micrograph_psize_A", np.float64),
        ("location/micrograph_uid", "U128"),
    ]

    data = np.zeros(n_particles, dtype=dtype)
    for i in range(n_particles):
        data["blob/path"][i] = f"/data/micrographs/micro_{i:03d}.mrc"
        data["blob/idx"][i] = i
        data["blob/psize_A"][i] = 1.5
        data["pick_stats/ncc_score"][i] = 0.5 + i * 0.01
        data["ctf/df1_A"][i] = 15000 + i * 10
        data["ctf/df2_A"][i] = 15050 + i * 10
        data["ctf/df_angle_rad"][i] = 0.1
        data["ctf/phase_shift_rad"][i] = 0.05
        data["ctf/accel_kv"][i] = 300
        data["ctf/cs_mm"][i] = 2.7
        data["ctf/amp_contrast"][i] = 0.1
        data["pick_stats/angle_rad"][i] = 0.2
        data["alignments2D/class"][i] = i % 3
        data["location/center_x_frac"][i] = 0.5
        data["location/center_y_frac"][i] = 0.5
        data["location/micrograph_shape"][i] = np.array([4096, 4096])
        data["location/micrograph_psize_A"][i] = 1.5
        data["location/micrograph_uid"][i] = f"micrograph-{i:03d}"

    with open(path, "wb") as handle:
        np.save(handle, data)

    return path


def _collect_required_relion_columns(df: pd.DataFrame) -> List[str]:
    return [
        "rlnImageName",
        "rlnMicrographName",
        "rlnCoordinateX",
        "rlnCoordinateY",
        "rlnDefocusU",
        "rlnDefocusV",
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_convert_cs_to_star(tmp_path: Path) -> None:
    cs_path = tmp_path / "particles.cs"
    star_path = tmp_path / "particles.star"

    _create_dummy_cs_file(cs_path)

    tool = ConversionTool()
    tool.cryosparc_to_relion_star(cs_path, star_path)

    assert star_path.exists(), "STAR file should be created"

    particles_df, optics_df = read_star_particles(star_path)
    assert len(particles_df) > 0

    missing = [col for col in _collect_required_relion_columns(particles_df) if col not in particles_df.columns]
    assert not missing, f"STAR file missing required RELION columns: {missing}"
    assert optics_df is not None and len(optics_df) >= 1


def test_parquet_export_with_common_group(tmp_path: Path) -> None:
    cs_path = tmp_path / "particles.cs"
    parquet_root = tmp_path / "parquet"

    _create_dummy_cs_file(cs_path)

    tool = ConversionTool()
    df_unified, parquet_path = tool.cryosparc_to_parquet(
        cs_path,
        parquet_root,
        dataset_id="datasetA",
        stage="particle_picking",
        session_id="session1",
        job_uid="J001",
    )

    assert Path(parquet_path).exists()
    df_read = pd.read_parquet(parquet_path)

    for required in [
        "common__micrograph_id",
        "common__coord_x_pix",
        "common__coord_y_pix",
        "common__particle_uid",
    ]:
        assert required in df_read.columns

    # Ensure raw CryoSPARC fields are preserved
    assert any(col.startswith("cs__blob/path") for col in df_read.columns)


def test_import_relion_star_into_cryosparc(tmp_path: Path) -> None:
    cs_path = tmp_path / "particles.cs"
    star_path = tmp_path / "particles.star"

    _create_dummy_cs_file(cs_path)
    ConversionTool().cryosparc_to_relion_star(cs_path, star_path)

    settings = CryoSPARCSettings(
        host="localhost",
        base_port=39000,
        username="user@example.com",
        password="password",
        license_id="LICENSE",
    )

    tools = CryoSPARCTools(settings)
    result = tools.import_particles_from_star(
        project_uid="P1",
        workspace_uid="W1",
        star_path=str(star_path),
        data_sign="negative",
        wait_for_completion=False,
    )

    assert result["job_type"] == "import_particles"
    assert result["status"] == "queued"
    assert result["project_uid"] == "P1"
    assert result["workspace_uid"] == "W1"


def test_relion_star_is_reusable_in_relion(tmp_path: Path) -> None:
    cs_path = tmp_path / "particles.cs"
    star_path = tmp_path / "particles.star"

    _create_dummy_cs_file(cs_path)
    ConversionTool().cryosparc_to_relion_star(cs_path, star_path)

    particles_df, _ = read_star_particles(star_path)
    missing = [col for col in _collect_required_relion_columns(particles_df) if col not in particles_df.columns]
    assert not missing

    tool = ConversionTool()
    df_unified, parquet_path = tool.relion_star_to_parquet(
        star_path,
        tmp_path / "parquet_relion",
        dataset_id="datasetA",
        stage="reconstruction",
        session_id="session2",
        job_uid="run01",
    )

    assert Path(parquet_path).exists()

    assert "common__angle_psi_deg" in df_unified.columns
    assert any(col.startswith("rln__rlnImageName") for col in df_unified.columns)


