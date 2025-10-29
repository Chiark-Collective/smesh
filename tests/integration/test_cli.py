from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from typer.testing import CliRunner

from smesh.cli.main import app


def _write_ascii_ply(path: Path) -> None:
    vertices = [
        (-1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def test_cli_lidar_npz(tmp_path: Path) -> None:
    mesh_path = tmp_path / "plane.ply"
    _write_ascii_ply(mesh_path)

    config = {
        "type": "lidar",
        "mesh": {"path": str(mesh_path.name)},
        "trajectory": {
            "kind": "static",
            "xyz": [0.5, 0.5, 2.0],
            "rpy_deg": [0.0, 0.0, 0.0],
        },
        "pattern": {
            "kind": "oscillating",
            "fov_deg": 20.0,
            "line_rate_hz": 1.0,
            "pulses_per_line": 16,
        },
        "sensor": {
            "kind": "lidar",
            "max_range_m": 10.0,
            "multi_return": False,
            "beam_divergence_mrad": 0.3,
            "num_lines": 1,
        },
        "noise": {"kind": "lidar", "keep_prob": 1.0},
        "attributes": ["range", "incidence", "scan_angle", "intensity", "returns"],
        "output": {"path": "out_scan.npz", "format": "npz"},
        "seed": 42,
    }

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    runner = CliRunner()
    result = runner.invoke(app, ["sample", str(cfg_path)])

    assert result.exit_code == 0, result.stdout

    out_path = tmp_path / "out_scan.npz"
    assert out_path.exists()
    data = np.load(out_path)
    xyz = data["xyz"]
    assert xyz.shape[0] > 0
    assert np.allclose(xyz[:, 2], 0.0)  # points lie on the plane z=0


def test_cli_lidar_las(tmp_path: Path) -> None:
    mesh_path = tmp_path / "plane.ply"
    _write_ascii_ply(mesh_path)

    config = {
        "type": "lidar",
        "mesh": {"path": str(mesh_path.name)},
        "trajectory": {
            "kind": "static",
            "xyz": [0.0, 0.0, 3.0],
        },
        "pattern": {
            "kind": "oscillating",
            "fov_deg": 30.0,
            "line_rate_hz": 1.0,
            "pulses_per_line": 32,
        },
        "sensor": {
            "kind": "lidar",
            "max_range_m": 15.0,
            "multi_return": False,
            "beam_divergence_mrad": 0.3,
            "num_lines": 1,
        },
        "noise": {"kind": "lidar", "keep_prob": 1.0},
        "attributes": ["range", "scan_angle", "returns", "intensity"],
        "output": {"path": "out_scan.las", "format": "las"},
        "seed": 7,
    }

    cfg_path = tmp_path / "las_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    runner = CliRunner()
    result = runner.invoke(app, ["sample", str(cfg_path)])
    assert result.exit_code == 0, result.stdout

    out_path = tmp_path / "out_scan.las"
    assert out_path.exists()

    import laspy

    with laspy.open(out_path) as reader:
        points = reader.read()
        assert len(points.x) > 0
        assert np.max(points.z) <= 0.01
        point_format = reader.header.point_format
        assert "Range" in point_format.extra_dimension_names
        assert "intensity" in point_format.dimension_names


def test_cli_run_with_overrides(tmp_path: Path) -> None:
    mesh_path = tmp_path / "plane.ply"
    _write_ascii_ply(mesh_path)

    config = {
        "type": "lidar",
        "mesh": {"path": str(mesh_path.name)},
        "trajectory": {"kind": "static", "xyz": [0.0, 0.0, 2.0]},
        "pattern": {"kind": "oscillating", "fov_deg": 10.0, "line_rate_hz": 1.0, "pulses_per_line": 8},
        "sensor": {"kind": "lidar", "max_range_m": 5.0, "multi_return": False, "beam_divergence_mrad": 0.3, "num_lines": 1},
        "noise": {"kind": "lidar", "keep_prob": 1.0},
        "output": {"path": "out_scan.npz", "format": "npz"},
    }

    cfg_path = tmp_path / "config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    override_path = tmp_path / "custom_output.ply"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(cfg_path),
            "--output",
            str(override_path),
            "--seed",
            "99",
            "--attribute",
            "range",
            "--attribute",
            "incidence",
            "--log-level",
            "DEBUG",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert override_path.exists()
    with open(override_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    assert header == "ply"


def test_mesh_generate_command(tmp_path: Path) -> None:
    output = tmp_path / "demo_mesh.ply"
    runner = CliRunner()
    result = runner.invoke(app, ["mesh", "generate", str(output), "--preset", "demo", "--size", "12"])
    assert result.exit_code == 0, result.stdout
    assert output.exists()
    with open(output, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        second = f.readline().strip()
    assert header == "ply"
    assert second == "format ascii 1.0"


def test_cli_sample_lidar_smoke(tmp_path: Path) -> None:
    mesh_path = tmp_path / "plane.ply"
    _write_ascii_ply(mesh_path)

    out_path = tmp_path / "quick_scan.npz"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "sample-lidar",
            "--mesh",
            str(mesh_path),
            "--output",
            str(out_path),
            "--altitude-m",
            "5.0",
            "--line-spacing-m",
            "1.0",
            "--speed-mps",
            "5.0",
            "--pulses-per-line",
            "8",
            "--line-rate-hz",
            "5.0",
            "--num-lines",
            "1",
            "--batch-size",
            "16",
            "--keep-prob",
            "1.0",
            "--max-range-m",
            "20.0",
            "--sigma-range-m",
            "0.0",
            "--sigma-angle-deg",
            "0.0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert out_path.exists()
    cloud = np.load(out_path)
    assert cloud["xyz"].shape[0] > 0
    assert "gps_time" in cloud
    assert cloud["gps_time"].shape[0] == cloud["xyz"].shape[0]


def test_cli_sample_lidar_rejects_zero_keep_prob(tmp_path: Path) -> None:
    mesh_path = tmp_path / "plane.ply"
    _write_ascii_ply(mesh_path)
    out_path = tmp_path / "bad_scan.npz"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "sample-lidar",
            "--mesh",
            str(mesh_path),
            "--output",
            str(out_path),
            "--keep-prob",
            "0.0",
        ],
    )

    assert result.exit_code != 0
    combined = result.stdout + result.stderr
    assert "within (0, 1]" in combined


def test_cli_total_station(tmp_path: Path) -> None:
    mesh_path = tmp_path / "plane.ply"
    _write_ascii_ply(mesh_path)

    config = {
        "type": "totalstation",
        "mesh": {"path": str(mesh_path.name)},
        "trajectory": {"kind": "static", "xyz": [0.0, 0.0, 5.0]},
        "pattern": {
            "kind": "raster",
            "az_range_deg": [-5.0, 5.0],
            "el_range_deg": [-70.0, -10.0],
            "step_deg": 2.0,
        },
        "sensor": {"kind": "totalstation", "max_range_m": 200.0, "boresight_rpy_deg": [0.0, 90.0, 0.0]},
        "output": {"path": "out_total.npz", "format": "npz"},
        "attributes": ["range", "incidence"],
        "seed": 3,
    }

    cfg_path = tmp_path / "total_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    runner = CliRunner()
    result = runner.invoke(app, ["sample", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    data = np.load(tmp_path / "out_total.npz")
    assert "xyz" in data and data["xyz"].shape[0] > 0


def test_cli_photogrammetry(tmp_path: Path) -> None:
    mesh_path = tmp_path / "plane.ply"
    _write_ascii_ply(mesh_path)

    config = {
        "type": "photogrammetry",
        "mesh": {"path": str(mesh_path.name)},
        "trajectory": {"kind": "static", "xyz": [0.0, 0.0, 2.0]},
        "pattern": {
            "kind": "camera",
            "resolution_px": [8, 6],
            "focal_px": [400.0, 400.0],
            "pixel_stride": 2,
        },
        "sensor": {
            "kind": "camera",
            "num_frames": 1,
        },
        "noise": {
            "kind": "photogrammetry",
            "pixel_sigma": 0.0,
        },
        "output": {"path": "out_photo.npz", "format": "npz"},
        "attributes": ["range", "color_normal"],
    }

    cfg_path = tmp_path / "photo_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    runner = CliRunner()
    result = runner.invoke(app, ["sample", str(cfg_path)])
    assert result.exit_code == 0, result.stdout
    data = np.load(tmp_path / "out_photo.npz")
    assert "pixel_u" in data and "pixel_v" in data
