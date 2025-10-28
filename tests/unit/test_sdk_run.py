from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from smesh.config import load_config
from smesh.sdk import sample_from_config


def _write_test_mesh(path: Path) -> None:
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


def _write_config(path: Path, mesh_name: str, output_name: str) -> None:
    config = {
        "type": "lidar",
        "mesh": {"path": mesh_name},
        "trajectory": {"kind": "static", "xyz": [0.0, 0.0, 2.0]},
        "pattern": {"kind": "oscillating", "fov_deg": 15.0, "line_rate_hz": 1.0, "pulses_per_line": 12},
        "sensor": {
            "kind": "lidar",
            "max_range_m": 20.0,
            "multi_return": False,
            "beam_divergence_mrad": 0.3,
            "num_lines": 1,
        },
        "noise": {"kind": "lidar", "keep_prob": 1.0},
        "attributes": ["range", "incidence", "scan_angle"],
        "sampler": {"batch_size_rays": 256},
        "output": {"path": output_name, "format": "npz"},
        "seed": 123,
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


def test_sample_from_config_path(tmp_path: Path) -> None:
    mesh_path = tmp_path / "plane.ply"
    _write_test_mesh(mesh_path)

    cfg_path = tmp_path / "scenario.yaml"
    _write_config(cfg_path, mesh_path.name, "scan.npz")

    result = sample_from_config(cfg_path)

    assert result.output_path.exists()
    data = np.load(result.output_path)
    xyz = data["xyz"]
    assert xyz.shape[0] > 0
    assert np.allclose(xyz[:, 2], 0.0, atol=1e-6)
    assert result.stats["points"] == xyz.shape[0]
    assert str(result.output_path).endswith("scan.npz")


def test_sample_from_config_object_override(tmp_path: Path) -> None:
    mesh_path = tmp_path / "plane.ply"
    _write_test_mesh(mesh_path)

    cfg_path = tmp_path / "scenario.yaml"
    _write_config(cfg_path, mesh_path.name, "first_scan.npz")

    cfg = load_config(cfg_path)

    override_path = tmp_path / "override_scan.npz"
    result = sample_from_config(
        cfg,
        output=override_path,
        seed=999,
        attributes=["range"],
    )

    assert result.output_path == override_path.resolve()
    assert result.output_path.exists()
    data = np.load(result.output_path)
    assert data["xyz"].shape[0] == result.stats["points"]
    assert result.config.attributes == ["range"]
