from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import laspy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from smesh.cli.main import _execute_sample  # type: ignore
from smesh.config import load_config
from smesh.core.scene import MeshScene

try:
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
    _HAVE_VTK = True
except Exception:  # pragma: no cover - optional dependency
    vtk_to_numpy = None  # type: ignore
    _HAVE_VTK = False

matplotlib.use("Agg")


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    config_path: Optional[Path] = None
    mesh_path: Optional[Path] = None


PREVIEW_EXAMPLES: List[ExampleSpec] = [
    ExampleSpec(name="raw_mesh", mesh_path=Path("examples/meshes/preview_scene.ply")),
    ExampleSpec(name="aerial_lidar", config_path=Path("examples/configs/preview/aerial_lidar_preview.yaml")),
    ExampleSpec(name="mobile_lidar", config_path=Path("examples/configs/preview/mobile_lidar_preview.yaml")),
    ExampleSpec(name="total_station", config_path=Path("examples/configs/preview/total_station_preview.yaml")),
    ExampleSpec(name="photogrammetry", config_path=Path("examples/configs/preview/photogrammetry_preview.yaml")),
]

FULL_EXAMPLES: List[ExampleSpec] = [
    ExampleSpec(name="aerial_lidar", config_path=Path("examples/configs/aerial_lidar.yaml")),
    ExampleSpec(name="mobile_lidar", config_path=Path("examples/configs/mobile_lidar.yaml")),
    ExampleSpec(name="total_station", config_path=Path("examples/configs/total_station.yaml")),
    ExampleSpec(name="photogrammetry", config_path=Path("examples/configs/photogrammetry.yaml")),
]

OUTPUT_DIR = Path("examples/outputs")
IMAGE_DIR = Path("examples/images")


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def run_example(spec: ExampleSpec, overwrite: bool = True, log_level: str = "INFO") -> Path:
    if spec.config_path is None:
        raise ValueError(f"Example '{spec.name}' does not have a config path.")
    cfg = load_config(spec.config_path)
    ext = cfg.output.format.lower()
    override_ext = "las" if ext == "laz" else ext
    out_path = OUTPUT_DIR / f"{spec.name}.{override_ext}"
    if out_path.exists() and not overwrite:
        logging.info("Skipping %s (output exists)", spec.name)
        return out_path

    _execute_sample(  # type: ignore
        config=spec.config_path,
        output_override=out_path,
        seed_override=None,
        engine_override=None,
        attribute_override=None,
        log_level=log_level,
    )
    return out_path


def _load_ascii_ply_vertices(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as fh:
        header_complete = False
        vertex_count: Optional[int] = None
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading {path}")
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                parts = stripped.split()
                vertex_count = int(parts[-1])
            if stripped == "end_header":
                header_complete = True
                break
        if not header_complete or vertex_count is None:
            raise ValueError(f"Could not parse vertex header for {path}")
        vertices = []
        for _ in range(vertex_count):
            parts = fh.readline().split()
            if len(parts) < 3:
                raise ValueError(f"Malformed vertex line in {path}")
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(vertices, dtype=np.float32)


def _load_mesh_vertices(mesh_path: Path) -> np.ndarray:
    scene = MeshScene(mesh_path=str(mesh_path))
    if _HAVE_VTK:
        try:
            poly = scene.vtk_polydata()
            pts = poly.GetPoints()
            if pts is not None and vtk_to_numpy is not None:
                arr = vtk_to_numpy(pts.GetData()).astype(np.float32, copy=False)
                return arr
        except Exception:
            pass
    return _load_ascii_ply_vertices(mesh_path)


def _load_points(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npz":
        with np.load(path) as data:
            xyz = data["xyz"]
    elif ext in {".las", ".laz"}:
        with laspy.open(path) as reader:
            points = reader.read()
            xyz = np.column_stack([points.x, points.y, points.z])
    else:
        raise ValueError(f"Unsupported output format for rendering: {path}")
    return xyz.astype(np.float32, copy=False)


def render_points(name: str, xyz: np.ndarray, max_points: int = 200_000) -> Path:
    if xyz.size == 0:
        raise ValueError(f"No points to render for {name}")
    rng = np.random.default_rng(0)
    if xyz.shape[0] > max_points:
        idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[idx]

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    # Use quantile stretch so nearly flat scenes still show contrast
    z_lo, z_hi = np.quantile(z, [0.02, 0.98])
    if np.isclose(z_lo, z_hi):
        colors = np.full_like(z, 0.5, dtype=np.float32)
    else:
        colors = np.clip((z - z_lo) / (z_hi - z_lo), 0.0, 1.0).astype(np.float32, copy=False)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax_top = fig.add_subplot(2, 1, 1)
    sc = ax_top.scatter(x, y, c=colors, s=1, cmap="viridis")
    ax_top.set_title(f"{name.replace('_', ' ').title()} – XY (top-down)")
    ax_top.set_xlabel("X [m]")
    ax_top.set_ylabel("Y [m]")
    ax_top.set_aspect("equal", adjustable="box")
    # Pad extents so extremely skinny strips still have some breathing room
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_pad = 0.05 * x_range if x_range > 0 else 1.0
    target_y_range = max(y_range, 0.2 * x_range)
    extra_y = max(0.0, target_y_range - y_range)
    y_pad = 0.05 * target_y_range + extra_y / 2.0
    ax_top.set_xlim(x_min - x_pad, x_max + x_pad)
    ax_top.set_ylim(y_min - y_pad, y_max + y_pad)
    fig.colorbar(sc, ax=ax_top, fraction=0.046, pad=0.04, label="Normalized height")

    ax_side = fig.add_subplot(2, 1, 2)
    ax_side.scatter(x, z, c=colors, s=1, cmap="viridis")
    ax_side.set_title("Elevation Profile (XZ)")
    ax_side.set_xlabel("X [m]")
    ax_side.set_ylabel("Z [m]")
    z_min, z_max = np.min(z), np.max(z)
    z_range = z_max - z_min
    z_pad = 0.05 * z_range if z_range > 0 else 1.0
    ax_side.set_ylim(z_min - z_pad, z_max + z_pad)

    fig.tight_layout()
    out_path = IMAGE_DIR / f"{name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_examples(names: List[str], overwrite_outputs: bool, log_level: str, *, full: bool = False) -> None:
    _ensure_dirs()
    example_list = FULL_EXAMPLES if full else PREVIEW_EXAMPLES
    selected = example_list if not names else [spec for spec in example_list if spec.name in names]
    if not selected:
        raise ValueError("No matching examples selected.")
    for spec in selected:
        if spec.config_path is None and spec.mesh_path is None:
            logging.warning("Skipping '%s' (no config or mesh path provided)", spec.name)
            continue
        if spec.config_path is None:
            image_path = IMAGE_DIR / f"{spec.name}.png"
            if image_path.exists() and not overwrite_outputs:
                logging.info("Skipping %s (image exists)", spec.name)
                continue
            logging.info("Rendering raw mesh for '%s'", spec.name)
            xyz = _load_mesh_vertices(spec.mesh_path)  # type: ignore[arg-type]
            image_path = render_points(spec.name, xyz)
            logging.info("Saved %s", image_path)
            continue

        logging.info("Running example '%s'", spec.name)
        out_path = run_example(spec, overwrite=overwrite_outputs, log_level=log_level)
        logging.info("Rendering image for '%s'", spec.name)
        if not out_path.exists():
            logging.warning("Output %s missing, skipping render", out_path)
            continue
        xyz = _load_points(out_path)
        if xyz.size == 0:
            logging.warning("No points generated for %s, skipping image", spec.name)
            continue
        image_path = render_points(spec.name, xyz)
        logging.info("Saved %s", image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Smesh example outputs and preview images.")
    parser.add_argument("--example", "-e", action="append", help="Example name to run (default: all).")
    parser.add_argument("--no-overwrite", action="store_true", help="Skip generating outputs if they already exist.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--full", action="store_true", help="Use full-resolution configs instead of quick previews.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(message)s")
    generate_examples(args.example or [], overwrite_outputs=not args.no_overwrite, log_level=args.log_level, full=args.full)


if __name__ == "__main__":
    main()
