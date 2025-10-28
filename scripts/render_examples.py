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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from smesh.cli.main import _execute_sample  # type: ignore
from smesh.config import load_config
from smesh.core.scene import MeshScene

try:  # pragma: no cover - optional dependency
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy  # type: ignore
    _HAVE_VTK = True
except Exception:  # pragma: no cover - optional dependency
    vtk = None  # type: ignore
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


def _parse_ascii_ply_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    with path.open("r", encoding="utf-8") as fh:
        vertex_count = 0
        face_count = 0
        vertex_props: list[str] = []
        reading_vertices = False
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"Unexpected EOF while reading {path}")
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
                reading_vertices = True
            elif stripped.startswith("element face"):
                face_count = int(stripped.split()[-1])
                reading_vertices = False
            elif stripped.startswith("property") and reading_vertices:
                vertex_props.append(stripped.split()[-1])
            elif stripped == "end_header":
                break

        if vertex_count == 0:
            raise ValueError(f"PLY file {path} missing vertex data")

        prop_index = {name: idx for idx, name in enumerate(vertex_props)}
        required = {"x", "y", "z"}
        if not required.issubset(prop_index):
            raise ValueError(f"PLY file {path} missing coordinates {required}")

        has_colors = all(name in prop_index for name in ("red", "green", "blue"))
        vertices = np.zeros((vertex_count, 3), dtype=np.float32)
        colors: Optional[np.ndarray]
        colors = np.zeros((vertex_count, 3), dtype=np.uint8) if has_colors else None

        for idx in range(vertex_count):
            parts = fh.readline().split()
            if len(parts) < len(vertex_props):
                raise ValueError(f"Malformed vertex line in {path}")
            vertices[idx, 0] = float(parts[prop_index["x"]])
            vertices[idx, 1] = float(parts[prop_index["y"]])
            vertices[idx, 2] = float(parts[prop_index["z"]])
            if colors is not None:
                colors[idx, 0] = int(parts[prop_index["red"]])
                colors[idx, 1] = int(parts[prop_index["green"]])
                colors[idx, 2] = int(parts[prop_index["blue"]])

        faces: list[list[int]] = []
        for _ in range(face_count):
            parts = fh.readline().split()
            if not parts:
                continue
            polygon_size = int(parts[0])
            indices = [int(p) for p in parts[1:1 + polygon_size]]
            if polygon_size < 3:
                continue
            for j in range(1, polygon_size - 1):
                faces.append([indices[0], indices[j], indices[j + 1]])

    faces_array = np.asarray(faces, dtype=np.int32) if faces else np.empty((0, 3), dtype=np.int32)
    return vertices, faces_array, colors


def _render_mesh_vtk(mesh_path: Path, out_path: Path) -> None:
    if not _HAVE_VTK or vtk is None:
        raise RuntimeError("VTK is not available for mesh rendering")

    scene = MeshScene(mesh_path=str(mesh_path))
    poly = scene.vtk_polydata()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0.78, 0.78, 0.78)
    prop.SetAmbient(0.25)
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.25)
    prop.SetSpecularPower(20.0)
    prop.EdgeVisibilityOn()
    prop.SetEdgeColor(0.45, 0.45, 0.45)
    prop.SetLineWidth(0.5)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1.0, 1.0, 1.0)

    bounds = poly.GetBounds()
    center = (
        0.5 * (bounds[0] + bounds[1]),
        0.5 * (bounds[2] + bounds[3]),
        0.5 * (bounds[4] + bounds[5]),
    )
    radius = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) * 0.5
    radius = radius if radius > 0 else 1.0

    camera = renderer.GetActiveCamera()
    distance = radius * 2.6
    camera.SetPosition(center[0] + distance, center[1] - distance, center[2] + distance * 0.9)
    camera.SetFocalPoint(*center)
    camera.SetViewUp(0.0, 0.0, 1.0)
    camera.SetClippingRange(0.01, distance * 6.0)

    light = vtk.vtkLight()
    light.SetLightTypeToSceneLight()
    light.SetPosition(center[0] + radius * 1.5, center[1] + radius * 1.2, center[2] + radius * 2.5)
    light.SetFocalPoint(*center)
    light.SetIntensity(0.9)
    renderer.AddLight(light)

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetOffScreenRendering(True)
    render_window.SetSize(900, 700)
    render_window.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)
    w2i.Update()

    png_writer = vtk.vtkPNGWriter()
    png_writer.SetFileName(str(out_path))
    png_writer.SetInputConnection(w2i.GetOutputPort())
    png_writer.Write()


def _render_mesh_matplotlib(mesh_path: Path, out_path: Path) -> None:
    vertices, faces, colors = _parse_ascii_ply_mesh(mesh_path)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    if faces.size > 0:
        tris = vertices[faces]
        if colors is not None:
            face_colors = colors[faces].mean(axis=1) / 255.0
        else:
            face_colors = np.full((faces.shape[0], 3), 0.75, dtype=np.float32)
        if face_colors.shape[1] == 3:
            alpha = np.full((face_colors.shape[0], 1), 1.0, dtype=np.float32)
            face_colors = np.concatenate([face_colors, alpha], axis=1)
        collection = Poly3DCollection(tris, facecolors=face_colors, edgecolors=(0.3, 0.3, 0.3, 0.4), linewidths=0.2)
        ax.add_collection3d(collection)
    else:
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c=vertices[:, 2], cmap="viridis", s=2)

    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    centers = (mins + maxs) * 0.5
    spans = maxs - mins
    max_range = spans.max() if np.any(spans > 0) else 1.0

    ax.set_xlim(centers[0] - max_range * 0.6, centers[0] + max_range * 0.6)
    ax.set_ylim(centers[1] - max_range * 0.6, centers[1] + max_range * 0.6)
    ax.set_zlim(centers[2] - max_range * 0.6, centers[2] + max_range * 0.6)

    ax.view_init(elev=35.0, azim=-45.0)
    ax.set_box_aspect((spans[0] + 1e-6, spans[1] + 1e-6, spans[2] + 1e-6))
    ax.set_facecolor((1.0, 1.0, 1.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticks([])
    ax.grid(False)
    ax.set_title("Raw Mesh Perspective", pad=12)

    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


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


def render_raw_mesh(name: str, mesh_path: Path, overwrite: bool) -> Path:
    image_path = IMAGE_DIR / f"{name}.png"
    if image_path.exists() and not overwrite:
        return image_path

    if mesh_path is None:
        raise ValueError("Raw mesh rendering requires a mesh path")

    if _HAVE_VTK and vtk is not None:
        try:
            _render_mesh_vtk(mesh_path, image_path)
            return image_path
        except Exception as exc:
            logging.warning("VTK rendering failed for %s: %s; falling back to matplotlib", mesh_path, exc)

    _render_mesh_matplotlib(mesh_path, image_path)
    return image_path


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
            logging.info("Rendering raw mesh for '%s'", spec.name)
            image_path = render_raw_mesh(spec.name, spec.mesh_path, overwrite_outputs)  # type: ignore[arg-type]
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
