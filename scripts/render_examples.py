from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import laspy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from smesh.cli.main import _execute_sample  # type: ignore
from smesh.config import load_config

matplotlib.use("Agg")


PREVIEW_EXAMPLES: List[Tuple[str, Path]] = [
    ("aerial_lidar", Path("examples/configs/preview/aerial_lidar_preview.yaml")),
    ("mobile_lidar", Path("examples/configs/preview/mobile_lidar_preview.yaml")),
    ("total_station", Path("examples/configs/preview/total_station_preview.yaml")),
    ("photogrammetry", Path("examples/configs/preview/photogrammetry_preview.yaml")),
]

FULL_EXAMPLES: List[Tuple[str, Path]] = [
    ("aerial_lidar", Path("examples/configs/aerial_lidar.yaml")),
    ("mobile_lidar", Path("examples/configs/mobile_lidar.yaml")),
    ("total_station", Path("examples/configs/total_station.yaml")),
    ("photogrammetry", Path("examples/configs/photogrammetry.yaml")),
]

OUTPUT_DIR = Path("examples/outputs")
IMAGE_DIR = Path("examples/images")


def _ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def run_example(name: str, config_path: Path, overwrite: bool = True, log_level: str = "INFO") -> Path:
    cfg = load_config(config_path)
    ext = cfg.output.format.lower()
    override_ext = "las" if ext == "laz" else ext
    out_path = OUTPUT_DIR / f"{name}.{override_ext}"
    if out_path.exists() and not overwrite:
        logging.info("Skipping %s (output exists)", name)
        return out_path

    _execute_sample(  # type: ignore
        config=config_path,
        output_override=out_path,
        seed_override=None,
        engine_override=None,
        attribute_override=None,
        log_level=log_level,
    )
    return out_path


def _load_points(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in {".npz", ".npz"}:
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

    z = xyz[:, 2]
    z_min, z_max = np.min(z), np.max(z)
    if np.isclose(z_min, z_max):
        colors = np.full_like(z, 0.5, dtype=np.float32)
    else:
        colors = (z - z_min) / (z_max - z_min)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax_top = fig.add_subplot(2, 1, 1)
    sc = ax_top.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=1, cmap="viridis")
    ax_top.set_title(f"{name.replace('_', ' ').title()} – XY (top-down)")
    ax_top.set_xlabel("X [m]")
    ax_top.set_ylabel("Y [m]")
    ax_top.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax_top, fraction=0.046, pad=0.04, label="Normalized height")

    ax_side = fig.add_subplot(2, 1, 2)
    ax_side.scatter(xyz[:, 0], xyz[:, 2], c=colors, s=1, cmap="viridis")
    ax_side.set_title("Elevation Profile (XZ)")
    ax_side.set_xlabel("X [m]")
    ax_side.set_ylabel("Z [m]")
    ax_side.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    out_path = IMAGE_DIR / f"{name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_examples(names: List[str], overwrite_outputs: bool, log_level: str, *, full: bool = False) -> None:
    _ensure_dirs()
    example_list = FULL_EXAMPLES if full else PREVIEW_EXAMPLES
    selected = example_list if not names else [(name, path) for name, path in example_list if name in names]
    if not selected:
        raise ValueError("No matching examples selected.")
    for name, config_path in selected:
        logging.info("Running example '%s'", name)
        out_path = run_example(name, config_path, overwrite=overwrite_outputs, log_level=log_level)
        logging.info("Rendering image for '%s'", name)
        if not out_path.exists():
            logging.warning("Output %s missing, skipping render", out_path)
            continue
        xyz = _load_points(out_path)
        if xyz.size == 0:
            logging.warning("No points generated for %s, skipping image", name)
            continue
        image_path = render_points(name, xyz)
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
