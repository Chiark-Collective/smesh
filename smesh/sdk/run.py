from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Union

import numpy as np

from ..config import ScenarioConfig, load_config
from ..core.sampler import Sampler, SamplerConfig
from ..core.scene import MeshScene
from ..runtime.builders import (
    build_noise,
    build_pattern,
    build_sensor,
    build_trajectory,
    build_writer,
)


def _ray_bundle_iter(sensor, rng: np.random.Generator) -> Iterable:
    for batch in sensor.batches(rng):
        yield batch.bundle


@dataclass(frozen=True)
class ConfigRunResult:
    """Summary of a sampling run driven by a configuration file."""

    stats: Dict[str, int]
    output_path: Path
    config: ScenarioConfig


def sample_from_config(
    config: Union[str, Path, ScenarioConfig],
    *,
    output: Optional[Path] = None,
    seed: Optional[int] = None,
    engine: Optional[str] = None,
    attributes: Optional[Sequence[str]] = None,
) -> ConfigRunResult:
    """Run a sampling scenario described by a configuration file or object.

    Parameters
    ----------
    config:
        Path to a YAML file or a pre-loaded :class:`~smesh.config.schema.ScenarioConfig`.
    output:
        Optional override for the output file produced by the run. The extension
        drives the format (``.las``, ``.laz``, ``.npz``, or ``.ply``).
    seed:
        Optional RNG seed. Falls back to the value in the config or ``12345``.
    engine:
        Optional override for the intersector/engine selection.
    attributes:
        Optional iterable of attribute names to compute. When omitted the
        configuration's attribute list is used as-is.

    Returns
    -------
    ConfigRunResult
        Includes basic statistics (points, rays), the resolved output path, and
        the resolved configuration object used for the run.
    """

    cfg = load_config(config) if not isinstance(config, ScenarioConfig) else config.model_copy(deep=True)

    if engine:
        cfg.engine = engine
    if attributes is not None:
        cfg.attributes = list(attributes)

    if output is not None:
        out_path = Path(output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.output.path = out_path
        ext = out_path.suffix.lower()
        if ext not in {".las", ".laz", ".npz", ".ply"}:
            raise ValueError(f"Unsupported output extension '{ext}'")
        cfg.output.format = ext.lstrip(".")
        if ext == ".las":
            cfg.output.compress = False
        elif ext == ".laz":
            cfg.output.compress = True if cfg.output.compress is None else cfg.output.compress
    else:
        cfg.output.path = Path(cfg.output.path).resolve()
        cfg.output.path.parent.mkdir(parents=True, exist_ok=True)

    scene = MeshScene(Path(cfg.mesh.path))
    trajectory = build_trajectory(cfg, scene)
    pattern = build_pattern(cfg)
    noise = build_noise(cfg)
    sensor = build_sensor(cfg, trajectory, pattern, noise)

    beam_divergence = cfg.sampler.beam_divergence_mrad
    if beam_divergence is None and hasattr(cfg.sensor, "beam_divergence_mrad"):
        beam_divergence = getattr(cfg.sensor, "beam_divergence_mrad")
    if beam_divergence is None:
        beam_divergence = 0.3

    sampler_cfg = SamplerConfig(
        intersector=cfg.engine or cfg.sampler.intersector,
        batch_size_rays=cfg.sampler.batch_size_rays,
        attributes=list(cfg.attributes),
        beam_divergence_mrad=beam_divergence,
    )
    sampler = Sampler(scene, cfg=sampler_cfg)
    writer = build_writer(cfg)

    run_seed = seed if seed is not None else (cfg.seed if cfg.seed is not None else 12345)
    rng = np.random.default_rng(run_seed)

    try:
        stats = sampler.run_to_writer(writer, _ray_bundle_iter(sensor, rng))
    finally:
        close = getattr(writer, "close", None)
        if callable(close):
            close()

    return ConfigRunResult(stats=stats, output_path=Path(cfg.output.path), config=cfg)

