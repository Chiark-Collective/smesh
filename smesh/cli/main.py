from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import typer

from ..config import ScenarioConfig, load_config
from ..core.exporter import LasWriter, NpzWriter, PlyWriter
from ..core.sampler import Sampler, SamplerConfig
from ..core.scene import MeshScene
from ..motion.pose import Pose
from ..motion.trajectory import LawnmowerTrajectory, StaticTrajectory, Trajectory
from ..sensors.lidar import LidarSensor
from ..sensors.totalstation import TotalStationSensor
from ..sensors.camera import CameraSensor
from ..sensors.noise import LidarNoise, PhotogrammetryNoise
from ..sensors.patterns import (
    OscillatingMirrorPattern,
    RasterAzElPattern,
    ScanPattern,
    SpinningPattern,
    CameraPattern,
)
from ..examples.synthetic import generate_mesh

app = typer.Typer(help="Smesh sampling utilities")
mesh_app = typer.Typer(help="Synthetic mesh helpers")
app.add_typer(mesh_app, name="mesh")

DEFAULT_LIDAR_ATTRIBUTES = ["range", "incidence", "scan_angle", "intensity", "returns", "gps_time"]


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="[%(levelname)s] %(message)s")
    logging.getLogger("smesh").setLevel(numeric)


def _build_trajectory(cfg: ScenarioConfig, scene: MeshScene) -> Trajectory:
    traj_cfg = cfg.trajectory
    if traj_cfg.kind == "static":
        pose = Pose.from_xyz_rpy(traj_cfg.xyz, traj_cfg.rpy_deg)
        return StaticTrajectory(pose, start_time_s=traj_cfg.start_time_s)
    if traj_cfg.kind == "lawnmower":
        return LawnmowerTrajectory(
            scene,
            altitude_m=traj_cfg.altitude_m,
            speed_mps=traj_cfg.speed_mps,
            line_spacing_m=traj_cfg.line_spacing_m,
            heading_deg=traj_cfg.heading_deg,
            start_time_s=traj_cfg.start_time_s,
            altitude_mode=getattr(traj_cfg, "altitude_mode", "above_top"),
        )
    raise ValueError(f"Unsupported trajectory kind: {traj_cfg.kind}")


def _build_pattern(cfg: ScenarioConfig) -> ScanPattern:
    pattern_cfg = cfg.pattern
    if pattern_cfg is None:
        raise ValueError("Scenario requires a pattern configuration")
    if pattern_cfg.kind == "oscillating":
        return OscillatingMirrorPattern(
            fov_deg=pattern_cfg.fov_deg,
            line_rate_hz=pattern_cfg.line_rate_hz,
            pulses_per_line=pattern_cfg.pulses_per_line,
        )
    if pattern_cfg.kind == "spinning":
        return SpinningPattern(
            vertical_angles_deg=pattern_cfg.vertical_angles_deg,
            rpm=pattern_cfg.rpm,
            prf_hz=pattern_cfg.prf_hz,
        )
    if pattern_cfg.kind == "raster":
        return RasterAzElPattern(
            az_range_deg=pattern_cfg.az_range_deg,
            el_range_deg=pattern_cfg.el_range_deg,
            step_deg=pattern_cfg.step_deg,
        )
    if pattern_cfg.kind == "camera":
        return CameraPattern(
            resolution_px=pattern_cfg.resolution_px,
            focal_px=pattern_cfg.focal_px,
            principal_px=pattern_cfg.principal_px,
            pixel_stride=pattern_cfg.pixel_stride,
        )
    raise ValueError(f"Unsupported pattern kind: {pattern_cfg.kind}")


def _build_noise(cfg: ScenarioConfig) -> Optional[object]:
    noise_cfg = cfg.noise
    if noise_cfg is None:
        return None
    if noise_cfg.kind == "lidar":
        return LidarNoise(
            sigma_range_m=noise_cfg.sigma_range_m,
            sigma_angle_deg=noise_cfg.sigma_angle_deg,
            keep_prob=noise_cfg.keep_prob,
            sigma_time_s=noise_cfg.sigma_time_s,
        )
    if noise_cfg.kind == "photogrammetry":
        return PhotogrammetryNoise(
            pixel_sigma=noise_cfg.pixel_sigma,
            sigma_time_s=noise_cfg.sigma_time_s,
        )
    raise ValueError(f"Unsupported noise model: {noise_cfg.kind}")


def _build_sensor(
    cfg: ScenarioConfig,
    trajectory: Trajectory,
    pattern: ScanPattern,
    noise: Optional[object],
):
    sensor_cfg = cfg.sensor
    if sensor_cfg.kind == "lidar":
        boresight_R = np.eye(3)
        if sensor_cfg.boresight_rpy_deg is not None:
            pose = Pose.from_xyz_rpy((0.0, 0.0, 0.0), sensor_cfg.boresight_rpy_deg)
            boresight_R = pose.R

        lever_arm = np.zeros(3)
        if sensor_cfg.lever_arm_m is not None:
            lever_arm = np.asarray(sensor_cfg.lever_arm_m, dtype=np.float64)

        lidar_noise = noise if isinstance(noise, LidarNoise) else None
        return LidarSensor(
            pattern=pattern,
            trajectory=trajectory,
            noise=lidar_noise,
            max_range_m=sensor_cfg.max_range_m,
            multi_return=sensor_cfg.multi_return,
            boresight_R=boresight_R,
            lever_arm_t=lever_arm,
            num_lines=sensor_cfg.num_lines,
        )

    if sensor_cfg.kind == "totalstation":
        boresight_R = np.eye(3)
        if sensor_cfg.boresight_rpy_deg is not None:
            pose = Pose.from_xyz_rpy((0.0, 0.0, 0.0), sensor_cfg.boresight_rpy_deg)
            boresight_R = pose.R
        lever_arm = np.zeros(3)
        if sensor_cfg.lever_arm_m is not None:
            lever_arm = np.asarray(sensor_cfg.lever_arm_m, dtype=np.float64)
        if not isinstance(pattern, RasterAzElPattern):
            raise ValueError("Total station sensor requires raster pattern")
        return TotalStationSensor(
            pattern=pattern,
            trajectory=trajectory,
            max_range_m=sensor_cfg.max_range_m,
            boresight_R=boresight_R,
            lever_arm_t=lever_arm,
        )

    if sensor_cfg.kind == "camera":
        boresight_R = np.eye(3)
        if sensor_cfg.boresight_rpy_deg is not None:
            pose = Pose.from_xyz_rpy((0.0, 0.0, 0.0), sensor_cfg.boresight_rpy_deg)
            boresight_R = pose.R
        lever_arm = np.zeros(3)
        if sensor_cfg.lever_arm_m is not None:
            lever_arm = np.asarray(sensor_cfg.lever_arm_m, dtype=np.float64)
        if not isinstance(pattern, CameraPattern):
            raise ValueError("Camera sensor requires camera pattern")
        cam_noise = noise if isinstance(noise, PhotogrammetryNoise) else None
        return CameraSensor(
            pattern=pattern,
            trajectory=trajectory,
            frame_rate_hz=sensor_cfg.frame_rate_hz,
            exposure_interval_s=sensor_cfg.exposure_interval_s,
            num_frames=sensor_cfg.num_frames,
            noise=cam_noise,
            boresight_R=boresight_R,
            lever_arm_t=lever_arm,
        )

    raise ValueError(f"Unsupported sensor kind: {sensor_cfg.kind}")


def _build_writer(cfg: ScenarioConfig):
    out_cfg = cfg.output
    format_lower = out_cfg.format.lower()
    if format_lower in {"las", "laz"}:
        compress = out_cfg.compress
        if compress is None:
            compress = format_lower == "laz"
        return LasWriter(
            str(out_cfg.path),
            point_format=out_cfg.point_format,
            compress=compress,
        )
    if format_lower == "npz":
        return NpzWriter(str(out_cfg.path))
    if format_lower == "ply":
        return PlyWriter(str(out_cfg.path))
    raise ValueError(f"Unsupported output format: {out_cfg.format}")


def _ray_bundle_iter(sensor, rng: np.random.Generator) -> Iterable:
    for batch in sensor.batches(rng):
        yield batch.bundle


def _execute_sample(
    config: Path,
    output_override: Optional[Path],
    seed_override: Optional[int],
    engine_override: Optional[str],
    attribute_override: Optional[List[str]],
    log_level: str,
) -> None:
    _configure_logging(log_level)
    cfg = load_config(config)
    if engine_override:
        cfg.engine = engine_override
    if attribute_override:
        cfg.attributes = list(attribute_override)
    if output_override is not None:
        out = output_override.resolve()
        cfg.output.path = out
        ext = out.suffix.lower()
        if ext in {".las", ".laz", ".npz", ".ply"}:
            cfg.output.format = ext.lstrip(".")
            if ext == ".las":
                cfg.output.compress = False
            if ext == ".laz":
                cfg.output.compress = True if cfg.output.compress is None else cfg.output.compress
        else:
            raise typer.BadParameter(f"Unsupported output extension '{ext}'", param_hint="--output")

    scene = MeshScene(cfg.mesh.path)
    trajectory = _build_trajectory(cfg, scene)
    pattern = _build_pattern(cfg)
    noise = _build_noise(cfg)
    sensor = _build_sensor(cfg, trajectory, pattern, noise)

    beam_divergence = cfg.sampler.beam_divergence_mrad
    if beam_divergence is None and hasattr(cfg.sensor, "beam_divergence_mrad"):
        beam_divergence = getattr(cfg.sensor, "beam_divergence_mrad")
    if beam_divergence is None:
        beam_divergence = 0.3

    sampler_cfg = SamplerConfig(
        intersector=cfg.engine or cfg.sampler.intersector,
        batch_size_rays=cfg.sampler.batch_size_rays,
        attributes=cfg.attributes,
        beam_divergence_mrad=beam_divergence,
    )
    sampler = Sampler(scene, cfg=sampler_cfg)
    writer = _build_writer(cfg)

    seed = seed_override if seed_override is not None else (cfg.seed if cfg.seed is not None else 12345)
    rng = np.random.default_rng(seed)
    stats = sampler.run_to_writer(writer, _ray_bundle_iter(sensor, rng))
    typer.echo(f"Completed {stats['points']} points from {stats['rays']} rays → {cfg.output.path}")


@app.command("sample")
def sample(
    config: Path = typer.Argument(..., exists=True, readable=True, help="Path to YAML configuration file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Override output path (extension sets format)."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override random seed."),
    engine: Optional[str] = typer.Option(None, "--engine", help="Override intersector/engine choice."),
    attribute: Optional[List[str]] = typer.Option(None, "--attribute", "-a", help="Override attribute list."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (e.g. INFO, DEBUG)."),
) -> None:
    """Run a sampling scenario specified by a YAML config."""

    attribute_override = list(attribute) if attribute else None
    _execute_sample(config, output, seed, engine, attribute_override, log_level)


@app.command("run")
def run(
    config: Path = typer.Argument(..., exists=True, readable=True, help="Path to YAML configuration file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Override output path (extension sets format)."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override random seed."),
    engine: Optional[str] = typer.Option(None, "--engine", help="Override intersector/engine choice."),
    attribute: Optional[List[str]] = typer.Option(None, "--attribute", "-a", help="Override attribute list."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (e.g. INFO, DEBUG)."),
) -> None:
    """Alias for `sample`"""

    attribute_override = list(attribute) if attribute else None
    _execute_sample(config, output, seed, engine, attribute_override, log_level)


@app.command("sample-lidar")
def sample_lidar_cli(
    mesh: Path = typer.Option(Path("examples/meshes/preview_scene.ply"), "--mesh", help="Input mesh path.", exists=True, file_okay=True, dir_okay=False, readable=True),
    output: Path = typer.Option(Path("examples/outputs/aerial_lidar_cli.las"), "--output", "-o", help="Output point cloud path (.las/.laz/.npz/.ply)."),
    altitude_m: float = typer.Option(40.0, "--altitude-m", help="Altitude parameter passed to the lawnmower trajectory."),
    altitude_mode: str = typer.Option("above_top", "--altitude-mode", help="Altitude interpretation: above_top, above_ground, or absolute."),
    speed_mps: float = typer.Option(20.0, "--speed-mps", help="Platform speed in metres per second."),
    line_spacing_m: float = typer.Option(30.0, "--line-spacing-m", help="Cross-track spacing between sweeps."),
    heading_deg: float = typer.Option(0.0, "--heading-deg", help="Heading of the flight path in degrees."),
    fov_deg: float = typer.Option(50.0, "--fov-deg", help="Oscillating mirror field of view in degrees."),
    line_rate_hz: float = typer.Option(20.0, "--line-rate-hz", help="Oscillation line rate in Hz."),
    pulses_per_line: int = typer.Option(200, "--pulses-per-line", help="Pulses emitted per line."),
    max_range_m: float = typer.Option(200.0, "--max-range-m", help="Maximum range for valid returns."),
    beam_divergence_mrad: float = typer.Option(0.3, "--beam-divergence-mrad", help="Beam divergence used for footprint estimation."),
    num_lines: Optional[int] = typer.Option(None, "--num-lines", help="Number of scan lines to distribute across the trajectory."),
    sigma_range_m: float = typer.Option(0.05, "--sigma-range-m", help="Per-ray range noise (metres)."),
    sigma_angle_deg: float = typer.Option(0.1, "--sigma-angle-deg", help="Per-ray angular noise (degrees)."),
    keep_prob: float = typer.Option(0.99, "--keep-prob", help="Probability of keeping each ray after dropout."),
    sigma_time_s: float = typer.Option(0.0, "--sigma-time-s", help="Timestamp jitter applied by the noise model (seconds)."),
    batch_size_rays: int = typer.Option(20_000, "--batch-size", help="Sampler batch size in rays."),
    seed: int = typer.Option(101, "--seed", help="Random seed for deterministic sampling."),
    attribute: List[str] = typer.Option([], "--attribute", "-a", help="Attributes to compute (defaults to preview set)."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (e.g. INFO, DEBUG)."),
) -> None:
    """Quick aerial-style LiDAR sampling driven entirely from CLI options."""

    if keep_prob < 0.0 or keep_prob > 1.0:
        raise typer.BadParameter("keep_prob must be within [0, 1].", param_hint="--keep-prob")
    allowed_modes = {"above_top", "above_ground", "absolute"}
    if altitude_mode not in allowed_modes:
        raise typer.BadParameter(f"altitude_mode must be one of {sorted(allowed_modes)}.", param_hint="--altitude-mode")

    attrs = attribute or DEFAULT_LIDAR_ATTRIBUTES
    _configure_logging(log_level)

    mesh = mesh.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    scene = MeshScene(mesh)
    trajectory = LawnmowerTrajectory(
        scene,
        altitude_m=altitude_m,
        speed_mps=speed_mps,
        line_spacing_m=line_spacing_m,
        heading_deg=heading_deg,
        start_time_s=0.0,
        altitude_mode=altitude_mode,  # type: ignore[arg-type]
    )
    pattern = OscillatingMirrorPattern(
        fov_deg=fov_deg,
        line_rate_hz=line_rate_hz,
        pulses_per_line=pulses_per_line,
    )

    noise = None
    if sigma_range_m > 0 or sigma_angle_deg > 0 or sigma_time_s > 0 or keep_prob < 1.0:
        noise = LidarNoise(
            sigma_range_m=sigma_range_m,
            sigma_angle_deg=sigma_angle_deg,
            keep_prob=keep_prob,
            sigma_time_s=sigma_time_s,
        )

    sensor = LidarSensor(
        pattern=pattern,
        trajectory=trajectory,
        noise=noise,
        max_range_m=max_range_m,
        multi_return=True,
        num_lines=num_lines,
    )

    sampler_cfg = SamplerConfig(
        intersector="auto",
        batch_size_rays=batch_size_rays,
        attributes=list(attrs),
        beam_divergence_mrad=beam_divergence_mrad,
    )
    sampler = Sampler(scene, cfg=sampler_cfg)

    fmt = output.suffix.lower()
    if fmt == ".las":
        writer = LasWriter(str(output), compress=False)
    elif fmt == ".laz":
        writer = LasWriter(str(output), compress=True)
    elif fmt == ".npz":
        writer = NpzWriter(str(output))
    elif fmt == ".ply":
        writer = PlyWriter(str(output))
    else:
        raise typer.BadParameter("Output must end with .las, .laz, .npz, or .ply", param_hint="--output")

    rng = np.random.default_rng(seed)
    stats = sampler.run_to_writer(writer, _ray_bundle_iter(sensor, rng))
    typer.echo(f"Completed {stats['points']} points from {stats['rays']} rays → {output}")


@mesh_app.command("generate")
def mesh_generate(
    output: Path = typer.Argument(..., help="Output mesh path (.ply)."),
    preset: str = typer.Option("demo", "--preset", help="Synthetic mesh preset (demo, plane, ramp)."),
    size: float = typer.Option(10.0, "--size", help="Scene extent scaling factor."),
) -> None:
    """Generate a synthetic mesh useful for sampling demos."""

    out = output.resolve()
    generate_mesh(preset=preset, size=size, path=out)
    typer.echo(f"Wrote synthetic mesh to {out}")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
