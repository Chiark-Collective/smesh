from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..config import ScenarioConfig
from ..core.exporter import LasWriter, NpzWriter, PlyWriter
from ..core.scene import MeshScene
from ..motion.pose import Pose
from ..motion.trajectory import LawnmowerTrajectory, StaticTrajectory, Trajectory
from ..sensors.camera import CameraSensor
from ..sensors.lidar import LidarSensor
from ..sensors.noise import LidarNoise, PhotogrammetryNoise
from ..sensors.patterns import (
    CameraPattern,
    OscillatingMirrorPattern,
    RasterAzElPattern,
    ScanPattern,
    SpinningPattern,
)
from ..sensors.totalstation import TotalStationSensor


def build_trajectory(cfg: ScenarioConfig, scene: MeshScene) -> Trajectory:
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


def build_pattern(cfg: ScenarioConfig) -> ScanPattern:
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


def build_noise(cfg: ScenarioConfig) -> Optional[object]:
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


def build_sensor(
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


def build_writer(cfg: ScenarioConfig):
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

