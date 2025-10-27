from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..core.intersector import RayBundle
from ..core.utils import ensure_unit_vectors
from ..motion.trajectory import Trajectory
from .base import SensorBatch
from .patterns import CameraPattern
from .noise import PhotogrammetryNoise


@dataclass
class CameraSensor:
    pattern: CameraPattern
    trajectory: Trajectory
    frame_rate_hz: Optional[float] = None
    exposure_interval_s: Optional[float] = None
    num_frames: Optional[int] = None
    noise: Optional[PhotogrammetryNoise] = None
    boresight_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    lever_arm_t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.boresight_R = np.asarray(self.boresight_R, dtype=np.float64).reshape(3, 3)
        self.lever_arm_t = np.asarray(self.lever_arm_t, dtype=np.float64).reshape(3)
        if self.frame_rate_hz is not None and self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")
        if self.exposure_interval_s is not None and self.exposure_interval_s <= 0:
            raise ValueError("exposure_interval_s must be positive")

    def _frame_times(self) -> np.ndarray:
        entries = list(self.trajectory.timeline())
        if not entries:
            raise RuntimeError("Trajectory provides no keyframes for camera sensor")
        start_time = entries[0][0]
        end_time = entries[-1][0]
        if self.num_frames is not None:
            count = max(1, int(self.num_frames))
            return np.linspace(start_time, end_time, count, dtype=np.float64)
        if self.exposure_interval_s is not None:
            interval = float(self.exposure_interval_s)
        elif self.frame_rate_hz is not None:
            interval = 1.0 / float(self.frame_rate_hz)
        else:
            return np.array([start_time], dtype=np.float64)
        duration = max(end_time - start_time, 0.0)
        count = int(np.floor(duration / interval)) + 1
        return start_time + np.arange(count, dtype=np.float64) * interval

    def batches(self, rng: Optional[np.random.Generator] = None) -> Iterable[SensorBatch]:
        if rng is None:
            rng = np.random.default_rng()

        frame_times = self._frame_times()
        for frame_id, time_s in enumerate(frame_times):
            pose = self.trajectory.sample(float(time_s))
            pattern_sample = self.pattern.sample(frame_id, start_time_s=0.0)
            pixel_uv = np.column_stack([pattern_sample.meta["pixel_u"], pattern_sample.meta["pixel_v"]])

            if self.noise is not None:
                pixel_uv = self.noise.jitter_pixels(pixel_uv, rng)
                pattern_sample.meta["pixel_u"] = pixel_uv[:, 0].astype(np.float32, copy=False)
                pattern_sample.meta["pixel_v"] = pixel_uv[:, 1].astype(np.float32, copy=False)
                dirs_sensor = self.pattern.directions_from_pixels(pixel_uv)
            else:
                dirs_sensor = pattern_sample.directions
            dirs_body = (self.boresight_R @ dirs_sensor.T).T
            dirs_world = (pose.R @ dirs_body.T).T
            dirs_world = ensure_unit_vectors(dirs_world)

            origin = pose.t + pose.R @ self.lever_arm_t
            origins = np.tile(origin, (dirs_world.shape[0], 1))

            gps_time = np.full((dirs_world.shape[0],), time_s, dtype=np.float64)
            meta = {
                "gps_time": gps_time,
                "frame_id": np.full((dirs_world.shape[0],), frame_id, dtype=np.int32),
                "pixel_u": pattern_sample.meta["pixel_u"],
                "pixel_v": pattern_sample.meta["pixel_v"],
            }

            if self.noise is not None and self.noise.sigma_time_s > 0:
                meta["gps_time"] = self.noise.time_jitter(meta["gps_time"], rng)

            bundle = RayBundle(
                origins=origins.astype(np.float32, copy=False),
                directions=dirs_world.astype(np.float32, copy=False),
                max_range=1e6,
                multi_hit=False,
                meta=meta,
            )
            yield SensorBatch(bundle=bundle, aux=meta)
