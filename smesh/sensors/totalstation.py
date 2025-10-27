from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from ..core.intersector import RayBundle
from ..core.utils import ensure_unit_vectors
from ..motion.trajectory import Trajectory
from .base import SensorBatch
from .patterns import RasterAzElPattern


@dataclass
class TotalStationSensor:
    pattern: RasterAzElPattern
    trajectory: Trajectory
    max_range_m: float = 1000.0
    boresight_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    lever_arm_t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.boresight_R = np.asarray(self.boresight_R, dtype=np.float64).reshape(3, 3)
        self.lever_arm_t = np.asarray(self.lever_arm_t, dtype=np.float64).reshape(3)
        if self.max_range_m <= 0:
            raise ValueError("max_range_m must be positive")

    def batches(self, rng: Optional[np.random.Generator] = None) -> Iterable[SensorBatch]:
        entries = list(self.trajectory.timeline())
        if not entries:
            raise RuntimeError("Trajectory provided no keyframes for total station sensor")
        pose_time, pose = entries[0]
        pattern_sample = self.pattern.sample(0, start_time_s=0.0)

        dirs_sensor = pattern_sample.directions
        dirs_body = (self.boresight_R @ dirs_sensor.T).T
        dirs_world = (pose.R @ dirs_body.T).T
        dirs_world = ensure_unit_vectors(dirs_world)

        origin = pose.t + pose.R @ self.lever_arm_t
        origins = np.tile(origin, (dirs_world.shape[0], 1))

        rel_time = np.asarray(pattern_sample.relative_time_s, dtype=np.float64)
        gps_time = pose_time + rel_time
        meta = {
            "gps_time": gps_time.astype(np.float64, copy=False),
            "relative_time_s": rel_time,
        }
        for key, value in pattern_sample.meta.items():
            meta[key] = value

        bundle = RayBundle(
            origins=origins.astype(np.float32, copy=False),
            directions=dirs_world.astype(np.float32, copy=False),
            max_range=float(self.max_range_m),
            multi_hit=False,
            meta=meta,
        )
        yield SensorBatch(bundle=bundle, aux=meta)
