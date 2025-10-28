from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import numpy as np

from ..core.intersector import RayBundle
from ..core.utils import ensure_unit_vectors
from ..motion.trajectory import Trajectory
from .base import SensorBatch
from .patterns import ScanPattern, PatternSample
from .noise import LidarNoise


@dataclass
class LidarSensor:
    """LiDAR sensor that combines a trajectory with a scan pattern."""

    pattern: ScanPattern
    trajectory: Trajectory
    noise: Optional[LidarNoise] = None
    max_range_m: float = 1000.0
    multi_return: bool = True
    boresight_R: np.ndarray = field(default_factory=lambda: np.eye(3))
    lever_arm_t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    num_lines: Optional[int] = None

    def __post_init__(self) -> None:
        self.boresight_R = np.asarray(self.boresight_R, dtype=np.float64).reshape(3, 3)
        self.lever_arm_t = np.asarray(self.lever_arm_t, dtype=np.float64).reshape(3)
        if self.max_range_m <= 0.0:
            raise ValueError("max_range_m must be positive.")
        if self.num_lines is not None and self.num_lines <= 0:
            raise ValueError("num_lines must be positive when provided.")

    def _line_times(self, cadence: float, timeline: Iterable[tuple[float, object]]) -> np.ndarray:
        times_list = list(timeline)
        if not times_list:
            raise ValueError("Trajectory timeline is empty.")
        traj_start = times_list[0][0]
        traj_end = times_list[-1][0]

        start_time = self.start_time_s if self.start_time_s is not None else traj_start
        end_time = self.end_time_s if self.end_time_s is not None else traj_end
        if end_time < start_time:
            end_time = start_time

        if self.num_lines is not None:
            count = self.num_lines
            if count <= 0:
                raise ValueError("num_lines must be positive when provided.")
            if count == 1:
                return np.asarray([start_time], dtype=np.float64)
            if np.isclose(end_time, start_time):
                offsets = np.arange(count, dtype=np.float64) * max(cadence, 1e-9)
                return start_time + offsets
            return np.linspace(start_time, end_time, count, dtype=np.float64)
        else:
            duration = max(end_time - start_time, 0.0)
            count = max(1, int(np.floor(duration / cadence)) + 1)
            indices = np.arange(count, dtype=np.int64)
            line_times = start_time + indices * cadence
            return line_times

    def batches(self, rng: Optional[np.random.Generator] = None) -> Iterable[SensorBatch]:
        if rng is None:
            rng = np.random.default_rng()

        cadence = self.pattern.cadence_s()
        timeline_entries = list(self.trajectory.timeline())
        line_times = self._line_times(cadence, timeline_entries)

        for line_idx, line_time in enumerate(line_times):
            pose = self.trajectory.sample(float(line_time))
            pattern_sample = self.pattern.sample(line_idx, start_time_s=0.0)
            dirs_sensor = pattern_sample.directions

            if self.noise is not None:
                dirs_sensor = self.noise.jitter_directions(dirs_sensor, rng)

            dirs_body = (self.boresight_R @ dirs_sensor.T).T
            dirs_world = (pose.R @ dirs_body.T).T
            dirs_world = ensure_unit_vectors(dirs_world)

            origin = pose.t + pose.R @ self.lever_arm_t
            origins = np.tile(origin, (dirs_world.shape[0], 1))

            times = line_time + pattern_sample.relative_time_s
            if self.noise is not None:
                times = self.noise.time_jitter(times, rng)

            meta: Dict[str, np.ndarray] = {
                "gps_time": times.astype(np.float64, copy=False),
            }
            for key, value in pattern_sample.meta.items():
                meta[key] = value

            if self.noise is not None:
                keep = self.noise.dropout_mask(dirs_world.shape[0], rng)
                if not np.any(keep):
                    continue
                origins = origins[keep]
                dirs_world = dirs_world[keep]
                meta = {k: v[keep] if len(v) == len(keep) else v for k, v in meta.items()}

            bundle = RayBundle(
                origins=origins.astype(np.float32, copy=False),
                directions=dirs_world.astype(np.float32, copy=False),
                max_range=float(self.max_range_m),
                multi_hit=bool(self.multi_return),
                meta=meta,
            )
            yield SensorBatch(bundle=bundle, aux=meta)
