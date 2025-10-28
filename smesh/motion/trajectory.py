from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Literal

import numpy as np

from .pose import Pose

try:  # optional import for type checking / hints
    from smesh.core.scene import MeshScene  # pragma: no cover
except Exception:  # pragma: no cover - avoids circular import at runtime
    MeshScene = None  # type: ignore


class Trajectory:
    """Base interface for platform motion."""

    def sample(self, t: float) -> Pose:
        raise NotImplementedError

    def timeline(self) -> Iterable[tuple[float, Pose]]:
        raise NotImplementedError


@dataclass
class StaticTrajectory(Trajectory):
    """A trajectory with a single, fixed pose."""

    pose: Pose
    start_time_s: float = 0.0

    def sample(self, t: float) -> Pose:
        return self.pose

    def timeline(self) -> Iterable[tuple[float, Pose]]:
        yield (self.start_time_s, self.pose)


class PolylineTrajectory(Trajectory):
    """Piecewise-linear trajectory through waypoints at constant speed."""

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        speed_mps: float,
        start_time_s: float = 0.0,
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("PolylineTrajectory requires at least two waypoints.")
        if speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive.")

        self._points = np.asarray(waypoints, dtype=np.float64)
        self._speed = float(speed_mps)
        self._start_time = float(start_time_s)

        seg_vecs = np.diff(self._points, axis=0)
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        if np.any(seg_lengths == 0):
            raise ValueError("Consecutive waypoints must be distinct.")

        seg_durations = seg_lengths / self._speed
        times = np.concatenate([[0.0], np.cumsum(seg_durations)])

        self._segment_vectors = seg_vecs
        self._segment_lengths = seg_lengths
        self._segment_durations = seg_durations
        self._times = self._start_time + times
        self._poses = self._build_keyframes()

    def _build_keyframes(self) -> List[Pose]:
        poses: List[Pose] = []
        # Orientation: yaw aligned with horizontal projection of segment
        seg_dirs = np.vstack(
            [self._segment_vectors, self._segment_vectors[-1]]
        )  # reuse last for final pose
        for idx, point in enumerate(self._points):
            dir_vec = seg_dirs[idx]
            yaw = np.arctan2(dir_vec[1], dir_vec[0])
            pose = Pose.from_xyz_rpy(tuple(point), (0.0, 0.0, np.degrees(yaw)))
            poses.append(pose)
        return poses

    def sample(self, t: float) -> Pose:
        if t <= self._times[0]:
            return self._poses[0]
        if t >= self._times[-1]:
            return self._poses[-1]

        idx = np.searchsorted(self._times, t, side="right") - 1
        t0, t1 = self._times[idx], self._times[idx + 1]
        alpha = (t - t0) / max(t1 - t0, 1e-9)
        p0 = self._points[idx]
        p1 = self._points[idx + 1]
        pos = (1.0 - alpha) * p0 + alpha * p1

        seg_vec = self._segment_vectors[idx]
        yaw = np.arctan2(seg_vec[1], seg_vec[0])
        pose = Pose.from_xyz_rpy(tuple(pos), (0.0, 0.0, np.degrees(yaw)))
        return pose

    def timeline(self) -> Iterable[tuple[float, Pose]]:
        for t, pose in zip(self._times, self._poses):
            yield (float(t), pose)


class LawnmowerTrajectory(Trajectory):
    """Generates a lawnmower (serpentine) flight pattern over a scene."""

    def __init__(
        self,
        scene: "MeshScene",
        altitude_m: float,
        speed_mps: float,
        line_spacing_m: float,
        heading_deg: float = 0.0,
        start_time_s: float = 0.0,
        altitude_mode: Literal["above_top", "above_ground", "absolute"] = "above_top",
    ) -> None:
        if altitude_mode not in {"above_top", "above_ground", "absolute"}:
            raise ValueError("altitude_mode must be one of 'above_top', 'above_ground', or 'absolute'.")
        if altitude_mode != "absolute" and altitude_m <= 0.0:
            raise ValueError("altitude_m must be positive for relative altitude modes.")
        if speed_mps <= 0.0:
            raise ValueError("speed_mps must be positive.")
        if line_spacing_m <= 0.0:
            raise ValueError("line_spacing_m must be positive.")

        bounds = scene.bounds()
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        if altitude_mode == "above_top":
            altitude = float(zmax + altitude_m)
        elif altitude_mode == "above_ground":
            if altitude_m < 0.0:
                raise ValueError("altitude_m must be non-negative when altitude_mode='above_ground'.")
            altitude = float(zmin + altitude_m)
        else:  # absolute
            altitude = float(altitude_m)

        heading_rad = np.deg2rad(heading_deg)
        forward = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        cross = np.array([-np.sin(heading_rad), np.cos(heading_rad)])

        corners = np.array(
            [
                [xmin, ymin],
                [xmin, ymax],
                [xmax, ymin],
                [xmax, ymax],
            ],
            dtype=np.float64,
        )
        proj_forward = corners @ forward
        proj_cross = corners @ cross

        min_f, max_f = proj_forward.min(), proj_forward.max()
        min_c, max_c = proj_cross.min(), proj_cross.max()
        width_cross = max_c - min_c
        num_lines = max(1, int(np.ceil(width_cross / line_spacing_m)) + 1)

        points: List[np.ndarray] = []
        for line_idx in range(num_lines):
            c_val = min_c + line_idx * line_spacing_m
            start_f, end_f = min_f, max_f
            if line_idx % 2 == 1:
                start_f, end_f = end_f, start_f

            start_xy = forward * start_f + cross * c_val
            end_xy = forward * end_f + cross * c_val
            start_pt = np.array([start_xy[0], start_xy[1], altitude])
            end_pt = np.array([end_xy[0], end_xy[1], altitude])

            if not points or not np.allclose(points[-1], start_pt):
                points.append(start_pt)
            points.append(end_pt)

        self._poly = PolylineTrajectory(points, speed_mps=speed_mps, start_time_s=start_time_s)

    def sample(self, t: float) -> Pose:
        return self._poly.sample(t)

    def timeline(self) -> Iterable[tuple[float, Pose]]:
        yield from self._poly.timeline()
