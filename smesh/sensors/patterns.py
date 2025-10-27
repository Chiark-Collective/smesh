from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

from ..core.utils import ensure_unit_vectors


@dataclass
class PatternSample:
    """Bundle of unit directions and per-ray metadata from a scan pattern."""

    directions: np.ndarray
    relative_time_s: np.ndarray
    meta: Dict[str, np.ndarray]


class ScanPattern:
    """Base class for generating sensor-frame ray directions."""

    def cadence_s(self) -> float:
        """Duration represented by each pattern sample."""
        raise NotImplementedError

    def sample(self, step_index: int, *, start_time_s: float = 0.0) -> PatternSample:
        raise NotImplementedError


class OscillatingMirrorPattern(ScanPattern):
    """1D oscillating mirror pattern for aerial line scanners."""

    def __init__(self, fov_deg: float, line_rate_hz: float, pulses_per_line: int) -> None:
        if fov_deg <= 0.0:
            raise ValueError("fov_deg must be positive.")
        if line_rate_hz <= 0.0:
            raise ValueError("line_rate_hz must be positive.")
        if pulses_per_line <= 0:
            raise ValueError("pulses_per_line must be positive.")

        self.fov_deg = float(fov_deg)
        self.line_rate_hz = float(line_rate_hz)
        self.pulses_per_line = int(pulses_per_line)
        self._line_period = 1.0 / self.line_rate_hz

    @property
    def line_period_s(self) -> float:
        return self._line_period

    def cadence_s(self) -> float:
        return self._line_period

    def sample(self, step_index: int, *, start_time_s: float = 0.0) -> PatternSample:
        scan_angles = np.linspace(
            -0.5 * self.fov_deg,
            0.5 * self.fov_deg,
            self.pulses_per_line,
            dtype=np.float64,
        )
        angles_rad = np.deg2rad(scan_angles)

        dirs = np.column_stack(
            [
                np.sin(angles_rad),
                np.zeros_like(angles_rad),
                -np.cos(angles_rad),
            ]
        )
        dirs = ensure_unit_vectors(dirs.astype(np.float64, copy=False))

        offsets = np.linspace(
            0.0,
            self._line_period,
            self.pulses_per_line,
            endpoint=False,
            dtype=np.float64,
        )

        meta: Dict[str, np.ndarray] = {
            "scan_angle_deg": scan_angles.astype(np.float32, copy=False),
            "scanline_id": np.full(self.pulses_per_line, step_index, dtype=np.uint32),
            "pulse_index": np.arange(self.pulses_per_line, dtype=np.uint32),
        }
        rel_time = offsets + start_time_s
        return PatternSample(directions=dirs, relative_time_s=rel_time, meta=meta)


class SpinningPattern(ScanPattern):
    """360° spinning LiDAR pattern with discrete vertical channels."""

    def __init__(self, vertical_angles_deg: Sequence[float], rpm: float, prf_hz: int) -> None:
        if not vertical_angles_deg:
            raise ValueError("Provide at least one vertical angle.")
        if rpm <= 0.0:
            raise ValueError("rpm must be positive.")
        if prf_hz <= 0:
            raise ValueError("prf_hz must be positive.")

        self.vertical_angles_deg = np.asarray(vertical_angles_deg, dtype=np.float64)
        self.rpm = float(rpm)
        self.prf_hz = int(prf_hz)
        self._rev_period = 60.0 / self.rpm
        pulses_per_rev = max(1, int(round(self.prf_hz * self._rev_period)))
        self._pulses_per_rev = pulses_per_rev
        self._azimuth_step = 360.0 / pulses_per_rev

    @property
    def pulses_per_revolution(self) -> int:
        return self._pulses_per_rev

    def sample(self, step_index: int, *, start_time_s: float = 0.0) -> PatternSample:
        azimuths = (np.arange(self._pulses_per_rev, dtype=np.float64) * self._azimuth_step) % 360.0
        times = np.arange(self._pulses_per_rev, dtype=np.float64) / float(self.prf_hz)

        dirs_list = []
        meta_scan_angle = []
        meta_channel = []
        meta_revolution = []
        meta_azimuth = []
        meta_pulse = []

        for pulse_idx, (az_deg, t_offset) in enumerate(zip(azimuths, times)):
            az_rad = np.deg2rad(az_deg)
            cos_az = np.cos(az_rad)
            sin_az = np.sin(az_rad)
            for channel_id, vert_deg in enumerate(self.vertical_angles_deg):
                el_rad = np.deg2rad(vert_deg)
                cos_el = np.cos(el_rad)
                sin_el = np.sin(el_rad)
                direction = np.array(
                    [
                        cos_el * cos_az,
                        cos_el * sin_az,
                        sin_el,
                    ],
                    dtype=np.float64,
                )
                dirs_list.append(direction)
                meta_scan_angle.append(vert_deg)
                meta_channel.append(channel_id)
                meta_revolution.append(step_index)
                meta_azimuth.append(az_deg)
                meta_pulse.append(pulse_idx)

        dirs = ensure_unit_vectors(np.asarray(dirs_list, dtype=np.float64))
        rel_time = (
            start_time_s
            + np.repeat(times, len(self.vertical_angles_deg))
            + step_index * self._rev_period
        )
        meta = {
            "scan_angle_deg": np.asarray(meta_scan_angle, dtype=np.float32),
            "channel_id": np.asarray(meta_channel, dtype=np.uint16),
            "revolution_id": np.asarray(meta_revolution, dtype=np.uint32),
            "azimuth_deg": np.asarray(meta_azimuth, dtype=np.float32),
            "pulse_index": np.asarray(meta_pulse, dtype=np.uint32),
        }
        return PatternSample(directions=dirs, relative_time_s=rel_time, meta=meta)

    def cadence_s(self) -> float:
        return self._rev_period


class RasterAzElPattern(ScanPattern):
    """Raster pattern over azimuth/elevation grid (total station style)."""

    def __init__(
        self,
        az_range_deg: tuple[float, float],
        el_range_deg: tuple[float, float],
        step_deg: float,
    ) -> None:
        if step_deg <= 0.0:
            raise ValueError("step_deg must be positive.")
        self.az_min, self.az_max = map(float, az_range_deg)
        self.el_min, self.el_max = map(float, el_range_deg)
        self.step_deg = float(step_deg)

        self._azimuths = np.arange(self.az_min, self.az_max + 0.001, self.step_deg, dtype=np.float64)
        self._elevations = np.arange(self.el_min, self.el_max + 0.001, self.step_deg, dtype=np.float64)

    def sample(self, step_index: int, *, start_time_s: float = 0.0) -> PatternSample:
        dirs = []
        scan_angles = []
        pixel_u = []
        pixel_v = []

        az_indices = np.arange(len(self._azimuths))
        for row_idx, el in enumerate(self._elevations):
            iter_idx = az_indices if row_idx % 2 == 0 else az_indices[::-1]
            for col_idx, az_idx in enumerate(iter_idx):
                az = self._azimuths[az_idx]
                az_rad = np.deg2rad(az)
                el_rad = np.deg2rad(el)
                cos_el = np.cos(el_rad)
                direction = np.array(
                    [cos_el * np.cos(az_rad), cos_el * np.sin(az_rad), np.sin(el_rad)],
                    dtype=np.float64,
                )
                dirs.append(direction)
                scan_angles.append(el)
                pixel_u.append(float(az_idx))
                pixel_v.append(float(row_idx))

        dirs = ensure_unit_vectors(np.asarray(dirs, dtype=np.float64))
        rel_time = start_time_s + np.arange(len(dirs), dtype=np.float64)
        meta = {
            "scan_angle_deg": np.asarray(scan_angles, dtype=np.float32),
            "pixel_u": np.asarray(pixel_u, dtype=np.float32),
            "pixel_v": np.asarray(pixel_v, dtype=np.float32),
        }
        return PatternSample(directions=dirs, relative_time_s=rel_time, meta=meta)

    def cadence_s(self) -> float:
        return 1.0


class CameraPattern(ScanPattern):
    """Generates unit directions for a pinhole camera model."""

    def __init__(
        self,
        resolution_px: tuple[int, int],
        focal_px: tuple[float, float],
        principal_px: tuple[float, float] | None = None,
        pixel_stride: int = 1,
    ) -> None:
        width, height = resolution_px
        if width <= 0 or height <= 0:
            raise ValueError("resolution_px must be positive")
        if focal_px[0] <= 0 or focal_px[1] <= 0:
            raise ValueError("focal_px must be positive")
        if pixel_stride <= 0:
            raise ValueError("pixel_stride must be positive")

        self.width = int(width)
        self.height = int(height)
        self.fx = float(focal_px[0])
        self.fy = float(focal_px[1])
        if principal_px is None:
            self.cx = (self.width - 1) / 2.0
            self.cy = (self.height - 1) / 2.0
        else:
            self.cx = float(principal_px[0])
            self.cy = float(principal_px[1])
        self.pixel_stride = int(pixel_stride)

    def cadence_s(self) -> float:
        return 1.0

    def sample(self, step_index: int, *, start_time_s: float = 0.0) -> PatternSample:
        us = np.arange(0, self.width, self.pixel_stride, dtype=np.float64)
        vs = np.arange(0, self.height, self.pixel_stride, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        uu = uu.reshape(-1)
        vv = vv.reshape(-1)

        dirs = self.directions_from_pixels(np.column_stack([uu, vv]))

        meta = {
            "pixel_u": uu.astype(np.float32, copy=False),
            "pixel_v": vv.astype(np.float32, copy=False),
        }
        rel_time = np.zeros_like(uu, dtype=np.float64) + start_time_s
        return PatternSample(directions=dirs, relative_time_s=rel_time, meta=meta)

    def directions_from_pixels(self, pixels: np.ndarray) -> np.ndarray:
        uv = np.asarray(pixels, dtype=np.float64)
        x = (uv[:, 0] - self.cx) / self.fx
        y = (uv[:, 1] - self.cy) / self.fy
        z = np.full_like(x, -1.0)
        dirs = np.column_stack([x, y, z])
        return ensure_unit_vectors(dirs)
