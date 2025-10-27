from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.utils import ensure_unit_vectors


@dataclass
class NoiseModel:
    """Base class for sensor noise models."""

    def jitter_directions(
        self,
        dirs: np.ndarray,
        rng: np.random.Generator,
        sigma_deg: Optional[float] = None,
    ) -> np.ndarray:
        return dirs

    def jitter_ranges(
        self,
        ranges: np.ndarray,
        rng: np.random.Generator,
        sigma_m: Optional[np.ndarray | float] = None,
    ) -> np.ndarray:
        return ranges

    def dropout_mask(self, n: int, rng: np.random.Generator, keep_prob: float) -> np.ndarray:
        return np.ones(n, dtype=bool)

    def time_jitter(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        sigma_s: Optional[float] = None,
    ) -> np.ndarray:
        return t


class LidarNoise(NoiseModel):
    """Gaussian angle/range noise with Bernoulli dropouts."""

    def __init__(
        self,
        sigma_range_m: float = 0.0,
        sigma_angle_deg: float = 0.0,
        keep_prob: float = 1.0,
        sigma_time_s: float = 0.0,
    ) -> None:
        if not (0.0 < keep_prob <= 1.0):
            raise ValueError("keep_prob must be in (0, 1].")
        self.sigma_range_m = float(max(0.0, sigma_range_m))
        self.sigma_angle_deg = float(max(0.0, sigma_angle_deg))
        self.keep_prob = float(keep_prob)
        self.sigma_time_s = float(max(0.0, sigma_time_s))

    def jitter_directions(
        self,
        dirs: np.ndarray,
        rng: np.random.Generator,
        sigma_deg: Optional[float] = None,
    ) -> np.ndarray:
        sigma = self.sigma_angle_deg if sigma_deg is None else float(max(0.0, sigma_deg))
        if sigma == 0.0:
            return dirs
        sigma_rad = np.deg2rad(sigma)
        perturb = rng.normal(scale=sigma_rad, size=dirs.shape)
        # Make perturbation orthogonal so magnitude interprets as angle
        proj = np.sum(perturb * dirs, axis=1, keepdims=True)
        perturb = perturb - proj * dirs
        jittered = dirs + perturb
        return ensure_unit_vectors(jittered)

    def jitter_ranges(
        self,
        ranges: np.ndarray,
        rng: np.random.Generator,
        sigma_m: Optional[np.ndarray | float] = None,
    ) -> np.ndarray:
        if sigma_m is None:
            sigma = self.sigma_range_m
        elif np.isscalar(sigma_m):
            sigma = float(max(0.0, sigma_m))  # type: ignore[arg-type]
        else:
            sigma = np.asarray(sigma_m, dtype=np.float64)
        if (np.isscalar(sigma) and sigma == 0.0) or (isinstance(sigma, np.ndarray) and np.all(sigma == 0.0)):
            return ranges
        noise = rng.normal(scale=sigma, size=ranges.shape)
        out = np.clip(ranges + noise, 0.0, None)
        return out.astype(np.float32, copy=False)

    def dropout_mask(self, n: int, rng: np.random.Generator, keep_prob: Optional[float] = None) -> np.ndarray:
        p = self.keep_prob if keep_prob is None else float(keep_prob)
        if not (0.0 <= p <= 1.0):
            raise ValueError("keep_prob must lie in [0, 1].")
        return rng.random(n) < p

    def time_jitter(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        sigma_s: Optional[float] = None,
    ) -> np.ndarray:
        sigma = self.sigma_time_s if sigma_s is None else float(max(0.0, sigma_s))
        if sigma == 0.0:
            return t
        jitter = rng.normal(scale=sigma, size=t.shape)
        return (t + jitter).astype(np.float64, copy=False)


class PhotogrammetryNoise(NoiseModel):
    """Simple photogrammetry noise model for pixel jitter and timing."""

    def __init__(self, pixel_sigma: float = 0.0, sigma_time_s: float = 0.0) -> None:
        self.pixel_sigma = float(max(0.0, pixel_sigma))
        self.sigma_time_s = float(max(0.0, sigma_time_s))

    def jitter_pixels(
        self,
        pixels: np.ndarray,
        rng: np.random.Generator,
        pixel_sigma: Optional[float] = None,
    ) -> np.ndarray:
        sigma = self.pixel_sigma if pixel_sigma is None else float(max(0.0, pixel_sigma))
        if sigma == 0.0:
            return pixels
        noise = rng.normal(scale=sigma, size=pixels.shape)
        return pixels + noise

    def time_jitter(
        self,
        t: np.ndarray,
        rng: np.random.Generator,
        sigma_s: Optional[float] = None,
    ) -> np.ndarray:
        sigma = self.sigma_time_s if sigma_s is None else float(max(0.0, sigma_s))
        if sigma == 0.0:
            return t
        return t + rng.normal(scale=sigma, size=t.shape)
