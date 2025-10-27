from __future__ import annotations
import numpy as np
import math
import logging

def get_logger(name: str = "smesh") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def ensure_unit_vectors(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.clip(norms, eps, None)
    return v / norms

def radians(deg: float | np.ndarray) -> float | np.ndarray:
    return np.deg2rad(deg)

def degrees(rad: float | np.ndarray) -> float | np.ndarray:
    return np.rad2deg(rad)
