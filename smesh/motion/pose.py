from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Pose:
    t: np.ndarray   # (3,)
    R: np.ndarray   # (3,3)

    @staticmethod
    def from_xyz_rpy(xyz: tuple[float,float,float], rpy_deg: tuple[float,float,float]) -> "Pose":
        rx, ry, rz = np.deg2rad(rpy_deg)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
        Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
        Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
        R = Rz @ Ry @ Rx
        return Pose(t=np.array(xyz, dtype=float), R=R.astype(float))

    def apply(self, p_body: np.ndarray) -> np.ndarray:
        return (self.R @ p_body.T).T + self.t
