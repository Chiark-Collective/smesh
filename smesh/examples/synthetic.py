from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def _grid_plane(size: float, divisions: int, z: float, color: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lin = np.linspace(-size / 2.0, size / 2.0, divisions + 1, dtype=np.float32)
    xv, yv = np.meshgrid(lin, lin, indexing="ij")
    vertices = np.column_stack([xv.ravel(), yv.ravel(), np.full_like(xv.ravel(), z)])

    faces = []
    for i in range(divisions):
        for j in range(divisions):
            idx0 = i * (divisions + 1) + j
            idx1 = idx0 + 1
            idx2 = idx0 + (divisions + 1)
            idx3 = idx2 + 1
            faces.append([idx0, idx1, idx3])
            faces.append([idx0, idx3, idx2])
    faces_arr = np.asarray(faces, dtype=np.int64)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (vertices.shape[0], 1))
    return vertices.astype(np.float32), faces_arr, colors


def _box(center: Tuple[float, float, float], size: Tuple[float, float, float], color: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, cz = center
    sx, sy, sz = size
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0
    vertices = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float32)

    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 7, 6], [4, 6, 5],  # top
        [0, 4, 5], [0, 5, 1],  # front
        [1, 5, 6], [1, 6, 2],  # right
        [2, 6, 7], [2, 7, 3],  # back
        [3, 7, 4], [3, 4, 0],  # left
    ], dtype=np.int64)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (vertices.shape[0], 1))
    return vertices, faces, colors


def _ramp(length: float, width: float, height: float, color: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lx, wy, hz = length / 2.0, width / 2.0, height
    vertices = np.array([
        [-lx, -wy, 0.0],
        [lx, -wy, 0.0],
        [lx, wy, 0.0],
        [-lx, wy, 0.0],
        [-lx, wy, hz],
        [lx, wy, hz],
    ], dtype=np.float32)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],  # base (downward facing)
        [3, 5, 2], [3, 4, 5],  # back wall
        [0, 4, 3],             # left wall
        [1, 2, 5],             # right wall
        [0, 1, 5], [0, 5, 4],  # sloped deck
    ], dtype=np.int64)
    colors = np.tile(np.asarray(color, dtype=np.uint8), (vertices.shape[0], 1))
    return vertices, faces, colors


def _merge_parts(parts: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    offset = 0
    for verts, tri, col in parts:
        vertices.append(verts)
        colors.append(col)
        faces.append(tri + offset)
        offset += verts.shape[0]
    all_vertices = np.vstack(vertices).astype(np.float32, copy=False)
    all_faces = np.vstack(faces).astype(np.int64, copy=False)
    all_colors = np.vstack(colors).astype(np.uint8, copy=False)
    normals = _compute_vertex_normals(all_vertices, all_faces)
    return all_vertices, all_faces, all_colors, normals


def _compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    tris = vertices[faces]
    face_normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lens = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = np.divide(face_normals, np.clip(lens, 1e-8, None), out=np.zeros_like(face_normals), where=lens > 0)
    for idx, tri in enumerate(faces):
        normals[tri] += face_normals[idx]
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, np.clip(lens, 1e-8, None), out=np.zeros_like(normals), where=lens > 0)
    return normals.astype(np.float32, copy=False)


def _write_ascii_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, normals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for (x, y, z), (nx, ny, nz), (r, g, b) in zip(vertices, normals, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f} {int(r)} {int(g)} {int(b)}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def generate_mesh(preset: str, size: float, path: Path) -> None:
    preset = preset.lower()
    if preset == "plane":
        plane = _grid_plane(size=size, divisions=40, z=0.0, color=(180, 200, 180))
        vertices, faces, colors = plane
        normals = _compute_vertex_normals(vertices, faces)
        _write_ascii_ply(path, vertices, faces, colors, normals)
        return

    if preset == "ramp":
        plane = _grid_plane(size=size, divisions=30, z=0.0, color=(200, 200, 200))
        ramp = _ramp(length=size * 0.8, width=size * 0.4, height=size * 0.2, color=(200, 160, 120))
        ramp_vertices = ramp[0] + np.array([0.0, -size * 0.1, 0.0], dtype=np.float32)
        parts = [plane, (ramp_vertices, ramp[1], ramp[2])]
        vertices, faces, colors, normals = _merge_parts(parts)
        _write_ascii_ply(path, vertices, faces, colors, normals)
        return

    if preset == "demo":
        plane = _grid_plane(size=size, divisions=40, z=0.0, color=(180, 200, 180))
        ramp = _ramp(length=size * 0.9, width=size * 0.4, height=size * 0.3, color=(200, 160, 120))
        ramp_vertices = ramp[0] + np.array([0.0, -size * 0.15, 0.0], dtype=np.float32)
        box1 = _box(center=(size * 0.15, size * 0.2, size * 0.25), size=(size * 0.3, size * 0.3, size * 0.5), color=(180, 180, 240))
        box2 = _box(center=(-size * 0.3, -size * 0.1, size * 0.15), size=(size * 0.2, size * 0.2, size * 0.3), color=(240, 180, 180))
        wall = _box(center=(0.0, -size * 0.35, size * 0.25), size=(size * 0.9, size * 0.05, size * 0.5), color=(200, 200, 220))

        parts = [
            plane,
            (ramp_vertices, ramp[1], ramp[2]),
            box1,
            box2,
            wall,
        ]
        vertices, faces, colors, normals = _merge_parts(parts)
        _write_ascii_ply(path, vertices, faces, colors, normals)
        return

    if preset == "preview":
        plane_vertices, plane_faces, plane_colors = _grid_plane(size=size, divisions=20, z=0.0, color=(185, 200, 185))
        ramp = _ramp(length=size * 0.45, width=size * 0.28, height=size * 0.2, color=(210, 170, 130))
        ramp_vertices = ramp[0] + np.array([-size * 0.2, -size * 0.08, 0.0], dtype=np.float32)
        # Drop the ramp base to keep the ground mesh continuous.
        ramp_faces = ramp[1][2:]
        plane = (plane_vertices, plane_faces, plane_colors)
        box = _box(center=(size * 0.2, size * 0.15, size * 0.2), size=(size * 0.25, size * 0.25, size * 0.4), color=(180, 190, 240))
        parts = [
            plane,
            (ramp_vertices, ramp_faces, ramp[2]),
            box,
        ]
        vertices, faces, colors, normals = _merge_parts(parts)
        _write_ascii_ply(path, vertices, faces, colors, normals)
        return

    raise ValueError(f"Unknown synthetic mesh preset '{preset}'.")
