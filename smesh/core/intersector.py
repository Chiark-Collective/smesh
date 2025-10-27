from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional, Iterable, Tuple, List, Dict
import numpy as np
from .scene import MeshScene
from .utils import get_logger, ensure_unit_vectors

_log = get_logger()

try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import vtk_to_numpy
    _HAVE_VTK = True
except Exception:
    vtk = None  # type: ignore
    _HAVE_VTK = False

try:
    import trimesh  # type: ignore
    from trimesh.ray import ray_pyembree  # type: ignore
    _HAVE_EMBREE = True
except Exception:
    ray_pyembree = None  # type: ignore
    _HAVE_EMBREE = False


@dataclass
class RayBundle:
    origins: np.ndarray          # (M, 3)
    directions: np.ndarray       # (M, 3) unit
    max_range: float = 1e6
    multi_hit: bool = False
    meta: dict[str, np.ndarray] = field(default_factory=dict)  # per-ray metadata

    def __post_init__(self) -> None:
        assert self.origins.shape == self.directions.shape
        self.directions = ensure_unit_vectors(self.directions)


@dataclass
class RayHits:
    points: np.ndarray                 # (K, 3)
    distances: np.ndarray              # (K,)
    ray_index: np.ndarray              # (K,) maps each hit to a source ray
    cell_ids: np.ndarray               # (K,) triangle ids (or -1 if unknown)
    hit_count_per_ray: np.ndarray      # (M,)

    def empty_like(self, n_rays: int) -> "RayHits":
        return RayHits(
            points=np.zeros((0, 3), dtype=np.float32),
            distances=np.zeros((0,), dtype=np.float32),
            ray_index=np.zeros((0,), dtype=np.int64),
            cell_ids=np.zeros((0,), dtype=np.int64),
            hit_count_per_ray=np.zeros((n_rays,), dtype=np.int64),
        )


class Intersector(Protocol):
    def intersect(self, scene: MeshScene, bundle: RayBundle) -> RayHits: ...


class NumpyIntersector:
    """Pure NumPy fallback intersector using brute-force ray-triangle tests."""

    def __init__(self, epsilon: float = 1e-8) -> None:
        self.epsilon = float(epsilon)

    def _ray_triangle_intersect(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        tri: np.ndarray,
        max_range: float,
    ) -> Optional[Tuple[float, np.ndarray]]:
        v0, v1, v2 = tri
        edge1 = v1 - v0
        edge2 = v2 - v0
        pvec = np.cross(direction, edge2)
        det = np.dot(edge1, pvec)
        if abs(det) < self.epsilon:
            return None
        inv_det = 1.0 / det
        tvec = origin - v0
        u = np.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return None
        qvec = np.cross(tvec, edge1)
        v = np.dot(direction, qvec) * inv_det
        if v < 0.0 or (u + v) > 1.0:
            return None
        t = np.dot(edge2, qvec) * inv_det
        if t < 0.0 or t > max_range:
            return None
        point = origin + direction * t
        return float(t), point.astype(np.float32, copy=False)

    def intersect(self, scene: MeshScene, bundle: RayBundle) -> RayHits:
        verts, faces = scene.triangle_arrays()
        tris = verts[faces]

        origins = np.asarray(bundle.origins, dtype=np.float64)
        dirs = ensure_unit_vectors(np.asarray(bundle.directions, dtype=np.float64))
        n_rays = origins.shape[0]
        max_range = float(bundle.max_range)

        points_list: List[np.ndarray] = []
        distances_list: List[float] = []
        ray_index_list: List[int] = []
        cell_ids_list: List[int] = []
        hit_counts = np.zeros((n_rays,), dtype=np.int64)

        for ray_idx in range(n_rays):
            o = origins[ray_idx]
            d = dirs[ray_idx]
            hits_for_ray: List[Tuple[float, np.ndarray, int]] = []
            for tri_idx, tri in enumerate(tris):
                result = self._ray_triangle_intersect(o, d, tri, max_range)
                if result is None:
                    continue
                dist, point = result
                hits_for_ray.append((dist, point, tri_idx))

            if not hits_for_ray:
                continue

            hits_for_ray.sort(key=lambda x: x[0])
            if not bundle.multi_hit:
                hits_for_ray = hits_for_ray[:1]

            hit_counts[ray_idx] = len(hits_for_ray)
            for dist, point, tri_idx in hits_for_ray:
                points_list.append(point)
                distances_list.append(dist)
                ray_index_list.append(ray_idx)
                cell_ids_list.append(tri_idx)

        if not points_list:
            return RayHits(
                points=np.zeros((0, 3), dtype=np.float32),
                distances=np.zeros((0,), dtype=np.float32),
                ray_index=np.zeros((0,), dtype=np.int64),
                cell_ids=np.zeros((0,), dtype=np.int64),
                hit_count_per_ray=hit_counts,
            )

        return RayHits(
            points=np.vstack(points_list).astype(np.float32, copy=False),
            distances=np.asarray(distances_list, dtype=np.float32),
            ray_index=np.asarray(ray_index_list, dtype=np.int64),
            cell_ids=np.asarray(cell_ids_list, dtype=np.int64),
            hit_count_per_ray=hit_counts,
        )


class VTKIntersector:
    """VTK OBBTree-based intersector (works with vtkPolyData scenes)."""
    def __init__(self, obb_tree: bool = True) -> None:
        if not _HAVE_VTK:
            raise RuntimeError("VTK is not available.")
        self._obb_tree = obb_tree

    def intersect(self, scene: MeshScene, bundle: RayBundle) -> RayHits:
        poly = scene.vtk_polydata()

        if self._obb_tree:
            tree = vtk.vtkOBBTree()
            tree.SetDataSet(poly)
            tree.BuildLocator()
            locator = tree
        else:
            locator = vtk.vtkCellLocator()
            locator.SetDataSet(poly)
            locator.BuildLocator()

        origins = np.asarray(bundle.origins, dtype=np.float64)
        dirs = np.asarray(bundle.directions, dtype=np.float64)
        n_rays = origins.shape[0]
        maxr = float(bundle.max_range)

        points_list: List[List[float]] = []
        distances_list: List[float] = []
        ray_index_list: List[int] = []
        cell_ids_list: List[int] = []
        hit_counts = np.zeros((n_rays,), dtype=np.int64)

        # Reusable objects
        isect_pts = vtk.vtkPoints()
        cell_ids = vtk.vtkIdList()

        for i in range(n_rays):
            o = origins[i]
            d = dirs[i]
            p0 = o
            p1 = o + d * maxr

            if isinstance(locator, vtk.vtkOBBTree):
                isect_pts.Reset()
                cell_ids.Reset()
                code = locator.IntersectWithLine(p0, p1, isect_pts, cell_ids)
                if code == 0:
                    continue
                m = isect_pts.GetNumberOfPoints()
                # Fetch and compute distances
                # OBBTree returns intersections ordered along the segment
                pts = [isect_pts.GetPoint(j) for j in range(m)]
                dists = [np.linalg.norm(np.asarray(pts[j]) - o) for j in range(m)]
                cids = [cell_ids.GetId(j) for j in range(cell_ids.GetNumberOfIds())]
                # If counts differ (shouldn't), pad/truncate
                if len(cids) < m:
                    cids.extend([-1] * (m - len(cids)))
                if not bundle.multi_hit and m > 1:
                    # Keep only the closest hit
                    idx = int(np.argmin(dists))
                    pts = [pts[idx]]
                    dists = [dists[idx]]
                    cids = [cids[idx]]

                for p, dist, cid in zip(pts, dists, cids):
                    points_list.append(list(p))
                    distances_list.append(float(dist))
                    ray_index_list.append(i)
                    cell_ids_list.append(int(cid))
                hit_counts[i] = len(pts)
            else:
                # Generic cell locator: do first-hit only
                t = vtk.reference(0.0)
                pt = [0.0, 0.0, 0.0]
                pcoords = [0.0, 0.0, 0.0]
                sub_id = vtk.reference(0)
                cell_id = vtk.reference(0)
                code = locator.IntersectWithLine(p0, p1, 1e-6, t, pt, pcoords, sub_id, cell_id)
                if code == 0:
                    continue
                points_list.append(list(pt))
                distances_list.append(float(np.linalg.norm(np.asarray(pt) - o)))
                ray_index_list.append(i)
                cell_ids_list.append(int(cell_id))
                hit_counts[i] = 1

        if len(points_list) == 0:
            return RayHits(
                points=np.zeros((0, 3), dtype=np.float32),
                distances=np.zeros((0,), dtype=np.float32),
                ray_index=np.zeros((0,), dtype=np.int64),
                cell_ids=np.zeros((0,), dtype=np.int64),
                hit_count_per_ray=hit_counts,
            )

        return RayHits(
            points=np.asarray(points_list, dtype=np.float32),
            distances=np.asarray(distances_list, dtype=np.float32),
            ray_index=np.asarray(ray_index_list, dtype=np.int64),
            cell_ids=np.asarray(cell_ids_list, dtype=np.int64),
            hit_count_per_ray=hit_counts,
        )


class EmbreeIntersector:
    """Embree via trimesh.ray.ray_pyembree (optional dependency)."""
    def __init__(self) -> None:
        if not _HAVE_EMBREE:
            raise RuntimeError("pyembree not available. pip install trimesh[ray].")
        self._inter = None

    def _ensure_intersector(self, scene: MeshScene) -> None:
        if self._inter is not None:
            return
        # Build trimesh object from VTK or path
        if hasattr(scene, "_trimesh") and scene._trimesh is not None:
            tm = scene._trimesh
        else:
            # Convert vtkPolyData to trimesh
            poly = scene.vtk_polydata()
            import numpy as np
            from vtk.util.numpy_support import vtk_to_numpy
            pts = vtk_to_numpy(poly.GetPoints().GetData())
            faces = vtk_to_numpy(poly.GetPolys().GetData()).reshape(-1, 4)[:, 1:4]
            tm = trimesh.Trimesh(vertices=pts, faces=faces, process=False)
        self._inter = ray_pyembree.RayMeshIntersector(tm)

    def intersect(self, scene: MeshScene, bundle: RayBundle) -> RayHits:
        self._ensure_intersector(scene)
        origins = np.asarray(bundle.origins, dtype=np.float64)
        dirs = ensure_unit_vectors(np.asarray(bundle.directions, dtype=np.float64))
        maxr = float(bundle.max_range)

        if bundle.multi_hit:
            locs, ray_ids, tri_ids = self._inter.intersects_location(
                origins, dirs, multiple_hits=True, return_id=True
            )
            # Filter by max range
            dists = np.linalg.norm(locs - origins[ray_ids], axis=1)
            mask = dists <= maxr
            locs = locs[mask]
            ray_ids = ray_ids[mask]
            tri_ids = tri_ids[mask]
            dists = dists[mask]

            # Count per-ray
            hit_counts = np.zeros((origins.shape[0],), dtype=np.int64)
            for i in ray_ids:
                hit_counts[i] += 1

            return RayHits(
                points=locs.astype(np.float32, copy=False),
                distances=dists.astype(np.float32, copy=False),
                ray_index=ray_ids.astype(np.int64, copy=False),
                cell_ids=tri_ids.astype(np.int64, copy=False),
                hit_count_per_ray=hit_counts,
            )
        else:
            # First hit only
            locs, idx_ray, tri_ids = self._inter.intersects_location(
                origins, dirs, multiple_hits=False, return_id=True
            )
            dists = np.linalg.norm(locs - origins[idx_ray], axis=1)
            mask = dists <= maxr
            locs = locs[mask]
            idx_ray = idx_ray[mask]
            tri_ids = tri_ids[mask]
            dists = dists[mask]

            hit_counts = np.zeros((origins.shape[0],), dtype=np.int64)
            for i in idx_ray:
                hit_counts[i] = 1

            return RayHits(
                points=locs.astype(np.float32, copy=False),
                distances=dists.astype(np.float32, copy=False),
                ray_index=idx_ray.astype(np.int64, copy=False),
                cell_ids=tri_ids.astype(np.int64, copy=False),
                hit_count_per_ray=hit_counts,
            )


class AutoIntersector:
    """Picks the fastest available backend: Embree if present, else VTK."""
    def __init__(self) -> None:
        self._impl: Optional[Intersector] = None

    def intersect(self, scene: MeshScene, bundle: RayBundle) -> RayHits:
        if self._impl is None:
            self._impl = self._choose(scene)
        return self._impl.intersect(scene, bundle)

    def _choose(self, scene: MeshScene) -> Intersector:
        global _HAVE_EMBREE, _HAVE_VTK
        if _HAVE_EMBREE:
            _log.info("AutoIntersector: using Embree.")
            return EmbreeIntersector()
        if _HAVE_VTK:
            _log.info("AutoIntersector: using VTK OBBTree.")
            return VTKIntersector(obb_tree=True)
        if scene.has_numpy_mesh():
            _log.info("AutoIntersector: using NumPy fallback intersector.")
            return NumpyIntersector()
        raise RuntimeError("No intersector backend found. Install VTK or trimesh[ray].")
