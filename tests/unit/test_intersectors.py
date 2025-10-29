from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import pytest

from smesh.core import intersector as inter
from smesh.core.intersector import RayBundle


class _FakeScalar:
    def __init__(self, value: float) -> None:
        self._value = value

    def get(self) -> float:
        return self._value

    def set(self, value: float) -> None:
        self._value = value


class _FakeLocator:
    def __init__(self) -> None:
        self.calls = 0

    def SetDataSet(self, _poly) -> None:  # pragma: no cover - unused in test but keeps interface
        pass

    def BuildLocator(self) -> None:  # pragma: no cover - unused in test but keeps interface
        pass

    def IntersectWithLine(self, p0, p1, tol, t, pt, pcoords, sub_id, cell_id) -> int:  # noqa: D401 - interface mirrors VTK
        self.calls += 1
        # Simulate a hit midway along the segment and set the referenced cell id.
        mid = [(a + b) * 0.5 for a, b in zip(p0, p1)]
        pt[0], pt[1], pt[2] = mid
        if hasattr(cell_id, "set"):
            cell_id.set(17)
        return 1


class _FakeVtkPoints:
    def Reset(self) -> None:
        pass

    def GetNumberOfPoints(self) -> int:  # pragma: no cover - only used in OBB branch
        return 0

    def GetPoint(self, _idx: int) -> tuple[float, float, float]:  # pragma: no cover - only used in OBB branch
        raise IndexError


class _FakeVtkIdList:
    def Reset(self) -> None:
        pass

    def GetNumberOfIds(self) -> int:  # pragma: no cover - only used in OBB branch
        return 0

    def GetId(self, _idx: int) -> int:  # pragma: no cover - only used in OBB branch
        raise IndexError


class _FakePoly:
    def GetMTime(self) -> int:
        return 1


def test_vtk_intersector_reference_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = SimpleNamespace(
        vtkPoints=lambda: _FakeVtkPoints(),
        vtkIdList=lambda: _FakeVtkIdList(),
        vtkCellLocator=lambda: _FakeLocator(),
        vtkOBBTree=type("FakeOBBTree", (), {}),
        mutable=lambda value: _FakeScalar(value),
        reference=lambda value: _FakeScalar(value),
    )

    monkeypatch.setattr(inter, "_HAVE_VTK", True)
    monkeypatch.setattr(inter, "vtk", fake_module)

    locator = _FakeLocator()
    scene = SimpleNamespace(vtk_polydata=lambda: _FakePoly())
    bundle = RayBundle(
        origins=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        directions=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        max_range=2.0,
        multi_hit=False,
    )

    vtk_inter = inter.VTKIntersector(obb_tree=False)
    vtk_inter._locator = locator
    vtk_inter._cached_scene_id = id(scene)
    vtk_inter._cached_scene_mtime = scene.vtk_polydata().GetMTime()

    hits = vtk_inter.intersect(scene, bundle)
    assert hits.cell_ids.shape == (1,)
    assert hits.cell_ids[0] == 17
    assert np.isclose(hits.distances[0], 1.0)
    assert hits.hit_count_per_ray.tolist() == [1]


def test_embree_intersector_orders_multi_hits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inter, "_HAVE_EMBREE", True)
    monkeypatch.setattr(inter.EmbreeIntersector, "_ensure_intersector", lambda self, scene: None)

    class FakeEmbree:
        def intersects_location(self, origins, dirs, multiple_hits=True, return_id=True):
            assert multiple_hits and return_id
            locs = np.array(
                [
                    [0.0, 0.0, 0.9],
                    [0.0, 0.0, 0.1],
                    [0.0, 0.0, 0.4],
                ],
                dtype=np.float64,
            )
            ray_ids = np.array([1, 0, 1], dtype=np.int64)
            tri_ids = np.array([20, 30, 10], dtype=np.int64)
            return locs, ray_ids, tri_ids

    embree = inter.EmbreeIntersector()
    embree._inter = FakeEmbree()

    scene = object()
    bundle = RayBundle(
        origins=np.zeros((2, 3), dtype=np.float32),
        directions=np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (2, 1)),
        max_range=5.0,
        multi_hit=True,
    )

    hits = embree.intersect(scene, bundle)
    assert hits.ray_index.tolist() == [0, 1, 1]
    assert np.allclose(hits.distances, [0.1, 0.4, 0.9])
    assert hits.cell_ids.tolist() == [30, 10, 20]
    assert hits.hit_count_per_ray.tolist() == [1, 2]
