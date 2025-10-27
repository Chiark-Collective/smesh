import numpy as np
import pytest

from smesh.core.intersector import RayBundle
from smesh.core.pointcloud import PointBatch


def test_raybundle_normalizes_directions() -> None:
    origins = np.zeros((3, 3), dtype=np.float64)
    dirs = np.array([[1.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64)
    bundle = RayBundle(origins=origins, directions=dirs)
    norms = np.linalg.norm(bundle.directions, axis=1)
    assert np.allclose(norms, 1.0)


def test_pointbatch_rejects_mismatched_attribute_lengths() -> None:
    xyz = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        PointBatch(xyz=xyz, attrs={"intensity": np.array([1.0], dtype=np.float32)})
