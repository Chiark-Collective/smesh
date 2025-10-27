import numpy as np
from smesh.core.pointcloud import PointBatch
from smesh.core.attributes import (
    RangeComputer,
    ReturnNumberComputer,
    IncidenceAngleComputer,
    IntensityLambertianComputer,
    BeamFootprintComputer,
    ColorNormalProbe,
)


class DummyScene:
    def __init__(self, normals: np.ndarray | None = None, rgb: np.ndarray | None = None) -> None:
        self._normals = normals
        self._rgb = rgb

    def probe_attributes(self, xyz: np.ndarray) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if self._normals is not None:
            out["normal"] = self._normals
        if self._rgb is not None:
            out["rgb"] = self._rgb
        return out


def make_batch(n: int, **attrs: np.ndarray) -> PointBatch:
    xyz = np.zeros((n, 3), dtype=np.float32)
    return PointBatch(xyz=xyz, attrs={k: v for k, v in attrs.items()})


def test_range_computer_uses_hit_distance() -> None:
    distances = np.array([12.5, 3.0], dtype=np.float32)
    batch = make_batch(2, _ray_hit_distance=distances.copy())
    RangeComputer().compute(batch, DummyScene())
    assert "range_m" in batch.attrs
    np.testing.assert_allclose(batch.attrs["range_m"], distances)


def test_return_number_computer_assigns_counts() -> None:
    ray_index = np.array([0, 0, 1, 1], dtype=np.int64)
    hit_counts = np.array([2, 2], dtype=np.int64)
    batch = make_batch(4, _ray_index=ray_index, _hit_count_per_ray=hit_counts)
    ReturnNumberComputer().compute(batch, DummyScene())
    np.testing.assert_array_equal(batch.attrs["return_number"], np.array([1, 2, 1, 2], dtype=np.uint8))
    np.testing.assert_array_equal(batch.attrs["number_of_returns"], np.array([2, 2, 2, 2], dtype=np.uint8))


def test_incidence_angle_computer_zero_for_normal_incidence() -> None:
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (3, 1))
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (3, 1))
    batch = make_batch(3, _ray_dir=dirs.astype(np.float32))
    IncidenceAngleComputer().compute(batch, DummyScene(normals=normals.astype(np.float32)))
    assert "incidence_deg" in batch.attrs
    assert np.allclose(batch.attrs["incidence_deg"], 0.0)


def test_incidence_angle_increases_with_tilt() -> None:
    dirs = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    normals = np.array([[0.0, 0.0, 1.0], [np.sqrt(0.5), 0.0, np.sqrt(0.5)]], dtype=np.float32)
    batch = make_batch(2, _ray_dir=dirs.astype(np.float32))
    IncidenceAngleComputer().compute(batch, DummyScene(normals=normals))
    inc = batch.attrs["incidence_deg"]
    assert inc[0] < 1.0  # approx zero
    assert 40.0 < inc[1] < 50.0  # around 45 degrees


def test_intensity_lambertian_monotonic_with_incidence() -> None:
    batch = make_batch(
        3,
        incidence_deg=np.array([0.0, 45.0, 80.0], dtype=np.float32),
        range_m=np.ones(3, dtype=np.float32),
    )
    IntensityLambertianComputer().compute(batch, DummyScene())
    intensities = batch.attrs["intensity01"]
    assert np.isclose(intensities[0], 1.0)
    assert intensities[0] > intensities[1] > intensities[2]


def test_beam_footprint_scales_with_range() -> None:
    ranges = np.array([10.0, 20.0], dtype=np.float32)
    batch = make_batch(2, range_m=ranges)
    comp = BeamFootprintComputer(divergence_mrad=0.5)
    comp.compute(batch, DummyScene())
    fp = batch.attrs["beam_footprint_m"]
    expected = 0.5 * 0.5e-3 * ranges
    np.testing.assert_allclose(fp[:, 0], expected)
    np.testing.assert_allclose(fp[:, 1], expected)


def test_color_normal_probe_respects_scene_probe() -> None:
    rgb = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    normals = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    batch = make_batch(2)
    ColorNormalProbe().compute(batch, DummyScene(normals=normals, rgb=rgb))
    np.testing.assert_array_equal(batch.attrs["rgb"], rgb)
    np.testing.assert_allclose(batch.attrs["normal"], normals)
