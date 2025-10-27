from typing import Any

import numpy as np

from smesh.core.intersector import RayBundle, RayHits
from smesh.core.sampler import Sampler, SamplerConfig


class DummyScene:
    def probe_attributes(self, xyz: np.ndarray) -> dict[str, np.ndarray]:
        return {}


class DummyIntersector:
    def __init__(self, hits: RayHits) -> None:
        self._hits = hits

    def intersect(self, scene: Any, bundle: RayBundle) -> RayHits:
        return self._hits


class DummyWriter:
    def __init__(self) -> None:
        self.batches: list = []

    def write_batch(self, batch) -> None:
        self.batches.append(batch)

    def close(self) -> None:
        pass


def test_sampler_forwards_meta_attributes() -> None:
    origins = np.zeros((3, 3), dtype=np.float32)
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (3, 1))
    meta = {
        "gps_time": np.array([0.0, 0.1, 0.2], dtype=np.float64),
        "scan_angle_deg": np.array([-5.0, 0.0, 5.0], dtype=np.float32),
        "scanline_id": np.array([1, 1, 1], dtype=np.uint32),
    }
    bundle = RayBundle(origins=origins, directions=dirs, max_range=100.0, multi_hit=False, meta=meta)

    hits = RayHits(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        distances=np.array([10.0, 12.0, 14.0], dtype=np.float32),
        ray_index=np.array([0, 1, 2], dtype=np.int64),
        cell_ids=np.zeros(3, dtype=np.int64),
        hit_count_per_ray=np.ones(3, dtype=np.int64),
    )

    scene = DummyScene()
    intersector = DummyIntersector(hits)
    sampler = Sampler(scene, intersector=intersector, cfg=SamplerConfig(attributes=["range"]))
    writer = DummyWriter()

    sampler.run_to_writer(writer, [bundle])
    assert len(writer.batches) == 1
    batch = writer.batches[0]
    np.testing.assert_allclose(batch.attrs["gps_time"], np.array([0.0, 0.1, 0.2]))
    np.testing.assert_array_equal(batch.attrs["scanline_id"], np.array([1, 1, 1], dtype=np.uint32))
    np.testing.assert_allclose(batch.attrs["scan_angle_deg"], np.array([-5.0, 0.0, 5.0]))


def test_sampler_chunks_large_bundles() -> None:
    origins = np.zeros((5, 3), dtype=np.float32)
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]], dtype=np.float32), (5, 1))
    meta = {"gps_time": np.linspace(0.0, 0.4, 5, dtype=np.float64)}
    bundle = RayBundle(origins=origins, directions=dirs, max_range=100.0, multi_hit=False, meta=meta)

    class RecordingIntersector:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def intersect(self, scene: Any, sub_bundle: RayBundle) -> RayHits:
            self.calls.append(len(sub_bundle.origins))
            return RayHits(
                points=np.zeros((0, 3), dtype=np.float32),
                distances=np.zeros((0,), dtype=np.float32),
                ray_index=np.zeros((0,), dtype=np.int64),
                cell_ids=np.zeros((0,), dtype=np.int64),
                hit_count_per_ray=np.zeros(len(sub_bundle.origins), dtype=np.int64),
            )

    intersector = RecordingIntersector()
    scene = DummyScene()
    cfg = SamplerConfig(attributes=[], batch_size_rays=2)
    sampler = Sampler(scene, intersector=intersector, cfg=cfg)
    writer = DummyWriter()

    stats = sampler.run_to_writer(writer, [bundle])
    assert intersector.calls == [2, 2, 1]
    assert stats["rays"] == 5
    assert writer.batches == []
