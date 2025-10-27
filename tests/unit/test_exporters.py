import numpy as np

from smesh.core.exporter import NpzWriter, PlyWriter
from smesh.core.pointcloud import PointBatch


def test_npz_writer_fills_missing_attributes(tmp_path) -> None:
    path = tmp_path / "cloud.npz"
    writer = NpzWriter(str(path))
    batch1 = PointBatch(
        xyz=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        attrs={"rgb": np.array([[10, 20, 30]], dtype=np.uint8)},
    )
    batch2 = PointBatch(
        xyz=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
        attrs={},
    )
    writer.write_batch(batch1)
    writer.write_batch(batch2)
    writer.close()

    with np.load(path) as data:
        xyz = data["xyz"]
        rgb = data["rgb"]
    assert xyz.shape == (2, 3)
    assert rgb.shape == (2, 3)
    np.testing.assert_array_equal(rgb[0], np.array([10, 20, 30], dtype=np.uint8))
    np.testing.assert_array_equal(rgb[1], np.zeros(3, dtype=np.uint8))


def test_ply_writer_concatenates_batches(tmp_path) -> None:
    path = tmp_path / "points.ply"
    writer = PlyWriter(str(path))
    batch1 = PointBatch(
        xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    )
    batch2 = PointBatch(
        xyz=np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    )
    writer.write_batch(batch1)
    writer.write_batch(batch2)
    writer.close()

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    assert "element vertex 4" in lines
    end_idx = lines.index("end_header")
    data = np.array([[float(val) for val in row.split()] for row in lines[end_idx + 1 :]])
    assert data.shape == (4, 3)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    data_sorted = data[np.lexsort((data[:, 2], data[:, 1], data[:, 0]))]
    expected_sorted = expected[np.lexsort((expected[:, 2], expected[:, 1], expected[:, 0]))]
    np.testing.assert_allclose(data_sorted, expected_sorted)
