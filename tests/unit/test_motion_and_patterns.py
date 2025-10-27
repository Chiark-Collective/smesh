import numpy as np

from smesh.motion.pose import Pose
from smesh.motion.trajectory import StaticTrajectory, PolylineTrajectory, LawnmowerTrajectory
from smesh.sensors.patterns import (
    OscillatingMirrorPattern,
    SpinningPattern,
    RasterAzElPattern,
    CameraPattern,
)
from smesh.sensors.noise import LidarNoise, PhotogrammetryNoise
from smesh.sensors.lidar import LidarSensor
from smesh.sensors.totalstation import TotalStationSensor
from smesh.sensors.camera import CameraSensor


class DummyScene:
    def __init__(self, bounds: tuple[float, float, float, float, float, float]) -> None:
        self._bounds = bounds

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        return self._bounds


def test_static_trajectory_timeline() -> None:
    pose = Pose.from_xyz_rpy((0, 0, 10), (0, 0, 0))
    traj = StaticTrajectory(pose, start_time_s=5.0)
    timeline = list(traj.timeline())
    assert timeline == [(5.0, pose)]
    assert traj.sample(123.0) == pose


def test_polyline_trajectory_interpolates_midpoint() -> None:
    waypoints = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    traj = PolylineTrajectory(waypoints, speed_mps=2.0, start_time_s=0.0)
    pose_mid = traj.sample(2.5)
    np.testing.assert_allclose(pose_mid.t, np.array([5.0, 0.0, 0.0]))


def test_lawnmower_trajectory_serpentine_spacing() -> None:
    scene = DummyScene((0.0, 100.0, 0.0, 60.0, 0.0, 5.0))
    traj = LawnmowerTrajectory(
        scene,
        altitude_m=10.0,
        speed_mps=10.0,
        line_spacing_m=20.0,
        heading_deg=0.0,
        start_time_s=0.0,
    )
    timeline = list(traj.timeline())
    points = np.array([pose.t for _, pose in timeline])
    # Check altitude offset
    assert np.allclose(points[:, 2], 15.0)
    # Unique cross-track coordinates should be spaced ~20 m
    unique_y = np.unique(np.round(points[:, 1], decimals=3))
    diffs = np.diff(unique_y)
    if len(diffs) > 0:
        np.testing.assert_allclose(diffs, np.full_like(diffs, 20.0), atol=1e-3)
    # Direction alternates along x
    deltas = np.diff(points[:, :2], axis=0)
    non_zero = deltas[np.linalg.norm(deltas, axis=1) > 1e-6]
    signs = np.sign(non_zero[:, 0])
    if len(signs) > 1:
        assert np.all(signs[1:] != signs[:-1])


def test_oscillating_pattern_span() -> None:
    pattern = OscillatingMirrorPattern(fov_deg=60.0, line_rate_hz=80.0, pulses_per_line=1000)
    sample = pattern.sample(step_index=3, start_time_s=0.0)
    dirs = sample.directions
    norms = np.linalg.norm(dirs, axis=1)
    assert dirs.shape == (1000, 3)
    assert np.allclose(norms, 1.0, atol=1e-6)
    angles = sample.meta["scan_angle_deg"]
    assert np.isclose(angles.min(), -30.0)
    assert np.isclose(angles.max(), 30.0)
    assert np.all(sample.meta["scanline_id"] == 3)


def test_spinning_pattern_channel_metadata() -> None:
    pattern = SpinningPattern(vertical_angles_deg=[-15.0, 0.0, 15.0], rpm=600.0, prf_hz=90000)
    sample = pattern.sample(step_index=2, start_time_s=0.1)
    dirs = sample.directions
    norms = np.linalg.norm(dirs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert np.all(sample.meta["channel_id"] < 3)
    assert np.all(sample.meta["revolution_id"] == 2)
    assert sample.meta["scan_angle_deg"].shape[0] == dirs.shape[0]


def test_raster_pattern_serpentine_order() -> None:
    pattern = RasterAzElPattern(az_range_deg=(0.0, 10.0), el_range_deg=(0.0, 5.0), step_deg=5.0)
    sample = pattern.sample(step_index=0, start_time_s=0.0)
    pixel_u = sample.meta["pixel_u"]
    rows = sample.meta["pixel_v"]
    # Rows should alternate direction (serpentine)
    row0 = pixel_u[rows == 0]
    row1 = pixel_u[rows == 1]
    assert np.array_equal(row0, np.array([0, 1, 2], dtype=np.float32))
    assert np.array_equal(row1, np.array([2, 1, 0], dtype=np.float32))


def test_lidar_noise_direction_norms() -> None:
    dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (500, 1))
    noise = LidarNoise(sigma_angle_deg=0.5, sigma_range_m=0.02, keep_prob=0.9)
    rng = np.random.default_rng(123)
    jittered = noise.jitter_directions(dirs, rng)
    norms = np.linalg.norm(jittered, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_lidar_noise_dropout_rate() -> None:
    noise = LidarNoise(keep_prob=0.7)
    rng = np.random.default_rng(42)
    mask = noise.dropout_mask(5000, rng)
    keep_rate = mask.mean()
    assert abs(keep_rate - 0.7) < 0.05


def test_lidar_noise_range_nonnegative() -> None:
    ranges = np.full(1000, 10.0, dtype=np.float32)
    noise = LidarNoise(sigma_range_m=1.0)
    rng = np.random.default_rng(7)
    jittered = noise.jitter_ranges(ranges, rng)
    assert np.all(jittered >= 0.0)
    assert abs(jittered.mean() - 10.0) < 0.2


def test_photogrammetry_noise_pixel_jitter() -> None:
    noise = PhotogrammetryNoise(pixel_sigma=0.5)
    rng = np.random.default_rng(11)
    pixels = np.zeros((1000, 2), dtype=np.float32)
    jittered = noise.jitter_pixels(pixels, rng)
    assert jittered.shape == pixels.shape
    assert jittered.std() > 0.3


def test_lidar_sensor_batches_generate_raybundles() -> None:
    pose = Pose.from_xyz_rpy((0.0, 0.0, 100.0), (0.0, 0.0, 0.0))
    traj = StaticTrajectory(pose)
    pattern = OscillatingMirrorPattern(fov_deg=20.0, line_rate_hz=50.0, pulses_per_line=16)
    sensor = LidarSensor(pattern=pattern, trajectory=traj, noise=LidarNoise(keep_prob=1.0), max_range_m=500.0, num_lines=2)

    rng = np.random.default_rng(0)
    batches = list(sensor.batches(rng))
    assert len(batches) == 2

    first = batches[0].bundle
    assert first.origins.shape[0] == 16
    assert first.directions.shape == (16, 3)
    assert np.all(first.directions[:, 2] < 0.0)
    assert "gps_time" in first.meta
    assert np.all(np.diff(first.meta["gps_time"]) >= 0.0)
    assert np.all(first.meta["scanline_id"] == 0)


def test_lidar_sensor_respects_dropout() -> None:
    pose = Pose.from_xyz_rpy((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    traj = StaticTrajectory(pose)
    pattern = OscillatingMirrorPattern(fov_deg=10.0, line_rate_hz=10.0, pulses_per_line=32)
    noise = LidarNoise(keep_prob=0.4)
    sensor = LidarSensor(pattern=pattern, trajectory=traj, noise=noise, max_range_m=100.0, num_lines=1)
    rng = np.random.default_rng(123)
    batches = list(sensor.batches(rng))
    assert len(batches) == 1
    count = batches[0].bundle.origins.shape[0]
    assert 5 <= count < 32


def test_camera_pattern_generates_negative_z() -> None:
    pattern = CameraPattern(resolution_px=(4, 3), focal_px=(200.0, 200.0))
    sample = pattern.sample(step_index=0, start_time_s=0.0)
    dirs = sample.directions
    assert dirs.shape[0] == 12
    assert np.all(dirs[:, 2] < 0)


def test_total_station_sensor_single_batch() -> None:
    scene_pose = Pose.from_xyz_rpy((0.0, 0.0, 10.0), (0.0, 0.0, 0.0))
    traj = StaticTrajectory(scene_pose)
    pattern = RasterAzElPattern(az_range_deg=(-5.0, 5.0), el_range_deg=(-2.0, 2.0), step_deg=2.0)
    sensor = TotalStationSensor(pattern=pattern, trajectory=traj, max_range_m=500.0)
    batches = list(sensor.batches())
    assert len(batches) == 1
    bundle = batches[0].bundle
    assert bundle.origins.shape[0] == bundle.directions.shape[0]
    assert np.allclose(bundle.origins[:, 2], 10.0)
    assert "pixel_u" in bundle.meta and "pixel_v" in bundle.meta


def test_camera_sensor_frames_and_metadata() -> None:
    pose = Pose.from_xyz_rpy((0.0, 0.0, 5.0), (0.0, 0.0, 0.0))
    traj = StaticTrajectory(pose)
    pattern = CameraPattern(resolution_px=(4, 4), focal_px=(100.0, 100.0), pixel_stride=2)
    noise = PhotogrammetryNoise(pixel_sigma=0.1, sigma_time_s=0.0)
    sensor = CameraSensor(pattern=pattern, trajectory=traj, num_frames=2, noise=noise)
    rng = np.random.default_rng(5)
    batches = list(sensor.batches(rng))
    assert len(batches) == 2
    bundle = batches[0].bundle
    assert "pixel_u" in bundle.meta and "pixel_v" in bundle.meta
    assert "gps_time" in bundle.meta
    assert np.all(bundle.meta["frame_id"] == 0)
