"""Unit tests for System 7 sensor implementations.

Covers the concrete sensor classes (LiDAR, stereo camera, IMU + wheel
encoder, beacon network, force-torque, sun sensor). All scene interaction is
exercised through light test doubles so the suite is GPU-free and
deterministic, mirroring the physics mock-test convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from moon_rover.sensors import (
    BeaconConfig,
    CameraConfig,
    EncoderConfig,
    FTConfig,
    GenesisForceTorqueSensor,
    GenesisIMUSensor,
    GenesisLiDARScanner,
    GenesisSunSensor,
    GenesisWheelEncoder,
    IMUConfig,
    LiDARConfig,
    RaycastStereoCamera,
    SunSensorConfig,
    TrilaterationBeaconNetwork,
)


# --------------------------------------------------------------------------
# Scene test doubles
# --------------------------------------------------------------------------


class FlatTerrainScene:
    """Heightfield scene: flat ground at z=0 with an optional square bump."""

    def __init__(self, bump_height: float = 0.0, bump_radius: float = 1.0):
        self.bump_height = bump_height
        self.bump_radius = bump_radius

    def get_terrain_height(self, x: float, y: float) -> float:
        if abs(x) <= self.bump_radius and abs(y) <= self.bump_radius:
            return self.bump_height
        return 0.0

    def get_terrain_normal(self, x: float, y: float) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])


class ConstantBatchScene:
    """Batch raycaster that reports a fixed hit distance with +z normals."""

    def __init__(self, distance: float = 5.0):
        self.distance = distance

    def raycast_batch(self, origins, directions, max_range):
        n = np.asarray(origins).reshape(-1, 3).shape[0]
        d = min(self.distance, max_range)
        dirs = np.asarray(directions).reshape(-1, 3)
        return {
            "distances": np.full(n, d),
            "positions": np.asarray(origins).reshape(-1, 3) + dirs * d,
            "normals": np.tile([0.0, 0.0, 1.0], (n, 1)),
        }


# --------------------------------------------------------------------------
# LiDAR
# --------------------------------------------------------------------------


def _lidar_cfg(**kw) -> LiDARConfig:
    base = dict(
        num_channels=8,
        h_resolution_deg=10.0,
        elevation_range_deg=(-30.0, 0.0),
        max_range_m=50.0,
        range_noise_sigma_m=0.02,
        intensity_noise_sigma=0.01,
        rotation_rate_hz=10.0,
        min_range_m=0.5,
        max_returns=1,
        seed=42,
    )
    base.update(kw)
    return LiDARConfig(**base)


def test_lidar_config_validation():
    s = GenesisLiDARScanner()
    with pytest.raises(ValueError):
        s.configure(_lidar_cfg(max_range_m=0.1))  # <= min_range_m
    with pytest.raises(ValueError):
        s.configure(_lidar_cfg(h_resolution_deg=0.0))


def test_lidar_scan_flat_ground():
    s = GenesisLiDARScanner()
    s.configure(_lidar_cfg())
    pose = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])  # 2 m up
    cloud = s.scan(FlatTerrainScene(), pose)
    assert cloud.points.shape[0] > 0
    assert cloud.points.shape[1] == 3
    assert cloud.intensities.min() >= 0.0 and cloud.intensities.max() <= 1.0
    assert cloud.rings.min() >= 0 and cloud.rings.max() < 8
    assert np.all(np.isfinite(cloud.points))
    # Downward rays from 2 m: hit range should be >= height.
    ranges = np.linalg.norm(cloud.points, axis=1)
    assert ranges.min() >= 0.5


def test_lidar_determinism():
    pose = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    a = GenesisLiDARScanner()
    a.configure(_lidar_cfg())
    c1 = a.scan(FlatTerrainScene(), pose)
    b = GenesisLiDARScanner()
    b.configure(_lidar_cfg())
    c2 = b.scan(FlatTerrainScene(), pose)
    np.testing.assert_array_equal(c1.points, c2.points)
    np.testing.assert_array_equal(c1.intensities, c2.intensities)


def test_lidar_partial_scan_requires_prior():
    s = GenesisLiDARScanner()
    s.configure(_lidar_cfg())
    with pytest.raises(RuntimeError):
        s.get_partial_scan(8)
    s.scan(FlatTerrainScene(), np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0]))
    partial = s.get_partial_scan(4)
    full = s.scan(FlatTerrainScene(), np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0]))
    assert partial.points.shape[0] <= full.points.shape[0]


def test_lidar_batch_scene_and_dust():
    s = GenesisLiDARScanner()
    s.configure(_lidar_cfg())
    pose = np.array([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0])
    cloud = s.scan(ConstantBatchScene(distance=5.0), pose)
    assert cloud.points.shape[0] > 0
    dusty = s.apply_dust_interference(80.0, cloud)
    # Heavy dust attenuates intensity and never increases the real return count
    # beyond original + sparse false returns.
    assert dusty.intensities.max() <= 1.0
    assert dusty.points.shape[0] <= cloud.points.shape[0] + 50


# --------------------------------------------------------------------------
# Stereo camera
# --------------------------------------------------------------------------


def _cam_cfg(**kw) -> CameraConfig:
    base = dict(
        resolution=(32, 24),
        baseline_m=0.12,
        fov_h_deg=60.0,
        fov_v_deg=45.0,
        focal_length_px=30.0,
        frame_rate_hz=30.0,
        depth_range_m=(0.5, 20.0),
        depth_noise_sigma=0.01,
        seed=7,
    )
    base.update(kw)
    return CameraConfig(**base)


def test_stereo_capture_shapes():
    cam = RaycastStereoCamera()
    cam.configure(_cam_cfg())
    pose = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    frame = cam.capture(ConstantBatchScene(distance=4.0), pose)
    assert frame.left_rgb.shape == (24, 32, 3)
    assert frame.right_rgb.shape == (24, 32, 3)
    assert frame.left_rgb.dtype == np.uint8
    assert frame.depth_map.shape == (24, 32)
    finite = np.isfinite(frame.depth_map)
    assert finite.any()
    assert np.nanmin(frame.depth_map[finite]) >= 0.5


def test_stereo_config_validation():
    cam = RaycastStereoCamera()
    with pytest.raises(ValueError):
        cam.configure(_cam_cfg(baseline_m=0.0))
    with pytest.raises(ValueError):
        cam.configure(_cam_cfg(depth_range_m=(5.0, 1.0)))


def test_stereo_rock_detection():
    cam = RaycastStereoCamera()
    cam.configure(_cam_cfg())
    # Background at 10 m with a closer 5x5 blob (a "rock") at 6 m.
    depth = np.full((24, 32), 10.0, dtype=np.float32)
    depth[10:15, 14:19] = 6.0
    rocks = cam.detect_rocks(depth)
    assert len(rocks) >= 1
    r = rocks[0]
    assert set(r) == {"bbox", "area_px", "center_3d", "height_m"}
    assert r["area_px"] >= 6
    assert r["height_m"] > 0.0


def test_stereo_navcam_pitched():
    cam = RaycastStereoCamera()
    cam.configure(_cam_cfg())
    pose = np.array([0.0, 0.0, 1.5, 1.0, 0.0, 0.0, 0.0])
    frame = cam.get_navcam_frame(FlatTerrainScene(), pose)
    # Downward-pitched camera over flat ground should see the surface.
    assert np.isfinite(frame.depth_map).any()


# --------------------------------------------------------------------------
# IMU + wheel encoder
# --------------------------------------------------------------------------


def test_imu_noise_and_bias_drift():
    imu = GenesisIMUSensor()
    imu.configure(
        IMUConfig(
            update_rate_hz=100.0,
            gyro_noise_sigma=0.01,
            accel_noise_sigma=0.05,
            gyro_bias_drift_deg_hr=50.0,
            seed=1,
        )
    )
    accs, gyros = [], []
    for _ in range(2000):
        r = imu.read(np.array([0.0, 0.0, 1.62]), np.zeros(3))
        accs.append(r.accel_xyz)
        gyros.append(r.gyro_xyz)
    accs = np.array(accs)
    # Accelerometer mean tracks the true signal within noise.
    assert abs(accs[:, 2].mean() - 1.62) < 0.05
    # Bias has random-walked away from zero.
    assert np.linalg.norm(imu.get_bias_state()) > 0.0


def test_imu_determinism():
    cfg = IMUConfig(
        update_rate_hz=100.0,
        gyro_noise_sigma=0.01,
        accel_noise_sigma=0.05,
        gyro_bias_drift_deg_hr=10.0,
        seed=99,
    )
    a, b = GenesisIMUSensor(), GenesisIMUSensor()
    a.configure(cfg)
    b.configure(cfg)
    for _ in range(50):
        ra = a.read(np.ones(3), np.ones(3) * 0.1)
        rb = b.read(np.ones(3), np.ones(3) * 0.1)
        np.testing.assert_array_equal(ra.gyro_xyz, rb.gyro_xyz)


def test_wheel_encoder_counts_and_velocity():
    enc = GenesisWheelEncoder()
    enc.configure(EncoderConfig(counts_per_rev=1024, update_rate_hz=100.0))
    omega = [2 * np.pi, 2 * np.pi]  # 1 rev/s on each wheel
    vels = []
    last = None
    prev_count = 0
    for _ in range(100):  # 1 s -> ~1 rev -> ~1024 counts
        last = enc.read(omega)
        assert last.counts[0] >= prev_count  # monotonic for forward motion
        prev_count = last.counts[0]
        vels.append(last.angular_velocities[0])
    assert last.counts[0] == pytest.approx(1024, abs=2)
    # Per-sample velocity is quantized (stair-steps), but the mean converges.
    assert float(np.mean(vels)) == pytest.approx(2 * np.pi, rel=0.02)


# --------------------------------------------------------------------------
# Beacon network
# --------------------------------------------------------------------------


def _beacon(pos, rng=2000.0, sigma=0.5) -> BeaconConfig:
    return BeaconConfig(
        position_xyz=np.array(pos, dtype=float),
        signal_range_m=rng,
        power_w=20.0,
        noise_sigma_m=sigma,
    )


def _net_with_4() -> TrilaterationBeaconNetwork:
    net = TrilaterationBeaconNetwork(seed=3)
    net.add_beacon("b0", _beacon([100.0, 0.0, 10.0]))
    net.add_beacon("b1", _beacon([-100.0, 0.0, 10.0]))
    net.add_beacon("b2", _beacon([0.0, 100.0, 10.0]))
    net.add_beacon("b3", _beacon([0.0, -100.0, 30.0]))
    return net


def test_beacon_fix_accuracy():
    net = _net_with_4()
    true = np.array([12.0, -7.0, 1.0])
    fix = net.compute_fix(true)
    assert fix is not None
    assert fix.num_beacons == 4
    assert np.linalg.norm(fix.position_xyz - true) < 3.0
    assert fix.covariance.shape == (3, 3)
    assert np.isfinite(fix.gdop)


def test_beacon_insufficient_returns_none():
    net = TrilaterationBeaconNetwork(seed=0)
    net.add_beacon("b0", _beacon([10.0, 0.0, 1.0]))
    net.add_beacon("b1", _beacon([0.0, 10.0, 1.0]))
    assert net.compute_fix(np.zeros(3)) is None  # < 3 beacons


def test_beacon_duplicate_and_remove():
    net = _net_with_4()
    with pytest.raises(ValueError):
        net.add_beacon("b0", _beacon([0.0, 0.0, 0.0]))
    net.remove_beacon("b0")
    with pytest.raises(KeyError):
        net.remove_beacon("b0")


def test_beacon_gdop_and_coverage():
    net = _net_with_4()
    assert np.isfinite(net.get_gdop_at(np.zeros(3)))
    # With only 3 visible beacons GDOP is undefined (inf).
    net.remove_beacon("b3")
    assert net.get_gdop_at(np.zeros(3)) == float("inf")
    cov = net.get_coverage_map(50.0)
    assert cov.ndim == 2 and cov.size > 0


# --------------------------------------------------------------------------
# Force-torque
# --------------------------------------------------------------------------


def _ft_cfg(**kw) -> FTConfig:
    base = dict(
        force_range_n=200.0,
        torque_range_nm=20.0,
        resolution_force=0.1,
        resolution_torque=0.01,
        update_rate_hz=1000.0,
        seed=5,
    )
    base.update(kw)
    return FTConfig(**base)


def test_ft_read_and_quantization():
    ft = GenesisForceTorqueSensor()
    ft.configure(_ft_cfg())
    r = ft.read(np.array([10.0, 0.0, -5.0, 0.1, 0.0, 0.0]))
    assert r.force_xyz.shape == (3,)
    assert r.torque_xyz.shape == (3,)
    # Within a few LSB of the input.
    assert abs(r.force_xyz[0] - 10.0) < 0.5
    assert not ft.check_overload()


def test_ft_overload_and_clip():
    ft = GenesisForceTorqueSensor()
    ft.configure(_ft_cfg())
    r = ft.read(np.array([500.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert ft.check_overload()
    assert abs(r.force_xyz[0]) <= 200.0  # saturated to range


def test_ft_tare():
    ft = GenesisForceTorqueSensor()
    ft.configure(_ft_cfg())
    bias = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ft.tare(bias)
    r = ft.read(bias)  # taring the same load -> ~zero
    assert abs(r.force_xyz[0]) < 0.5


def test_ft_config_validation():
    ft = GenesisForceTorqueSensor()
    with pytest.raises(ValueError):
        ft.configure(_ft_cfg(force_range_n=0.0))


# --------------------------------------------------------------------------
# Sun sensor
# --------------------------------------------------------------------------


def test_sun_valid_reading():
    s = GenesisSunSensor()
    s.configure(SunSensorConfig(accuracy_deg=0.5, update_rate_hz=1.0, seed=11))
    r = s.read(sun_azimuth_true=123.0, sun_elevation=30.0, in_shadow=False)
    assert r.valid
    assert abs(((r.azimuth_deg - 123.0 + 180) % 360) - 180) < 3.0


def test_sun_shadow_and_horizon_invalid():
    s = GenesisSunSensor()
    s.configure(
        SunSensorConfig(
            accuracy_deg=0.5,
            update_rate_hz=1.0,
            elevation_threshold_deg=0.0,
            seed=1,
        )
    )
    assert not s.read(90.0, 30.0, in_shadow=True).valid
    assert not s.read(90.0, -5.0, in_shadow=False).valid


def test_sun_determinism():
    cfg = SunSensorConfig(accuracy_deg=1.0, update_rate_hz=1.0, seed=4)
    a, b = GenesisSunSensor(), GenesisSunSensor()
    a.configure(cfg)
    b.configure(cfg)
    for _ in range(20):
        ra = a.read(45.0, 20.0, False)
        rb = b.read(45.0, 20.0, False)
        assert ra.azimuth_deg == rb.azimuth_deg
