"""Unit tests for src/moon_rover/data/logging/streams.py.

Covers:
- Camera frames written to HDF5 with timestamps and shape consistency.
- LIDAR scans persisted to HDF5 via flat points + offsets table.
- Occupancy map snapshots stored as resizable 4D HDF5 dataset.
- Rover state / sensor / event records written to MCAP with per-stream topics.
- Async worker drains the queue via flush() without blocking the producer.
- close() finalizes both backends and is idempotent.
- Disabled backends silently drop traffic; shape mismatches raise.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from mcap.reader import make_reader

from moon_rover.data.logging.streams import (
    LogConfig,
    MultiStreamLogger,
    StreamType,
)


def _new_logger(tmp_path: Path, **overrides) -> MultiStreamLogger:
    cfg = LogConfig(
        output_dir=str(tmp_path),
        enable_hdf5=overrides.pop("enable_hdf5", True),
        enable_mcap=overrides.pop("enable_mcap", True),
        **overrides,
    )
    logger = MultiStreamLogger()
    logger.initialize(cfg)
    return logger


def _read_mcap_messages(path: Path) -> list[tuple[str, dict, int]]:
    out: list[tuple[str, dict, int]] = []
    with open(path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages():
            payload = json.loads(message.data.decode("utf-8"))
            out.append((channel.topic, payload, message.log_time))
    return out


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_initialize_creates_output_files(tmp_path):
    logger = _new_logger(tmp_path)
    try:
        assert (tmp_path / "log.h5").exists()
        assert (tmp_path / "log.mcap").exists()
    finally:
        logger.close()


def test_double_initialize_raises(tmp_path):
    logger = _new_logger(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="already initialized"):
            logger.initialize(LogConfig(output_dir=str(tmp_path)))
    finally:
        logger.close()


def test_close_is_idempotent(tmp_path):
    logger = _new_logger(tmp_path)
    logger.close()
    logger.close()  # second call should be a no-op


def test_logging_before_initialize_raises(tmp_path):
    logger = MultiStreamLogger()
    with pytest.raises(RuntimeError, match="has not been initialized"):
        logger.log_event("boot", {"ok": True})


def test_logging_after_close_raises(tmp_path):
    logger = _new_logger(tmp_path)
    logger.close()
    with pytest.raises(RuntimeError, match="is closed"):
        logger.log_event("late", {})


# ---------------------------------------------------------------------------
# HDF5 — cameras
# ---------------------------------------------------------------------------


def test_camera_rgb_round_trip(tmp_path):
    logger = _new_logger(tmp_path, enable_mcap=False)
    frame_a = np.random.randint(0, 255, (4, 6, 3), dtype=np.uint8)
    frame_b = np.random.randint(0, 255, (4, 6, 3), dtype=np.uint8)
    try:
        logger.log_camera_frame(frame_a, StreamType.CAMERA_RGB)
        logger.log_camera_frame(frame_b, StreamType.CAMERA_RGB)
        logger.flush()
    finally:
        logger.close()

    with h5py.File(tmp_path / "log.h5", "r") as f:
        frames = f["camera_rgb/frames"][:]
        ts = f["camera_rgb/timestamps_ns"][:]
    assert frames.shape == (2, 4, 6, 3)
    np.testing.assert_array_equal(frames[0], frame_a)
    np.testing.assert_array_equal(frames[1], frame_b)
    assert ts.shape == (2,)
    assert ts[1] >= ts[0]


def test_camera_depth_uint16_preserved(tmp_path):
    logger = _new_logger(tmp_path, enable_mcap=False)
    depth = np.arange(48, dtype=np.uint16).reshape(6, 8)
    try:
        logger.log_camera_frame(depth, StreamType.CAMERA_DEPTH)
        logger.flush()
    finally:
        logger.close()

    with h5py.File(tmp_path / "log.h5", "r") as f:
        ds = f["camera_depth/frames"]
        assert ds.dtype == np.uint16
        np.testing.assert_array_equal(ds[0], depth)


def test_camera_frame_shape_mismatch_is_logged_not_raised(tmp_path, caplog):
    logger = _new_logger(tmp_path, enable_mcap=False)
    good = np.zeros((4, 6, 3), dtype=np.uint8)
    bad = np.zeros((5, 6, 3), dtype=np.uint8)
    try:
        logger.log_camera_frame(good, StreamType.CAMERA_RGB)
        logger.flush()
        with caplog.at_level("ERROR", logger="moon_rover.data.logging.streams"):
            logger.log_camera_frame(bad, StreamType.CAMERA_RGB)
            logger.flush()
        worker_errs = [
            r for r in caplog.records if "failed to write record" in r.getMessage()
        ]
        assert worker_errs, "worker did not log the dispatch failure"
        exc_msgs = [str(r.exc_info[1]) for r in worker_errs if r.exc_info]
        assert any("does not match" in m for m in exc_msgs)
    finally:
        logger.close()


def test_log_camera_with_non_camera_stream_raises(tmp_path):
    logger = _new_logger(tmp_path, enable_mcap=False)
    try:
        with pytest.raises(ValueError, match="CAMERA_RGB or CAMERA_DEPTH"):
            logger.log_camera_frame(np.zeros((1, 1, 3), np.uint8), StreamType.SENSOR_IMU)
    finally:
        logger.close()


# ---------------------------------------------------------------------------
# HDF5 — LIDAR
# ---------------------------------------------------------------------------


def test_lidar_scans_round_trip_with_offsets(tmp_path):
    logger = _new_logger(tmp_path, enable_mcap=False)
    scan_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    scan_b = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.float32)
    try:
        logger.log_sensor_reading(StreamType.SENSOR_LIDAR, {"points": scan_a, "timestamp": 1.0})
        logger.log_sensor_reading(StreamType.SENSOR_LIDAR, {"points": scan_b, "timestamp": 2.0})
        logger.flush()
    finally:
        logger.close()

    with h5py.File(tmp_path / "log.h5", "r") as f:
        points = f["sensor_lidar/points"][:]
        offsets = f["sensor_lidar/offsets"][:]
        ts = f["sensor_lidar/timestamps_ns"][:]
    assert points.shape == (5, 3)
    np.testing.assert_array_equal(offsets, np.array([0, 2, 5], dtype=np.uint64))
    np.testing.assert_allclose(points[0:2], scan_a)
    np.testing.assert_allclose(points[2:5], scan_b)
    assert list(ts) == [1_000_000_000, 2_000_000_000]


def test_lidar_empty_scan_records_offset(tmp_path):
    logger = _new_logger(tmp_path, enable_mcap=False)
    scan = np.array([[0, 0, 0]], dtype=np.float32)
    empty = np.empty((0, 3), dtype=np.float32)
    try:
        logger.log_sensor_reading(StreamType.SENSOR_LIDAR, {"points": scan})
        logger.log_sensor_reading(StreamType.SENSOR_LIDAR, {"points": empty})
        logger.flush()
    finally:
        logger.close()

    with h5py.File(tmp_path / "log.h5", "r") as f:
        offsets = f["sensor_lidar/offsets"][:]
        ts = f["sensor_lidar/timestamps_ns"][:]
    np.testing.assert_array_equal(offsets, np.array([0, 1, 1], dtype=np.uint64))
    assert ts.shape == (2,)


# ---------------------------------------------------------------------------
# HDF5 — occupancy map
# ---------------------------------------------------------------------------


def test_occupancy_map_snapshots_round_trip(tmp_path):
    logger = _new_logger(tmp_path, enable_mcap=False)
    grid_a = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    grid_b = (grid_a * 0.5).astype(np.float32)
    try:
        logger.log_sensor_reading(
            StreamType.OCCUPANCY_MAP, {"grid": grid_a, "timestamp": 0.5}
        )
        logger.log_sensor_reading(
            StreamType.OCCUPANCY_MAP, {"grid": grid_b, "timestamp": 1.0}
        )
        logger.flush()
    finally:
        logger.close()

    with h5py.File(tmp_path / "log.h5", "r") as f:
        frames = f["occupancy_map/frames"][:]
        ts = f["occupancy_map/timestamps_ns"][:]
    assert frames.shape == (2, 3, 4)
    np.testing.assert_allclose(frames[0], grid_a)
    np.testing.assert_allclose(frames[1], grid_b)
    assert list(ts) == [500_000_000, 1_000_000_000]


# ---------------------------------------------------------------------------
# MCAP — time-series records
# ---------------------------------------------------------------------------


def test_rover_state_written_to_mcap_per_rover_topic(tmp_path):
    logger = _new_logger(tmp_path, enable_hdf5=False)
    try:
        logger.log_rover_state(
            "rover_a", {"position": [1.0, 2.0, 3.0], "velocity": [0.1, 0.0, 0.0]}
        )
        logger.log_rover_state(
            "rover_b", {"position": [4.0, 5.0, 6.0], "velocity": [0.0, -0.1, 0.0]}
        )
        logger.flush()
    finally:
        logger.close()

    msgs = _read_mcap_messages(tmp_path / "log.mcap")
    topics = sorted({topic for topic, _, _ in msgs})
    assert topics == ["/rover/rover_a/state", "/rover/rover_b/state"]
    payload_a = next(p for t, p, _ in msgs if t == "/rover/rover_a/state")
    assert payload_a["position"] == [1.0, 2.0, 3.0]


def test_imu_sensor_written_to_mcap(tmp_path):
    logger = _new_logger(tmp_path, enable_hdf5=False)
    try:
        logger.log_sensor_reading(
            StreamType.SENSOR_IMU,
            {"accel_xyz": [0.0, 0.0, -1.62], "gyro_xyz": [0, 0, 0], "timestamp": 0.5},
        )
        logger.flush()
    finally:
        logger.close()

    msgs = _read_mcap_messages(tmp_path / "log.mcap")
    assert len(msgs) == 1
    topic, payload, log_time = msgs[0]
    assert topic == "/sensor_imu"
    assert payload["accel_xyz"] == [0.0, 0.0, -1.62]
    assert log_time == 500_000_000


def test_events_written_to_events_topic(tmp_path):
    logger = _new_logger(tmp_path, enable_hdf5=False)
    try:
        logger.log_event("phase_change", {"from": "boot", "to": "drive"})
        logger.log_event("fault", {"code": "MOTOR_OVERHEAT", "wheel": 2})
        logger.flush()
    finally:
        logger.close()

    msgs = _read_mcap_messages(tmp_path / "log.mcap")
    assert {topic for topic, _, _ in msgs} == {"/events"}
    event_types = sorted(p["event_type"] for _, p, _ in msgs)
    assert event_types == ["fault", "phase_change"]


def test_numpy_payload_serialized_to_json(tmp_path):
    logger = _new_logger(tmp_path, enable_hdf5=False)
    try:
        logger.log_event(
            "telemetry",
            {
                "pose": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "count": np.int64(7),
                "ok": np.bool_(True),
            },
        )
        logger.flush()
    finally:
        logger.close()

    msgs = _read_mcap_messages(tmp_path / "log.mcap")
    assert len(msgs) == 1
    _, payload, _ = msgs[0]
    assert payload["pose"] == [1.0, 2.0, 3.0]
    assert payload["count"] == 7
    assert payload["ok"] is True


# ---------------------------------------------------------------------------
# Routing and backend toggles
# ---------------------------------------------------------------------------


def test_camera_drop_when_hdf5_disabled(tmp_path):
    logger = _new_logger(tmp_path, enable_hdf5=False, enable_mcap=True)
    try:
        logger.log_camera_frame(np.zeros((2, 2, 3), np.uint8), StreamType.CAMERA_RGB)
        logger.flush()
    finally:
        logger.close()
    assert not (tmp_path / "log.h5").exists()


def test_rover_state_drop_when_mcap_disabled(tmp_path):
    logger = _new_logger(tmp_path, enable_hdf5=True, enable_mcap=False)
    try:
        logger.log_rover_state("rover_a", {"position": [0, 0, 0]})
        logger.flush()
    finally:
        logger.close()
    assert not (tmp_path / "log.mcap").exists()


# ---------------------------------------------------------------------------
# Async + size reporting
# ---------------------------------------------------------------------------


def test_burst_writes_drain_via_flush(tmp_path):
    """High-frequency producer should not block; flush() drains everything."""
    logger = _new_logger(tmp_path)
    n = 200
    try:
        for i in range(n):
            logger.log_sensor_reading(
                StreamType.SENSOR_IMU,
                {"accel_xyz": [0.0, 0.0, float(i)], "timestamp": i * 0.001},
            )
        logger.flush()
    finally:
        logger.close()

    msgs = _read_mcap_messages(tmp_path / "log.mcap")
    assert len(msgs) == n


def test_get_estimated_size_grows_after_writes(tmp_path):
    logger = _new_logger(tmp_path)
    try:
        logger.flush()
        empty_size = logger.get_estimated_size_bytes()
        for _ in range(50):
            logger.log_event("tick", {"v": 1})
            logger.log_camera_frame(
                np.zeros((8, 8, 3), dtype=np.uint8), StreamType.CAMERA_RGB
            )
        logger.flush()
        populated_size = logger.get_estimated_size_bytes()
    finally:
        logger.close()
    assert populated_size > empty_size


def test_extract_timestamp_units(tmp_path):
    """`timestamp` is seconds (int or float); `timestamp_ns` is explicit ns."""
    logger = _new_logger(tmp_path, enable_hdf5=False)
    try:
        logger.log_event("a", {"timestamp": 1.5})  # float seconds
        logger.log_event("b", {"timestamp": 2})  # int seconds
        logger.log_event("c", {"timestamp_ns": 3_000_000_000})  # explicit ns
        logger.flush()
    finally:
        logger.close()

    msgs = _read_mcap_messages(tmp_path / "log.mcap")
    by_event = {p["event_type"]: t for _, p, t in msgs}
    assert by_event["a"] == 1_500_000_000
    assert by_event["b"] == 2_000_000_000
    assert by_event["c"] == 3_000_000_000
