"""
System 13: Data Logging Streams

Configurable multi-stream data logging for rover telemetry, sensor data,
camera streams, and mission events. Dense sensor arrays (LIDAR, camera,
occupancy maps) are written to HDF5; time-series structured records (rover
state, IMU, GPS, cable, power, events) are written to MCAP. All writes are
serialized on a background worker thread so the simulation loop never blocks
on disk I/O.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import numpy.typing as npt
from mcap.writer import Writer as _MCAPWriter

_LOG = logging.getLogger(__name__)


@dataclass
class LogConfig:
    """
    Configuration for data logging system.

    Attributes:
        output_dir: Directory path for log files.
        rover_state_rate_hz: Rover state logging rate (position, velocity, etc).
            Default: 50 Hz.
        sensor_rate_hz: Generic sensor logging rate. Varies by sensor type
            (lidar, imu, etc.). Default: varies per sensor.
        camera_fps: Camera frame capture rate in frames per second.
            Default: 30 fps.
        event_driven: If True, log events asynchronously when they occur
            (e.g., fault detection, phase change). Default: True.
        enable_hdf5: Enable HDF5 format output for structured data. Default: True.
        enable_mcap: Enable MCAP format output (ROS-compatible binary log).
            Default: False.
        enable_mp4: Enable MP4 video encoding for camera streams. Default: False.
    """

    output_dir: str
    rover_state_rate_hz: int = 50
    sensor_rate_hz: int = 100
    camera_fps: int = 30
    event_driven: bool = True
    enable_hdf5: bool = True
    enable_mcap: bool = False
    enable_mp4: bool = False


class StreamType(Enum):
    """
    Enumeration of data stream types for logging.

    Attributes:
        ROVER_STATE: Rover pose, velocity, orientation (50 Hz).
        SENSOR_LIDAR: Lidar point clouds (5-10 Hz depending on mode).
        SENSOR_IMU: Inertial measurement unit data (200 Hz).
        SENSOR_GPS: GPS position fixes and quality metrics (1 Hz).
        CAMERA_RGB: RGB color camera frames (30 fps).
        CAMERA_DEPTH: Depth map from stereo camera (30 fps).
        CABLE_STATE: Cable tension, spool state, length deployed (10 Hz).
        MISSION_EVENTS: Mission phase changes, faults, commands (event-driven).
        POWER_THERMAL: Battery voltage/current, motor temperatures (10 Hz).
        OCCUPANCY_MAP: Probabilistic occupancy grid snapshots (1 Hz).
    """

    ROVER_STATE = "rover_state"
    SENSOR_LIDAR = "sensor_lidar"
    SENSOR_IMU = "sensor_imu"
    SENSOR_GPS = "sensor_gps"
    CAMERA_RGB = "camera_rgb"
    CAMERA_DEPTH = "camera_depth"
    CABLE_STATE = "cable_state"
    MISSION_EVENTS = "mission_events"
    POWER_THERMAL = "power_thermal"
    OCCUPANCY_MAP = "occupancy_map"


_HDF5_STREAMS = frozenset(
    {
        StreamType.SENSOR_LIDAR,
        StreamType.CAMERA_RGB,
        StreamType.CAMERA_DEPTH,
        StreamType.OCCUPANCY_MAP,
    }
)
_MCAP_STREAMS = frozenset(
    {
        StreamType.ROVER_STATE,
        StreamType.SENSOR_IMU,
        StreamType.SENSOR_GPS,
        StreamType.CABLE_STATE,
        StreamType.MISSION_EVENTS,
        StreamType.POWER_THERMAL,
    }
)


class DataLogger(ABC):
    """
    Abstract interface for multi-stream data logging.

    Manages simultaneous logging of rover states, sensor readings, camera frames,
    and mission events to various output formats (HDF5, MCAP, MP4) with
    configurable sampling rates and event-driven triggers.

    Key features:
    - Asynchronous buffering for high-rate streams
    - Configurable compression and output formats
    - Real-time size estimation for storage management
    - Automatic rotation and archival
    """

    @abstractmethod
    def initialize(self, config: LogConfig) -> None:
        """
        Initialize logging system with configuration.

        Creates output directories, opens file handles, and prepares buffers
        for configured streams.

        Args:
            config: Logging configuration.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def log_rover_state(
        self,
        rover_id: str,
        state: dict,
    ) -> None:
        """
        Log rover pose and motion state.

        Logs rover state at configured rate (50 Hz). State dict should contain
        keys like 'position', 'velocity', 'orientation', 'accel', etc.

        Args:
            rover_id: Rover identifier.
            state: State dictionary with position, velocity, orientation, etc.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def log_sensor_reading(
        self,
        sensor_type: StreamType,
        data: dict,
    ) -> None:
        """
        Log generic sensor measurement data.

        Logs sensor readings (IMU, GPS, lidar, etc.) at appropriate rates.
        Data dict contains sensor-specific fields (e.g., IMU contains
        'accel_xyz', 'gyro_xyz', 'timestamp').

        Args:
            sensor_type: Type of sensor measurement (SENSOR_IMU, SENSOR_GPS, etc).
            data: Sensor-specific data dictionary.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def log_camera_frame(
        self,
        frame: npt.NDArray[np.uint8],
        stream_type: StreamType,
    ) -> None:
        """
        Log camera frame (RGB or depth).

        Logs raw or compressed camera imagery at configured fps. Depth frames
        can be stored as uint16 (millimeters) for compression efficiency.

        Args:
            frame: Image array. RGB: HxWx3, Depth: HxWx1.
            stream_type: CAMERA_RGB or CAMERA_DEPTH.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def log_event(
        self,
        event_type: str,
        payload: dict,
    ) -> None:
        """
        Log mission event with arbitrary payload.

        Records event-driven data (phase changes, fault detection, command
        issuance, etc.) with timestamp and associated metadata.

        Args:
            event_type: Event category (e.g., 'phase_change', 'fault_detected').
            payload: Event-specific data dictionary.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def flush(self) -> None:
        """
        Flush all buffered data to disk.

        Ensures all queued logging data is written to output files. Called
        periodically or on demand for data durability.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """
        Close logging system and finalize output files.

        Flushes all pending data, closes file handles, and performs cleanup.
        After close() is called, no further logging is possible without
        reinitializing.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def get_estimated_size_bytes(self) -> int:
        """
        Get estimated cumulative size of all logged data.

        Returns running estimate of total storage used by all output streams,
        useful for monitoring disk quota and archival decisions.

        Returns:
            Estimated total size in bytes.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Internal record types pushed across the worker queue
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _StateRecord:
    timestamp_ns: int
    rover_id: str
    state: dict


@dataclass(slots=True)
class _SensorRecord:
    timestamp_ns: int
    stream: StreamType
    data: dict


@dataclass(slots=True)
class _CameraRecord:
    timestamp_ns: int
    stream: StreamType
    frame: np.ndarray


@dataclass(slots=True)
class _EventRecord:
    timestamp_ns: int
    event_type: str
    payload: dict


class _FlushSentinel:
    __slots__ = ("event",)

    def __init__(self) -> None:
        self.event = threading.Event()


class _StopSentinel:
    __slots__ = ()


# ---------------------------------------------------------------------------
# HDF5 backend — dense sensor arrays (cameras, LIDAR, occupancy maps)
# ---------------------------------------------------------------------------


class _HDF5Backend:
    """HDF5 sink for dense sensor arrays.

    Layout:
        /sensor_lidar/
            points          (N_total_points, D)  float32   — D = 3 or 4
            offsets         (N_scans + 1,)       uint64    — points[off[i]:off[i+1]]
            timestamps_ns   (N_scans,)           int64
        /camera_rgb/
            frames          (N, H, W, C)         uint8     — resizable along axis 0
            timestamps_ns   (N,)                 int64
        /camera_depth/
            frames          (N, H, W) or (N, H, W, 1)  uint16 — resizable
            timestamps_ns   (N,)                 int64
        /occupancy_map/
            frames          (N, H, W)            float32   — resizable
            timestamps_ns   (N,)                 int64
    """

    _COMPRESSION = "gzip"
    _COMPRESSION_OPTS = 4

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file: Optional[h5py.File] = h5py.File(path, "w")
        self._frame_shape: dict[StreamType, tuple[int, ...]] = {}
        self._lidar_total_points = 0

    @property
    def path(self) -> Path:
        return self._path

    def write_camera_frame(
        self, stream: StreamType, frame: np.ndarray, ts_ns: int
    ) -> None:
        assert self._file is not None
        group_name = stream.value
        if group_name not in self._file:
            grp = self._file.create_group(group_name)
            chunk = (1,) + frame.shape
            grp.create_dataset(
                "frames",
                shape=(0,) + frame.shape,
                maxshape=(None,) + frame.shape,
                dtype=frame.dtype,
                chunks=chunk,
                compression=self._COMPRESSION,
                compression_opts=self._COMPRESSION_OPTS,
            )
            grp.create_dataset(
                "timestamps_ns",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(256,),
            )
            self._frame_shape[stream] = frame.shape
        else:
            expected = self._frame_shape.get(stream)
            if expected is not None and frame.shape != expected:
                raise ValueError(
                    f"{stream.name} frame shape {frame.shape} does not match "
                    f"established shape {expected}"
                )
            grp = self._file[group_name]

        frames = grp["frames"]
        timestamps = grp["timestamps_ns"]
        n = frames.shape[0]
        frames.resize(n + 1, axis=0)
        frames[n] = frame
        timestamps.resize(n + 1, axis=0)
        timestamps[n] = ts_ns

    def write_lidar_scan(self, data: dict, ts_ns: int) -> None:
        assert self._file is not None
        points = data.get("points")
        if points is None:
            raise ValueError("LIDAR scan missing 'points' array")
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2:
            raise ValueError(
                f"LIDAR 'points' must be 2-D (N, D); got shape {points.shape}"
            )

        group_name = StreamType.SENSOR_LIDAR.value
        if group_name not in self._file:
            d = points.shape[1]
            grp = self._file.create_group(group_name)
            grp.create_dataset(
                "points",
                shape=(0, d),
                maxshape=(None, d),
                dtype=np.float32,
                chunks=(max(1024, points.shape[0] or 1024), d),
                compression=self._COMPRESSION,
                compression_opts=self._COMPRESSION_OPTS,
            )
            grp.create_dataset(
                "offsets",
                shape=(1,),
                maxshape=(None,),
                dtype=np.uint64,
                chunks=(256,),
                data=np.zeros(1, dtype=np.uint64),
            )
            grp.create_dataset(
                "timestamps_ns",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(256,),
            )
            self._lidar_total_points = 0
        else:
            grp = self._file[group_name]
            existing_d = grp["points"].shape[1]
            if points.shape[1] != existing_d:
                raise ValueError(
                    f"LIDAR point dimension {points.shape[1]} does not match "
                    f"established dimension {existing_d}"
                )

        pts_ds = grp["points"]
        off_ds = grp["offsets"]
        ts_ds = grp["timestamps_ns"]

        n_new = points.shape[0]
        old_total = self._lidar_total_points
        new_total = old_total + n_new
        if n_new > 0:
            pts_ds.resize(new_total, axis=0)
            pts_ds[old_total:new_total] = points
        self._lidar_total_points = new_total

        n_scans = off_ds.shape[0]
        off_ds.resize(n_scans + 1, axis=0)
        off_ds[n_scans] = new_total

        n_ts = ts_ds.shape[0]
        ts_ds.resize(n_ts + 1, axis=0)
        ts_ds[n_ts] = ts_ns

    def write_occupancy_map(self, data: dict, ts_ns: int) -> None:
        assert self._file is not None
        grid = data.get("grid")
        if grid is None:
            grid = data.get("map")
        if grid is None:
            raise ValueError("Occupancy map missing 'grid' (or 'map') array")
        grid = np.asarray(grid, dtype=np.float32)
        if grid.ndim != 2:
            raise ValueError(
                f"Occupancy 'grid' must be 2-D (H, W); got shape {grid.shape}"
            )

        group_name = StreamType.OCCUPANCY_MAP.value
        if group_name not in self._file:
            grp = self._file.create_group(group_name)
            grp.create_dataset(
                "frames",
                shape=(0,) + grid.shape,
                maxshape=(None,) + grid.shape,
                dtype=np.float32,
                chunks=(1,) + grid.shape,
                compression=self._COMPRESSION,
                compression_opts=self._COMPRESSION_OPTS,
            )
            grp.create_dataset(
                "timestamps_ns",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int64,
                chunks=(256,),
            )
            self._frame_shape[StreamType.OCCUPANCY_MAP] = grid.shape
        else:
            expected = self._frame_shape.get(StreamType.OCCUPANCY_MAP)
            if expected is not None and grid.shape != expected:
                raise ValueError(
                    f"Occupancy grid shape {grid.shape} does not match "
                    f"established shape {expected}"
                )
            grp = self._file[group_name]

        frames = grp["frames"]
        timestamps = grp["timestamps_ns"]
        n = frames.shape[0]
        frames.resize(n + 1, axis=0)
        frames[n] = grid
        timestamps.resize(n + 1, axis=0)
        timestamps[n] = ts_ns

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


# ---------------------------------------------------------------------------
# MCAP backend — time-series structured records
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


class _MCAPBackend:
    """MCAP sink for time-series structured records.

    Each topic is registered lazily with a generic JSON schema. Messages are
    encoded as UTF-8 JSON bytes. The resulting MCAP file is a valid ROS2 bag
    and readable by Foxglove Studio, mcap CLI, and other downstream tooling.
    """

    _SCHEMA_NAME = "moon_rover/JsonRecord"

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fp = open(path, "wb")
        self._writer = _MCAPWriter(self._fp)
        self._writer.start()
        self._schema_id = self._writer.register_schema(
            name=self._SCHEMA_NAME, encoding="jsonschema", data=b"{}"
        )
        self._channels: dict[str, int] = {}
        self._sequences: dict[str, int] = defaultdict(int)
        self._finished = False

    @property
    def path(self) -> Path:
        return self._path

    def _channel_for(self, topic: str) -> int:
        ch_id = self._channels.get(topic)
        if ch_id is None:
            ch_id = self._writer.register_channel(
                topic=topic,
                message_encoding="json",
                schema_id=self._schema_id,
            )
            self._channels[topic] = ch_id
        return ch_id

    def write_message(self, topic: str, ts_ns: int, payload: dict) -> None:
        ch_id = self._channel_for(topic)
        seq = self._sequences[topic]
        self._sequences[topic] = seq + 1
        data = json.dumps(payload, default=_json_default).encode("utf-8")
        self._writer.add_message(
            channel_id=ch_id,
            log_time=ts_ns,
            data=data,
            publish_time=ts_ns,
            sequence=seq,
        )

    def flush(self) -> None:
        if not self._finished:
            self._fp.flush()

    def close(self) -> None:
        if not self._finished:
            try:
                self._writer.finish()
            finally:
                self._fp.close()
                self._finished = True


# ---------------------------------------------------------------------------
# Concrete DataLogger
# ---------------------------------------------------------------------------


class MultiStreamLogger(DataLogger):
    """Production multi-stream data logger.

    Routing:
        Dense arrays (LIDAR, camera, occupancy map) -> HDF5
        Time-series records (rover state, IMU, GPS, cable, power, events) -> MCAP

    All ``log_*`` methods enqueue a record and return immediately; a background
    worker thread serializes and writes them, keeping the simulation loop free
    of disk I/O. ``flush()`` waits for the in-flight queue to drain before
    syncing file handles. ``close()`` flushes, stops the worker, and finalizes
    the output files.
    """

    _WORKER_JOIN_TIMEOUT_S = 30.0

    def __init__(self) -> None:
        self._config: Optional[LogConfig] = None
        self._hdf5: Optional[_HDF5Backend] = None
        self._mcap: Optional[_MCAPBackend] = None
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        self._initialized = False
        self._closed = False

    def initialize(self, config: LogConfig) -> None:
        if self._initialized:
            raise RuntimeError("MultiStreamLogger already initialized")

        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if config.enable_hdf5:
            self._hdf5 = _HDF5Backend(out_dir / "log.h5")
        if config.enable_mcap:
            self._mcap = _MCAPBackend(out_dir / "log.mcap")

        self._config = config
        self._queue = queue.Queue()
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="MoonRoverLogWorker",
            daemon=True,
        )
        self._worker.start()
        self._initialized = True

    def log_rover_state(self, rover_id: str, state: dict) -> None:
        self._ensure_open()
        if self._mcap is None:
            return
        ts = _extract_timestamp_ns(state)
        assert self._queue is not None
        self._queue.put(_StateRecord(ts, rover_id, dict(state)))

    def log_sensor_reading(self, sensor_type: StreamType, data: dict) -> None:
        self._ensure_open()
        ts = _extract_timestamp_ns(data)
        if sensor_type in _HDF5_STREAMS:
            if self._hdf5 is None:
                return
        elif sensor_type in _MCAP_STREAMS:
            if self._mcap is None:
                return
        else:
            raise ValueError(f"Unknown sensor stream: {sensor_type!r}")
        assert self._queue is not None
        self._queue.put(_SensorRecord(ts, sensor_type, dict(data)))

    def log_camera_frame(
        self, frame: npt.NDArray[np.uint8], stream_type: StreamType
    ) -> None:
        self._ensure_open()
        if stream_type not in (StreamType.CAMERA_RGB, StreamType.CAMERA_DEPTH):
            raise ValueError(
                f"log_camera_frame requires CAMERA_RGB or CAMERA_DEPTH; got {stream_type!r}"
            )
        if self._hdf5 is None:
            return
        ts = time.time_ns()
        assert self._queue is not None
        self._queue.put(_CameraRecord(ts, stream_type, np.array(frame, copy=True)))

    def log_event(self, event_type: str, payload: dict) -> None:
        self._ensure_open()
        if self._mcap is None:
            return
        ts = _extract_timestamp_ns(payload)
        assert self._queue is not None
        self._queue.put(_EventRecord(ts, event_type, dict(payload)))

    def flush(self) -> None:
        self._ensure_open()
        sentinel = _FlushSentinel()
        assert self._queue is not None
        self._queue.put(sentinel)
        sentinel.event.wait()

    def close(self) -> None:
        if not self._initialized or self._closed:
            return
        assert self._queue is not None
        self._queue.put(_StopSentinel())
        if self._worker is not None:
            self._worker.join(timeout=self._WORKER_JOIN_TIMEOUT_S)
        if self._hdf5 is not None:
            self._hdf5.close()
        if self._mcap is not None:
            self._mcap.close()
        self._closed = True

    def get_estimated_size_bytes(self) -> int:
        total = 0
        for backend in (self._hdf5, self._mcap):
            if backend is None:
                continue
            p = backend.path
            try:
                total += p.stat().st_size
            except OSError:
                pass
        return total

    def _ensure_open(self) -> None:
        if not self._initialized:
            raise RuntimeError("MultiStreamLogger has not been initialized")
        if self._closed:
            raise RuntimeError("MultiStreamLogger is closed")

    def _worker_loop(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            try:
                if isinstance(item, _StopSentinel):
                    return
                if isinstance(item, _FlushSentinel):
                    try:
                        if self._hdf5 is not None:
                            self._hdf5.flush()
                        if self._mcap is not None:
                            self._mcap.flush()
                    finally:
                        item.event.set()
                    continue
                self._dispatch(item)
            except Exception:
                _LOG.exception("MultiStreamLogger worker failed to write record")

    def _dispatch(self, item: object) -> None:
        if isinstance(item, _StateRecord):
            assert self._mcap is not None
            self._mcap.write_message(
                topic=f"/rover/{item.rover_id}/state",
                ts_ns=item.timestamp_ns,
                payload=item.state,
            )
            return
        if isinstance(item, _SensorRecord):
            if item.stream == StreamType.SENSOR_LIDAR:
                assert self._hdf5 is not None
                self._hdf5.write_lidar_scan(item.data, item.timestamp_ns)
            elif item.stream == StreamType.OCCUPANCY_MAP:
                assert self._hdf5 is not None
                self._hdf5.write_occupancy_map(item.data, item.timestamp_ns)
            else:
                assert self._mcap is not None
                self._mcap.write_message(
                    topic=f"/{item.stream.value}",
                    ts_ns=item.timestamp_ns,
                    payload=item.data,
                )
            return
        if isinstance(item, _CameraRecord):
            assert self._hdf5 is not None
            self._hdf5.write_camera_frame(item.stream, item.frame, item.timestamp_ns)
            return
        if isinstance(item, _EventRecord):
            assert self._mcap is not None
            self._mcap.write_message(
                topic="/events",
                ts_ns=item.timestamp_ns,
                payload={"event_type": item.event_type, **item.payload},
            )
            return
        raise TypeError(f"Unknown log record type: {type(item).__name__}")


def _extract_timestamp_ns(data: dict) -> int:
    """Pick a nanosecond timestamp from the record payload.

    Resolution order:
        1. ``data['timestamp_ns']`` — explicit nanoseconds (int).
        2. ``data['timestamp']`` — seconds (int or float).
        3. ``time.time_ns()`` — wall clock fallback.
    """
    if not isinstance(data, dict):
        return time.time_ns()
    ts_ns = data.get("timestamp_ns")
    if ts_ns is not None:
        return int(ts_ns)
    ts = data.get("timestamp")
    if ts is not None:
        return int(float(ts) * 1e9)
    return time.time_ns()
