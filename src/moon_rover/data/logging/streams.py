"""
System 13: Data Logging Streams

Configurable multi-stream data logging for rover telemetry, sensor data,
camera streams, and mission events. Supports HDF5, MCAP, and video output
formats with configurable sampling rates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt


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
