"""
System 11.2: Localisation — Extended Kalman Filter

Extended Kalman Filter implementation for multi-sensor fusion combining IMU,
wheel encoders, GPS, sun sensor, and lidar scan matching for robust localization
in lunar regolith with challenging GPS conditions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


@dataclass
class EKFState:
    """
    Complete state estimate for Extended Kalman Filter.

    Maintains 15-dimensional state vector: position (3), velocity (3), orientation
    (3 Euler angles), and gyroscope bias (3) with associated covariance matrix.

    Attributes:
        position_xyz: 3-element [x, y, z] position in world frame (meters).
        velocity_xyz: 3-element [vx, vy, vz] velocity in world frame (m/s).
        orientation_rpy: 3-element [roll, pitch, yaw] Euler angles (radians).
        gyro_bias_xyz: 3-element gyroscope bias [bx, by, bz] (rad/s).
        covariance: 15x15 symmetric positive-definite covariance matrix.
    """

    position_xyz: npt.NDArray[np.float64]
    velocity_xyz: npt.NDArray[np.float64]
    orientation_rpy: npt.NDArray[np.float64]
    gyro_bias_xyz: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]


@dataclass
class EKFConfig:
    """
    Configuration for Extended Kalman Filter.

    Specifies state dimensions, sensor update rates, and fusion strategy.

    Attributes:
        state_dim: Dimensionality of state vector. Fixed at 15.
        imu_rate_hz: IMU prediction update rate (Hz). Default: 200 Hz.
        encoder_rate_hz: Wheel encoder measurement rate (Hz). Default: 50 Hz.
        gps_rate_hz: GPS measurement rate (Hz). Default: 1 Hz.
        scan_match_rate_hz: Lidar scan matching (ICP) rate (Hz). Default: 5 Hz.
    """

    state_dim: int = 15
    imu_rate_hz: int = 200
    encoder_rate_hz: int = 50
    gps_rate_hz: int = 1
    scan_match_rate_hz: int = 5


class IMUReading(Protocol):
    """Protocol for IMU sensor readings."""

    @property
    def timestamp(self) -> float:
        """Sensor timestamp in seconds."""
        ...

    @property
    def accel_xyz(self) -> npt.NDArray[np.float64]:
        """3-element acceleration vector (m/s^2)."""
        ...

    @property
    def gyro_xyz(self) -> npt.NDArray[np.float64]:
        """3-element angular velocity vector (rad/s)."""
        ...


class EncoderReading(Protocol):
    """Protocol for wheel encoder measurements."""

    @property
    def timestamp(self) -> float:
        """Sensor timestamp in seconds."""
        ...

    @property
    def left_distance_m(self) -> float:
        """Distance traveled by left wheel (meters)."""
        ...

    @property
    def right_distance_m(self) -> float:
        """Distance traveled by right wheel (meters)."""
        ...

    @property
    def track_width_m(self) -> float:
        """Distance between left and right wheels (meters)."""
        ...


class GPSFix(Protocol):
    """Protocol for GPS position fixes."""

    @property
    def timestamp(self) -> float:
        """Sensor timestamp in seconds."""
        ...

    @property
    def position_xyz(self) -> npt.NDArray[np.float64]:
        """3-element [x, y, z] position (meters)."""
        ...

    @property
    def position_covariance(self) -> npt.NDArray[np.float64]:
        """3x3 covariance matrix for position (m^2)."""
        ...

    @property
    def fix_quality(self) -> str:
        """Quality indicator: 'poor', 'good', 'excellent'."""
        ...


class SunReading(Protocol):
    """Protocol for sun sensor heading measurements."""

    @property
    def timestamp(self) -> float:
        """Sensor timestamp in seconds."""
        ...

    @property
    def heading_rad(self) -> float:
        """Estimated heading/yaw angle (radians)."""
        ...

    @property
    def confidence(self) -> float:
        """Confidence in measurement [0, 1]."""
        ...


class LocalizationEKF(ABC):
    """
    Extended Kalman Filter for multi-sensor localization.

    Fuses measurements from IMU (200 Hz), wheel encoders (50 Hz), GPS (1 Hz),
    sun sensor (1 Hz), and lidar scan matching (5 Hz) to maintain a robust
    position and orientation estimate in challenging lunar terrain.

    The filter uses constant-velocity motion model with IMU-derived acceleration
    and angular velocity predictions, corrected by relative motion from encoders
    and absolute references from GPS and sun sensor.
    """

    @abstractmethod
    def initialize(
        self,
        config: EKFConfig,
        initial_state: EKFState,
    ) -> None:
        """
        Initialize EKF with configuration and initial state.

        Args:
            config: EKF configuration specifying rates and dimensions.
            initial_state: Initial estimate of rover state with covariance.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, imu_reading: IMUReading) -> EKFState:
        """
        Predict step using IMU measurement (200 Hz).

        Propagates state forward using constant-velocity model with IMU
        acceleration and gyro angular velocity. Updates time-evolving
        covariance to account for process noise.

        Args:
            imu_reading: IMU measurement containing acceleration and gyro rates.

        Returns:
            Predicted EKF state after IMU update.
        """
        raise NotImplementedError

    @abstractmethod
    def update_encoder(self, encoder_reading: EncoderReading) -> EKFState:
        """
        Measurement update from wheel encoders (50 Hz).

        Uses differential wheel odometry to provide relative displacement
        measurement. Helps constrain translation between GPS updates and
        corrects accumulated drift from IMU integration.

        Args:
            encoder_reading: Wheel distance measurements from left and right wheels.

        Returns:
            Corrected EKF state after encoder measurement.
        """
        raise NotImplementedError

    @abstractmethod
    def update_gps(self, gps_fix: GPSFix) -> EKFState:
        """
        Measurement update from GPS position fix (1 Hz).

        Incorporates absolute position measurement with dynamic covariance
        based on fix quality. Helps anchor global position and reset drift
        from local odometry.

        Args:
            gps_fix: GPS position and associated uncertainty.

        Returns:
            Corrected EKF state after GPS measurement.
        """
        raise NotImplementedError

    @abstractmethod
    def update_sun_sensor(self, sun_reading: SunReading) -> EKFState:
        """
        Measurement update from sun sensor for heading correction (1 Hz).

        Uses absolute heading measurement from sun position (lunar local time
        known). Provides drift-free yaw estimate to prevent accumulated
        gyro bias from rotating the rover's orientation estimate.

        Args:
            sun_reading: Sun-based heading measurement and confidence.

        Returns:
            Corrected EKF state after sun sensor measurement.
        """
        raise NotImplementedError

    @abstractmethod
    def update_scan_match(
        self,
        transform: npt.NDArray[np.float64],
    ) -> EKFState:
        """
        Measurement update from lidar scan matching (ICP) (5 Hz).

        Registers current lidar scan against map to estimate incremental pose
        change. Provides high-fidelity relative odometry for path-relative
        localization between sparse GPS updates.

        Args:
            transform: 4x4 homogeneous transformation matrix from scan matching.

        Returns:
            Corrected EKF state after scan match measurement.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> EKFState:
        """
        Retrieve current EKF state estimate.

        Returns:
            Current state with position, velocity, orientation, and covariance.
        """
        raise NotImplementedError

    @abstractmethod
    def get_position_accuracy(self) -> float:
        """
        Get 1-sigma position uncertainty from covariance.

        Computes the trace of the 3x3 position block of the covariance matrix.

        Returns:
            1-sigma position error in meters.
        """
        raise NotImplementedError

    @abstractmethod
    def check_mahalanobis_gate(
        self,
        innovation: npt.NDArray[np.float64],
        sensor_cov: npt.NDArray[np.float64],
    ) -> bool:
        """
        Outlier rejection using Mahalanobis distance gating.

        Checks if measurement innovation (residual) is consistent with predicted
        measurement uncertainty. Rejects measurements beyond chi-squared threshold
        to improve robustness to outliers and GPS multipath.

        Args:
            innovation: Measurement residual vector.
            sensor_cov: Sensor measurement covariance matrix.

        Returns:
            True if measurement passes gate (should be incorporated), False otherwise.
        """
        raise NotImplementedError
