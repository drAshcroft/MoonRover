"""System 7.3: IMU and Wheel Odometry Sensors.

This module defines inertial measurement (accelerometer + gyroscope) and
wheel encoder interfaces. Includes realistic noise, bias, and drift models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class IMUConfig:
    """Configuration for inertial measurement unit (IMU).

    Attributes:
        update_rate_hz: IMU sampling frequency in Hz (typically 200 Hz for rovers).
        gyro_noise_sigma: Gaussian noise standard deviation on gyroscope in rad/s.
        accel_noise_sigma: Gaussian noise standard deviation on accelerometer in m/s².
        gyro_bias_drift_deg_hr: Gyroscope bias drift rate in degrees/hour.
                                Random walk model: bias changes slowly over time.
    """

    update_rate_hz: float
    gyro_noise_sigma: float
    accel_noise_sigma: float
    gyro_bias_drift_deg_hr: float


@dataclass
class EncoderConfig:
    """Configuration for wheel encoder.

    Attributes:
        counts_per_rev: Encoder resolution in counts per wheel revolution.
                        Higher = finer position/velocity resolution.
        update_rate_hz: Encoder sampling frequency in Hz.
    """

    counts_per_rev: int
    update_rate_hz: float


@dataclass
class IMUReading:
    """Single IMU measurement snapshot.

    Attributes:
        accel_xyz: Acceleration in [x, y, z] in m/s² (body frame).
        gyro_xyz: Angular velocity in [x, y, z] in rad/s (body frame).
        timestamp: Measurement timestamp in seconds.
    """

    accel_xyz: NDArray
    gyro_xyz: NDArray
    timestamp: float


@dataclass
class EncoderReading:
    """Wheel encoder state measurement.

    Attributes:
        counts: List of encoder counts for each wheel (one per wheel).
                Accumulated since system start.
        angular_velocities: Computed angular velocities for each wheel in rad/s.
        timestamp: Measurement timestamp in seconds.
    """

    counts: list[int]
    angular_velocities: list[float]
    timestamp: float


class IMUSensor(ABC):
    """Abstract base class for inertial measurement unit (IMU).

    Measures linear acceleration and angular velocity in body frame.
    Includes realistic noise and bias drift models.
    """

    @abstractmethod
    def configure(self, config: IMUConfig) -> None:
        """Initialize IMU with sensor parameters.

        Args:
            config: IMU configuration object.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, true_accel: NDArray, true_gyro: NDArray) -> IMUReading:
        """Generate IMU measurement with noise and bias.

        Applies sensor noise (Gaussian) and gyroscope bias (random walk) to
        true acceleration and angular velocity values.

        Args:
            true_accel: True acceleration [x, y, z] in m/s² (body frame).
            true_gyro: True angular velocity [x, y, z] in rad/s (body frame).

        Returns:
            IMUReading with noisy measurements and timestamp.
        """
        raise NotImplementedError

    @abstractmethod
    def get_bias_state(self) -> NDArray:
        """Return current estimated gyroscope bias vector.

        Bias drifts slowly (random walk) during operation. This returns the
        current accumulated bias offset, used for bias estimation filters.

        Returns:
            Gyroscope bias [x, y, z] in rad/s.
        """
        raise NotImplementedError


class WheelEncoder(ABC):
    """Abstract base class for wheel encoder sensor.

    Measures wheel rotation via incremental encoder. Provides both
    absolute counts and derived angular velocities.
    """

    @abstractmethod
    def configure(self, config: EncoderConfig) -> None:
        """Initialize encoder with sensor parameters.

        Args:
            config: Encoder configuration object.
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, true_angular_velocities: list[float]) -> EncoderReading:
        """Generate encoder measurement from wheel angular velocities.

        Integrates angular velocity to compute encoder counts and
        computes derived angular velocity from count changes.

        Args:
            true_angular_velocities: True angular velocity for each wheel in rad/s.

        Returns:
            EncoderReading with counts and angular velocities.
        """
        raise NotImplementedError
