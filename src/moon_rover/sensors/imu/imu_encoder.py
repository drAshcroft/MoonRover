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
    seed: int = 0


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
    seed: int = 0


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


class GenesisIMUSensor(IMUSensor):
    """Strapdown IMU model: white noise + gyro bias random walk.

    Each :meth:`read` advances one sample period ``dt = 1/update_rate_hz``.
    The gyroscope bias follows a discrete random walk whose increment standard
    deviation is derived from the catalog ``gyro_bias_drift_deg_hr`` figure:
    ``sigma_step = bias_rate_rad_s * sqrt(dt)``. Accelerometer and gyroscope
    each get zero-mean Gaussian white noise. A single seeded generator keeps
    runs reproducible for replay.
    """

    def __init__(self) -> None:
        self._config: IMUConfig | None = None
        self._rng: np.random.Generator | None = None
        self._bias: NDArray = np.zeros(3)
        self._t: float = 0.0
        self._dt: float = 0.0
        self._bias_step_sigma: float = 0.0

    def configure(self, config: IMUConfig) -> None:
        if config.update_rate_hz <= 0.0:
            raise ValueError(
                f"update_rate_hz must be > 0, got {config.update_rate_hz}"
            )
        if config.gyro_noise_sigma < 0.0 or config.accel_noise_sigma < 0.0:
            raise ValueError("noise sigmas must be >= 0")
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._bias = np.zeros(3)
        self._t = 0.0
        self._dt = 1.0 / config.update_rate_hz
        # deg/hr -> rad/s, then random-walk increment over one sample.
        bias_rate_rad_s = np.radians(config.gyro_bias_drift_deg_hr) / 3600.0
        self._bias_step_sigma = bias_rate_rad_s * np.sqrt(self._dt)

    def read(self, true_accel: NDArray, true_gyro: NDArray) -> IMUReading:
        if self._config is None or self._rng is None:
            raise RuntimeError("configure() must be called before read()")
        rng = self._rng
        cfg = self._config
        true_accel = np.asarray(true_accel, dtype=np.float64).reshape(3)
        true_gyro = np.asarray(true_gyro, dtype=np.float64).reshape(3)

        if self._bias_step_sigma > 0.0:
            self._bias = self._bias + rng.normal(0.0, self._bias_step_sigma, 3)

        accel = true_accel + rng.normal(0.0, cfg.accel_noise_sigma, 3)
        gyro = (
            true_gyro
            + self._bias
            + rng.normal(0.0, cfg.gyro_noise_sigma, 3)
        )
        self._t += self._dt
        return IMUReading(
            accel_xyz=accel,
            gyro_xyz=gyro,
            timestamp=self._t,
        )

    def get_bias_state(self) -> NDArray:
        return self._bias.copy()


class GenesisWheelEncoder(WheelEncoder):
    """Incremental wheel encoder with tick quantization.

    Integrates each wheel's true angular velocity over one sample period into
    a fractional revolution accumulator, then reports the truncated integer
    count at ``counts_per_rev`` resolution. The derived angular velocity comes
    from the *quantized* count delta divided by ``dt``, so it carries the
    realistic stair-step quantization error of a real encoder rather than the
    clean input rate.
    """

    def __init__(self) -> None:
        self._config: EncoderConfig | None = None
        self._dt: float = 0.0
        self._counts: list[int] = []
        self._accum: NDArray | None = None  # fractional counts per wheel
        self._t: float = 0.0

    def configure(self, config: EncoderConfig) -> None:
        if config.counts_per_rev < 1:
            raise ValueError(
                f"counts_per_rev must be >= 1, got {config.counts_per_rev}"
            )
        if config.update_rate_hz <= 0.0:
            raise ValueError(
                f"update_rate_hz must be > 0, got {config.update_rate_hz}"
            )
        self._config = config
        self._dt = 1.0 / config.update_rate_hz
        self._counts = []
        self._accum = None
        self._t = 0.0

    def read(self, true_angular_velocities: list[float]) -> EncoderReading:
        if self._config is None:
            raise RuntimeError("configure() must be called before read()")
        cfg = self._config
        omega = np.asarray(true_angular_velocities, dtype=np.float64).reshape(-1)
        n = omega.shape[0]
        if self._accum is None:
            self._accum = np.zeros(n)
            self._counts = [0] * n
        elif self._accum.shape[0] != n:
            raise ValueError(
                f"wheel count changed: configured for {self._accum.shape[0]}, "
                f"got {n}"
            )

        # rev/s -> counts/s: omega/(2pi) * counts_per_rev.
        delta_counts = omega / (2.0 * np.pi) * cfg.counts_per_rev * self._dt
        self._accum = self._accum + delta_counts
        new_counts = np.trunc(self._accum).astype(np.int64)
        tick_delta = new_counts - np.asarray(self._counts, dtype=np.int64)
        self._counts = new_counts.tolist()

        # Angular velocity reconstructed from quantized ticks.
        ang_vel = (
            tick_delta / cfg.counts_per_rev * 2.0 * np.pi / self._dt
        ).tolist()
        self._t += self._dt
        return EncoderReading(
            counts=[int(c) for c in self._counts],
            angular_velocities=[float(v) for v in ang_vel],
            timestamp=self._t,
        )
