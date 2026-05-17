"""System 7.6: Sun Sensor.

This module defines a simple sun-tracking sensor for solar array orientation
and power estimation. Provides azimuth of sun direction with accuracy suitable
for solar panel attitude optimization.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SunSensorConfig:
    """Configuration for sun sensor.

    Attributes:
        accuracy_deg: Azimuth measurement accuracy in degrees (1-sigma noise).
        update_rate_hz: Sensor sampling frequency in Hz (typically 1 Hz).
        elevation_threshold_deg: Minimum sun elevation for a valid reading.
                                 At or below this the sun is treated as set.
        seed: RNG seed for reproducible measurement noise.
    """

    accuracy_deg: float
    update_rate_hz: float
    elevation_threshold_deg: float = 0.0
    seed: int = 0


@dataclass
class SunReading:
    """Sun direction measurement snapshot.

    Attributes:
        azimuth_deg: Sun azimuth angle in degrees (0 = north, 90 = east, 180 = south, 270 = west).
        valid: True if sun is above horizon and measurement is reliable. False if in shadow or below horizon.
        timestamp: Measurement timestamp in seconds.
    """

    azimuth_deg: float
    valid: bool
    timestamp: float


class SunSensor(ABC):
    """Abstract base class for sun-tracking sensor.

    Measures sun direction (azimuth) above lunar horizon. Used for:
    - Solar array orientation optimization
    - Power generation forecasting
    - Time-of-day determination
    - Shadow detection

    Returns invalid readings when rover is in shadow.
    """

    @abstractmethod
    def configure(self, config: SunSensorConfig) -> None:
        """Initialize sun sensor with parameters.

        Args:
            config: Sun sensor configuration object.
        """
        raise NotImplementedError

    @abstractmethod
    def read(
        self,
        sun_azimuth_true: float,
        sun_elevation: float,
        in_shadow: bool,
    ) -> SunReading:
        """Generate sun direction measurement.

        Measures sun azimuth angle with noise. Returns invalid reading if rover
        is in shadow or sun is below horizon.

        Args:
            sun_azimuth_true: True sun azimuth in degrees (0 = north, 90 = east, etc.).
            sun_elevation: Sun elevation angle above horizon in degrees (positive = above, negative = below).
            in_shadow: Boolean indicating if rover is in shadow (occluded from sun).

        Returns:
            SunReading with azimuth measurement and validity flag.
        """
        raise NotImplementedError


class GenesisSunSensor(SunSensor):
    """Coarse sun sensor: noisy azimuth, gated by shadow and horizon.

    A reading is valid only when the rover is not in shadow and the sun is
    above ``elevation_threshold_deg``. When valid, the true azimuth is
    perturbed by zero-mean Gaussian noise with standard deviation
    ``accuracy_deg`` and wrapped into ``[0, 360)``. Invalid readings echo the
    last good azimuth (or 0.0) with ``valid=False`` so consumers can hold.
    """

    def __init__(self) -> None:
        self._config: SunSensorConfig | None = None
        self._rng: np.random.Generator | None = None
        self._t: float = 0.0
        self._dt: float = 0.0
        self._last_azimuth: float = 0.0

    def configure(self, config: SunSensorConfig) -> None:
        if config.accuracy_deg < 0.0:
            raise ValueError(
                f"accuracy_deg must be >= 0, got {config.accuracy_deg}"
            )
        if config.update_rate_hz <= 0.0:
            raise ValueError(
                f"update_rate_hz must be > 0, got {config.update_rate_hz}"
            )
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._t = 0.0
        self._dt = 1.0 / config.update_rate_hz
        self._last_azimuth = 0.0

    def read(
        self,
        sun_azimuth_true: float,
        sun_elevation: float,
        in_shadow: bool,
    ) -> SunReading:
        if self._config is None or self._rng is None:
            raise RuntimeError("configure() must be called before read()")
        cfg = self._config
        self._t += self._dt

        if in_shadow or sun_elevation <= cfg.elevation_threshold_deg:
            return SunReading(
                azimuth_deg=self._last_azimuth,
                valid=False,
                timestamp=self._t,
            )

        noisy = float(sun_azimuth_true)
        if cfg.accuracy_deg > 0.0:
            noisy += float(self._rng.normal(0.0, cfg.accuracy_deg))
        noisy %= 360.0
        self._last_azimuth = noisy
        return SunReading(
            azimuth_deg=noisy,
            valid=True,
            timestamp=self._t,
        )
