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
        accuracy_deg: Azimuth measurement accuracy in degrees.
        update_rate_hz: Sensor sampling frequency in Hz (typically 1 Hz).
    """

    accuracy_deg: float
    update_rate_hz: float


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
