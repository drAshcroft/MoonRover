"""System 2.3: Lighting and Shadow System for Lunar Environment.

This module provides interfaces for modeling solar illumination and shadows
on the lunar surface, accounting for surface orientation and lunar cycles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class SolarConfig:
    """Configuration for solar system and lighting.

    Parameters:
        elevation_deg: Sun elevation angle in degrees (5-75 typical).
            0 = horizon, 90 = zenith.
        azimuth_deg: Sun azimuth angle in degrees (0-360).
            0 = North, 90 = East, 180 = South, 270 = West.
        lunar_day_cycle: Whether to simulate full lunar day/night cycle.
        time_scale: Time acceleration factor (1.0 = real-time, 100.0 = 100x faster).
    """
    elevation_deg: float
    azimuth_deg: float = 0.0
    lunar_day_cycle: bool = False
    time_scale: float = 1.0


@dataclass
class AlbedoMap:
    """Surface reflectance properties for different lunar terrain types.

    Parameters:
        mare_basalt: Albedo of dark mare basalt regions (~0.07 on Moon).
        highland_regolith: Albedo of bright highland regolith (~0.12 on Moon).
        data: Full albedo map as 2D array. Values 0-1 representing reflectance.
    """
    mare_basalt: float = 0.07
    highland_regolith: float = 0.12
    data: Optional[NDArray[np.float32]] = None


class SolarSystem(ABC):
    """Abstract interface for solar illumination and shadow computation.

    Manages sun position, illuminance calculation, and shadow generation
    for realistic lunar lighting.
    """

    @abstractmethod
    def configure(self, config: SolarConfig) -> None:
        """Configure solar system with sun position and cycle parameters.

        Parameters:
            config: SolarConfig with elevation, azimuth, cycle parameters.

        Raises:
            ValueError: If elevation is outside 0-90 degrees or azimuth outside 0-360.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, sim_time: float) -> None:
        """Update sun position based on simulation time.

        If lunar_day_cycle is enabled, the sun position orbits based on sim_time.
        A lunar day is 29.5 Earth days (~2.55M seconds).

        Parameters:
            sim_time: Elapsed simulation time in seconds.

        Raises:
            ValueError: If sim_time is negative.
        """
        raise NotImplementedError

    @abstractmethod
    def get_illuminance_at(self, position: NDArray[np.float32]) -> float:
        """Get illuminance (lux) at a world position.

        Illuminance depends on:
        - Sun elevation and azimuth
        - Surface orientation (normal) at position
        - Surface albedo
        - Shadows from terrain

        Parameters:
            position: 3D world position [x, y, z] in meters.

        Returns:
            Illuminance in lux. Range 0 to ~160,000 (lunar surface peak).
            0 = in shadow.

        Raises:
            ValueError: If position is outside terrain bounds.
        """
        raise NotImplementedError

    @abstractmethod
    def get_shadow_mask(self, camera_pose: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Generate binary shadow mask for a camera view.

        Computes which pixels are in shadow when viewed from camera_pose.
        Useful for rendering and sensor simulation.

        Parameters:
            camera_pose: Camera position and orientation [x, y, z, qx, qy, qz, qw].

        Returns:
            Binary mask (0=shadow, 1=illuminated). Shape matches camera resolution.

        Raises:
            RuntimeError: If camera parameters are not configured.
        """
        raise NotImplementedError
