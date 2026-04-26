"""System 7.4: GPS Beacon Network — Pseudo-GNSS Localization.

This module implements a local pseudo-GNSS system using stationary beacon
transmitters (e.g., moonbase antenna or active beacons). Provides position
fixes via trilateration with realistic geometric dilution of precision (GDOP).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class BeaconConfig:
    """Configuration for a single stationary GPS beacon.

    Attributes:
        position_xyz: 3D position [x, y, z] of beacon transmitter in world frame, meters.
        signal_range_m: Maximum line-of-sight signal range in meters.
        power_w: Transmitter power in Watts (affects SNR and ranging accuracy).
        noise_sigma_m: Gaussian noise standard deviation on range measurements in meters.
    """

    position_xyz: NDArray
    signal_range_m: float
    power_w: float
    noise_sigma_m: float


@dataclass
class GPSFix:
    """Position fix result from beacon trilateration.

    Attributes:
        position_xyz: Estimated rover position [x, y, z] in meters.
        gdop: Geometric dilution of precision (1.0-10.0+). Lower is better.
              1.0-5.0 = excellent to good, >10.0 = poor geometry.
        num_beacons: Number of beacons used in solution (typically >= 4 for 3D).
        covariance: 3x3 position covariance matrix (diagonal) representing position uncertainty.
    """

    position_xyz: NDArray
    gdop: float
    num_beacons: int
    covariance: NDArray


class BeaconNetwork(ABC):
    """Abstract base class for beacon-based positioning network.

    Manages a network of stationary beacon transmitters that provide
    pseudo-GNSS fixes via trilateration. Simulates geometric dilution
    of precision and ranging accuracy degradation with distance/obstruction.
    """

    @abstractmethod
    def add_beacon(self, beacon_id: str, config: BeaconConfig) -> None:
        """Register a new beacon transmitter in the network.

        Args:
            beacon_id: Unique identifier string for the beacon (e.g., "moonbase_primary", "antenna_1").
            config: BeaconConfig describing beacon location and signal parameters.

        Raises:
            ValueError: If beacon_id already exists.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_beacon(self, beacon_id: str) -> None:
        """Deactivate and remove a beacon from the network.

        Args:
            beacon_id: Unique identifier of beacon to remove.

        Raises:
            KeyError: If beacon_id not found.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_fix(self, rover_position: NDArray) -> GPSFix | None:
        """Compute position fix using beacons in range.

        Performs trilateration using all beacons within signal range
        of the rover. Returns None if fewer than 1 beacon is visible.

        Args:
            rover_position: Current rover position [x, y, z] in meters.
                            Used to determine which beacons are in range.

        Returns:
            GPSFix object if solution is valid, None if insufficient beacons.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gdop_at(self, position: NDArray) -> float:
        """Query geometric dilution of precision at a given location.

        GDOP measures the contribution of beacon geometry to position error.
        It multiplies the ranging noise to determine overall position uncertainty:
        position_error ≈ ranging_noise * GDOP.

        Args:
            position: Query position [x, y, z] in meters.

        Returns:
            GDOP value (1.0 = ideal geometry, >10.0 = poor geometry).
            Returns inf if fewer than 4 beacons visible at this location.
        """
        raise NotImplementedError

    @abstractmethod
    def get_coverage_map(self, grid_resolution: float) -> NDArray:
        """Generate 2D GDOP map over terrain surface.

        Samples GDOP at regular grid points (at constant altitude/height)
        to visualize positioning quality across traversal area.

        Args:
            grid_resolution: Grid spacing in meters.

        Returns:
            2D array of GDOP values at each grid point. NaN indicates no solution.
        """
        raise NotImplementedError

    @abstractmethod
    def get_visible_beacons(self, position: NDArray) -> list[str]:
        """Return list of beacon IDs visible from a given position.

        Args:
            position: Query position [x, y, z] in meters.

        Returns:
            List of beacon_id strings that are in signal range at this location.
        """
        raise NotImplementedError
