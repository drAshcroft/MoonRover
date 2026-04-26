"""System 10: Moonbase Infrastructure.

This module defines the lunar base operations and logistics hub. Provides
charging, equipment storage (cable reels and antennas), docking facilities,
and localization beacon infrastructure. Central to rover mission planning.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class MoonbaseConfig:
    """Configuration parameters for lunar base facility.

    Attributes:
        habitat_dims_m: Habitat structure dimensions [length, width, height] in meters.
        solar_array_config: Solar power array configuration (from power.systems module).
        power_bus_voltage: Electrical bus voltage in Volts (typically 48 V).
        comm_tower_height_m: Communication tower height in meters.
        num_docking_bays: Number of rover docking stations (typically 2).
        charge_rate_w: Power delivery rate at charging dock in Watts.
        num_cable_reels: Total cable reels available in depot (typically 8).
        num_antennas: Total antenna units available in depot (typically 20).
        landing_pad_radius_m: Radius of landing pad/safe zone around base in meters.
    """

    habitat_dims_m: tuple[float, float, float]
    solar_array_config: Any  # SolarArrayConfig from power.systems
    power_bus_voltage: float
    comm_tower_height_m: float
    num_docking_bays: int
    charge_rate_w: float
    num_cable_reels: int
    num_antennas: int
    landing_pad_radius_m: float


@dataclass
class DepotInventory:
    """Tracking of equipment inventory in base depot.

    Attributes:
        cable_reels_available: Number of unused cable reels remaining.
        antennas_available: Number of undeployed antennas remaining.
        assigned_items: Dictionary mapping rover_id to list of assigned items.
                        E.g., {"rover_1": ["cable_reel_5", "antenna_7"]}.
    """

    cable_reels_available: int
    antennas_available: int
    assigned_items: dict[str, list[str]] = field(default_factory=dict)


class Moonbase(ABC):
    """Abstract base class for lunar base facility.

    Moonbase serves as the mission hub providing:
    - Equipment logistics (cable reels, antenna units)
    - Power charging infrastructure
    - Localization beacon (primary GNSS reference)
    - Docking and rover servicing
    - Mission planning and coordination center

    Rovers depart from and return to base, exchanging equipment and recharging.
    """

    @abstractmethod
    def initialize(self, config: MoonbaseConfig, engine: Any) -> None:
        """Initialize moonbase facility with configuration and physics engine.

        Args:
            config: MoonbaseConfig describing facility layout and capabilities.
            engine: Physics engine for collision detection, docking validation, etc.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def get_primary_beacon(self) -> Any:
        """Get beacon configuration for primary localization reference.

        Moonbase hosts a fixed, primary beacon (e.g., communication antenna)
        that rovers use for pseudo-GNSS positioning.

        Returns:
            BeaconConfig object (from sensors.gps_beacon.network module)
            describing beacon location and signal characteristics.
        """
        raise NotImplementedError

    @abstractmethod
    def request_cable_reel(self, rover_id: str) -> bool:
        """Assign an available cable reel to a rover for deployment mission.

        Each rover can be assigned one cable reel per mission. The reel is
        held in the rover's cargo until deployed in the field.

        Args:
            rover_id: Unique identifier string for the requesting rover.

        Returns:
            True if reel successfully assigned, False if none available or rover already has one.
        """
        raise NotImplementedError

    @abstractmethod
    def request_antenna(self, rover_id: str) -> bool:
        """Assign an available antenna unit to a rover for deployment.

        Rovers can be assigned multiple antennas per mission for distributed
        communication network expansion.

        Args:
            rover_id: Unique identifier string for the requesting rover.

        Returns:
            True if antenna successfully assigned, False if none available.
        """
        raise NotImplementedError

    @abstractmethod
    def get_inventory(self) -> DepotInventory:
        """Query current depot inventory and allocations.

        Returns:
            DepotInventory object with available equipment counts and rover assignments.
        """
        raise NotImplementedError

    @abstractmethod
    def dock_rover(self, rover_id: str) -> bool:
        """Attempt to dock a rover at a charging bay.

        Validates rover position and orientation against docking bay specifications
        (alignment tolerance ±0.05m, orientation ±5°).

        Args:
            rover_id: Unique identifier of rover requesting dock.

        Returns:
            True if docking successful (rover now charging), False if misaligned or bays full.
        """
        raise NotImplementedError

    @abstractmethod
    def undock_rover(self, rover_id: str) -> None:
        """Release a docked rover from charging bay.

        Args:
            rover_id: Unique identifier of rover to release.

        Raises:
            ValueError: If rover is not currently docked.
        """
        raise NotImplementedError

    @abstractmethod
    def get_charge_state(self, rover_id: str) -> float:
        """Query charging progress of a docked rover.

        Args:
            rover_id: Unique identifier of rover (must be docked).

        Returns:
            Charging progress as fraction [0.0, 1.0]. 0.0 = empty, 1.0 = full.
            Returns -1.0 if rover not docked.
        """
        raise NotImplementedError
