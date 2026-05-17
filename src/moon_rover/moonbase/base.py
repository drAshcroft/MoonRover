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


@dataclass
class _DockingBay:
    """Internal state of one charging/docking bay."""

    index: int
    position_xyz: NDArray
    heading_deg: float
    occupant: str | None = None


class LunarMoonbase(Moonbase):
    """Lunar base hub: depot logistics, docking, charging, primary beacon.

    Bays are laid out evenly along the +X face of the habitat at a fixed
    approach heading. Docking validates a rover's registered pose against a
    free bay within ±0.05 m position and ±5° heading tolerance; a docked rover
    charges over time via :meth:`step`, modelled as
    ``charge_rate_w · dt / battery_capacity_wh`` clamped to ``[0, 1]``. The
    primary beacon sits atop the comm tower and is always available for
    pseudo-GNSS. ``engine`` is optional and only used for terrain height.
    """

    DOCK_POS_TOL_M = 0.05
    DOCK_HEADING_TOL_DEG = 5.0
    DEFAULT_BATTERY_WH = 1000.0

    def __init__(self) -> None:
        self._config: MoonbaseConfig | None = None
        self._engine: Any = None
        self._origin: NDArray = np.zeros(3)
        self._bays: list[_DockingBay] = []
        self._reels_avail: int = 0
        self._antennas_avail: int = 0
        self._assigned: dict[str, list[str]] = {}
        self._reel_serial: int = 0
        self._antenna_serial: int = 0
        self._charge: dict[str, float] = {}
        self._battery_wh: dict[str, float] = {}
        self._rover_pose: dict[str, tuple[NDArray, float]] = {}

    # -- initialization ------------------------------------------------
    def initialize(self, config: MoonbaseConfig, engine: Any) -> None:
        if config.num_docking_bays < 1:
            raise ValueError(
                f"num_docking_bays must be >= 1, got {config.num_docking_bays}"
            )
        if config.charge_rate_w <= 0.0:
            raise ValueError(
                f"charge_rate_w must be > 0, got {config.charge_rate_w}"
            )
        if config.num_cable_reels < 0 or config.num_antennas < 0:
            raise ValueError("inventory counts must be >= 0")
        self._config = config
        self._engine = engine
        z0 = 0.0
        if engine is not None and hasattr(engine, "get_terrain_height"):
            try:
                z0 = float(engine.get_terrain_height(0.0, 0.0))
            except Exception:
                z0 = 0.0
        self._origin = np.array([0.0, 0.0, z0], dtype=np.float64)

        length = config.habitat_dims_m[0]
        width = config.habitat_dims_m[1]
        # Bays spaced along Y just off the +X face, all facing -X (approach +X->base).
        n = config.num_docking_bays
        ys = np.linspace(-width / 2.0, width / 2.0, n)
        face_x = length / 2.0 + 1.0
        self._bays = [
            _DockingBay(
                index=i,
                position_xyz=self._origin + np.array([face_x, float(ys[i]), 0.0]),
                heading_deg=180.0,
            )
            for i in range(n)
        ]
        self._reels_avail = config.num_cable_reels
        self._antennas_avail = config.num_antennas
        self._assigned = {}
        self._charge = {}
        self._battery_wh = {}
        self._rover_pose = {}

    def _require(self) -> MoonbaseConfig:
        if self._config is None:
            raise RuntimeError("initialize() must be called before use")
        return self._config

    # -- registration hooks -------------------------------------------
    def set_rover_pose(
        self, rover_id: str, position_xyz: NDArray, heading_deg: float
    ) -> None:
        """Register/update a rover's pose (used for docking validation)."""
        self._rover_pose[rover_id] = (
            np.asarray(position_xyz, dtype=np.float64).reshape(3),
            float(heading_deg),
        )

    def set_rover_battery_capacity(self, rover_id: str, capacity_wh: float) -> None:
        """Set a rover's battery capacity (Wh) for charge-rate modelling."""
        self._battery_wh[rover_id] = float(capacity_wh)

    def step(self, dt: float) -> None:
        """Advance charging for all docked rovers by ``dt`` seconds."""
        cfg = self._require()
        for bay in self._bays:
            rid = bay.occupant
            if rid is None:
                continue
            cap_wh = self._battery_wh.get(rid, self.DEFAULT_BATTERY_WH)
            delta = cfg.charge_rate_w * dt / (cap_wh * 3600.0)
            self._charge[rid] = float(np.clip(self._charge.get(rid, 0.0) + delta, 0.0, 1.0))

    # -- beacon --------------------------------------------------------
    def get_primary_beacon(self) -> Any:
        cfg = self._require()
        from moon_rover.sensors.gps_beacon.network import BeaconConfig

        top = self._origin + np.array([0.0, 0.0, cfg.comm_tower_height_m])
        return BeaconConfig(
            position_xyz=top.astype(np.float64),
            signal_range_m=max(2000.0, cfg.landing_pad_radius_m * 20.0),
            power_w=50.0,
            noise_sigma_m=0.25,
        )

    # -- depot ---------------------------------------------------------
    def request_cable_reel(self, rover_id: str) -> bool:
        self._require()
        items = self._assigned.setdefault(rover_id, [])
        if self._reels_avail <= 0:
            return False
        if any(it.startswith("cable_reel_") for it in items):
            return False  # one reel per rover per mission
        self._reel_serial += 1
        items.append(f"cable_reel_{self._reel_serial}")
        self._reels_avail -= 1
        return True

    def request_antenna(self, rover_id: str) -> bool:
        self._require()
        if self._antennas_avail <= 0:
            return False
        self._antenna_serial += 1
        self._assigned.setdefault(rover_id, []).append(
            f"antenna_{self._antenna_serial}"
        )
        self._antennas_avail -= 1
        return True

    def get_inventory(self) -> DepotInventory:
        self._require()
        return DepotInventory(
            cable_reels_available=self._reels_avail,
            antennas_available=self._antennas_avail,
            assigned_items={k: list(v) for k, v in self._assigned.items()},
        )

    # -- docking -------------------------------------------------------
    def _free_bay_for(self, rover_id: str) -> _DockingBay | None:
        pose = self._rover_pose.get(rover_id)
        if pose is None:
            return None
        pos, heading = pose
        for bay in self._bays:
            if bay.occupant is not None:
                continue
            if np.linalg.norm(pos - bay.position_xyz) > self.DOCK_POS_TOL_M:
                continue
            dh = abs((heading - bay.heading_deg + 180.0) % 360.0 - 180.0)
            if dh > self.DOCK_HEADING_TOL_DEG:
                continue
            return bay
        return None

    def dock_rover(self, rover_id: str) -> bool:
        self._require()
        if any(b.occupant == rover_id for b in self._bays):
            return True  # already docked
        bay = self._free_bay_for(rover_id)
        if bay is None:
            return False
        bay.occupant = rover_id
        self._charge.setdefault(rover_id, 0.0)
        return True

    def undock_rover(self, rover_id: str) -> None:
        self._require()
        for bay in self._bays:
            if bay.occupant == rover_id:
                bay.occupant = None
                return
        raise ValueError(f"rover '{rover_id}' is not currently docked")

    def get_charge_state(self, rover_id: str) -> float:
        self._require()
        if not any(b.occupant == rover_id for b in self._bays):
            return -1.0
        return float(self._charge.get(rover_id, 0.0))
