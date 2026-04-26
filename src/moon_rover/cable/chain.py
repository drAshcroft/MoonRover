"""System 8: Cable System — Rigid-Link Chain Model.

This module implements the tethered cable system as a rigid-link chain.
Each link represents a discrete segment of cable that transitions from stored
(on spool) to actively grounded (on terrain) as the rover advances.
Tracks tension, friction losses, and electrical characteristics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CableConfig:
    """Configuration for tethered cable system.

    Attributes:
        link_length_m: Length of each rigid link segment in meters.
        link_diameter_m: Cross-section diameter of cable in meters.
        link_mass_kg: Mass of one link segment in kilograms.
        total_length_m: Total cable length (number of links = total_length / link_length).
        joint_damping: Damping coefficient at each link joint (simulates friction).
        joint_stiffness: Stiffness coefficient at each link joint.
        terrain_friction: Coulomb friction coefficient between cable and terrain.
        max_tension_n: Maximum safe tension before cable failure in Newtons.
        bend_radius_min_m: Minimum bend radius at joint before mechanical failure.
        voltage_dc: DC power bus voltage for electrical current in Volts.
        resistance_per_m: Electrical resistance per meter of cable in Ohms/m.
    """

    link_length_m: float
    link_diameter_m: float
    link_mass_kg: float
    total_length_m: float
    joint_damping: float
    joint_stiffness: float
    terrain_friction: float
    max_tension_n: float
    bend_radius_min_m: float
    voltage_dc: float
    resistance_per_m: float


@dataclass
class CableLinkState:
    """State of a single rigid link segment in the cable chain.

    Attributes:
        position_xyz: 3D position of link center [x, y, z] in meters.
        orientation_quat: Quaternion (w, x, y, z) orientation of link.
        active: True if link is grounded on terrain, False if still stored on spool.
        contact_terrain: True if link is currently touching terrain surface.
        tension_n: Tension force at this link in Newtons.
    """

    position_xyz: NDArray
    orientation_quat: NDArray
    active: bool
    contact_terrain: bool
    tension_n: float


@dataclass
class SpoolState:
    """State of cable spool mechanism.

    Attributes:
        remaining_length_m: Cable still stored on spool in meters.
        angular_velocity: Spool rotation speed in rad/s (positive = paying out).
        tension_n: Tension in cable at spool end in Newtons.
        brake_engaged: True if mechanical brake is holding spool stationary.
    """

    remaining_length_m: float
    angular_velocity: float
    tension_n: float
    brake_engaged: bool


class CableSystem(ABC):
    """Abstract base class for tethered cable deployment system.

    Models cable as a chain of rigid links. As rover moves, links sequentially
    transition from stored (on spool) to active (on ground) state. Tracks tension,
    friction losses, and electrical distribution along cable length.

    Key constraint: All cable links must be pre-allocated at initialization (no dynamic allocation).
    """

    @abstractmethod
    def initialize(self, config: CableConfig, engine: Any) -> None:
        """Initialize cable system and pre-allocate all link objects.

        CRITICAL: This method must allocate ALL cable links at construction time.
        Do NOT dynamically create new link objects during simulation. All links are
        created in the STORED state and transition to ACTIVE as rover advances.

        Args:
            config: Cable configuration parameters.
            engine: Physics engine (for forces, constraints, etc.).

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float) -> None:
        """Update cable system state for one simulation step.

        Updates link positions, tensions, friction losses, and transitions
        links from stored to active state as needed.

        Args:
            dt: Simulation time step in seconds.
        """
        raise NotImplementedError

    @abstractmethod
    def activate_next_link(self, rover_position: NDArray) -> bool:
        """Transition the next cable link from stored (spool) to active (grounded).

        Called when rover has advanced sufficiently (approximately one link_length)
        past the last active link.

        Args:
            rover_position: Current rover position [x, y, z] in meters.

        Returns:
            True if transition successful, False if no more links available.
        """
        raise NotImplementedError

    @abstractmethod
    def get_link_states(self) -> list[CableLinkState]:
        """Return state of all cable links (stored and active).

        Returns:
            List of CableLinkState objects, one per link in order from spool to rover.
        """
        raise NotImplementedError

    @abstractmethod
    def get_total_drag_force(self) -> NDArray:
        """Compute total drag force from all grounded cable links.

        Friction forces on grounded links sum to create a net drag force
        opposing rover motion.

        Returns:
            3D drag force vector [Fx, Fy, Fz] in Newtons.
        """
        raise NotImplementedError

    @abstractmethod
    def get_spool_state(self) -> SpoolState:
        """Return current spool mechanism state.

        Returns:
            SpoolState with remaining cable length, motor state, and tension.
        """
        raise NotImplementedError

    @abstractmethod
    def command_spool(self, feed_rate: float) -> None:
        """Command spool feed rate (pay-out or reel-in).

        Args:
            feed_rate: Desired feed rate in m/s.
                       Positive = pay out (unreel), negative = reel in.
        """
        raise NotImplementedError

    @abstractmethod
    def engage_brake(self) -> None:
        """Mechanically lock the spool to prevent cable motion.

        Used to halt cable payout and hold rover in place via cable tension.
        """
        raise NotImplementedError

    @abstractmethod
    def check_tension_fault(self) -> bool:
        """Check if cable tension exceeds safe limit.

        Returns:
            True if any link has tension > max_tension_n, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def check_bend_radius_fault(self) -> list[int]:
        """Identify links where bend radius exceeds mechanical limit.

        Returns:
            List of link indices (0-based) where bend_radius < bend_radius_min_m.
            Empty list if no violations.
        """
        raise NotImplementedError

    @abstractmethod
    def get_electrical_state(self) -> dict[str, float]:
        """Query electrical characteristics of cable and current state.

        Returns:
            Dictionary with:
            - "voltage_dc": Bus voltage in Volts.
            - "current_a": Current flowing through cable in Amps.
            - "voltage_drop_v": Cumulative voltage drop along cable length.
            - "power_w": Power dissipated as heat in cable resistance.
        """
        raise NotImplementedError
