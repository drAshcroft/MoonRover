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


# Lunar surface gravity (m/s^2) used for cable-ground friction loads.
_MOON_G = 1.62


class RigidLinkCableSystem(CableSystem):
    """Analytic rigid-link tether: pre-allocated chain, spool, drag, electrical.

    The cable is a fixed list of :class:`CableLinkState` objects allocated once
    in :meth:`initialize` (no per-step allocation). Links start ``STORED`` on
    the spool and flip to grounded ``active`` as the rover advances roughly one
    ``link_length_m``; each grounded link is laid at the rover position recorded
    when it was activated, so the deployed cable traces the rover's path.

    Tension is integrated from the rover end back toward the spool: each
    grounded link adds its Coulomb ground friction (``mu * m * g_moon``) to the
    running tension, which is what the rover feels as drag. The model is fully
    deterministic and engine-free (``engine`` is accepted for API symmetry and
    optional terrain height lookup). One instance models one cable run, so
    multiple rovers are handled by multiple independent instances with no
    shared state.
    """

    def __init__(self) -> None:
        self._config: CableConfig | None = None
        self._engine: Any = None
        self._links: list[CableLinkState] = []
        self._num_links: int = 0
        self._n_active: int = 0
        self._spool: SpoolState | None = None
        self._feed_rate: float = 0.0
        self._anchor: NDArray = np.zeros(3)
        self._last_rover_pos: NDArray | None = None
        self._motion_dir: NDArray = np.zeros(3)
        self._load_current_a: float = 0.0

    # -- initialization ------------------------------------------------
    def initialize(self, config: CableConfig, engine: Any) -> None:
        if config.link_length_m <= 0.0:
            raise ValueError(f"link_length_m must be > 0, got {config.link_length_m}")
        if config.total_length_m <= 0.0:
            raise ValueError(
                f"total_length_m must be > 0, got {config.total_length_m}"
            )
        if config.max_tension_n <= 0.0:
            raise ValueError(f"max_tension_n must be > 0, got {config.max_tension_n}")
        if config.link_mass_kg <= 0.0:
            raise ValueError(f"link_mass_kg must be > 0, got {config.link_mass_kg}")
        self._config = config
        self._engine = engine
        self._num_links = max(
            1, int(np.ceil(config.total_length_m / config.link_length_m))
        )
        # Pre-allocate ALL links once, in STORED state (no dynamic allocation).
        self._links = [
            CableLinkState(
                position_xyz=np.zeros(3, dtype=np.float64),
                orientation_quat=np.array([1.0, 0.0, 0.0, 0.0]),
                active=False,
                contact_terrain=False,
                tension_n=0.0,
            )
            for _ in range(self._num_links)
        ]
        self._n_active = 0
        self._spool = SpoolState(
            remaining_length_m=config.total_length_m,
            angular_velocity=0.0,
            tension_n=0.0,
            brake_engaged=False,
        )
        self._feed_rate = 0.0
        self._last_rover_pos = None
        self._motion_dir = np.zeros(3)

    def _require(self) -> CableConfig:
        if self._config is None or self._spool is None:
            raise RuntimeError("initialize() must be called before use")
        return self._config

    def _terrain_z(self, x: float, y: float) -> float:
        eng = self._engine
        if eng is not None and hasattr(eng, "get_terrain_height"):
            try:
                return float(eng.get_terrain_height(x, y))
            except Exception:
                return 0.0
        return 0.0

    # -- per-step dynamics --------------------------------------------
    def step(self, dt: float) -> None:
        cfg = self._require()
        spool = self._spool
        assert spool is not None
        if not spool.brake_engaged and self._feed_rate != 0.0:
            paid = self._feed_rate * dt
            new_remaining = float(
                np.clip(spool.remaining_length_m - paid, 0.0, cfg.total_length_m)
            )
            spool.remaining_length_m = new_remaining
            # Spool angular velocity from linear feed (radius ~ link diameter).
            radius = max(cfg.link_diameter_m, 1e-3)
            spool.angular_velocity = self._feed_rate / radius
        else:
            spool.angular_velocity = 0.0

        # Integrate tension from the rover end back to the spool.
        per_link_friction = (
            cfg.terrain_friction * cfg.link_mass_kg * _MOON_G
        )
        running = 0.0
        for i in range(self._n_active - 1, -1, -1):
            running += per_link_friction
            self._links[i].tension_n = running
        spool.tension_n = running

    def activate_next_link(self, rover_position: NDArray) -> bool:
        cfg = self._require()
        rover_position = np.asarray(rover_position, dtype=np.float64).reshape(3)
        if self._last_rover_pos is not None:
            d = rover_position - self._last_rover_pos
            n = np.linalg.norm(d)
            if n > 1e-9:
                self._motion_dir = d / n
        self._last_rover_pos = rover_position.copy()

        if self._n_active >= self._num_links:
            return False
        spool = self._spool
        assert spool is not None
        if spool.remaining_length_m < cfg.link_length_m - 1e-9:
            return False

        link = self._links[self._n_active]
        lay = rover_position.copy()
        lay[2] = self._terrain_z(lay[0], lay[1])
        link.position_xyz = lay
        link.active = True
        link.contact_terrain = True
        link.tension_n = 0.0
        self._n_active += 1
        spool.remaining_length_m = max(
            0.0, spool.remaining_length_m - cfg.link_length_m
        )
        return True

    # -- queries -------------------------------------------------------
    def get_link_states(self) -> list[CableLinkState]:
        return [
            CableLinkState(
                position_xyz=l.position_xyz.copy(),
                orientation_quat=l.orientation_quat.copy(),
                active=l.active,
                contact_terrain=l.contact_terrain,
                tension_n=l.tension_n,
            )
            for l in self._links
        ]

    def get_total_drag_force(self) -> NDArray:
        cfg = self._require()
        n_ground = sum(1 for l in self._links if l.active and l.contact_terrain)
        mag = cfg.terrain_friction * cfg.link_mass_kg * _MOON_G * n_ground
        if mag <= 0.0 or np.linalg.norm(self._motion_dir) < 1e-9:
            return np.zeros(3, dtype=np.float64)
        # Drag opposes rover motion.
        return (-self._motion_dir * mag).astype(np.float64)

    def get_spool_state(self) -> SpoolState:
        self._require()
        spool = self._spool
        assert spool is not None
        return SpoolState(
            remaining_length_m=spool.remaining_length_m,
            angular_velocity=spool.angular_velocity,
            tension_n=spool.tension_n,
            brake_engaged=spool.brake_engaged,
        )

    def command_spool(self, feed_rate: float) -> None:
        self._require()
        assert self._spool is not None
        if self._spool.brake_engaged and feed_rate != 0.0:
            return  # brake holds the spool; ignore feed commands
        self._feed_rate = float(feed_rate)

    def engage_brake(self) -> None:
        self._require()
        assert self._spool is not None
        self._spool.brake_engaged = True
        self._spool.angular_velocity = 0.0
        self._feed_rate = 0.0

    def release_brake(self) -> None:
        """Release the spool brake so feed commands take effect again."""
        self._require()
        assert self._spool is not None
        self._spool.brake_engaged = False

    def check_tension_fault(self) -> bool:
        cfg = self._require()
        return any(l.tension_n > cfg.max_tension_n for l in self._links)

    def check_bend_radius_fault(self) -> list[int]:
        cfg = self._require()
        active = [i for i, l in enumerate(self._links) if l.active]
        bad: list[int] = []
        for k in range(1, len(active) - 1):
            a = self._links[active[k - 1]].position_xyz
            b = self._links[active[k]].position_xyz
            c = self._links[active[k + 1]].position_xyz
            v1 = b - a
            v2 = c - b
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            turn = np.arccos(cos_a)  # exterior angle at the joint
            if turn < 1e-6:
                continue
            # Radius of the circular arc that fits the turn over one link.
            bend_radius = cfg.link_length_m / (2.0 * np.sin(turn / 2.0))
            if bend_radius < cfg.bend_radius_min_m:
                bad.append(active[k])
        return bad

    def set_electrical_load(self, current_a: float) -> None:
        """Set the DC load current drawn through the cable (Amps)."""
        self._load_current_a = float(current_a)

    def get_electrical_state(self) -> dict[str, float]:
        cfg = self._require()
        deployed_len = self._n_active * cfg.link_length_m
        # Round-trip conductor length (out and return).
        resistance = cfg.resistance_per_m * deployed_len * 2.0
        current = self._load_current_a
        v_drop = current * resistance
        power = current * current * resistance
        return {
            "voltage_dc": float(cfg.voltage_dc),
            "current_a": float(current),
            "voltage_drop_v": float(v_drop),
            "power_w": float(power),
        }
