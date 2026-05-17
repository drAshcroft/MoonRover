"""System 9: Antenna System.

This module defines the deployable antenna unit that extends rover
communication range via a high-gain dish. Tracks deployment state,
mechanical configuration, and activation status.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class AntennaState(Enum):
    """State machine for antenna deployment lifecycle.

    States represent discrete phases of antenna deployment and use:
    - STORED: Compact, folded configuration; mounted on rover chassis.
    - GRIPPED: Held by arm gripper during placement procedure.
    - CARRIED: In transit on rover (e.g., arm holding it while traversing).
    - PLACED: Positioned on ground with base plate; not yet fully deployed.
    - DEPLOYED: Fully extended with mast vertical. Base must contact terrain.
    - ACTIVE: Commissioned and transmitting/receiving. Awaiting signal from moonbase.
    - FAILED: Mechanical failure or deployment error detected.
    """

    STORED = "stored"
    GRIPPED = "gripped"
    CARRIED = "carried"
    PLACED = "placed"
    DEPLOYED = "deployed"
    ACTIVE = "active"
    FAILED = "failed"


@dataclass
class AntennaConfig:
    """Physical and dimensional parameters of antenna unit.

    Attributes:
        base_plate_m: Base plate dimensions [length, width, height] in meters.
        base_mass_kg: Mass of base plate assembly in kilograms.
        mast_height_m: Height of vertical mast when fully extended in meters.
        mast_radius_m: Mast outer radius in meters.
        mast_mass_kg: Mass of mast structure in kilograms.
        dish_diameter_m: High-gain parabolic dish diameter in meters.
        dish_mass_kg: Mass of dish reflector in kilograms.
        connector_mass_kg: Mass of RF connector/cable interface in kilograms.
        total_mass_kg: Total assembled antenna mass in kilograms
                       (sum: base + mast + dish + connector).
    """

    base_plate_m: tuple[float, float, float]
    base_mass_kg: float
    mast_height_m: float
    mast_radius_m: float
    mast_mass_kg: float
    dish_diameter_m: float
    dish_mass_kg: float
    connector_mass_kg: float
    total_mass_kg: float


@dataclass
class DeploymentQuality:
    """Assessment of antenna deployment success and mechanical condition.

    Attributes:
        tilt_deg: Base plate tilt angle from vertical in degrees.
                  0 = perfectly level, >8 = degraded (acceptable up to ~5°).
        base_contact_corners: Number of base plate corners in firm ground contact (0-4).
                              4 = fully supported, <2 = unstable.
        position_error_m: Lateral position error from optimal deployment site in meters.
        connector_engaged: True if RF connector is properly mated and latched.
        status: Deployment status string:
                - "full": All criteria met, full functionality.
                - "degraded": One or more non-critical criteria unmet (tilt, contact).
                - "failed": Deployment unsuitable for operation (not engaged, excessive tilt, etc.).
    """

    tilt_deg: float
    base_contact_corners: int
    position_error_m: float
    connector_engaged: bool
    status: str


class AntennaUnit(ABC):
    """Abstract base class for deployable antenna subsystem.

    Models a mast-mounted high-gain dish antenna that extends rover
    communication range. Manages deployment state transitions, mechanical
    positioning, and deployment quality assessment.
    """

    @abstractmethod
    def get_state(self) -> AntennaState:
        """Return current antenna state in deployment lifecycle.

        Returns:
            Current AntennaState (STORED, GRIPPED, CARRIED, PLACED, DEPLOYED, ACTIVE, or FAILED).
        """
        raise NotImplementedError

    @abstractmethod
    def transition(self, new_state: AntennaState) -> bool:
        """Attempt transition to a new antenna state.

        Validates requested state transition according to state machine rules.
        Valid transitions (examples):
        - STORED -> GRIPPED (arm grasps antenna)
        - GRIPPED -> PLACED (arm places base plate on ground)
        - PLACED -> DEPLOYED (mast extends)
        - DEPLOYED -> ACTIVE (RF connector engaged, activation complete)

        Args:
            new_state: Target AntennaState.

        Returns:
            True if transition is valid and successful, False if invalid or preconditions unmet.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_deployment(self) -> DeploymentQuality:
        """Assess quality and readiness of antenna deployment.

        Checks mechanical constraints:
        - Base plate tilt < 8° (vertical)
        - Base plate corners in contact (minimum 2-3 corners)
        - Position within acceptable range of target site
        - RF connector properly engaged

        Returns:
            DeploymentQuality assessment with status flags.
        """
        raise NotImplementedError

    @abstractmethod
    def get_beacon_config(self) -> Any | None:
        """Get beacon configuration if antenna is active.

        When antenna reaches ACTIVE state and is properly positioned,
        it acts as a beacon transmitter for rover pseudo-GNSS localization.

        Returns:
            BeaconConfig object (from gps_beacon.network) if ACTIVE and deployed.
            None if antenna is not in ACTIVE state or deployment is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def get_physical_properties(self) -> dict[str, Any]:
        """Return antenna physical properties and current configuration.

        Returns:
            Dictionary with:
            - "mass_kg": Total antenna mass.
            - "dimensions_m": List of relevant dimensions (base, mast, dish).
            - "center_of_mass": 3D position of center of mass.
            - "footprint_area_m2": Base plate contact area with ground.
            - "mast_length_m": Current mast extension length (increases during DEPLOYED state).
        """
        raise NotImplementedError


# Allowed state-machine edges. FAILED is reachable from any non-terminal state
# (handled separately) to model mechanical faults at any point.
_ANTENNA_TRANSITIONS: dict[AntennaState, set[AntennaState]] = {
    AntennaState.STORED: {AntennaState.GRIPPED},
    AntennaState.GRIPPED: {AntennaState.CARRIED, AntennaState.STORED},
    AntennaState.CARRIED: {AntennaState.PLACED, AntennaState.GRIPPED},
    AntennaState.PLACED: {AntennaState.DEPLOYED, AntennaState.GRIPPED},
    AntennaState.DEPLOYED: {AntennaState.ACTIVE},
    AntennaState.ACTIVE: set(),
    AntennaState.FAILED: set(),
}


class DeployableAntennaUnit(AntennaUnit):
    """Mast-mounted dish antenna with a 7-state deployment machine.

    Lifecycle: ``STORED → GRIPPED → CARRIED → PLACED → DEPLOYED → ACTIVE``,
    with ``FAILED`` reachable from any non-terminal state. Transitions are
    guarded: placing requires a known ground pose (:meth:`set_placement`),
    ``PLACED → DEPLOYED`` raises the mast to full height, and
    ``DEPLOYED → ACTIVE`` requires a non-failed deployment assessment with the
    RF connector engaged. Once ``ACTIVE`` the unit exposes a
    :class:`~moon_rover.sensors.gps_beacon.network.BeaconConfig` so it can join
    the pseudo-GNSS network.

    The ``engine`` argument is optional and only used for a terrain-height
    lookup at the placement site; the model is otherwise deterministic.
    """

    def __init__(self, config: AntennaConfig, engine: Any = None) -> None:
        self._config = config
        self._engine = engine
        self._state = AntennaState.STORED
        self._placement_xy: NDArray | None = None
        self._base_z: float = 0.0
        self._tilt_deg: float = 0.0
        self._base_contact_corners: int = 0
        self._position_error_m: float = 0.0
        self._connector_engaged: bool = False
        self._mast_length_m: float = 0.0
        self._beacon_range_m: float = 800.0
        self._beacon_power_w: float = 20.0
        self._beacon_noise_sigma_m: float = 0.5

    # -- external sensing hooks ---------------------------------------
    def set_placement(
        self,
        position_xy: NDArray,
        tilt_deg: float,
        base_contact_corners: int,
        position_error_m: float,
        connector_engaged: bool = False,
    ) -> None:
        """Record the sensed ground placement used for quality assessment."""
        self._placement_xy = np.asarray(position_xy, dtype=np.float64).reshape(2)
        eng = self._engine
        if eng is not None and hasattr(eng, "get_terrain_height"):
            try:
                self._base_z = float(
                    eng.get_terrain_height(
                        self._placement_xy[0], self._placement_xy[1]
                    )
                )
            except Exception:
                self._base_z = 0.0
        self._tilt_deg = float(tilt_deg)
        self._base_contact_corners = int(np.clip(base_contact_corners, 0, 4))
        self._position_error_m = float(position_error_m)
        self._connector_engaged = bool(connector_engaged)

    def set_connector_engaged(self, engaged: bool) -> None:
        self._connector_engaged = bool(engaged)

    def fail(self) -> None:
        """Force the unit into the terminal FAILED state."""
        self._state = AntennaState.FAILED

    # -- state machine -------------------------------------------------
    def get_state(self) -> AntennaState:
        return self._state

    def transition(self, new_state: AntennaState) -> bool:
        if new_state == self._state:
            return True
        if self._state in (AntennaState.ACTIVE, AntennaState.FAILED):
            # Terminal-ish: only ACTIVE -> FAILED is allowed.
            if self._state == AntennaState.ACTIVE and new_state == AntennaState.FAILED:
                self._state = AntennaState.FAILED
                return True
            return False
        if new_state == AntennaState.FAILED:
            self._state = AntennaState.FAILED
            return True
        if new_state not in _ANTENNA_TRANSITIONS[self._state]:
            return False

        # Precondition gates.
        if new_state == AntennaState.PLACED and self._placement_xy is None:
            return False
        if new_state == AntennaState.ACTIVE:
            q = self.evaluate_deployment()
            if q.status == "failed" or not self._connector_engaged:
                return False

        self._state = new_state
        if new_state == AntennaState.DEPLOYED:
            self._mast_length_m = self._config.mast_height_m
        elif new_state in (AntennaState.STORED, AntennaState.GRIPPED):
            self._mast_length_m = 0.0
        return True

    # -- assessment ----------------------------------------------------
    def evaluate_deployment(self) -> DeploymentQuality:
        tilt = self._tilt_deg
        corners = self._base_contact_corners
        engaged = self._connector_engaged
        if tilt > 8.0 or corners < 2:
            status = "failed"
        elif tilt <= 5.0 and corners >= 4 and engaged:
            status = "full"
        else:
            status = "degraded"
        return DeploymentQuality(
            tilt_deg=tilt,
            base_contact_corners=corners,
            position_error_m=self._position_error_m,
            connector_engaged=engaged,
            status=status,
        )

    def get_beacon_config(self) -> Any | None:
        if self._state != AntennaState.ACTIVE or self._placement_xy is None:
            return None
        if self.evaluate_deployment().status == "failed":
            return None
        from moon_rover.sensors.gps_beacon.network import BeaconConfig

        pos = np.array(
            [
                self._placement_xy[0],
                self._placement_xy[1],
                self._base_z + self._mast_length_m,
            ],
            dtype=np.float64,
        )
        return BeaconConfig(
            position_xyz=pos,
            signal_range_m=self._beacon_range_m,
            power_w=self._beacon_power_w,
            noise_sigma_m=self._beacon_noise_sigma_m,
        )

    def get_physical_properties(self) -> dict[str, Any]:
        cfg = self._config
        if self._placement_xy is not None:
            base = np.array(
                [self._placement_xy[0], self._placement_xy[1], self._base_z]
            )
        else:
            base = np.zeros(3)
        com = base + np.array([0.0, 0.0, max(self._mast_length_m, 0.0) * 0.5])
        footprint = cfg.base_plate_m[0] * cfg.base_plate_m[1]
        return {
            "mass_kg": cfg.total_mass_kg,
            "dimensions_m": [
                list(cfg.base_plate_m),
                cfg.mast_height_m,
                cfg.dish_diameter_m,
            ],
            "center_of_mass": com.astype(np.float64),
            "footprint_area_m2": float(footprint),
            "mast_length_m": float(self._mast_length_m),
        }
