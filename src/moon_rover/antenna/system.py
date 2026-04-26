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
