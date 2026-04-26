"""System 4.4: Wheel-Terrain Interaction Model.

This module defines the physics models for wheel-regolith interactions,
including tire slip, traction generation, sinkage, and cable drag effects.
Integrates Pacejka Magic Formula for slip-based traction, Bekker-Wong model
for sinkage and soil deformation, and specialized models for cable entanglement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class WheelTerrainConfig:
    """Configuration for wheel-terrain interaction physics.

    Attributes:
        contact_model: Contact physics model to use. Options: "hertz_coulomb" (Hertzian with Coulomb friction).
        friction_anisotropic: If True, model directional friction variation in regolith.
        bekker_params: Dictionary of Bekker-Wong soil model parameters:
            - "cohesion_pa": Soil cohesion in Pa.
            - "friction_angle_deg": Soil internal friction angle in degrees.
            - "sinkage_exponent": Exponent in sinkage law (typically 0.8-1.2).
            - "k_phi": Friction modulus in kPa (relates normal stress to friction).
            - "k_c": Cohesion modulus (relates cohesion contribution).
        pacejka_params: Dictionary of Pacejka Magic Formula tire parameters:
            - "B": Stiffness factor (higher = sharper transition).
            - "C": Shape factor (typically 1.3-2.0).
            - "D": Peak friction factor (fraction of normal force).
            - "E": Curvature factor (exponent).
    """

    contact_model: str
    friction_anisotropic: bool
    bekker_params: dict[str, float]
    pacejka_params: dict[str, float]


class WheelTerrainModel(ABC):
    """Abstract base class for wheel-terrain interaction physics.

    Models the forces generated when a wheel rolls/slips on lunar regolith,
    including:
    - Slip-based traction (Pacejka Magic Formula)
    - Normal force and sinkage (Bekker-Wong model)
    - Cable drag when cables are present
    - Rut/groove formation from previous passes
    """

    @abstractmethod
    def compute_slip_ratio(self, wheel_angular_vel: float, ground_velocity: float) -> float:
        """Calculate normalized slip ratio for a rolling wheel.

        Slip ratio quantifies the mismatch between wheel surface speed and
        ground speed. Used as input to traction models.

        Args:
            wheel_angular_vel: Wheel rotation speed in rad/s.
            ground_velocity: Magnitude of wheel center velocity in m/s.

        Returns:
            Slip ratio in range [0, 1]:
            - 0.0: pure rolling (no slip)
            - 1.0: full spin-out (no forward motion, wheel spins in place)
            Computed as: (wheel_speed - ground_speed) / max(wheel_speed, ground_speed)
        """
        raise NotImplementedError

    @abstractmethod
    def compute_traction_force(self, slip_ratio: float, normal_force: float) -> float:
        """Generate longitudinal traction force using Pacejka Magic Formula.

        The Magic Formula models tire slip as a smooth curve from zero slip
        (pure rolling) to full slip (spinning in place). Peak traction occurs
        at intermediate slip ratios (typically 5-15% for rovers).

        Args:
            slip_ratio: Normalized slip ratio in range [0, 1].
            normal_force: Normal force at wheel-ground contact in Newtons.

        Returns:
            Longitudinal traction force in Newtons (positive = forward).
            Peak value typically occurs at slip_ratio ≈ 0.1.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_sinkage(self, wheel_load: float, soil_params: dict[str, float]) -> float:
        """Compute wheel sinkage into lunar regolith using Bekker-Wong model.

        Wheel sinkage (penetration depth) reduces effective wheel radius and
        increases contact patch area, affecting both rolling resistance and
        traction. The Bekker-Wong model empirically fits sinkage to normal load.

        Args:
            wheel_load: Normal force on wheel in Newtons.
            soil_params: Soil parameter dictionary containing:
                - "cohesion_pa": Cohesive strength in Pa.
                - "sinkage_exponent": Exponent in power law (typically 0.8-1.2).
                - Other Bekker coefficients as configured.

        Returns:
            Wheel sinkage depth in meters (typically 0.001-0.05 m for lunar regolith).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_rut_state(self, position: NDArray) -> float:
        """Query rut or groove depth from terrain state at a given position.

        If the terrain uses Material Point Method (MPM) or similar deformation
        tracking, this returns the current rut depth caused by previous rover passes.
        Used to compute rolling resistance and traction degradation.

        Args:
            position: 3D position (x, y, z) in world coordinates.

        Returns:
            Rut depth in meters (0.0 = no rut, positive = depression).
        """
        raise NotImplementedError

    @abstractmethod
    def compute_cable_drag_effect(self, cable_tension: float, normal_force: float) -> float:
        """Compute traction reduction due to cable drag and entanglement.

        When a cable is laid behind the rover, it exerts friction and can wrap
        around wheels, reducing available traction. This effect scales with
        cable tension and wheel load.

        Args:
            cable_tension: Tension in cable at wheel location in Newtons.
            normal_force: Normal force at wheel contact in Newtons.

        Returns:
            Traction reduction factor in range [0.0, 1.0]:
            - 1.0: no cable effect, full traction available
            - 0.0: cable completely prevents wheel rotation
            Computed as a function of cable tension and normal load.
        """
        raise NotImplementedError
