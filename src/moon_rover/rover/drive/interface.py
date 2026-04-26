"""System 4: Rover Drive Systems — Common interface for all drive configurations.

This module defines the abstract interface for rover drive systems, supporting
multiple kinematic configurations: 2-wheel differential (unicycle model),
3-wheel tricycle (bicycle/Ackermann steering), and 4-wheel skid-steer. All
drive types implement a common interface for unified control.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


class DriveType(Enum):
    """Enumeration of supported rover drive configurations."""

    TWO_WHEEL_DIFF = "two_wheel_differential"
    """Differential drive: independent control of left/right wheels (unicycle model)."""

    THREE_WHEEL_TRICYCLE = "three_wheel_tricycle"
    """Tricycle/Ackermann steering: front wheels steer, rear wheels drive."""

    FOUR_WHEEL_SKID = "four_wheel_skid"
    """Skid-steer drive: independent left/right track control."""


@dataclass
class WheelState:
    """Real-time state of a single wheel during motion.

    Attributes:
        angular_velocity: Wheel rotation speed in rad/s.
        torque: Torque applied to wheel axle in N·m.
        slip_ratio: Normalized slip ratio (wheel speed - ground speed) / max(wheel speed, ground speed).
                    0 = no slip, 1 = full slip (spinning in place).
        sinkage_depth: Wheel sinkage into regolith in meters (computed by wheel-terrain model).
        contact_force: Normal force at wheel-terrain contact in Newtons.
    """

    angular_velocity: float
    torque: float
    slip_ratio: float
    sinkage_depth: float
    contact_force: float


@dataclass
class DriveCommand:
    """Desired kinematic motion for the rover.

    Attributes:
        linear_velocity_mps: Forward velocity in m/s.
        angular_velocity_radps: Rotation rate in rad/s (positive = counter-clockwise when viewed from above).
    """

    linear_velocity_mps: float
    angular_velocity_radps: float


@dataclass
class DriveConfig:
    """Configuration parameters for rover drive system.

    Attributes:
        drive_type: Selected drive configuration (2-wheel, 3-wheel, or 4-wheel).
        track_width_m: Lateral distance between left/right wheels or tracks in meters.
        wheelbase_m: Longitudinal distance between front and rear axles in meters.
        wheel_radius_m: Effective radius of wheels in meters.
        max_torque_nm: Maximum torque available per wheel/motor in N·m.
        max_steer_angle_rad: Maximum steering angle in radians (only used for tricycle/Ackermann).
                             Optional; defaults to None for differential/skid-steer.
        num_wheels: Total number of driven wheels (2, 3, or 4).
    """

    drive_type: DriveType
    track_width_m: float
    wheelbase_m: float
    wheel_radius_m: float
    max_torque_nm: float
    max_steer_angle_rad: float | None = None
    num_wheels: int = 4


class DriveSystem(ABC):
    """Abstract base class for rover drive systems.

    Defines the unified interface for all drive configurations. Responsible for:
    - Kinematic forward/inverse solutions for each drive type
    - Converting high-level (v, ω) commands to per-wheel control signals
    - Tracking odometry and wheel states
    - Handling slip and terrain interaction effects

    All drive types (2-wheel unicycle, 3-wheel bicycle, 4-wheel skid-steer) inherit
    from this interface, allowing unified rover control regardless of physical configuration.
    """

    @abstractmethod
    def configure(self, config: DriveConfig) -> None:
        """Initialize drive system with configuration parameters.

        Args:
            config: Drive system configuration object containing geometry and limits.

        Raises:
            ValueError: If configuration parameters are invalid (e.g., wheel_radius_m <= 0).
        """
        raise NotImplementedError

    @abstractmethod
    def command(self, cmd: DriveCommand) -> None:
        """Issue a high-level kinematic command to the drive system.

        Converts the desired linear and angular velocity into per-wheel motor commands.
        The actual wheel speeds depend on the drive type:
        - 2-wheel: left_speed, right_speed
        - 3-wheel: steer_angle, front_speed, rear_speed (or equivalent)
        - 4-wheel: left_speed, right_speed (tracks may be coupled or independent)

        Args:
            cmd: Desired linear velocity (m/s) and angular velocity (rad/s).
        """
        raise NotImplementedError

    @abstractmethod
    def get_wheel_states(self) -> list[WheelState]:
        """Return real-time state of all wheels.

        Returns:
            List of WheelState objects, one per wheel, in consistent order
            (e.g., front-left, front-right, rear-left, rear-right for 4-wheel).
        """
        raise NotImplementedError

    @abstractmethod
    def get_odometry(self) -> tuple[NDArray, NDArray]:
        """Get integrated odometry from start of simulation.

        Returns:
            Tuple of:
            - position_xyz: 3D position as (x, y, z) in meters relative to start.
            - orientation_quat: Unit quaternion (w, x, y, z) representing rotation from start orientation.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_kinematics(self, wheel_speeds: list[float]) -> DriveCommand:
        """Compute rover velocity from per-wheel speeds (forward kinematics).

        Given the current wheel angular velocities (from encoders or motor feedback),
        determine the resulting rover velocity and rotation rate.

        Args:
            wheel_speeds: List of wheel angular velocities in rad/s.
                          Length must equal num_wheels.

        Returns:
            Computed DriveCommand (linear and angular velocity).
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_kinematics(self, cmd: DriveCommand) -> list[float]:
        """Compute per-wheel speeds from desired rover motion (inverse kinematics).

        Given a desired linear and angular velocity, compute the required wheel
        speeds to achieve that motion.

        Args:
            cmd: Desired DriveCommand (linear and angular velocity).

        Returns:
            List of required wheel angular velocities in rad/s, one per wheel.
        """
        raise NotImplementedError

    @abstractmethod
    def get_drive_type(self) -> DriveType:
        """Return the drive configuration type.

        Returns:
            DriveType enum indicating 2-wheel, 3-wheel, or 4-wheel configuration.
        """
        raise NotImplementedError
