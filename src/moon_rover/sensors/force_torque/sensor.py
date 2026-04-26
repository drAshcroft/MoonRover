"""System 7.5: Force-Torque Sensor — Wrist-Mounted.

This module defines the 6-axis force-torque (F/T) sensor for gripper and
manipulation force feedback. Mounted at arm wrist, measures forces and
torques at the end-effector for grasp control and object interaction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class FTConfig:
    """Configuration for 6-axis force-torque sensor.

    Attributes:
        force_range_n: Maximum measurable force in Newtons (symmetric ±).
        torque_range_nm: Maximum measurable torque in N·m (symmetric ±).
        resolution_force: Force measurement resolution in Newtons (smallest detectable step).
        resolution_torque: Torque measurement resolution in N·m.
        update_rate_hz: Sensor sampling frequency in Hz (typically 1000 Hz for manipulation).
    """

    force_range_n: float
    torque_range_nm: float
    resolution_force: float
    resolution_torque: float
    update_rate_hz: float


@dataclass
class FTReading:
    """Single 6-axis force-torque measurement.

    Attributes:
        force_xyz: Force components [Fx, Fy, Fz] in Newtons (wrist frame).
        torque_xyz: Torque components [Tx, Ty, Tz] in N·m (wrist frame).
        timestamp: Measurement timestamp in seconds.
    """

    force_xyz: NDArray
    torque_xyz: NDArray
    timestamp: float


class ForceTorqueSensor(ABC):
    """Abstract base class for 6-axis force-torque sensor.

    Measures forces and torques at the arm end-effector. Used for:
    - Grasp stability assessment
    - Contact detection during manipulation
    - Compliance control (soft contact)
    - Overload detection
    """

    @abstractmethod
    def configure(self, config: FTConfig) -> None:
        """Initialize F/T sensor with specifications.

        Args:
            config: FT sensor configuration object.

        Raises:
            ValueError: If configuration parameters are invalid (e.g., force_range_n <= 0).
        """
        raise NotImplementedError

    @abstractmethod
    def read(self, joint_reaction_forces: NDArray) -> FTReading:
        """Generate F/T measurement from arm reaction forces.

        Derives 6-axis force-torque from rigid body solver forces at the
        end-effector contact. Applies sensor noise and quantization.

        Args:
            joint_reaction_forces: Reaction force vector at end-effector contact (typically
                                   computed from rigid body dynamics solver).

        Returns:
            FTReading with measured forces and torques.
        """
        raise NotImplementedError

    @abstractmethod
    def check_overload(self) -> bool:
        """Check if sensor is experiencing overload condition.

        Returns:
            True if any force component exceeds force_range_n or torque exceeds torque_range_nm.
            False if all measurements within safe limits.
        """
        raise NotImplementedError
