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
    seed: int = 0


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


class GenesisForceTorqueSensor(ForceTorqueSensor):
    """6-axis F/T sensor: calibration offset, white noise, quantization, clip.

    :meth:`read` accepts the end-effector reaction wrench as a 6-vector
    ``[Fx, Fy, Fz, Tx, Ty, Tz]`` (a 3-vector is treated as force-only). The
    pipeline is: subtract the stored calibration (tare) offset, add zero-mean
    Gaussian noise of one resolution step (1 LSB) per axis, quantize to the
    sensor resolution, then saturate to the configured range. Overload is
    latched from the *unsaturated* magnitude so a clipped reading still flags.
    """

    def __init__(self) -> None:
        self._config: FTConfig | None = None
        self._rng: np.random.Generator | None = None
        self._offset: NDArray = np.zeros(6)
        self._overload: bool = False
        self._t: float = 0.0
        self._dt: float = 0.0

    def configure(self, config: FTConfig) -> None:
        if config.force_range_n <= 0.0:
            raise ValueError(
                f"force_range_n must be > 0, got {config.force_range_n}"
            )
        if config.torque_range_nm <= 0.0:
            raise ValueError(
                f"torque_range_nm must be > 0, got {config.torque_range_nm}"
            )
        if config.resolution_force <= 0.0 or config.resolution_torque <= 0.0:
            raise ValueError("resolution_force/torque must be > 0")
        if config.update_rate_hz <= 0.0:
            raise ValueError(
                f"update_rate_hz must be > 0, got {config.update_rate_hz}"
            )
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._offset = np.zeros(6)
        self._overload = False
        self._t = 0.0
        self._dt = 1.0 / config.update_rate_hz

    def set_calibration_offset(self, offset: NDArray) -> None:
        """Set the 6-vector bias subtracted from every subsequent reading."""
        self._offset = np.asarray(offset, dtype=np.float64).reshape(6).copy()

    def tare(self, current_wrench: NDArray) -> None:
        """Zero the sensor against the present load (store it as the offset)."""
        self._offset = np.asarray(current_wrench, dtype=np.float64).reshape(6).copy()

    def read(self, joint_reaction_forces: NDArray) -> FTReading:
        if self._config is None or self._rng is None:
            raise RuntimeError("configure() must be called before read()")
        cfg = self._config
        rng = self._rng
        w = np.asarray(joint_reaction_forces, dtype=np.float64).reshape(-1)
        if w.shape[0] == 3:
            w = np.concatenate([w, np.zeros(3)])
        elif w.shape[0] != 6:
            raise ValueError(
                f"joint_reaction_forces must have 3 or 6 elements, got {w.shape[0]}"
            )

        w = w - self._offset
        res = np.array([cfg.resolution_force] * 3 + [cfg.resolution_torque] * 3)
        w = w + rng.normal(0.0, res)  # ~1 LSB white noise per axis

        # Overload latched from the pre-saturation signal.
        force_mag = np.linalg.norm(w[:3])
        torque_mag = np.linalg.norm(w[3:])
        self._overload = bool(
            force_mag > cfg.force_range_n or torque_mag > cfg.torque_range_nm
        )

        quant = np.round(w / res) * res
        quant[:3] = np.clip(quant[:3], -cfg.force_range_n, cfg.force_range_n)
        quant[3:] = np.clip(quant[3:], -cfg.torque_range_nm, cfg.torque_range_nm)

        self._t += self._dt
        return FTReading(
            force_xyz=quant[:3],
            torque_xyz=quant[3:],
            timestamp=self._t,
        )

    def check_overload(self) -> bool:
        return self._overload
