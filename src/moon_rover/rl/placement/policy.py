"""
System 19: Antenna Placement Policy

Reinforcement learning and scripted policies for antenna placement arm control.
Learns contact-rich manipulation with force feedback for robust surface placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from moon_rover.rl.common.policy_interface import BasePolicy, PolicyInterface, PolicyMode


def _field(obs: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dataclass observation or a plain dict."""
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float64).ravel()
    return float(arr[0]) if arr.size else default


@dataclass
class PlacementObservation:
    """
    Observation state for antenna placement task.

    Complete sensory input to placement policy including arm kinematics,
    force-torque feedback, surface properties, and task progress.

    Attributes:
        joint_positions: 4-element array [q1, q2, q3, q4] joint angles (rad).
        joint_velocities: 4-element array [dq1, dq2, dq3, dq4] (rad/s).
        ft_force: 3-element force vector [fx, fy, fz] in end-effector frame (N).
        ft_torque: 3-element torque vector [mx, my, mz] (N.m).
        surface_normal: 3-element surface normal vector in world frame.
        surface_compliance: 1-element compliance metric [0, 1] (soft vs hard).
        height_above_surface: 1-element height of gripper above surface (m).
        time_remaining: 1-element fraction of episode time left [0, 1].
        previous_action: 4-element previous joint velocity command (rad/s).

    Total dimension: 4 + 4 + 3 + 3 + 3 + 1 + 1 + 1 + 4 = 24 dimensions.
    """

    joint_positions: npt.NDArray[np.float32]  # shape (4,)
    joint_velocities: npt.NDArray[np.float32]  # shape (4,)
    ft_force: npt.NDArray[np.float32]  # shape (3,)
    ft_torque: npt.NDArray[np.float32]  # shape (3,)
    surface_normal: npt.NDArray[np.float32]  # shape (3,)
    surface_compliance: npt.NDArray[np.float32]  # shape (1,)
    height_above_surface: npt.NDArray[np.float32]  # shape (1,)
    time_remaining: npt.NDArray[np.float32]  # shape (1,)
    previous_action: npt.NDArray[np.float32]  # shape (4,)


@dataclass
class PlacementAction:
    """
    Control action for antenna placement arm.

    Represents desired joint velocities for the 4-DOF arm. Normalized to
    [-1, 1] range for network policies and rescaled by maximum velocity.

    Attributes:
        joint_velocity_targets: 4-element array [dq1_cmd, dq2_cmd, dq3_cmd, dq4_cmd]
            Normalized to [-1, 1]. Rescaled to ±0.15 rad/s max by controller.
            20 Hz control rate with exponential moving average filtering (α=0.4)
            to smooth commands and improve stability.
    """

    joint_velocity_targets: npt.NDArray[np.float32]  # shape (4,), values in [-1, 1]


class PlacementReward:
    """
    Reward function specification for antenna placement task.

    Composite 7-component reward from architecture document:

    1. Alignment reward: Bonus for aligning antenna base with surface normal
       (contact frame aligned with world frame). Enables correct placement angle.

    2. Contact reward: Bonus for achieving desired normal force [50-200 N range].
       Penalizes under-contact (missed surface) and over-contact (crash).

    3. Stability reward: Bonus when force/torque in contact frame is balanced.
       Reduces rocking motions and instability at contact.

    4. Height reward: Guides arm downward toward surface. Enables descent phase.

    5. Time efficiency: Rewards completing task in fewer steps. Drives fast
       execution without sacrificing quality.

    6. Control smoothness: Penalty on jerk (d3x/dt3) to reduce wear and improve
       energy efficiency.

    7. Task completion: Large bonus when antenna placement conditions met and
       arm can safely release (contacts stable, alignment correct, force nominal).

    See architecture documentation for exact weight matrix and thresholds.
    """

    pass


class AntennaPlacementPolicy(BasePolicy):
    """Scripted antenna-placement controller (impedance + force feedback).

    Deterministic baseline used as a fallback and an RL training reference.
    Behaviour:

    1. Descend while ``height_above_surface`` exceeds ``contact_height_m`` —
       command negative vertical velocity proportional to the height error.
    2. On contact, regulate the normal force to ``target_force_n``: keep
       lowering while under-force, retract while over-force (impedance law).
    3. Align the wrist with ``surface_normal`` so the base plate seats flat.

    The dominant vertical command is exposed via :attr:`last_vertical_rate`
    (negative = lowering) for introspection and testing. Actions are 4 joint
    velocity targets normalized to [-1, 1].
    """

    def __init__(
        self,
        *,
        target_force_n: float = 120.0,
        contact_height_m: float = 0.02,
        k_height: float = 4.0,
        k_force: float = 0.01,
    ) -> None:
        super().__init__(
            mode=PolicyMode.SCRIPTED,
            confidence=1.0,
            supported_modes={PolicyMode.SCRIPTED, PolicyMode.FALLBACK},
        )
        self.target_force_n = float(target_force_n)
        self.contact_height_m = float(contact_height_m)
        self.k_height = float(k_height)
        self.k_force = float(k_force)
        self.last_vertical_rate: float = 0.0

    def _compute_action(self, observation: Any) -> dict[str, npt.NDArray]:
        height = _scalar(_field(observation, "height_above_surface"), 0.0)
        fz = float(np.asarray(_field(observation, "ft_force", [0.0, 0.0, 0.0])).ravel()[2])
        normal = np.asarray(
            _field(observation, "surface_normal", [0.0, 0.0, 1.0]), dtype=np.float64
        ).ravel()

        if height > self.contact_height_m:
            # Above the surface: descend proportionally to the height error.
            vertical = -np.clip(self.k_height * height, 0.0, 1.0)
        else:
            # In contact: impedance law on the normal force. Under-force
            # (fz < target) keeps pushing down (negative); over-force retracts.
            force_err = self.target_force_n - fz
            vertical = float(np.clip(-self.k_force * force_err, -1.0, 1.0))
        self.last_vertical_rate = float(vertical)

        # Wrist alignment: roll the last joint toward the surface-normal tilt.
        tilt = float(np.arctan2(float(normal[0]), max(float(normal[2]), 1e-6)))
        wrist = float(np.clip(-tilt, -1.0, 1.0))

        # Map vertical command onto shoulder + elbow, alignment onto the wrist.
        targets = np.array([0.0, vertical, vertical, wrist], dtype=np.float32)
        return {"joint_velocity_targets": targets}
