"""
System 22: Manipulator Arm Policy — Pickup + Carry-Stable

Reinforcement learning and scripted policies for antenna pickup and
stable carrying. Two-phase control: aggressive contact-rich pickup followed
by conservative stability maintenance during transit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from moon_rover.rl.common.policy_interface import BasePolicy, PolicyInterface, PolicyMode


def _field(obs: Any, name: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


@dataclass
class PickupObservation:
    """
    Observation state for antenna pickup task.

    Rich visual and proprioceptive feedback enabling contact-rich manipulation
    to locate, orient, grasp, and lift antenna from lunar surface.

    Attributes:
        joint_pos: 4-element arm joint positions [q1, q2, q3, q4] (rad).
        joint_vel: 4-element arm joint velocities [dq1, dq2, dq3, dq4] (rad/s).
        ft_6axis: 6-element force-torque sensor [fx, fy, fz, mx, my, mz]
            in gripper frame (N, N.m).
        gripper_jaw: 1-element gripper opening [0, 1] where 0 = fully closed,
            1 = fully open.
        wrist_depth_patch: 32x32 depth image from wrist-mounted camera (m).
            Enables visual servoing to antenna surface and gaps for grasping.
        antenna_relative_pose: 6-element pose relative to gripper frame
            [dx, dy, dz, roll, pitch, yaw] (m, rad). Ground truth for training,
            visual estimates for real deployment.

    Total dimension: 4 + 4 + 6 + 1 + (32*32) + 6 = 1039+ dimensions.
    """

    joint_pos: npt.NDArray[np.float32]  # shape (4,)
    joint_vel: npt.NDArray[np.float32]  # shape (4,)
    ft_6axis: npt.NDArray[np.float32]  # shape (6,)
    gripper_jaw: npt.NDArray[np.float32]  # shape (1,)
    wrist_depth_patch: npt.NDArray[np.float32]  # shape (32, 32)
    antenna_relative_pose: npt.NDArray[np.float32]  # shape (6,)


@dataclass
class PickupAction:
    """
    Control action for antenna pickup manipulation.

    Attributes:
        joint_velocity_targets: 4-element desired joint velocities
            [dq1_cmd, dq2_cmd, dq3_cmd, dq4_cmd] normalized to [-1, 1].
        gripper_command: 1-element gripper control [0, 1] or binary {0, 1}.
            0 = open gripper, 1 = close gripper (engage force control).
    """

    joint_velocity_targets: npt.NDArray[np.float32]  # shape (4,), [-1, 1]
    gripper_command: npt.NDArray[np.float32]  # shape (1,), [0, 1]


@dataclass
class CarryStableObservation:
    """
    Observation state for antenna carry-stable task.

    Minimal but informative feedback for maintaining stability while carrying
    antenna back to rover. Focuses on chassis motion and arm stability rather
    than visual details.

    Attributes:
        joint_pos: 4-element arm joint positions (rad).
        joint_vel: 4-element arm joint velocities (rad/s).
        ft_6axis: 6-element force-torque in gripper frame (N, N.m).
        chassis_accel: 3-element chassis acceleration [ax, ay, az] (m/s^2).
            From IMU, used to detect bumps and terrain disturbances.
        chassis_gyro: 3-element chassis angular velocity [gx, gy, gz] (rad/s).
            Detects roll/pitch changes indicating antenna shift.
        rover_speed: 1-element rover forward velocity (m/s).
        rover_yaw_rate: 1-element rover yaw/turning rate (rad/s).

    Total dimension: 4 + 4 + 6 + 3 + 3 + 1 + 1 = 22 dimensions.
    """

    joint_pos: npt.NDArray[np.float32]  # shape (4,)
    joint_vel: npt.NDArray[np.float32]  # shape (4,)
    ft_6axis: npt.NDArray[np.float32]  # shape (6,)
    chassis_accel: npt.NDArray[np.float32]  # shape (3,)
    chassis_gyro: npt.NDArray[np.float32]  # shape (3,)
    rover_speed: npt.NDArray[np.float32]  # shape (1,)
    rover_yaw_rate: npt.NDArray[np.float32]  # shape (1,)


@dataclass
class CarryStableAction:
    """
    Control action for antenna carry-stable task.

    Attributes:
        joint_velocity_targets: 4-element desired joint velocities.
            Tightly bounded to ±0.05 rad/s to maintain stability and prevent
            antenna swinging. Updated at 50 Hz with tight feedback control.
    """

    joint_velocity_targets: npt.NDArray[np.float32]  # shape (4,), [-0.05, 0.05]


class PickupPolicy(BasePolicy):
    """Scripted antenna-pickup controller (reactive reach + force grasp).

    Deterministic baseline implementing a two-stage reach-and-grasp:

    1. Reach: drive joint velocities to reduce the positional error in
       ``antenna_relative_pose`` (first three components, the gripper-frame
       offset to the antenna). A 4-DOF Jacobian-free heuristic maps the
       dominant Cartesian error onto base/shoulder/elbow joints.
    2. Grasp: once the antenna is within ``grasp_tolerance_m`` the gripper
       command latches closed (1.0); otherwise it stays open (0.0).

    :attr:`grasp_closed` records whether the gripper has latched.
    """

    def __init__(self, *, grasp_tolerance_m: float = 0.05, k_reach: float = 5.0) -> None:
        super().__init__(
            mode=PolicyMode.SCRIPTED,
            confidence=1.0,
            supported_modes={PolicyMode.SCRIPTED, PolicyMode.FALLBACK},
        )
        self.grasp_tolerance_m = float(grasp_tolerance_m)
        self.k_reach = float(k_reach)
        self.grasp_closed: bool = False

    def reset(self) -> None:
        super().reset()
        self.grasp_closed = False

    def _compute_action(self, observation: Any) -> dict[str, npt.NDArray]:
        pose = np.asarray(
            _field(observation, "antenna_relative_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dtype=np.float64,
        ).ravel()
        offset = pose[:3]
        dist = float(np.linalg.norm(offset))

        if dist <= self.grasp_tolerance_m:
            self.grasp_closed = True

        if self.grasp_closed:
            joint_targets = np.zeros(4, dtype=np.float32)
            gripper = 1.0
        else:
            # Reach: base yaws toward dx/dy, shoulder/elbow drive dz + reach.
            base = np.clip(self.k_reach * np.arctan2(offset[1], offset[0] + 1e-6), -1.0, 1.0)
            reach = np.clip(self.k_reach * float(np.hypot(offset[0], offset[1])), -1.0, 1.0)
            lift = np.clip(self.k_reach * offset[2], -1.0, 1.0)
            joint_targets = np.array([base, reach, -reach, lift], dtype=np.float32)
            gripper = 0.0

        return {
            "joint_velocity_targets": joint_targets,
            "gripper_command": np.array([gripper], dtype=np.float32),
        }


class CarryStablePolicy(BasePolicy):
    """Scripted carry-stability controller (impedance damping).

    Deterministic baseline that keeps the carried antenna steady during
    transit. It opposes residual joint motion and counteracts chassis
    disturbances sensed by the IMU gyro, with commands tightly bounded to
    ``max_joint_rate`` (default ±0.05 rad/s) to prevent swinging.
    """

    def __init__(self, *, max_joint_rate: float = 0.05, k_damp: float = 0.8, k_gyro: float = 0.3) -> None:
        super().__init__(
            mode=PolicyMode.SCRIPTED,
            confidence=1.0,
            supported_modes={PolicyMode.SCRIPTED, PolicyMode.FALLBACK},
        )
        self.max_joint_rate = float(max_joint_rate)
        self.k_damp = float(k_damp)
        self.k_gyro = float(k_gyro)

    def _compute_action(self, observation: Any) -> dict[str, npt.NDArray]:
        joint_vel = np.asarray(
            _field(observation, "joint_vel", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64
        ).ravel()[:4]
        if joint_vel.size < 4:
            joint_vel = np.pad(joint_vel, (0, 4 - joint_vel.size))
        gyro = np.asarray(
            _field(observation, "chassis_gyro", [0.0, 0.0, 0.0]), dtype=np.float64
        ).ravel()

        # Damp residual joint motion; add a gyro-driven counter-disturbance term.
        disturbance = float(np.linalg.norm(gyro)) if gyro.size else 0.0
        cmd = -self.k_damp * joint_vel - self.k_gyro * disturbance * np.sign(joint_vel)
        cmd = np.clip(cmd, -self.max_joint_rate, self.max_joint_rate)
        return {"joint_velocity_targets": cmd.astype(np.float32)}
