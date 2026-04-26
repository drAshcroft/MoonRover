"""
System 22: Manipulator Arm Policy — Pickup + Carry-Stable

Reinforcement learning and scripted policies for antenna pickup and
stable carrying. Two-phase control: aggressive contact-rich pickup followed
by conservative stability maintenance during transit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from moon_rover.rl.common.policy_interface import PolicyInterface


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


class PickupPolicy(PolicyInterface):
    """
    Policy for antenna pickup manipulation.

    Implements the PolicyInterface for antenna pickup task. Both scripted
    and RL implementations provide identical observe/act interface.

    RL policy learns contact-rich manipulation with visual servoing to
    locate and grasp antenna. Scripted policy uses reactive reaching and
    force-feedback grasping.

    High-dimensional observation space (1039+ dims) including 32x32 depth
    image requires CNN feature extraction or visual transformer architecture.

    See PolicyInterface for full method documentation.
    """

    pass


class CarryStablePolicy(PolicyInterface):
    """
    Policy for antenna carry stability control.

    Implements the PolicyInterface for carry-stable task. Both scripted
    and RL implementations provide identical observe/act interface.

    RL policy learns compliant arm control that absorbs terrain disturbances
    and prevents antenna swinging. Scripted policy uses impedance control
    tuned for stability over speed.

    Low-dimensional observation space (22 dims) enables fast training and
    efficient inference for safety-critical carry phase.

    See PolicyInterface for full method documentation.
    """

    pass
