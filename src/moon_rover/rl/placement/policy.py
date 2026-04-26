"""
System 19: Antenna Placement Policy

Reinforcement learning and scripted policies for antenna placement arm control.
Learns contact-rich manipulation with force feedback for robust surface placement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from moon_rover.rl.common.policy_interface import PolicyInterface


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


class AntennaPlacementPolicy(PolicyInterface):
    """
    Policy for antenna placement arm control.

    Implements the PolicyInterface for antenna placement task. Both scripted
    and RL implementations provide identical observe/act interface.

    RL policy learns contact-rich manipulation through interaction. Scripted
    policy uses impedance control and force feedback heuristics.

    See PolicyInterface for full method documentation.
    """

    pass
