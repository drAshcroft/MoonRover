"""System 6: Manipulator Arm and End-Effector.

This module defines the arm kinematics, control interface, and end-effector
management. Supports arbitrary DOF arms with configurable joint limits,
payload constraints, and force-torque feedback for grasp control.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ArmConfig:
    """Configuration for robotic manipulator arm.

    Attributes:
        num_dof: Number of degrees of freedom (typically 4-6).
        joint_limits: List of tuples (min_angle_rad, max_angle_rad) for each joint.
                      Length must equal num_dof.
        reach_m: Maximum reach from base to end-effector in meters.
        payload_kg: Maximum payload the gripper can safely handle in kilograms.
        joint_accuracy_deg: Joint positioning repeatability in degrees.
    """

    num_dof: int
    joint_limits: list[tuple[float, float]]
    reach_m: float
    payload_kg: float
    joint_accuracy_deg: float


@dataclass
class GripperConfig:
    """Configuration for end-effector gripper.

    Attributes:
        num_fingers: Number of fingers/digits (typically 2-3).
        max_open_m: Maximum opening distance in meters.
        max_force_n: Maximum gripping force in Newtons (per finger).
        compliance_model: Compliance model for finger interaction (e.g., "linear", "exponential").
    """

    num_fingers: int
    max_open_m: float
    max_force_n: float
    compliance_model: str


@dataclass
class ArmState:
    """Real-time state of arm and end-effector during operation.

    Attributes:
        joint_positions: Array of joint angles in radians (length = num_dof).
        joint_velocities: Array of joint angular velocities in rad/s.
        joint_torques: Array of torques applied at each joint in N·m.
        gripper_position: Current gripper opening distance in meters (0 = fully closed, max_open_m = fully open).
        ft_reading: 6-axis force-torque sensor reading at wrist [Fx, Fy, Fz, Tx, Ty, Tz] in [N, N, N, N·m, N·m, N·m].
    """

    joint_positions: NDArray
    joint_velocities: NDArray
    joint_torques: NDArray
    gripper_position: float
    ft_reading: NDArray


@dataclass
class GraspQuality:
    """Assessment of gripper contact and grasp stability.

    Attributes:
        force_n: Total gripping force in Newtons (sum across all fingers).
        stable: True if grasp meets stability criteria (force > threshold, contact > min_points).
        contact_points: Number of contact points across all fingers.
    """

    force_n: float
    stable: bool
    contact_points: int


class ManipulatorArm(ABC):
    """Abstract base class for rover robotic arm.

    Manages arm kinematics (forward and inverse solutions), joint control,
    and end-effector interaction. Supports arbitrary DOF and payload constraints.
    """

    @abstractmethod
    def configure(self, arm_config: ArmConfig, gripper_config: GripperConfig) -> None:
        """Initialize arm with geometric and control parameters.

        Args:
            arm_config: Arm geometry, joint limits, and payload configuration.
            gripper_config: End-effector gripper configuration.

        Raises:
            ValueError: If configuration parameters are inconsistent (e.g., joint_limits length != num_dof).
        """
        raise NotImplementedError

    @abstractmethod
    def inverse_kinematics(self, target_pose: NDArray) -> NDArray:
        """Compute joint angles to reach a target end-effector pose.

        Solves the inverse kinematics problem: given a desired end-effector
        position and orientation, compute the required joint angles.

        Args:
            target_pose: End-effector target pose as 4x4 homogeneous transformation matrix
                         or [x, y, z, qw, qx, qy, qz] (position + quaternion).

        Returns:
            Joint angle solution in radians (length = num_dof).
            May be multiple solutions; returns one valid solution or closest to current pose.

        Raises:
            ValueError: If target pose is unreachable (outside workspace).
        """
        raise NotImplementedError

    @abstractmethod
    def forward_kinematics(self, joint_angles: NDArray) -> NDArray:
        """Compute end-effector pose from joint angles.

        Solves forward kinematics: given joint angles, compute the resulting
        end-effector position and orientation.

        Args:
            joint_angles: Array of joint angles in radians (length = num_dof).

        Returns:
            End-effector pose as 4x4 homogeneous transformation matrix
            or [x, y, z, qw, qx, qy, qz] (position + unit quaternion).
        """
        raise NotImplementedError

    @abstractmethod
    def command_joints(self, velocities: NDArray) -> None:
        """Command joint velocities for arm motion.

        Issues velocity commands to arm joints (typically velocity control loop).

        Args:
            velocities: Desired joint angular velocities in rad/s (length = num_dof).
        """
        raise NotImplementedError

    @abstractmethod
    def command_gripper(self, open_fraction: float) -> None:
        """Command gripper opening.

        Args:
            open_fraction: Gripper opening as fraction [0.0, 1.0]:
                           - 0.0: fully closed
                           - 1.0: fully open (max_open_m distance)
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> ArmState:
        """Return current arm and gripper state.

        Returns:
            ArmState object with joint positions, velocities, torques, and FT sensor reading.
        """
        raise NotImplementedError

    @abstractmethod
    def get_grasp_quality(self) -> GraspQuality:
        """Evaluate current gripper grasp quality.

        Analyzes force-torque sensor data and contact points to assess
        if current grasp is stable and adequate.

        Returns:
            GraspQuality object indicating grasp force and stability.
        """
        raise NotImplementedError

    @abstractmethod
    def is_in_stow_position(self) -> bool:
        """Check if arm is in stowed (safe) position.

        The stow position is typically folded against rover body to minimize
        center of mass shift and reduce tip-over risk during traversal.

        Returns:
            True if arm is at or very close to stow position, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def stow(self) -> None:
        """Move arm to stowed position.

        Issues motion commands to move arm into stowed configuration.
        Blocks until stow motion completes.
        """
        raise NotImplementedError
