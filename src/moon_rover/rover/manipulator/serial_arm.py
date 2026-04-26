"""Concrete serial-chain ManipulatorArm with FK, damped-least-squares IK, and gripper.

Implements :class:`ManipulatorArm` for an arbitrary number of revolute joints
(rotating about a configured axis) plus a parallel-jaw gripper. The kinematics
are general: each joint is parameterised by its rotation axis and the link
translation that follows it, so the class covers both the default 4-DOF arm in
``configs/rover.yaml`` and any future 5/6-DOF reconfiguration.

IK uses damped least squares (DLS) on the 3D position Jacobian, which is stable
near singularities and works for the typical rover-arm reach (2 m). Orientation
is not enforced during IK — the rover placement task cares primarily about
end-effector position plus an approximate tool-Z axis; tightening to full 6-DOF
pose IK can be added later via a CasADi/Pinocchio backend without touching the
call sites.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from moon_rover.rover.manipulator.arm import (
    ArmConfig,
    ArmState,
    GraspQuality,
    GripperConfig,
    ManipulatorArm,
)


# ---------------------------------------------------------------------------
# Kinematic chain definition
# ---------------------------------------------------------------------------


@dataclass
class _JointSpec:
    """One revolute joint in the serial chain."""

    axis: NDArray  # unit vector in the parent frame
    link_length: float  # translation along local +X after the joint
    lower: float
    upper: float


class SerialArm(ManipulatorArm):
    """Revolute serial-chain arm with DLS inverse kinematics.

    Parameters:
        base_offset: Translation from rover base frame to joint-1 origin.
        joint_axes: List of 3-vectors giving each joint's rotation axis (parent
            frame). Defaults to alternating +Z, +Y, +Y, ... consistent with
            the default 4-DOF arm.
        default_stow_angles: Joint angles used by :meth:`stow`. Defaults to
            zeros (straight arm).
    """

    def __init__(
        self,
        base_offset: Sequence[float] = (0.0, 0.0, 0.0),
        joint_axes: Optional[Sequence[Sequence[float]]] = None,
        default_stow_angles: Optional[Sequence[float]] = None,
    ) -> None:
        self._base_offset = np.asarray(base_offset, dtype=np.float64).flatten()
        if self._base_offset.shape != (3,):
            raise ValueError("base_offset must be length-3")
        self._requested_axes = joint_axes
        self._default_stow_angles = default_stow_angles

        self._arm_config: Optional[ArmConfig] = None
        self._gripper_config: Optional[GripperConfig] = None
        self._joints: List[_JointSpec] = []
        self._joint_pos: NDArray = np.zeros(0, dtype=np.float64)
        self._joint_vel: NDArray = np.zeros(0, dtype=np.float64)
        self._joint_torque: NDArray = np.zeros(0, dtype=np.float64)
        self._gripper_open_m: float = 0.0
        self._gripper_target_fraction: float = 1.0
        self._ft_reading: NDArray = np.zeros(6, dtype=np.float64)
        self._stow_angles: NDArray = np.zeros(0, dtype=np.float64)
        self._link_lengths: NDArray = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # ABC: configure
    # ------------------------------------------------------------------

    def configure(self, arm_config: ArmConfig, gripper_config: GripperConfig) -> None:
        if arm_config.num_dof <= 0:
            raise ValueError(f"num_dof must be positive, got {arm_config.num_dof}")
        if len(arm_config.joint_limits) != arm_config.num_dof:
            raise ValueError(
                f"joint_limits length ({len(arm_config.joint_limits)}) must equal "
                f"num_dof ({arm_config.num_dof})"
            )
        if arm_config.reach_m <= 0.0:
            raise ValueError(f"reach_m must be positive, got {arm_config.reach_m}")
        if gripper_config.num_fingers <= 0:
            raise ValueError("gripper must have at least one finger")

        axes = self._resolve_axes(arm_config.num_dof)
        link_lengths = self._resolve_link_lengths(arm_config.num_dof, arm_config.reach_m)

        self._joints = []
        for i, (axis, link, (lo, hi)) in enumerate(
            zip(axes, link_lengths, arm_config.joint_limits)
        ):
            ax = np.asarray(axis, dtype=np.float64).flatten()
            norm = float(np.linalg.norm(ax))
            if norm < 1e-9:
                raise ValueError(f"Joint {i} axis must be a non-zero vector")
            self._joints.append(
                _JointSpec(
                    axis=ax / norm,
                    link_length=float(link),
                    lower=float(lo),
                    upper=float(hi),
                )
            )

        n = arm_config.num_dof
        self._arm_config = arm_config
        self._gripper_config = gripper_config
        self._joint_pos = np.zeros(n, dtype=np.float64)
        self._joint_vel = np.zeros(n, dtype=np.float64)
        self._joint_torque = np.zeros(n, dtype=np.float64)
        self._gripper_open_m = gripper_config.max_open_m
        self._gripper_target_fraction = 1.0
        self._ft_reading = np.zeros(6, dtype=np.float64)
        self._link_lengths = np.asarray(link_lengths, dtype=np.float64)

        if self._default_stow_angles is not None:
            stow = np.asarray(self._default_stow_angles, dtype=np.float64).flatten()
            if stow.size != n:
                raise ValueError(
                    f"default_stow_angles length ({stow.size}) must equal num_dof ({n})"
                )
        else:
            stow = np.zeros(n, dtype=np.float64)
        self._stow_angles = self._clamp_limits(stow)

    # ------------------------------------------------------------------
    # ABC: kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(self, joint_angles: NDArray) -> NDArray:
        """End-effector pose as a 4x4 homogeneous transform in rover-base frame."""
        angles = np.asarray(joint_angles, dtype=np.float64).flatten()
        self._require_configured(angles.size)

        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = self._base_offset
        for spec, theta in zip(self._joints, angles):
            T = T @ _rotation_matrix(spec.axis, float(theta))
            T = T @ _translation_matrix(np.array([spec.link_length, 0.0, 0.0]))
        return T

    def inverse_kinematics(self, target_pose: NDArray) -> NDArray:
        """DLS inverse kinematics for position (orientation ignored).

        Returns:
            Joint angles (length = num_dof) that put the end-effector at the
            target XYZ in the rover-base frame.

        Raises:
            ValueError: If target is outside the reachable workspace or the
                solver fails to converge after ``max_iterations``.
        """
        self._require_configured()
        target_xyz = self._extract_target_xyz(target_pose)

        # Reject wildly out-of-reach targets up front to give a clear error.
        reach = float(np.sum(self._link_lengths)) + 1e-6
        base_to_target = float(np.linalg.norm(target_xyz - self._base_offset))
        if base_to_target > reach + 1e-4:
            raise ValueError(
                f"Target {target_xyz.tolist()} outside arm workspace "
                f"(distance {base_to_target:.3f} m > reach {reach:.3f} m)"
            )

        q = self._joint_pos.copy()
        damping = 1e-2
        tolerance = 1e-3
        max_iter = 200
        for _ in range(max_iter):
            T = self.forward_kinematics(q)
            error = target_xyz - T[:3, 3]
            if float(np.linalg.norm(error)) < tolerance:
                return self._clamp_limits(q)
            J = self._position_jacobian(q)
            # Damped pseudo-inverse: dq = J^T (J J^T + λ^2 I)^-1 · error
            JJt = J @ J.T + (damping ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(JJt, error)
            q = self._clamp_limits(q + dq)

        raise ValueError(
            f"Inverse kinematics failed to converge for target {target_xyz.tolist()}"
        )

    # ------------------------------------------------------------------
    # ABC: command
    # ------------------------------------------------------------------

    def command_joints(self, velocities: NDArray) -> None:
        self._require_configured()
        v = np.asarray(velocities, dtype=np.float64).flatten()
        if v.size != self._joint_pos.size:
            raise ValueError(
                f"Expected {self._joint_pos.size} joint velocities, got {v.size}"
            )
        self._joint_vel = v

    def command_gripper(self, open_fraction: float) -> None:
        self._require_configured()
        f = float(np.clip(open_fraction, 0.0, 1.0))
        self._gripper_target_fraction = f
        assert self._gripper_config is not None
        self._gripper_open_m = f * self._gripper_config.max_open_m

    # ------------------------------------------------------------------
    # Explicit integrator — used by the simulation loop to advance arm state
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        """Integrate joint positions from current velocity command."""
        self._require_configured()
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        self._joint_pos = self._clamp_limits(self._joint_pos + self._joint_vel * dt)

    # ------------------------------------------------------------------
    # ABC: state / grasp / stow
    # ------------------------------------------------------------------

    def get_state(self) -> ArmState:
        self._require_configured()
        return ArmState(
            joint_positions=self._joint_pos.astype(np.float32),
            joint_velocities=self._joint_vel.astype(np.float32),
            joint_torques=self._joint_torque.astype(np.float32),
            gripper_position=self._gripper_open_m,
            ft_reading=self._ft_reading.astype(np.float32),
        )

    def get_grasp_quality(self) -> GraspQuality:
        self._require_configured()
        assert self._gripper_config is not None
        # Force magnitude from FT sensor (first 3 components = Fx, Fy, Fz).
        force = float(np.linalg.norm(self._ft_reading[:3]))
        closed_enough = self._gripper_open_m < 0.5 * self._gripper_config.max_open_m
        stable = (force >= 1.0) and closed_enough
        contact_points = self._gripper_config.num_fingers if closed_enough else 0
        return GraspQuality(force_n=force, stable=stable, contact_points=contact_points)

    def is_in_stow_position(self) -> bool:
        self._require_configured()
        if self._stow_angles.size == 0:
            return True
        error = self._joint_pos - self._stow_angles
        # Stowed when every joint is within ~2° of its stow target.
        return bool(np.all(np.abs(error) < math.radians(2.0)))

    def stow(self) -> None:
        self._require_configured()
        # Instantaneous move to stow; the simulation loop integrates velocity
        # but callers can also snap to the pose via this method.
        self._joint_pos = self._stow_angles.copy()
        self._joint_vel = np.zeros_like(self._joint_vel)

    # ------------------------------------------------------------------
    # Extended API used by higher-level controllers
    # ------------------------------------------------------------------

    def set_ft_reading(self, ft: Sequence[float]) -> None:
        """Update the simulated wrist force-torque reading."""
        arr = np.asarray(ft, dtype=np.float64).flatten()
        if arr.size != 6:
            raise ValueError(f"ft must be length 6 (Fx,Fy,Fz,Tx,Ty,Tz), got {arr.size}")
        self._ft_reading = arr

    def set_joint_positions(self, angles: Sequence[float]) -> None:
        """Hard-set joint positions (e.g. to restore from a snapshot)."""
        self._require_configured()
        arr = np.asarray(angles, dtype=np.float64).flatten()
        if arr.size != self._joint_pos.size:
            raise ValueError(
                f"Expected {self._joint_pos.size} joint angles, got {arr.size}"
            )
        self._joint_pos = self._clamp_limits(arr)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _position_jacobian(self, angles: NDArray) -> NDArray:
        """3xN position Jacobian in the rover-base frame."""
        n = angles.size
        J = np.zeros((3, n), dtype=np.float64)

        # Accumulate transforms joint-by-joint and record axis + origin in world.
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = self._base_offset
        origins: List[NDArray] = []
        axes_world: List[NDArray] = []

        for spec, theta in zip(self._joints, angles):
            # Joint axis expressed in the base frame (before applying its own rotation).
            axis_world = T[:3, :3] @ spec.axis
            axes_world.append(axis_world)
            origins.append(T[:3, 3].copy())
            T = T @ _rotation_matrix(spec.axis, float(theta))
            T = T @ _translation_matrix(np.array([spec.link_length, 0.0, 0.0]))

        end_effector = T[:3, 3]
        for i in range(n):
            J[:, i] = np.cross(axes_world[i], end_effector - origins[i])
        return J

    def _resolve_axes(self, num_dof: int) -> List[NDArray]:
        if self._requested_axes is not None:
            axes = [np.asarray(a, dtype=np.float64) for a in self._requested_axes]
            if len(axes) != num_dof:
                raise ValueError(
                    f"joint_axes length ({len(axes)}) must equal num_dof ({num_dof})"
                )
            return axes
        # Default: first joint yaws about Z, remaining joints pitch about Y.
        axes = [np.array([0.0, 0.0, 1.0])]
        for _ in range(num_dof - 1):
            axes.append(np.array([0.0, 1.0, 0.0]))
        return axes

    @staticmethod
    def _resolve_link_lengths(num_dof: int, reach_m: float) -> List[float]:
        # Distribute the total reach uniformly across links.
        if num_dof == 0:
            return []
        per_link = reach_m / num_dof
        return [per_link] * num_dof

    def _clamp_limits(self, angles: NDArray) -> NDArray:
        out = angles.copy()
        for i, spec in enumerate(self._joints):
            out[i] = float(np.clip(out[i], spec.lower, spec.upper))
        return out

    def _extract_target_xyz(self, target_pose: NDArray) -> NDArray:
        pose = np.asarray(target_pose, dtype=np.float64)
        if pose.shape == (4, 4):
            return pose[:3, 3].copy()
        flat = pose.flatten()
        if flat.size >= 3:
            return flat[:3].astype(np.float64)
        raise ValueError(f"target_pose must be 4x4 or length-3+: got shape {pose.shape}")

    def _require_configured(self, n_expected: Optional[int] = None) -> None:
        if self._arm_config is None:
            raise RuntimeError("SerialArm.configure() must be called first")
        if n_expected is not None and n_expected != self._joint_pos.size:
            raise ValueError(
                f"Expected {self._joint_pos.size} joint values, got {n_expected}"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rotation_matrix(axis: NDArray, theta: float) -> NDArray:
    """4x4 homogeneous rotation about ``axis`` by ``theta`` (Rodrigues' formula)."""
    ax = axis / max(float(np.linalg.norm(axis)), 1e-12)
    c, s = math.cos(theta), math.sin(theta)
    K = np.array(
        [
            [0.0, -ax[2], ax[1]],
            [ax[2], 0.0, -ax[0]],
            [-ax[1], ax[0], 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _translation_matrix(offset: NDArray) -> NDArray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = offset
    return T


def arm_config_from_yaml(arm_cfg: dict) -> Tuple[ArmConfig, GripperConfig]:
    """Build :class:`ArmConfig` and :class:`GripperConfig` from ``rover.yaml.arm``."""
    num_dof = int(arm_cfg.get("num_dof", 4))

    joints_section = arm_cfg.get("joints", {}) or {}
    joint_limits: List[Tuple[float, float]] = []
    for i in range(1, num_dof + 1):
        jkey = f"joint_{i}"
        j = joints_section.get(jkey, {})
        lo = float(j.get("lower_limit_rad", -math.pi))
        hi = float(j.get("upper_limit_rad", math.pi))
        joint_limits.append((lo, hi))

    arm = ArmConfig(
        num_dof=num_dof,
        joint_limits=joint_limits,
        reach_m=float(arm_cfg.get("reach_m", 2.0)),
        payload_kg=float(arm_cfg.get("payload_kg", 5.0)),
        joint_accuracy_deg=float(arm_cfg.get("joint_accuracy_deg", 0.5)),
    )
    gripper_cfg = arm_cfg.get("gripper", {}) or {}
    gripper = GripperConfig(
        num_fingers=int(gripper_cfg.get("num_fingers", 2)),
        max_open_m=float(gripper_cfg.get("stroke_m", 0.1)),
        max_force_n=float(gripper_cfg.get("max_grip_force_n", 500.0)),
        compliance_model=str(gripper_cfg.get("compliance_model", "linear")),
    )
    return arm, gripper
