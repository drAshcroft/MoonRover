"""Unit tests for SerialArm forward/inverse kinematics and gripper behaviour."""
from __future__ import annotations

import math

import numpy as np
import pytest

from moon_rover.rover.manipulator.arm import ArmConfig, GripperConfig
from moon_rover.rover.manipulator.serial_arm import SerialArm, arm_config_from_yaml


def _arm_config(num_dof: int = 4, reach: float = 2.0) -> ArmConfig:
    limits = [(-math.pi, math.pi)] * num_dof
    return ArmConfig(
        num_dof=num_dof,
        joint_limits=limits,
        reach_m=reach,
        payload_kg=5.0,
        joint_accuracy_deg=0.5,
    )


def _gripper_config() -> GripperConfig:
    return GripperConfig(num_fingers=2, max_open_m=0.1, max_force_n=500.0, compliance_model="linear")


def _make(arm_cfg: ArmConfig | None = None, gripper_cfg: GripperConfig | None = None) -> SerialArm:
    arm = SerialArm()
    arm.configure(arm_cfg or _arm_config(), gripper_cfg or _gripper_config())
    return arm


# ---------------------------------------------------------------------------
# Configure validation
# ---------------------------------------------------------------------------


def test_configure_rejects_joint_limits_length_mismatch():
    cfg = _arm_config(num_dof=4)
    cfg.joint_limits = [(-1.0, 1.0)] * 3
    arm = SerialArm()
    with pytest.raises(ValueError):
        arm.configure(cfg, _gripper_config())


def test_configure_rejects_non_positive_dof():
    cfg = _arm_config()
    cfg.num_dof = 0
    cfg.joint_limits = []
    arm = SerialArm()
    with pytest.raises(ValueError):
        arm.configure(cfg, _gripper_config())


def test_configure_rejects_non_positive_reach():
    cfg = _arm_config()
    cfg.reach_m = 0.0
    arm = SerialArm()
    with pytest.raises(ValueError):
        arm.configure(cfg, _gripper_config())


def test_fk_before_configure_raises():
    arm = SerialArm()
    with pytest.raises(RuntimeError):
        arm.forward_kinematics(np.zeros(4))


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------


def test_fk_zero_angles_extends_along_x_by_reach():
    arm = _make(_arm_config(num_dof=4, reach=2.0))
    T = arm.forward_kinematics(np.zeros(4))
    # Links sum to reach_m; end-effector lies at (reach, 0, 0).
    assert T[:3, 3] == pytest.approx([2.0, 0.0, 0.0], rel=1e-9, abs=1e-9)


def test_fk_shape_is_4x4_homogeneous():
    arm = _make()
    T = arm.forward_kinematics(np.zeros(4))
    assert T.shape == (4, 4)
    # Bottom row is [0, 0, 0, 1].
    assert T[3, :].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_fk_yaw_rotates_end_effector_around_z():
    arm = _make(_arm_config(num_dof=2, reach=2.0))
    # First joint is about +Z. With theta=pi/2 the straight arm points at +Y.
    q = np.array([math.pi / 2, 0.0])
    T = arm.forward_kinematics(q)
    assert T[:3, 3] == pytest.approx([0.0, 2.0, 0.0], abs=1e-9)


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------


def test_ik_reaches_interior_target_within_tolerance():
    arm = _make(_arm_config(num_dof=4, reach=2.0))
    # The straight-arm q=0 state is a kinematic singularity in extension; seed
    # the joints so DLS starts off-singular, then IK back to an FK-derived target.
    known = np.array([0.2, -0.4, -0.3, 0.1], dtype=np.float64)
    arm.set_joint_positions(known)
    target = arm.forward_kinematics(known)[:3, 3].copy()
    arm.set_joint_positions(known + 0.05)
    q = arm.inverse_kinematics(target)
    T = arm.forward_kinematics(q)
    assert np.linalg.norm(T[:3, 3] - target) < 1e-2


def test_ik_rejects_unreachable_target():
    arm = _make(_arm_config(num_dof=4, reach=1.0))
    far = np.array([10.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        arm.inverse_kinematics(far)


def test_ik_accepts_4x4_homogeneous_pose():
    arm = _make()
    # Establish an off-singular starting pose and an FK-derived reachable target.
    known = np.array([0.15, -0.35, -0.2, 0.05], dtype=np.float64)
    arm.set_joint_positions(known)
    xyz = arm.forward_kinematics(known)[:3, 3]
    pose = np.eye(4)
    pose[:3, 3] = xyz
    arm.set_joint_positions(known + 0.05)
    q = arm.inverse_kinematics(pose)
    T = arm.forward_kinematics(q)
    assert T[:3, 3] == pytest.approx(xyz, abs=1e-2)


def test_ik_clamps_result_to_joint_limits():
    cfg = _arm_config(num_dof=3, reach=2.0)
    cfg.joint_limits = [(-0.2, 0.2)] * 3  # tight limits
    arm = _make(cfg)
    # Any solution must obey the limits, even if the target wasn't reached.
    try:
        q = arm.inverse_kinematics(np.array([1.5, 0.3, 0.0]))
    except ValueError:
        return
    for angle, (lo, hi) in zip(q, cfg.joint_limits):
        assert lo - 1e-9 <= angle <= hi + 1e-9


# ---------------------------------------------------------------------------
# Commands and integration
# ---------------------------------------------------------------------------


def test_command_joints_integrates_over_update():
    arm = _make()
    arm.command_joints(np.array([0.5, 0.0, 0.0, 0.0]))
    arm.update(0.1)
    state = arm.get_state()
    assert state.joint_positions[0] == pytest.approx(0.05, rel=1e-6)


def test_command_joints_wrong_length_raises():
    arm = _make()
    with pytest.raises(ValueError):
        arm.command_joints(np.zeros(3))


def test_update_rejects_negative_dt():
    arm = _make()
    with pytest.raises(ValueError):
        arm.update(-0.01)


# ---------------------------------------------------------------------------
# Gripper
# ---------------------------------------------------------------------------


def test_command_gripper_clamps_fraction_to_unit_interval():
    arm = _make()
    arm.command_gripper(5.0)
    assert arm.get_state().gripper_position == pytest.approx(0.1)
    arm.command_gripper(-2.0)
    assert arm.get_state().gripper_position == pytest.approx(0.0)


def test_grasp_quality_requires_force_and_closed_gripper():
    arm = _make()
    # Fully open + no force → unstable.
    arm.command_gripper(1.0)
    q = arm.get_grasp_quality()
    assert q.stable is False
    assert q.contact_points == 0

    # Closed + nonzero FT force → stable.
    arm.command_gripper(0.0)
    arm.set_ft_reading([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q = arm.get_grasp_quality()
    assert q.stable is True
    assert q.force_n == pytest.approx(5.0, rel=1e-6)
    assert q.contact_points == 2


# ---------------------------------------------------------------------------
# Stow
# ---------------------------------------------------------------------------


def test_stow_snaps_to_default_angles_and_reports_in_stow():
    arm = _make()
    arm.command_joints(np.array([0.4, 0.3, 0.2, 0.1]))
    arm.update(1.0)
    assert not arm.is_in_stow_position()
    arm.stow()
    assert arm.is_in_stow_position()


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def test_arm_config_from_yaml_builds_expected_limits():
    raw = {
        "num_dof": 3,
        "reach_m": 1.5,
        "payload_kg": 4.0,
        "joint_accuracy_deg": 0.3,
        "joints": {
            "joint_1": {"lower_limit_rad": -1.0, "upper_limit_rad": 1.0},
            "joint_2": {"lower_limit_rad": -2.0, "upper_limit_rad": 2.0},
            "joint_3": {"lower_limit_rad": -0.5, "upper_limit_rad": 0.5},
        },
        "gripper": {"num_fingers": 3, "stroke_m": 0.08,
                    "max_grip_force_n": 300.0, "compliance_model": "exponential"},
    }
    arm_cfg, gripper_cfg = arm_config_from_yaml(raw)
    assert arm_cfg.num_dof == 3
    assert arm_cfg.joint_limits == [(-1.0, 1.0), (-2.0, 2.0), (-0.5, 0.5)]
    assert gripper_cfg.num_fingers == 3
    assert gripper_cfg.max_open_m == pytest.approx(0.08)
