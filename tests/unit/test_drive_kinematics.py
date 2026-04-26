"""Pure-logic unit tests for rover DriveSystem kinematics.

These tests exercise the configure/IK/FK contract of the three concrete drive
implementations without attaching to a Genesis entity; they run fast under the
standard unit-test lane and do not require the Genesis CPU kernels.
"""
from __future__ import annotations

import math

import pytest

from moon_rover.rover.drive.genesis_drive import (
    FourWheelSkidSteerDrive,
    ThreeWheelTricycleDrive,
    TwoWheelDifferentialDrive,
    create_drive_system,
    drive_config_from_profile,
)
from moon_rover.rover.drive.interface import (
    DriveCommand,
    DriveConfig,
    DriveType,
)


def _two_wheel_config() -> DriveConfig:
    return DriveConfig(
        drive_type=DriveType.TWO_WHEEL_DIFF,
        track_width_m=1.2,
        wheelbase_m=0.0,
        wheel_radius_m=0.3,
        max_torque_nm=100.0,
        num_wheels=2,
    )


def _tricycle_config() -> DriveConfig:
    return DriveConfig(
        drive_type=DriveType.THREE_WHEEL_TRICYCLE,
        track_width_m=1.0,
        wheelbase_m=1.4,
        wheel_radius_m=0.25,
        max_torque_nm=80.0,
        max_steer_angle_rad=math.radians(35.0),
        num_wheels=3,
    )


def _skid_config() -> DriveConfig:
    return DriveConfig(
        drive_type=DriveType.FOUR_WHEEL_SKID,
        track_width_m=1.1,
        wheelbase_m=1.0,
        wheel_radius_m=0.3,
        max_torque_nm=90.0,
        num_wheels=4,
    )


# ---------------------------------------------------------------------------
# Two-wheel differential
# ---------------------------------------------------------------------------


def test_two_wheel_ik_forward_matches_v_over_r():
    drive = TwoWheelDifferentialDrive()
    drive.configure(_two_wheel_config())
    # Pure forward at 1 m/s with 0.3 m wheels: both wheels spin at 1/0.3 rad/s.
    speeds = drive.inverse_kinematics(DriveCommand(1.0, 0.0))
    assert speeds == pytest.approx([1.0 / 0.3, 1.0 / 0.3], rel=1e-9)


def test_two_wheel_ik_spin_gives_equal_and_opposite_speeds():
    drive = TwoWheelDifferentialDrive()
    drive.configure(_two_wheel_config())
    # In-place spin: v=0, omega=1 rad/s, track=1.2 → wheel speed = ±(0.6)/0.3 = 2.
    speeds = drive.inverse_kinematics(DriveCommand(0.0, 1.0))
    assert speeds[0] == pytest.approx(-2.0, rel=1e-9)
    assert speeds[1] == pytest.approx(+2.0, rel=1e-9)


def test_two_wheel_fk_is_inverse_of_ik():
    drive = TwoWheelDifferentialDrive()
    drive.configure(_two_wheel_config())
    for v, w in [(0.5, 0.0), (0.0, 0.5), (1.2, -0.4), (-0.75, 0.9)]:
        speeds = drive.inverse_kinematics(DriveCommand(v, w))
        twist = drive.forward_kinematics(speeds)
        assert twist.linear_velocity_mps == pytest.approx(v, rel=1e-9, abs=1e-9)
        assert twist.angular_velocity_radps == pytest.approx(w, rel=1e-9, abs=1e-9)


def test_two_wheel_configure_rejects_wrong_drive_type():
    drive = TwoWheelDifferentialDrive()
    cfg = _two_wheel_config()
    cfg.drive_type = DriveType.FOUR_WHEEL_SKID
    with pytest.raises(ValueError):
        drive.configure(cfg)


def test_two_wheel_configure_rejects_invalid_geometry():
    drive = TwoWheelDifferentialDrive()
    bad = _two_wheel_config()
    bad.wheel_radius_m = 0.0
    with pytest.raises(ValueError):
        drive.configure(bad)


def test_two_wheel_fk_wrong_count_raises():
    drive = TwoWheelDifferentialDrive()
    drive.configure(_two_wheel_config())
    with pytest.raises(ValueError):
        drive.forward_kinematics([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Skid-steer
# ---------------------------------------------------------------------------


def test_skid_steer_ik_returns_four_wheels_paired_by_side():
    drive = FourWheelSkidSteerDrive()
    drive.configure(_skid_config())
    speeds = drive.inverse_kinematics(DriveCommand(1.0, 0.5))
    assert len(speeds) == 4
    # Front-left == rear-left, front-right == rear-right.
    assert speeds[0] == pytest.approx(speeds[2], rel=1e-9)
    assert speeds[1] == pytest.approx(speeds[3], rel=1e-9)


def test_skid_steer_fk_ik_round_trip():
    drive = FourWheelSkidSteerDrive()
    drive.configure(_skid_config())
    for v, w in [(1.0, 0.0), (0.3, -0.8), (-0.5, 0.4)]:
        speeds = drive.inverse_kinematics(DriveCommand(v, w))
        twist = drive.forward_kinematics(speeds)
        assert twist.linear_velocity_mps == pytest.approx(v, rel=1e-9, abs=1e-9)
        assert twist.angular_velocity_radps == pytest.approx(w, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# Three-wheel tricycle
# ---------------------------------------------------------------------------


def test_tricycle_pure_forward_sets_zero_steering():
    drive = ThreeWheelTricycleDrive()
    drive.configure(_tricycle_config())
    speeds = drive.inverse_kinematics(DriveCommand(1.5, 0.0))
    assert drive._steer_angle_target == pytest.approx(0.0, abs=1e-9)
    assert speeds == pytest.approx([1.5 / 0.25, 1.5 / 0.25], rel=1e-9)


def test_tricycle_steering_saturates_at_config_limit():
    drive = ThreeWheelTricycleDrive()
    cfg = _tricycle_config()
    drive.configure(cfg)
    # Very large yaw demand at low speed saturates steering.
    drive.inverse_kinematics(DriveCommand(0.2, 5.0))
    max_delta = cfg.max_steer_angle_rad
    assert abs(drive._steer_angle_target) == pytest.approx(max_delta, rel=1e-6)


def test_tricycle_fk_returns_bicycle_model_yaw_rate():
    drive = ThreeWheelTricycleDrive()
    drive.configure(_tricycle_config())
    # Command that keeps steering within limits.
    drive.inverse_kinematics(DriveCommand(1.0, 0.3))
    twist = drive.forward_kinematics([1.0 / 0.25, 1.0 / 0.25])
    # Forward speed recovered exactly.
    assert twist.linear_velocity_mps == pytest.approx(1.0, rel=1e-9)
    # Yaw rate should be (v / L) * tan(delta) ≈ original 0.3 rad/s target.
    assert twist.angular_velocity_radps == pytest.approx(0.3, rel=1e-3)


# ---------------------------------------------------------------------------
# Factory / profile loading
# ---------------------------------------------------------------------------


def test_create_drive_system_builds_correct_concrete_type():
    two = create_drive_system(_two_wheel_config())
    tri = create_drive_system(_tricycle_config())
    skid = create_drive_system(_skid_config())
    assert isinstance(two, TwoWheelDifferentialDrive)
    assert isinstance(tri, ThreeWheelTricycleDrive)
    assert isinstance(skid, FourWheelSkidSteerDrive)
    assert two.get_drive_type() is DriveType.TWO_WHEEL_DIFF
    assert tri.get_drive_type() is DriveType.THREE_WHEEL_TRICYCLE
    assert skid.get_drive_type() is DriveType.FOUR_WHEEL_SKID


def test_drive_config_from_profile_parses_steering_degrees():
    profile = {
        "track_width_m": 1.1,
        "wheelbase_m": 1.4,
        "wheel_radius_m": 0.25,
        "max_torque_nm": 75.0,
        "num_wheels": 3,
        "steering": {"max_steering_angle_degrees": 30.0},
    }
    cfg = drive_config_from_profile(profile, DriveType.THREE_WHEEL_TRICYCLE)
    assert cfg.track_width_m == pytest.approx(1.1)
    assert cfg.wheelbase_m == pytest.approx(1.4)
    assert cfg.max_steer_angle_rad == pytest.approx(math.radians(30.0))
