"""Unit tests for live-pose manipulation task planning."""

from __future__ import annotations

import math

import numpy as np
import pytest

from moon_rover.navigation.manipulation.sequencer import (
    ArmManipulationSequencer,
    ManipulationTask,
    ManipulationTaskContext,
)


class _RecordingArm:
    def __init__(self) -> None:
        self.trajectories: list[list[np.ndarray]] = []
        self.gripper_commands: list[float] = []

    def follow_trajectory(self, waypoints: list[np.ndarray]) -> bool:
        self.trajectories.append(list(waypoints))
        return True

    def command_gripper(self, open_fraction: float) -> None:
        self.gripper_commands.append(open_fraction)

    def abort(self) -> bool:
        return True


def _yaw_quat(degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    return np.array([0.0, 0.0, math.sin(radians / 2.0), math.cos(radians / 2.0)])


def _pose(position, quat=None) -> np.ndarray:
    orientation = _yaw_quat(90.0) if quat is None else np.asarray(quat)
    return np.concatenate([np.asarray(position, dtype=np.float64), orientation])


def _context() -> ManipulationTaskContext:
    rover = _pose([10.0, 5.0, 0.5])
    return ManipulationTaskContext(
        rover_world_pose=rover,
        depot_world_pose=_pose([10.0, 5.8, 0.5]),
        antenna_world_pose=_pose([10.0, 6.0, 0.5]),
        surveyed_target_world_position=np.array([9.0, 5.0, 0.5]),
        surveyed_surface_normal_world=np.array([0.0, 0.0, 1.0]),
        antenna_port_world_pose=_pose([10.0, 5.5, 0.7]),
        cable_reel_world_pose=_pose([10.0, 5.9, 0.5]),
    )


def test_pickup_uses_live_antenna_pose_in_rover_body_frame():
    arm = _RecordingArm()

    assert ArmManipulationSequencer().execute_task(
        ManipulationTask.ANTENNA_PICKUP, arm, _context()
    )

    grasp = arm.trajectories[0][2]
    np.testing.assert_allclose(grasp, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-8)
    assert arm.gripper_commands == [0.0]


def test_placement_uses_surveyed_world_target_and_surface_normal():
    arm = _RecordingArm()

    assert ArmManipulationSequencer().execute_task(
        ManipulationTask.ANTENNA_PLACEMENT, arm, _context()
    )

    contact = arm.trajectories[0][2]
    np.testing.assert_allclose(contact, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-8)
    assert arm.gripper_commands == [1.0]


def test_cable_connection_uses_live_antenna_port_pose():
    arm = _RecordingArm()

    assert ArmManipulationSequencer().execute_task(
        ManipulationTask.CABLE_CONNECTION, arm, _context()
    )

    contact = arm.trajectories[0][2]
    np.testing.assert_allclose(contact, [0.5, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], atol=1e-8)


def test_cable_reel_pickup_uses_live_reel_pose():
    arm = _RecordingArm()

    assert ArmManipulationSequencer().execute_task(
        ManipulationTask.CABLE_REEL_PICKUP, arm, _context()
    )

    grasp = arm.trajectories[0][2]
    np.testing.assert_allclose(grasp[:3], [0.9, 0.0, 0.0], atol=1e-8)


@pytest.mark.parametrize(
    "task",
    [
        ManipulationTask.ANTENNA_PICKUP,
        ManipulationTask.ANTENNA_PLACEMENT,
        ManipulationTask.CABLE_CONNECTION,
        ManipulationTask.CABLE_REEL_PICKUP,
    ],
)
def test_live_pose_tasks_reject_missing_context(task: ManipulationTask):
    with pytest.raises(ValueError, match="requires a ManipulationTaskContext"):
        ArmManipulationSequencer().execute_task(task, _RecordingArm())


def test_transport_stow_does_not_require_live_geometry():
    arm = _RecordingArm()

    assert ArmManipulationSequencer().execute_task(ManipulationTask.TRANSPORT_STOW, arm)
    assert len(arm.trajectories) == 1


def test_placement_rejects_zero_surface_normal():
    context = _context()
    context = ManipulationTaskContext(
        **{
            **context.__dict__,
            "surveyed_surface_normal_world": np.zeros(3),
        }
    )

    with pytest.raises(ValueError, match="must not be a zero vector"):
        ArmManipulationSequencer().execute_task(
            ManipulationTask.ANTENNA_PLACEMENT, _RecordingArm(), context
        )
