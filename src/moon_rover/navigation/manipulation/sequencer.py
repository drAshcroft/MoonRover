"""System 11.5: Manipulation Planning — Task Sequencer.

This module provides task-level sequencing and planning for the rover's
manipulation arm (e.g., antenna pickup, placement, and cable connection).

The sequencer coordinates arm motion planning, end-effector control, and
gripper actuation to achieve high-level manipulation objectives.

Classes:
    ManipulationTask (Enum): Discrete manipulation task types
    ManipulationSequencer (ABC): Abstract interface for arm planning and control

Typical Usage:
    sequencer = ManipulationSequencer(...)
    waypoints = sequencer.plan_pickup(depot_pose, target_pose)
    context = ManipulationTaskContext(
        rover_world_pose=...,
        depot_world_pose=...,
        antenna_world_pose=...,
    )
    success = sequencer.execute_task(ManipulationTask.ANTENNA_PICKUP, arm, context)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import List, Optional


class ManipulationTask(Enum):
    """Discrete manipulation tasks performed by the rover arm.

    Tasks are sequenced based on mission objectives and ordered by
    dependencies (e.g., pickup before placement).

    Attributes:
        ANTENNA_PICKUP: Grasp antenna from depot and load onto rover
        CABLE_REEL_PICKUP: Grasp cable reel and mount to cable dispensing system
        TRANSPORT_STOW: Stow antenna/reel to safe position for transit
        ANTENNA_PLACEMENT: Deploy antenna at target location with orientation
        CABLE_CONNECTION: Connect cable between antenna and rover/base
    """
    ANTENNA_PICKUP = "antenna_pickup"
    CABLE_REEL_PICKUP = "cable_reel_pickup"
    TRANSPORT_STOW = "transport_stow"
    ANTENNA_PLACEMENT = "antenna_placement"
    CABLE_CONNECTION = "cable_connection"


@dataclass(frozen=True)
class ManipulationTaskContext:
    """Live world-frame geometry needed to plan a manipulation task.

    The arm planner emits rover-body-frame waypoints. Callers provide surveyed
    targets and sensed entity poses in world coordinates alongside the rover
    pose used to transform them into that planning frame.
    """

    rover_world_pose: np.ndarray
    antenna_world_pose: Optional[np.ndarray] = None
    depot_world_pose: Optional[np.ndarray] = None
    surveyed_target_world_position: Optional[np.ndarray] = None
    surveyed_surface_normal_world: Optional[np.ndarray] = None
    antenna_port_world_pose: Optional[np.ndarray] = None
    cable_reel_world_pose: Optional[np.ndarray] = None


class ManipulationSequencer(ABC):
    """Abstract interface for robotic arm manipulation planning and control.

    The sequencer generates motion plans (waypoints) for arm tasks and
    executes high-level manipulation objectives. It abstracts away low-level
    inverse kinematics, collision checking, and gripper control.

    Typical Workflow:
        1. Call plan_* methods to generate waypoint trajectories
        2. Send waypoints to arm controller for execution
        3. Monitor completion and error states
        4. Execute high-level tasks via execute_task()

    Abstract Methods:
        plan_pickup: Generate approach and grasp waypoints
        plan_placement: Generate approach and release waypoints
        plan_cable_connection: Generate connection approach waypoints
        execute_task: Execute a named manipulation task
        get_stow_with_payload_pose: Query safe stow configuration with payload
    """

    @abstractmethod
    def plan_pickup(
        self,
        depot_pose: np.ndarray,
        antenna_pose: np.ndarray
    ) -> List[np.ndarray]:
        """Plan arm trajectory for picking up an antenna from the depot.

        Generates a sequence of arm end-effector waypoints to approach,
        grasp, and retract an antenna from a depot location.

        Args:
            depot_pose (np.ndarray): Depot location pose [x, y, z, qx, qy, qz, qw]
                                     (7-element pose in rover body frame)
            antenna_pose (np.ndarray): Target antenna pose relative to depot
                                       (7-element pose in rover body frame)

        Returns:
            List[np.ndarray]: Ordered list of arm end-effector waypoints.
                             Each waypoint is [x, y, z, qx, qy, qz, qw].
                             Sequence: approach -> contact -> grasp -> retract.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            depot_pose = np.array([0.5, 0, -0.3, 0, 0, 0, 1])  # 0.5m forward, 0.3m down
            antenna_pose = np.array([0.5, 0, -0.35, 0, 0, 0, 1])
            waypoints = sequencer.plan_pickup(depot_pose, antenna_pose)
            # waypoints[0]: approach position above antenna
            # waypoints[1]: contact position at antenna top
            # waypoints[2]: grasp position (slight retraction)
        """
        raise NotImplementedError("plan_pickup implementation pending")

    @abstractmethod
    def plan_placement(
        self,
        target_position: np.ndarray,
        surface_normal: np.ndarray
    ) -> List[np.ndarray]:
        """Plan arm trajectory for placing an antenna at a target location.

        Generates a sequence of arm end-effector waypoints to approach a
        target surface, orient the antenna correctly, and release it.

        Args:
            target_position (np.ndarray): Target deployment location [x, y, z]
                                          (3-element position in rover body frame)
            surface_normal (np.ndarray): Surface normal at target for orientation [nx, ny, nz]
                                        (unit vector in rover body frame)

        Returns:
            List[np.ndarray]: Ordered list of arm end-effector waypoints.
                             Each waypoint is [x, y, z, qx, qy, qz, qw].
                             Sequence: approach -> align -> contact -> release -> retract.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            target = np.array([0.6, 0.0, 0.0])  # 0.6m forward in rover body frame
            normal = np.array([0.0, 0.0, 1.0])    # Vertical mounting
            waypoints = sequencer.plan_placement(target, normal)
        """
        raise NotImplementedError("plan_placement implementation pending")

    @abstractmethod
    def plan_cable_connection(
        self,
        antenna_port_pose: np.ndarray
    ) -> List[np.ndarray]:
        """Plan arm trajectory for connecting a cable to an antenna port.

        Generates arm waypoints to approach an antenna's connector port,
        align the cable connector, and mate the connection.

        Args:
            antenna_port_pose (np.ndarray): Antenna connector port pose in rover body frame
                                            [x, y, z, qx, qy, qz, qw]

        Returns:
            List[np.ndarray]: Ordered list of arm end-effector waypoints.
                             Each waypoint is [x, y, z, qx, qy, qz, qw].
                             Sequence: approach -> align -> contact -> mate -> retract.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            port_pose = np.array([0.5, 0.0, 0.15, 0, 0, 0.707, 0.707])
            waypoints = sequencer.plan_cable_connection(port_pose)
        """
        raise NotImplementedError("plan_cable_connection implementation pending")

    @abstractmethod
    def execute_task(
        self,
        task: ManipulationTask,
        arm: 'ManipulatorArm',
        context: Optional[ManipulationTaskContext] = None,
    ) -> bool:
        """Execute a high-level manipulation task.

        Orchestrates the complete manipulation task including motion planning,
        execution, error recovery, and status monitoring.

        Args:
            task (ManipulationTask): The task to execute (pickup, placement, etc.)
            arm (ManipulatorArm): The arm controller object for motion execution.
                                 Expected interface:
                                   - arm.follow_trajectory(waypoints) -> bool
                                   - arm.get_status() -> str
                                   - arm.abort() -> bool
            context: Live world-frame poses required by antenna and cable tasks.

        Returns:
            bool: True if task completed successfully, False on failure.
                 Failures may include: planning failure, motion failure,
                 gripper error, or timeout.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            context = ManipulationTaskContext(
                rover_world_pose=rover_pose,
                depot_world_pose=depot_pose,
                antenna_world_pose=antenna_pose,
            )
            success = sequencer.execute_task(ManipulationTask.ANTENNA_PICKUP, arm, context)
            if success:
                print("Antenna picked up successfully")
            else:
                print("Pickup failed; attempting recovery")
        """
        raise NotImplementedError("execute_task implementation pending")

    @abstractmethod
    def get_stow_with_payload_pose(self) -> np.ndarray:
        """Get the arm's safe stow configuration while holding a payload.

        Returns the arm joint configuration (or end-effector pose) for
        safe transit with a payload (e.g., antenna). This pose ensures
        the payload is secured and balanced during rover motion.

        Returns:
            np.ndarray: Joint configuration [q1, q2, ..., qn] or
                       end-effector pose [x, y, z, qx, qy, qz, qw]
                       depending on sequencer implementation.
                       Shape depends on arm kinematics (typically 6-7 DOF).

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            stow_config = sequencer.get_stow_with_payload_pose()
            arm.move_to_config(stow_config)
        """
        raise NotImplementedError("get_stow_with_payload_pose implementation pending")


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

import logging  # noqa: E402

logger = logging.getLogger(__name__)


def _pose7(x: float, y: float, z: float, qx: float = 0.0, qy: float = 0.0,
           qz: float = 0.0, qw: float = 1.0) -> np.ndarray:
    return np.array([x, y, z, qx, qy, qz, qw], dtype=np.float64)


def _surface_normal_to_quat(normal: np.ndarray) -> np.ndarray:
    """Convert surface normal to quaternion orienting Z-axis toward normal."""
    normal = _require_vector(normal, 3, "surface_normal")
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError("surface_normal must not be a zero vector")
    n = normal / norm
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, n)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-9:
        if n[2] > 0:
            return np.array([0.0, 0.0, 0.0, 1.0])
        else:
            return np.array([0.0, 1.0, 0.0, 0.0])
    axis = cross / cross_norm
    angle = float(np.arccos(np.clip(np.dot(z, n), -1.0, 1.0)))
    s = np.sin(angle / 2.0)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)])


def _require_vector(value: np.ndarray, length: int, name: str) -> np.ndarray:
    """Return a finite float vector of the required length."""
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (length,):
        raise ValueError(f"{name} must contain exactly {length} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _normalize_quat(quat: np.ndarray, name: str) -> np.ndarray:
    """Normalize an [x, y, z, w] quaternion."""
    vector = _require_vector(quat, 4, name)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must not be a zero quaternion")
    return vector / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Return the conjugate of an [x, y, z, w] quaternion."""
    x, y, z, w = quat
    return np.array([-x, -y, -z, w], dtype=np.float64)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two normalized [x, y, z, w] quaternions."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return _normalize_quat(
        np.array(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ]
        ),
        "quaternion product",
    )


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector by an [x, y, z, w] quaternion."""
    q_xyz = quat[:3]
    return vector + (2.0 * np.cross(q_xyz, np.cross(q_xyz, vector) + quat[3] * vector))


def _world_position_to_body(position_world: np.ndarray, rover_world_pose: np.ndarray) -> np.ndarray:
    """Transform a world-frame position into rover body coordinates."""
    rover_pose = _require_vector(rover_world_pose, 7, "rover_world_pose")
    position = _require_vector(position_world, 3, "world position")
    rover_inverse = _quat_conjugate(_normalize_quat(rover_pose[3:7], "rover_world_pose quaternion"))
    return _quat_rotate(rover_inverse, position - rover_pose[:3])


def _world_vector_to_body(vector_world: np.ndarray, rover_world_pose: np.ndarray) -> np.ndarray:
    """Rotate a world-frame direction into rover body coordinates."""
    rover_pose = _require_vector(rover_world_pose, 7, "rover_world_pose")
    vector = _require_vector(vector_world, 3, "world vector")
    rover_inverse = _quat_conjugate(_normalize_quat(rover_pose[3:7], "rover_world_pose quaternion"))
    return _quat_rotate(rover_inverse, vector)


def _world_pose_to_body(pose_world: np.ndarray, rover_world_pose: np.ndarray) -> np.ndarray:
    """Transform a world-frame pose into rover body coordinates."""
    pose = _require_vector(pose_world, 7, "world pose")
    rover_pose = _require_vector(rover_world_pose, 7, "rover_world_pose")
    body_position = _world_position_to_body(pose[:3], rover_pose)
    rover_inverse = _quat_conjugate(_normalize_quat(rover_pose[3:7], "rover_world_pose quaternion"))
    body_quat = _quat_multiply(rover_inverse, _normalize_quat(pose[3:7], "world pose quaternion"))
    return _pose7(*body_position, *body_quat)


class ArmManipulationSequencer(ManipulationSequencer):
    """Concrete manipulation sequencer for antenna placement pipeline.

    Waypoints are expressed in rover body frame as 7-element poses
    [x, y, z, qx, qy, qz, qw]. Approach, grasp, carry, place, and stow
    phases are each broken into a small set of linearly interpolated
    intermediate poses to allow smooth arm trajectory following.

    The execute_task() method orchestrates a complete task using the arm's
    follow_trajectory() interface and monitors success/failure.
    """

    # Nominal arm geometry constants (can be overridden via subclass or config)
    _ARM_REACH_M: float = 0.8          # max arm reach in metres
    _APPROACH_STANDOFF_M: float = 0.15 # how far above target to approach from
    _CARRY_HEIGHT_M: float = 0.30      # z of payload during carry
    _CARRY_X_M: float = 0.40           # forward offset during carry
    _MAX_TASK_RETRIES: int = 2

    def __init__(
        self,
        approach_standoff_m: float = 0.15,
        carry_height_m: float = 0.30,
    ) -> None:
        self._approach_standoff_m = approach_standoff_m
        self._carry_height_m = carry_height_m
        self._current_task: ManipulationTask | None = None

    # ------------------------------------------------------------------
    # Waypoint builders
    # ------------------------------------------------------------------

    def plan_pickup(
        self,
        depot_pose: np.ndarray,
        antenna_pose: np.ndarray,
    ) -> List[np.ndarray]:
        """Generate approach → pre-grasp → grasp → retract waypoints."""
        target = antenna_pose[:3].copy()
        standoff = np.array([0.0, 0.0, self._approach_standoff_m])

        approach = _pose7(*(target + standoff))
        pre_grasp = _pose7(*(target + standoff * 0.5))
        grasp = _pose7(*target, *antenna_pose[3:] if len(antenna_pose) >= 7 else [0, 0, 0, 1])
        retract = _pose7(*(target + standoff * 1.5))
        carry = self.get_stow_with_payload_pose()

        return [approach, pre_grasp, grasp, retract, carry]

    def plan_placement(
        self,
        target_position: np.ndarray,
        surface_normal: np.ndarray,
    ) -> List[np.ndarray]:
        """Generate approach → align → contact → release → retract waypoints."""
        tgt = target_position[:3].copy()
        q = _surface_normal_to_quat(surface_normal)
        normal_unit = surface_normal / (np.linalg.norm(surface_normal) + 1e-9)

        above = tgt + normal_unit * self._approach_standoff_m * 2.0
        approach = _pose7(*above, *q)
        align = _pose7(*(tgt + normal_unit * self._approach_standoff_m), *q)
        contact = _pose7(*tgt, *q)
        release = _pose7(*tgt, *q)           # gripper opens at this pose
        retract = _pose7(*above, *q)

        return [approach, align, contact, release, retract]

    def plan_cable_connection(
        self,
        antenna_port_pose: np.ndarray,
    ) -> List[np.ndarray]:
        """Generate approach → align → contact → mate → retract waypoints."""
        port_pos = antenna_port_pose[:3].copy()
        q = antenna_port_pose[3:7] if len(antenna_port_pose) >= 7 else np.array([0, 0, 0, 1.0])

        # Approach from +Z above the port
        above = port_pos + np.array([0.0, 0.0, self._approach_standoff_m * 2.0])
        approach = _pose7(*above, *q)
        pre_align = _pose7(*(port_pos + np.array([0.0, 0.0, self._approach_standoff_m])), *q)
        contact = _pose7(*port_pos, *q)
        mated = _pose7(*port_pos, *q)    # small push to seat connector
        retract = _pose7(*above, *q)

        return [approach, pre_align, contact, mated, retract]

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    def execute_task(
        self,
        task: ManipulationTask,
        arm: "ManipulatorArm",
        context: Optional[ManipulationTaskContext] = None,
    ) -> bool:
        self._validate_context(task, context)
        self._current_task = task
        success = False
        for attempt in range(self._MAX_TASK_RETRIES + 1):
            try:
                success = self._run_task(task, arm, context)
                if success:
                    break
                logger.warning("Task %s attempt %d/%d failed, retrying", task.value, attempt + 1, self._MAX_TASK_RETRIES + 1)
            except Exception as exc:  # noqa: BLE001
                logger.error("Task %s raised exception on attempt %d: %s", task.value, attempt + 1, exc)
                try:
                    arm.abort()
                except Exception:  # noqa: BLE001
                    pass
        self._current_task = None
        return success

    def _run_task(
        self,
        task: ManipulationTask,
        arm: "ManipulatorArm",
        context: Optional[ManipulationTaskContext],
    ) -> bool:
        if task == ManipulationTask.TRANSPORT_STOW:
            stow = self.get_stow_with_payload_pose()
            return bool(arm.follow_trajectory([stow]))

        if task == ManipulationTask.ANTENNA_PICKUP:
            assert context is not None
            assert context.depot_world_pose is not None
            assert context.antenna_world_pose is not None
            depot = _world_pose_to_body(context.depot_world_pose, context.rover_world_pose)
            antenna = _world_pose_to_body(context.antenna_world_pose, context.rover_world_pose)
            waypoints = self.plan_pickup(depot, antenna)
            if not arm.follow_trajectory(waypoints[:-2]):  # approach → grasp
                return False
            # Close gripper
            try:
                arm.command_gripper(0.0)
            except AttributeError:
                pass
            # Retract and stow
            return bool(arm.follow_trajectory(waypoints[-2:]))

        if task == ManipulationTask.ANTENNA_PLACEMENT:
            assert context is not None
            assert context.surveyed_target_world_position is not None
            assert context.surveyed_surface_normal_world is not None
            target = _world_position_to_body(
                context.surveyed_target_world_position, context.rover_world_pose
            )
            normal = _world_vector_to_body(
                context.surveyed_surface_normal_world, context.rover_world_pose
            )
            waypoints = self.plan_placement(target, normal)
            if not arm.follow_trajectory(waypoints[:3]):  # approach → contact
                return False
            # Open gripper to release
            try:
                arm.command_gripper(1.0)
            except AttributeError:
                pass
            return bool(arm.follow_trajectory(waypoints[3:]))  # retract

        if task == ManipulationTask.CABLE_CONNECTION:
            assert context is not None
            assert context.antenna_port_world_pose is not None
            port = _world_pose_to_body(context.antenna_port_world_pose, context.rover_world_pose)
            waypoints = self.plan_cable_connection(port)
            return bool(arm.follow_trajectory(waypoints))

        if task == ManipulationTask.CABLE_REEL_PICKUP:
            assert context is not None
            assert context.depot_world_pose is not None
            assert context.cable_reel_world_pose is not None
            depot = _world_pose_to_body(context.depot_world_pose, context.rover_world_pose)
            reel = _world_pose_to_body(context.cable_reel_world_pose, context.rover_world_pose)
            waypoints = self.plan_pickup(depot, reel)
            if not arm.follow_trajectory(waypoints[:-2]):
                return False
            try:
                arm.command_gripper(0.0)
            except AttributeError:
                pass
            return bool(arm.follow_trajectory(waypoints[-2:]))

        logger.warning("Unknown task type: %s", task)
        return False

    @staticmethod
    def _validate_context(
        task: ManipulationTask,
        context: Optional[ManipulationTaskContext],
    ) -> None:
        """Reject tasks whose live planning geometry is incomplete."""
        required_by_task = {
            ManipulationTask.ANTENNA_PICKUP: ("depot_world_pose", "antenna_world_pose"),
            ManipulationTask.ANTENNA_PLACEMENT: (
                "surveyed_target_world_position",
                "surveyed_surface_normal_world",
            ),
            ManipulationTask.CABLE_CONNECTION: ("antenna_port_world_pose",),
            ManipulationTask.CABLE_REEL_PICKUP: ("depot_world_pose", "cable_reel_world_pose"),
        }
        required = required_by_task.get(task, ())
        if not required:
            return
        if context is None:
            raise ValueError(f"{task.value} requires a ManipulationTaskContext")
        rover_pose = _require_vector(context.rover_world_pose, 7, "rover_world_pose")
        _normalize_quat(rover_pose[3:7], "rover_world_pose quaternion")
        missing = [name for name in required if getattr(context, name) is None]
        if missing:
            raise ValueError(f"{task.value} context missing: {', '.join(missing)}")
        vector_lengths = {
            "depot_world_pose": 7,
            "antenna_world_pose": 7,
            "surveyed_target_world_position": 3,
            "surveyed_surface_normal_world": 3,
            "antenna_port_world_pose": 7,
            "cable_reel_world_pose": 7,
        }
        for name in required:
            value = _require_vector(getattr(context, name), vector_lengths[name], name)
            if vector_lengths[name] == 7:
                _normalize_quat(value[3:7], f"{name} quaternion")
            if name == "surveyed_surface_normal_world" and np.linalg.norm(value) <= 1e-12:
                raise ValueError("surveyed_surface_normal_world must not be a zero vector")

    def get_stow_with_payload_pose(self) -> np.ndarray:
        """Return body-frame stow pose: tucked in, centred, slightly raised."""
        return _pose7(
            x=self._CARRY_X_M,
            y=0.0,
            z=self._CARRY_HEIGHT_M,
            qx=0.0, qy=0.0, qz=0.0, qw=1.0,
        )
