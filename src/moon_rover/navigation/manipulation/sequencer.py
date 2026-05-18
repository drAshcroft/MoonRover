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
    success = sequencer.execute_task(ManipulationTask.ANTENNA_PICKUP, arm)
"""

from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from typing import List


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
                                          (3-element position in world frame)
            surface_normal (np.ndarray): Surface normal at target for orientation [nx, ny, nz]
                                        (unit vector indicating antenna mounting direction)

        Returns:
            List[np.ndarray]: Ordered list of arm end-effector waypoints.
                             Each waypoint is [x, y, z, qx, qy, qz, qw].
                             Sequence: approach -> align -> contact -> release -> retract.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            target = np.array([50.0, 10.0, 0.0])  # 50m east, 10m north, ground level
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
            antenna_port_pose (np.ndarray): Antenna connector port pose
                                            [x, y, z, qx, qy, qz, qw]
                                            (7-element pose in world frame)

        Returns:
            List[np.ndarray]: Ordered list of arm end-effector waypoints.
                             Each waypoint is [x, y, z, qx, qy, qz, qw].
                             Sequence: approach -> align -> contact -> mate -> retract.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            port_pose = np.array([50.0, 10.0, 0.5, 0, 0, 0.707, 0.707])
            waypoints = sequencer.plan_cable_connection(port_pose)
        """
        raise NotImplementedError("plan_cable_connection implementation pending")

    @abstractmethod
    def execute_task(
        self,
        task: ManipulationTask,
        arm: 'ManipulatorArm'
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

        Returns:
            bool: True if task completed successfully, False on failure.
                 Failures may include: planning failure, motion failure,
                 gripper error, or timeout.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            success = sequencer.execute_task(ManipulationTask.ANTENNA_PICKUP, arm)
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
    n = normal / (np.linalg.norm(normal) + 1e-9)
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
    ) -> bool:
        self._current_task = task
        success = False
        for attempt in range(self._MAX_TASK_RETRIES + 1):
            try:
                success = self._run_task(task, arm)
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

    def _run_task(self, task: ManipulationTask, arm: "ManipulatorArm") -> bool:
        if task == ManipulationTask.TRANSPORT_STOW:
            stow = self.get_stow_with_payload_pose()
            return bool(arm.follow_trajectory([stow]))

        if task == ManipulationTask.ANTENNA_PICKUP:
            # Build nominal pick from depot directly in front of rover
            depot = _pose7(0.5, 0.0, -0.3)
            antenna = _pose7(0.5, 0.0, -0.35)
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
            # Placement location assumed to be set by caller via a target pose;
            # use a nominal ground-level target directly forward of the rover.
            target = np.array([0.6, 0.0, 0.0])
            normal = np.array([0.0, 0.0, 1.0])
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
            port = _pose7(0.5, 0.0, 0.15)
            waypoints = self.plan_cable_connection(port)
            return bool(arm.follow_trajectory(waypoints))

        if task == ManipulationTask.CABLE_REEL_PICKUP:
            depot = _pose7(0.4, 0.0, -0.25)
            reel = _pose7(0.4, 0.0, -0.30)
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

    def get_stow_with_payload_pose(self) -> np.ndarray:
        """Return body-frame stow pose: tucked in, centred, slightly raised."""
        return _pose7(
            x=self._CARRY_X_M,
            y=0.0,
            z=self._CARRY_HEIGHT_M,
            qx=0.0, qy=0.0, qz=0.0, qw=1.0,
        )
