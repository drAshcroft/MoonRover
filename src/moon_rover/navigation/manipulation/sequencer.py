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
