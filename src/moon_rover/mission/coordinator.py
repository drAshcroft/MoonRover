"""
System 12.2: Multi-Rover Coordination

Abstract interfaces for multi-rover coordination including shared world model
maintenance, cable crossing avoidance, communication modeling, and deadlock
resolution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import numpy.typing as npt


@dataclass
class RoverStatus:
    """
    Current operational status of a single rover.

    Attributes:
        rover_id: Unique rover identifier.
        position: 3-element [x, y, z] position in world frame (meters).
        heading: Rover heading/yaw angle (radians).
        phase: Current mission phase (MissionPhase enum value).
        battery_soc: Battery state of charge [0, 1].
        cable_length_remaining: Remaining cable on spool (meters).
        current_target: Currently assigned GridPoint, or None if none.
    """

    rover_id: str
    position: npt.NDArray[np.float64]
    heading: float
    phase: str
    battery_soc: float
    cable_length_remaining: float
    current_target: Optional[object] = None


class MultiRoverCoordinator(ABC):
    """
    Coordinator for multi-rover missions with shared world modeling.

    Maintains:
    - Shared occupancy map fed by all rovers' sensors
    - Rover position tracking for deadlock/collision detection
    - Cable path tracking to prevent rover-cable collisions
    - Communication delay modeling for realistic constraints
    - Deadlock resolution recommendations

    Uses communication relay via base station (50ms typical delay).
    """

    @abstractmethod
    def register_rover(
        self,
        rover_id: str,
        drive_type: str,
    ) -> None:
        """
        Register a rover with the coordinator.

        Args:
            rover_id: Unique rover identifier.
            drive_type: Rover type: 'diff_drive' or 'skid_steer'.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def update_rover_status(self, status: RoverStatus) -> None:
        """
        Update coordinator's knowledge of rover state.

        Called each time rover transmits position, battery, phase status via
        communication link. Updates shared world model with latest measurements.

        Args:
            status: Current rover operational status.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def get_shared_world_model(self) -> object:
        """
        Retrieve shared occupancy map built from all rover observations.

        Returns unified environmental model incorporating lidar/camera data
        from all rovers fused into single coordinate frame.

        Returns:
            Shared world model (occupancy map or similar).
        """
        raise NotImplementedError

    @abstractmethod
    def check_cable_crossing(
        self,
        rover_id: str,
        planned_path: list[npt.NDArray[np.float64]],
    ) -> bool:
        """
        Check if planned path would cross existing cable deployments.

        Queries known cable positions from other rovers and validates that
        proposed path maintains 1.5 m clearance from all cables.

        Args:
            rover_id: Rover requesting path check.
            planned_path: List of waypoint positions [x, y, z].

        Returns:
            True if path is clear of cable crossings, False if crossing detected.
        """
        raise NotImplementedError

    @abstractmethod
    def get_communication_delay(
        self,
        sender: str,
        receiver: str,
    ) -> float:
        """
        Get communication delay between two rovers via relay.

        Models realistic communication through lunar base relay station. Typical
        delay 50 ms assuming moonbase relay network. Can be asymmetric if one
        rover has poor antenna orientation.

        Args:
            sender: Sending rover ID (or 'base' for moonbase).
            receiver: Receiving rover ID (or 'base').

        Returns:
            One-way communication delay in seconds.
        """
        raise NotImplementedError

    @abstractmethod
    def resolve_deadlock(
        self,
        rover_a: str,
        rover_b: str,
    ) -> dict:
        """
        Generate deadlock resolution strategy for two rovers.

        Computes reassignment of target waypoints to resolve spatial deadlock.
        May involve:
        - Swapping target assignments
        - Introducing wait-at-waypoint instructions
        - Reordering visit sequence

        Args:
            rover_a: First deadlocked rover ID.
            rover_b: Second deadlocked rover ID.

        Returns:
            Dict with keys 'rover_a_targets', 'rover_b_targets' containing
            reassigned GridPoint lists, plus 'priority' indicating which rover
            should yield (move first).
        """
        raise NotImplementedError
