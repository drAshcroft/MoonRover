"""
System 12: Mission Management System

High-level mission orchestration managing rover deployment sequences, antenna
placement grid assignments, fault detection and recovery, and multi-rover
coordination for the lunar cable deployment mission.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt


class MissionPhase(Enum):
    """
    Rover mission state machine phases.

    Attributes:
        PLANNING: Initial phase, no commands issued.
        DEPOT_PICKUP: At base station, deploying from depot.
        TRANSIT: En route to antenna placement location.
        ANTENNA_DEPLOY: Arrived at grid point, antenna placement underway.
        CABLE_CONNECT: Connecting cable to placed antenna (manual/teleop step).
        RETURN_TO_BASE: Returning to depot after mission segment.
        CHARGING: At base, charging batteries.
        FAULT_RECOVERY: Executing fault mitigation procedure.
        COMPLETE: Mission segment complete, awaiting next assignment.
    """

    PLANNING = "planning"
    DEPOT_PICKUP = "depot_pickup"
    TRANSIT = "transit"
    ANTENNA_DEPLOY = "antenna_deploy"
    CABLE_CONNECT = "cable_connect"
    RETURN_TO_BASE = "return_to_base"
    CHARGING = "charging"
    FAULT_RECOVERY = "fault_recovery"
    COMPLETE = "complete"


class FaultType(Enum):
    """
    Enumeration of mission-critical faults.

    Attributes:
        ANTENNA_TILT: Antenna placement arm unable to achieve target orientation.
        CABLE_SNAG: Cable snagged on terrain feature, tension spike.
        CABLE_EXHAUSTED: Entire spool length deployed (end-of-mission normal).
        ROVER_STUCK: Mobility system unable to move (rock obstruction, high-centering).
        BATTERY_LOW: Battery SOC below safe operating threshold.
        GPS_LOST: No GPS fix for extended period, localization unreliable.
        MOTOR_OVERHEAT: Drive or arm motor temperature exceeds limits.
    """

    ANTENNA_TILT = "antenna_tilt"
    CABLE_SNAG = "cable_snag"
    CABLE_EXHAUSTED = "cable_exhausted"
    ROVER_STUCK = "rover_stuck"
    BATTERY_LOW = "battery_low"
    GPS_LOST = "gps_lost"
    MOTOR_OVERHEAT = "motor_overheat"


@dataclass
class GridPoint:
    """
    Single antenna deployment location on lunar surface grid.

    Attributes:
        position_xyz: 3-element [x, y, z] position in world coordinates (meters).
        row: Row index in deployment grid.
        col: Column index in deployment grid.
        assigned_rover: Rover ID assigned to this point, or None if unassigned.
        status: Current status: 'unvisited', 'visited', 'equipped', 'complete'.
        antenna_id: ID of antenna placed at this location, or None.
    """

    position_xyz: npt.NDArray[np.float64]
    row: int
    col: int
    assigned_rover: Optional[str] = None
    status: str = "unvisited"
    antenna_id: Optional[str] = None


@dataclass
class MissionConfig:
    """
    Configuration for mission planning and execution.

    Attributes:
        grid_origin: 3-element [x, y, z] origin of antenna deployment grid.
        grid_rows: Number of rows in antenna grid.
        grid_cols: Number of columns in antenna grid.
        grid_spacing_m: Spacing between grid points in meters.
        num_rovers: Number of rovers participating in mission.
        max_retries: Maximum retry attempts for failed mission segments. Default: 3.
    """

    grid_origin: npt.NDArray[np.float64]
    grid_rows: int
    grid_cols: int
    grid_spacing_m: float
    num_rovers: int
    max_retries: int = 3


class MissionManager(ABC):
    """
    Central mission management system for multi-rover antenna deployment.

    Handles:
    - Deployment sequence optimization to minimize cable routing complexity
    - Rover-to-grid-point assignment balancing workload
    - Phase tracking and advancement for each rover
    - Fault detection and recovery strategy selection
    - Multi-rover deadlock detection and resolution
    - Cable exclusion zone maintenance for collision avoidance
    """

    @abstractmethod
    def initialize(self, config: MissionConfig) -> None:
        """
        Initialize mission manager with configuration.

        Args:
            config: Mission configuration with grid layout and rover count.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def plan_deployment_order(self) -> list[GridPoint]:
        """
        Determine optimal deployment order for antenna placement.

        Optimizes sequence to minimize cable length and routing complexity by
        computing a spanning tree or greedy nearest-neighbor traversal. First
        point placed should be close to depot.

        Returns:
            Ordered list of grid points for deployment sequence.
        """
        raise NotImplementedError

    @abstractmethod
    def assign_rovers(
        self,
        rover_ids: list[str],
    ) -> dict[str, list[GridPoint]]:
        """
        Assign rovers to subsets of grid points for parallel deployment.

        Distributes grid points among rovers to balance workload and minimize
        cross-rover cable interference. Uses assignment algorithms (Hungarian,
        greedy nearest-neighbor) to optimize total distance.

        Args:
            rover_ids: List of rover identifiers.

        Returns:
            Dict mapping rover_id -> list of assigned GridPoints.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_phase(self, rover_id: str) -> MissionPhase:
        """
        Query current mission phase for a rover.

        Args:
            rover_id: Rover identifier.

        Returns:
            Current MissionPhase for this rover.
        """
        raise NotImplementedError

    @abstractmethod
    def advance_phase(self, rover_id: str) -> MissionPhase:
        """
        Advance rover to next mission phase.

        Performs phase transition including state sanity checks. For example,
        cannot advance from ANTENNA_DEPLOY to CABLE_CONNECT until arm reports
        successful placement.

        Args:
            rover_id: Rover identifier.

        Returns:
            New MissionPhase after advancement.

        Raises:
            RuntimeError: If advancement preconditions not met.
        """
        raise NotImplementedError

    @abstractmethod
    def detect_fault(
        self,
        rover_id: str,
        telemetry: dict,
    ) -> Optional[FaultType]:
        """
        Detect mission-critical faults from rover telemetry.

        Monitors sensor values and state diagnostics to identify fault
        conditions. Uses thresholds and heuristics (e.g., cable tension
        spikes, temperature excursions, GPS outages).

        Args:
            rover_id: Rover identifier.
            telemetry: Telemetry dict with keys like 'cable_tension_n',
                'motor_temp_c', 'gps_fix_quality', 'battery_soc', etc.

        Returns:
            FaultType if fault detected, None otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def execute_recovery(
        self,
        rover_id: str,
        fault: FaultType,
    ) -> bool:
        """
        Execute fault recovery procedure for specified fault type.

        Selects and executes appropriate recovery action:
        - CABLE_SNAG: Reverse briefly, then retry
        - ROVER_STUCK: Deploy emergency wiggle maneuver, or request rescue
        - BATTERY_LOW: Abort mission, return to depot immediately
        - GPS_LOST: Switch to dead reckoning, slow speed, request assistance
        - MOTOR_OVERHEAT: Pause, cool down, resume

        Args:
            rover_id: Rover identifier.
            fault: Fault type to recover from.

        Returns:
            True if recovery successful, False if recovery exhausted.
        """
        raise NotImplementedError

    @abstractmethod
    def get_grid_status(self) -> list[GridPoint]:
        """
        Retrieve current status of all grid points.

        Returns:
            List of all GridPoints with current assignments and status.
        """
        raise NotImplementedError

    @abstractmethod
    def check_deadlock(
        self,
        rover_positions: dict[str, npt.NDArray[np.float64]],
    ) -> list[tuple[str, str]]:
        """
        Detect and report deadlocked rover pairs.

        Identifies rovers that are in spatial deadlock (e.g., blocking each
        other's planned paths) and unable to progress. Returns pairs that
        require intervention or replanning.

        Args:
            rover_positions: Dict mapping rover_id -> [x, y, z] position.

        Returns:
            List of (rover_id_a, rover_id_b) deadlock pairs.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cable_exclusion_zones(self) -> list[npt.NDArray[np.float64]]:
        """
        Get cylindrical exclusion zones around all deployed cable segments.

        Returns zones as point clouds or implicit cylinder representations
        for path planning collision checks.

        Returns:
            List of Nx3 point clouds defining cable exclusion boundaries
            (1.5 m buffer around cable centerline).
        """
        raise NotImplementedError
