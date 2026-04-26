"""
System 11.3: Global Path Planning

Abstract path planning system supporting both A* and D* Lite algorithms for
static and dynamic environments. Integrates traversability maps, cable avoidance,
and energy-aware routing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class PlannerConfig:
    """
    Configuration for global path planner.

    Attributes:
        grid_resolution_m: Resolution of planning grid in meters. Default: 0.50 m.
        algorithm: Planning algorithm choice. Options: 'a_star' or 'd_star_lite'.
            Default: 'a_star'.
        heading_change_penalty: Cost penalty for direction changes. Higher values
            favor smoother paths. Default: 0.1.
        cable_clearance_m: Minimum clearance distance from known cables (meters).
            Default: depends on cable map buffer.
        replan_trigger_distance_m: Distance traveled before replanning is triggered
            if new obstacles detected. Default: 5.0 m.
    """

    grid_resolution_m: float = 0.50
    algorithm: str = "a_star"
    heading_change_penalty: float = 0.1
    cable_clearance_m: float = 1.5
    replan_trigger_distance_m: float = 5.0


@dataclass
class PlannedPath:
    """
    Result of path planning query.

    Attributes:
        waypoints: List of Nx3 arrays, each element [x, y, z] (meters).
        total_distance_m: Total path length in meters.
        estimated_cable_drag_energy: Estimated energy dissipated by cable drag
            along path (Joules). Used for battery-aware planning.
        risk_score: Aggregate risk metric [0, 1] accounting for traversability,
            cable proximity, and path stability.
    """

    waypoints: list[npt.NDArray[np.float64]]
    total_distance_m: float
    estimated_cable_drag_energy: float
    risk_score: float


class GlobalPathPlanner(ABC):
    """
    Abstract base class for global path planning.

    Handles long-horizon path planning from start to goal while respecting
    traversability constraints, cable avoidance, and energy considerations.
    Supports both static planning (A*) and dynamic replanning (D* Lite).
    """

    @abstractmethod
    def configure(self, config: PlannerConfig) -> None:
        """
        Configure the path planner with algorithm parameters.

        Args:
            config: Planner configuration specifying algorithm, resolution, etc.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def plan(
        self,
        start: npt.NDArray[np.float64],
        goal: npt.NDArray[np.float64],
        traversability_map: object,
        cable_map: object,
    ) -> PlannedPath:
        """
        Compute global path from start to goal.

        Searches for optimal path respecting traversability constraints and
        cable exclusion zones. If no feasible path exists, raises ValueError.

        Args:
            start: 3-element [x, y, z] start position (meters).
            goal: 3-element [x, y, z] goal position (meters).
            traversability_map: Traversability map providing cost field.
            cable_map: Cable map for exclusion zone queries.

        Returns:
            PlannedPath with waypoints, distance, energy estimate, and risk.

        Raises:
            ValueError: If no feasible path exists.
        """
        raise NotImplementedError

    @abstractmethod
    def replan(
        self,
        current_pos: npt.NDArray[np.float64],
        new_obstacles: list[npt.NDArray[np.float64]],
    ) -> PlannedPath:
        """
        Incremental replanning using D* Lite when environment changes.

        Uses D* Lite algorithm to efficiently replan when new obstacles are
        detected, without recomputing entire search space. Falls back to full
        replanning if environment changes are substantial.

        Args:
            current_pos: Current rover position [x, y, z] (meters).
            new_obstacles: List of newly detected obstacle regions (Nx3 arrays).

        Returns:
            Replanned path from current position to original goal.

        Raises:
            ValueError: If no feasible path to goal exists.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cable_aware_cost(
        self,
        path: list[npt.NDArray[np.float64]],
        cable_positions: npt.NDArray[np.float64],
    ) -> float:
        """
        Compute cable drag energy cost for a candidate path.

        Estimates energy dissipated by cable drag along path based on:
        - Cable length deployed
        - Cable tension as function of distance from spool
        - Terrain slope along path

        Used for multi-objective optimization balancing path length vs. energy.

        Args:
            path: Sequence of waypoint positions [x, y, z].
            cable_positions: Known deployed cable positions (Nx3 array).

        Returns:
            Estimated cable drag energy in Joules.
        """
        raise NotImplementedError
