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


# ---------------------------------------------------------------------------
# Concrete implementation: A* + D* Lite on a discrete grid
# ---------------------------------------------------------------------------

import heapq  # noqa: E402
from typing import Iterator  # noqa: E402


def _grid_neighbours(
    node: tuple[int, int],
    shape: tuple[int, int],
    diagonal: bool = True,
) -> Iterator[tuple[tuple[int, int], float]]:
    """Yield (neighbour, move_cost) for a 2-D grid node."""
    x, y = node
    cardinals = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for dx, dy in cardinals:
        nx2, ny2 = x + dx, y + dy
        if 0 <= nx2 < shape[0] and 0 <= ny2 < shape[1]:
            yield (nx2, ny2), 1.0
    if diagonal:
        for dx, dy in diagonals:
            nx2, ny2 = x + dx, y + dy
            if 0 <= nx2 < shape[0] and 0 <= ny2 < shape[1]:
                yield (nx2, ny2), np.sqrt(2.0)


class AStarDStarPathPlanner(GlobalPathPlanner):
    """Grid-based global path planner supporting A* (static) and D* Lite (dynamic).

    Internally maintains a cost grid derived from the traversability map and
    cable exclusion zones. Planning is done in grid coordinates; waypoints are
    returned in world coordinates.

    D* Lite re-uses the previously computed cost values and only re-expands
    cells whose edge costs have changed, making incremental replanning efficient.
    """

    def __init__(self) -> None:
        self._config: PlannerConfig | None = None
        self._cost_grid: npt.NDArray[np.float32] | None = None
        self._grid_shape: tuple[int, int] = (200, 200)
        self._origin: npt.NDArray[np.float64] = np.zeros(2)
        self._resolution: float = 0.5
        # D* Lite state
        self._g: dict[tuple[int, int], float] = {}
        self._rhs: dict[tuple[int, int], float] = {}
        self._open_heap: list = []
        self._km: float = 0.0
        self._goal_grid: tuple[int, int] | None = None
        self._last_start_grid: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, world: npt.NDArray[np.float64]) -> tuple[int, int]:
        gx = int((world[0] - self._origin[0]) / self._resolution)
        gy = int((world[1] - self._origin[1]) / self._resolution)
        h, w = self._grid_shape
        return max(0, min(gx, h - 1)), max(0, min(gy, w - 1))

    def _grid_to_world(self, node: tuple[int, int]) -> npt.NDArray[np.float64]:
        x = self._origin[0] + (node[0] + 0.5) * self._resolution
        y = self._origin[1] + (node[1] + 0.5) * self._resolution
        return np.array([x, y, 0.0], dtype=np.float64)

    def _edge_cost(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        if self._cost_grid is None:
            return 1.0
        cx = float(self._cost_grid[b[0], b[1]])
        if cx >= 1.0:
            return float("inf")
        move = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        penalty = 1.0 + cx * 5.0
        return move * penalty

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _heading_penalty(self, prev: tuple[int, int], curr: tuple[int, int], nxt: tuple[int, int]) -> float:
        if self._config is None:
            return 0.0
        v1 = (curr[0] - prev[0], curr[1] - prev[1])
        v2 = (nxt[0] - curr[0], nxt[1] - curr[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = np.sqrt(v1[0] ** 2 + v1[1] ** 2) + 1e-9
        mag2 = np.sqrt(v2[0] ** 2 + v2[1] ** 2) + 1e-9
        cos_a = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        turn_angle = np.arccos(cos_a)
        return self._config.heading_change_penalty * turn_angle

    # ------------------------------------------------------------------
    # A* search
    # ------------------------------------------------------------------

    def _astar(
        self,
        start_g: tuple[int, int],
        goal_g: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        open_heap: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, start_g))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_g: None}
        g_score: dict[tuple[int, int], float] = {start_g: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current == goal_g:
                return self._reconstruct_path(came_from, goal_g)
            for nb, _ in _grid_neighbours(current, self._grid_shape):
                ec = self._edge_cost(current, nb)
                if ec == float("inf"):
                    continue
                tentative = g_score[current] + ec
                # Add heading change penalty when we have a predecessor
                prev = came_from.get(current)
                if prev is not None:
                    tentative += self._heading_penalty(prev, current, nb)
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    f = tentative + self._heuristic(nb, goal_g)
                    heapq.heappush(open_heap, (f, nb))
        return None

    @staticmethod
    def _reconstruct_path(
        came_from: dict[tuple[int, int], tuple[int, int] | None],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
        node: tuple[int, int] | None = goal
        while node is not None:
            path.append(node)
            node = came_from.get(node)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # D* Lite helpers (Koenig & Likhachev 2002)
    # ------------------------------------------------------------------

    def _dsl_key(self, s: tuple[int, int], s_start: tuple[int, int]) -> tuple[float, float]:
        g = self._g.get(s, float("inf"))
        rhs = self._rhs.get(s, float("inf"))
        k2 = min(g, rhs)
        k1 = k2 + self._heuristic(s_start, s) + self._km
        return (k1, k2)

    def _dsl_update_vertex(self, u: tuple[int, int], s_start: tuple[int, int]) -> None:
        if u != self._goal_grid:
            succ_costs = [
                self._g.get(nb, float("inf")) + self._edge_cost(u, nb)
                for nb, _ in _grid_neighbours(u, self._grid_shape)
            ]
            self._rhs[u] = min(succ_costs) if succ_costs else float("inf")
        if self._g.get(u, float("inf")) != self._rhs.get(u, float("inf")):
            k = self._dsl_key(u, s_start)
            heapq.heappush(self._open_heap, (k, u))

    def _dsl_compute_shortest_path(self, s_start: tuple[int, int]) -> None:
        while self._open_heap:
            k_old, u = heapq.heappop(self._open_heap)
            k_new = self._dsl_key(u, s_start)
            if k_old < k_new:
                heapq.heappush(self._open_heap, (k_new, u))
                continue
            g_u = self._g.get(u, float("inf"))
            rhs_u = self._rhs.get(u, float("inf"))
            k_start = self._dsl_key(s_start, s_start)
            g_start = self._g.get(s_start, float("inf"))
            rhs_start = self._rhs.get(s_start, float("inf"))
            if k_old > k_start and min(g_start, rhs_start) == rhs_start:
                break
            if g_u > rhs_u:
                self._g[u] = rhs_u
                for nb, _ in _grid_neighbours(u, self._grid_shape):
                    self._dsl_update_vertex(nb, s_start)
            else:
                self._g[u] = float("inf")
                self._dsl_update_vertex(u, s_start)
                for nb, _ in _grid_neighbours(u, self._grid_shape):
                    self._dsl_update_vertex(nb, s_start)

    def _dsl_extract_path(
        self, s_start: tuple[int, int]
    ) -> list[tuple[int, int]] | None:
        path = [s_start]
        current = s_start
        max_steps = self._grid_shape[0] * self._grid_shape[1]
        for _ in range(max_steps):
            if current == self._goal_grid:
                return path
            candidates = [
                (self._g.get(nb, float("inf")) + self._edge_cost(current, nb), nb)
                for nb, _ in _grid_neighbours(current, self._grid_shape)
            ]
            if not candidates:
                return None
            best_cost, best_nb = min(candidates, key=lambda x: x[0])
            if best_cost == float("inf"):
                return None
            path.append(best_nb)
            current = best_nb
        return None

    def _init_dstar(self, start_g: tuple[int, int], goal_g: tuple[int, int]) -> None:
        self._g = {}
        self._rhs = {}
        self._open_heap = []
        self._km = 0.0
        self._goal_grid = goal_g
        self._last_start_grid = start_g
        self._rhs[goal_g] = 0.0
        heapq.heappush(self._open_heap, (self._dsl_key(goal_g, start_g), goal_g))
        self._dsl_compute_shortest_path(start_g)

    # ------------------------------------------------------------------
    # Cost grid construction
    # ------------------------------------------------------------------

    def _build_cost_grid(
        self, traversability_map: object, cable_map: object
    ) -> npt.NDArray[np.float32]:
        from moon_rover.navigation.perception.mapping import GridTraversabilityMap, SpatialCableMap

        grid = np.zeros(self._grid_shape, dtype=np.float32)
        if isinstance(traversability_map, GridTraversabilityMap):
            field = traversability_map.get_cost_field()
            h = min(field.shape[0], self._grid_shape[0])
            w = min(field.shape[1], self._grid_shape[1])
            grid[:h, :w] = field[:h, :w]

        if isinstance(cable_map, SpatialCableMap) and self._config is not None:
            clearance = self._config.cable_clearance_m
            for pts in cable_map.get_cable_positions().values():
                for pt in pts:
                    gx, gy = self._world_to_grid(pt[:2])
                    r = max(1, int(clearance / self._resolution))
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            if dx ** 2 + dy ** 2 <= r ** 2:
                                nx2 = gx + dx
                                ny2 = gy + dy
                                if 0 <= nx2 < self._grid_shape[0] and 0 <= ny2 < self._grid_shape[1]:
                                    grid[nx2, ny2] = max(grid[nx2, ny2], 0.95)
        return grid

    # ------------------------------------------------------------------
    # GlobalPathPlanner interface
    # ------------------------------------------------------------------

    def configure(self, config: PlannerConfig) -> None:
        self._config = config
        self._resolution = config.grid_resolution_m

    def plan(
        self,
        start: npt.NDArray[np.float64],
        goal: npt.NDArray[np.float64],
        traversability_map: object,
        cable_map: object,
    ) -> PlannedPath:
        if self._config is None:
            self.configure(PlannerConfig())

        self._cost_grid = self._build_cost_grid(traversability_map, cable_map)
        start_g = self._world_to_grid(start)
        goal_g = self._world_to_grid(goal)
        self._goal_grid = goal_g  # store for replan() regardless of algorithm

        algo = self._config.algorithm if self._config else "a_star"
        if algo == "d_star_lite":
            self._init_dstar(start_g, goal_g)
            grid_path = self._dsl_extract_path(start_g)
        else:
            grid_path = self._astar(start_g, goal_g)

        if grid_path is None:
            raise ValueError(f"No feasible path from {start} to {goal}")

        waypoints = [self._grid_to_world(node) for node in grid_path]
        total_dist = sum(
            float(np.linalg.norm(waypoints[i + 1][:2] - waypoints[i][:2]))
            for i in range(len(waypoints) - 1)
        )
        cable_energy = self.get_cable_aware_cost(waypoints, np.empty((0, 3)))
        risk = float(np.mean([
            float(self._cost_grid[g[0], g[1]]) for g in grid_path
        ]))
        return PlannedPath(
            waypoints=waypoints,
            total_distance_m=total_dist,
            estimated_cable_drag_energy=cable_energy,
            risk_score=risk,
        )

    def replan(
        self,
        current_pos: npt.NDArray[np.float64],
        new_obstacles: list[npt.NDArray[np.float64]],
    ) -> PlannedPath:
        if self._config is None or self._cost_grid is None or self._goal_grid is None:
            raise ValueError("Planner not initialised — call plan() before replan()")

        # Inflate new obstacles in cost grid
        for obs_pts in new_obstacles:
            for pt in obs_pts:
                gx, gy = self._world_to_grid(pt[:2])
                r = 2
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        nx2, ny2 = gx + dx, gy + dy
                        if 0 <= nx2 < self._grid_shape[0] and 0 <= ny2 < self._grid_shape[1]:
                            old = float(self._cost_grid[nx2, ny2])
                            self._cost_grid[nx2, ny2] = min(1.0, old + 0.5)

        start_g = self._world_to_grid(current_pos)
        goal_g = self._goal_grid

        if self._config.algorithm == "d_star_lite" and self._last_start_grid is not None:
            self._km += self._heuristic(self._last_start_grid, start_g)
            self._last_start_grid = start_g
            for obs_pts in new_obstacles:
                for pt in obs_pts:
                    gx, gy = self._world_to_grid(pt[:2])
                    self._dsl_update_vertex((gx, gy), start_g)
            self._dsl_compute_shortest_path(start_g)
            grid_path = self._dsl_extract_path(start_g)
        else:
            grid_path = self._astar(start_g, goal_g)

        if grid_path is None:
            raise ValueError("No feasible path after replanning")

        waypoints = [self._grid_to_world(node) for node in grid_path]
        total_dist = sum(
            float(np.linalg.norm(waypoints[i + 1][:2] - waypoints[i][:2]))
            for i in range(len(waypoints) - 1)
        )
        cable_energy = self.get_cable_aware_cost(waypoints, np.empty((0, 3)))
        risk = float(np.mean([float(self._cost_grid[g[0], g[1]]) for g in grid_path]))
        return PlannedPath(
            waypoints=waypoints,
            total_distance_m=total_dist,
            estimated_cable_drag_energy=cable_energy,
            risk_score=risk,
        )

    def get_cable_aware_cost(
        self,
        path: list[npt.NDArray[np.float64]],
        cable_positions: npt.NDArray[np.float64],
    ) -> float:
        if len(path) < 2:
            return 0.0
        # Estimate cable drag energy: tension proportional to deployed length,
        # modulated by terrain slope proxy (delta-z along path).
        total = 0.0
        cable_len = float(np.linalg.norm(cable_positions)) if cable_positions.size else 0.0
        base_tension_n = max(10.0, cable_len * 0.5)  # N, rough linear model
        for i in range(len(path) - 1):
            seg = path[i + 1] - path[i]
            ds = float(np.linalg.norm(seg[:2]))
            dz = float(seg[2]) if len(seg) > 2 else 0.0
            slope_factor = 1.0 + max(0.0, dz / (ds + 1e-9))  # upslope increases drag
            total += base_tension_n * ds * slope_factor
        return total
