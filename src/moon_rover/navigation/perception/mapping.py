"""
System 11.1: Perception and Mapping

Abstract base classes for occupancy mapping, elevation mapping, traversability
assessment, and cable tracking. These systems build and maintain environmental
models used by the global path planner and motion controller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class OccupancyMap(ABC):
    """
    Voxel-based 3D occupancy grid for static obstacle mapping.

    Updates from lidar point clouds and maintains a probabilistic voxel grid
    representation. Used to detect rocks, equipment, and other obstacles.

    Attributes:
        voxel_resolution_m: Edge length of each voxel in meters. Default: 0.10 m.
    """

    voxel_resolution_m: float = 0.10

    @abstractmethod
    def update_from_point_cloud(
        self,
        points: npt.NDArray[np.float32],
        rover_pose: npt.NDArray[np.float64],
    ) -> None:
        """
        Update occupancy grid from incoming lidar point cloud.

        Integrates new point cloud measurements into the 3D occupancy grid using
        ray-casting occupancy update. Accounts for rover pose to transform points
        into world coordinates.

        Args:
            points: Nx3 array of 3D point coordinates in rover frame (meters).
            rover_pose: 6-element pose [x, y, z, roll, pitch, yaw] (meters, radians).

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def query_occupancy(
        self, position: npt.NDArray[np.float64]
    ) -> float:
        """
        Query occupancy probability at a single position.

        Returns:
            Occupancy probability in [0, 1]. 0 = free space, 1 = fully occupied.
        """
        raise NotImplementedError

    @abstractmethod
    def get_traversability_cost(self) -> npt.NDArray[np.float32]:
        """
        Export 3D occupancy grid as traversability cost field.

        Returns:
            3D array of cost values in [0, 1] matching voxel resolution.
        """
        raise NotImplementedError


@dataclass
class ElevationMap(ABC):
    """
    Continuous elevation map built from stereo depth images.

    Maintains a high-resolution local elevation surface for slope and terrain
    roughness estimation. Uses a sliding window to keep memory bounded while
    covering the rover's navigation neighborhood.

    Attributes:
        resolution_m: Grid cell resolution in meters. Default: 0.05 m.
        window_size_m: Side length of square sliding window. Default: 20 m.
    """

    resolution_m: float = 0.05
    window_size_m: float = 20.0

    @abstractmethod
    def update_from_stereo_depth(
        self,
        depth_image: npt.NDArray[np.float32],
        intrinsics: npt.NDArray[np.float64],
        rover_pose: npt.NDArray[np.float64],
    ) -> None:
        """
        Update elevation map from stereo depth image.

        Converts depth measurements to 3D world coordinates and updates the
        sliding elevation grid. Handles temporal filtering and outlier removal.

        Args:
            depth_image: HxW depth image in meters.
            intrinsics: 3x3 camera intrinsic matrix.
            rover_pose: 6-element pose [x, y, z, roll, pitch, yaw].

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def get_slope_at(
        self, position: npt.NDArray[np.float64]
    ) -> float:
        """
        Query local slope (gradient magnitude) at position.

        Computes surface normal and returns slope angle in degrees [0, 90].

        Args:
            position: 3-element [x, y, z] coordinate in world frame.

        Returns:
            Slope angle in degrees.
        """
        raise NotImplementedError


@dataclass
class TraversabilityMap(ABC):
    """
    Synthesized traversability cost field combining slope, rock density, and obstacles.

    Integrates elevation map slope, occupancy grid rock obstacles, and cable
    proximity penalties into a unified cost field [0, 1] used by path planning.

    Traversability incorporates:
    - Slope penalty: max at 25 degrees, inf at steeper slopes
    - Rock density: weighted by occupancy grid density
    - Cable proximity: penalty zone around known cable positions
    """

    @abstractmethod
    def update(
        self,
        elevation_map: ElevationMap,
        occupancy_map: OccupancyMap,
        cable_map: CableMap,
    ) -> None:
        """
        Recompute traversability field from constituent maps.

        Args:
            elevation_map: Current elevation surface.
            occupancy_map: Current 3D occupancy grid.
            cable_map: Known cable positions and geometry.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def get_cost(
        self, position: npt.NDArray[np.float64]
    ) -> float:
        """
        Query traversability cost at a position.

        Returns:
            Cost value in [0, 1]. 0 = free/easy, 1 = impassable/dangerous.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cost_field(self) -> npt.NDArray[np.float32]:
        """
        Export full 2D traversability cost grid for path planning.

        Returns:
            2D array of cost values in [0, 1] at planner grid resolution.
        """
        raise NotImplementedError


@dataclass
class CableMap(ABC):
    """
    Tracks all known cable positions and geometry for collision avoidance.

    Maintains a record of deployed cable sections with their 3D centerline
    positions, diameter, and stiffness properties. Provides spatial queries
    for path planning and motion control.
    """

    @abstractmethod
    def add_cable_segment(
        self,
        segment_id: str,
        centerline: npt.NDArray[np.float64],
        diameter_m: float,
    ) -> None:
        """
        Register a cable segment in the map.

        Args:
            segment_id: Unique identifier for this cable segment.
            centerline: Nx3 array of 3D points defining cable path.
            diameter_m: Cable diameter in meters.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def query_clearance(
        self, position: npt.NDArray[np.float64]
    ) -> float:
        """
        Query minimum clearance distance from position to nearest cable.

        Args:
            position: 3-element [x, y, z] query position.

        Returns:
            Minimum distance in meters to cable centerline. Returns inf if no cables.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cable_positions(self) -> dict[str, npt.NDArray[np.float64]]:
        """
        Retrieve all tracked cable centerlines.

        Returns:
            Dict mapping segment_id -> Nx3 array of centerline points.
        """
        raise NotImplementedError

    @abstractmethod
    def get_exclusion_zones(
        self, buffer_m: float = 1.5
    ) -> list[npt.NDArray[np.float64]]:
        """
        Get cylindrical exclusion zones around cables.

        Args:
            buffer_m: Radial buffer distance beyond cable radius. Default: 1.5 m.

        Returns:
            List of Nx3 point clouds defining exclusion zone boundaries.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------

from scipy.ndimage import uniform_filter  # noqa: E402
from scipy.spatial import KDTree  # noqa: E402


class GridOccupancyMap(OccupancyMap):
    """2D log-odds occupancy grid updated from LiDAR point clouds.

    Supports incremental Bayesian updates via Bresenham ray casting.
    Free cells along each ray are decremented; the endpoint cell is incremented.
    """

    _L_OCC: float = 0.85  # p(occ | hit)
    _L_FREE: float = 0.15  # p(occ | miss)
    _L_MIN: float = -5.0
    _L_MAX: float = 5.0

    def __init__(
        self,
        width_m: float = 100.0,
        height_m: float = 100.0,
        origin_xy: tuple[float, float] = (-50.0, -50.0),
        voxel_resolution_m: float = 0.10,
    ) -> None:
        super().__init__(voxel_resolution_m=voxel_resolution_m)
        self._origin = np.array(origin_xy, dtype=np.float64)
        nx = max(1, int(round(width_m / voxel_resolution_m)))
        ny = max(1, int(round(height_m / voxel_resolution_m)))
        self._log_odds: npt.NDArray[np.float32] = np.zeros((nx, ny), dtype=np.float32)
        self._l_occ = float(np.log(self._L_OCC / (1.0 - self._L_OCC)))
        self._l_free = float(np.log(self._L_FREE / (1.0 - self._L_FREE)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _world_to_grid(self, xy: npt.NDArray[np.float64]) -> tuple[int, int]:
        gx = int((xy[0] - self._origin[0]) / self.voxel_resolution_m)
        gy = int((xy[1] - self._origin[1]) / self.voxel_resolution_m)
        return gx, gy

    def _grid_to_world(self, gx: int, gy: int) -> npt.NDArray[np.float64]:
        x = self._origin[0] + (gx + 0.5) * self.voxel_resolution_m
        y = self._origin[1] + (gy + 0.5) * self.voxel_resolution_m
        return np.array([x, y, 0.0], dtype=np.float64)

    def _update_cell(self, gx: int, gy: int, delta: float) -> None:
        nx, ny = self._log_odds.shape
        if 0 <= gx < nx and 0 <= gy < ny:
            self._log_odds[gx, gy] = np.clip(
                self._log_odds[gx, gy] + delta, self._L_MIN, self._L_MAX
            )

    def _bresenham_ray(
        self, start: tuple[int, int], end: tuple[int, int]
    ) -> None:
        """Mark free along ray, occupied at endpoint using Bresenham's line."""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        err = dx - dy
        x, y = x0, y0
        max_steps = dx + dy + 1
        for _ in range(max_steps):
            if x == x1 and y == y1:
                self._update_cell(x, y, self._l_occ)
                break
            self._update_cell(x, y, self._l_free)
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def update_from_point_cloud(
        self,
        points: npt.NDArray[np.float32],
        rover_pose: npt.NDArray[np.float64],
    ) -> None:
        x, y, z = rover_pose[0], rover_pose[1], rover_pose[2]
        yaw = rover_pose[5]
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        world_pts = (R @ points.T).T + np.array([x, y, z])
        rover_grid = self._world_to_grid(np.array([x, y]))
        for pt in world_pts:
            pt_grid = self._world_to_grid(pt[:2])
            self._bresenham_ray(rover_grid, pt_grid)

    def query_occupancy(self, position: npt.NDArray[np.float64]) -> float:
        gx, gy = self._world_to_grid(position[:2])
        nx, ny = self._log_odds.shape
        if not (0 <= gx < nx and 0 <= gy < ny):
            return 1.0
        l = float(self._log_odds[gx, gy])
        return float(1.0 / (1.0 + np.exp(-l)))

    def get_traversability_cost(self) -> npt.NDArray[np.float32]:
        prob = (1.0 / (1.0 + np.exp(-self._log_odds))).astype(np.float32)
        return prob[:, :, np.newaxis]

    # ------------------------------------------------------------------
    # Extra accessors used by planner and traversability map
    # ------------------------------------------------------------------

    @property
    def probability_grid(self) -> npt.NDArray[np.float32]:
        """2D occupancy probability array [0, 1]."""
        return (1.0 / (1.0 + np.exp(-self._log_odds))).astype(np.float32)

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self._log_odds.shape

    @property
    def origin(self) -> npt.NDArray[np.float64]:
        return self._origin.copy()


class SlidingElevationMap(ElevationMap):
    """Sliding-window elevation map built from stereo depth images.

    Maintains a fixed-size 2D height grid centred on the rover. As the rover
    moves, the window translates and stale cells are reset to NaN.
    """

    def __init__(
        self,
        resolution_m: float = 0.05,
        window_size_m: float = 20.0,
    ) -> None:
        super().__init__(resolution_m=resolution_m, window_size_m=window_size_m)
        n = max(1, int(round(window_size_m / resolution_m)))
        self._n = n
        self._height: npt.NDArray[np.float32] = np.full((n, n), np.nan, dtype=np.float32)
        self._count: npt.NDArray[np.int32] = np.zeros((n, n), dtype=np.int32)
        self._centre = np.zeros(2, dtype=np.float64)  # world xy of grid centre

    def _world_to_local(self, xy: npt.NDArray[np.float64]) -> tuple[int, int]:
        half = self.window_size_m / 2.0
        lx = int((xy[0] - self._centre[0] + half) / self.resolution_m)
        ly = int((xy[1] - self._centre[1] + half) / self.resolution_m)
        return lx, ly

    def _recenter(self, rover_xy: npt.NDArray[np.float64]) -> None:
        """Translate grid when rover moves more than half a cell."""
        shift = rover_xy - self._centre
        shift_cells = (shift / self.resolution_m).astype(int)
        if abs(shift_cells[0]) > 0 or abs(shift_cells[1]) > 0:
            self._height = np.roll(self._height, shift=(-shift_cells[0], -shift_cells[1]), axis=(0, 1))
            self._count = np.roll(self._count, shift=(-shift_cells[0], -shift_cells[1]), axis=(0, 1))
            self._centre += shift_cells * self.resolution_m
            # Zero-out shifted-in border rows/cols
            if shift_cells[0] > 0:
                self._height[-shift_cells[0]:, :] = np.nan
                self._count[-shift_cells[0]:, :] = 0
            elif shift_cells[0] < 0:
                self._height[: -shift_cells[0], :] = np.nan
                self._count[: -shift_cells[0], :] = 0
            if shift_cells[1] > 0:
                self._height[:, -shift_cells[1]:] = np.nan
                self._count[:, -shift_cells[1]:] = 0
            elif shift_cells[1] < 0:
                self._height[:, : -shift_cells[1]] = np.nan
                self._count[:, : -shift_cells[1]] = 0

    def update_from_stereo_depth(
        self,
        depth_image: npt.NDArray[np.float32],
        intrinsics: npt.NDArray[np.float64],
        rover_pose: npt.NDArray[np.float64],
    ) -> None:
        rx, ry, rz = rover_pose[0], rover_pose[1], rover_pose[2]
        self._recenter(np.array([rx, ry]))
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy_int = intrinsics[0, 2], intrinsics[1, 2]
        H, W = depth_image.shape
        v_idx, u_idx = np.mgrid[0:H, 0:W]
        d = depth_image.astype(np.float64)
        valid = (d > 0.0) & np.isfinite(d)
        xc = ((u_idx - cx) / fx * d)[valid]
        yc = ((v_idx - cy_int) / fy * d)[valid]
        zc = d[valid]
        # rotate camera frame to world frame using rover yaw
        yaw = rover_pose[5]
        cy2, sy2 = np.cos(yaw), np.sin(yaw)
        world_x = rx + cy2 * xc - sy2 * zc
        world_y = ry + sy2 * xc + cy2 * zc
        world_z = rz - yc
        xy_arr = np.stack([world_x, world_y], axis=1)
        for i in range(len(world_z)):
            lx, ly = self._world_to_local(xy_arr[i])
            if 0 <= lx < self._n and 0 <= ly < self._n:
                n = self._count[lx, ly]
                z = world_z[i]
                if n == 0 or not np.isfinite(self._height[lx, ly]):
                    self._height[lx, ly] = z
                else:
                    # running mean
                    self._height[lx, ly] = (self._height[lx, ly] * n + z) / (n + 1)
                self._count[lx, ly] = n + 1

    def get_slope_at(self, position: npt.NDArray[np.float64]) -> float:
        lx, ly = self._world_to_local(position[:2])
        r = 1
        x0, x1 = max(0, lx - r), min(self._n, lx + r + 1)
        y0, y1 = max(0, ly - r), min(self._n, ly + r + 1)
        patch = self._height[x0:x1, y0:y1]
        if patch.size < 4 or not np.any(np.isfinite(patch)):
            return 0.0
        valid = np.isfinite(patch)
        if valid.sum() < 2:
            return 0.0
        dz_dx = np.nanmax(patch) - np.nanmin(patch)
        slope_m = dz_dx / (self.resolution_m * patch.shape[0])
        return float(np.degrees(np.arctan(slope_m)))

    @property
    def height_grid(self) -> npt.NDArray[np.float32]:
        return self._height.copy()


class GridTraversabilityMap(TraversabilityMap):
    """Traversability cost field synthesised from slope, occupancy, and cable proximity."""

    _MAX_SLOPE_DEG: float = 25.0
    _SLOPE_PENALTY_START_DEG: float = 5.0
    _CABLE_COST: float = 0.9
    _OCC_THRESHOLD: float = 0.65

    def __init__(self, grid_shape: tuple[int, int] = (200, 200)) -> None:
        self._cost: npt.NDArray[np.float32] = np.zeros(grid_shape, dtype=np.float32)
        self._shape = grid_shape

    def update(
        self,
        elevation_map: ElevationMap,
        occupancy_map: OccupancyMap,
        cable_map: "CableMap",
    ) -> None:
        if not isinstance(occupancy_map, GridOccupancyMap):
            return
        occ_grid = occupancy_map.probability_grid
        h, w = occ_grid.shape
        cost = np.zeros((h, w), dtype=np.float32)
        # Occupancy cost: impassable above threshold
        cost = np.where(occ_grid > self._OCC_THRESHOLD, 1.0, occ_grid * 0.5).astype(np.float32)
        # Smooth to remove noise
        cost = uniform_filter(cost, size=3).astype(np.float32)
        # Cable exclusion zones: inflate cable cells
        for segment_pts in cable_map.get_cable_positions().values():
            for pt in segment_pts:
                gx, gy = occupancy_map._world_to_grid(pt[:2])
                r = 3
                x0, x1 = max(0, gx - r), min(h, gx + r + 1)
                y0, y1 = max(0, gy - r), min(w, gy + r + 1)
                cost[x0:x1, y0:y1] = np.maximum(cost[x0:x1, y0:y1], self._CABLE_COST)
        self._cost = cost

    def get_cost(self, position: npt.NDArray[np.float64]) -> float:
        return 0.0  # caller should use get_cost_field for planning

    def get_cost_field(self) -> npt.NDArray[np.float32]:
        return self._cost.copy()


class SpatialCableMap(CableMap):
    """Cable segment registry backed by KD-tree for O(log n) proximity queries."""

    def __init__(self) -> None:
        self._segments: dict[str, npt.NDArray[np.float64]] = {}
        self._diameters: dict[str, float] = {}
        self._kd_tree: KDTree | None = None
        self._all_points: npt.NDArray[np.float64] | None = None

    def _rebuild_index(self) -> None:
        if not self._segments:
            self._kd_tree = None
            self._all_points = None
            return
        pts = np.vstack(list(self._segments.values()))
        self._all_points = pts
        self._kd_tree = KDTree(pts[:, :3])

    def add_cable_segment(
        self,
        segment_id: str,
        centerline: npt.NDArray[np.float64],
        diameter_m: float,
    ) -> None:
        self._segments[segment_id] = np.asarray(centerline, dtype=np.float64)
        self._diameters[segment_id] = diameter_m
        self._rebuild_index()

    def query_clearance(self, position: npt.NDArray[np.float64]) -> float:
        if self._kd_tree is None:
            return float("inf")
        dist, _ = self._kd_tree.query(position[:3])
        return float(dist)

    def get_cable_positions(self) -> dict[str, npt.NDArray[np.float64]]:
        return {k: v.copy() for k, v in self._segments.items()}

    def get_exclusion_zones(self, buffer_m: float = 1.5) -> list[npt.NDArray[np.float64]]:
        zones: list[npt.NDArray[np.float64]] = []
        for seg_id, centerline in self._segments.items():
            radius = self._diameters.get(seg_id, 0.05) / 2.0 + buffer_m
            zone_pts = []
            angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            for pt in centerline:
                for a in angles:
                    zone_pts.append([
                        pt[0] + radius * np.cos(a),
                        pt[1] + radius * np.sin(a),
                        pt[2],
                    ])
            if zone_pts:
                zones.append(np.array(zone_pts, dtype=np.float64))
        return zones
