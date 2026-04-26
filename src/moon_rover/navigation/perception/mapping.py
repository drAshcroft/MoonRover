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
