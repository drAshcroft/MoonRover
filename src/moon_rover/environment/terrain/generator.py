"""System 2.1: Terrain Generation Pipeline.

This module provides interfaces for generating realistic lunar terrain with
craters, rocks, rilles, and navigation meshes for pathfinding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class TerrainConfig:
    """Configuration for terrain generation.

    Parameters:
        seed: Random seed for reproducible terrain generation.
        size_m: Terrain size in meters. Typical 100x100 m for rover missions.
        fBm_octaves: Number of Fractal Brownian Motion octaves (8 typical). Higher = more detail.
        fBm_amplitude: Height variation amplitude in meters (0-4 m typical).
        crater_params: Dictionary with crater generation parameters:
            - "count": Number of craters to generate
            - "min_radius_m": Minimum crater radius in meters
            - "max_radius_m": Maximum crater radius in meters
            - "depth_ratio": Depth as fraction of radius
        rock_density: Rock count per square meter. Typical 0.1-0.5.
        rille_enabled: Whether to generate rilles (narrow valleys).
        moonbase_position: 3D position [x, y, z] of moonbase (anchor point).
        resolution: Heightfield resolution in pixels. 512 or 1024 typical.
    """
    seed: int
    size_m: float = 100.0
    fBm_octaves: int = 8
    fBm_amplitude: float = 2.0
    crater_params: dict = None
    rock_density: float = 0.2
    rille_enabled: bool = True
    moonbase_position: Tuple[float, float, float] = (50.0, 50.0, 0.0)
    resolution: int = 512

    def __post_init__(self) -> None:
        """Set default crater parameters if not provided."""
        if self.crater_params is None:
            self.crater_params = {
                "count": 15,
                "min_radius_m": 1.0,
                "max_radius_m": 5.0,
                "depth_ratio": 0.3
            }


@dataclass
class TerrainOutput:
    """Generated terrain data and derived products.

    Attributes:
        height_field: 2D height map (height_field[i, j] is z-coordinate in meters).
            Shape is (resolution, resolution).
        slope_map: Slope magnitude at each point in degrees (0-90). Shape (resolution, resolution).
        normal_map: Per-pixel surface normal vectors. Shape (resolution, resolution, 3).
        rock_positions: List of (x, y, z, radius) tuples for rocks on terrain.
        crater_list: List of crater objects with position, radius, depth.
        nav_mesh: Binary traversability grid (0=impassable, 1=traversable).
            Shape (resolution, resolution). Based on slope and obstacles.
    """
    height_field: NDArray[np.float32]
    slope_map: NDArray[np.float32]
    normal_map: NDArray[np.float32]
    rock_positions: List[Tuple[float, float, float, float]]
    crater_list: List[dict]
    nav_mesh: NDArray[np.uint8]


class TerrainGenerator(ABC):
    """Abstract interface for lunar terrain generation.

    Generates heightfields, craters, rocks, and navigation meshes using
    fractional Brownian motion, crater algorithms, and derived slope/normal maps.
    """

    @abstractmethod
    def generate(self, config: TerrainConfig) -> TerrainOutput:
        """Generate complete terrain from configuration.

        Pipeline:
        1. Generate base heightfield using fBm
        2. Carve craters
        3. Place rocks and boulders
        4. Generate rilles if enabled
        5. Compute slope and normal maps
        6. Build navigation mesh from slope and obstacles

        Parameters:
            config: TerrainConfig with all generation parameters.

        Returns:
            TerrainOutput with heightfield, slopes, normals, rocks, craters, nav_mesh.

        Raises:
            ValueError: If config parameters are out of valid ranges.
        """
        raise NotImplementedError

    @abstractmethod
    def export_genesis_heightfield(self, output: TerrainOutput) -> Any:
        """Convert terrain output to Genesis-compatible heightfield format.

        Genesis uses specific heightfield representations (morphs.Terrain).
        This method provides the conversion.

        Parameters:
            output: TerrainOutput from generate().

        Returns:
            Genesis-compatible heightfield object (gs.morphs.Terrain or similar).

        Raises:
            RuntimeError: If Genesis library is not available.
        """
        raise NotImplementedError

    @abstractmethod
    def export_nav_mesh(self, output: TerrainOutput) -> NDArray[np.uint8]:
        """Export navigation mesh as binary traversability grid.

        The nav mesh combines slope constraints and obstacle detection:
        - Slope > max_traversable_slope -> impassable
        - Proximity to rocks -> impassable
        - Otherwise -> traversable

        Parameters:
            output: TerrainOutput from generate().

        Returns:
            Binary grid (0=impassable, 1=traversable). Shape (resolution, resolution).
        """
        raise NotImplementedError
