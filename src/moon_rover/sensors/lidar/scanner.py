"""System 7.1: LiDAR Scanner — Spinning Multi-Beam Sensor.

This module models a spinning multi-beam LiDAR scanner (e.g., Velodyne VLP-32C).
Generates point clouds in 3D space with range, intensity, and ring channel data.
Supports dust interference for realistic lunar dust accumulation effects.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class LiDARConfig:
    """Configuration for spinning multi-beam LiDAR scanner.

    Attributes:
        num_channels: Number of vertical laser channels (e.g., 32 for VLP-32C).
        h_resolution_deg: Horizontal (azimuth) angular resolution in degrees.
                          Typical: 0.1-0.4 deg. Lower = finer angular resolution.
        elevation_range_deg: Tuple (min_elevation, max_elevation) in degrees.
                             Defines vertical field-of-view span.
                             E.g., (-25, 15) = 40° total vertical FOV.
        max_range_m: Maximum sensing range in meters.
        range_noise_sigma_m: Gaussian noise standard deviation on range measurements in meters.
        intensity_noise_sigma: Gaussian noise standard deviation on intensity (reflectivity) values.
        rotation_rate_hz: Spinning frequency in Hz (e.g., 20 Hz = 50 ms per full rotation).
        min_range_m: Minimum range for valid measurements in meters (closer readings rejected).
        max_returns: Maximum number of returns per laser pulse (typically 1-2).
    """

    num_channels: int
    h_resolution_deg: float
    elevation_range_deg: tuple[float, float]
    max_range_m: float
    range_noise_sigma_m: float
    intensity_noise_sigma: float
    rotation_rate_hz: float
    min_range_m: float
    max_returns: int


@dataclass
class PointCloud:
    """3D point cloud data from LiDAR scan.

    Attributes:
        points: Array of 3D points (N x 3) in sensor frame [x, y, z] in meters.
        intensities: Array of reflectivity/intensity values (N,) in range [0, 1].
                     1.0 = highly reflective, 0.0 = absorptive.
        rings: Array of channel indices (N,) indicating which vertical channel produced each point.
               0 = lowest, num_channels-1 = highest.
        timestamps: Array of measurement timestamps (N,) in seconds relative to scan start.
                    Accounts for spinning during acquisition.
    """

    points: NDArray
    intensities: NDArray
    rings: NDArray
    timestamps: NDArray


class LiDARScanner(ABC):
    """Abstract base class for spinning multi-beam LiDAR sensor.

    Performs 3D range measurements in a sweeping pattern, generating point clouds
    at high frame rate. Simulates realistic effects including:
    - Range noise and minimum range cutoff
    - Intensity (reflectivity) simulation
    - Ring/channel stratification
    - Dust/aerosol interference
    """

    @abstractmethod
    def configure(self, config: LiDARConfig) -> None:
        """Initialize LiDAR with sensor parameters.

        Args:
            config: LiDAR configuration object.

        Raises:
            ValueError: If configuration parameters are invalid (e.g., max_range_m <= min_range_m).
        """
        raise NotImplementedError

    @abstractmethod
    def scan(self, scene: Any, sensor_pose: NDArray) -> PointCloud:
        """Capture a complete 3D point cloud from current sensor pose.

        Performs full 360° rotation scan (or configured FOV) from the given pose.
        Uses scene.raycast_batch() for efficient ray-tracing against terrain/objects.

        Args:
            scene: Scene object with raycast_batch(rays) method supporting batch ray-terrain queries.
            sensor_pose: 4x4 homogeneous transformation matrix of sensor in world frame,
                         or [x, y, z, qw, qx, qy, qz].

        Returns:
            Complete PointCloud with all measurement channels.
        """
        raise NotImplementedError

    @abstractmethod
    def get_partial_scan(self, num_rays: int) -> PointCloud:
        """Return partial scan snapshot for obstacle avoidance tasks.

        Generates a lower-density partial scan at higher frequency (e.g., 240 Hz)
        for real-time reactive navigation without full point cloud computation.

        Args:
            num_rays: Number of horizontal rays in partial scan.
                      Typically 1/3 to 1/2 of full scan density.

        Returns:
            Partial PointCloud with fewer points but same timestamp.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_dust_interference(self, sph_density: float, cloud: PointCloud) -> PointCloud:
        """Simulate dust/aerosol interference on point cloud measurements.

        Lunar dust in air (suspended at landing sites) causes:
        - Increased range noise
        - False returns (dust particles)
        - Intensity attenuation
        - Reduced max range

        Args:
            sph_density: Suspended particle density (mg/m³). 0 = no dust, >100 = heavy dust.
            cloud: Input PointCloud to apply dust effects to.

        Returns:
            Modified PointCloud with dust-induced noise and attenuation.
        """
        raise NotImplementedError
