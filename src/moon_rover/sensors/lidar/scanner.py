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
    dropout_probability: float = 0.0
    seed: int = 0


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


class GenesisLiDARScanner(LiDARScanner):
    """Spinning multi-beam LiDAR backed by scene ray-casting.

    Casts ``num_channels`` × ``azimuth_steps`` rays per full revolution against
    the supplied scene (see :class:`~moon_rover.sensors._raycast.RaycastAdapter`
    for the accepted scene protocol), then applies a Velodyne-class sensor
    model: Gaussian range noise, minimum/maximum range gating, random per-shot
    dropout, Lambertian intensity with range falloff, ring stratification, and
    azimuth-dependent acquisition timestamps.

    Determinism: a single seeded ``numpy`` generator drives all stochastic
    effects, so identical (scene, pose) inputs reproduce identical clouds.
    """

    def __init__(self) -> None:
        self._config: LiDARConfig | None = None
        self._rng: np.random.Generator | None = None
        self._elevations_rad: NDArray | None = None
        self._azimuths_rad: NDArray | None = None
        self._last_scene: Any = None
        self._last_pose: NDArray | None = None
        self._scan_count: int = 0

    # -- configuration -------------------------------------------------
    def configure(self, config: LiDARConfig) -> None:
        if config.num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {config.num_channels}")
        if config.h_resolution_deg <= 0.0:
            raise ValueError(
                f"h_resolution_deg must be > 0, got {config.h_resolution_deg}"
            )
        if config.max_range_m <= config.min_range_m:
            raise ValueError(
                f"max_range_m ({config.max_range_m}) must exceed "
                f"min_range_m ({config.min_range_m})"
            )
        if config.min_range_m < 0.0:
            raise ValueError(f"min_range_m must be >= 0, got {config.min_range_m}")
        if not 0.0 <= config.dropout_probability <= 1.0:
            raise ValueError(
                f"dropout_probability must be in [0, 1], got {config.dropout_probability}"
            )
        lo, hi = config.elevation_range_deg
        if hi < lo:
            raise ValueError(
                f"elevation_range_deg must be (min, max) with max >= min, got {(lo, hi)}"
            )
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._scan_count = 0
        if config.num_channels == 1:
            self._elevations_rad = np.radians(np.array([0.5 * (lo + hi)]))
        else:
            self._elevations_rad = np.radians(
                np.linspace(lo, hi, config.num_channels)
            )
        n_az = max(1, int(round(360.0 / config.h_resolution_deg)))
        self._azimuths_rad = np.radians(
            np.linspace(0.0, 360.0, n_az, endpoint=False)
        )

    # -- internal helpers ---------------------------------------------
    def _require_config(self) -> LiDARConfig:
        if self._config is None or self._rng is None:
            raise RuntimeError("configure() must be called before use")
        return self._config

    def _ray_dirs_sensor(self, azimuths: NDArray) -> tuple[NDArray, NDArray]:
        """Build (rays, 3) unit directions and (rays,) ring indices.

        Sensor frame: +x forward, +y left, +z up. Rays are ordered
        channel-major then azimuth so timestamps grow monotonically.
        """
        el = self._elevations_rad
        az_grid, el_grid = np.meshgrid(azimuths, el, indexing="ij")
        ce = np.cos(el_grid)
        dirs = np.stack(
            [ce * np.cos(az_grid), ce * np.sin(az_grid), np.sin(el_grid)], axis=-1
        ).reshape(-1, 3)
        rings = np.broadcast_to(
            np.arange(el.shape[0]), az_grid.shape
        ).reshape(-1)
        return dirs, rings

    def _build_cloud(
        self,
        scene: Any,
        pose: NDArray,
        azimuths: NDArray,
    ) -> PointCloud:
        from moon_rover.sensors._raycast import RaycastAdapter, decompose_pose

        cfg = self._require_config()
        rng = self._rng
        assert rng is not None
        rot, trans = decompose_pose(pose)
        dirs_local, rings_all = self._ray_dirs_sensor(azimuths)
        dirs_world = dirs_local @ rot.T
        origins = np.broadcast_to(trans, dirs_world.shape)

        adapter = RaycastAdapter(scene)
        distances, positions, normals = adapter.cast(
            origins, dirs_world, cfg.max_range_m
        )

        n_az = azimuths.shape[0]
        n_ch = self._elevations_rad.shape[0]
        # Azimuth fraction -> acquisition time within one revolution.
        rev_period = 1.0 / cfg.rotation_rate_hz if cfg.rotation_rate_hz > 0 else 0.0
        az_frac = (azimuths % (2 * np.pi)) / (2 * np.pi)
        ts_all = np.repeat(az_frac * rev_period, n_ch)

        valid = np.isfinite(distances)
        noisy = distances.copy()
        if cfg.range_noise_sigma_m > 0.0:
            noisy[valid] += rng.normal(
                0.0, cfg.range_noise_sigma_m, size=int(valid.sum())
            )
        valid &= noisy >= cfg.min_range_m
        valid &= noisy <= cfg.max_range_m
        if cfg.dropout_probability > 0.0:
            keep = rng.random(noisy.shape) >= cfg.dropout_probability
            valid &= keep

        idx = np.where(valid)[0]
        pts = dirs_local[idx] * noisy[idx, None]

        # Lambertian reflectivity with inverse-square range falloff.
        base_albedo = 0.5
        cos_inc = np.ones(idx.shape[0])
        nrm = normals[idx]
        nmag = np.linalg.norm(nrm, axis=1)
        has_n = nmag > 1e-6
        if np.any(has_n):
            d = dirs_world[idx][has_n]
            cos_inc[has_n] = np.clip(
                np.abs(np.sum(-d * (nrm[has_n] / nmag[has_n, None]), axis=1)),
                0.05,
                1.0,
            )
        falloff = np.clip(
            1.0 - (noisy[idx] / cfg.max_range_m) ** 2 * 0.5, 0.1, 1.0
        )
        intensities = base_albedo * cos_inc * falloff
        if cfg.intensity_noise_sigma > 0.0:
            intensities = intensities + rng.normal(
                0.0, cfg.intensity_noise_sigma, size=idx.shape[0]
            )
        intensities = np.clip(intensities, 0.0, 1.0)

        self._last_scene = scene
        self._last_pose = np.asarray(pose, dtype=np.float64)
        return PointCloud(
            points=pts.astype(np.float64),
            intensities=intensities.astype(np.float64),
            rings=rings_all[idx].astype(np.int32),
            timestamps=ts_all[idx].astype(np.float64),
        )

    # -- public API ----------------------------------------------------
    def scan(self, scene: Any, sensor_pose: NDArray) -> PointCloud:
        cfg = self._require_config()
        self._scan_count += 1
        return self._build_cloud(scene, sensor_pose, self._azimuths_rad)

    def get_partial_scan(self, num_rays: int) -> PointCloud:
        cfg = self._require_config()
        if self._last_scene is None or self._last_pose is None:
            raise RuntimeError(
                "get_partial_scan() requires a prior scan(); call scan() first"
            )
        if num_rays < 1:
            raise ValueError(f"num_rays must be >= 1, got {num_rays}")
        full = self._azimuths_rad
        n = min(num_rays, full.shape[0])
        sel = np.linspace(0, full.shape[0] - 1, n).astype(int)
        return self._build_cloud(self._last_scene, self._last_pose, full[sel])

    def apply_dust_interference(
        self, sph_density: float, cloud: PointCloud
    ) -> PointCloud:
        cfg = self._require_config()
        rng = self._rng
        assert rng is not None
        if sph_density <= 0.0 or cloud.points.shape[0] == 0:
            return cloud
        ranges = np.linalg.norm(cloud.points, axis=1)
        # Beer-Lambert transmittance; extinction grows with particle density.
        ext_coeff = 4.0e-4 * sph_density  # 1/m at given mg/m^3
        transmittance = np.exp(-ext_coeff * ranges)

        # Heavily attenuated returns are lost entirely.
        survive = rng.random(ranges.shape) < np.clip(transmittance + 0.05, 0.0, 1.0)
        # Extra range scatter scales with optical depth.
        extra_sigma = cfg.range_noise_sigma_m * (1.0 + sph_density / 50.0)
        jitter = rng.normal(0.0, extra_sigma, size=cloud.points.shape) * (
            1.0 - transmittance[:, None]
        )
        pts = cloud.points + jitter
        intensities = cloud.intensities * transmittance

        pts = pts[survive]
        intensities = intensities[survive]
        rings = cloud.rings[survive]
        timestamps = cloud.timestamps[survive]

        # Suspended particles produce sparse near-field false returns.
        n_false = rng.poisson(sph_density / 40.0)
        if n_false > 0:
            fr = rng.uniform(cfg.min_range_m, min(15.0, cfg.max_range_m), n_false)
            faz = rng.uniform(0.0, 2 * np.pi, n_false)
            fel = rng.uniform(-0.2, 0.2, n_false)
            fpts = np.stack(
                [
                    fr * np.cos(fel) * np.cos(faz),
                    fr * np.cos(fel) * np.sin(faz),
                    fr * np.sin(fel),
                ],
                axis=-1,
            )
            pts = np.vstack([pts, fpts])
            intensities = np.concatenate(
                [intensities, rng.uniform(0.0, 0.15, n_false)]
            )
            rings = np.concatenate(
                [rings, rng.integers(0, cfg.num_channels, n_false).astype(np.int32)]
            )
            timestamps = np.concatenate(
                [timestamps, rng.uniform(0.0, timestamps.max(initial=0.0) + 1e-9, n_false)]
            )

        return PointCloud(
            points=pts,
            intensities=np.clip(intensities, 0.0, 1.0),
            rings=rings,
            timestamps=timestamps,
        )
