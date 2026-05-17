"""System 7.4: GPS Beacon Network — Pseudo-GNSS Localization.

This module implements a local pseudo-GNSS system using stationary beacon
transmitters (e.g., moonbase antenna or active beacons). Provides position
fixes via trilateration with realistic geometric dilution of precision (GDOP).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class BeaconConfig:
    """Configuration for a single stationary GPS beacon.

    Attributes:
        position_xyz: 3D position [x, y, z] of beacon transmitter in world frame, meters.
        signal_range_m: Maximum line-of-sight signal range in meters.
        power_w: Transmitter power in Watts (affects SNR and ranging accuracy).
        noise_sigma_m: Gaussian noise standard deviation on range measurements in meters.
    """

    position_xyz: NDArray
    signal_range_m: float
    power_w: float
    noise_sigma_m: float


@dataclass
class GPSFix:
    """Position fix result from beacon trilateration.

    Attributes:
        position_xyz: Estimated rover position [x, y, z] in meters.
        gdop: Geometric dilution of precision (1.0-10.0+). Lower is better.
              1.0-5.0 = excellent to good, >10.0 = poor geometry.
        num_beacons: Number of beacons used in solution (typically >= 4 for 3D).
        covariance: 3x3 position covariance matrix (diagonal) representing position uncertainty.
    """

    position_xyz: NDArray
    gdop: float
    num_beacons: int
    covariance: NDArray


class BeaconNetwork(ABC):
    """Abstract base class for beacon-based positioning network.

    Manages a network of stationary beacon transmitters that provide
    pseudo-GNSS fixes via trilateration. Simulates geometric dilution
    of precision and ranging accuracy degradation with distance/obstruction.
    """

    @abstractmethod
    def add_beacon(self, beacon_id: str, config: BeaconConfig) -> None:
        """Register a new beacon transmitter in the network.

        Args:
            beacon_id: Unique identifier string for the beacon (e.g., "moonbase_primary", "antenna_1").
            config: BeaconConfig describing beacon location and signal parameters.

        Raises:
            ValueError: If beacon_id already exists.
        """
        raise NotImplementedError

    @abstractmethod
    def remove_beacon(self, beacon_id: str) -> None:
        """Deactivate and remove a beacon from the network.

        Args:
            beacon_id: Unique identifier of beacon to remove.

        Raises:
            KeyError: If beacon_id not found.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_fix(self, rover_position: NDArray) -> GPSFix | None:
        """Compute position fix using beacons in range.

        Performs trilateration using all beacons within signal range
        of the rover. Returns None if fewer than 1 beacon is visible.

        Args:
            rover_position: Current rover position [x, y, z] in meters.
                            Used to determine which beacons are in range.

        Returns:
            GPSFix object if solution is valid, None if insufficient beacons.
        """
        raise NotImplementedError

    @abstractmethod
    def get_gdop_at(self, position: NDArray) -> float:
        """Query geometric dilution of precision at a given location.

        GDOP measures the contribution of beacon geometry to position error.
        It multiplies the ranging noise to determine overall position uncertainty:
        position_error ≈ ranging_noise * GDOP.

        Args:
            position: Query position [x, y, z] in meters.

        Returns:
            GDOP value (1.0 = ideal geometry, >10.0 = poor geometry).
            Returns inf if fewer than 4 beacons visible at this location.
        """
        raise NotImplementedError

    @abstractmethod
    def get_coverage_map(self, grid_resolution: float) -> NDArray:
        """Generate 2D GDOP map over terrain surface.

        Samples GDOP at regular grid points (at constant altitude/height)
        to visualize positioning quality across traversal area.

        Args:
            grid_resolution: Grid spacing in meters.

        Returns:
            2D array of GDOP values at each grid point. NaN indicates no solution.
        """
        raise NotImplementedError

    @abstractmethod
    def get_visible_beacons(self, position: NDArray) -> list[str]:
        """Return list of beacon IDs visible from a given position.

        Args:
            position: Query position [x, y, z] in meters.

        Returns:
            List of beacon_id strings that are in signal range at this location.
        """
        raise NotImplementedError


class TrilaterationBeaconNetwork(BeaconNetwork):
    """Pseudo-GNSS network solved by iterated least-squares trilateration.

    Each visible beacon yields a noisy pseudorange (true distance plus
    zero-mean Gaussian error with the beacon's own ``noise_sigma_m``). A
    Gauss-Newton solve recovers the rover position; geometry quality is
    reported as GDOP = ``sqrt(trace((HᵀH)⁻¹))`` over the unit line-of-sight
    matrix ``H``. A 3D fix needs at least three in-range beacons; GDOP is only
    defined (finite) with four or more.

    A seeded generator makes the simulated ranging error reproducible.
    """

    MIN_BEACONS_FIX = 3
    MIN_BEACONS_GDOP = 4

    def __init__(self, seed: int = 0) -> None:
        self._beacons: dict[str, BeaconConfig] = {}
        self._rng = np.random.default_rng(seed)

    # -- network management -------------------------------------------
    def add_beacon(self, beacon_id: str, config: BeaconConfig) -> None:
        if beacon_id in self._beacons:
            raise ValueError(f"beacon_id '{beacon_id}' already exists")
        cfg = BeaconConfig(
            position_xyz=np.asarray(config.position_xyz, dtype=np.float64).reshape(3),
            signal_range_m=float(config.signal_range_m),
            power_w=float(config.power_w),
            noise_sigma_m=float(config.noise_sigma_m),
        )
        self._beacons[beacon_id] = cfg

    def remove_beacon(self, beacon_id: str) -> None:
        if beacon_id not in self._beacons:
            raise KeyError(f"beacon_id '{beacon_id}' not found")
        del self._beacons[beacon_id]

    # -- geometry helpers ---------------------------------------------
    def _visible(self, position: NDArray) -> list[str]:
        p = np.asarray(position, dtype=np.float64).reshape(3)
        out = []
        for bid, cfg in self._beacons.items():
            if np.linalg.norm(cfg.position_xyz - p) <= cfg.signal_range_m:
                out.append(bid)
        return out

    def _geometry_matrix(self, position: NDArray, beacon_ids: list[str]) -> NDArray:
        p = np.asarray(position, dtype=np.float64).reshape(3)
        rows = []
        for bid in beacon_ids:
            d = p - self._beacons[bid].position_xyz
            n = np.linalg.norm(d)
            if n < 1e-9:
                n = 1e-9
            rows.append(d / n)
        return np.asarray(rows, dtype=np.float64)

    def _gdop_from(self, h: NDArray) -> float:
        if h.shape[0] < self.MIN_BEACONS_GDOP:
            return float("inf")
        try:
            cov = np.linalg.inv(h.T @ h)
        except np.linalg.LinAlgError:
            return float("inf")
        tr = np.trace(cov)
        return float(np.sqrt(tr)) if tr > 0 else float("inf")

    def get_visible_beacons(self, position: NDArray) -> list[str]:
        return self._visible(position)

    def get_gdop_at(self, position: NDArray) -> float:
        vis = self._visible(position)
        if len(vis) < self.MIN_BEACONS_GDOP:
            return float("inf")
        return self._gdop_from(self._geometry_matrix(position, vis))

    def compute_fix(self, rover_position: NDArray) -> GPSFix | None:
        true_pos = np.asarray(rover_position, dtype=np.float64).reshape(3)
        vis = self._visible(true_pos)
        if len(vis) < self.MIN_BEACONS_FIX:
            return None

        anchors = np.array(
            [self._beacons[b].position_xyz for b in vis], dtype=np.float64
        )
        sigmas = np.array(
            [self._beacons[b].noise_sigma_m for b in vis], dtype=np.float64
        )
        true_ranges = np.linalg.norm(anchors - true_pos, axis=1)
        meas = true_ranges + self._rng.normal(0.0, np.maximum(sigmas, 1e-9))

        # Gauss-Newton from the anchor centroid.
        est = anchors.mean(axis=0) + np.array([0.0, 0.0, 1.0])
        for _ in range(50):
            diff = est - anchors
            pred = np.linalg.norm(diff, axis=1)
            pred = np.where(pred < 1e-9, 1e-9, pred)
            h = diff / pred[:, None]
            residual = meas - pred
            try:
                step, *_ = np.linalg.lstsq(h, residual, rcond=None)
            except np.linalg.LinAlgError:
                break
            est = est + step
            if np.linalg.norm(step) < 1e-6:
                break

        h_final = self._geometry_matrix(est, vis)
        try:
            unit_cov = np.linalg.inv(h_final.T @ h_final)
        except np.linalg.LinAlgError:
            return None
        sigma_sq = float(np.mean(np.maximum(sigmas, 1e-9) ** 2))
        covariance = unit_cov * sigma_sq
        tr = np.trace(unit_cov)
        gdop = (
            float(np.sqrt(tr))
            if (len(vis) >= self.MIN_BEACONS_GDOP and tr > 0)
            else float("inf")
        )
        return GPSFix(
            position_xyz=est.astype(np.float64),
            gdop=gdop,
            num_beacons=len(vis),
            covariance=covariance.astype(np.float64),
        )

    def get_coverage_map(self, grid_resolution: float) -> NDArray:
        if grid_resolution <= 0.0:
            raise ValueError(
                f"grid_resolution must be > 0, got {grid_resolution}"
            )
        if not self._beacons:
            return np.zeros((0, 0), dtype=np.float64)
        pos = np.array(
            [c.position_xyz for c in self._beacons.values()], dtype=np.float64
        )
        margin = max(c.signal_range_m for c in self._beacons.values()) * 0.5
        x_min, y_min = pos[:, 0].min() - margin, pos[:, 1].min() - margin
        x_max, y_max = pos[:, 0].max() + margin, pos[:, 1].max() + margin
        z = float(pos[:, 2].mean())
        xs = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        ys = np.arange(y_min, y_max + grid_resolution, grid_resolution)
        grid = np.full((ys.shape[0], xs.shape[0]), np.nan, dtype=np.float64)
        for j, gy in enumerate(ys):
            for i, gx in enumerate(xs):
                g = self.get_gdop_at(np.array([gx, gy, z]))
                grid[j, i] = g if np.isfinite(g) else np.nan
        return grid
