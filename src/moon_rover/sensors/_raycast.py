"""Shared scene ray-casting adapter and pose helpers for active sensors.

Active range sensors (LiDAR, stereo camera) need to shoot rays at the world.
The ``scene`` object they are handed is duck-typed so the sensors stay
decoupled from the physics backend and remain unit-testable with light
doubles. A scene is accepted if it provides **either**:

* ``raycast_batch(origins, directions, max_range) -> dict`` returning
  ``{"distances": (N,), "positions": (N, 3), "normals": (N, 3)}`` — the same
  shape as :meth:`GenesisPhysicsEngine.query_raycaster`. Misses are encoded as
  ``inf`` / ``nan`` distance or a ``"hit"`` boolean mask.
* ``get_terrain_height(x, y) -> float`` (optionally
  ``get_terrain_normal(x, y) -> (3,)``) — in which case rays are intersected
  against the heightfield analytically by sphere-marching plus bisection. This
  is what lets the sensors run directly against
  :class:`~moon_rover.core.physics._genesis_engine.GenesisPhysicsEngine`.

All math is plain deterministic NumPy: identical inputs give identical
outputs, which keeps replay bit-stable.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> NDArray:
    """Return the 3x3 rotation matrix for a (w, x, y, z) quaternion."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def decompose_pose(pose: NDArray) -> tuple[NDArray, NDArray]:
    """Split a sensor pose into ``(rotation_3x3, translation_3)``.

    Accepts a 4x4 homogeneous transform or a 7-vector
    ``[x, y, z, qw, qx, qy, qz]``.
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (4, 4):
        return pose[:3, :3].copy(), pose[:3, 3].copy()
    if pose.shape == (7,):
        return quat_to_matrix(pose[3], pose[4], pose[5], pose[6]), pose[:3].copy()
    raise ValueError(
        f"pose must be a 4x4 matrix or a 7-vector [x,y,z,qw,qx,qy,qz], got shape {pose.shape}"
    )


class RaycastAdapter:
    """Uniform ray-cast interface over a duck-typed scene.

    Resolves the scene capability once at construction so the per-frame hot
    path has no branching surprises.
    """

    def __init__(self, scene: Any):
        self._scene = scene
        self._mode: str
        if scene is not None and hasattr(scene, "raycast_batch"):
            self._mode = "batch"
        elif scene is not None and hasattr(scene, "get_terrain_height"):
            self._mode = "terrain"
        else:
            raise TypeError(
                "scene must expose raycast_batch(origins, directions, max_range) "
                "or get_terrain_height(x, y); got "
                f"{type(scene).__name__}"
            )

    def cast(
        self, origins: NDArray, directions: NDArray, max_range: float
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Cast a batch of rays.

        Args:
            origins: (N, 3) ray start points in world frame.
            directions: (N, 3) ray directions in world frame (need not be unit).
            max_range: Rays that do not hit within this distance are misses.

        Returns:
            ``(distances, positions, normals)``:
            - ``distances`` (N,): hit distance, ``inf`` for misses.
            - ``positions`` (N, 3): world hit point (``nan`` for misses).
            - ``normals`` (N, 3): unit surface normal (zeros if unavailable).
        """
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        directions = directions / norms
        if self._mode == "batch":
            return self._cast_batch(origins, directions, max_range)
        return self._cast_terrain(origins, directions, max_range)

    def _cast_batch(
        self, origins: NDArray, directions: NDArray, max_range: float
    ) -> tuple[NDArray, NDArray, NDArray]:
        result = self._scene.raycast_batch(origins, directions, max_range)
        n = origins.shape[0]
        if isinstance(result, dict):
            distances = np.asarray(
                result.get("distances", result.get("distance")), dtype=np.float64
            ).reshape(-1)
            positions = result.get("positions", result.get("points"))
            normals = result.get("normals")
            hit = result.get("hit")
        else:  # tuple/list: (distances, positions[, normals])
            distances = np.asarray(result[0], dtype=np.float64).reshape(-1)
            positions = result[1] if len(result) > 1 else None
            normals = result[2] if len(result) > 2 else None
            hit = None
        miss = ~np.isfinite(distances) | (distances <= 0.0) | (distances > max_range)
        if hit is not None:
            miss |= ~np.asarray(hit, dtype=bool).reshape(-1)
        distances = np.where(miss, np.inf, distances)
        if positions is None:
            positions = origins + directions * np.where(
                miss[:, None], 0.0, distances[:, None]
            )
        positions = np.asarray(positions, dtype=np.float64).reshape(n, 3)
        positions[miss] = np.nan
        if normals is None:
            normals = np.zeros((n, 3), dtype=np.float64)
        else:
            normals = np.asarray(normals, dtype=np.float64).reshape(n, 3)
        return distances, positions, normals

    def _cast_terrain(
        self, origins: NDArray, directions: NDArray, max_range: float
    ) -> tuple[NDArray, NDArray, NDArray]:
        scene = self._scene
        n = origins.shape[0]
        # March all rays together; a hit is a sign change of
        # f(t) = ray_z(t) - terrain_height(ray_xy(t)).
        n_steps = 256
        step = max_range / n_steps
        t_prev = np.zeros(n)
        f_prev = self._terrain_residual(origins)
        distances = np.full(n, np.inf)
        active = np.ones(n, dtype=bool)
        for i in range(1, n_steps + 1):
            t_cur = i * step
            pts = origins + directions * t_cur
            f_cur = self._terrain_residual(pts)
            crossed = active & (f_prev > 0.0) & (f_cur <= 0.0)
            if np.any(crossed):
                idx = np.where(crossed)[0]
                lo = np.full(idx.shape, t_prev[idx])
                hi = np.full(idx.shape, t_cur)
                for _ in range(24):  # bisection refine
                    mid = 0.5 * (lo + hi)
                    fm = self._terrain_residual(
                        origins[idx] + directions[idx] * mid[:, None]
                    )
                    above = fm > 0.0
                    lo = np.where(above, mid, lo)
                    hi = np.where(above, hi, mid)
                distances[idx] = 0.5 * (lo + hi)
                active[idx] = False
            t_prev = np.where(active, t_cur, t_prev)
            f_prev = np.where(active, f_cur, f_prev)
            if not np.any(active):
                break
        miss = ~np.isfinite(distances)
        positions = origins + directions * np.where(miss[:, None], 0.0, distances[:, None])
        positions[miss] = np.nan
        normals = np.zeros((n, 3), dtype=np.float64)
        if hasattr(scene, "get_terrain_normal"):
            for j in np.where(~miss)[0]:
                normals[j] = np.asarray(
                    scene.get_terrain_normal(positions[j, 0], positions[j, 1]),
                    dtype=np.float64,
                )
        return distances, positions, normals

    def _terrain_residual(self, pts: NDArray) -> NDArray:
        """z - terrain_height(x, y) for each point; >0 means above ground."""
        scene = self._scene
        out = np.empty(pts.shape[0])
        for i in range(pts.shape[0]):
            out[i] = pts[i, 2] - scene.get_terrain_height(pts[i, 0], pts[i, 1])
        return out
