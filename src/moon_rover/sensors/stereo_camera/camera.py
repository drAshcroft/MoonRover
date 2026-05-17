"""System 7.2: Stereo Camera System.

This module defines the stereo vision sensor for depth estimation, obstacle
detection, and visual navigation. Generates synchronized left/right RGB frames
and computed depth maps with rock detection capability.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class CameraConfig:
    """Configuration for stereo camera pair.

    Attributes:
        resolution: Tuple (width_px, height_px) image resolution in pixels.
        baseline_m: Horizontal distance between left and right camera optical centers in meters.
        fov_h_deg: Horizontal field of view in degrees.
        fov_v_deg: Vertical field of view in degrees.
        focal_length_px: Focal length in pixels (relates to field of view and image resolution).
        frame_rate_hz: Capture frequency in Hz (typically 30 Hz for stereo computation).
        depth_range_m: Tuple (min_depth, max_depth) of valid depth measurements in meters.
        depth_noise_sigma: Standard deviation of Gaussian noise on depth in meters.
    """

    resolution: tuple[int, int]
    baseline_m: float
    fov_h_deg: float
    fov_v_deg: float
    focal_length_px: float
    frame_rate_hz: float
    depth_range_m: tuple[float, float]
    depth_noise_sigma: float
    seed: int = 0


@dataclass
class StereoFrame:
    """Synchronized stereo image pair with computed depth map.

    Attributes:
        left_rgb: Left camera RGB image as HxWx3 uint8 array.
        right_rgb: Right camera RGB image as HxWx3 uint8 array (same shape as left_rgb).
        depth_map: Computed stereo depth map as HxW float32 array in meters.
                   0.0 or NaN indicates invalid/no-match pixels.
        timestamp: Frame timestamp in seconds (relative to simulation start).
    """

    left_rgb: NDArray
    right_rgb: NDArray
    depth_map: NDArray
    timestamp: float


class StereoCamera(ABC):
    """Abstract base class for stereo vision sensor.

    Provides synchronized stereo image pairs and stereo-computed depth maps
    for 3D perception, obstacle detection, and visual navigation.
    """

    @abstractmethod
    def configure(self, config: CameraConfig) -> None:
        """Initialize stereo camera system with parameters.

        Args:
            config: Stereo camera configuration object.

        Raises:
            ValueError: If configuration parameters are invalid (e.g., baseline_m <= 0).
        """
        raise NotImplementedError

    @abstractmethod
    def capture(self, scene: Any, camera_pose: NDArray) -> StereoFrame:
        """Capture stereo image pair and compute depth map.

        Renders left and right views from scene at given pose, performs stereo
        matching (correlation) to compute depth map.

        Args:
            scene: Scene object with rendering capability.
            camera_pose: 4x4 homogeneous transformation matrix of camera in world frame,
                         or [x, y, z, qw, qx, qy, qz].

        Returns:
            StereoFrame with left/right RGB and depth map.
        """
        raise NotImplementedError

    @abstractmethod
    def detect_rocks(self, depth_map: NDArray) -> list[dict]:
        """Detect rock obstacles in depth map.

        Uses connected component labeling on depth discontinuities to identify
        individual rocks/obstacles. Each detection includes bounding box, area, and
        estimated 3D position.

        Args:
            depth_map: Depth map from capture() or external source (HxW float32).

        Returns:
            List of rock detections, each a dict with:
            - "bbox": Bounding box [x_min, y_min, x_max, y_max] in pixels.
            - "area_px": Connected component area in pixels.
            - "center_3d": Estimated 3D position [x, y, z] in camera frame.
            - "height_m": Estimated rock height above surrounding terrain.
        """
        raise NotImplementedError

    @abstractmethod
    def get_navcam_frame(self, scene: Any, navcam_pose: NDArray) -> StereoFrame:
        """Capture navigation camera view (downward-pointing stereo pair).

        Acquires stereo images from a downward-facing camera mount (typically 45° below
        horizontal) for close-range ground obstacle detection and safe foothold validation.

        Args:
            scene: Scene object with rendering capability.
            navcam_pose: 4x4 transformation matrix of navcam in world frame.

        Returns:
            StereoFrame from downward-looking stereo pair.
        """
        raise NotImplementedError


class RaycastStereoCamera(StereoCamera):
    """Stereo camera that renders by per-pixel ray-casting the scene.

    Each pixel of the left and right cameras is a ray cast against the scene
    (see :class:`~moon_rover.sensors._raycast.RaycastAdapter`). The hit range
    is projected onto the optical axis to form a ground-truth depth map; an
    RGB image is synthesised by Lambertian shading of the hit normal so depth
    discontinuities and terrain relief are visible. Gaussian depth noise scaled
    by the configured ``depth_noise_sigma`` is applied within the valid depth
    band; pixels with no hit or out-of-band depth are ``nan``.

    Camera frame: +x optical axis (forward), +y left, +z up. The stereo pair
    is separated along the camera y-axis by ``baseline_m``.
    """

    def __init__(self) -> None:
        self._config: CameraConfig | None = None
        self._rng: np.random.Generator | None = None
        self._dirs_cam: NDArray | None = None  # (H, W, 3) unit rays, optical +x
        self._frame_count: int = 0

    # -- configuration -------------------------------------------------
    def configure(self, config: CameraConfig) -> None:
        w, h = config.resolution
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {config.resolution}")
        if config.baseline_m <= 0.0:
            raise ValueError(f"baseline_m must be > 0, got {config.baseline_m}")
        if config.focal_length_px <= 0.0:
            raise ValueError(
                f"focal_length_px must be > 0, got {config.focal_length_px}"
            )
        lo, hi = config.depth_range_m
        if hi <= lo or lo < 0.0:
            raise ValueError(
                f"depth_range_m must be (min, max) with 0 <= min < max, got {(lo, hi)}"
            )
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._frame_count = 0

        f = config.focal_length_px
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        # Optical axis +x; image x -> -y (left positive), image y -> -z (up positive).
        dx = np.ones_like(us, dtype=np.float64)
        dy = (cx - us) / f
        dz = (cy - vs) / f
        dirs = np.stack([dx, dy, dz], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
        self._dirs_cam = dirs  # (H, W, 3)

    def _require_config(self) -> CameraConfig:
        if self._config is None or self._rng is None:
            raise RuntimeError("configure() must be called before use")
        return self._config

    # -- rendering -----------------------------------------------------
    def _render_one(
        self, scene: Any, rot: NDArray, origin: NDArray
    ) -> tuple[NDArray, NDArray]:
        """Render a single camera. Returns (rgb uint8 HxWx3, depth float HxW)."""
        from moon_rover.sensors._raycast import RaycastAdapter

        cfg = self._config
        assert cfg is not None and self._dirs_cam is not None
        w, h = cfg.resolution
        dirs_cam = self._dirs_cam.reshape(-1, 3)
        dirs_world = dirs_cam @ rot.T
        origins = np.broadcast_to(origin, dirs_world.shape)
        forward = rot[:, 0]  # camera +x in world

        adapter = RaycastAdapter(scene)
        dist, pos, nrm = adapter.cast(origins, dirs_world, cfg.depth_range_m[1] * 2.0)

        # Depth = projection of the hit vector onto the optical axis.
        depth = dist * (dirs_world @ forward)
        lo, hi = cfg.depth_range_m
        valid = np.isfinite(depth) & (depth >= lo) & (depth <= hi)

        # Lambertian shade from a fixed key light; flat-shade misses to sky.
        light = np.array([0.3, 0.2, 0.93])
        light = light / np.linalg.norm(light)
        nmag = np.linalg.norm(nrm, axis=1)
        shade = np.full(depth.shape, 0.15)
        has_n = nmag > 1e-6
        if np.any(has_n):
            unit_n = nrm[has_n] / nmag[has_n, None]
            shade[has_n] = 0.2 + 0.8 * np.clip(unit_n @ light, 0.0, 1.0)
        # Where no normal but a valid hit (batch backends without normals),
        # fall back to a range-shaded grey so structure is still visible.
        no_n = valid & ~has_n
        if np.any(no_n):
            shade[no_n] = np.clip(1.0 - (depth[no_n] - lo) / (hi - lo), 0.1, 1.0)
        shade[~valid] = 0.08  # background / sky

        rng = self._rng
        assert rng is not None
        gray = np.clip(shade * 255.0, 0, 255)
        gray += rng.normal(0.0, 1.5, size=gray.shape)
        rgb = np.clip(gray, 0, 255).astype(np.uint8)
        rgb = np.repeat(rgb.reshape(h, w, 1), 3, axis=2)

        depth_out = np.full(depth.shape, np.nan)
        d_valid = depth[valid]
        if cfg.depth_noise_sigma > 0.0 and d_valid.size > 0:
            # Stereo error grows with depth (triangulation), so scale noise.
            scale = 1.0 + (d_valid - lo) / max(hi - lo, 1e-6)
            d_valid = d_valid + rng.normal(0.0, cfg.depth_noise_sigma, d_valid.size) * scale
        depth_out[valid] = d_valid
        return rgb, depth_out.reshape(h, w).astype(np.float32)

    def _camera_axes(self, pose: NDArray) -> tuple[NDArray, NDArray]:
        from moon_rover.sensors._raycast import decompose_pose

        rot, trans = decompose_pose(pose)
        return rot, trans

    def _capture_pair(self, scene: Any, pose: NDArray) -> StereoFrame:
        cfg = self._require_config()
        rot, trans = self._camera_axes(pose)
        left_axis = rot[:, 1]  # camera +y points left
        half = 0.5 * cfg.baseline_m
        left_origin = trans + left_axis * half
        right_origin = trans - left_axis * half

        left_rgb, depth = self._render_one(scene, rot, left_origin)
        right_rgb, _ = self._render_one(scene, rot, right_origin)
        ts = self._frame_count / cfg.frame_rate_hz if cfg.frame_rate_hz > 0 else 0.0
        self._frame_count += 1
        return StereoFrame(
            left_rgb=left_rgb,
            right_rgb=right_rgb,
            depth_map=depth,
            timestamp=ts,
        )

    # -- public API ----------------------------------------------------
    def capture(self, scene: Any, camera_pose: NDArray) -> StereoFrame:
        return self._capture_pair(scene, camera_pose)

    def get_navcam_frame(self, scene: Any, navcam_pose: NDArray) -> StereoFrame:
        """Capture from a stereo pair pitched 45° downward from ``navcam_pose``."""
        from moon_rover.sensors._raycast import decompose_pose

        rot, trans = decompose_pose(navcam_pose)
        pitch = np.radians(45.0)  # nose-down about camera +y (left) axis
        cp, sp = np.cos(pitch), np.sin(pitch)
        # Rotate optical axis (+x) down toward -z.
        r_pitch = np.array(
            [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64
        )
        tilted = np.eye(4)
        tilted[:3, :3] = rot @ r_pitch
        tilted[:3, 3] = trans
        return self._capture_pair(scene, tilted)

    def detect_rocks(self, depth_map: NDArray) -> list[dict]:
        cfg = self._require_config()
        depth_map = np.asarray(depth_map, dtype=np.float64)
        h, w = depth_map.shape
        valid = np.isfinite(depth_map)
        if valid.sum() < 9:
            return []

        # A rock protrudes: its measured depth is closer than the locally
        # smoothed background depth. Build the background with a separable
        # box mean over valid pixels, then flag negative residuals.
        filled = np.where(valid, depth_map, 0.0)
        k = max(3, (min(h, w) // 12) | 1)  # odd window
        bg = _box_mean(filled, valid.astype(np.float64), k)
        residual = bg - depth_map  # >0 where closer than background
        thresh = max(0.05, 3.0 * cfg.depth_noise_sigma)
        mask = valid & (residual > thresh)

        labels, n = _connected_components(mask)
        f = cfg.focal_length_px
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rocks: list[dict] = []
        for lbl in range(1, n + 1):
            ys, xs = np.where(labels == lbl)
            if xs.size < 6:  # reject speckle
                continue
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cu, cv = float(xs.mean()), float(ys.mean())
            z = float(np.median(depth_map[ys, xs]))  # optical-axis depth
            # Back-project centroid (camera frame: +x fwd, +y left, +z up).
            px = z
            py = (cx - cu) / f * z
            pz = (cy - cv) / f * z
            rocks.append(
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "area_px": int(xs.size),
                    "center_3d": np.array([px, py, pz], dtype=np.float64),
                    "height_m": float(np.clip(residual[ys, xs].max(), 0.0, None)),
                }
            )
        rocks.sort(key=lambda r: r["area_px"], reverse=True)
        return rocks


def _box_mean(values: NDArray, weights: NDArray, k: int) -> NDArray:
    """Weighted box mean with window ``k`` (NaN-safe via the weight mask)."""
    pad = k // 2
    vp = np.pad(values, pad, mode="edge")
    wp = np.pad(weights, pad, mode="edge")
    csum_v = np.cumsum(np.cumsum(vp, axis=0), axis=1)
    csum_w = np.cumsum(np.cumsum(wp, axis=0), axis=1)

    def _window(c: NDArray) -> NDArray:
        c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
        h, w = values.shape
        ys = np.arange(h)
        xs = np.arange(w)
        y0, y1 = ys, ys + k
        x0, x1 = xs, xs + k
        a = c[np.ix_(y1, x1)]
        b = c[np.ix_(y0, x1)]
        cc = c[np.ix_(y1, x0)]
        d = c[np.ix_(y0, x0)]
        return a - b - cc + d

    sv = _window(csum_v)
    sw = _window(csum_w)
    return np.where(sw > 1e-9, sv / np.maximum(sw, 1e-9), values)


def _connected_components(mask: NDArray) -> tuple[NDArray, int]:
    """4-connected labeling via iterative flood fill (no SciPy dependency)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx] != 0:
                continue
            cur += 1
            stack = [(sy, sx)]
            labels[sy, sx] = cur
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and labels[ny, nx] == 0
                    ):
                        labels[ny, nx] = cur
                        stack.append((ny, nx))
    return labels, cur
