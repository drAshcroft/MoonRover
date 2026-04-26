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
