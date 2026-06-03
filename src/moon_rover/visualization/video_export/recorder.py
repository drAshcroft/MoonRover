"""Video recording orchestration for Genesis scenes.

This module turns a stepping physics engine into an MP4 demo. It owns a single
offscreen camera and drives the engine's camera API across the scene lifecycle:

    recorder.attach(engine)   # CONSTRUCTION — add the camera before build_scene()
    recorder.start()          # SIMULATION  — begin accumulating frames
    recorder.capture()        # SIMULATION  — once per frame while stepping
    recorder.close()          # SIMULATION  — encode + write the .mp4

The camera can either hold a fixed wide shot of the scene or **track** a body
(e.g. the rover), following it with a fixed offset so the action stays framed.
Tracking pose is resolved at capture time via ``engine.get_body_pose``, so the
tracked body only needs to exist once the scene is built — not at ``attach``.

The recorder is engine-agnostic: any object exposing ``add_camera``,
``set_camera_pose``, ``start_camera_recording``, ``render_camera``,
``stop_camera_recording`` and (for tracking) ``get_body_pose`` works, which keeps
it unit-testable against a lightweight fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


@dataclass
class RecorderConfig:
    """Framing and encoding settings for a single demo camera.

    Attributes:
        output_path: Destination ``.mp4`` (parent dirs are created on close).
        resolution: (width, height) in pixels; coerced to even values so the
            H.264 encoder accepts the frames.
        fov: Vertical field of view in degrees.
        fps: Playback frame rate of the encoded video.
        camera_name: Engine-side camera identifier.
        track_body: When set, the camera follows this body each frame using
            ``camera_offset`` / ``lookat_offset``. When ``None`` the camera holds
            the fixed ``camera_position`` / ``camera_lookat`` shot.
        camera_offset: Camera position relative to the tracked body (tracking).
        lookat_offset: Aim point relative to the tracked body (tracking).
        camera_position: Fixed camera position (non-tracking).
        camera_lookat: Fixed aim point (non-tracking).
    """

    output_path: Path
    resolution: Tuple[int, int] = (1280, 720)
    fov: float = 45.0
    fps: int = 30
    camera_name: str = "demo_camera"

    track_body: Optional[str] = None
    camera_offset: Vec3 = (6.0, -6.0, 4.0)
    lookat_offset: Vec3 = (0.0, 0.0, 0.5)

    camera_position: Vec3 = (8.0, -8.0, 5.0)
    camera_lookat: Vec3 = (0.0, 0.0, 0.5)

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        width, height = int(self.resolution[0]), int(self.resolution[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"resolution must be positive; got {self.resolution!r}")
        # Round up to even dimensions for libx264.
        self.resolution = (width + (width % 2), height + (height % 2))
        if int(self.fps) <= 0:
            raise ValueError(f"fps must be > 0; got {self.fps!r}")
        self.fps = int(self.fps)
        if not 0.0 < float(self.fov) < 180.0:
            raise ValueError(f"fov must be in (0, 180); got {self.fov!r}")


class VideoRecorder:
    """Drives a single engine camera to produce one MP4 demo of a run."""

    def __init__(self, config: RecorderConfig) -> None:
        self.config = config
        self._engine: Any = None
        self._camera: Any = None
        self._recording = False
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        """Number of frames captured so far."""
        return self._frame_count

    @property
    def is_recording(self) -> bool:
        return self._recording

    def attach(self, engine: Any) -> Any:
        """Register the camera with ``engine`` — call before ``build_scene()``.

        Raises:
            TypeError: If the engine does not expose the camera API.
        """
        for method in (
            "add_camera",
            "set_camera_pose",
            "start_camera_recording",
            "render_camera",
            "stop_camera_recording",
        ):
            if not hasattr(engine, method):
                raise TypeError(
                    f"engine {type(engine).__name__} lacks '{method}()'; "
                    "cannot record video"
                )
        if self.config.track_body is not None and not hasattr(engine, "get_body_pose"):
            raise TypeError(
                "engine lacks 'get_body_pose()' required for camera tracking"
            )
        self._engine = engine
        init_pos, init_lookat = self._initial_pose()
        self._camera = engine.add_camera(
            name=self.config.camera_name,
            resolution=self.config.resolution,
            pos=init_pos,
            lookat=init_lookat,
            fov=self.config.fov,
        )
        return self._camera

    def set_track_body(self, body_name: Optional[str]) -> None:
        """Set or change the tracked body before recording starts."""
        self.config.track_body = body_name

    def start(self) -> None:
        """Begin recording — call after ``build_scene()`` (SIMULATION phase)."""
        if self._engine is None:
            raise RuntimeError("attach(engine) must be called before start()")
        if self._recording:
            return
        self._engine.start_camera_recording(self.config.camera_name)
        self._recording = True

    def capture(self) -> None:
        """Render one frame. No-op until ``start()`` has been called."""
        if not self._recording:
            return
        if self.config.track_body is not None:
            pos, lookat = self._tracking_pose()
            self._engine.set_camera_pose(
                self.config.camera_name, pos=pos, lookat=lookat
            )
        self._engine.render_camera(self.config.camera_name)
        self._frame_count += 1

    def close(self) -> Optional[Path]:
        """Encode and write the video. Returns the output path, or None if idle.

        Safe to call when not recording or with zero frames (returns None
        without writing). Always leaves the recorder in a stopped state.
        """
        if not self._recording or self._engine is None:
            self._recording = False
            return None
        if self._frame_count == 0:
            # Nothing was captured; stop cleanly without writing a broken file.
            self._recording = False
            return None
        path = self.config.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._engine.stop_camera_recording(
            self.config.camera_name, save_path=str(path), fps=self.config.fps
        )
        self._recording = False
        return path

    # -- internal -----------------------------------------------------
    def _initial_pose(self) -> Tuple[Vec3, Vec3]:
        if self.config.track_body is not None:
            # Body pose is unknown until the scene is built; seed with the
            # offset relative to the world origin and correct on first capture.
            return self.config.camera_offset, self.config.lookat_offset
        return self.config.camera_position, self.config.camera_lookat

    def _tracking_pose(self) -> Tuple[Vec3, Vec3]:
        body_pos, _ = self._engine.get_body_pose(self.config.track_body)
        body_pos = np.asarray(body_pos, dtype=np.float64).reshape(3)
        pos = body_pos + np.asarray(self.config.camera_offset, dtype=np.float64)
        lookat = body_pos + np.asarray(self.config.lookat_offset, dtype=np.float64)
        return tuple(pos.tolist()), tuple(lookat.tolist())
