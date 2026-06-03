"""Unit tests for the VideoRecorder video-export orchestrator.

Driven against a lightweight fake engine that records the camera call sequence,
so the recorder's lifecycle and tracking logic are verified without Genesis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moon_rover.visualization.video_export import RecorderConfig, VideoRecorder


class _FakeCameraEngine:
    """Implements the engine camera API and logs every call."""

    def __init__(self, body_pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))):
        self.added = None
        self.recording_started = False
        self.frames = 0
        self.poses: list[tuple] = []
        self.stopped_with: dict | None = None
        self._body_pose = (np.asarray(body_pose[0], float), np.asarray(body_pose[1], float))

    def add_camera(self, name, resolution, pos, lookat, fov):
        self.added = {
            "name": name,
            "resolution": resolution,
            "pos": pos,
            "lookat": lookat,
            "fov": fov,
        }
        return object()

    def set_camera_pose(self, name, pos=None, lookat=None):
        self.poses.append((pos, lookat))

    def start_camera_recording(self, name):
        self.recording_started = True

    def render_camera(self, name):
        self.frames += 1

    def stop_camera_recording(self, name, save_path, fps):
        self.stopped_with = {"name": name, "save_path": save_path, "fps": fps}
        Path(save_path).write_bytes(b"FAKEMP4")  # stand in for the encoder

    def get_body_pose(self, name):
        return self._body_pose


# ---------------------------------------------------------------------------
# RecorderConfig validation
# ---------------------------------------------------------------------------


def test_config_rounds_resolution_to_even(tmp_path):
    cfg = RecorderConfig(output_path=tmp_path / "v.mp4", resolution=(641, 481))
    assert cfg.resolution == (642, 482)


def test_config_rejects_bad_values(tmp_path):
    with pytest.raises(ValueError, match="resolution"):
        RecorderConfig(output_path=tmp_path / "v.mp4", resolution=(0, 480))
    with pytest.raises(ValueError, match="fps"):
        RecorderConfig(output_path=tmp_path / "v.mp4", fps=0)
    with pytest.raises(ValueError, match="fov"):
        RecorderConfig(output_path=tmp_path / "v.mp4", fov=0.0)


# ---------------------------------------------------------------------------
# Fixed-shot lifecycle
# ---------------------------------------------------------------------------


def test_fixed_shot_records_and_writes(tmp_path):
    out = tmp_path / "demos" / "fixed.mp4"
    cfg = RecorderConfig(
        output_path=out,
        track_body=None,
        camera_position=(8.0, -8.0, 5.0),
        camera_lookat=(0.0, 0.0, 0.5),
        fps=30,
    )
    engine = _FakeCameraEngine()
    rec = VideoRecorder(cfg)
    rec.attach(engine)

    # Fixed shot seeds the camera at the configured pose.
    assert engine.added["pos"] == (8.0, -8.0, 5.0)
    assert engine.added["lookat"] == (0.0, 0.0, 0.5)

    rec.start()
    for _ in range(5):
        rec.capture()
    path = rec.close()

    assert engine.recording_started is True
    assert engine.frames == 5
    assert rec.frame_count == 5
    # No pose updates for a fixed shot.
    assert engine.poses == []
    assert path == out
    assert out.exists()
    assert engine.stopped_with == {"name": cfg.camera_name, "save_path": str(out), "fps": 30}


# ---------------------------------------------------------------------------
# Tracking-shot lifecycle
# ---------------------------------------------------------------------------


def test_tracking_shot_follows_body(tmp_path):
    cfg = RecorderConfig(
        output_path=tmp_path / "track.mp4",
        track_body="rover_001",
        camera_offset=(6.0, -6.0, 4.0),
        lookat_offset=(0.0, 0.0, 0.5),
    )
    engine = _FakeCameraEngine(body_pose=((10.0, 20.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
    rec = VideoRecorder(cfg)
    rec.attach(engine)
    rec.start()
    rec.capture()

    # Camera pose tracked the body + offsets.
    pos, lookat = engine.poses[-1]
    assert pos == pytest.approx((16.0, 14.0, 4.0))
    assert lookat == pytest.approx((10.0, 20.0, 0.5))
    assert engine.frames == 1


def test_set_track_body_changes_subject(tmp_path):
    cfg = RecorderConfig(output_path=tmp_path / "t.mp4", track_body=None)
    rec = VideoRecorder(cfg)
    rec.set_track_body("rover_001")
    assert rec.config.track_body == "rover_001"


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_capture_is_noop_before_start(tmp_path):
    engine = _FakeCameraEngine()
    rec = VideoRecorder(RecorderConfig(output_path=tmp_path / "v.mp4"))
    rec.attach(engine)
    rec.capture()  # before start()
    assert engine.frames == 0


def test_close_without_frames_writes_nothing(tmp_path):
    out = tmp_path / "v.mp4"
    engine = _FakeCameraEngine()
    rec = VideoRecorder(RecorderConfig(output_path=out, track_body=None))
    rec.attach(engine)
    rec.start()
    assert rec.close() is None
    assert engine.stopped_with is None
    assert not out.exists()


def test_attach_rejects_engine_without_camera_api(tmp_path):
    class _NoCamera:
        pass

    rec = VideoRecorder(RecorderConfig(output_path=tmp_path / "v.mp4"))
    with pytest.raises(TypeError, match="add_camera"):
        rec.attach(_NoCamera())


def test_attach_tracking_requires_get_body_pose(tmp_path):
    class _NoPose:
        def add_camera(self, **kwargs):
            return object()

        def set_camera_pose(self, *a, **k):
            pass

        def start_camera_recording(self, *a, **k):
            pass

        def render_camera(self, *a, **k):
            pass

        def stop_camera_recording(self, *a, **k):
            pass

    rec = VideoRecorder(
        RecorderConfig(output_path=tmp_path / "v.mp4", track_body="rover_001")
    )
    with pytest.raises(TypeError, match="get_body_pose"):
        rec.attach(_NoPose())


def test_start_before_attach_raises(tmp_path):
    rec = VideoRecorder(RecorderConfig(output_path=tmp_path / "v.mp4"))
    with pytest.raises(RuntimeError, match="attach"):
        rec.start()
