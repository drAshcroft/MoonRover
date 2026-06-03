"""Unit tests for the GenesisPhysicsEngine camera / video-capture API.

All tests run against the mock_genesis fixture (no real GPU/render). They verify
the lifecycle contract and that calls forward to the Genesis camera object with
the expected arguments — real frame encoding is covered by the opt-in
tests/physics_real smoke test.
"""

from __future__ import annotations

import pytest

from moon_rover.core.physics.engine import GenesisConfig, GenesisPhysicsEngine


def _make_engine(mock_genesis, build=False):
    gs_mock, scene_mock = mock_genesis
    engine = GenesisPhysicsEngine()
    engine.configure(
        GenesisConfig(
            gravity_vector=(0.0, 0.0, -1.622),
            timestep=1.0 / 240.0,
            use_gpu=False,
            random_seed=42,
        )
    )
    return engine, gs_mock, scene_mock


def _built_engine_with_camera(mock_genesis, **cam_kwargs):
    engine, gs_mock, scene_mock = _make_engine(mock_genesis)
    cam = engine.add_camera("demo", **cam_kwargs)
    engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
    engine.build_scene()
    return engine, scene_mock, cam


# ---------------------------------------------------------------------------
# add_camera
# ---------------------------------------------------------------------------


def test_add_camera_registers_and_forwards_args(mock_genesis):
    engine, _, scene_mock = _make_engine(mock_genesis)
    cam = engine.add_camera(
        "demo", resolution=(640, 480), pos=(1.0, 2.0, 3.0), lookat=(0.0, 0.0, 0.5), fov=50.0
    )
    assert cam is scene_mock.add_camera.return_value
    scene_mock.add_camera.assert_called_once()
    kwargs = scene_mock.add_camera.call_args.kwargs
    assert kwargs["res"] == (640, 480)
    assert kwargs["pos"] == (1.0, 2.0, 3.0)
    assert kwargs["lookat"] == (0.0, 0.0, 0.5)
    assert kwargs["fov"] == 50.0
    assert kwargs["GUI"] is False


def test_add_camera_duplicate_name_raises(mock_genesis):
    engine, _, _ = _make_engine(mock_genesis)
    engine.add_camera("demo")
    with pytest.raises(ValueError, match="already registered"):
        engine.add_camera("demo")


def test_add_camera_rejects_nonpositive_resolution(mock_genesis):
    engine, _, _ = _make_engine(mock_genesis)
    with pytest.raises(ValueError, match="resolution"):
        engine.add_camera("demo", resolution=(0, 480))


def test_add_camera_rejects_out_of_range_fov(mock_genesis):
    engine, _, _ = _make_engine(mock_genesis)
    with pytest.raises(ValueError, match="fov"):
        engine.add_camera("demo", fov=200.0)


def test_add_camera_after_build_raises_phase(mock_genesis):
    engine, scene_mock, _ = _built_engine_with_camera(mock_genesis)
    with pytest.raises(RuntimeError, match="construction"):
        engine.add_camera("late")


# ---------------------------------------------------------------------------
# recording lifecycle
# ---------------------------------------------------------------------------


def test_recording_lifecycle_forwards_to_camera(mock_genesis, tmp_path):
    engine, _, cam = _built_engine_with_camera(mock_genesis)
    out = tmp_path / "demo.mp4"

    engine.start_camera_recording("demo")
    cam.start_recording.assert_called_once()

    engine.render_camera("demo")
    engine.render_camera("demo")
    assert cam.render.call_count == 2

    engine.stop_camera_recording("demo", save_path=str(out), fps=24)
    cam.stop_recording.assert_called_once_with(save_to_filename=str(out), fps=24)


def test_set_camera_pose_forwards_to_camera(mock_genesis):
    engine, _, cam = _built_engine_with_camera(mock_genesis)
    engine.set_camera_pose("demo", pos=(5.0, 6.0, 7.0), lookat=(0.0, 0.0, 1.0))
    cam.set_pose.assert_called_once_with(pos=(5.0, 6.0, 7.0), lookat=(0.0, 0.0, 1.0))


def test_stop_recording_without_start_raises(mock_genesis, tmp_path):
    engine, _, _ = _built_engine_with_camera(mock_genesis)
    with pytest.raises(RuntimeError, match="not recording"):
        engine.stop_camera_recording("demo", save_path=str(tmp_path / "x.mp4"))


def test_stop_recording_rejects_nonpositive_fps(mock_genesis, tmp_path):
    engine, _, _ = _built_engine_with_camera(mock_genesis)
    engine.start_camera_recording("demo")
    with pytest.raises(ValueError, match="fps"):
        engine.stop_camera_recording("demo", save_path=str(tmp_path / "x.mp4"), fps=0)


def test_recording_requires_simulation_phase(mock_genesis):
    engine, _, _ = _make_engine(mock_genesis)
    engine.add_camera("demo")
    with pytest.raises(RuntimeError, match="simulation"):
        engine.start_camera_recording("demo")


def test_unknown_camera_name_raises(mock_genesis):
    engine, _, _ = _built_engine_with_camera(mock_genesis)
    with pytest.raises(ValueError, match="not registered"):
        engine.render_camera("missing")


def test_cameras_cleared_on_teardown(mock_genesis):
    engine, _, _ = _built_engine_with_camera(mock_genesis)
    engine.teardown()
    assert engine._cameras == {}
