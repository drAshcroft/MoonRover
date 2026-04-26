"""Unit tests for GenesisPhysicsEngine.

All tests run without a real GPU via the mock_genesis fixture in conftest.py.
Run with: pytest tests/physics/ -v -m 'not gpu'
"""

from __future__ import annotations

import pickle
from unittest.mock import MagicMock

import numpy as np
import pytest

from moon_rover.core.physics.engine import GenesisConfig, GenesisPhysicsEngine, ScenePhase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(mock_genesis, config=None):
    """Create a configured (but not yet built) engine."""
    gs_mock, _ = mock_genesis
    engine = GenesisPhysicsEngine()
    cfg = config or GenesisConfig(
        gravity_vector=(0.0, 0.0, -1.622),
        timestep=1.0 / 240.0,
        use_gpu=False,
        random_seed=42,
    )
    engine.configure(cfg)
    return engine


def _make_built_engine(mock_genesis, config=None):
    """Create a fully built engine with one entity in SIMULATION phase."""
    gs_mock, scene_mock = mock_genesis
    engine = _make_engine(mock_genesis, config)
    engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
    engine.build_scene()
    return engine, scene_mock


# ---------------------------------------------------------------------------
# Phase enforcement (8 tests)
# ---------------------------------------------------------------------------

class TestPhaseEnforcement:

    def test_step_before_build_raises(self, mock_genesis, default_config):
        engine = _make_engine(mock_genesis, default_config)
        with pytest.raises(RuntimeError, match="simulation"):
            engine.step(default_config.timestep)

    def test_add_entity_after_build_raises(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine, _ = _make_built_engine(mock_genesis)
        with pytest.raises(RuntimeError, match="construction"):
            engine.add_entity("extra", gs_mock.morphs.Box(), gs_mock.materials.Rigid())

    def test_save_snapshot_in_construction_raises(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        with pytest.raises(RuntimeError, match="simulation"):
            engine.save_snapshot()

    def test_restore_snapshot_in_construction_raises(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        with pytest.raises(RuntimeError, match="simulation"):
            engine.restore_snapshot(b"data")

    def test_get_body_pose_in_construction_raises(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        with pytest.raises(RuntimeError, match="simulation"):
            engine.get_body_pose("box")

    def test_set_body_pose_in_construction_raises(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        with pytest.raises(RuntimeError, match="simulation"):
            engine.set_body_pose("box", np.zeros(3), np.array([0, 0, 0, 1]))

    def test_register_raycaster_after_build_raises(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        with pytest.raises(RuntimeError, match="construction"):
            engine.register_raycaster(
                "lidar", "box", 0,
                {"num_channels": 32, "elevation_range_deg": (-25, 15), "h_resolution_deg": 0.2},
                30.0,
            )

    def test_teardown_twice_raises(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        engine.teardown()
        with pytest.raises(RuntimeError):
            engine.teardown()


# ---------------------------------------------------------------------------
# Genesis singleton (3 tests)
# ---------------------------------------------------------------------------

class TestGenesisSingleton:

    def test_gs_init_called_once_across_two_configs(self, mock_genesis, default_config):
        gs_mock, _ = mock_genesis
        e1 = GenesisPhysicsEngine()
        e1.configure(default_config)
        e2 = GenesisPhysicsEngine()
        e2.configure(default_config)
        gs_mock.init.assert_called_once()

    def test_configure_rejects_cpu_then_gpu_runtime_mismatch(self, mock_genesis, default_config):
        gs_mock, _ = mock_genesis
        cpu_engine = GenesisPhysicsEngine()
        cpu_engine.configure(default_config)

        gpu_engine = GenesisPhysicsEngine()
        gpu_config = GenesisConfig(
            gravity_vector=default_config.gravity_vector,
            timestep=default_config.timestep,
            contact_iterations=default_config.contact_iterations,
            use_gpu=True,
            random_seed=default_config.random_seed,
        )
        with pytest.raises(RuntimeError, match="process-global"):
            gpu_engine.configure(gpu_config)

        gs_mock.init.assert_called_once()

    def test_configure_rejects_gpu_then_cpu_runtime_mismatch(self, mock_genesis, default_config):
        gs_mock, _ = mock_genesis
        gpu_config = GenesisConfig(
            gravity_vector=default_config.gravity_vector,
            timestep=default_config.timestep,
            contact_iterations=default_config.contact_iterations,
            use_gpu=True,
            random_seed=default_config.random_seed,
        )
        gpu_engine = GenesisPhysicsEngine()
        gpu_engine.configure(gpu_config)

        cpu_engine = GenesisPhysicsEngine()
        with pytest.raises(RuntimeError, match="process-global"):
            cpu_engine.configure(default_config)

        gs_mock.init.assert_called_once()

    def test_configure_rejects_seed_mismatch(self, mock_genesis, default_config):
        gs_mock, _ = mock_genesis
        first = GenesisPhysicsEngine()
        first.configure(default_config)

        second = GenesisPhysicsEngine()
        different_seed = GenesisConfig(
            gravity_vector=default_config.gravity_vector,
            timestep=default_config.timestep,
            contact_iterations=default_config.contact_iterations,
            use_gpu=default_config.use_gpu,
            random_seed=999,
        )
        with pytest.raises(RuntimeError, match="seed"):
            second.configure(different_seed)

        gs_mock.init.assert_called_once()

    def test_gs_destroy_called_on_teardown_when_policy_always(self, mock_genesis, monkeypatch):
        gs_mock, _ = mock_genesis
        monkeypatch.setenv("MOON_ROVER_GENESIS_DESTROY_POLICY", "always")
        engine, _ = _make_built_engine(mock_genesis)
        engine.teardown()
        gs_mock.destroy.assert_called_once()

    def test_teardown_waits_for_last_runtime_owner_before_destroy(self, mock_genesis, monkeypatch, default_config):
        gs_mock, _ = mock_genesis
        monkeypatch.setenv("MOON_ROVER_GENESIS_DESTROY_POLICY", "always")

        first = GenesisPhysicsEngine()
        first.configure(default_config)
        second = GenesisPhysicsEngine()
        second.configure(default_config)

        first.teardown()
        gs_mock.destroy.assert_not_called()
        assert GenesisPhysicsEngine._gs_initialized is True
        assert GenesisPhysicsEngine._gs_runtime_owner_count == 1

        second.teardown()
        gs_mock.destroy.assert_called_once()
        assert GenesisPhysicsEngine._gs_initialized is False
        assert GenesisPhysicsEngine._gs_runtime_owner_count == 0

    def test_gs_initialized_flag_reset_after_forced_destroy(self, mock_genesis, monkeypatch):
        monkeypatch.setenv("MOON_ROVER_GENESIS_DESTROY_POLICY", "always")
        engine, _ = _make_built_engine(mock_genesis)
        engine.teardown()
        assert GenesisPhysicsEngine._gs_initialized is False
        assert GenesisPhysicsEngine._gs_runtime_config is None

    def test_gs_initialized_flag_preserved_after_safe_windows_teardown(self, mock_genesis, monkeypatch):
        gs_mock, _ = mock_genesis
        monkeypatch.delenv("MOON_ROVER_GENESIS_DESTROY_POLICY", raising=False)
        monkeypatch.setattr("moon_rover.core.physics._genesis_engine.os.name", "nt")
        engine, _ = _make_built_engine(mock_genesis)
        engine.teardown()
        gs_mock.destroy.assert_not_called()
        assert GenesisPhysicsEngine._gs_initialized is True
        assert GenesisPhysicsEngine._gs_runtime_config is not None
        assert GenesisPhysicsEngine._gs_runtime_owner_count == 0


# ---------------------------------------------------------------------------
# configure() (4 tests)
# ---------------------------------------------------------------------------

class TestConfigure:

    def test_configure_creates_scene(self, mock_genesis, default_config):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis, default_config)
        gs_mock.Scene.assert_called_once()

    def test_configure_passes_gravity(self, mock_genesis):
        gs_mock, _ = mock_genesis
        config = GenesisConfig(gravity_vector=(0.0, 0.0, -1.622), use_gpu=False)
        engine = GenesisPhysicsEngine()
        engine.configure(config)
        call_kwargs = gs_mock.options.SimOptions.call_args
        assert call_kwargs is not None
        gravity_arg = call_kwargs.kwargs.get("gravity") or call_kwargs.args[0]
        assert gravity_arg == (0.0, 0.0, -1.622)

    def test_configure_uses_cpu_backend_when_not_gpu(self, mock_genesis, default_config):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis, default_config)
        init_kwargs = gs_mock.init.call_args.kwargs
        assert init_kwargs["backend"] == gs_mock.cpu

    def test_solver_backends_has_rigid_body_key(self, mock_genesis, default_config):
        engine = _make_engine(mock_genesis, default_config)
        backends = engine.solver_backends
        assert "rigid_body" in backends
        assert "mpm" in backends


# ---------------------------------------------------------------------------
# Entity registry (5 tests)
# ---------------------------------------------------------------------------

class TestEntityRegistry:

    def test_add_entity_returns_entity(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis)
        entity = engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        assert entity is not None

    def test_add_entity_duplicate_name_raises(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        with pytest.raises(ValueError, match="already registered"):
            engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())

    def test_get_entity_unknown_raises_key_error(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        with pytest.raises(KeyError):
            engine.get_entity("does_not_exist")

    def test_list_entities_after_add(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("box1", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.add_entity("box2", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        names = engine.list_entities()
        assert "box1" in names
        assert "box2" in names
        assert len(names) == 2

    def test_add_terrain_entity_populates_terrain_record(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        height_field = np.zeros((64, 64), dtype=np.float32)
        engine.add_terrain_entity("terrain", height_field, [100.0, 100.0])
        assert engine._terrain is not None
        assert engine._terrain.resolution_x == 64
        assert engine._terrain.resolution_y == 64

    def test_add_terrain_entity_uses_regolith_contact_material_defaults(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis)
        height_field = np.zeros((16, 16), dtype=np.float32)
        engine.add_terrain_entity("terrain", height_field, [10.0, 10.0])
        material_kwargs = gs_mock.materials.Rigid.call_args.kwargs
        assert material_kwargs["rho"] == pytest.approx(1800.0)
        assert material_kwargs["friction"] == pytest.approx(1.2)
        assert material_kwargs["coup_restitution"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# build_scene() (4 tests)
# ---------------------------------------------------------------------------

class TestBuildScene:

    def test_build_calls_scene_build(self, mock_genesis):
        gs_mock, scene_mock = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.build_scene()
        scene_mock.build.assert_called_once()

    def test_phase_becomes_simulation_after_build(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        assert engine.get_phase() == ScenePhase.SIMULATION

    def test_build_with_no_entities_raises(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        with pytest.raises(RuntimeError, match="no entities"):
            engine.build_scene()

    def test_build_scene_with_multiple_envs_raises(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        with pytest.raises(NotImplementedError, match="n_envs=1"):
            engine.build_scene(n_envs=2)


# ---------------------------------------------------------------------------
# step() (5 tests)
# ---------------------------------------------------------------------------

class TestStep:

    def test_step_calls_scene_step(self, mock_genesis):
        engine, scene_mock = _make_built_engine(mock_genesis)
        engine.step(1.0 / 240.0)
        scene_mock.step.assert_called()

    def test_step_increments_sim_time(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        engine.step(1.0 / 240.0)
        assert abs(engine.get_sim_time() - 1.0 / 240.0) < 1e-9

    def test_step_increments_step_count(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        engine.step(1.0 / 240.0)
        engine.step(1.0 / 240.0)
        assert engine.get_step_count() == 2

    def test_step_mismatched_dt_raises(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        with pytest.raises(ValueError, match="dt"):
            engine.step(0.01)   # configured dt is 1/240 ≈ 0.00417

    def test_step_caches_prev_velocities_for_tracked_acceleration(self, mock_genesis):
        """Tracked bodies should snapshot pre-step velocity for acceleration telemetry."""
        gs_mock, scene_mock = mock_genesis
        engine = _make_engine(mock_genesis)
        entity = engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        entity.get_vel.return_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        entity.get_ang.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        engine.build_scene()

        engine.get_body_acceleration("box")
        engine.step(1.0 / 240.0)
        rec = engine._entities["box"]
        np.testing.assert_allclose(rec.prev_lin_vel, [1.0, 2.0, 3.0], atol=1e-6)
        assert rec.accel_prev_step == 1

    def test_step_skips_velocity_queries_without_acceleration_tracking(self, mock_genesis):
        gs_mock, scene_mock = mock_genesis
        engine = _make_engine(mock_genesis)
        entity = engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.build_scene()
        entity.get_vel.reset_mock()
        entity.get_ang.reset_mock()

        engine.step(1.0 / 240.0)

        entity.get_vel.assert_not_called()
        entity.get_ang.assert_not_called()


# ---------------------------------------------------------------------------
# Body state queries (8 tests)
# ---------------------------------------------------------------------------

class TestBodyStateQueries:

    def test_get_body_pose_returns_float32(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        pos, quat = engine.get_body_pose("box")
        assert pos.dtype == np.float32
        assert quat.dtype == np.float32
        assert pos.shape == (3,)
        assert quat.shape == (4,)

    def test_get_body_velocity_returns_float32(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        lin, ang = engine.get_body_velocity("box")
        assert lin.dtype == np.float32
        assert ang.dtype == np.float32

    def test_get_body_pose_nonzero_env_idx_raises(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        with pytest.raises(ValueError, match="env_idx=1"):
            engine.get_body_pose("box", env_idx=1)

    def test_get_body_pose_multi_env_scene_raises(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        engine._n_envs = 2
        with pytest.raises(NotImplementedError, match="n_envs=2"):
            engine.get_body_pose("box", env_idx=0)

    def test_get_body_acceleration_zero_on_static_body(self, mock_genesis):
        """Static body: velocity unchanged → acceleration = 0."""
        engine, _ = _make_built_engine(mock_genesis)
        engine.get_body_acceleration("box")
        engine.step(1.0 / 240.0)
        lin_a, ang_a = engine.get_body_acceleration("box")
        np.testing.assert_allclose(lin_a, [0.0, 0.0, 0.0], atol=1e-5)

    def test_get_body_acceleration_after_velocity_change(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis)
        entity = engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.build_scene()

        # Before step: vel = 0; after step: vel = [1,0,0]
        entity.get_vel.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        entity.get_ang.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        engine.get_body_acceleration("box")
        dt = 1.0 / 240.0

        def step_side_effect():
            entity.get_vel.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        engine._scene.step.side_effect = step_side_effect
        engine.step(dt)

        lin_a, _ = engine.get_body_acceleration("box")
        expected_ax = 1.0 / dt
        assert abs(lin_a[0] - expected_ax) < 1.0  # approx — prev was 0, curr is 1

    def test_get_body_acceleration_first_call_primes_telemetry(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        lin_a, ang_a = engine.get_body_acceleration("box")
        np.testing.assert_allclose(lin_a, [0.0, 0.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(ang_a, [0.0, 0.0, 0.0], atol=1e-5)
        rec = engine._entities["box"]
        assert rec.accel_tracking_enabled is True
        assert rec.accel_prev_step == 0

    def test_get_body_acceleration_strict_mode_raises_on_velocity_query_failure(
        self, mock_genesis, monkeypatch
    ):
        engine, _ = _make_built_engine(mock_genesis)
        entity = engine._entities["box"].genesis_entity
        entity.get_vel.side_effect = RuntimeError("velocity unavailable")
        monkeypatch.setenv("MOON_ROVER_GENESIS_STRICT_DIAGNOSTICS", "1")

        with pytest.raises(RuntimeError, match="get_body_acceleration\\(\\) telemetry prime"):
            engine.get_body_acceleration("box")

    def test_get_dof_positions_shape(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        dofs = engine.get_dof_positions("box")
        assert dofs.ndim == 1
        assert dofs.dtype == np.float32

    def test_get_link_poses_list_length(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        poses = engine.get_link_poses("box")
        assert isinstance(poses, list)
        assert len(poses) >= 1
        pos, quat = poses[0]
        assert pos.shape == (3,)
        assert quat.shape == (4,)


class TestSingleEnvGating:

    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("get_body_velocity", ("box",)),
            ("get_body_acceleration", ("box",)),
            ("get_link_poses", ("box",)),
            ("get_link_velocities", ("box",)),
            ("get_dof_positions", ("box",)),
            ("get_dof_velocities", ("box",)),
            ("set_body_pose", ("box", np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]))),
            ("set_body_velocity", ("box", np.zeros(3), np.zeros(3))),
            ("set_dof_positions", ("box", np.zeros(6))),
            ("set_dof_velocities", ("box", np.zeros(6))),
            ("apply_dof_forces", ("box", np.zeros(6))),
        ],
    )
    def test_nonzero_env_idx_raises(self, mock_genesis, method_name, args):
        engine, _ = _make_built_engine(mock_genesis)
        method = getattr(engine, method_name)
        with pytest.raises(ValueError, match="env_idx=1"):
            method(*args, env_idx=1)


# ---------------------------------------------------------------------------
# Terrain queries (4 tests)
# ---------------------------------------------------------------------------

class TestTerrainQueries:

    def _engine_with_flat_terrain(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        height_field = np.zeros((64, 64), dtype=np.float32)
        engine.add_terrain_entity("terrain", height_field, [100.0, 100.0])
        gs_mock, _ = mock_genesis
        engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.build_scene()
        return engine

    def test_flat_terrain_height_is_zero(self, mock_genesis):
        engine = self._engine_with_flat_terrain(mock_genesis)
        h = engine.get_terrain_height(50.0, 50.0)
        assert abs(h) < 1e-6

    def test_terrain_height_bilinear_midpoint(self, mock_genesis):
        """Ramp from 0 at x=0 to 10 at x=100: midpoint should be ~5."""
        engine = _make_engine(mock_genesis)
        H, W = 64, 64
        height_field = np.zeros((H, W), dtype=np.float32)
        for c in range(W):
            height_field[:, c] = (c / (W - 1)) * 10.0
        engine.add_terrain_entity("terrain", height_field, [100.0, 100.0])
        gs_mock, _ = mock_genesis
        engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.build_scene()
        h = engine.get_terrain_height(50.0, 50.0)
        assert abs(h - 5.0) < 0.5   # bilinear mid — within 0.5 m

    def test_flat_terrain_normal_is_up(self, mock_genesis):
        engine = self._engine_with_flat_terrain(mock_genesis)
        normal = engine.get_terrain_normal(50.0, 50.0)
        np.testing.assert_allclose(normal, [0.0, 0.0, 1.0], atol=1e-5)

    def test_ramp_terrain_normal_matches_world_space_slope(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        height_field = np.zeros((64, 64), dtype=np.float32)
        for c in range(height_field.shape[1]):
            height_field[:, c] = (c / (height_field.shape[1] - 1)) * 10.0
        engine.add_terrain_entity("terrain", height_field, [100.0, 100.0])

        normal = engine.get_terrain_normal(50.0, 50.0)
        expected = np.array([-0.1, 0.0, 1.0], dtype=np.float32)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(normal, expected, atol=2e-2)

    def test_terrain_height_no_terrain_raises(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        with pytest.raises(RuntimeError, match="terrain"):
            engine.get_terrain_height(0.0, 0.0)


# ---------------------------------------------------------------------------
# Snapshot round-trip (6 tests)
# ---------------------------------------------------------------------------

class TestSnapshot:

    def test_save_snapshot_returns_bytes(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        snap = engine.save_snapshot()
        assert isinstance(snap, bytes)
        assert len(snap) > 0

    def test_restore_snapshot_restores_sim_time(self, mock_genesis):
        engine, scene_mock = _make_built_engine(mock_genesis)
        engine.step(1.0 / 240.0)
        expected_time = engine.get_sim_time()   # capture snapshot time
        snap = engine.save_snapshot()
        engine.step(1.0 / 240.0)
        engine.step(1.0 / 240.0)
        scene_mock.step.reset_mock()
        engine.restore_snapshot(snap)
        assert abs(engine.get_sim_time() - expected_time) < 1e-9
        scene_mock.step.assert_not_called()

    def test_restore_snapshot_restores_step_count(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        engine.step(1.0 / 240.0)
        snap = engine.save_snapshot()
        engine.step(1.0 / 240.0)
        engine.step(1.0 / 240.0)
        engine.restore_snapshot(snap)
        assert engine.get_step_count() == 1

    def test_restore_snapshot_restores_quaternion(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        entity = engine._entities["box"].genesis_entity
        entity.get_quat.return_value = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
        snap = engine.save_snapshot()
        entity.set_quat.reset_mock()

        engine.restore_snapshot(snap)

        entity.set_quat.assert_called_once()
        np.testing.assert_allclose(
            entity.set_quat.call_args.args[0],
            [0.1, 0.2, 0.3, 0.9],
        )

    def test_restore_snapshot_restores_acceleration_tracking_state(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        engine.get_body_acceleration("box")
        engine.step(1.0 / 240.0)
        snap = engine.save_snapshot()

        rec = engine._entities["box"]
        rec.accel_tracking_enabled = False
        rec.accel_prev_step = -1
        rec.prev_lin_vel = np.zeros(3, dtype=np.float32)
        rec.prev_ang_vel = np.zeros(3, dtype=np.float32)

        engine.restore_snapshot(snap)

        assert rec.accel_tracking_enabled is True
        assert rec.accel_prev_step == 1
        np.testing.assert_allclose(rec.prev_lin_vel, [0.0, 0.0, 0.0], atol=1e-6)

    def test_restore_snapshot_restores_pose_velocity_and_dofs(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        entity = engine._entities["box"].genesis_entity
        entity.get_pos.return_value = np.array([1.2, -0.4, 0.9], dtype=np.float32)
        entity.get_quat.return_value = np.array([0.1, 0.2, -0.1, 0.96], dtype=np.float32)
        entity.get_vel.return_value = np.array([0.5, -0.3, 1.1], dtype=np.float32)
        entity.get_ang.return_value = np.array([0.2, -0.6, 0.4], dtype=np.float32)
        entity.get_dofs_pos.return_value = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32)
        entity.get_dofs_vel.return_value = np.array([0.3, 0.2, 0.1, -0.1, -0.2, -0.3], dtype=np.float32)

        snap = engine.save_snapshot()

        entity.set_pos.reset_mock()
        entity.set_quat.reset_mock()
        entity.set_vel.reset_mock()
        entity.set_ang.reset_mock()
        entity.set_dofs_pos.reset_mock()
        entity.set_dofs_velocity.reset_mock()

        engine.restore_snapshot(snap)

        np.testing.assert_allclose(entity.set_pos.call_args.args[0], [1.2, -0.4, 0.9])
        np.testing.assert_allclose(entity.set_quat.call_args.args[0], [0.1, 0.2, -0.1, 0.96])
        np.testing.assert_allclose(entity.set_vel.call_args.args[0], [0.5, -0.3, 1.1])
        np.testing.assert_allclose(entity.set_ang.call_args.args[0], [0.2, -0.6, 0.4])
        np.testing.assert_allclose(
            entity.set_dofs_pos.call_args.args[0],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            entity.set_dofs_velocity.call_args.args[0],
            [0.3, 0.2, 0.1, -0.1, -0.2, -0.3],
        )

    def test_restore_snapshot_dt_mismatch_raises(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        bad_snap = pickle.dumps({
            "version": 1,
            "sim_time": 0.0,
            "step_count": 0,
            "dt": 1.0 / 60.0,
            "n_envs": 1,
            "entities": {},
        })
        with pytest.raises(ValueError, match="dt="):
            engine.restore_snapshot(bad_snap)

    def test_restore_snapshot_corrupt_raises_value_error(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        with pytest.raises(ValueError, match="corrupt"):
            engine.restore_snapshot(b"not_valid_pickle_data")

    def test_restore_snapshot_wrong_version_raises(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        bad_snap = pickle.dumps({"version": 99, "entities": {}, "dt": 1.0 / 240.0})
        with pytest.raises(ValueError, match="version"):
            engine.restore_snapshot(bad_snap)

    def test_restore_snapshot_entity_mismatch_raises(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        bad_snap = pickle.dumps({
            "version":    1,
            "sim_time":   0.0,
            "step_count": 0,
            "dt":         1.0 / 240.0,
            "n_envs":     1,
            "entities":   {"ghost_entity": {}},
        })
        with pytest.raises(ValueError, match="ghost_entity"):
            engine.restore_snapshot(bad_snap)

    def test_save_snapshot_missing_velocity_state_raises_runtime_error(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        entity = engine._entities["box"].genesis_entity
        entity.get_vel.side_effect = RuntimeError("velocity unavailable")

        with pytest.raises(RuntimeError, match="save_snapshot\\(\\) state capture"):
            engine.save_snapshot()

    def test_restore_snapshot_strict_mode_raises_when_orientation_restore_skipped(
        self, mock_genesis, monkeypatch
    ):
        engine, _ = _make_built_engine(mock_genesis)
        entity = engine._entities["box"].genesis_entity
        snap = engine.save_snapshot()
        entity.set_quat = None
        entity.n_dofs = 0
        monkeypatch.setenv("MOON_ROVER_GENESIS_STRICT_DIAGNOSTICS", "1")

        with pytest.raises(RuntimeError, match="orientation restore"):
            engine.restore_snapshot(snap)


# ---------------------------------------------------------------------------
# Raycaster (3 tests)
# ---------------------------------------------------------------------------

class TestRaycaster:

    def _engine_with_raycaster(self, mock_genesis):
        gs_mock, scene_mock = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("rover", gs_mock.morphs.URDF("rover.urdf"), gs_mock.materials.Rigid())
        engine.register_raycaster(
            "lidar", "rover", 0,
            {"num_channels": 32, "elevation_range_deg": (-25.0, 15.0), "h_resolution_deg": 0.2},
            30.0,
        )
        engine.build_scene()
        return engine, scene_mock

    def test_register_raycaster_calls_add_sensor(self, mock_genesis):
        gs_mock, scene_mock = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("rover", gs_mock.morphs.URDF("rover.urdf"), gs_mock.materials.Rigid())
        engine.register_raycaster(
            "lidar", "rover", 0,
            {"num_channels": 32, "elevation_range_deg": (-25.0, 15.0), "h_resolution_deg": 0.2},
            30.0,
        )
        scene_mock.add_sensor.assert_called_once()
        gs_mock.sensors.SphericalPattern.assert_called_once()
        pattern_kwargs = gs_mock.sensors.SphericalPattern.call_args.kwargs
        assert pattern_kwargs["fov"] == (360.0, (-25.0, 15.0))
        assert pattern_kwargs["n_points"] == (1800, 32)
        raycaster_kwargs = gs_mock.sensors.Raycaster.call_args.kwargs
        assert raycaster_kwargs["entity_idx"] == engine._entities["rover"].genesis_entity.idx
        assert raycaster_kwargs["link_idx_local"] == 0
        assert raycaster_kwargs["max_range"] == 30.0
        assert raycaster_kwargs["no_hit_value"] == 30.0
        assert raycaster_kwargs["return_world_frame"] is True

    def test_register_raycaster_rejects_invalid_pattern(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("rover", gs_mock.morphs.URDF("rover.urdf"), gs_mock.materials.Rigid())

        with pytest.raises(ValueError, match="h_resolution_deg"):
            engine.register_raycaster(
                "lidar", "rover", 0,
                {"num_channels": 32, "elevation_range_deg": (-25.0, 15.0), "h_resolution_deg": 0.0},
                30.0,
            )

    def test_query_raycaster_returns_correct_keys(self, mock_genesis):
        engine, _ = self._engine_with_raycaster(mock_genesis)
        result = engine.query_raycaster("lidar")
        assert "distances" in result
        assert "positions" in result
        assert "normals" in result

    def test_query_raycaster_supports_read_tuple(self, mock_genesis):
        engine, _ = self._engine_with_raycaster(mock_genesis)

        class ReadOnlySensor:
            def read(self):
                return (
                    np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=np.float32),
                    np.array([[7.0, 8.0]], dtype=np.float32),
                )

        engine._raycasters["lidar"].genesis_sensor = ReadOnlySensor()
        result = engine.query_raycaster("lidar")

        np.testing.assert_allclose(result["distances"], [7.0, 8.0])
        np.testing.assert_allclose(
            result["positions"],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )
        np.testing.assert_allclose(result["normals"], np.zeros((2, 3), dtype=np.float32))

    def test_query_unknown_raycaster_raises(self, mock_genesis):
        engine, _ = self._engine_with_raycaster(mock_genesis)
        with pytest.raises(KeyError):
            engine.query_raycaster("does_not_exist")


# ---------------------------------------------------------------------------
# Contact queries (2 tests)
# ---------------------------------------------------------------------------

class TestContactQueries:

    def test_get_body_contacts_empty_when_none(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        contacts = engine.get_body_contacts("box")
        assert contacts == []

    def test_get_body_contacts_from_genesis_contact_dict(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("ground", gs_mock.morphs.Plane(), gs_mock.materials.Rigid())
        engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.build_scene()

        ground = engine._entities["ground"].genesis_entity
        box = engine._entities["box"].genesis_entity
        ground.links[0].idx = 0
        ground.geoms[0].idx = 0
        box.links[0].idx = 1
        box.geoms[0].idx = 1
        box.get_contacts.return_value = {
            "link_a": np.array([[0, 0]], dtype=np.int32),
            "link_b": np.array([[1, 1]], dtype=np.int32),
            "geom_a": np.array([[0, 0]], dtype=np.int32),
            "geom_b": np.array([[1, 1]], dtype=np.int32),
            "valid_mask": np.array([[True, False]], dtype=np.bool_),
            "penetration": np.array([[0.01, 0.02]], dtype=np.float32),
            "position": np.array(
                [[[0.1, 0.2, 0.0], [9.0, 9.0, 9.0]]],
                dtype=np.float32,
            ),
            "normal": np.array(
                [[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]],
                dtype=np.float32,
            ),
            "force_a": np.array(
                [[[0.0, 0.0, -2.5], [0.0, 0.0, -99.0]]],
                dtype=np.float32,
            ),
            "force_b": np.array(
                [[[0.0, 0.0, 2.5], [0.0, 0.0, 99.0]]],
                dtype=np.float32,
            ),
        }

        contacts = engine.get_body_contacts("box")

        assert len(contacts) == 1
        assert contacts[0]["body_b"] == "ground"
        np.testing.assert_allclose(contacts[0]["pos"], [0.1, 0.2, 0.0])
        np.testing.assert_allclose(contacts[0]["normal"], [0.0, 0.0, -1.0])
        assert contacts[0]["force_n"] == pytest.approx(2.5)
        assert contacts[0]["penetration"] == pytest.approx(0.01)
        assert engine.is_in_contact("box", "ground") is True

    def test_get_body_contacts_from_scene_object_list(self, mock_genesis):
        gs_mock, scene_mock = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("box1", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.add_entity("box2", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.build_scene()

        contact = MagicMock()
        contact.entity_a = engine._entities["box1"].genesis_entity
        contact.entity_b = engine._entities["box2"].genesis_entity
        contact.position = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        contact.normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        contact.force_normal = 4.5
        scene_mock.get_contacts.return_value = [contact]

        contacts = engine.get_body_contacts("box1")

        assert len(contacts) == 1
        assert contacts[0]["body_b"] == "box2"
        assert contacts[0]["force_n"] == pytest.approx(4.5)

    def test_is_in_contact_false_when_no_contacts(self, mock_genesis):
        gs_mock, _ = mock_genesis
        engine = _make_engine(mock_genesis)
        engine.add_entity("box1", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.add_entity("box2", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.build_scene()
        assert engine.is_in_contact("box1", "box2") is False

    def test_get_body_contacts_strict_mode_raises_when_contact_query_fails(
        self, mock_genesis, monkeypatch
    ):
        engine, _ = _make_built_engine(mock_genesis)
        entity = engine._entities["box"].genesis_entity
        entity.get_contacts.side_effect = RuntimeError("contact api drift")
        monkeypatch.setenv("MOON_ROVER_GENESIS_STRICT_DIAGNOSTICS", "1")

        with pytest.raises(RuntimeError, match="entity contact query"):
            engine.get_body_contacts("box")


# ---------------------------------------------------------------------------
# solver_backends (2 tests)
# ---------------------------------------------------------------------------

class TestSolverBackends:

    def test_solver_backends_returns_dict(self, mock_genesis, default_config):
        engine = _make_engine(mock_genesis, default_config)
        backends = engine.solver_backends
        assert isinstance(backends, dict)

    def test_solver_backends_defensive_copy(self, mock_genesis, default_config):
        engine = _make_engine(mock_genesis, default_config)
        b1 = engine.solver_backends
        b1["injected"] = "bad"
        b2 = engine.solver_backends
        assert "injected" not in b2


# ---------------------------------------------------------------------------
# Teardown (3 tests)
# ---------------------------------------------------------------------------

class TestTeardown:

    def test_teardown_clears_entities(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        engine.teardown()
        assert engine._entities == {}

    def test_teardown_clears_terrain(self, mock_genesis):
        engine = _make_engine(mock_genesis)
        height_field = np.zeros((64, 64), dtype=np.float32)
        engine.add_terrain_entity("terrain", height_field, [100.0, 100.0])
        gs_mock, _ = mock_genesis
        engine.add_entity("box", gs_mock.morphs.Box(), gs_mock.materials.Rigid())
        engine.build_scene()
        engine.teardown()
        assert engine._terrain is None

    def test_teardown_phase_becomes_teardown(self, mock_genesis):
        engine, _ = _make_built_engine(mock_genesis)
        engine.teardown()
        assert engine.get_phase() == ScenePhase.TEARDOWN

    def test_teardown_never_policy_skips_destroy(self, mock_genesis, monkeypatch):
        gs_mock, _ = mock_genesis
        monkeypatch.setenv("MOON_ROVER_GENESIS_DESTROY_POLICY", "never")
        engine, _ = _make_built_engine(mock_genesis)
        engine.teardown()
        gs_mock.destroy.assert_not_called()


# ---------------------------------------------------------------------------
# GenesisConfig.from_yaml (integration — no Genesis, just YAML)
# ---------------------------------------------------------------------------

class TestGenesisConfigFromYaml:

    def test_from_yaml_loads_gravity(self):
        config = GenesisConfig.from_yaml("configs/physics.yaml")
        # physics.yaml gravity.value: [0, 0, -1.622]
        assert abs(config.gravity_vector[2] - (-1.622)) < 1e-4

    def test_from_yaml_loads_timestep(self):
        config = GenesisConfig.from_yaml("configs/physics.yaml")
        assert abs(config.timestep - 0.004166) < 1e-6

    def test_from_yaml_loads_seed(self):
        config = GenesisConfig.from_yaml("configs/physics.yaml")
        assert config.random_seed == 42
