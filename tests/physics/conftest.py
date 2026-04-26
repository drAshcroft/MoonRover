"""Shared fixtures for physics engine unit tests.

Provides a complete mock of the Genesis 0.4.4 API so tests run without a GPU.
The mock is injected via monkeypatch into the _genesis_engine module's `gs` reference.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from moon_rover.core.physics.engine import GenesisConfig, ScenePhase


# ---------------------------------------------------------------------------
# Default GenesisConfig for tests
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> GenesisConfig:
    """Minimal GenesisConfig for use in unit tests (CPU, no GPU required)."""
    return GenesisConfig(
        gravity_vector=(0.0, 0.0, -1.622),
        timestep=1.0 / 240.0,
        contact_iterations=30,
        use_gpu=False,
        random_seed=42,
    )


# ---------------------------------------------------------------------------
# Genesis entity mock factory
# ---------------------------------------------------------------------------

def make_entity_mock(name: str = "entity") -> MagicMock:
    """Build a realistic mock of a gs.RigidEntity."""
    entity = MagicMock(name=f"entity_{name}")
    try:
        entity.idx = int(name)
    except ValueError:
        entity.idx = 0
    entity.get_pos.return_value = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    entity.get_quat.return_value = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    entity.get_vel.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    entity.get_ang.return_value = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    entity.get_dofs_pos.return_value = np.zeros(6, dtype=np.float32)
    entity.get_dofs_vel.return_value = np.zeros(6, dtype=np.float32)
    entity.get_links_pos.return_value = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
    entity.get_links_quat.return_value = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    entity.get_links_vel.return_value = np.zeros((1, 3), dtype=np.float32)
    entity.get_links_ang.return_value = np.zeros((1, 3), dtype=np.float32)
    entity.n_dofs = 6
    link_mock = MagicMock()
    link_mock.idx = entity.idx
    entity.links = [link_mock]
    geom_mock = MagicMock()
    geom_mock.idx = entity.idx
    entity.geoms = [geom_mock]
    return entity


# ---------------------------------------------------------------------------
# Genesis sensor mock
# ---------------------------------------------------------------------------

def make_sensor_mock() -> MagicMock:
    """Build a mock raycaster sensor."""
    sensor = MagicMock()
    data = MagicMock()
    data.distances = np.ones(100, dtype=np.float32) * 5.0
    data.positions = np.zeros((100, 3), dtype=np.float32)
    data.normals = np.tile([0.0, 0.0, 1.0], (100, 1)).astype(np.float32)
    sensor.get_data.return_value = data
    return sensor


# ---------------------------------------------------------------------------
# Full Genesis module mock (injected into _genesis_engine.gs)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_genesis(monkeypatch):
    """Replace the `gs` module in _genesis_engine with a complete MagicMock.

    This fixture is autouse so all tests in tests/physics/ run without a GPU.
    It resets GenesisPhysicsEngine's class-level singleton flag before each test.
    """
    import moon_rover.core.physics._genesis_engine as engine_mod

    # Reset singleton between tests
    engine_mod.GenesisPhysicsEngine._gs_initialized = False
    engine_mod.GenesisPhysicsEngine._gs_runtime_config = None
    engine_mod.GenesisPhysicsEngine._gs_runtime_owner_count = 0

    gs_mock = MagicMock(name="genesis")
    gs_mock.cuda = "cuda"
    gs_mock.cpu = "cpu"
    gs_mock.init = MagicMock()
    gs_mock.destroy = MagicMock()

    # Options namespace
    sim_opts_mock = MagicMock()
    rigid_opts_mock = MagicMock()
    mpm_opts_mock = MagicMock()
    vis_opts_mock = MagicMock()
    gs_mock.options = MagicMock()
    gs_mock.options.SimOptions = MagicMock(return_value=sim_opts_mock)
    gs_mock.options.RigidOptions = MagicMock(return_value=rigid_opts_mock)
    gs_mock.options.MPMOptions = MagicMock(return_value=mpm_opts_mock)
    gs_mock.options.VisOptions = MagicMock(return_value=vis_opts_mock)

    # Scene mock
    scene_mock = MagicMock(name="scene")
    scene_mock.build = MagicMock()
    scene_mock.step = MagicMock()
    scene_mock.get_contacts = MagicMock(return_value=[])

    entity_counter = {"n": 0}

    def add_entity_side_effect(**kwargs):
        entity_counter["n"] += 1
        return make_entity_mock(str(entity_counter["n"]))

    scene_mock.add_entity = MagicMock(side_effect=add_entity_side_effect)
    scene_mock.add_sensor = MagicMock(side_effect=lambda sensor: make_sensor_mock())

    gs_mock.Scene = MagicMock(return_value=scene_mock)

    # Morphs / materials (return plain mocks)
    gs_mock.morphs = MagicMock()
    gs_mock.morphs.Box = MagicMock(return_value=MagicMock())
    gs_mock.morphs.Plane = MagicMock(return_value=MagicMock())
    gs_mock.morphs.Sphere = MagicMock(return_value=MagicMock())
    gs_mock.morphs.URDF = MagicMock(return_value=MagicMock())
    gs_mock.morphs.Terrain = MagicMock(return_value=MagicMock())
    gs_mock.materials = MagicMock()
    gs_mock.materials.Rigid = MagicMock(return_value=MagicMock())
    gs_mock.sensors = MagicMock()
    gs_mock.sensors.Raycaster = MagicMock(return_value=MagicMock())
    gs_mock.sensors.RaycastPattern = MagicMock(return_value=MagicMock())
    gs_mock.sensors.SphericalPattern = MagicMock(return_value=MagicMock())

    monkeypatch.setattr(engine_mod, "gs", gs_mock)

    return gs_mock, scene_mock
