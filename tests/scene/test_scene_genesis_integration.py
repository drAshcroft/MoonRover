"""End-to-end integration test: SceneComposer + GenesisPhysicsEngine.

Exercises the full pipeline with a real GenesisPhysicsEngine and the project
fixture configs.  Automatically skipped when Genesis is not installed.

Run with:
    pytest tests/scene/test_scene_genesis_integration.py -v -m genesis
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Genesis availability guard
# ──────────────────────────────────────────────────────────────────────────────

try:
    import genesis as gs  # noqa: F401
    _GENESIS_AVAILABLE = True
except ImportError:
    _GENESIS_AVAILABLE = False

pytestmark = pytest.mark.genesis  # tag so `-m not genesis` can skip entire file

_CONFIGS = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
_SCENE_YAML   = os.path.join(_CONFIGS, "scene.yaml")
_ROVER_YAML   = os.path.join(_CONFIGS, "rover.yaml")
_MISSION_YAML = os.path.join(_CONFIGS, "mission.yaml")
_PHYSICS_YAML = os.path.join(_CONFIGS, "physics.yaml")
_SENSORS_YAML = os.path.join(_CONFIGS, "sensors.yaml")

# Integration test terrain resolution — small enough for fast CI
_TERRAIN_RES = 64
_TERRAIN_SIZE_M = 100.0


def _skip_no_genesis():
    if not _GENESIS_AVAILABLE:
        pytest.skip("Genesis not installed — skipping real-engine integration tests")


# ──────────────────────────────────────────────────────────────────────────────
# Flat terrain generator stub (replaces LunarTerrainGenerator until implemented)
# ──────────────────────────────────────────────────────────────────────────────

class _FlatTerrainGenerator:
    """Minimal stub TerrainGenerator that produces a flat height field.

    Used in integration tests while the full LunarTerrainGenerator is still
    pending implementation.  Passes the height field straight to the engine.
    """

    def generate(self, config):
        from moon_rover.environment.terrain.generator import TerrainOutput
        size = _TERRAIN_RES
        hf   = np.zeros((size, size), dtype=np.float32)
        sm   = np.zeros((size, size), dtype=np.float32)
        nm   = np.zeros((size, size, 3), dtype=np.float32)
        nm[:, :, 2] = 1.0  # all normals pointing up
        nav  = np.ones((size, size), dtype=np.uint8)
        return TerrainOutput(
            height_field=hf,
            slope_map=sm,
            normal_map=nm,
            rock_positions=[],
            crater_list=[],
            nav_mesh=nav,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Minimal URDF stubs (replace full GenesisURDFBuilder until Task f4611142)
# ──────────────────────────────────────────────────────────────────────────────

def _simple_box_urdf(name: str, mass: float = 10.0, size: str = "1.0 0.5 0.3") -> str:
    sx, sy, sz = (float(part) for part in size.split())
    ixx = (mass / 12.0) * ((sy * sy) + (sz * sz))
    iyy = (mass / 12.0) * ((sx * sx) + (sz * sz))
    izz = (mass / 12.0) * ((sx * sx) + (sy * sy))
    return f"""<?xml version="1.0"?>
<robot name="{name}">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <visual><geometry><box size="{size}"/></geometry></visual>
    <collision><geometry><box size="{size}"/></geometry></collision>
  </link>
</robot>
"""


class _StubURDFBuilder:
    """Produces simple box-body URDF strings for integration testing.

    Replaces GenesisURDFBuilder (Task f4611142) so the full SceneComposer
    pipeline can be exercised without the full URDF generation system.
    """

    def build_rover(self, config: dict) -> str:
        mass = float(config.get("mass_kg", 60.0))
        return _simple_box_urdf("rover", mass=mass, size="1.5 0.8 0.4")

    def build_antenna(self) -> str:
        return _simple_box_urdf("antenna", mass=4.5, size="0.4 0.4 0.05")

    def build_moonbase(self) -> str:
        return _simple_box_urdf("moonbase", mass=5000.0, size="10.0 8.0 4.0")

    def validate(self, urdf_xml: str, stage) -> list:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Cable composer stub (limits link count for Genesis snode budget)
# ──────────────────────────────────────────────────────────────────────────────

# Genesis's internal snode limit is hit when too many physics entities are
# registered.  120 cable links (60 m / 0.5 m per link) exceeds this during
# build_scene().  The stub forces link_length_m = _CI_LINK_LEN so at most
# _CI_MAX_LINKS links are created per rover, keeping the entity count small.
_CI_LINK_LEN = 10.0   # 10 m per link → 6 links for a 60 m cable
_CI_MAX_LINKS = 10    # safety cap


class _LimitedCableComposer:
    """CableComposer wrapper that caps link count for integration tests.

    Overrides link_length_m on each rover's CableConfig so the full
    cable pre-allocation path is exercised without hitting Genesis's
    internal sparse-node limit during build_scene().
    """

    def compose(self, rover_specs, mission_cfg, engine):
        import math
        from moon_rover.core.scene.cable_composer import CableComposer
        from moon_rover.cable.chain import CableConfig

        # Patch each rover spec's cable_config to use large link segments
        for rs in rover_specs:
            cfg = rs.cable_config
            capped_links = min(
                math.ceil(cfg.total_length_m / _CI_LINK_LEN),
                _CI_MAX_LINKS,
            )
            rs.cable_config = CableConfig(
                link_length_m=_CI_LINK_LEN,
                link_diameter_m=cfg.link_diameter_m,
                link_mass_kg=cfg.link_mass_kg * (_CI_LINK_LEN / cfg.link_length_m),
                total_length_m=capped_links * _CI_LINK_LEN,
                joint_damping=cfg.joint_damping,
                joint_stiffness=cfg.joint_stiffness,
                terrain_friction=cfg.terrain_friction,
                max_tension_n=cfg.max_tension_n,
                bend_radius_min_m=cfg.bend_radius_min_m,
                voltage_dc=cfg.voltage_dc,
                resistance_per_m=cfg.resistance_per_m,
            )

        return CableComposer().compose(rover_specs, mission_cfg, engine)


# ──────────────────────────────────────────────────────────────────────────────
# Shared setup helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_engine():
    """Instantiate and configure a real GenesisPhysicsEngine (CPU, no viewer)."""
    from moon_rover.core.physics.engine import GenesisPhysicsEngine, GenesisConfig
    engine = GenesisPhysicsEngine()
    config = GenesisConfig.from_yaml(_PHYSICS_YAML)
    engine.configure(config, show_viewer=False)
    return engine


def _make_composer():
    """Return a GenesisSceneComposer loaded with all fixture configs.

    Injects stubs for not-yet-implemented subsystems:
      - FlatTerrainGenerator  (replaces LunarTerrainGenerator)
      - StubURDFBuilder       (replaces GenesisURDFBuilder, Task f4611142)
      - LimitedCableComposer  (caps cable link count to avoid Genesis snode limit)
    so the full SceneComposer pipeline can be exercised end-to-end.
    """
    from moon_rover.core.scene.composer import GenesisSceneComposer
    from moon_rover.core.scene.terrain_composer import TerrainComposer
    from moon_rover.core.scene.moonbase_composer import MoonbaseComposer
    from moon_rover.core.scene.rover_composer import RoverComposer
    from moon_rover.core.scene.antenna_composer import AntennaComposer

    stub_builder = _StubURDFBuilder()

    c = GenesisSceneComposer(
        terrain_composer=TerrainComposer(generator=_FlatTerrainGenerator()),
        moonbase_composer=MoonbaseComposer(urdf_builder=stub_builder),
        rover_composer=RoverComposer(urdf_builder=stub_builder),
        antenna_composer=AntennaComposer(urdf_builder=stub_builder),
        cable_composer=_LimitedCableComposer(),
    )
    c.load_scene_config(_SCENE_YAML)
    c.load_rover_config(_ROVER_YAML)
    c.load_mission_config(_MISSION_YAML)
    c.load_physics_config(_PHYSICS_YAML)
    c.load_sensors_config(_SENSORS_YAML)
    return c


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def _add_ground_plane(engine) -> None:
    """Add a minimal flat ground plane so build_scene() has at least one entity."""
    engine.add_terrain_entity(
        name="ground",
        height_field=np.zeros((16, 16), dtype=np.float32),
        size=[_TERRAIN_SIZE_M, _TERRAIN_SIZE_M],
    )


class TestGenesisEngineBootstrap:
    """Verify the engine can be configured and torn down cleanly."""

    def test_configure_and_teardown(self):
        _skip_no_genesis()
        engine = _make_engine()
        _add_ground_plane(engine)
        engine.build_scene()
        engine.teardown()

    def test_phase_transitions(self):
        _skip_no_genesis()
        from moon_rover.core.physics.engine import ScenePhase
        engine = _make_engine()
        assert engine.get_phase() == ScenePhase.CONSTRUCTION
        _add_ground_plane(engine)
        engine.build_scene()
        assert engine.get_phase() == ScenePhase.SIMULATION
        engine.teardown()


class TestValidateConfigsWithRealEngine:
    """Validate configs pass cleanly before composing."""

    def test_validate_returns_no_errors(self):
        _skip_no_genesis()
        c = _make_composer()
        errors = c.validate_configs()
        schema_errors = [e for e in errors if not e.startswith("scene.rover.sensor_config_ref")]
        assert schema_errors == [], f"Unexpected validation errors:\n" + "\n".join(errors)


class TestComposePipelineWithRealGenesis:
    """Full end-to-end compose_scene() with the real GenesisPhysicsEngine."""

    @pytest.fixture(scope="class")
    def composed_scene(self):
        _skip_no_genesis()
        engine = _make_engine()
        composer = _make_composer()
        scene = composer.compose_scene(engine)
        yield scene, engine
        try:
            engine.teardown()
        except Exception:
            pass

    def test_scene_terrain_registered(self, composed_scene):
        scene, engine = composed_scene
        assert scene.terrain is not None
        assert scene.terrain.entity_handle is not None

    def test_terrain_height_field_shape(self, composed_scene):
        scene, _ = composed_scene
        hf = scene.terrain.height_field
        assert isinstance(hf, np.ndarray)
        assert hf.dtype == np.float32
        assert hf.ndim == 2

    def test_nav_mesh_matches_height_field(self, composed_scene):
        scene, _ = composed_scene
        assert scene.terrain.nav_mesh.shape == scene.terrain.height_field.shape

    def test_moonbase_registered(self, composed_scene):
        scene, engine = composed_scene
        assert scene.moonbase is not None
        assert scene.moonbase.entity_handle is not None

    def test_rovers_registered(self, composed_scene):
        scene, engine = composed_scene
        assert len(scene.rovers) >= 1
        for rs in scene.rovers:
            assert rs.entity_handle is not None, \
                f"rover {rs.rover_id!r} has no entity handle"

    def test_antennas_registered(self, composed_scene):
        scene, _ = composed_scene
        assert len(scene.antennas) == len(scene.rovers)
        for ant in scene.antennas:
            assert ant.entity_handle is not None

    def test_cable_links_pre_allocated(self, composed_scene):
        scene, _ = composed_scene
        for cs in scene.cables:
            expected = math.ceil(cs.config.total_length_m / cs.config.link_length_m)
            assert len(cs.link_entity_handles) == expected, \
                f"cable for {cs.rover_id!r}: expected {expected} links, got {len(cs.link_entity_handles)}"
            assert all(h is not None for h in cs.link_entity_handles)

    def test_engine_in_simulation_phase_after_compose(self, composed_scene):
        _, engine = composed_scene
        from moon_rover.core.physics.engine import ScenePhase
        assert engine.get_phase() == ScenePhase.SIMULATION

    def test_entity_handles_in_physics_handles_dict(self, composed_scene):
        scene, _ = composed_scene
        assert "terrain" in scene.physics_entity_handles
        assert "moonbase" in scene.physics_entity_handles
        for rs in scene.rovers:
            assert rs.rover_id in scene.physics_entity_handles

    def test_no_any_typed_fields_in_rover_spec(self, composed_scene):
        """Every typed field on SceneRoverSpec must carry its concrete type."""
        from moon_rover.core.scene.specs import DriveType
        from moon_rover.cable.chain import CableConfig

        scene, _ = composed_scene
        rs = scene.rovers[0]
        assert isinstance(rs.drive_type, DriveType), \
            f"drive_type is {type(rs.drive_type)}, not DriveType"
        assert isinstance(rs.cable_config, CableConfig), \
            f"cable_config is {type(rs.cable_config)}, not CableConfig"
        assert len(rs.initial_pose) == 7, \
            "initial_pose must be 7-tuple (x,y,z,qx,qy,qz,qw)"
        assert isinstance(rs.sensor_handles, dict)


class TestGenesisTerrainQueries:
    """Verify terrain height/normal queries work post-compose."""

    @pytest.fixture(scope="class")
    def engine_with_scene(self):
        _skip_no_genesis()
        engine = _make_engine()
        composer = _make_composer()
        composer.compose_scene(engine)
        yield engine
        try:
            engine.teardown()
        except Exception:
            pass

    def test_terrain_height_at_origin(self, engine_with_scene):
        engine = engine_with_scene
        get_height = getattr(engine, "get_terrain_height", None)
        if get_height is None:
            pytest.skip("Engine has no get_terrain_height — terrain query API not available")
        h = get_height(0.0, 0.0)
        assert isinstance(h, float)

    def test_terrain_normal_at_origin(self, engine_with_scene):
        engine = engine_with_scene
        get_normal = getattr(engine, "get_terrain_normal", None)
        if get_normal is None:
            pytest.skip("Engine has no get_terrain_normal — terrain query API not available")
        n = get_normal(0.0, 0.0)
        assert len(n) == 3
        norm = math.sqrt(sum(v**2 for v in n))
        assert abs(norm - 1.0) < 1e-4


class TestGenesisSimulationStep:
    """Run a small number of physics steps after composing the scene."""

    def test_ten_steps_do_not_crash(self):
        _skip_no_genesis()
        engine = _make_engine()
        composer = _make_composer()
        composer.compose_scene(engine)

        import yaml
        with open(_PHYSICS_YAML) as fh:
            phys = yaml.safe_load(fh)
        dt = float(phys["timestep"]["seconds"])

        for _ in range(10):
            engine.step(dt)

        engine.teardown()

    def test_rover_pose_queryable_after_step(self):
        _skip_no_genesis()
        engine = _make_engine()
        composer = _make_composer()
        scene = composer.compose_scene(engine)

        import yaml
        with open(_PHYSICS_YAML) as fh:
            phys = yaml.safe_load(fh)
        dt = float(phys["timestep"]["seconds"])
        engine.step(dt)

        get_pose = getattr(engine, "get_body_pose", None)
        if get_pose is None:
            pytest.skip("Engine has no get_body_pose")

        rover_id = scene.rovers[0].rover_id
        pos, quat = get_pose(rover_id)
        assert len(pos) == 3
        assert len(quat) == 4

        engine.teardown()
