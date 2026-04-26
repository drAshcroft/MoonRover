"""Contract tests: verify the Scene object satisfies each downstream subsystem's interface.

These tests document the EXACT fields and types that each pending subsystem will
access from the Scene returned by compose_scene().  If any field is Any-typed, None,
or the wrong type, the test fails with a descriptive message.

This file serves as a living specification — adding a new subsystem's contract is
a one-line assertion addition.

Run with:
    pytest tests/scene/test_scene_contract.py -v
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from tests.scene.test_scene_composer import (
    MockPhysicsEngine,
    _build_scene_composer_with_mocks,
    _mock_material,
)

_CONFIGS = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
_SCENE_YAML   = os.path.join(_CONFIGS, "scene.yaml")
_ROVER_YAML   = os.path.join(_CONFIGS, "rover.yaml")
_MISSION_YAML = os.path.join(_CONFIGS, "mission.yaml")
_PHYSICS_YAML = os.path.join(_CONFIGS, "physics.yaml")
_SENSORS_YAML = os.path.join(_CONFIGS, "sensors.yaml")


@pytest.fixture(scope="module")
def scene_1rover():
    """Fully composed 1-rover Scene for contract assertions."""
    engine = MockPhysicsEngine()
    c = _build_scene_composer_with_mocks(engine, rover_ids=("rover_001",))
    c.load_scene_config(_SCENE_YAML)
    c.load_rover_config(_ROVER_YAML)
    c.load_mission_config(_MISSION_YAML)
    c.load_physics_config(_PHYSICS_YAML)
    c.load_sensors_config(_SENSORS_YAML)
    return c.compose_scene(engine)


@pytest.fixture(scope="module")
def scene_2rover():
    """Fully composed 2-rover Scene for multi-rover contract assertions."""
    engine = MockPhysicsEngine()
    c = _build_scene_composer_with_mocks(engine, rover_ids=("rover_001", "rover_002"))
    c.load_scene_config(_SCENE_YAML)
    c.load_rover_config(_ROVER_YAML)
    c.load_mission_config(_MISSION_YAML)
    c.load_physics_config(_PHYSICS_YAML)
    c.load_sensors_config(_SENSORS_YAML)
    return c.compose_scene(engine)


# ──────────────────────────────────────────────────────────────────────────────
# Terrain contracts → TerrainGenerator / PathPlanner
# ──────────────────────────────────────────────────────────────────────────────

class TestTerrainContract:
    def test_height_field_is_float32_ndarray(self, scene_1rover):
        hf = scene_1rover.terrain.height_field
        assert isinstance(hf, np.ndarray), \
            f"height_field must be ndarray, got {type(hf)}"
        assert hf.dtype == np.float32, \
            f"height_field must be float32, got {hf.dtype}"

    def test_height_field_is_2d(self, scene_1rover):
        hf = scene_1rover.terrain.height_field
        assert hf.ndim == 2, \
            f"height_field must be 2D (resolution×resolution), got ndim={hf.ndim}"

    def test_nav_mesh_is_uint8_ndarray(self, scene_1rover):
        """PathPlanner contract: nav_mesh must be binary uint8."""
        nm = scene_1rover.terrain.nav_mesh
        assert isinstance(nm, np.ndarray), \
            f"nav_mesh must be ndarray, got {type(nm)}"
        assert nm.dtype == np.uint8, \
            f"nav_mesh must be uint8, got {nm.dtype}"

    def test_nav_mesh_same_shape_as_height_field(self, scene_1rover):
        assert scene_1rover.terrain.nav_mesh.shape == \
               scene_1rover.terrain.height_field.shape, \
            "nav_mesh and height_field must have matching shapes"

    def test_slope_map_is_float32_ndarray(self, scene_1rover):
        sm = scene_1rover.terrain.slope_map
        assert isinstance(sm, np.ndarray)
        assert sm.dtype == np.float32

    def test_normal_map_is_float32_ndarray(self, scene_1rover):
        nm = scene_1rover.terrain.normal_map
        assert isinstance(nm, np.ndarray)
        assert nm.dtype == np.float32
        assert nm.ndim == 3

    def test_terrain_size_m_is_positive_float(self, scene_1rover):
        assert isinstance(scene_1rover.terrain.size_m, float), \
            "size_m must be float for navigation subsystem"
        assert scene_1rover.terrain.size_m > 0.0

    def test_terrain_entity_handle_is_not_none(self, scene_1rover):
        assert scene_1rover.terrain.entity_handle is not None, \
            "terrain entity_handle must not be None — physics subsystem needs it"

    def test_terrain_material_is_not_none(self, scene_1rover):
        assert scene_1rover.terrain.material is not None, \
            "terrain material must be resolved (not None) — physics friction model needs it"


# ──────────────────────────────────────────────────────────────────────────────
# Rover contracts → DriveSystem
# ──────────────────────────────────────────────────────────────────────────────

class TestRoverContract:
    def test_drive_type_is_enum(self, scene_1rover):
        from moon_rover.core.scene.specs import DriveType
        dt = scene_1rover.rovers[0].drive_type
        assert isinstance(dt, DriveType), \
            f"drive_type must be DriveType enum, got {type(dt)} — " \
            "DriveSystem must not compare against raw strings"

    def test_initial_pose_has_7_components(self, scene_1rover):
        pose = scene_1rover.rovers[0].initial_pose
        assert len(pose) == 7, \
            f"initial_pose must be (x,y,z,qx,qy,qz,qw) — 7 elements, got {len(pose)}"

    def test_initial_pose_quaternion_is_unit(self, scene_1rover):
        _, _, _, qx, qy, qz, qw = scene_1rover.rovers[0].initial_pose
        norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        assert abs(norm - 1.0) < 1e-6, \
            f"Quaternion (qx,qy,qz,qw) must be unit, got norm={norm}"

    def test_rover_id_is_string(self, scene_1rover):
        assert isinstance(scene_1rover.rovers[0].rover_id, str)
        assert len(scene_1rover.rovers[0].rover_id) > 0

    def test_mass_kg_is_positive_float(self, scene_1rover):
        m = scene_1rover.rovers[0].mass_kg
        assert isinstance(m, float), f"mass_kg must be float, got {type(m)}"
        assert m > 0.0

    def test_wheel_radius_m_is_positive_float(self, scene_1rover):
        r = scene_1rover.rovers[0].wheel_radius_m
        assert isinstance(r, float)
        assert r > 0.0

    def test_num_wheels_is_positive_int(self, scene_1rover):
        n = scene_1rover.rovers[0].num_wheels
        assert isinstance(n, int)
        assert n >= 2

    def test_urdf_str_is_non_empty_string(self, scene_1rover):
        urdf = scene_1rover.rovers[0].urdf_str
        assert isinstance(urdf, str), f"urdf_str must be str, got {type(urdf)}"
        assert len(urdf) > 0

    def test_urdf_str_parseable_by_yourdfpy(self, scene_1rover):
        """DriveSystem contract: URDF must be parseable."""
        try:
            import yourdfpy
        except ImportError:
            pytest.skip("yourdfpy not installed")

        urdf = scene_1rover.rovers[0].urdf_str
        try:
            # Parse from string
            import io
            yourdfpy.URDF.load(io.StringIO(urdf))
        except Exception as exc:
            pytest.fail(
                f"urdf_str failed yourdfpy parsing — DriveSystem cannot use it: {exc}"
            )

    def test_entity_handle_is_not_none(self, scene_1rover):
        assert scene_1rover.rovers[0].entity_handle is not None, \
            "entity_handle must not be None — physics engine needs it for actuation"

    def test_sensor_handles_is_dict(self, scene_1rover):
        sh = scene_1rover.rovers[0].sensor_handles
        assert isinstance(sh, dict), \
            f"sensor_handles must be dict, got {type(sh)}"


# ──────────────────────────────────────────────────────────────────────────────
# Cable contracts → CableSystem
# ──────────────────────────────────────────────────────────────────────────────

class TestCableContract:
    def test_cable_config_is_cable_config_type(self, scene_1rover):
        from moon_rover.cable.chain import CableConfig
        cfg = scene_1rover.cables[0].config
        assert isinstance(cfg, CableConfig), \
            f"cable config must be CableConfig, got {type(cfg)} — CableSystem needs typed config"

    def test_cable_link_count_matches_math(self, scene_1rover):
        cs = scene_1rover.cables[0]
        expected = math.ceil(cs.config.total_length_m / cs.config.link_length_m)
        actual = len(cs.link_entity_handles)
        assert actual == expected, \
            f"link count must be ceil({cs.config.total_length_m}/{cs.config.link_length_m})={expected}, got {actual}"

    def test_cable_rover_id_matches_rover(self, scene_1rover):
        assert scene_1rover.cables[0].rover_id == scene_1rover.rovers[0].rover_id, \
            "cable.rover_id must match rover.rover_id — CableSystem must know which rover it tethers"

    def test_cable_id_is_string(self, scene_1rover):
        assert isinstance(scene_1rover.cables[0].cable_id, str)
        assert len(scene_1rover.cables[0].cable_id) > 0

    def test_all_link_handles_non_none(self, scene_1rover):
        handles = scene_1rover.cables[0].link_entity_handles
        assert all(h is not None for h in handles), \
            "All cable link handles must be non-None — pre-allocation constraint"

    def test_max_tension_positive(self, scene_1rover):
        assert scene_1rover.cables[0].config.max_tension_n > 0


# ──────────────────────────────────────────────────────────────────────────────
# Antenna contracts → AntennaSystem
# ──────────────────────────────────────────────────────────────────────────────

class TestAntennaContract:
    def test_initial_state_is_stored(self, scene_1rover):
        from moon_rover.antenna.system import AntennaState
        state = scene_1rover.antennas[0].initial_state
        assert state == AntennaState.STORED, \
            f"initial_state must be STORED at compose time, got {state}"

    def test_antenna_rover_id_matches_rover(self, scene_1rover):
        assert scene_1rover.antennas[0].rover_id == scene_1rover.rovers[0].rover_id

    def test_cable_attachment_entity_name_matches_rover_id(self, scene_1rover):
        ant = scene_1rover.antennas[0]
        assert ant.cable_attachment_entity_name == ant.rover_id, \
            "cable_attachment_entity_name must equal rover_id so CableComposer can link them"

    def test_antenna_config_is_antenna_config_type(self, scene_1rover):
        from moon_rover.antenna.system import AntennaConfig
        cfg = scene_1rover.antennas[0].antenna_config
        assert isinstance(cfg, AntennaConfig), \
            f"antenna_config must be AntennaConfig, got {type(cfg)}"

    def test_antenna_entity_handle_is_not_none(self, scene_1rover):
        assert scene_1rover.antennas[0].entity_handle is not None


# ──────────────────────────────────────────────────────────────────────────────
# Moonbase contracts → PowerSystem
# ──────────────────────────────────────────────────────────────────────────────

class TestMoonbaseContract:
    def test_charge_rate_w_is_positive_float(self, scene_1rover):
        cw = scene_1rover.moonbase.config.charge_rate_w
        assert isinstance(cw, float), \
            f"charge_rate_w must be float, got {type(cw)} — PowerSystem needs typed value"
        assert cw > 0.0, "charge_rate_w must be > 0"

    def test_moonbase_world_position_is_3_tuple(self, scene_1rover):
        pos = scene_1rover.moonbase.world_position
        assert len(pos) == 3, "world_position must be 3-tuple (x, y, z)"
        assert all(isinstance(v, float) for v in pos)

    def test_beacon_config_is_beacon_config_type(self, scene_1rover):
        from moon_rover.core.scene.specs import BeaconConfig
        bc = scene_1rover.moonbase.beacon
        assert isinstance(bc, BeaconConfig), f"beacon must be BeaconConfig, got {type(bc)}"

    def test_docking_config_has_positive_ports(self, scene_1rover):
        assert scene_1rover.moonbase.docking.num_ports >= 1

    def test_moonbase_entity_handle_is_not_none(self, scene_1rover):
        assert scene_1rover.moonbase.entity_handle is not None, \
            "moonbase entity_handle must not be None — it's registered as fixed body"


# ──────────────────────────────────────────────────────────────────────────────
# Multi-rover contracts
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiRoverContract:
    def test_rover_cable_pairing_is_bijective(self, scene_2rover):
        """Each rover must have exactly one cable, and vice versa."""
        rover_ids = [rs.rover_id for rs in scene_2rover.rovers]
        cable_rover_ids = [cs.rover_id for cs in scene_2rover.cables]
        assert sorted(rover_ids) == sorted(cable_rover_ids), \
            "rover_ids and cable rover_ids must be identical sets"

    def test_rover_antenna_pairing_is_bijective(self, scene_2rover):
        rover_ids = [rs.rover_id for rs in scene_2rover.rovers]
        antenna_rover_ids = [ant.rover_id for ant in scene_2rover.antennas]
        assert sorted(rover_ids) == sorted(antenna_rover_ids)

    def test_physics_handles_include_all_cables(self, scene_2rover):
        # cable link handles are registered under the zero-padded entity name.
        for cs in scene_2rover.cables:
            first_link_key = f"{cs.rover_id}_cable_link_0000"
            assert first_link_key in scene_2rover.physics_entity_handles, \
                f"Missing cable link handle {first_link_key!r} in physics_entity_handles"
