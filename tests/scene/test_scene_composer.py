"""Integration tests for GenesisSceneComposer.

Covers the full compose_scene() pipeline, all YAML loaders, validate_configs(),
and multi-rover scenarios.  No real Genesis engine or URDF builder is imported —
all external dependencies are replaced with mock objects.

Run with:
    pytest tests/scene/test_scene_composer.py -v
"""

from __future__ import annotations

import math
import os
import textwrap
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixture YAML paths (real project configs — no synthesis needed)
# ──────────────────────────────────────────────────────────────────────────────

_CONFIGS = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs"
)
_SCENE_YAML   = os.path.join(_CONFIGS, "scene.yaml")
_ROVER_YAML   = os.path.join(_CONFIGS, "rover.yaml")
_MISSION_YAML = os.path.join(_CONFIGS, "mission.yaml")
_PHYSICS_YAML = os.path.join(_CONFIGS, "physics.yaml")
_SENSORS_YAML = os.path.join(_CONFIGS, "sensors.yaml")


# ──────────────────────────────────────────────────────────────────────────────
# Mock helpers
# ──────────────────────────────────────────────────────────────────────────────

class _PhaseTracker:
    """Tracks physics engine phase transitions."""
    CONSTRUCTION = "construction"
    SIMULATION   = "simulation"

    def __init__(self, start="construction"):
        self._phase = start
        self.build_scene_calls = 0

    def get_phase(self):
        from moon_rover.core.physics.engine import ScenePhase
        return ScenePhase(self._phase)

    def build_scene(self):
        self.build_scene_calls += 1
        self._phase = self.SIMULATION


class MockPhysicsEngine:
    """Records all calls made during scene construction for assertion."""

    def __init__(self, start_phase="construction"):
        self._tracker = _PhaseTracker(start_phase)
        self.entity_calls: List[Dict[str, Any]] = []
        self.terrain_calls: List[Dict[str, Any]] = []
        self.raycaster_calls: List[Dict[str, Any]] = []
        self.pose_calls: List[Dict[str, Any]] = []
        self._handle_counter = 0

    def _next_handle(self) -> object:
        self._handle_counter += 1
        return object()  # opaque unique handle

    def get_phase(self):
        return self._tracker.get_phase()

    def build_scene(self):
        self._tracker.build_scene()

    def add_entity(self, *, name: str, morph: str, material, entity_type: str):
        handle = self._next_handle()
        self.entity_calls.append(
            {"name": name, "morph": morph, "material": material, "entity_type": entity_type}
        )
        return handle

    def add_terrain_entity(self, *, name: str, height_field, size: list):
        handle = self._next_handle()
        self.terrain_calls.append({"name": name, "height_field": height_field, "size": size})
        return handle

    def register_raycaster(self, *, name: str, link_entity: str, link_idx: int, pattern_config: dict):
        handle = self._next_handle()
        self.raycaster_calls.append(
            {"name": name, "link_entity": link_entity, "link_idx": link_idx}
        )
        return handle

    def set_body_pose(self, name: str, position, quaternion):
        self.pose_calls.append(
            {"name": name, "position": list(position), "quaternion": list(quaternion)}
        )

    @property
    def build_scene_called_count(self) -> int:
        return self._tracker.build_scene_calls


def _make_mock_terrain_output(size: int = 64):
    """Return a minimal TerrainOutput-like object."""
    from moon_rover.environment.terrain.generator import TerrainOutput
    hf = np.zeros((size, size), dtype=np.float32)
    sm = np.zeros((size, size), dtype=np.float32)
    nm = np.zeros((size, size, 3), dtype=np.float32)
    nav = np.ones((size, size), dtype=np.uint8)
    return TerrainOutput(
        height_field=hf,
        slope_map=sm,
        normal_map=nm,
        rock_positions=[],
        crater_list=[],
        nav_mesh=nav,
    )


def _mock_material():
    """Return a mock MaterialProperties object."""
    mat = MagicMock()
    mat.name = "lunar_regolith"
    return mat


def _mock_terrain_composer(engine, scene_cfg):
    """Return a pre-built SceneTerrainSpec with a valid entity handle."""
    from moon_rover.core.scene.terrain_composer import TerrainComposer
    from moon_rover.core.scene.specs import SceneTerrainSpec
    from moon_rover.environment.terrain.generator import TerrainConfig

    output = _make_mock_terrain_output()
    handle = engine.add_terrain_entity(
        name="terrain",
        height_field=output.height_field,
        size=[100.0, 100.0],
    )
    cfg = TerrainConfig(
        seed=42,
        size_m=100.0,
        fBm_octaves=8,
        fBm_amplitude=2.0,
        crater_params={"count": 0},
        rock_density=0.0,
        rille_enabled=False,
        moonbase_position=(0.0, 0.0, 0.0),
        resolution=64,
    )
    return SceneTerrainSpec(
        config=cfg,
        height_field=output.height_field,
        slope_map=output.slope_map,
        normal_map=output.normal_map,
        rock_positions=[],
        crater_list=[],
        nav_mesh=output.nav_mesh,
        material=_mock_material(),
        size_m=100.0,
        entity_handle=handle,
    )


def _mock_moonbase_composer(engine):
    from moon_rover.core.scene.specs import (
        SceneMoonbaseSpec, BeaconConfig, DockingConfig,
    )
    from moon_rover.moonbase.base import MoonbaseConfig

    handle = engine.add_entity(
        name="moonbase", morph="<urdf/>", material=None, entity_type="fixed"
    )
    config = MoonbaseConfig(
        habitat_dims_m=(10.0, 8.0, 4.0),
        solar_array_config=None,
        power_bus_voltage=48.0,
        comm_tower_height_m=5.0,
        num_docking_bays=2,
        charge_rate_w=500.0,
        num_cable_reels=8,
        num_antennas=20,
        landing_pad_radius_m=15.0,
    )
    return SceneMoonbaseSpec(
        config=config,
        world_position=(0.0, 0.0, 0.0),
        beacon=BeaconConfig(
            position_xyz=(0.0, 0.0, 0.0),
            frequency_hz=1.0,
            signal_strength=1.0,
            communication_range_m=1000.0,
        ),
        docking=DockingConfig(
            num_ports=2,
            port_positions=[(2.0, 0.0, 0.0)],
            charge_rate_w=500.0,
        ),
        entity_handle=handle,
    )


def _mock_rover_spec(engine, rover_id: str = "rover_001", profile_key: str = "four_wheel_skid"):
    from moon_rover.core.scene.specs import SceneRoverSpec, DriveType
    from moon_rover.cable.chain import CableConfig

    handle = engine.add_entity(
        name=rover_id, morph="<urdf/>", material=None, entity_type="articulated"
    )
    cable_cfg = CableConfig(
        link_length_m=0.5,
        link_diameter_m=0.01,
        link_mass_kg=0.075,
        total_length_m=60.0,
        joint_damping=0.1,
        joint_stiffness=100.0,
        terrain_friction=0.4,
        max_tension_n=500.0,
        bend_radius_min_m=0.05,
        voltage_dc=48.0,
        resistance_per_m=0.005,
    )
    return SceneRoverSpec(
        rover_id=rover_id,
        urdf_str="<robot name='test'><link name='base'/></robot>",
        drive_type=DriveType.FOUR_WHEEL_SKID,
        initial_pose=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0),
        cable_config=cable_cfg,
        sensor_handles={},
        entity_handle=handle,
        num_wheels=4,
        mass_kg=60.0,
        wheel_radius_m=0.35,
    )


def _mock_antenna_spec(engine, rover_id: str):
    from moon_rover.core.scene.specs import SceneAntennaSpec
    from moon_rover.antenna.system import AntennaConfig, AntennaState

    handle = engine.add_entity(
        name=f"{rover_id}_antenna", morph="<urdf/>", material=None, entity_type="rigid"
    )
    cfg = AntennaConfig(
        base_plate_m=(0.4, 0.4, 0.05),
        base_mass_kg=2.5,
        mast_height_m=1.2,
        mast_radius_m=0.02,
        mast_mass_kg=1.0,
        dish_diameter_m=0.6,
        dish_mass_kg=0.8,
        connector_mass_kg=0.2,
        total_mass_kg=4.5,
    )
    return SceneAntennaSpec(
        rover_id=rover_id,
        antenna_config=cfg,
        initial_state=AntennaState.STORED,
        cable_attachment_entity_name=rover_id,
        entity_handle=handle,
    )


def _mock_cable_spec(engine, rover_spec):
    import math
    from moon_rover.core.scene.specs import SceneCableSpec

    cfg = rover_spec.cable_config
    num_links = math.ceil(cfg.total_length_m / cfg.link_length_m)
    handles = []
    for i in range(num_links):
        handles.append(
            engine.add_entity(
                name=f"{rover_spec.rover_id}_cable_link_{i:04d}",
                morph="<urdf/>",
                material=None,
                entity_type="rigid",
            )
        )
    return SceneCableSpec(
        rover_id=rover_spec.rover_id,
        cable_id="cable_main",
        config=cfg,
        link_entity_handles=handles,
    )


def _build_scene_composer_with_mocks(engine, rover_ids=("rover_001",)):
    """Return a GenesisSceneComposer wired with lightweight mock sub-composers."""
    from moon_rover.core.scene.composer import GenesisSceneComposer

    # Build the specs using the mock engine (so calls are recorded)
    terrain_spec = _mock_terrain_composer(engine, scene_cfg={})
    moonbase_spec = _mock_moonbase_composer(engine)
    rover_specs = [_mock_rover_spec(engine, rid) for rid in rover_ids]
    antenna_specs = [_mock_antenna_spec(engine, rid) for rid in rover_ids]
    cable_specs = [_mock_cable_spec(engine, rs) for rs in rover_specs]

    class _StaticTerrainComposer:
        def compose(self, *args, **kwargs): return terrain_spec

    class _StaticMoonbaseComposer:
        def compose(self, *args, **kwargs): return moonbase_spec

    class _StaticRoverComposer:
        def compose(self, *args, **kwargs): return rover_specs

    class _StaticAntennaComposer:
        def compose(self, *args, **kwargs): return antenna_specs

    class _StaticCableComposer:
        def compose(self, *args, **kwargs): return cable_specs

    class _NoopSensorRegistrar:
        def register(self, *args, **kwargs): pass

    class _MockMaterialLib:
        def get_material(self, name): return _mock_material()
        def validate_all_referenced_materials(self, names): return []

    return GenesisSceneComposer(
        terrain_composer=_StaticTerrainComposer(),
        moonbase_composer=_StaticMoonbaseComposer(),
        rover_composer=_StaticRoverComposer(),
        antenna_composer=_StaticAntennaComposer(),
        cable_composer=_StaticCableComposer(),
        sensor_registrar=_NoopSensorRegistrar(),
        material_library=_MockMaterialLib(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Tests: YAML loaders
# ──────────────────────────────────────────────────────────────────────────────

class TestYAMLLoaders:
    def test_load_scene_config_valid(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        c = GenesisSceneComposer()
        cfg = c.load_scene_config(_SCENE_YAML)
        assert "terrain" in cfg
        assert "moonbase" in cfg

    def test_load_rover_config_valid(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        c = GenesisSceneComposer()
        cfg = c.load_rover_config(_ROVER_YAML)
        assert "profiles" in cfg
        assert len(cfg["profiles"]) >= 3

    def test_load_mission_config_valid(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        c = GenesisSceneComposer()
        cfg = c.load_mission_config(_MISSION_YAML)
        assert "rovers" in cfg
        assert len(cfg["rovers"]) >= 1

    def test_load_physics_config_valid(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        c = GenesisSceneComposer()
        cfg = c.load_physics_config(_PHYSICS_YAML)
        assert "gravity" in cfg
        assert "timestep" in cfg

    def test_load_sensors_config_valid(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        c = GenesisSceneComposer()
        cfg = c.load_sensors_config(_SENSORS_YAML)
        assert "lidar" in cfg

    def test_load_sensors_config_none(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        c = GenesisSceneComposer()
        result = c.load_sensors_config(None)
        assert result is None

    def test_load_missing_file_raises_file_not_found(self, tmp_path):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        c = GenesisSceneComposer()
        with pytest.raises(FileNotFoundError):
            c.load_scene_config(str(tmp_path / "nonexistent.yaml"))

    def test_load_invalid_yaml_raises_value_error(self, tmp_path):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        bad = tmp_path / "bad.yaml"
        bad.write_text("key: [unclosed bracket", encoding="utf-8")
        c = GenesisSceneComposer()
        with pytest.raises(ValueError, match="Invalid YAML"):
            c.load_scene_config(str(bad))


# ──────────────────────────────────────────────────────────────────────────────
# Tests: validate_configs()
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateConfigs:
    def _loaded_composer(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer

        class _NullMaterialLib:
            def validate_all_referenced_materials(self, names): return []

        c = GenesisSceneComposer(material_library=_NullMaterialLib())
        c.load_scene_config(_SCENE_YAML)
        c.load_rover_config(_ROVER_YAML)
        c.load_mission_config(_MISSION_YAML)
        c.load_physics_config(_PHYSICS_YAML)
        c.load_sensors_config(_SENSORS_YAML)
        return c

    def test_valid_configs_return_empty_errors(self):
        c = self._loaded_composer()
        errors = c.validate_configs()
        # JSON schema errors for valid configs should be empty
        schema_errors = [e for e in errors if e.startswith("[")]
        assert schema_errors == [], f"Unexpected schema errors: {schema_errors}"

    def test_missing_required_config_returns_error(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        c = GenesisSceneComposer()
        c.load_scene_config(_SCENE_YAML)
        # rover, mission, physics not loaded
        errors = c.validate_configs()
        assert errors, "Expected errors for missing configs"
        assert any("not loaded" in e for e in errors)

    def test_bad_type_ref_returns_error(self, tmp_path):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        import yaml

        bad_scene = tmp_path / "scene.yaml"
        # Write minimal scene with a broken type_ref
        import yaml as _yaml
        with open(_SCENE_YAML) as fh:
            scene_data = _yaml.safe_load(fh)
        scene_data.setdefault("rover", {})["type_ref"] = "nonexistent_profile"
        bad_scene.write_text(_yaml.dump(scene_data), encoding="utf-8")

        class _NullMaterialLib:
            def validate_all_referenced_materials(self, names): return []

        c = GenesisSceneComposer(material_library=_NullMaterialLib())
        c.load_scene_config(str(bad_scene))
        c.load_rover_config(_ROVER_YAML)
        c.load_mission_config(_MISSION_YAML)
        c.load_physics_config(_PHYSICS_YAML)
        errors = c.validate_configs()
        assert any("type_ref" in e and "nonexistent_profile" in e for e in errors), errors

    def test_unknown_material_returns_error(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer

        class _StrictMaterialLib:
            def validate_all_referenced_materials(self, names):
                return names  # report all as missing

        c = GenesisSceneComposer(material_library=_StrictMaterialLib())
        c.load_scene_config(_SCENE_YAML)
        c.load_rover_config(_ROVER_YAML)
        c.load_mission_config(_MISSION_YAML)
        c.load_physics_config(_PHYSICS_YAML)
        errors = c.validate_configs()
        assert any("MaterialLibrary" in e for e in errors), errors

    def test_unmatched_cable_id_returns_error(self, tmp_path):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        import yaml as _yaml

        with open(_MISSION_YAML) as fh:
            mission_data = _yaml.safe_load(fh)
        # Give the first rover a cable_id that doesn't match the top-level cable
        mission_data["cable"]["cable_id"] = "cable_main"
        for rv in mission_data.get("rovers", []):
            rv["cable_id"] = "cable_other"

        bad_mission = tmp_path / "mission.yaml"
        bad_mission.write_text(_yaml.dump(mission_data), encoding="utf-8")

        class _NullMaterialLib:
            def validate_all_referenced_materials(self, names): return []

        c = GenesisSceneComposer(material_library=_NullMaterialLib())
        c.load_scene_config(_SCENE_YAML)
        c.load_rover_config(_ROVER_YAML)
        c.load_mission_config(str(bad_mission))
        c.load_physics_config(_PHYSICS_YAML)
        errors = c.validate_configs()
        assert any("cable_id" in e for e in errors), errors

    def test_errors_collected_not_fail_fast(self, tmp_path):
        """Multiple violations should all appear in the error list."""
        from moon_rover.core.scene.composer import GenesisSceneComposer
        import yaml as _yaml

        with open(_MISSION_YAML) as fh:
            mission_data = _yaml.safe_load(fh)
        mission_data["cable"]["cable_id"] = "cable_main"
        for rv in mission_data.get("rovers", []):
            rv["cable_id"] = "cable_other"
            rv["rover_profile"] = "nonexistent_profile"  # second violation

        bad_mission = tmp_path / "mission.yaml"
        bad_mission.write_text(_yaml.dump(mission_data), encoding="utf-8")

        class _StrictMaterialLib:
            def validate_all_referenced_materials(self, names): return names

        c = GenesisSceneComposer(material_library=_StrictMaterialLib())
        c.load_scene_config(_SCENE_YAML)
        c.load_rover_config(_ROVER_YAML)
        c.load_mission_config(str(bad_mission))
        c.load_physics_config(_PHYSICS_YAML)
        errors = c.validate_configs()
        assert len(errors) >= 2, f"Expected ≥2 errors, got: {errors}"


# ──────────────────────────────────────────────────────────────────────────────
# Tests: compose_scene() with mock engine
# ──────────────────────────────────────────────────────────────────────────────

class TestComposeScene:
    def _composer_with_configs(self, engine, rover_ids=("rover_001",)):
        c = _build_scene_composer_with_mocks(engine, rover_ids)
        c.load_scene_config(_SCENE_YAML)
        c.load_rover_config(_ROVER_YAML)
        c.load_mission_config(_MISSION_YAML)
        c.load_physics_config(_PHYSICS_YAML)
        c.load_sensors_config(_SENSORS_YAML)
        return c

    def test_scene_has_all_required_fields(self):
        engine = MockPhysicsEngine()
        c = self._composer_with_configs(engine)
        scene = c.compose_scene(engine)

        assert scene.terrain is not None
        assert scene.moonbase is not None
        assert scene.rovers is not None and len(scene.rovers) >= 1
        assert scene.antennas is not None and len(scene.antennas) >= 1
        assert scene.cables is not None and len(scene.cables) >= 1

    def test_build_scene_called_exactly_once(self):
        engine = MockPhysicsEngine()
        c = self._composer_with_configs(engine)
        c.compose_scene(engine)
        assert engine.build_scene_called_count == 1

    def test_engine_construction_phase_required(self):
        engine = MockPhysicsEngine(start_phase="simulation")
        c = self._composer_with_configs(engine)
        with pytest.raises(RuntimeError, match="CONSTRUCTION"):
            c.compose_scene(engine)

    def test_missing_config_raises_runtime_error(self):
        from moon_rover.core.scene.composer import GenesisSceneComposer
        engine = MockPhysicsEngine()
        c = GenesisSceneComposer()
        c.load_scene_config(_SCENE_YAML)
        # rover, mission, physics not loaded
        with pytest.raises(RuntimeError, match="not loaded"):
            c.compose_scene(engine)

    def test_validate_errors_raise_value_error(self, tmp_path):
        """compose_scene() must fail if validate_configs() returns errors."""
        class _StrictMaterialLib:
            def validate_all_referenced_materials(self, names): return names

        engine = MockPhysicsEngine()
        c = _build_scene_composer_with_mocks(engine)
        c._material_library = _StrictMaterialLib()  # inject strict lib
        c.load_scene_config(_SCENE_YAML)
        c.load_rover_config(_ROVER_YAML)
        c.load_mission_config(_MISSION_YAML)
        c.load_physics_config(_PHYSICS_YAML)
        c.load_sensors_config(_SENSORS_YAML)

        with pytest.raises(ValueError, match="Config validation failed"):
            c.compose_scene(engine)

    def test_terrain_fields_populated(self):
        engine = MockPhysicsEngine()
        c = self._composer_with_configs(engine)
        scene = c.compose_scene(engine)

        assert scene.terrain.height_field is not None
        assert scene.terrain.height_field.dtype == np.float32
        assert scene.terrain.height_field.ndim == 2
        assert scene.terrain.nav_mesh is not None
        assert scene.terrain.nav_mesh.dtype == np.uint8
        assert scene.terrain.nav_mesh.shape == scene.terrain.height_field.shape

    def test_terrain_entity_registered_with_engine(self):
        engine = MockPhysicsEngine()
        c = self._composer_with_configs(engine)
        c.compose_scene(engine)
        terrain_calls = [t for t in engine.terrain_calls if t["name"] == "terrain"]
        assert len(terrain_calls) == 1

    def test_initial_layout_places_entities_readably(self):
        engine = MockPhysicsEngine()
        c = self._composer_with_configs(engine)
        c.compose_scene(engine)

        pose_by_name = {entry["name"]: entry for entry in engine.pose_calls}

        assert pose_by_name["moonbase"]["position"] == [0.0, 0.0, 0.0]
        assert pose_by_name["rover_001"]["position"] == [0.0, 0.0, 0.5]
        assert pose_by_name["rover_001_antenna"]["position"][2] > 0.5
        assert pose_by_name["rover_001_cable_link_0000"]["position"][0] >= 200.0

    def test_sub_composer_failure_includes_stage_name(self):
        class _BoomTerrainComposer:
            def compose(self, *a, **kw):
                raise RuntimeError("disk full")

        engine = MockPhysicsEngine()
        c = _build_scene_composer_with_mocks(engine)
        c._terrain_composer = _BoomTerrainComposer()
        c.load_scene_config(_SCENE_YAML)
        c.load_rover_config(_ROVER_YAML)
        c.load_mission_config(_MISSION_YAML)
        c.load_physics_config(_PHYSICS_YAML)
        c.load_sensors_config(_SENSORS_YAML)

        with pytest.raises(RuntimeError, match="TerrainComposer"):
            c.compose_scene(engine)


# ──────────────────────────────────────────────────────────────────────────────
# Tests: multi-rover composition
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiRoverComposition:
    def _multi_rover_composer(self, engine, rover_ids=("rover_001", "rover_002")):
        c = _build_scene_composer_with_mocks(engine, rover_ids)
        c.load_scene_config(_SCENE_YAML)
        c.load_rover_config(_ROVER_YAML)
        c.load_mission_config(_MISSION_YAML)
        c.load_physics_config(_PHYSICS_YAML)
        c.load_sensors_config(_SENSORS_YAML)
        return c

    def test_two_rovers_produce_two_rover_specs(self):
        engine = MockPhysicsEngine()
        scene = self._multi_rover_composer(engine).compose_scene(engine)
        assert len(scene.rovers) == 2
        assert {rs.rover_id for rs in scene.rovers} == {"rover_001", "rover_002"}

    def test_two_rovers_produce_two_antenna_specs(self):
        engine = MockPhysicsEngine()
        scene = self._multi_rover_composer(engine).compose_scene(engine)
        assert len(scene.antennas) == 2

    def test_two_rovers_produce_two_cable_specs(self):
        engine = MockPhysicsEngine()
        scene = self._multi_rover_composer(engine).compose_scene(engine)
        assert len(scene.cables) == 2

    def test_cable_rover_ids_match_rover_specs(self):
        engine = MockPhysicsEngine()
        scene = self._multi_rover_composer(engine).compose_scene(engine)
        rover_ids = {rs.rover_id for rs in scene.rovers}
        cable_rover_ids = {cs.rover_id for cs in scene.cables}
        assert rover_ids == cable_rover_ids

    def test_physics_entity_handles_includes_all_rovers(self):
        engine = MockPhysicsEngine()
        scene = self._multi_rover_composer(engine).compose_scene(engine)
        assert "rover_001" in scene.physics_entity_handles
        assert "rover_002" in scene.physics_entity_handles
        assert "terrain" in scene.physics_entity_handles
        assert "moonbase" in scene.physics_entity_handles
