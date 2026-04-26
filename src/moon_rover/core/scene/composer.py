"""System 3: Scene Composition System — YAML-driven Scene Construction.

This module defines the scene composition interface and the GenesisSceneComposer
concrete implementation.  The composer orchestrates all sub-modules (terrain,
moonbase, rover, antenna, cable, sensors) into a single typed Scene object that
downstream subsystems consume without casting.

Typical usage::

    composer = GenesisSceneComposer()
    composer.load_scene_config("configs/scene.yaml")
    composer.load_rover_config("configs/rover.yaml")
    composer.load_mission_config("configs/mission.yaml")
    composer.load_physics_config("configs/physics.yaml")
    composer.load_sensors_config("configs/sensors.yaml")  # optional

    errors = composer.validate_configs()
    if errors:
        raise ValueError("\\n".join(errors))

    scene = composer.compose_scene(engine)
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import yaml

from moon_rover.core.scene.specs import (
    Scene,
    SceneTerrainSpec,
    SceneRoverSpec,
    SceneAntennaSpec,
    SceneMoonbaseSpec,
    SceneCableSpec,
)

if TYPE_CHECKING:
    from moon_rover.core.physics.engine import PhysicsEngine


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class SceneComposer(ABC):
    """Abstract interface for YAML-driven scene composition.

    Coordinates loading of multiple configuration files (scene, rover, mission,
    physics, sensors) and composing them into a complete, validated simulation
    scene ready for the physics engine.
    """

    @abstractmethod
    def load_scene_config(self, scene_yaml: str) -> Dict[str, Any]:
        """Load scene configuration from a YAML file.

        Expected top-level keys: terrain, sun, rover, mission, moonbase,
        environment, visualization, debug.

        Parameters:
            scene_yaml: Path to scene YAML configuration file.

        Returns:
            Parsed YAML as dictionary.

        Raises:
            FileNotFoundError: If scene_yaml does not exist.
            ValueError: If YAML syntax is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def load_rover_config(self, rover_yaml: str) -> Dict[str, Any]:
        """Load rover configuration from a YAML file.

        Expected top-level keys: profiles (two_wheel_diff, three_wheel_tricycle,
        four_wheel_skid), arm, power, structure, debug.

        Parameters:
            rover_yaml: Path to rover YAML configuration file.

        Returns:
            Parsed YAML as dictionary.

        Raises:
            FileNotFoundError: If rover_yaml does not exist.
            ValueError: If YAML syntax is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def load_mission_config(self, mission_yaml: str) -> Dict[str, Any]:
        """Load mission configuration from a YAML file.

        Expected top-level keys: mission, grid, rovers, deployment, cable,
        waypoints, success_criteria, contingency, analysis, debug.

        Parameters:
            mission_yaml: Path to mission YAML configuration file.

        Returns:
            Parsed YAML as dictionary.

        Raises:
            FileNotFoundError: If mission_yaml does not exist.
            ValueError: If YAML syntax is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def load_physics_config(self, physics_yaml: str) -> Dict[str, Any]:
        """Load physics engine configuration from a YAML file.

        Expected top-level keys matching GenesisConfig: gravity, timestep,
        contact_solver, broadphase, substeps, gpu_backend, determinism,
        performance, damping.

        Parameters:
            physics_yaml: Path to physics YAML configuration file.

        Returns:
            Parsed YAML as dictionary.

        Raises:
            FileNotFoundError: If physics_yaml does not exist.
            ValueError: If YAML syntax is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def load_sensors_config(self, sensors_yaml: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load sensor suite configuration from a YAML file.

        Parameters:
            sensors_yaml: Path to sensors YAML file, or None to skip sensor
                          registration (sensors will not be registered with
                          the physics engine).

        Returns:
            Parsed YAML as dictionary, or None if sensors_yaml is None.

        Raises:
            FileNotFoundError: If sensors_yaml is provided but does not exist.
            ValueError: If YAML syntax is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_configs(self) -> List[str]:
        """Validate all loaded configurations.

        Performs structural validation (required fields present, correct types)
        AND cross-reference validation:
        - scene.rover.type_ref must exist in rover.yaml profiles
        - scene.terrain.material.name must exist in MaterialLibrary
        - scene.rover.sensor_config_ref must exist in sensors.yaml (if loaded)
        - mission.rovers[*].rover_profile must exist in rover profiles
        - mission.rovers[*].cable_id must match the cable section cable_id

        Returns:
            List of error message strings. Empty list means all valid.
        """
        raise NotImplementedError

    @abstractmethod
    def compose_scene(self, engine: "PhysicsEngine") -> Scene:
        """Compose all loaded configurations into a complete typed Scene.

        Orchestration order:
        1. Assert engine in CONSTRUCTION phase
        2. Assert validate_configs() returns no errors
        3. Load MaterialLibrary
        4. TerrainComposer → SceneTerrainSpec
        5. MoonbaseComposer → SceneMoonbaseSpec
        6. RoverComposer (per rover) → list[SceneRoverSpec]
        7. AntennaComposer (per rover) → list[SceneAntennaSpec]
        8. CableComposer (per rover) → list[SceneCableSpec]
        9. SensorRegistrar → populate sensor_handles on rover specs
        10. engine.build_scene() — exactly once
        11. Apply initial rover poses post-build
        12. Return typed Scene

        Parameters:
            engine: Configured PhysicsEngine in CONSTRUCTION phase.

        Returns:
            Fully populated typed Scene.

        Raises:
            RuntimeError: If engine is not in CONSTRUCTION phase, or if required
                          configs have not been loaded.
            ValueError: If validate_configs() returns any errors (all listed).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

_SCHEMAS_DIR = os.path.join(os.path.dirname(__file__), "schemas")

_SCHEMA_FILES = {
    "scene":   "scene_schema.json",
    "rover":   "rover_schema.json",
    "mission": "mission_schema.json",
    "physics": "physics_schema.json",
    "sensors": "sensors_schema.json",
}


def _load_schema(name: str) -> Dict[str, Any]:
    """Load a JSON Schema file from the schemas/ directory."""
    path = os.path.join(_SCHEMAS_DIR, _SCHEMA_FILES[name])
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _validate_schema(config: Dict[str, Any], schema_name: str) -> List[str]:
    """Validate a config dict against a named JSON Schema.

    Returns a list of error messages (empty = valid). Uses jsonschema if
    available; falls back to a no-op with a warning if not installed.
    """
    try:
        import jsonschema  # type: ignore[import]
    except ImportError:
        return [
            f"jsonschema not installed — structural validation of {schema_name} skipped. "
            "Run: pip install jsonschema"
        ]

    schema = _load_schema(schema_name)
    validator = jsonschema.Draft202012Validator(schema)
    errors = []
    for err in sorted(validator.iter_errors(config), key=lambda e: list(e.absolute_path)):
        path = ".".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"[{schema_name}] {path}: {err.message}")
    return errors


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load and parse a YAML file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be parsed as YAML.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path!r}")
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path!r}: {exc}") from exc


class GenesisSceneComposer(SceneComposer):
    """Concrete scene composer for the Genesis physics backend.

    All sub-composers (terrain, moonbase, rover, antenna, cable, sensors) are
    injected at construction time to allow unit-testing without a real physics
    engine or Genesis import.

    Attributes:
        _scene_cfg: Parsed scene.yaml dict, or None if not yet loaded.
        _rover_cfg: Parsed rover.yaml dict, or None if not yet loaded.
        _mission_cfg: Parsed mission.yaml dict, or None if not yet loaded.
        _physics_cfg: Parsed physics.yaml dict, or None if not yet loaded.
        _sensors_cfg: Parsed sensors.yaml dict, or None (optional).
    """

    _REQUIRED_CONFIGS = ("scene", "rover", "mission", "physics")

    def __init__(
        self,
        terrain_composer=None,
        moonbase_composer=None,
        rover_composer=None,
        antenna_composer=None,
        cable_composer=None,
        sensor_registrar=None,
        material_library=None,
    ) -> None:
        """Initialise composer with optional injected sub-composers.

        Sub-composers default to None and are lazily instantiated by
        compose_scene() when not provided, using the standard concrete
        implementations.  Inject mocks for testing.
        """
        self._scene_cfg: Optional[Dict[str, Any]] = None
        self._rover_cfg: Optional[Dict[str, Any]] = None
        self._mission_cfg: Optional[Dict[str, Any]] = None
        self._physics_cfg: Optional[Dict[str, Any]] = None
        self._sensors_cfg: Optional[Dict[str, Any]] = None

        # Sub-composers (injected or lazily created)
        self._terrain_composer = terrain_composer
        self._moonbase_composer = moonbase_composer
        self._rover_composer = rover_composer
        self._antenna_composer = antenna_composer
        self._cable_composer = cable_composer
        self._sensor_registrar = sensor_registrar
        self._material_library = material_library

    # ------------------------------------------------------------------
    # YAML loaders
    # ------------------------------------------------------------------

    def load_scene_config(self, scene_yaml: str) -> Dict[str, Any]:
        """Load and cache scene.yaml.

        Parameters:
            scene_yaml: Path to scene YAML file.

        Returns:
            Parsed config dict.

        Raises:
            FileNotFoundError: File not found.
            ValueError: Invalid YAML syntax.
        """
        self._scene_cfg = _load_yaml(scene_yaml)
        return self._scene_cfg

    def load_rover_config(self, rover_yaml: str) -> Dict[str, Any]:
        """Load and cache rover.yaml.

        Parameters:
            rover_yaml: Path to rover YAML file.

        Returns:
            Parsed config dict.

        Raises:
            FileNotFoundError: File not found.
            ValueError: Invalid YAML syntax.
        """
        self._rover_cfg = _load_yaml(rover_yaml)
        return self._rover_cfg

    def load_mission_config(self, mission_yaml: str) -> Dict[str, Any]:
        """Load and cache mission.yaml.

        Parameters:
            mission_yaml: Path to mission YAML file.

        Returns:
            Parsed config dict.

        Raises:
            FileNotFoundError: File not found.
            ValueError: Invalid YAML syntax.
        """
        self._mission_cfg = _load_yaml(mission_yaml)
        return self._mission_cfg

    def load_physics_config(self, physics_yaml: str) -> Dict[str, Any]:
        """Load and cache physics.yaml.

        Parameters:
            physics_yaml: Path to physics YAML file.

        Returns:
            Parsed config dict.

        Raises:
            FileNotFoundError: File not found.
            ValueError: Invalid YAML syntax.
        """
        self._physics_cfg = _load_yaml(physics_yaml)
        return self._physics_cfg

    def load_sensors_config(self, sensors_yaml: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load and cache sensors.yaml, or store None if path is None.

        Parameters:
            sensors_yaml: Path to sensors YAML file, or None to skip.

        Returns:
            Parsed config dict, or None.

        Raises:
            FileNotFoundError: File not found (only if path is not None).
            ValueError: Invalid YAML syntax.
        """
        if sensors_yaml is None:
            self._sensors_cfg = None
            return None
        self._sensors_cfg = _load_yaml(sensors_yaml)
        return self._sensors_cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_configs(self) -> List[str]:
        """Validate all loaded configurations and cross-references.

        Returns:
            List of human-readable error strings. Empty = all valid.
        """
        errors: List[str] = []

        # ── structural: required configs loaded ──────────────────────
        missing = [
            name for name in self._REQUIRED_CONFIGS
            if getattr(self, f"_{name}_cfg") is None
        ]
        if missing:
            errors.append(
                f"Required configs not loaded: {', '.join(missing)}. "
                "Call load_*_config() before validate_configs()."
            )
            # Cannot do cross-reference checks without the configs
            return errors

        scene = self._scene_cfg
        rover = self._rover_cfg
        mission = self._mission_cfg
        sensors = self._sensors_cfg

        # ── JSON Schema structural validation ────────────────────────
        for schema_name, cfg in [
            ("scene", scene),
            ("rover", rover),
            ("mission", mission),
            ("physics", self._physics_cfg),
        ]:
            errors.extend(_validate_schema(cfg, schema_name))

        if sensors is not None:
            errors.extend(_validate_schema(sensors, "sensors"))

        # ── cross-reference: rover type_ref ─────────────────────────
        type_ref = scene.get("rover", {}).get("type_ref")
        profiles = rover.get("profiles", {})
        if type_ref and type_ref not in profiles:
            errors.append(
                f"scene.rover.type_ref {type_ref!r} not found in rover.yaml profiles. "
                f"Available: {sorted(profiles)}"
            )

        # ── cross-reference: terrain material name ───────────────────
        material_name = (
            scene.get("terrain", {}).get("material", {}).get("name")
        )
        if material_name and self._material_library is not None:
            missing_mats = self._material_library.validate_all_referenced_materials(
                [material_name]
            )
            for m in missing_mats:
                errors.append(
                    f"scene.terrain.material.name {m!r} not found in MaterialLibrary."
                )

        # ── cross-reference: sensor_config_ref ──────────────────────
        # scene.yaml uses sensor_config_ref to name the suite_id declared
        # inside sensors.yaml's sensor_suite.suite_id field.
        sensor_ref = scene.get("rover", {}).get("sensor_config_ref")
        if sensor_ref:
            if sensors is None:
                errors.append(
                    f"scene.rover.sensor_config_ref is {sensor_ref!r} but no "
                    "sensors config has been loaded. Call load_sensors_config()."
                )
            else:
                suite_id = sensors.get("sensor_suite", {}).get("suite_id")
                if suite_id != sensor_ref:
                    errors.append(
                        f"scene.rover.sensor_config_ref {sensor_ref!r} does not match "
                        f"sensors.yaml sensor_suite.suite_id {suite_id!r}."
                    )

        # ── cross-reference: mission rover profiles ──────────────────
        mission_rovers = mission.get("rovers", [])
        for rv in mission_rovers:
            profile = rv.get("rover_profile")
            if profile and profile not in profiles:
                errors.append(
                    f"mission.rovers[{rv.get('rover_id')!r}].rover_profile "
                    f"{profile!r} not in rover.yaml profiles. "
                    f"Available: {sorted(profiles)}"
                )

        # ── cross-reference: mission cable_id ───────────────────────
        # Each rover may have its own cable (e.g. cable_001, cable_002…).
        # When only one rover uses a cable, its cable_id must match the mission
        # cable section's cable_id. With multiple rovers we only verify that
        # every rover.cable_id is a non-empty string.
        cable_id_in_mission = mission.get("cable", {}).get("cable_id")
        rover_cable_ids = [rv.get("cable_id") for rv in mission_rovers if rv.get("cable_id")]
        unique_cable_ids = set(rover_cable_ids)
        if len(unique_cable_ids) == 1 and cable_id_in_mission:
            # Single cable shared by all rovers — must match the cable section.
            (sole_cable_id,) = unique_cable_ids
            if sole_cable_id != cable_id_in_mission:
                errors.append(
                    f"All rovers share cable_id {sole_cable_id!r} which does not "
                    f"match mission.cable.cable_id {cable_id_in_mission!r}."
                )

        return errors

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def _check_configs_loaded(self) -> None:
        """Raise RuntimeError if any required config is missing."""
        missing = [
            name for name in self._REQUIRED_CONFIGS
            if getattr(self, f"_{name}_cfg") is None
        ]
        if missing:
            raise RuntimeError(
                f"Cannot compose scene — required configs not loaded: "
                f"{', '.join(missing)}. Call load_*_config() first."
            )

    def compose_scene(self, engine: "PhysicsEngine") -> Scene:
        """Compose a fully-typed Scene from all loaded configs.

        Parameters:
            engine: PhysicsEngine in CONSTRUCTION phase.

        Returns:
            Typed Scene with all entities registered and initial poses applied.

        Raises:
            RuntimeError: Engine not in CONSTRUCTION phase, or missing configs.
            ValueError: Config validation errors (all listed in the message).
        """
        from moon_rover.core.physics.engine import ScenePhase

        # 1. Pre-flight checks
        self._check_configs_loaded()

        if engine.get_phase() != ScenePhase.CONSTRUCTION:
            raise RuntimeError(
                f"compose_scene() requires engine in CONSTRUCTION phase; "
                f"current phase: {engine.get_phase().value!r}"
            )

        errors = self.validate_configs()
        if errors:
            raise ValueError(
                "Config validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        # 2. Lazily import and instantiate sub-composers if not injected
        terrain_composer = self._terrain_composer or self._default_terrain_composer()
        moonbase_composer = self._moonbase_composer or self._default_moonbase_composer()
        rover_composer = self._rover_composer or self._default_rover_composer()
        antenna_composer = self._antenna_composer or self._default_antenna_composer()
        cable_composer = self._cable_composer or self._default_cable_composer()
        sensor_registrar = self._sensor_registrar or self._default_sensor_registrar()
        material_library = self._material_library or self._default_material_library()

        entity_handles: Dict[str, Any] = {}

        # 3. Terrain
        try:
            terrain_spec: SceneTerrainSpec = terrain_composer.compose(
                self._scene_cfg, engine, material_library
            )
        except Exception as exc:
            raise RuntimeError(f"TerrainComposer failed: {exc}") from exc
        entity_handles["terrain"] = terrain_spec.entity_handle

        # 4. Moonbase
        try:
            moonbase_spec: SceneMoonbaseSpec = moonbase_composer.compose(
                self._scene_cfg, engine
            )
        except Exception as exc:
            raise RuntimeError(f"MoonbaseComposer failed: {exc}") from exc
        entity_handles["moonbase"] = moonbase_spec.entity_handle

        # 5. Rovers
        try:
            rover_specs: List[SceneRoverSpec] = rover_composer.compose(
                self._scene_cfg, self._rover_cfg, self._mission_cfg, engine
            )
        except Exception as exc:
            raise RuntimeError(f"RoverComposer failed: {exc}") from exc
        for rs in rover_specs:
            entity_handles[rs.rover_id] = rs.entity_handle

        # 6. Antennas
        try:
            antenna_specs: List[SceneAntennaSpec] = antenna_composer.compose(
                rover_specs, engine
            )
        except Exception as exc:
            raise RuntimeError(f"AntennaComposer failed: {exc}") from exc
        for ant in antenna_specs:
            entity_handles[f"{ant.rover_id}_antenna"] = ant.entity_handle

        # 7. Cables
        try:
            cable_specs: List[SceneCableSpec] = cable_composer.compose(
                rover_specs, self._mission_cfg, engine
            )
        except Exception as exc:
            raise RuntimeError(f"CableComposer failed: {exc}") from exc
        for cs in cable_specs:
            for i, h in enumerate(cs.link_entity_handles):
                entity_handles[f"{cs.rover_id}_cable_link_{i:04d}"] = h

        # 8. Sensors (mutates rover_specs in-place)
        try:
            sensor_registrar.register(self._sensors_cfg, rover_specs, engine)
        except Exception as exc:
            raise RuntimeError(f"SensorRegistrar failed: {exc}") from exc

        # 9. Build scene — exactly once
        engine.build_scene()

        # 10. Apply initial poses/layout post-build
        self._apply_initial_layout(
            terrain_spec,
            moonbase_spec,
            rover_specs,
            antenna_specs,
            cable_specs,
            engine,
        )

        return Scene(
            terrain=terrain_spec,
            moonbase=moonbase_spec,
            rovers=rover_specs,
            antennas=antenna_specs,
            cables=cable_specs,
            physics_entity_handles=entity_handles,
        )

    @staticmethod
    def _apply_initial_layout(
        terrain_spec: SceneTerrainSpec,
        moonbase_spec: SceneMoonbaseSpec,
        rover_specs: List[SceneRoverSpec],
        antenna_specs: List[SceneAntennaSpec],
        cable_specs: List[SceneCableSpec],
        engine: "PhysicsEngine",
    ) -> None:
        """Set initial entity poses via the physics engine post-build.

        Stored cable links are staged well away from the active scene until
        the cable deployment subsystem starts using them. This keeps manual
        viewer checks readable instead of leaving a stack of link bodies at
        the world origin.
        """
        set_pose = getattr(engine, "set_body_pose", None)
        if set_pose is None:
            return

        identity_quat = [0.0, 0.0, 0.0, 1.0]

        try:
            set_pose("moonbase", list(moonbase_spec.world_position), identity_quat)
        except Exception:
            pass

        rover_positions: Dict[str, List[float]] = {}
        for rs in rover_specs:
            x, y, z, qx, qy, qz, qw = rs.initial_pose
            rover_positions[rs.rover_id] = [x, y, z]
            try:
                set_pose(rs.rover_id, [x, y, z], [qx, qy, qz, qw])
            except Exception:
                pass

        for ant in antenna_specs:
            rover_pos = rover_positions.get(ant.rover_id)
            if rover_pos is None:
                continue
            antenna_z = rover_pos[2] + max(0.35, ant.antenna_config.base_plate_m[2] * 2.0)
            try:
                set_pose(
                    f"{ant.rover_id}_antenna",
                    [rover_pos[0], rover_pos[1], antenna_z],
                    identity_quat,
                )
            except Exception:
                pass

        terrain_span = max(float(terrain_spec.size_m), 100.0)
        storage_origin_x = float(moonbase_spec.world_position[0]) + (terrain_span * 2.0)
        storage_origin_y = float(moonbase_spec.world_position[1]) + (terrain_span * 2.0)
        storage_origin_z = max(float(moonbase_spec.world_position[2]) + 2.0, 2.0)

        for cable_idx, cs in enumerate(cable_specs):
            storage_x = storage_origin_x + (cable_idx * 6.0)
            for link_idx, _handle in enumerate(cs.link_entity_handles):
                try:
                    set_pose(
                        f"{cs.rover_id}_cable_link_{link_idx:04d}",
                        [
                            storage_x,
                            storage_origin_y + (link_idx * 0.35),
                            storage_origin_z + (link_idx * 0.02),
                        ],
                        identity_quat,
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Default sub-composer factories (import-guarded)
    # ------------------------------------------------------------------

    def _default_terrain_composer(self):
        from moon_rover.core.scene.terrain_composer import TerrainComposer
        return TerrainComposer()

    def _default_moonbase_composer(self):
        from moon_rover.core.scene.moonbase_composer import MoonbaseComposer
        return MoonbaseComposer()

    def _default_rover_composer(self):
        from moon_rover.core.scene.rover_composer import RoverComposer
        return RoverComposer()

    def _default_antenna_composer(self):
        from moon_rover.core.scene.antenna_composer import AntennaComposer
        return AntennaComposer()

    def _default_cable_composer(self):
        from moon_rover.core.scene.cable_composer import CableComposer
        return CableComposer()

    def _default_sensor_registrar(self):
        from moon_rover.core.scene.sensor_registrar import SensorRegistrar
        return SensorRegistrar()

    def _default_material_library(self):
        from moon_rover.core.assets.material_library import MaterialLibrary
        lib = MaterialLibrary()
        # Load builtins; callers may supply a richer library via injection
        return lib
