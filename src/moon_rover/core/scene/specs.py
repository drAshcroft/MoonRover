"""Typed domain model for composed simulation scenes.

These dataclasses form the typed contract between the SceneComposer pipeline
and downstream subsystems (DriveSystem, TerrainGenerator, PowerSystem, etc.).
No field may be typed as Any — all downstream code must be able to access
spec fields without casting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from moon_rover.environment.terrain.generator import TerrainConfig, TerrainOutput
    from moon_rover.core.assets.urdf_builder import MaterialProperties
    from moon_rover.antenna.system import AntennaConfig, AntennaState
    from moon_rover.moonbase.base import MoonbaseConfig
    from moon_rover.cable.chain import CableConfig


class DriveType(Enum):
    """Rover drive configuration type.

    Maps directly to the profile keys in rover.yaml.
    """
    TWO_WHEEL_DIFF = "two_wheel_diff"
    THREE_WHEEL_TRICYCLE = "three_wheel_tricycle"
    FOUR_WHEEL_SKID = "four_wheel_skid"


@dataclass
class BeaconConfig:
    """Moonbase localization beacon parameters.

    Used by the GPS beacon network and rover pseudo-GNSS subsystems.

    Attributes:
        position_xyz: Beacon world position [x, y, z] in meters.
        frequency_hz: Beacon signal update rate in Hz.
        signal_strength: Relative signal strength (1.0 = nominal).
        communication_range_m: Maximum effective range in meters.
    """
    position_xyz: Tuple[float, float, float]
    frequency_hz: float
    signal_strength: float
    communication_range_m: float


@dataclass
class DockingConfig:
    """Moonbase docking bay parameters for charging and servicing.

    Attributes:
        num_ports: Number of simultaneous docking positions.
        port_positions: List of port positions relative to moonbase center [x, y, z].
        charge_rate_w: Power delivery per port in Watts.
    """
    num_ports: int
    port_positions: List[Tuple[float, float, float]]
    charge_rate_w: float


@dataclass
class SceneTerrainSpec:
    """Typed terrain specification produced by TerrainComposer.

    Carries the full TerrainOutput alongside resolved material properties
    and the physics entity handle for the terrain body.

    Attributes:
        config: TerrainConfig used to generate this terrain.
        height_field: 2D height map, shape (resolution, resolution), dtype float32.
        slope_map: Slope magnitude per point in degrees, same shape as height_field.
        normal_map: Per-pixel surface normals, shape (resolution, resolution, 3).
        rock_positions: List of (x, y, z, radius) tuples for rock obstacles.
        crater_list: List of crater dicts with position, radius, depth keys.
        nav_mesh: Binary traversability grid, shape (resolution, resolution), dtype uint8.
                  0 = impassable, 1 = traversable. Used by PathPlanner.
        material: Resolved material properties for terrain surface.
        size_m: Terrain side length in meters (terrain is square).
        entity_handle: Physics engine entity handle returned by add_terrain_entity().
    """
    config: "TerrainConfig"
    height_field: NDArray[np.float32]
    slope_map: NDArray[np.float32]
    normal_map: NDArray[np.float32]
    rock_positions: List[Tuple[float, float, float, float]]
    crater_list: List[Dict[str, Any]]
    nav_mesh: NDArray[np.uint8]
    material: "MaterialProperties"
    size_m: float
    entity_handle: Any  # Genesis entity handle; opaque to SceneComposer


@dataclass
class SceneRoverSpec:
    """Typed rover specification produced by RoverComposer.

    Carries everything the DriveSystem, SensorSubsystem, and CableSystem
    need to initialise and control the rover.

    Attributes:
        rover_id: Unique string identifier (e.g. "rover_001").
        urdf_str: Complete URDF XML string for this rover configuration.
        drive_type: Drive configuration enum — TWO_WHEEL_DIFF, THREE_WHEEL_TRICYCLE,
                    or FOUR_WHEEL_SKID. Consumers must not use the string profile key.
        initial_pose: Initial world pose [x, y, z, qx, qy, qz, qw]. Applied post-build.
        cable_config: CableConfig for the cable attached to this rover.
        sensor_handles: Dict mapping sensor name to physics engine handle or reference.
                        Keys: 'lidar', 'force_torque', 'sun_sensor', 'imu'.
                        Values are engine-specific handles populated by SensorRegistrar.
        entity_handle: Physics engine entity handle returned by add_entity().
        num_wheels: Number of wheels on this rover (2, 3, or 4).
        mass_kg: Total rover mass in kg.
        wheel_radius_m: Wheel radius in meters.
    """
    rover_id: str
    urdf_str: str
    drive_type: DriveType
    initial_pose: Tuple[float, float, float, float, float, float, float]
    cable_config: "CableConfig"
    sensor_handles: Dict[str, Any]
    entity_handle: Any
    num_wheels: int
    mass_kg: float
    wheel_radius_m: float


@dataclass
class SceneAntennaSpec:
    """Typed antenna specification produced by AntennaComposer.

    Attributes:
        rover_id: ID of the rover this antenna is assigned to.
        antenna_config: Physical configuration of the antenna unit.
        initial_state: Initial deployment state — always STORED at compose time.
        cable_attachment_entity_name: Entity name to link cable system to antenna.
        entity_handle: Physics engine entity handle returned by add_entity().
    """
    rover_id: str
    antenna_config: "AntennaConfig"
    initial_state: "AntennaState"
    cable_attachment_entity_name: str
    entity_handle: Any


@dataclass
class SceneMoonbaseSpec:
    """Typed moonbase specification produced by MoonbaseComposer.

    Attributes:
        config: MoonbaseConfig with facility dimensions and capabilities.
        world_position: Absolute world position [x, y, z] in meters.
        beacon: Primary localization beacon configuration.
        docking: Docking bay layout and charging parameters.
        entity_handle: Physics engine entity handle (fixed/kinematic body).
    """
    config: "MoonbaseConfig"
    world_position: Tuple[float, float, float]
    beacon: BeaconConfig
    docking: DockingConfig
    entity_handle: Any


@dataclass
class SceneCableSpec:
    """Typed cable specification produced by CableComposer.

    Carries the CableConfig and all pre-allocated link entity handles.
    All links are allocated at CONSTRUCTION time — never during simulation.

    Attributes:
        rover_id: ID of the rover this cable is tethered to.
        cable_id: Cable identifier from mission.yaml.
        config: CableConfig with all mechanical and electrical parameters.
        link_entity_handles: List of physics entity handles, one per cable link.
                             Length = ceil(config.total_length_m / config.link_length_m).
                             All links start in STORED (inactive) state.
    """
    rover_id: str
    cable_id: str
    config: "CableConfig"
    link_entity_handles: List[Any] = field(default_factory=list)


@dataclass
class Scene:
    """Complete, fully-typed simulation scene.

    Produced by SceneComposer.compose_scene() after engine.build_scene() completes.
    All fields are concrete typed specs — no Any-typed containers at the top level.

    Attributes:
        terrain: Terrain spec including height_field, nav_mesh, material, and handles.
        moonbase: Moonbase spec with facility config, beacon, and docking info.
        rovers: Ordered list of rover specs, one per rover in mission.yaml.
        antennas: Ordered list of antenna specs, parallel to rovers list.
        cables: Ordered list of cable specs, parallel to rovers list.
        physics_entity_handles: Flat map of entity_name -> physics handle for
                                all entities registered with the engine.
    """
    terrain: SceneTerrainSpec
    moonbase: SceneMoonbaseSpec
    rovers: List[SceneRoverSpec]
    antennas: List[SceneAntennaSpec]
    cables: List[SceneCableSpec]
    physics_entity_handles: Dict[str, Any] = field(default_factory=dict)
