"""RoverComposer — maps rover.yaml profiles and mission.yaml rovers to SceneRoverSpecs.

For each rover listed in mission.yaml, resolves its profile from rover.yaml,
generates a validated URDF, and registers the rover with the physics engine.
Handles 1–N rovers, all three drive configurations, and Euler→quaternion conversion
for initial poses.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from moon_rover.core.scene.specs import DriveType, SceneRoverSpec
from moon_rover.cable.chain import CableConfig

if TYPE_CHECKING:
    from moon_rover.core.physics.engine import PhysicsEngine
    from moon_rover.core.assets.urdf_builder import URDFBuilder, URDFValidationStage
    from moon_rover.core.assets.material_library import MaterialLibrary


# Profile key → DriveType enum
_DRIVE_TYPE_MAP: Dict[str, DriveType] = {
    "two_wheel_diff": DriveType.TWO_WHEEL_DIFF,
    "three_wheel_tricycle": DriveType.THREE_WHEEL_TRICYCLE,
    "four_wheel_skid": DriveType.FOUR_WHEEL_SKID,
}


def _euler_deg_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert Euler angles (degrees, ZYX convention) to quaternion (qx, qy, qz, qw).

    Parameters:
        roll: Rotation around X axis in degrees.
        pitch: Rotation around Y axis in degrees.
        yaw: Rotation around Z axis in degrees.

    Returns:
        Tuple (qx, qy, qz, qw).
    """
    r = math.radians(roll)
    p = math.radians(pitch)
    y = math.radians(yaw)

    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


class RoverComposer:
    """Converts mission.yaml rover list + rover.yaml profiles into SceneRoverSpecs.

    Parameters:
        urdf_builder: URDFBuilder implementation (injected for testing).
        material_lib: MaterialLibrary (injected for testing).
    """

    def __init__(
        self,
        urdf_builder: Optional["URDFBuilder"] = None,
        material_lib: Optional["MaterialLibrary"] = None,
    ) -> None:
        self._urdf_builder = urdf_builder
        self._material_lib = material_lib

    def compose(
        self,
        scene_cfg: Dict[str, Any],
        rover_cfg: Dict[str, Any],
        mission_cfg: Dict[str, Any],
        engine: "PhysicsEngine",
    ) -> List[SceneRoverSpec]:
        """Compose all rovers listed in mission.yaml.

        Parameters:
            scene_cfg: Parsed scene.yaml — used for initial pose of first rover.
            rover_cfg: Parsed rover.yaml — provides drive profiles.
            mission_cfg: Parsed mission.yaml — lists rovers to deploy.
            engine: PhysicsEngine in CONSTRUCTION phase.

        Returns:
            List of SceneRoverSpec, one per mission rover.

        Raises:
            ValueError: If a rover_profile is missing from rover.yaml profiles,
                        or if URDF validation fails for any rover.
        """
        from moon_rover.core.assets.urdf_builder import URDFValidationStage

        builder = self._urdf_builder or self._default_urdf_builder()
        profiles = rover_cfg.get("profiles", {})
        mission_rovers: List[Dict[str, Any]] = mission_cfg.get("rovers", [])

        # Initial pose from scene.yaml (shared starting point for rover_001;
        # additional rovers offset by grid_position in mission config).
        scene_rover = scene_cfg.get("rover", {})
        base_pos = scene_rover.get("position", [0.0, 0.0, 0.5])
        base_ori = scene_rover.get("orientation", [0.0, 0.0, 0.0])

        results: List[SceneRoverSpec] = []

        for idx, rv in enumerate(mission_rovers):
            rover_id: str = rv.get("rover_id", f"rover_{idx:03d}")
            profile_key: str = rv.get("rover_profile", "four_wheel_skid")

            if profile_key not in profiles:
                raise ValueError(
                    f"rover {rover_id!r}: rover_profile {profile_key!r} not found "
                    f"in rover.yaml profiles. Available: {sorted(profiles)}"
                )

            profile = profiles[profile_key]
            drive_type = _DRIVE_TYPE_MAP.get(profile_key, DriveType.FOUR_WHEEL_SKID)

            # Build URDF config dict from profile
            urdf_config = self._profile_to_urdf_config(rover_id, profile, rover_cfg)

            # Generate and validate URDF
            urdf_str = builder.build_rover(urdf_config)
            errors = builder.validate(urdf_str, URDFValidationStage.PHYSICS_SANITY)
            if errors:
                raise ValueError(
                    f"URDF validation failed for rover {rover_id!r} "
                    f"(profile {profile_key!r}):\n" +
                    "\n".join(f"  • {e}" for e in errors)
                )

            # Material for rover chassis
            chassis_mat_name = (
                rover_cfg.get("structure", {}).get("chassis_material", "aluminum")
            )
            material = None
            if self._material_lib is not None:
                try:
                    material = self._material_lib.get_material(chassis_mat_name)
                except KeyError:
                    pass  # Material not required for entity registration

            # Register with physics engine
            entity_handle = engine.add_entity(
                name=rover_id,
                morph=urdf_str,
                material=material,
                entity_type="articulated",
            )

            # Compute initial pose — offset by grid_position for multi-rover
            grid_pos = rv.get("grid_position", [0, 0])
            x = float(base_pos[0]) + float(grid_pos[0]) * 20.0
            y = float(base_pos[1]) + float(grid_pos[1]) * 20.0
            z = float(base_pos[2])
            qx, qy, qz, qw = _euler_deg_to_quaternion(
                float(base_ori[0]),
                float(base_ori[1]),
                float(base_ori[2]),
            )

            # Build CableConfig placeholder (CableComposer will populate fully)
            cable_config = self._default_cable_config(mission_cfg)

            results.append(SceneRoverSpec(
                rover_id=rover_id,
                urdf_str=urdf_str,
                drive_type=drive_type,
                initial_pose=(x, y, z, qx, qy, qz, qw),
                cable_config=cable_config,
                sensor_handles={},
                entity_handle=entity_handle,
                num_wheels=int(profile.get("num_wheels", 4)),
                mass_kg=float(profile.get("mass_kg", 60.0)),
                wheel_radius_m=float(profile.get("wheel_radius_m", 0.35)),
            ))

        return results

    @staticmethod
    def _profile_to_urdf_config(
        rover_id: str,
        profile: Dict[str, Any],
        rover_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert a rover.yaml profile dict into a URDFBuilder.build_rover() config."""
        arm = rover_cfg.get("arm", {})
        power = rover_cfg.get("power", {})
        structure = rover_cfg.get("structure", {})
        gripper = arm.get("gripper", {}) if isinstance(arm, dict) else {}

        arm_joints = arm.get("joints", {}) if isinstance(arm, dict) else {}
        num_dof = int(arm.get("num_dof", 4))
        joint_axes: list[list[float]] = []
        joint_limits: list[list[float]] = []
        for i in range(1, num_dof + 1):
            joint_spec = arm_joints.get(f"joint_{i}", {}) if isinstance(arm_joints, dict) else {}
            joint_axes.append(list(joint_spec.get("axis", [0, 0, 1] if i == 1 else [0, 1, 0])))
            joint_limits.append([
                float(joint_spec.get("lower_limit_rad", -3.14159)),
                float(joint_spec.get("upper_limit_rad", 3.14159)),
            ])

        return {
            "rover_id": rover_id,
            "mass_kg": float(profile.get("mass_kg", 60.0)),
            "wheel_radius": float(profile.get("wheel_radius_m", 0.35)),
            "wheel_count": int(profile.get("num_wheels", 4)),
            "track_width": float(profile.get("track_width_m", 1.5)),
            "wheelbase": float(profile.get("wheelbase_m", 1.8)),
            "wheel_positions": profile.get("wheel_positions", []),
            "inertia": profile.get("inertia", [3.0, 6.0, 3.5]),
            "com_offset": profile.get("com_offset", [0.0, 0.0, -0.25]),
            "material_name": structure.get("chassis_material", "aluminum"),
            "dimensions": profile.get("dimensions") or structure.get(
                "dimensions", [2.0, 1.5, 0.8]
            ),
            "arm_dof": num_dof,
            "arm_reach_m": float(arm.get("reach_m", 2.0)),
            "arm_link_lengths": arm.get("link_lengths"),
            "arm_joint_axes": joint_axes,
            "arm_joint_limits": joint_limits,
            "arm_base_position": profile.get("arm_base_position") or arm.get(
                "base_position", [0.5, 0.0, 0.3]
            ),
            "gripper_type": gripper.get("type", "parallel_jaw"),
            "gripper_stroke_m": float(gripper.get("stroke_m", 0.1)),
            "gripper_finger_length_m": float(gripper.get("finger_length_m", 0.1)),
            "gripper_finger_width_m": float(gripper.get("finger_width_m", 0.03)),
            "battery_capacity_wh": (
                power.get("battery", {}).get("capacity_wh", 2000.0)
            ),
            "drive_type": profile.get("skid_steer") and "skid_steer" or "differential",
        }

    @staticmethod
    def _default_cable_config(mission_cfg: Dict[str, Any]) -> CableConfig:
        """Build a CableConfig from mission.yaml cable section defaults."""
        cable = mission_cfg.get("cable", {})
        total_length_m = float(cable.get("length_m", 60.0))
        # link_length_m can be tuned via mission.yaml cable.link_length_m;
        # 0.5 m is production-grade resolution.
        link_length_m = float(cable.get("link_length_m", 0.5))
        linear_density = float(cable.get("linear_density_kg_m", 0.15))
        power_section = cable.get("power", {})

        return CableConfig(
            link_length_m=link_length_m,
            link_diameter_m=float(cable.get("diameter_mm", 10.0)) / 1000.0,
            link_mass_kg=linear_density * link_length_m,
            total_length_m=total_length_m,
            joint_damping=0.1,
            joint_stiffness=100.0,
            terrain_friction=0.4,
            max_tension_n=float(cable.get("max_tension_n", 500.0)),
            bend_radius_min_m=0.05,
            voltage_dc=float(power_section.get("voltage_dc", 48.0)) if power_section.get("enabled") else 0.0,
            resistance_per_m=float(power_section.get("resistance_per_m_ohms", 0.005)),
        )

    @staticmethod
    def _default_urdf_builder() -> "URDFBuilder":
        try:
            from moon_rover.core.assets.genesis_urdf_builder import GenesisURDFBuilder
            return GenesisURDFBuilder()
        except ImportError:
            raise ImportError(
                "No concrete URDFBuilder available. "
                "Inject a URDFBuilder into RoverComposer(urdf_builder=...)."
            )
