"""CableComposer — maps mission.yaml cable config to registered SceneCableSpecs.

Converts the cable section of mission.yaml into CableConfig instances, pre-allocates
ALL cable link entities with the physics engine during CONSTRUCTION phase, and returns
SceneCableSpecs ready for the CableSystem to operate on.

Critical constraint: every link entity must be registered before engine.build_scene()
is called. No dynamic allocation is permitted during simulation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from moon_rover.core.scene.specs import SceneCableSpec
from moon_rover.cable.chain import CableConfig

if TYPE_CHECKING:
    from moon_rover.core.scene.specs import SceneRoverSpec
    from moon_rover.core.physics.engine import PhysicsEngine


# Minimal URDF for a single rigid cable link — a thin cylinder.
# Length and radius are overridden per CableConfig at compose time.
_LINK_URDF_TEMPLATE = """\
<?xml version="1.0"?>
<robot name="{name}">
  <link name="link">
    <inertial>
      <mass value="{mass}"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <inertia ixx="{ixx}" ixy="0" ixz="0" iyy="{iyy}" iyz="0" izz="{izz}"/>
    </inertial>
    <visual>
      <geometry>
        <cylinder length="{length}" radius="{radius}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="{length}" radius="{radius}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""


def _link_urdf(name: str, config: CableConfig) -> str:
    radius = config.link_diameter_m / 2.0
    length = config.link_length_m
    mass = config.link_mass_kg
    ixx_iyy = (mass / 12.0) * ((3.0 * radius * radius) + (length * length))
    izz = 0.5 * mass * radius * radius
    return _LINK_URDF_TEMPLATE.format(
        name=name,
        mass=mass,
        ixx=ixx_iyy,
        iyy=ixx_iyy,
        izz=izz,
        length=length,
        radius=radius,
    )


class CableComposer:
    """Pre-allocates all cable link entities for every rover in the mission.

    One cable is created per rover. The number of links is determined by:
        num_links = ceil(total_length_m / link_length_m)

    All links start in STORED (inactive) state. The CableSystem transitions them
    to ACTIVE as the rover pays out cable during simulation.

    Parameters:
        None — CableComposer has no injectable dependencies (link URDF is
               generated internally from CableConfig).
    """

    def compose(
        self,
        rover_specs: List["SceneRoverSpec"],
        mission_cfg: Dict[str, Any],
        engine: "PhysicsEngine",
    ) -> List[SceneCableSpec]:
        """Pre-allocate cable link entities for all rovers.

        Parameters:
            rover_specs: List of SceneRoverSpec; one cable is composed per rover.
                         The CableConfig placeholder already stored in each spec
                         is used as the source of truth (populated by RoverComposer).
            mission_cfg: Parsed mission.yaml — provides cable_id if present.
            engine: PhysicsEngine in CONSTRUCTION phase.

        Returns:
            List of SceneCableSpec, parallel to rover_specs.

        Raises:
            ValueError: If rover_specs is empty or CableConfig is invalid.
        """
        if not rover_specs:
            raise ValueError("rover_specs must contain at least one rover.")

        mission_cable = mission_cfg.get("cable", {})
        results: List[SceneCableSpec] = []

        for rover_spec in rover_specs:
            rover_id = rover_spec.rover_id
            config = rover_spec.cable_config

            if config.link_length_m <= 0:
                raise ValueError(
                    f"Cable for rover {rover_id!r}: link_length_m must be > 0, "
                    f"got {config.link_length_m}"
                )
            if config.total_length_m <= 0:
                raise ValueError(
                    f"Cable for rover {rover_id!r}: total_length_m must be > 0, "
                    f"got {config.total_length_m}"
                )

            num_links = math.ceil(config.total_length_m / config.link_length_m)
            cable_id = mission_cable.get("cable_id", f"cable_{rover_id}")

            link_handles: List[Any] = []
            for link_idx in range(num_links):
                link_name = f"{rover_id}_cable_link_{link_idx:04d}"
                urdf = _link_urdf(link_name, config)
                handle = engine.add_entity(
                    name=link_name,
                    morph=urdf,
                    material=None,
                    entity_type="rigid",
                )
                link_handles.append(handle)

            results.append(SceneCableSpec(
                rover_id=rover_id,
                cable_id=cable_id,
                config=config,
                link_entity_handles=link_handles,
            ))

        return results
