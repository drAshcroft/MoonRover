"""AntennaComposer — maps rover specs to registered SceneAntennaSpecs.

For each rover in the mission, creates one antenna unit collocated with
the rover chassis in STORED initial state. All antennas are registered
with the physics engine as rigid bodies before build_scene() is called.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from moon_rover.core.scene.specs import SceneAntennaSpec
from moon_rover.antenna.system import AntennaConfig, AntennaState

if TYPE_CHECKING:
    from moon_rover.core.scene.specs import SceneRoverSpec
    from moon_rover.core.physics.engine import PhysicsEngine
    from moon_rover.core.assets.urdf_builder import URDFBuilder


# Standard lunar surface antenna dimensions per system specification
_DEFAULT_ANTENNA_CONFIG = AntennaConfig(
    base_plate_m=(0.4, 0.4, 0.05),
    base_mass_kg=2.5,
    mast_height_m=1.2,
    mast_radius_m=0.02,
    mast_mass_kg=1.0,
    dish_diameter_m=0.6,
    dish_mass_kg=0.8,
    connector_mass_kg=0.2,
    total_mass_kg=4.5,  # 2.5 + 1.0 + 0.8 + 0.2
)


class AntennaComposer:
    """Creates and registers antenna units for each rover in the mission.

    Each rover receives exactly one deployable antenna unit. All antennas
    start in STORED state (folded on rover chassis) and are registered as
    rigid bodies with the physics engine during CONSTRUCTION phase.

    Parameters:
        urdf_builder: URDFBuilder implementation. If None, a default concrete
                      implementation is instantiated lazily.
        antenna_config: AntennaConfig to apply to all antennas. If None, the
                        standard lunar antenna defaults are used.
    """

    def __init__(
        self,
        urdf_builder: Optional["URDFBuilder"] = None,
        antenna_config: Optional[AntennaConfig] = None,
    ) -> None:
        self._urdf_builder = urdf_builder
        self._antenna_config = antenna_config or _DEFAULT_ANTENNA_CONFIG

    def compose(
        self,
        rover_specs: List["SceneRoverSpec"],
        engine: "PhysicsEngine",
    ) -> List[SceneAntennaSpec]:
        """Build and register one antenna per rover with the physics engine.

        Parameters:
            rover_specs: List of SceneRoverSpec from RoverComposer; one antenna
                         is created per rover, sharing the same rover_id.
            engine: PhysicsEngine in CONSTRUCTION phase.

        Returns:
            List of SceneAntennaSpec, parallel to rover_specs (same order/length).

        Raises:
            ValueError: If rover_specs is empty.
        """
        if not rover_specs:
            raise ValueError("rover_specs must contain at least one rover.")

        builder = self._urdf_builder or self._default_urdf_builder()
        urdf_xml = builder.build_antenna()

        results: List[SceneAntennaSpec] = []

        for rover_spec in rover_specs:
            rover_id = rover_spec.rover_id
            antenna_entity_name = f"{rover_id}_antenna"

            # Register antenna as a rigid body collocated with rover chassis.
            # Initial state is STORED — physically coincident with the rover.
            # The arm/deployment system will move it to terrain during simulation.
            entity_handle = engine.add_entity(
                name=antenna_entity_name,
                morph=urdf_xml,
                material=None,
                entity_type="rigid",
            )

            results.append(SceneAntennaSpec(
                rover_id=rover_id,
                antenna_config=self._antenna_config,
                initial_state=AntennaState.STORED,
                cable_attachment_entity_name=rover_id,
                entity_handle=entity_handle,
            ))

        return results

    @staticmethod
    def _default_urdf_builder() -> "URDFBuilder":
        """Lazily import and return the default concrete URDFBuilder."""
        try:
            from moon_rover.core.assets.genesis_urdf_builder import GenesisURDFBuilder
            return GenesisURDFBuilder()
        except ImportError:
            raise ImportError(
                "No concrete URDFBuilder available. "
                "Inject a URDFBuilder into AntennaComposer(urdf_builder=...)."
            )
