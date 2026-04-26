"""MoonbaseComposer — maps scene.yaml moonbase config to a SceneMoonbaseSpec.

Registers the moonbase as a FIXED (kinematic, zero-mass) rigid body so it
does not move under lunar gravity, and populates beacon and docking configs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from moon_rover.core.scene.specs import BeaconConfig, DockingConfig, SceneMoonbaseSpec

if TYPE_CHECKING:
    from moon_rover.core.physics.engine import PhysicsEngine
    from moon_rover.core.assets.urdf_builder import URDFBuilder
    from moon_rover.moonbase.base import MoonbaseConfig


class MoonbaseComposer:
    """Converts scene.yaml moonbase section into a registered SceneMoonbaseSpec.

    The moonbase is registered as a fixed (kinematic) body — it must not move
    under gravity. This is enforced by passing a zero-mass morph to the engine.

    Parameters:
        urdf_builder: URDFBuilder implementation. If None, a default concrete
                      implementation is instantiated lazily.
    """

    def __init__(self, urdf_builder: Optional["URDFBuilder"] = None) -> None:
        self._urdf_builder = urdf_builder

    def compose(
        self,
        scene_cfg: Dict[str, Any],
        engine: "PhysicsEngine",
    ) -> SceneMoonbaseSpec:
        """Build and register the moonbase with the physics engine.

        Parameters:
            scene_cfg: Parsed scene.yaml dict.
            engine: PhysicsEngine in CONSTRUCTION phase.

        Returns:
            SceneMoonbaseSpec with beacon, docking configs, and entity handle.
        """
        mb_cfg = scene_cfg.get("moonbase", {})

        # ── position and orientation ─────────────────────────────────
        pos_raw = mb_cfg.get("position", [0.0, 0.0, 0.0])
        world_position: Tuple[float, float, float] = (
            float(pos_raw[0]),
            float(pos_raw[1]),
            float(pos_raw[2]),
        )

        # ── beacon ───────────────────────────────────────────────────
        beacon_raw = mb_cfg.get("beacon", {})
        beacon = BeaconConfig(
            position_xyz=world_position,
            frequency_hz=float(beacon_raw.get("frequency_hz", 1.0)),
            signal_strength=float(beacon_raw.get("signal_strength", 1.0)),
            communication_range_m=float(
                beacon_raw.get("communication_range_m", 1000.0)
            ),
        )

        # ── docking ──────────────────────────────────────────────────
        docking_raw = mb_cfg.get("docking", {})
        raw_ports = docking_raw.get("port_positions", [[2.0, 0.0, 0.0]])
        port_positions: List[Tuple[float, float, float]] = [
            (float(p[0]), float(p[1]), float(p[2])) for p in raw_ports
        ]
        docking = DockingConfig(
            num_ports=int(docking_raw.get("num_ports", 2)),
            port_positions=port_positions,
            charge_rate_w=float(docking_raw.get("charging_power_w", 500.0)),
        )

        # ── build MoonbaseConfig for downstream subsystems ───────────
        moonbase_config = self._build_moonbase_config(mb_cfg, docking)

        # ── build URDF and register as fixed body ────────────────────
        builder = self._urdf_builder or self._default_urdf_builder()
        urdf_xml = builder.build_moonbase()

        # Fixed body: pass fixed=True or zero mass morph depending on engine API.
        # The engine's add_entity signature accepts entity_type="fixed" to signal
        # that the body should be kinematic (immovable).
        entity_handle = engine.add_entity(
            name="moonbase",
            morph=urdf_xml,
            material=None,
            entity_type="fixed",
        )

        return SceneMoonbaseSpec(
            config=moonbase_config,
            world_position=world_position,
            beacon=beacon,
            docking=docking,
            entity_handle=entity_handle,
        )

    @staticmethod
    def _build_moonbase_config(
        mb_cfg: Dict[str, Any],
        docking: DockingConfig,
    ) -> "MoonbaseConfig":
        """Construct a MoonbaseConfig from scene.yaml moonbase section."""
        from moon_rover.moonbase.base import MoonbaseConfig

        return MoonbaseConfig(
            habitat_dims_m=(10.0, 8.0, 4.0),  # standard dimensions
            solar_array_config=None,           # populated by PowerSystem later
            power_bus_voltage=48.0,
            comm_tower_height_m=5.0,
            num_docking_bays=docking.num_ports,
            charge_rate_w=docking.charge_rate_w,
            num_cable_reels=8,
            num_antennas=20,
            landing_pad_radius_m=float(
                mb_cfg.get("docking", {}).get("landing_pad_radius_m", 15.0)
            ),
        )

    @staticmethod
    def _default_urdf_builder() -> "URDFBuilder":
        """Lazily import and return the default concrete URDFBuilder."""
        try:
            from moon_rover.core.assets.genesis_urdf_builder import GenesisURDFBuilder
            return GenesisURDFBuilder()
        except ImportError:
            raise ImportError(
                "No concrete URDFBuilder available. "
                "Either implement GenesisURDFBuilder or inject a URDFBuilder "
                "into MoonbaseComposer(urdf_builder=...)."
            )
