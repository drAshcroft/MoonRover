"""TerrainComposer — maps scene.yaml terrain config to a SceneTerrainSpec.

Converts the terrain section of scene.yaml into a TerrainConfig, drives the
TerrainGenerator, registers the resulting height-field with the physics engine,
and returns a fully populated SceneTerrainSpec.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, TYPE_CHECKING

from moon_rover.core.scene.specs import SceneTerrainSpec
from moon_rover.environment.terrain.generator import TerrainConfig, TerrainGenerator

if TYPE_CHECKING:
    from moon_rover.core.physics.engine import PhysicsEngine
    from moon_rover.core.assets.material_library import MaterialLibrary


class TerrainComposer:
    """Converts scene.yaml terrain section into a registered SceneTerrainSpec.

    Parameters:
        generator: TerrainGenerator implementation. If None, a default
                   concrete implementation is instantiated lazily.
    """

    def __init__(self, generator: Optional[TerrainGenerator] = None) -> None:
        self._generator = generator

    def compose(
        self,
        scene_cfg: Dict[str, Any],
        engine: "PhysicsEngine",
        material_lib: "MaterialLibrary",
    ) -> SceneTerrainSpec:
        """Generate terrain and register it with the physics engine.

        Parameters:
            scene_cfg: Parsed scene.yaml dict.
            engine: PhysicsEngine in CONSTRUCTION phase.
            material_lib: Loaded MaterialLibrary for material resolution.

        Returns:
            SceneTerrainSpec with all terrain data and the engine entity handle.

        Raises:
            KeyError: If the terrain material name is not in MaterialLibrary.
            ValueError: If terrain config values are invalid.
        """
        tc = scene_cfg.get("terrain", {})

        # ── build TerrainConfig from scene.yaml terrain section ──────
        size_m: float = float(tc.get("size_m", 100.0))
        resolution: int = int(tc.get("resolution", 256))
        height_scale_m: float = float(tc.get("height_scale_m", 2.0))
        seed: int = int(tc.get("seed", 42))

        features = tc.get("features", {})
        craters_enabled: bool = bool(features.get("craters", True))
        num_craters: int = int(features.get("num_craters", 15))
        num_rock_clusters: int = int(features.get("num_rock_clusters", 10))
        rille_enabled: bool = bool(features.get("ridges", True))

        # Rock density: clusters per m² (approximate from cluster count)
        rock_density: float = num_rock_clusters / max(size_m * size_m, 1.0)

        crater_params = {
            "count": num_craters if craters_enabled else 0,
            "min_radius_m": 1.0,
            "max_radius_m": max(5.0, size_m * 0.05),
            "depth_ratio": 0.3,
        }

        config = TerrainConfig(
            seed=seed,
            size_m=size_m,
            fBm_octaves=8,
            fBm_amplitude=height_scale_m,
            crater_params=crater_params,
            rock_density=rock_density,
            rille_enabled=rille_enabled,
            moonbase_position=(0.0, 0.0, 0.0),
            resolution=resolution,
        )

        # ── generate terrain ─────────────────────────────────────────
        generator = self._generator or self._default_generator()
        output = generator.generate(config)

        # ── resolve material ─────────────────────────────────────────
        material_name: str = (
            tc.get("material", {}).get("name", "lunar_regolith")
        )
        material = material_lib.get_material(material_name)  # raises KeyError on miss

        # ── register with physics engine ─────────────────────────────
        entity_handle = engine.add_terrain_entity(
            name="terrain",
            height_field=output.height_field,
            size=[size_m, size_m],
        )

        return SceneTerrainSpec(
            config=config,
            height_field=output.height_field,
            slope_map=output.slope_map,
            normal_map=output.normal_map,
            rock_positions=output.rock_positions,
            crater_list=output.crater_list,
            nav_mesh=output.nav_mesh,
            material=material,
            size_m=size_m,
            entity_handle=entity_handle,
        )

    @staticmethod
    def _default_generator() -> TerrainGenerator:
        """Lazily import and return the default concrete TerrainGenerator."""
        try:
            from moon_rover.environment.terrain.lunar_generator import (
                LunarTerrainGenerator,
            )
            return LunarTerrainGenerator()
        except ImportError:
            raise ImportError(
                "No concrete TerrainGenerator available. "
                "Either install the lunar_generator module or inject a "
                "TerrainGenerator into TerrainComposer(generator=...)."
            )
