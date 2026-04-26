"""Material Definitions Library for Lunar Assets.

This module provides a centralized library of material properties used across
the simulation for rovers, terrain, and structures.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import yaml

from moon_rover.core.assets.urdf_builder import MaterialProperties


# Built-in lunar and structural material definitions.
_BUILTIN_MATERIALS: Dict[str, MaterialProperties] = {
    "lunar_regolith": MaterialProperties(
        friction=0.6,
        density=1500.0,      # kg/m³ — compacted lunar regolith
        restitution=0.02,
        name="lunar_regolith",
    ),
    "rock": MaterialProperties(
        friction=0.8,
        density=2700.0,      # kg/m³ — basalt
        restitution=0.1,
        name="rock",
    ),
    "aluminum": MaterialProperties(
        friction=0.35,
        density=2700.0,      # kg/m³
        restitution=0.15,
        name="aluminum",
    ),
    "glass": MaterialProperties(
        friction=0.2,
        density=2500.0,      # kg/m³
        restitution=0.05,
        name="glass",
    ),
    # Common aliases
    "metal": MaterialProperties(
        friction=0.35,
        density=2700.0,
        restitution=0.15,
        name="metal",
    ),
    "rubber": MaterialProperties(
        friction=0.9,
        density=1100.0,
        restitution=0.6,
        name="rubber",
    ),
}


class MaterialLibrary:
    """Centralized repository of material properties.

    Loads material definitions from YAML files and provides lookup by name.
    Built-in materials (lunar_regolith, rock, aluminum, glass) are always
    available without any YAML file.  Additional materials can be loaded
    via load_from_yaml().

    Materials are immutable after construction; repeated load_from_yaml()
    calls merge new materials into the library (existing names are overwritten).
    """

    def __init__(self) -> None:
        # Start with builtins; load_from_yaml() may extend/override these.
        self._materials: Dict[str, MaterialProperties] = dict(_BUILTIN_MATERIALS)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_from_yaml(self, path: str) -> None:
        """Load material definitions from a YAML file.

        Expected YAML structure::

            materials:
              - name: "lunar_regolith"
                friction: 0.6
                density: 1500
                restitution: 0.02

        Parameters:
            path: Absolute or relative path to YAML material definition file.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If YAML structure is invalid or materials have missing fields.
        """
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict) or "materials" not in data:
            raise ValueError(
                f"YAML at {path!r} must have a top-level 'materials' list."
            )

        for entry in data["materials"]:
            required = ("name", "friction", "density", "restitution")
            missing = [k for k in required if k not in entry]
            if missing:
                raise ValueError(
                    f"Material entry missing required keys {missing}: {entry!r}"
                )
            mat = MaterialProperties(
                friction=float(entry["friction"]),
                density=float(entry["density"]),
                restitution=float(entry["restitution"]),
                name=str(entry["name"]),
            )
            self._materials[mat.name] = mat

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_material(self, name: str) -> MaterialProperties:
        """Retrieve a material by name.

        Parameters:
            name: Material identifier (e.g., "lunar_regolith", "aluminum").

        Returns:
            MaterialProperties object with all properties for the material.

        Raises:
            KeyError: If material name is not found in library.
        """
        try:
            return self._materials[name]
        except KeyError:
            raise KeyError(
                f"Material {name!r} not found in MaterialLibrary. "
                f"Available: {sorted(self._materials)}"
            ) from None

    def list_materials(self) -> List[str]:
        """Get list of all loaded material names.

        Returns:
            List of material identifiers currently available in the library.
        """
        return sorted(self._materials)

    def validate_all_referenced_materials(self, names: List[str]) -> List[str]:
        """Return names not present in the library (for cross-reference validation).

        Parameters:
            names: List of material names to validate.

        Returns:
            List of names that are missing from the library. Empty = all valid.
        """
        return [n for n in names if n not in self._materials]
