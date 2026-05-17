"""System 2.2: Regolith Physics Model (Material Point Method).

This module provides an interface for simulating lunar regolith behavior
using Material Point Method (MPM), including wheel sinkage and cable-soil interaction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from moon_rover.environment.terrain.generator import TerrainOutput


@dataclass
class RegolithConfig:
    """Configuration for regolith Material Point Method simulation.

    Parameters:
        particle_resolution_near: Particle resolution near wheels/cables in meters (0.02 typical).
        particle_resolution_far: Particle resolution far from interaction points (0.10 typical).
        bulk_density_compacted: Bulk density of compacted regolith in kg/m^3 (1500 typical).
        bulk_density_loose: Bulk density of loose regolith in kg/m^3 (900 typical).
        friction_angle_deg: Internal friction angle in degrees (35-40 for lunar regolith).
        cohesion_kpa: Cohesion in kPa (0.5 typical for loose lunar soil).
        constitutive_model: Soil model ("drucker_prager", "cam_clay", "mcc").
        mpm_enabled: Opt-in switch for the Genesis MPM soil bed. Default
            ``False`` — the deterministic analytic Bekker-Wong rut field is
            always active and is the authoritative data product. When ``True``
            the high-fidelity MPM granular bed is also built; this requires a
            CUDA backend and a sufficiently small substep (CFL: substeps >= 14
            at timestep 1/240), making MPM scenes ~10x slower than rigid scenes.
            The bed is never enabled implicitly.
    """
    particle_resolution_near: float = 0.02
    particle_resolution_far: float = 0.10
    bulk_density_compacted: float = 1500.0
    bulk_density_loose: float = 900.0
    friction_angle_deg: float = 37.5
    cohesion_kpa: float = 0.5
    constitutive_model: str = "drucker_prager"
    mpm_enabled: bool = False


class RegolithSimulation(ABC):
    """Abstract interface for Material Point Method regolith simulation.

    Simulates lunar soil deformation, compaction, and interaction with wheels and cables.
    Uses MPM for accurate granular material behavior without explicit particle tracking.
    """

    @abstractmethod
    def initialize(self, config: RegolithConfig, terrain: TerrainOutput) -> None:
        """Initialize MPM simulation with terrain and material parameters.

        Sets up material point grid, initializes state variables (stress, strain, velocity),
        and prepares for wheel/cable interaction.

        Parameters:
            config: RegolithConfig with soil material properties.
            terrain: TerrainOutput from terrain generator (provides initial heightfield).

        Raises:
            ValueError: If config parameters are invalid (density < 0, angles out of range, etc.).
            RuntimeError: If MPM backend is not available.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance regolith simulation by dt seconds.

        Executes:
        1. Update constitutive model stresses
        2. Apply gravity and external forces
        3. Solve implicit dynamics
        4. Update material points

        Parameters:
            dt: Timestep in seconds. Typically matches physics engine timestep.
        """
        raise NotImplementedError

    @abstractmethod
    def get_sinkage_at(self, position: NDArray[np.float32]) -> float:
        """Get wheel sinkage depth at specified position.

        Sinkage is the vertical displacement of the soil surface due to wheel loading.
        Computed from the difference between undisturbed and current terrain height.

        Parameters:
            position: 3D position [x, y, z] in world coordinates.

        Returns:
            Sinkage depth in meters (positive downward). 0 if at undisturbed terrain.

        Raises:
            ValueError: If position is outside terrain bounds.
        """
        raise NotImplementedError

    @abstractmethod
    def get_drag_force(self, cable_positions: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute drag forces on cables from regolith interaction.

        Cables experience drag as they are pulled through soil. This method
        computes drag forces at each cable segment from soil resistance.

        Parameters:
            cable_positions: Array of cable node positions. Shape (N, 3) for N nodes.

        Returns:
            Drag force vectors for each node. Shape (N, 3).
            Units: Newtons.

        Raises:
            ValueError: If any position is outside terrain bounds.
        """
        raise NotImplementedError
