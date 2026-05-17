"""Default lightweight terramechanics stack (rigid + analytic, no MPM).

This is the **canonical default** wheel/regolith interaction path for normal
simulation, RL, and replay. It is GPU-free, runs at normal substeps, and is
CPU-deterministic — it deliberately does *not* use the Genesis MPM particle
solver, avoiding the MPM CFL substep tax (~10x slowdown) and the CUDA
requirement (see :class:`GenesisMPMRegolith` and CLAUDE.md for the opt-in MPM
tier).

It composes three pieces that already exist independently into one turnkey
object:

* **Rigid heightfield terrain** — provided by the physics engine, already
  regolith-tuned (rho=1800, friction=1.2, coup_restitution=0.02). This is the
  actual contact surface the wheels roll on; the engine owns it. This module
  does not touch the engine and therefore needs no GPU.
* :class:`~moon_rover.rover.drive.lunar_wheel_terrain.LunarRegolithWheelTerrain`
  — analytic slip ratio (kinematic), Pacejka traction, Bekker-Wong sinkage,
  and the cable-drag traction-degradation factor.
* :class:`~moon_rover.environment.regolith.GenesisMPMRegolith` constructed with
  ``engine=None`` — its analytic core gives deterministic per-cell rut depth
  with repeated-pass compaction and embedded-cable soil drag. Its
  :meth:`get_sinkage_at` is wired straight into the wheel-terrain model as the
  ``rut_sampler`` so rut history feeds back into the wheel model.

The result: register a wheel contact each step via :meth:`update_wheel` and you
get slip, traction, sinkage, and accumulating rut feedback with no MPM, at
normal substeps, validated on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from moon_rover.environment.regolith import GenesisMPMRegolith, RegolithConfig
from moon_rover.rover.drive.lunar_wheel_terrain import LunarRegolithWheelTerrain
from moon_rover.rover.drive.wheel_terrain import WheelTerrainConfig

if TYPE_CHECKING:
    from moon_rover.environment.terrain.generator import TerrainOutput

__all__ = [
    "TerramechanicsWheelState",
    "AnalyticTerramechanics",
    "default_analytic_terramechanics",
    "flat_regolith_terrain",
]


@dataclass(frozen=True)
class TerramechanicsWheelState:
    """Per-wheel terramechanics readout for a single contact update.

    Attributes:
        slip_ratio: Kinematic longitudinal slip in [0, 1].
        traction_force: Longitudinal traction in N (Pacejka), already scaled by
            the cable-drag degradation factor.
        traction_scale: Cable-drag traction multiplier in [0, 1] applied to the
            raw Pacejka force (1.0 = no cable effect).
        sinkage: Instantaneous Bekker-Wong sinkage for this load in m.
        rut_depth: Accumulated rut/compaction depth at this position in m,
            including the contribution of this pass.
    """

    slip_ratio: float
    traction_force: float
    traction_scale: float
    sinkage: float
    rut_depth: float


def flat_regolith_terrain(resolution: int = 64) -> "TerrainOutput":
    """Build a flat z=0 :class:`TerrainOutput` for the analytic stack.

    The analytic terramechanics path only needs a (square) heightfield; a flat
    bed is the canonical reference surface for slip/sinkage/rut studies and for
    the CPU integration tests. Real missions pass the generated terrain instead.
    """
    from moon_rover.environment.terrain.generator import TerrainOutput

    res = int(resolution)
    if res < 2:
        raise ValueError(f"resolution must be >= 2, got {resolution}")
    return TerrainOutput(
        height_field=np.zeros((res, res), dtype=np.float32),
        slope_map=np.zeros((res, res), dtype=np.float32),
        normal_map=np.tile(np.array([0, 0, 1], np.float32), (res, res, 1)),
        rock_positions=[],
        crater_list=[],
        nav_mesh=np.ones((res, res), dtype=np.uint8),
    )


class AnalyticTerramechanics:
    """Turnkey default regolith-interaction stack (rigid + analytic, no MPM).

    Owns an analytic-only regolith deformation model and a lunar wheel-terrain
    model wired to it. After :meth:`initialize`, call :meth:`update_wheel` once
    per wheel per physics step; repeated passes over the same track accumulate
    rut depth deterministically. No engine, GPU, or MPM solver is involved.

    Parameters:
        terrain_size_m: Side length (m) of the square terrain the heightfield
            spans (matches the ``size_m`` the engine registers the terrain
            with). World wheel positions must lie in ``[0, terrain_size_m]``.
        wheel_radius_m: Wheel radius (m) — Bekker-Wong length term.
        wheel_width_m: Effective wheel contact width (m).
        regolith_config: Soil properties. ``mpm_enabled`` is forced ``False``
            here — this stack is the analytic tier by definition.
        wheel_terrain_config: Optional override for the Pacejka/Bekker
            parameters; defaults to the lunar regolith production tuning.
    """

    def __init__(
        self,
        *,
        terrain_size_m: float = 100.0,
        wheel_radius_m: float = 0.30,
        wheel_width_m: float = 0.15,
        regolith_config: Optional[RegolithConfig] = None,
        wheel_terrain_config: Optional[WheelTerrainConfig] = None,
    ) -> None:
        self._regolith = GenesisMPMRegolith(
            engine=None,  # analytic-only by construction — no MPM, no GPU
            terrain_size_m=terrain_size_m,
            wheel_radius_m=wheel_radius_m,
        )
        # rut_sampler is the analytic regolith field: rut history feeds back
        # into the wheel-terrain model so rolling resistance/traction see it.
        self._wheel_terrain = LunarRegolithWheelTerrain(
            config=wheel_terrain_config,
            wheel_width_m=wheel_width_m,
            wheel_radius_m=wheel_radius_m,
            rut_sampler=self._regolith.get_sinkage_at,
        )
        self._terrain_size_m = float(terrain_size_m)
        self._initialized = False

    @property
    def wheel_terrain(self) -> LunarRegolithWheelTerrain:
        """The underlying analytic wheel-terrain model."""
        return self._wheel_terrain

    @property
    def regolith(self) -> GenesisMPMRegolith:
        """The underlying analytic regolith deformation model."""
        return self._regolith

    def initialize(
        self,
        terrain: "TerrainOutput",
        regolith_config: Optional[RegolithConfig] = None,
    ) -> None:
        """Initialize the analytic regolith field over ``terrain``.

        Parameters:
            terrain: Heightfield source (use :func:`flat_regolith_terrain` for
                the flat reference bed).
            regolith_config: Soil properties; ``mpm_enabled`` is coerced to
                ``False`` so this stack never builds an MPM bed.

        Raises:
            ValueError: Propagated from regolith config/heightfield validation.
        """
        cfg = regolith_config or RegolithConfig()
        if cfg.mpm_enabled:
            # This stack is the analytic tier by definition; enabling MPM here
            # is a category error. Make the contract explicit rather than
            # silently building (or silently ignoring) an MPM bed.
            raise ValueError(
                "AnalyticTerramechanics is the GPU-free analytic tier; "
                "RegolithConfig.mpm_enabled must be False. Use GenesisMPMRegolith "
                "with an engine directly for the opt-in MPM tier."
            )
        self._regolith.initialize(cfg, terrain)
        self._initialized = True

    def step(self, dt: float) -> None:
        """Advance the regolith model's simulation clock by ``dt`` seconds."""
        self._require_initialized()
        self._regolith.step(dt)

    def update_wheel(
        self,
        position: NDArray[np.float32],
        *,
        wheel_angular_vel: float,
        ground_velocity: float,
        wheel_load_n: float,
        contact_radius_m: float,
        cable_tension_n: float = 0.0,
    ) -> TerramechanicsWheelState:
        """Compute the terramechanics state for one wheel and record the pass.

        Registers the contact with the regolith model (accumulating ruts with
        repeated passes over the same cell), then returns slip, sinkage, the
        cable-degraded traction force, and the post-pass rut depth.

        Parameters:
            position: World (x, y, z) of the wheel contact; XY must lie inside
                ``[0, terrain_size_m]``.
            wheel_angular_vel: Wheel spin in rad/s.
            ground_velocity: Wheel-centre ground speed magnitude in m/s.
            wheel_load_n: Normal load on the wheel in N (>= 0).
            contact_radius_m: Contact-patch radius in m (> 0).
            cable_tension_n: Cable tension at the wheel in N (>= 0).

        Returns:
            A :class:`TerramechanicsWheelState`.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
            ValueError: Propagated for out-of-bounds position or invalid load.
        """
        self._require_initialized()
        pos = np.asarray(position, dtype=np.float64).reshape(-1)

        slip = self._wheel_terrain.compute_slip_ratio(
            wheel_angular_vel, ground_velocity
        )
        raw_traction = self._wheel_terrain.compute_traction_force(
            slip, wheel_load_n
        )
        scale = self._wheel_terrain.compute_cable_drag_effect(
            cable_tension_n, wheel_load_n
        )
        sinkage = self._wheel_terrain.compute_sinkage(wheel_load_n, {})

        # Record the pass so repeated passes over the same track deepen the rut.
        self._regolith.apply_wheel_pass(
            pos, wheel_load_n=float(wheel_load_n),
            contact_radius_m=float(contact_radius_m),
        )
        rut_depth = self._wheel_terrain.compute_rut_state(pos)

        return TerramechanicsWheelState(
            slip_ratio=float(slip),
            traction_force=float(raw_traction * scale),
            traction_scale=float(scale),
            sinkage=float(sinkage),
            rut_depth=float(rut_depth),
        )

    def get_drag_force(
        self, cable_positions: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Soil-resistance drag on embedded cable nodes (analytic).

        Delegates to the regolith model; see
        :meth:`GenesisMPMRegolith.get_drag_force`.
        """
        self._require_initialized()
        return self._regolith.get_drag_force(cable_positions)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "AnalyticTerramechanics.initialize() must be called first"
            )


def default_analytic_terramechanics(
    *,
    terrain: Optional["TerrainOutput"] = None,
    terrain_size_m: float = 100.0,
    wheel_radius_m: float = 0.30,
    wheel_width_m: float = 0.15,
    regolith_config: Optional[RegolithConfig] = None,
) -> AnalyticTerramechanics:
    """Build and initialize the canonical default terramechanics stack.

    This is the one-call entry point for the GPU-free slip/sinkage/rut path.

    Parameters:
        terrain: Heightfield source. If ``None``, a flat reference bed is used.
        terrain_size_m: Square terrain extent in m.
        wheel_radius_m: Wheel radius in m.
        wheel_width_m: Wheel contact width in m.
        regolith_config: Soil properties (``mpm_enabled`` must be False).

    Returns:
        An initialized :class:`AnalyticTerramechanics` ready for
        :meth:`AnalyticTerramechanics.update_wheel`.
    """
    stack = AnalyticTerramechanics(
        terrain_size_m=terrain_size_m,
        wheel_radius_m=wheel_radius_m,
        wheel_width_m=wheel_width_m,
        regolith_config=regolith_config,
    )
    stack.initialize(
        terrain if terrain is not None else flat_regolith_terrain(),
        regolith_config,
    )
    return stack
