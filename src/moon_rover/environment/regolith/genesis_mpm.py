"""Concrete lunar regolith deformation model (System 2.2).

`GenesisMPMRegolith` implements the
:class:`~moon_rover.environment.regolith.mpm_model.RegolithSimulation`
interface with two cooperating layers:

1. **Analytic deformation field (deterministic, CPU).**
   A Bekker-Wong sinkage law plus a pass-counting compaction model maintains
   a per-cell rut-depth grid over the terrain. This is the data product fed
   to :class:`LunarRegolithWheelTerrain` — its ``rut_sampler`` argument is
   exactly :meth:`get_sinkage_at`. Repeated wheel passes over the same cell
   accumulate additional sinkage as the soil compacts (rut formation).

2. **Genesis MPM soil bed (high fidelity, GPU/CUDA only — opt-in).**
   Off by default. When ``RegolithConfig.mpm_enabled`` is ``True`` *and* an
   engine in the construction phase is supplied, an MPM granular patch is
   added to its scene so rigid wheels contact a real deforming substrate.
   This path requires Genesis + a CUDA backend and substeps>=14 (CFL), is
   ~10x slower than rigid scenes, and is exercised only by
   ``@pytest.mark.gpu`` tests; the analytic field above remains the
   authoritative, replay-deterministic data source either way.

Genesis 0.4.4 cannot serialise MPM particle state (see CLAUDE.md), so the
analytic field — not MPM readback — is what callers query every step.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from moon_rover.environment.regolith.mpm_model import (
    RegolithConfig,
    RegolithSimulation,
)

if TYPE_CHECKING:
    from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine
    from moon_rover.environment.terrain.generator import TerrainOutput

__all__ = ["GenesisMPMRegolith"]

# Mean lunar surface gravity (m/s^2).
LUNAR_GRAVITY = 1.625

# Reference Bekker-Wong moduli for loose dry lunar regolith, consistent with
# rover.drive.lunar_wheel_terrain so the wheel model and the soil model agree.
_REF_K_PHI = 820.0e3      # friction modulus k_phi (Pa/m^n)
_REF_K_C = 14.0e3         # cohesion modulus k_c (Pa/m^(n-1))
_REF_DENSITY = 900.0      # density the reference moduli were tuned at (kg/m^3)
_REF_COHESION_KPA = 1.7   # cohesion the reference k_c was tuned at (kPa)

_VALID_MODELS = {"drucker_prager", "cam_clay", "mcc"}


class GenesisMPMRegolith(RegolithSimulation):
    """Lunar regolith deformation model.

    Parameters:
        engine: Optional physics engine in the construction phase. An MPM soil
            bed is added to its scene only when this is supplied *and*
            ``RegolithConfig.mpm_enabled`` is ``True`` (and Genesis + CUDA are
            available). With the default config (``mpm_enabled=False``) the
            model runs analytic-only and no MPM solver entity is created, even
            if an engine is passed. When ``None`` the model is always
            analytic-only (no GPU needed).
        terrain_size_m: Side length (m) of the square terrain the height field
            spans. ``TerrainOutput`` carries no extent, so it is supplied here
            (matches the ``size_m`` TerrainComposer registers with the engine).
        wheel_radius_m: Reference wheel radius for the Bekker-Wong length term.
        rut_compaction_gain: Extra sinkage accrued per additional pass over a
            cell, as a fraction of the single-pass sinkage (logarithmic).
        cable_diameter_m: Effective cable cross-section for soil-drag area.
    """

    def __init__(
        self,
        engine: Optional["GenesisPhysicsEngine"] = None,
        *,
        terrain_size_m: float = 100.0,
        wheel_radius_m: float = 0.30,
        rut_compaction_gain: float = 0.35,
        cable_diameter_m: float = 0.02,
    ) -> None:
        if terrain_size_m <= 0.0:
            raise ValueError("terrain_size_m must be > 0")
        if wheel_radius_m <= 0.0:
            raise ValueError("wheel_radius_m must be > 0")
        if rut_compaction_gain < 0.0:
            raise ValueError("rut_compaction_gain must be >= 0")
        if cable_diameter_m <= 0.0:
            raise ValueError("cable_diameter_m must be > 0")

        self._engine = engine
        self._size_m = float(terrain_size_m)
        self._wheel_radius_m = float(wheel_radius_m)
        self._rut_gain = float(rut_compaction_gain)
        self._cable_diameter_m = float(cable_diameter_m)

        self._config: Optional[RegolithConfig] = None
        self._undisturbed: Optional[NDArray[np.float64]] = None
        self._rut_depth: Optional[NDArray[np.float64]] = None
        self._pass_count: Optional[NDArray[np.int32]] = None
        self._k_phi: float = _REF_K_PHI
        self._k_c: float = _REF_K_C
        self._sim_time: float = 0.0
        self._mpm_entity: Any = None  # Genesis MPM entity when GPU path active
        self._mpm_bed_bounds: Optional[tuple] = None  # ((lo),(hi)) diagnostics

    # ------------------------------------------------------------------ #
    # RegolithSimulation interface
    # ------------------------------------------------------------------ #
    def initialize(
        self, config: RegolithConfig, terrain: "TerrainOutput"
    ) -> None:
        """Validate config, build the rut field, optionally add an MPM bed.

        Raises:
            ValueError: If config parameters are out of valid ranges or the
                terrain height field is malformed.
            RuntimeError: If an engine was supplied but the Genesis MPM
                backend is unavailable (no CUDA / Genesis not installed).
        """
        self._validate_config(config)

        hf = np.asarray(terrain.height_field, dtype=np.float64)
        if hf.ndim != 2 or hf.shape[0] != hf.shape[1] or hf.shape[0] < 2:
            raise ValueError(
                f"terrain.height_field must be square 2-D (>=2), got {hf.shape}"
            )

        self._config = config
        self._undisturbed = hf.copy()
        # The rut field is decoupled from the (coarse) terrain grid: it is
        # resolved at the far-field particle resolution so wheel-scale contact
        # patches actually land on multiple cells.
        res_rut = max(2, int(round(self._size_m / config.particle_resolution_far)) + 1)
        self._rut_depth = np.zeros((res_rut, res_rut), dtype=np.float64)
        self._pass_count = np.zeros((res_rut, res_rut), dtype=np.int32)
        self._sim_time = 0.0

        # Map soil properties onto Bekker-Wong moduli, anchored to the
        # reference loose-regolith calibration.
        density_ratio = config.bulk_density_loose / _REF_DENSITY
        cohesion_ratio = config.cohesion_kpa / _REF_COHESION_KPA
        phi = math.radians(config.friction_angle_deg)
        # k_phi scales with density and internal friction; k_c with cohesion.
        self._k_phi = _REF_K_PHI * density_ratio * (
            math.tan(phi) / math.tan(math.radians(35.0))
        )
        self._k_c = _REF_K_C * max(cohesion_ratio, 1e-6)

        # The MPM soil bed is opt-in and expensive (CUDA + substeps>=14 for
        # CFL stability, ~10x slower than rigid scenes). It is built ONLY when
        # explicitly requested via RegolithConfig.mpm_enabled. The default
        # config produces no MPM solver entity — the analytic Bekker-Wong rut
        # field above remains the authoritative, replay-deterministic data
        # source either way. We never silently downgrade: an enabled flag with
        # no engine is a misconfiguration, not an implicit analytic fallback.
        if config.mpm_enabled:
            if self._engine is None:
                raise RuntimeError(
                    "RegolithConfig.mpm_enabled=True but no engine was supplied "
                    "to GenesisMPMRegolith. The MPM soil bed requires a physics "
                    "engine in the CONSTRUCTION phase (construct with "
                    "engine=...). Set mpm_enabled=False (default) for the "
                    "analytic-only path."
                )
            self._build_mpm_bed(config, terrain)

    def step(self, dt: float) -> None:
        """Advance the simulation by ``dt`` seconds.

        The analytic rut field is event-driven (updated by
        :meth:`apply_wheel_pass`), so this advances the MPM bed when present
        and tracks simulation time.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
            ValueError: If ``dt`` <= 0.
        """
        if self._config is None:
            raise RuntimeError("step() called before initialize()")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        self._sim_time += dt
        # The MPM bed is driven by the shared Genesis scene; the owning engine
        # is responsible for scene.step(). Nothing extra to integrate here.

    def get_sinkage_at(self, position: NDArray[np.float32]) -> float:
        """Accumulated rut/sinkage depth (m, positive down) at a position.

        Bilinearly samples the rut-depth field. Returns 0 over undisturbed
        soil. Safe to pass directly as ``LunarRegolithWheelTerrain``'s
        ``rut_sampler``.

        Raises:
            ValueError: If the XY position is outside the terrain bounds.
        """
        self._require_initialized()
        pos = np.asarray(position, dtype=np.float64).reshape(-1)
        if pos.size < 2:
            raise ValueError("position must have at least X and Y components")
        return float(self._bilinear(self._rut_depth, pos[0], pos[1]))

    def get_drag_force(
        self, cable_positions: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Soil-resistance drag on embedded cable nodes.

        For each node embedded below the (rutted) soil surface the resisting
        force magnitude is the passive soil shear
        ``(c + rho*g*depth*tan(phi)) * A_node`` directed opposite the local
        cable tangent. Nodes above the surface contribute zero.

        Parameters:
            cable_positions: (N, 3) world-space node positions.

        Returns:
            (N, 3) float32 drag-force vectors in Newtons.

        Raises:
            ValueError: If shape is not (N, 3) or any XY is out of bounds.
        """
        self._require_initialized()
        pts = np.asarray(cable_positions, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(
                f"cable_positions must have shape (N, 3), got {pts.shape}"
            )
        n = pts.shape[0]
        forces = np.zeros((n, 3), dtype=np.float64)
        if n == 0:
            return forces.astype(np.float32)

        cfg = self._config
        rho = cfg.bulk_density_compacted
        phi = math.radians(cfg.friction_angle_deg)
        cohesion_pa = cfg.cohesion_kpa * 1e3

        for i in range(n):
            x, y, z = pts[i]
            surface_z = self._surface_z(x, y)  # raises on out-of-bounds
            depth = surface_z - z
            if depth <= 0.0:
                continue
            # Node tributary length from neighbour spacing.
            if n == 1:
                seg_len = self._cable_diameter_m
                tangent = np.array([1.0, 0.0, 0.0])
            else:
                lo = max(i - 1, 0)
                hi = min(i + 1, n - 1)
                tangent = pts[hi] - pts[lo]
                seg_len = 0.5 * float(np.linalg.norm(tangent)) or self._cable_diameter_m
            tnorm = float(np.linalg.norm(tangent))
            t_hat = tangent / tnorm if tnorm > 1e-9 else np.array([1.0, 0.0, 0.0])

            area = seg_len * self._cable_diameter_m
            sigma_v = rho * LUNAR_GRAVITY * depth
            shear = (cohesion_pa + sigma_v * math.tan(phi)) * area
            forces[i] = -shear * t_hat

        return forces.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Extended API (non-ABC): rut formation driver
    # ------------------------------------------------------------------ #
    def apply_wheel_pass(
        self,
        position: NDArray[np.float32],
        wheel_load_n: float,
        contact_radius_m: float,
    ) -> float:
        """Register a wheel contact, deforming the soil and forming ruts.

        Computes single-pass Bekker-Wong sinkage for ``wheel_load_n``, then
        for every grid cell inside ``contact_radius_m`` accumulates rut depth
        with a logarithmic compaction gain per repeated pass.

        Returns:
            The peak rut depth (m) under this contact after the pass.

        Raises:
            ValueError: If load/radius are invalid or the XY centre is out of
                bounds.
        """
        self._require_initialized()
        if wheel_load_n < 0.0:
            raise ValueError("wheel_load_n must be >= 0")
        if contact_radius_m <= 0.0:
            raise ValueError("contact_radius_m must be > 0")

        pos = np.asarray(position, dtype=np.float64).reshape(-1)
        cx, cy = float(pos[0]), float(pos[1])
        if not (0.0 <= cx <= self._size_m and 0.0 <= cy <= self._size_m):
            raise ValueError(
                f"contact centre ({cx}, {cy}) outside terrain bounds "
                f"[0, {self._size_m}]"
            )

        z_single = self._bekker_sinkage(wheel_load_n, contact_radius_m)

        res = self._rut_depth.shape[0]
        cell = self._size_m / (res - 1)
        span = int(math.ceil(contact_radius_m / cell)) + 1
        j_c = self._index(cx)
        i_c = self._index(cy)
        i_lo, i_hi = max(0, i_c - span), min(res - 1, i_c + span)
        j_lo, j_hi = max(0, j_c - span), min(res - 1, j_c + span)

        cells = [
            (i, j)
            for i in range(i_lo, i_hi + 1)
            for j in range(j_lo, j_hi + 1)
            if (j * cell - cx) ** 2 + (i * cell - cy) ** 2
            <= contact_radius_m ** 2
        ]
        # A contact patch finer than the grid still leaves a rut: fall back to
        # the single nearest cell so sinkage is never silently dropped.
        if not cells:
            cells = [(i_c, j_c)]

        peak = 0.0
        for i, j in cells:
            self._pass_count[i, j] += 1
            passes = int(self._pass_count[i, j])
            # Repeated passes compact the soil -> extra sinkage that
            # saturates logarithmically.
            z_total = z_single * (1.0 + self._rut_gain * math.log(passes))
            if z_total > self._rut_depth[i, j]:
                self._rut_depth[i, j] = z_total
            peak = max(peak, self._rut_depth[i, j])
        return float(peak)

    def get_sim_time(self) -> float:
        """Total simulated time advanced via :meth:`step` (seconds)."""
        return self._sim_time

    def mean_particle_speed(self) -> float:
        """Mean speed (m/s) of the MPM bed's particles, for settling checks.

        Settled regolith should have a near-zero mean speed; a value that
        stays high step after step indicates the bed never reaches static
        equilibrium (numerical particle noise / unsettled fall).

        Raises:
            RuntimeError: If no MPM bed was built (analytic-only mode), or the
                installed Genesis MPM state API differs from the expected
                0.4.4 surface.
        """
        if self._mpm_entity is None:
            raise RuntimeError(
                "no MPM bed built; mean_particle_speed() requires the "
                "GPU/Genesis path (construct with engine=...)"
            )
        try:
            state = self._mpm_entity.get_state()
            vel = state.vel
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(
                "Could not read MPM particle velocities via "
                f"entity.get_state().vel ({type(exc).__name__}: {exc}); the "
                "installed Genesis MPM state API differs from 0.4.4."
            ) from exc
        if hasattr(vel, "detach"):
            vel = vel.detach().cpu().numpy()
        v = np.asarray(vel, dtype=np.float64).reshape(-1, 3)
        if v.size == 0:
            return 0.0
        return float(np.linalg.norm(v, axis=1).mean())

    # ------------------------------------------------------------------ #
    # Validation / helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_config(config: RegolithConfig) -> None:
        if config.bulk_density_loose <= 0.0 or config.bulk_density_compacted <= 0.0:
            raise ValueError("bulk densities must be > 0")
        if config.bulk_density_loose > config.bulk_density_compacted:
            raise ValueError(
                "bulk_density_loose must be <= bulk_density_compacted"
            )
        if not 0.0 < config.friction_angle_deg < 90.0:
            raise ValueError(
                f"friction_angle_deg must be in (0, 90), "
                f"got {config.friction_angle_deg}"
            )
        if config.cohesion_kpa < 0.0:
            raise ValueError("cohesion_kpa must be >= 0")
        if config.particle_resolution_near <= 0.0 or config.particle_resolution_far <= 0.0:
            raise ValueError("particle resolutions must be > 0")
        if config.constitutive_model not in _VALID_MODELS:
            raise ValueError(
                f"constitutive_model must be one of {sorted(_VALID_MODELS)}, "
                f"got {config.constitutive_model!r}"
            )

    def _require_initialized(self) -> None:
        if self._config is None or self._rut_depth is None:
            raise RuntimeError("regolith model used before initialize()")

    def _bekker_sinkage(self, load_n: float, contact_radius_m: float) -> float:
        """Bekker-Wong sinkage z for a circular contact patch under ``load``."""
        fz = max(load_n, 0.0)
        if fz <= 0.0:
            return 0.0
        b = min(contact_radius_m, self._wheel_radius_m)
        coeff = (self._k_c / max(b, 1e-6)) + self._k_phi
        if coeff < 1e-6:
            return 0.0
        # p = Fz / A,  A ≈ pi r^2 ;  p = (kc/b + k_phi) z^n, n = 1
        area = math.pi * contact_radius_m ** 2
        pressure = fz / max(area, 1e-9)
        z = pressure / coeff  # n = 1
        # Physical cap: cannot sink deeper than the wheel radius.
        return float(min(z, self._wheel_radius_m))

    def _index(self, world_coord: float) -> int:
        res = self._rut_depth.shape[0]
        frac = world_coord / self._size_m if self._size_m > 0.0 else 0.0
        return int(np.clip(round(frac * (res - 1)), 0, res - 1))

    def _bilinear(
        self, grid: NDArray[np.float64], x: float, y: float
    ) -> float:
        """Bilinear sample of ``grid`` at world (x, y); bounds-checked."""
        size = self._size_m
        if not (0.0 <= x <= size and 0.0 <= y <= size):
            raise ValueError(
                f"position ({x}, {y}) outside terrain bounds [0, {size}]"
            )
        res = grid.shape[0]
        gx = x / size * (res - 1) if size > 0.0 else 0.0
        gy = y / size * (res - 1) if size > 0.0 else 0.0
        j0 = int(np.clip(math.floor(gx), 0, res - 1))
        i0 = int(np.clip(math.floor(gy), 0, res - 1))
        j1 = min(j0 + 1, res - 1)
        i1 = min(i0 + 1, res - 1)
        tx = gx - j0
        ty = gy - i0
        top = grid[i0, j0] * (1.0 - tx) + grid[i0, j1] * tx
        bot = grid[i1, j0] * (1.0 - tx) + grid[i1, j1] * tx
        return top * (1.0 - ty) + bot * ty

    def _surface_z(self, x: float, y: float) -> float:
        """Current (rutted) soil surface elevation at world (x, y)."""
        undisturbed = self._bilinear(self._undisturbed, x, y)
        rut = self._bilinear(self._rut_depth, x, y)
        return undisturbed - rut

    # ------------------------------------------------------------------ #
    # Genesis MPM bed (GPU/CUDA only — validated by @pytest.mark.gpu)
    # ------------------------------------------------------------------ #
    def _build_mpm_bed(
        self, config: RegolithConfig, terrain: "TerrainOutput"
    ) -> None:
        """Add an MPM granular soil patch to the supplied engine's scene.

        Requires Genesis with a CUDA backend. Raises an actionable
        RuntimeError (never a silent fallback) if the MPM API or backend is
        unavailable, consistent with the strict-diagnostics policy.
        """
        engine = self._engine
        scene = getattr(engine, "_scene", None)
        if scene is None:
            raise RuntimeError(
                "engine has no constructed scene; call initialize() while the "
                "engine is in the CONSTRUCTION phase before build_scene()"
            )

        try:
            import genesis as gs
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "Genesis is not installed; the MPM soil bed requires "
                "genesis==0.4.4 with a CUDA backend. Run analytic-only by "
                "constructing GenesisMPMRegolith(engine=None)."
            ) from exc

        engine_cfg = getattr(engine, "_config", None)
        if engine_cfg is None or not getattr(engine_cfg, "use_gpu", False):
            raise RuntimeError(
                "Genesis MPM regolith requires a CUDA backend "
                "(GenesisConfig.use_gpu=True). The MPM solver is not "
                "supported on the CPU reference backend."
            )

        # MPM granular material mapped from the soil config. Genesis 0.4.4
        # exposes sand-like granular MPM; surface drift in the exact ctor
        # signature must fail loudly, not silently degrade.
        try:
            mpm_material = gs.materials.MPM.Sand(
                rho=float(config.bulk_density_loose),
                friction_angle=float(config.friction_angle_deg),
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(
                "Failed to construct gs.materials.MPM.Sand for the regolith "
                f"bed ({type(exc).__name__}: {exc}). The installed Genesis "
                "MPM material API differs from the expected 0.4.4 surface; "
                "update _build_mpm_bed() to match before GPU validation."
            ) from exc

        # The MPM solver runs in a small, fixed simulation domain (set by
        # GenesisPhysicsEngine's MPMOptions, default ~[-1,1]^2 x [0,1] with
        # ~3-cell safety padding). A whole-terrain bed cannot fit, and MPM is
        # only meaningful as a *local* high-fidelity patch under wheel/cable
        # contact. We therefore size the bed to the solver's actual padded
        # boundary. The analytic rut field (whole-terrain) remains the
        # authoritative data product; this patch is a local physical
        # substrate in MPM-solver-local coordinates, not world terrain coords.
        solver = getattr(scene, "mpm_solver", None)
        boundary = getattr(solver, "boundary", None)
        lo = getattr(boundary, "lower", None)
        hi = getattr(boundary, "upper", None)
        if lo is None or hi is None:
            raise RuntimeError(
                "Could not read scene.mpm_solver.boundary.lower/upper to size "
                "the MPM regolith bed. The installed Genesis MPM solver API "
                "differs from the expected 0.4.4 surface; update "
                "_build_mpm_bed() to locate the solver domain."
            )
        safe_lo = np.asarray(lo, dtype=np.float64)
        safe_hi = np.asarray(hi, dtype=np.float64)
        extent = safe_hi - safe_lo
        if np.any(extent <= 0.0):
            raise RuntimeError(
                f"Degenerate MPM solver domain (lower={safe_lo}, "
                f"upper={safe_hi}); cannot place a regolith bed."
            )

        # CFL stability guard. Genesis MPM is explicit: it is unstable when
        # substep_dt (= SimOptions.dt / substeps) exceeds ~2e-2 * dx. Past
        # that limit the bed never settles — particles jitter/explode rather
        # than behaving like static regolith. Fail loudly with the fix
        # instead of silently producing unphysical "constant motion" sand.
        dx = float(getattr(solver, "_dx", 1.0 / 64.0))
        substeps = max(int(getattr(engine_cfg, "substeps", 1)), 1)
        timestep = float(getattr(engine_cfg, "timestep"))
        substep_dt = timestep / substeps
        suggested_dt = 2e-2 * dx
        if substep_dt > suggested_dt:
            min_substeps = int(math.ceil(timestep / suggested_dt))
            raise RuntimeError(
                f"MPM substep_dt={substep_dt:.3e}s exceeds the Genesis "
                f"stability limit {suggested_dt:.3e}s (=2e-2*dx, dx={dx:.5f}). "
                f"The regolith bed would be numerically unstable (sand never "
                f"settles). Set GenesisConfig.substeps >= {min_substeps} at "
                f"timestep {timestep:.6f}s, or reduce the timestep, before "
                f"building the MPM bed."
            )

        # Inset 10% in X/Y for lateral margin, but seat the bed *on the
        # solver floor*. Insetting the bottom upward would leave the whole
        # sand block free-falling onto the MPM boundary, which never settles
        # (perpetual particle noise). The bed must start resting on its
        # support so settled regolith is static and only deforms under load.
        inset_xy = 0.10 * extent
        usable_lo = safe_lo + inset_xy
        usable_hi = safe_hi - inset_xy
        usable_ext = usable_hi - usable_lo

        requested_thickness = max(
            10.0 * config.particle_resolution_far,
            4.0 * config.particle_resolution_near,
        )
        # Floor the bed just above the padded boundary (epsilon so particles
        # stay strictly inside) and cap thickness to the domain.
        z_eps = 0.02 * extent[2]
        max_thickness = (safe_hi[2] - z_eps) - (safe_lo[2] + z_eps)
        bed_thickness = float(min(requested_thickness, 0.6 * max_thickness))

        center_x = 0.5 * (usable_lo[0] + usable_hi[0])
        center_y = 0.5 * (usable_lo[1] + usable_hi[1])
        bed_bottom_z = float(safe_lo[2] + z_eps)
        pos_z = bed_bottom_z + 0.5 * bed_thickness
        bed_size = (
            float(usable_ext[0]),
            float(usable_ext[1]),
            bed_thickness,
        )
        self._mpm_bed_bounds = (
            (center_x - bed_size[0] / 2, center_y - bed_size[1] / 2, bed_bottom_z),
            (center_x + bed_size[0] / 2, center_y + bed_size[1] / 2,
             bed_bottom_z + bed_thickness),
        )

        try:
            morph = gs.morphs.Box(
                pos=(float(center_x), float(center_y), float(pos_z)),
                size=bed_size,
            )
            # Register through the engine (not scene.add_entity directly) so
            # the bed is in the engine's entity registry and build_scene()
            # accounts for it.
            self._mpm_entity = engine.add_entity(
                "regolith_mpm_bed",
                morph,
                mpm_material,
                entity_type="mpm",
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context
            raise RuntimeError(
                "Failed to add the MPM soil bed entity via engine.add_entity "
                f"({type(exc).__name__}: {exc}). Bed bounds "
                f"{self._mpm_bed_bounds} inside solver domain "
                f"[{safe_lo.tolist()}, {safe_hi.tolist()}]. Verify the engine "
                "is in the CONSTRUCTION phase and the morph/material API "
                "matches Genesis 0.4.4 before GPU validation."
            ) from exc
