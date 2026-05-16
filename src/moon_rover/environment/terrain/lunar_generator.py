"""Concrete lunar terrain generator (System 2.1).

`LunarTerrainGenerator` is the production implementation of the
:class:`~moon_rover.environment.terrain.generator.TerrainGenerator` interface.
It builds a reproducible lunar height-field from fractional Brownian motion,
carves impact craters with raised rims, optionally carves sinuous rilles,
scatters surface rocks, flattens the moonbase pad, and derives slope, normal
and navigation-mesh products consumed by the physics engine and path planner.

All randomness is driven by ``numpy.random.default_rng(config.seed)`` so a given
``TerrainConfig`` always produces byte-identical output, which the replay and
Monte-Carlo layers depend on.

Coordinate convention
----------------------
The height-field is a square ``(resolution, resolution)`` array.  ``height[i, j]``
is the terrain elevation in metres at world position::

    x = j / (resolution - 1) * size_m
    y = i / (resolution - 1) * size_m

i.e. the first axis is world ``y`` (rows) and the second axis is world ``x``
(columns).  This matches ``GenesisPhysicsEngine.add_terrain_entity``, which
reads ``res_y, res_x = height_field.shape``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from moon_rover.environment.terrain.generator import (
    TerrainConfig,
    TerrainGenerator,
    TerrainOutput,
)

__all__ = ["LunarTerrainGenerator"]


class LunarTerrainGenerator(TerrainGenerator):
    """Procedural lunar terrain generator.

    Parameters:
        max_traversable_slope_deg: Slope at or below which a cell is considered
            drivable when building the navigation mesh. Apollo/LRV-class rovers
            top out around 20-25 deg on regolith; 25 deg is the default.
        rock_clearance_m: Extra planar margin added around each rock radius when
            marking navigation-mesh cells impassable, accounting for rover width.
        rim_height_ratio: Crater rim height as a fraction of crater depth.
        rim_width_ratio: Crater rim falloff width as a fraction of crater radius.
        moonbase_pad_radius_m: Radius of the flattened pad around the moonbase
            anchor point. Set to 0 to disable pad flattening.
    """

    def __init__(
        self,
        max_traversable_slope_deg: float = 25.0,
        rock_clearance_m: float = 0.5,
        rim_height_ratio: float = 0.10,
        rim_width_ratio: float = 0.35,
        moonbase_pad_radius_m: float = 6.0,
    ) -> None:
        if not 0.0 < max_traversable_slope_deg < 90.0:
            raise ValueError(
                "max_traversable_slope_deg must be in (0, 90), "
                f"got {max_traversable_slope_deg}"
            )
        if rock_clearance_m < 0.0:
            raise ValueError(f"rock_clearance_m must be >= 0, got {rock_clearance_m}")
        if moonbase_pad_radius_m < 0.0:
            raise ValueError(
                f"moonbase_pad_radius_m must be >= 0, got {moonbase_pad_radius_m}"
            )
        self.max_traversable_slope_deg = float(max_traversable_slope_deg)
        self.rock_clearance_m = float(rock_clearance_m)
        self.rim_height_ratio = float(rim_height_ratio)
        self.rim_width_ratio = float(rim_width_ratio)
        self.moonbase_pad_radius_m = float(moonbase_pad_radius_m)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #
    def generate(self, config: TerrainConfig) -> TerrainOutput:
        """Generate complete terrain from configuration.

        See :meth:`TerrainGenerator.generate` for the full pipeline contract.
        """
        self._validate_config(config)

        rng = np.random.default_rng(config.seed)
        res = config.resolution
        size = float(config.size_m)
        # World metres between adjacent grid samples.
        cell_size = size / (res - 1)

        # World-space coordinate grids. xx varies along columns (axis 1, world
        # x), yy varies along rows (axis 0, world y).
        axis = np.linspace(0.0, size, res, dtype=np.float64)
        xx, yy = np.meshgrid(axis, axis)  # both shape (res, res)

        # 1. Base height-field from fractional Brownian motion.
        height = self._fbm(rng, res, config.fBm_octaves) * float(config.fBm_amplitude)

        # 2. Carve craters (raised-rim bowl model).
        crater_list = self._carve_craters(rng, height, xx, yy, config)

        # 3. Carve rilles (sinuous collapsed-lava-tube valleys).
        if config.rille_enabled:
            self._carve_rilles(rng, height, xx, yy, config)

        # 4. Flatten the moonbase pad so the base sits level.
        self._flatten_moonbase_pad(height, xx, yy, config)

        # 5. Scatter surface rocks (separate obstacles, not baked into height).
        rock_positions = self._place_rocks(
            rng, height, axis, config, cell_size
        )

        # 6. Derive slope and normal maps from the final height-field.
        slope_map, normal_map = self._compute_slope_normals(height, cell_size)

        # 7. Build the navigation mesh from slope + rock proximity.
        nav_mesh = self._build_nav_mesh(
            slope_map, rock_positions, axis, res, cell_size
        )

        return TerrainOutput(
            height_field=height.astype(np.float32),
            slope_map=slope_map.astype(np.float32),
            normal_map=normal_map.astype(np.float32),
            rock_positions=rock_positions,
            crater_list=crater_list,
            nav_mesh=nav_mesh.astype(np.uint8),
        )

    def export_genesis_heightfield(self, output: TerrainOutput) -> Any:
        """Convert terrain output to a Genesis ``gs.morphs.Terrain`` morph.

        Mirrors the morph construction in
        ``GenesisPhysicsEngine.add_terrain_entity`` so a caller can build the
        terrain morph directly from a :class:`TerrainOutput` if it is not going
        through :class:`TerrainComposer`.

        Raises:
            RuntimeError: If the Genesis library is not importable.
        """
        try:
            import genesis as gs
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Genesis is not installed; cannot export a Genesis "
                "heightfield. Install genesis==0.4.4 or consume "
                "TerrainOutput.height_field directly."
            ) from exc

        hf = np.asarray(output.height_field, dtype=np.float32)
        res_y, res_x = hf.shape
        size_m = float(output.config.size_m)
        horizontal_scale = size_m / max(res_x - 1, 1)
        return gs.morphs.Terrain(
            height_field=hf,
            horizontal_scale=horizontal_scale,
            vertical_scale=1.0,
            n_subterrains=(1, 1),
            subterrain_size=(float(res_x), float(res_y)),
        )

    def export_nav_mesh(self, output: TerrainOutput) -> NDArray[np.uint8]:
        """Return the binary traversability grid (0=impassable, 1=traversable).

        The navigation mesh is computed during :meth:`generate`; this accessor
        returns it as a defensive ``uint8`` copy so callers cannot mutate the
        cached output in place.
        """
        return np.asarray(output.nav_mesh, dtype=np.uint8).copy()

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_config(config: TerrainConfig) -> None:
        if config.resolution < 4:
            raise ValueError(
                f"resolution must be >= 4, got {config.resolution}"
            )
        if config.size_m <= 0.0:
            raise ValueError(f"size_m must be > 0, got {config.size_m}")
        if config.fBm_octaves < 1:
            raise ValueError(
                f"fBm_octaves must be >= 1, got {config.fBm_octaves}"
            )
        if config.fBm_amplitude < 0.0:
            raise ValueError(
                f"fBm_amplitude must be >= 0, got {config.fBm_amplitude}"
            )
        if config.rock_density < 0.0:
            raise ValueError(
                f"rock_density must be >= 0, got {config.rock_density}"
            )
        cp = config.crater_params or {}
        cmin = float(cp.get("min_radius_m", 1.0))
        cmax = float(cp.get("max_radius_m", 5.0))
        if cmin <= 0.0 or cmax <= 0.0:
            raise ValueError("crater radii must be > 0")
        if cmin > cmax:
            raise ValueError(
                f"crater min_radius_m ({cmin}) must be <= max_radius_m ({cmax})"
            )
        if int(cp.get("count", 0)) < 0:
            raise ValueError("crater count must be >= 0")
        if float(cp.get("depth_ratio", 0.3)) < 0.0:
            raise ValueError("crater depth_ratio must be >= 0")

    # ------------------------------------------------------------------ #
    # 1. fBm base field
    # ------------------------------------------------------------------ #
    def _fbm(
        self, rng: np.random.Generator, res: int, octaves: int
    ) -> NDArray[np.float64]:
        """Sum of value-noise octaves, normalised to roughly [-1, 1].

        Each octave is a random lattice smoothly interpolated with the
        quintic fade ``6t^5 - 15t^4 + 10t^3`` (Perlin's improved-noise fade),
        doubling frequency and halving amplitude per octave.
        """
        height = np.zeros((res, res), dtype=np.float64)
        amplitude = 1.0
        amplitude_sum = 0.0
        base_period = 4  # lattice cells across the terrain for octave 0

        for octave in range(octaves):
            period = base_period * (2 ** octave)
            # Cap lattice density at the grid resolution to avoid aliasing.
            period = min(period, max(res - 1, 1))
            height += amplitude * self._value_noise(rng, res, period)
            amplitude_sum += amplitude
            amplitude *= 0.5

        height /= max(amplitude_sum, 1e-9)
        # Re-centre to zero mean and scale so the result is ~[-1, 1].
        height -= height.mean()
        peak = np.abs(height).max()
        if peak > 1e-9:
            height /= peak
        return height

    @staticmethod
    def _value_noise(
        rng: np.random.Generator, res: int, period: int
    ) -> NDArray[np.float64]:
        """One octave of 2-D value noise sampled onto a ``res x res`` grid."""
        gw = period + 1
        lattice = rng.random((gw, gw), dtype=np.float64)

        coords = np.linspace(0.0, float(period), res, dtype=np.float64)
        idx0 = np.clip(np.floor(coords).astype(np.intp), 0, period - 1)
        idx1 = idx0 + 1
        t = coords - idx0
        # Quintic fade for C2-continuous interpolation.
        fade = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

        # Bilinear corners gathered via index broadcasting (square lattice).
        v00 = lattice[np.ix_(idx0, idx0)]
        v01 = lattice[np.ix_(idx0, idx1)]
        v10 = lattice[np.ix_(idx1, idx0)]
        v11 = lattice[np.ix_(idx1, idx1)]

        fx = fade[None, :]
        fy = fade[:, None]
        top = v00 * (1.0 - fx) + v01 * fx
        bot = v10 * (1.0 - fx) + v11 * fx
        return top * (1.0 - fy) + bot * fy

    # ------------------------------------------------------------------ #
    # 2. Craters
    # ------------------------------------------------------------------ #
    def _carve_craters(
        self,
        rng: np.random.Generator,
        height: NDArray[np.float64],
        xx: NDArray[np.float64],
        yy: NDArray[np.float64],
        config: TerrainConfig,
    ) -> List[Dict[str, Any]]:
        """Carve raised-rim bowl craters in place; return crater metadata."""
        cp = config.crater_params
        count = int(cp.get("count", 0))
        min_r = float(cp.get("min_radius_m", 1.0))
        max_r = float(cp.get("max_radius_m", 5.0))
        depth_ratio = float(cp.get("depth_ratio", 0.3))
        size = float(config.size_m)

        crater_list: List[Dict[str, Any]] = []
        for _ in range(count):
            cx = float(rng.uniform(0.0, size))
            cy = float(rng.uniform(0.0, size))
            radius = float(rng.uniform(min_r, max_r))
            depth = radius * depth_ratio

            dist = np.hypot(xx - cx, yy - cy)
            d = dist / radius  # normalised radial distance

            # Bowl cavity: -depth at centre, rising to 0 at the rim (d=1).
            cavity = np.where(d < 1.0, depth * (d * d - 1.0), 0.0)

            # Raised rim: Gaussian bump centred just outside the rim.
            rim_height = depth * self.rim_height_ratio
            rim_width = self.rim_width_ratio
            rim = rim_height * np.exp(-((d - 1.0) ** 2) / (2.0 * rim_width ** 2))
            rim = np.where(d > 1.0 - rim_width, rim, 0.0)

            height += cavity + rim

            z_floor = float(height[self._nearest_index(cy, size, height.shape[0]),
                                   self._nearest_index(cx, size, height.shape[1])])
            crater_list.append(
                {
                    "position": [cx, cy, z_floor],
                    "radius": radius,
                    "depth": depth,
                    "rim_height": rim_height,
                }
            )
        return crater_list

    # ------------------------------------------------------------------ #
    # 3. Rilles
    # ------------------------------------------------------------------ #
    def _carve_rilles(
        self,
        rng: np.random.Generator,
        height: NDArray[np.float64],
        xx: NDArray[np.float64],
        yy: NDArray[np.float64],
        config: TerrainConfig,
    ) -> None:
        """Carve 1-2 sinuous trench valleys in place (collapsed lava tubes)."""
        size = float(config.size_m)
        n_rilles = int(rng.integers(1, 3))  # 1 or 2

        for _ in range(n_rilles):
            # A rille is a sine-perturbed straight line crossing the terrain.
            vertical = bool(rng.integers(0, 2))
            base = float(rng.uniform(0.25 * size, 0.75 * size))
            amp = float(rng.uniform(0.05, 0.15) * size)
            wavelength = float(rng.uniform(0.4, 0.9) * size)
            half_width = float(rng.uniform(0.5, 2.0))
            depth = float(rng.uniform(0.5, 2.0))

            if vertical:
                centerline = base + amp * np.sin(2.0 * math.pi * yy / wavelength)
                offset = np.abs(xx - centerline)
            else:
                centerline = base + amp * np.sin(2.0 * math.pi * xx / wavelength)
                offset = np.abs(yy - centerline)

            # Smooth Gaussian trench cross-section.
            trench = -depth * np.exp(-(offset ** 2) / (2.0 * half_width ** 2))
            height += trench

    # ------------------------------------------------------------------ #
    # 4. Moonbase pad
    # ------------------------------------------------------------------ #
    def _flatten_moonbase_pad(
        self,
        height: NDArray[np.float64],
        xx: NDArray[np.float64],
        yy: NDArray[np.float64],
        config: TerrainConfig,
    ) -> None:
        """Blend the height-field toward a level pad around the moonbase."""
        pad_r = self.moonbase_pad_radius_m
        if pad_r <= 0.0:
            return

        bx, by = float(config.moonbase_position[0]), float(config.moonbase_position[1])
        dist = np.hypot(xx - bx, yy - by)

        # Target level = mean height inside the pad core.
        core = dist <= pad_r
        if not core.any():
            return
        target = float(height[core].mean())

        # Cosine blend over [pad_r, 2*pad_r]: full flatten inside the core,
        # untouched terrain beyond the falloff.
        falloff = 2.0 * pad_r
        w = np.clip((falloff - dist) / max(falloff - pad_r, 1e-9), 0.0, 1.0)
        w = np.where(dist <= pad_r, 1.0, w)
        # Smoothstep the blend weight for a seamless transition.
        w = w * w * (3.0 - 2.0 * w)
        height += (target - height) * w

    # ------------------------------------------------------------------ #
    # 5. Rocks
    # ------------------------------------------------------------------ #
    def _place_rocks(
        self,
        rng: np.random.Generator,
        height: NDArray[np.float64],
        axis: NDArray[np.float64],
        config: TerrainConfig,
        cell_size: float,
    ) -> List[Tuple[float, float, float, float]]:
        """Scatter surface rocks as ``(x, y, z, radius)`` obstacle tuples."""
        size = float(config.size_m)
        area = size * size
        n_rocks = int(round(config.rock_density * area))
        if n_rocks <= 0:
            return []

        rocks: List[Tuple[float, float, float, float]] = []
        res = height.shape[0]
        for _ in range(n_rocks):
            rx = float(rng.uniform(0.0, size))
            ry = float(rng.uniform(0.0, size))
            # Log-ish size distribution: many small rocks, few boulders.
            radius = float(rng.uniform(0.10, 0.60))
            if rng.random() < 0.05:
                radius = float(rng.uniform(0.60, 1.50))  # rare boulder

            i = self._nearest_index(ry, size, res)
            j = self._nearest_index(rx, size, res)
            ground_z = float(height[i, j])
            # Rock centre sits one radius above the terrain surface.
            rocks.append((rx, ry, ground_z + radius, radius))
        return rocks

    # ------------------------------------------------------------------ #
    # 6. Slope / normal maps
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_slope_normals(
        height: NDArray[np.float64], cell_size: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return (slope_deg, normal_map) from world-space height gradients."""
        # np.gradient axis 0 = world y (rows), axis 1 = world x (columns).
        dz_dy, dz_dx = np.gradient(height, cell_size)

        grad_mag = np.hypot(dz_dx, dz_dy)
        slope_deg = np.degrees(np.arctan(grad_mag))

        # Surface normal = normalize(-dz/dx, -dz/dy, 1).
        normals = np.empty(height.shape + (3,), dtype=np.float64)
        normals[..., 0] = -dz_dx
        normals[..., 1] = -dz_dy
        normals[..., 2] = 1.0
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals /= np.maximum(norm, 1e-9)
        return slope_deg, normals

    # ------------------------------------------------------------------ #
    # 7. Navigation mesh
    # ------------------------------------------------------------------ #
    def _build_nav_mesh(
        self,
        slope_map: NDArray[np.float64],
        rock_positions: List[Tuple[float, float, float, float]],
        axis: NDArray[np.float64],
        res: int,
        cell_size: float,
    ) -> NDArray[np.uint8]:
        """1 where slope is drivable and clear of rocks, else 0."""
        nav = (slope_map <= self.max_traversable_slope_deg).astype(np.uint8)

        if not rock_positions:
            return nav

        size = axis[-1]
        for (rx, ry, _rz, radius) in rock_positions:
            clear = radius + self.rock_clearance_m
            # Bounding box of affected cells (cheap, then exact circle test).
            i_c = self._nearest_index(ry, size, res)
            j_c = self._nearest_index(rx, size, res)
            span = int(math.ceil(clear / cell_size)) + 1
            i_lo, i_hi = max(0, i_c - span), min(res - 1, i_c + span)
            j_lo, j_hi = max(0, j_c - span), min(res - 1, j_c + span)
            if i_lo > i_hi or j_lo > j_hi:
                continue

            sub_y = axis[i_lo : i_hi + 1][:, None]
            sub_x = axis[j_lo : j_hi + 1][None, :]
            inside = (sub_x - rx) ** 2 + (sub_y - ry) ** 2 <= clear * clear
            nav[i_lo : i_hi + 1, j_lo : j_hi + 1][inside] = 0
        return nav

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _nearest_index(world_coord: float, size: float, res: int) -> int:
        """Map a world coordinate in [0, size] to the nearest grid index."""
        frac = 0.0 if size <= 0.0 else world_coord / size
        return int(np.clip(round(frac * (res - 1)), 0, res - 1))
