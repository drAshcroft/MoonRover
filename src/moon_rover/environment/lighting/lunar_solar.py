"""Concrete lunar solar illumination model (System 2.3).

`LunarSolarSystem` is the production implementation of the
:class:`~moon_rover.environment.lighting.solar.SolarSystem` interface.

It models the sun as a directional source on the airless lunar surface:

* Sun position is an (elevation, azimuth) pair.  With ``lunar_day_cycle``
  enabled, :meth:`update` sweeps the sun through a full synodic lunar day
  (29.53 Earth days) so long-duration missions see sunrise/noon/sunset/night.
* Irradiance (W/m^2) on an arbitrarily oriented surface follows the cosine
  law against the top-of-atmosphere solar constant; the Moon has no
  atmospheric attenuation so the surface value equals the orbital value.
* Illuminance (lux) is irradiance scaled by the luminous efficacy of
  extraterrestrial sunlight (~120 lm/W), giving the documented ~160 klx peak.

Consumers:

* ``PowerSystem`` uses :meth:`get_irradiance` / :meth:`get_sun_elevation_deg`
  for solar-array generation.
* ``SunSensor`` uses :meth:`get_sun_direction` for attitude tracking.
* ``ThermalModel`` uses irradiance for solar heating.

Coordinate frame
-----------------
World axes are ``x = East``, ``y = North``, ``z = Up``.  Azimuth is measured
clockwise from North (0 = North, 90 = East, 180 = South, 270 = West), matching
:class:`SolarConfig`.  :meth:`get_sun_direction` returns the unit vector
pointing *from the surface toward the sun*.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from moon_rover.environment.lighting.solar import (
    AlbedoMap,
    SolarConfig,
    SolarSystem,
)

__all__ = ["LunarSolarSystem"]

# Top-of-atmosphere solar irradiance at 1 AU (W/m^2). The Moon orbits at
# essentially the same heliocentric distance as Earth, and is airless, so the
# lunar surface value equals this to within annual eccentricity (~3.4%).
SOLAR_CONSTANT_W_M2 = 1361.0

# Luminous efficacy of extraterrestrial (unattenuated) sunlight. ~120 lm/W
# yields a sun-at-zenith surface illuminance of ~1.63e5 lux, matching the
# ~160,000 lux peak documented on the SolarSystem interface.
SOLAR_LUMINOUS_EFFICACY_LM_W = 120.0

# Synodic lunar day: 29.530589 Earth days, the sunrise-to-sunrise period
# experienced by a fixed point on the surface.
LUNAR_DAY_SECONDS = 29.530589 * 86400.0


class LunarSolarSystem(SolarSystem):
    """Directional-source solar model for the lunar surface.

    Parameters:
        albedo: Surface reflectance map. Currently used to expose albedo to
            callers (e.g. thermal/optical models); direct illuminance is the
            primary product.
        terrain_size_m: Optional terrain side length (square) in metres. When
            set, :meth:`get_illuminance_at` enforces XY bounds and can sample a
            terrain normal map.
        terrain_normal_map: Optional per-cell surface normals, shape
            ``(res, res, 3)`` as produced by the terrain generator. Used to
            orient the cosine law at a queried position.
    """

    def __init__(
        self,
        albedo: Optional[AlbedoMap] = None,
        terrain_size_m: Optional[float] = None,
        terrain_normal_map: Optional[NDArray[np.float32]] = None,
    ) -> None:
        self.albedo = albedo or AlbedoMap()
        self._config: Optional[SolarConfig] = None

        # Current (possibly time-varying) sun position. Elevation may go
        # negative during lunar night; consumers clamp via irradiance.
        self._elevation_deg: float = 0.0
        self._azimuth_deg: float = 0.0
        self._sim_time: float = 0.0

        self._terrain_size_m: Optional[float] = (
            float(terrain_size_m) if terrain_size_m is not None else None
        )
        self._terrain_normals: Optional[NDArray[np.float32]] = None
        if terrain_normal_map is not None:
            self.set_terrain(self._terrain_size_m, terrain_normal_map)

        self._camera_res: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ #
    # SolarSystem interface
    # ------------------------------------------------------------------ #
    def configure(self, config: SolarConfig) -> None:
        """Configure the sun position and day-cycle parameters.

        Raises:
            ValueError: elevation outside [0, 90], azimuth outside [0, 360],
                or time_scale <= 0.
        """
        if not 0.0 <= config.elevation_deg <= 90.0:
            raise ValueError(
                f"elevation_deg must be in [0, 90], got {config.elevation_deg}"
            )
        if not 0.0 <= config.azimuth_deg <= 360.0:
            raise ValueError(
                f"azimuth_deg must be in [0, 360], got {config.azimuth_deg}"
            )
        if config.time_scale <= 0.0:
            raise ValueError(
                f"time_scale must be > 0, got {config.time_scale}"
            )

        self._config = config
        self._elevation_deg = float(config.elevation_deg)
        self._azimuth_deg = float(config.azimuth_deg)
        self._sim_time = 0.0

    def update(self, sim_time: float) -> None:
        """Advance the sun position to ``sim_time`` seconds.

        With ``lunar_day_cycle`` disabled the sun is static and only the
        timestamp is recorded. With it enabled the azimuth sweeps a full
        360 deg per (scaled) synodic day and the elevation follows a sine
        whose positive peak is the configured ``elevation_deg``; the negative
        half-cycle represents lunar night (sun below the horizon).

        Raises:
            RuntimeError: If called before :meth:`configure`.
            ValueError: If ``sim_time`` is negative.
        """
        if self._config is None:
            raise RuntimeError("update() called before configure()")
        if sim_time < 0.0:
            raise ValueError(f"sim_time must be >= 0, got {sim_time}")

        self._sim_time = float(sim_time)
        if not self._config.lunar_day_cycle:
            return

        t_eff = sim_time * self._config.time_scale
        phase = (t_eff % LUNAR_DAY_SECONDS) / LUNAR_DAY_SECONDS  # [0, 1)

        peak_elev = self._config.elevation_deg
        # Sine day: 0 at "sunrise" (phase 0), peak at noon (phase 0.25),
        # 0 at "sunset" (phase 0.5), negative (night) through phase 1.
        self._elevation_deg = peak_elev * math.sin(2.0 * math.pi * phase)
        # Sun tracks across the sky; base azimuth is the configured offset.
        self._azimuth_deg = (self._config.azimuth_deg + 360.0 * phase) % 360.0

    def get_illuminance_at(self, position: NDArray[np.float32]) -> float:
        """Illuminance (lux) at a world position.

        Uses the terrain surface normal at ``position`` when a terrain normal
        map is configured, otherwise assumes a horizontal (+Z) surface. Returns
        0 when the sun is at or below the horizon.

        Raises:
            ValueError: If a terrain extent is configured and the XY position
                lies outside ``[0, terrain_size_m]``.
        """
        pos = np.asarray(position, dtype=np.float64).reshape(-1)
        if pos.size < 2:
            raise ValueError("position must have at least X and Y components")

        normal = np.array([0.0, 0.0, 1.0])
        if self._terrain_size_m is not None:
            size = self._terrain_size_m
            if not (0.0 <= pos[0] <= size and 0.0 <= pos[1] <= size):
                raise ValueError(
                    f"position {pos[:2].tolist()} outside terrain bounds "
                    f"[0, {size}]"
                )
            if self._terrain_normals is not None:
                normal = self._sample_terrain_normal(pos[0], pos[1])

        irradiance = self.get_irradiance(surface_normal=normal)
        return irradiance * SOLAR_LUMINOUS_EFFICACY_LM_W

    def get_shadow_mask(self, camera_pose: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Coarse global illumination mask for a configured camera.

        This model treats the sun as a single directional source, so without a
        terrain depth pass the mask is globally lit when the sun is above the
        horizon and globally dark otherwise. Per-pixel terrain self-shadowing
        is delegated to the renderer/perception layer.

        Raises:
            RuntimeError: If camera resolution was not set via
                :meth:`configure_camera`.
        """
        if self._camera_res is None:
            raise RuntimeError(
                "camera parameters not configured; call configure_camera()"
            )
        h, w = self._camera_res
        lit = 1 if self._elevation_deg > 0.0 else 0
        return np.full((h, w), lit, dtype=np.uint8)

    # ------------------------------------------------------------------ #
    # Production helpers (PowerSystem / SunSensor / ThermalModel)
    # ------------------------------------------------------------------ #
    def get_sun_elevation_deg(self) -> float:
        """Current sun elevation in degrees (negative during lunar night)."""
        return self._elevation_deg

    def get_sun_azimuth_deg(self) -> float:
        """Current sun azimuth in degrees, clockwise from North."""
        return self._azimuth_deg

    def get_sun_direction(self) -> NDArray[np.float64]:
        """Unit vector pointing from the surface toward the sun.

        Frame: ``x = East``, ``y = North``, ``z = Up``. During lunar night the
        returned vector has a negative Z component (sun below the horizon).
        """
        e = math.radians(self._elevation_deg)
        a = math.radians(self._azimuth_deg)
        cos_e = math.cos(e)
        vec = np.array(
            [cos_e * math.sin(a), cos_e * math.cos(a), math.sin(e)],
            dtype=np.float64,
        )
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-12 else vec

    def get_irradiance(
        self, surface_normal: Optional[NDArray[np.float32]] = None
    ) -> float:
        """Solar irradiance (W/m^2) on a surface.

        Cosine law against the lunar solar constant. With no normal supplied,
        a horizontal surface is assumed (irradiance = S * sin(elevation)).
        Returns 0 when the sun is at or below the horizon, or when the surface
        faces away from the sun.

        Parameters:
            surface_normal: Optional 3-vector; need not be normalised.
        """
        if self._elevation_deg <= 0.0:
            return 0.0

        sun_dir = self.get_sun_direction()
        if surface_normal is None:
            cos_incidence = sun_dir[2]  # dot with +Z
        else:
            n = np.asarray(surface_normal, dtype=np.float64).reshape(3)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                return 0.0
            cos_incidence = float(np.dot(sun_dir, n / nn))

        if cos_incidence <= 0.0:
            return 0.0
        return SOLAR_CONSTANT_W_M2 * cos_incidence

    # ------------------------------------------------------------------ #
    # Configuration setters
    # ------------------------------------------------------------------ #
    def set_terrain(
        self,
        terrain_size_m: Optional[float],
        terrain_normal_map: Optional[NDArray[np.float32]] = None,
    ) -> None:
        """Attach terrain extent and (optionally) a normal map for queries."""
        if terrain_size_m is not None:
            if terrain_size_m <= 0.0:
                raise ValueError(
                    f"terrain_size_m must be > 0, got {terrain_size_m}"
                )
            self._terrain_size_m = float(terrain_size_m)

        if terrain_normal_map is not None:
            nm = np.asarray(terrain_normal_map, dtype=np.float32)
            if nm.ndim != 3 or nm.shape[2] != 3 or nm.shape[0] != nm.shape[1]:
                raise ValueError(
                    "terrain_normal_map must have shape (res, res, 3), "
                    f"got {nm.shape}"
                )
            self._terrain_normals = nm

    def configure_camera(self, width: int, height: int) -> None:
        """Set the camera pixel resolution used by :meth:`get_shadow_mask`."""
        if width <= 0 or height <= 0:
            raise ValueError(
                f"camera resolution must be positive, got {width}x{height}"
            )
        self._camera_res = (int(height), int(width))

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _sample_terrain_normal(self, x: float, y: float) -> NDArray[np.float64]:
        """Nearest-cell terrain normal at world (x, y)."""
        assert self._terrain_normals is not None
        assert self._terrain_size_m is not None
        res = self._terrain_normals.shape[0]
        size = self._terrain_size_m
        frac_x = 0.0 if size <= 0.0 else x / size
        frac_y = 0.0 if size <= 0.0 else y / size
        j = int(np.clip(round(frac_x * (res - 1)), 0, res - 1))
        i = int(np.clip(round(frac_y * (res - 1)), 0, res - 1))
        n = self._terrain_normals[i, j].astype(np.float64)
        nn = np.linalg.norm(n)
        return n / nn if nn > 1e-12 else np.array([0.0, 0.0, 1.0])
