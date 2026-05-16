"""Unit tests for LunarSolarSystem (System 2.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from moon_rover.environment.lighting import (
    LunarSolarSystem,
    SolarConfig,
)
from moon_rover.environment.lighting.lunar_solar import (
    LUNAR_DAY_SECONDS,
    SOLAR_CONSTANT_W_M2,
)


# --------------------------------------------------------------------------- #
# Configuration / validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "elev,azim,tscale",
    [(-1.0, 0.0, 1.0), (91.0, 0.0, 1.0), (45.0, -5.0, 1.0),
     (45.0, 361.0, 1.0), (45.0, 0.0, 0.0)],
)
def test_configure_rejects_bad_values(elev, azim, tscale):
    sun = LunarSolarSystem()
    with pytest.raises(ValueError):
        sun.configure(SolarConfig(elevation_deg=elev, azimuth_deg=azim,
                                  time_scale=tscale))


def test_update_before_configure_raises():
    with pytest.raises(RuntimeError):
        LunarSolarSystem().update(1.0)


def test_update_negative_time_raises():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=30.0))
    with pytest.raises(ValueError):
        sun.update(-1.0)


# --------------------------------------------------------------------------- #
# Static sun (no day cycle)
# --------------------------------------------------------------------------- #
def test_static_sun_position_constant():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=40.0, azimuth_deg=90.0))
    sun.update(0.0)
    sun.update(10_000.0)
    assert sun.get_sun_elevation_deg() == pytest.approx(40.0)
    assert sun.get_sun_azimuth_deg() == pytest.approx(90.0)


def test_sun_direction_unit_and_frame():
    sun = LunarSolarSystem()
    # Sun due East (azimuth 90), 30 deg up.
    sun.configure(SolarConfig(elevation_deg=30.0, azimuth_deg=90.0))
    sun.update(0.0)
    d = sun.get_sun_direction()
    assert np.linalg.norm(d) == pytest.approx(1.0, abs=1e-9)
    # East = cos(30)*sin(90) = 0.866, North ~0, Up = sin(30) = 0.5.
    assert d[0] == pytest.approx(math.cos(math.radians(30.0)), abs=1e-9)
    assert abs(d[1]) < 1e-9
    assert d[2] == pytest.approx(0.5, abs=1e-9)


def test_zenith_sun_direction_is_up():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=90.0, azimuth_deg=0.0))
    sun.update(0.0)
    d = sun.get_sun_direction()
    np.testing.assert_allclose(d, [0.0, 0.0, 1.0], atol=1e-9)


# --------------------------------------------------------------------------- #
# Irradiance / illuminance
# --------------------------------------------------------------------------- #
def test_irradiance_horizontal_cosine_law():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=30.0))
    sun.update(0.0)
    # Horizontal surface: S * sin(elevation).
    expected = SOLAR_CONSTANT_W_M2 * math.sin(math.radians(30.0))
    assert sun.get_irradiance() == pytest.approx(expected, rel=1e-9)


def test_irradiance_zenith_is_solar_constant():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=90.0))
    sun.update(0.0)
    assert sun.get_irradiance() == pytest.approx(SOLAR_CONSTANT_W_M2, rel=1e-9)


def test_irradiance_zero_at_horizon():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=0.0))
    sun.update(0.0)
    assert sun.get_irradiance() == 0.0


def test_irradiance_normal_facing_away_is_zero():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=45.0))
    sun.update(0.0)
    # Surface normal pointing straight down faces away from the sun.
    assert sun.get_irradiance(surface_normal=np.array([0.0, 0.0, -1.0])) == 0.0


def test_irradiance_tilted_panel_toward_sun_is_higher():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=30.0, azimuth_deg=90.0))
    sun.update(0.0)
    flat = sun.get_irradiance()
    # Panel tilted to face the sun directly => full solar constant.
    aimed = sun.get_irradiance(surface_normal=sun.get_sun_direction())
    assert aimed == pytest.approx(SOLAR_CONSTANT_W_M2, rel=1e-9)
    assert aimed > flat


def test_illuminance_peak_near_160klx():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=90.0))
    sun.update(0.0)
    lux = sun.get_illuminance_at(np.array([0.0, 0.0, 0.0]))
    assert 150_000 <= lux <= 170_000


# --------------------------------------------------------------------------- #
# Lunar day/night cycle
# --------------------------------------------------------------------------- #
def test_day_cycle_noon_peak_and_night():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=60.0, lunar_day_cycle=True))

    # Phase 0.25 => noon => peak elevation.
    sun.update(0.25 * LUNAR_DAY_SECONDS)
    assert sun.get_sun_elevation_deg() == pytest.approx(60.0, abs=1e-6)
    assert sun.get_irradiance() > 0.0

    # Phase 0.75 => deep night => sun below horizon, no power.
    sun.update(0.75 * LUNAR_DAY_SECONDS)
    assert sun.get_sun_elevation_deg() < 0.0
    assert sun.get_irradiance() == 0.0
    assert sun.get_illuminance_at(np.array([0.0, 0.0, 0.0])) == 0.0


def test_day_cycle_azimuth_sweeps():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=45.0, azimuth_deg=0.0,
                              lunar_day_cycle=True))
    sun.update(0.0)
    a0 = sun.get_sun_azimuth_deg()
    sun.update(0.5 * LUNAR_DAY_SECONDS)
    a_half = sun.get_sun_azimuth_deg()
    assert a0 == pytest.approx(0.0, abs=1e-6)
    assert a_half == pytest.approx(180.0, abs=1e-6)


def test_time_scale_accelerates_cycle():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=60.0, lunar_day_cycle=True,
                              time_scale=1000.0))
    # Reach noon 1000x sooner in sim seconds.
    sun.update(0.25 * LUNAR_DAY_SECONDS / 1000.0)
    assert sun.get_sun_elevation_deg() == pytest.approx(60.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Terrain-aware illuminance
# --------------------------------------------------------------------------- #
def test_illuminance_out_of_bounds_raises():
    sun = LunarSolarSystem(terrain_size_m=50.0)
    sun.configure(SolarConfig(elevation_deg=45.0))
    sun.update(0.0)
    with pytest.raises(ValueError):
        sun.get_illuminance_at(np.array([60.0, 10.0, 0.0]))


def test_illuminance_uses_terrain_normal():
    res = 8
    normals = np.zeros((res, res, 3), dtype=np.float32)
    normals[..., 2] = 1.0  # flat, +Z everywhere
    sun = LunarSolarSystem(terrain_size_m=50.0, terrain_normal_map=normals)
    sun.configure(SolarConfig(elevation_deg=90.0))
    sun.update(0.0)
    lux = sun.get_illuminance_at(np.array([25.0, 25.0, 0.0]))
    assert lux > 150_000


def test_set_terrain_validates_normal_map_shape():
    sun = LunarSolarSystem()
    with pytest.raises(ValueError):
        sun.set_terrain(50.0, np.zeros((4, 4), dtype=np.float32))


# --------------------------------------------------------------------------- #
# Shadow mask
# --------------------------------------------------------------------------- #
def test_shadow_mask_requires_camera():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=45.0))
    sun.update(0.0)
    with pytest.raises(RuntimeError):
        sun.get_shadow_mask(np.zeros(7, dtype=np.float32))


def test_shadow_mask_lit_when_sun_up_dark_when_down():
    sun = LunarSolarSystem()
    sun.configure(SolarConfig(elevation_deg=30.0, lunar_day_cycle=True))
    sun.configure_camera(width=16, height=8)

    sun.update(0.25 * LUNAR_DAY_SECONDS)  # noon
    lit = sun.get_shadow_mask(np.zeros(7, dtype=np.float32))
    assert lit.shape == (8, 16)
    assert lit.dtype == np.uint8
    assert (lit == 1).all()

    sun.update(0.75 * LUNAR_DAY_SECONDS)  # night
    dark = sun.get_shadow_mask(np.zeros(7, dtype=np.float32))
    assert (dark == 0).all()


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_deterministic_for_same_inputs():
    def run():
        s = LunarSolarSystem()
        s.configure(SolarConfig(elevation_deg=55.0, azimuth_deg=120.0,
                                lunar_day_cycle=True, time_scale=10.0))
        s.update(123_456.0)
        return s.get_sun_elevation_deg(), s.get_sun_azimuth_deg(), s.get_irradiance()

    assert run() == run()
