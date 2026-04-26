"""Unit tests for LunarRegolithWheelTerrain wheel-terrain physics."""
from __future__ import annotations

import numpy as np
import pytest

from moon_rover.rover.drive.lunar_wheel_terrain import (
    LunarRegolithWheelTerrain,
    default_lunar_regolith_config,
)


def _model(**kwargs) -> LunarRegolithWheelTerrain:
    return LunarRegolithWheelTerrain(**kwargs)


# ---------------------------------------------------------------------------
# Slip ratio
# ---------------------------------------------------------------------------


def test_slip_ratio_is_zero_for_pure_rolling():
    m = _model(wheel_radius_m=0.3)
    # Wheel angular = 10 rad/s → 3 m/s surface speed; ground = 3 m/s → no slip.
    assert m.compute_slip_ratio(10.0, 3.0) == pytest.approx(0.0, abs=1e-9)


def test_slip_ratio_is_one_when_wheel_spins_in_place():
    m = _model(wheel_radius_m=0.3)
    assert m.compute_slip_ratio(10.0, 0.0) == pytest.approx(1.0, abs=1e-9)


def test_slip_ratio_is_clipped_to_zero_for_reverse_slip():
    # Negative slip (ground moving faster than wheel) is clipped to 0 since
    # the production contract is a [0, 1] forward-slip ratio.
    m = _model(wheel_radius_m=0.3)
    assert m.compute_slip_ratio(1.0, 10.0) == pytest.approx(0.0, abs=1e-9)


def test_slip_ratio_intermediate():
    m = _model(wheel_radius_m=0.3)
    # Wheel 10 rad/s → 3 m/s, ground 2 m/s → (3 - 2)/3 = 0.3333.
    assert m.compute_slip_ratio(10.0, 2.0) == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_slip_ratio_tiny_speeds_return_zero():
    m = _model(wheel_radius_m=0.3)
    assert m.compute_slip_ratio(0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Pacejka traction
# ---------------------------------------------------------------------------


def test_traction_zero_slip_yields_zero_force():
    m = _model()
    assert m.compute_traction_force(0.0, 500.0) == pytest.approx(0.0, abs=1e-9)


def test_traction_scales_linearly_with_normal_force():
    m = _model()
    f1 = m.compute_traction_force(0.5, 100.0)
    f2 = m.compute_traction_force(0.5, 200.0)
    assert f2 == pytest.approx(2.0 * f1, rel=1e-6)


def test_traction_never_exceeds_peak_d_coefficient():
    m = _model()
    fz = 500.0
    peak = 0.55 * fz  # D = 0.55 in default pacejka params.
    # Sweep slip and confirm ceiling is not violated.
    for s in np.linspace(0.0, 1.0, 21):
        ft = m.compute_traction_force(float(s), fz)
        assert ft <= peak + 1e-6


def test_traction_has_an_interior_peak():
    # Pacejka curves rise quickly, peak, then gently fall — the peak must sit
    # in the (0, 1) interior, not at the endpoints.
    m = _model()
    fz = 400.0
    values = [m.compute_traction_force(s, fz) for s in np.linspace(0.0, 1.0, 41)]
    peak_idx = int(np.argmax(values))
    assert 0 < peak_idx < len(values) - 1


def test_traction_clips_slip_to_valid_range():
    m = _model()
    f_hi = m.compute_traction_force(1.5, 300.0)
    f_cap = m.compute_traction_force(1.0, 300.0)
    assert f_hi == pytest.approx(f_cap, rel=1e-9)


# ---------------------------------------------------------------------------
# Bekker-Wong sinkage
# ---------------------------------------------------------------------------


def test_sinkage_zero_for_zero_load():
    m = _model()
    assert m.compute_sinkage(0.0, {}) == pytest.approx(0.0, abs=1e-9)


def test_sinkage_monotonic_in_load():
    m = _model()
    z_light = m.compute_sinkage(100.0, {})
    z_heavy = m.compute_sinkage(1000.0, {})
    assert z_heavy > z_light > 0.0


def test_sinkage_capped_below_half_wheel_radius():
    m = _model(wheel_radius_m=0.3)
    # Massive load to force the cap.
    z = m.compute_sinkage(1.0e8, {})
    assert z <= 0.5 * 0.3 + 1e-9


def test_sinkage_overrides_by_soil_params():
    m = _model()
    softer = m.compute_sinkage(500.0, {"k_phi": 100.0e3})  # much softer soil
    default = m.compute_sinkage(500.0, {})
    assert softer > default


# ---------------------------------------------------------------------------
# Cable drag
# ---------------------------------------------------------------------------


def test_cable_drag_no_tension_returns_full_traction():
    m = _model()
    assert m.compute_cable_drag_effect(0.0, 500.0) == pytest.approx(1.0, abs=1e-9)


def test_cable_drag_floors_at_zero_under_high_tension():
    m = _model()
    assert m.compute_cable_drag_effect(10_000.0, 100.0) == pytest.approx(0.0, abs=1e-9)


def test_cable_drag_interpolates_between_endpoints():
    m = _model()
    v = m.compute_cable_drag_effect(50.0, 500.0)
    assert 0.0 < v < 1.0


# ---------------------------------------------------------------------------
# Rut sampler
# ---------------------------------------------------------------------------


def test_rut_state_zero_without_sampler():
    m = _model()
    assert m.compute_rut_state(np.zeros(3)) == 0.0


def test_rut_state_delegates_to_sampler():
    calls: list = []

    def sampler(pos):
        calls.append(np.asarray(pos, dtype=np.float64).tolist())
        return 0.02

    m = _model(rut_sampler=sampler)
    got = m.compute_rut_state(np.array([1.0, 2.0, 0.0]))
    assert got == pytest.approx(0.02)
    assert len(calls) == 1


def test_rut_state_swallows_sampler_errors():
    def bad(_pos):
        raise RuntimeError("sampler failure")

    m = _model(rut_sampler=bad)
    assert m.compute_rut_state(np.zeros(3)) == 0.0


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_default_config_has_pacejka_and_bekker_blocks():
    cfg = default_lunar_regolith_config()
    for key in ("B", "C", "D", "E"):
        assert key in cfg.pacejka_params
    for key in ("k_phi", "k_c", "sinkage_exponent"):
        assert key in cfg.bekker_params
