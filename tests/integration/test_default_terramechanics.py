"""CPU integration test for the default analytic terramechanics stack.

Exercises the GPU-free slip/sinkage/rut path end-to-end (no MPM particle
solver, no CUDA, normal substeps): a loaded wheel proxy doing repeated passes
over a rigid regolith track must show slip-dependent traction and a rut that
deepens with each pass, and the whole trajectory must be bit-identical on
replay.

This guards the canonical default regolith-interaction model — see
terramechanics.py and CLAUDE.md ("Regolith interaction tiers").
"""

from __future__ import annotations

import numpy as np
import pytest

from moon_rover.rover.drive import (
    AnalyticTerramechanics,
    default_analytic_terramechanics,
    flat_regolith_terrain,
)
from moon_rover.environment.regolith import RegolithConfig

TERRAIN_M = 50.0
TRACK_XY = (25.0, 25.0)
WHEEL_LOAD_N = 250.0
CONTACT_R = 0.15


def _stack() -> AnalyticTerramechanics:
    return default_analytic_terramechanics(
        terrain=flat_regolith_terrain(64),
        terrain_size_m=TERRAIN_M,
        wheel_radius_m=0.30,
        wheel_width_m=0.15,
    )


def _pos(x: float = TRACK_XY[0], y: float = TRACK_XY[1]) -> np.ndarray:
    return np.array([x, y, 0.0], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Wiring / contract
# --------------------------------------------------------------------------- #
def test_factory_is_gpu_free_and_initialized():
    """Turnkey: one call yields an analytic-only, ready-to-use stack."""
    tm = _stack()
    # Analytic regolith core, no MPM bed, no engine.
    assert tm.regolith._engine is None
    assert tm.regolith._mpm_entity is None
    # rut_sampler is wired: undisturbed soil reads zero rut.
    assert tm.update_wheel(
        _pos(), wheel_angular_vel=0.0, ground_velocity=0.0,
        wheel_load_n=0.0, contact_radius_m=CONTACT_R,
    ).rut_depth == 0.0


def test_rejects_mpm_enabled_config():
    """The analytic tier must refuse an MPM-enabled config explicitly."""
    tm = AnalyticTerramechanics(terrain_size_m=TERRAIN_M)
    with pytest.raises(ValueError, match="analytic tier"):
        tm.initialize(flat_regolith_terrain(32), RegolithConfig(mpm_enabled=True))


def test_use_before_initialize_raises():
    tm = AnalyticTerramechanics(terrain_size_m=TERRAIN_M)
    with pytest.raises(RuntimeError, match="initialize"):
        tm.update_wheel(
            _pos(), wheel_angular_vel=1.0, ground_velocity=0.5,
            wheel_load_n=WHEEL_LOAD_N, contact_radius_m=CONTACT_R,
        )


# --------------------------------------------------------------------------- #
# Slip-dependent traction
# --------------------------------------------------------------------------- #
def test_traction_is_slip_dependent():
    """Zero slip => ~zero traction; traction rises with slip (Pacejka)."""
    tm = _stack()
    ground_v = 1.0
    # wheel_speed = omega * r (r=0.3). Pure rolling: omega = v/r => slip 0.
    rolling_omega = ground_v / 0.30

    s_zero = tm.update_wheel(
        _pos(), wheel_angular_vel=rolling_omega, ground_velocity=ground_v,
        wheel_load_n=WHEEL_LOAD_N, contact_radius_m=CONTACT_R,
    )
    assert s_zero.slip_ratio == pytest.approx(0.0, abs=1e-6)
    assert abs(s_zero.traction_force) < 1e-6

    # Sweep increasing wheel overspeed => increasing slip => increasing traction.
    tractions = []
    last_slip = -1.0
    for over in (1.1, 1.5, 2.5, 6.0):
        st = tm.update_wheel(
            _pos(8.0, 8.0),  # fresh patch so rut history doesn't confound
            wheel_angular_vel=rolling_omega * over,
            ground_velocity=ground_v,
            wheel_load_n=WHEEL_LOAD_N, contact_radius_m=CONTACT_R,
        )
        assert 0.0 < st.slip_ratio <= 1.0
        assert st.slip_ratio > last_slip
        last_slip = st.slip_ratio
        tractions.append(st.traction_force)

    # Pacejka Magic Formula: traction rises steeply from low slip to an
    # intermediate-slip peak, then declines slightly toward full spin-out.
    assert all(t > 0.0 for t in tractions)
    assert tractions[0] < max(tractions)          # low slip is not the peak
    assert np.argmax(tractions) not in (0,)       # peak at intermediate slip
    assert max(tractions) > 2.0 * abs(s_zero.traction_force + 1e-9)


def test_traction_scales_with_normal_load():
    tm = _stack()
    omega, v = 5.0, 1.0
    light = tm.update_wheel(
        _pos(5, 5), wheel_angular_vel=omega, ground_velocity=v,
        wheel_load_n=100.0, contact_radius_m=CONTACT_R,
    )
    heavy = tm.update_wheel(
        _pos(40, 40), wheel_angular_vel=omega, ground_velocity=v,
        wheel_load_n=400.0, contact_radius_m=CONTACT_R,
    )
    assert heavy.traction_force > light.traction_force
    assert heavy.sinkage > light.sinkage  # Bekker: heavier => deeper


def test_cable_drag_degrades_traction():
    tm = _stack()
    omega, v = 5.0, 1.0
    free = tm.update_wheel(
        _pos(5, 5), wheel_angular_vel=omega, ground_velocity=v,
        wheel_load_n=WHEEL_LOAD_N, contact_radius_m=CONTACT_R,
        cable_tension_n=0.0,
    )
    taut = tm.update_wheel(
        _pos(45, 45), wheel_angular_vel=omega, ground_velocity=v,
        wheel_load_n=WHEEL_LOAD_N, contact_radius_m=CONTACT_R,
        cable_tension_n=300.0,
    )
    assert free.traction_scale == pytest.approx(1.0)
    assert taut.traction_scale < 1.0
    assert taut.traction_force < free.traction_force


# --------------------------------------------------------------------------- #
# Accumulating ruts over repeated passes
# --------------------------------------------------------------------------- #
def _drive_track(tm: AnalyticTerramechanics, passes: int) -> list[float]:
    """Drive `passes` repeated wheel passes over the same track cell."""
    ruts = []
    for _ in range(passes):
        st = tm.update_wheel(
            _pos(), wheel_angular_vel=5.0, ground_velocity=1.0,
            wheel_load_n=WHEEL_LOAD_N, contact_radius_m=CONTACT_R,
        )
        tm.step(0.01)
        ruts.append(st.rut_depth)
    return ruts


def test_rut_deepens_with_repeated_passes():
    tm = _stack()
    ruts = _drive_track(tm, passes=6)
    assert ruts[0] > 0.0
    # Non-decreasing, and strictly deeper after several compaction passes.
    assert ruts == sorted(ruts)
    assert ruts[-1] > ruts[0]
    # Undisturbed soil away from the track stays at zero.
    off = tm.update_wheel(
        _pos(2.0, 2.0), wheel_angular_vel=0.0, ground_velocity=0.0,
        wheel_load_n=0.0, contact_radius_m=CONTACT_R,
    )
    assert off.rut_depth == 0.0


# --------------------------------------------------------------------------- #
# Replay determinism
# --------------------------------------------------------------------------- #
def test_deterministic_on_replay():
    """Two independent runs of the same trajectory are bit-identical."""

    def run() -> tuple[list[float], float]:
        tm = _stack()
        states = []
        for k in range(8):
            st = tm.update_wheel(
                _pos(10.0 + 0.5 * k, 20.0),
                wheel_angular_vel=4.0 + 0.1 * k,
                ground_velocity=1.0,
                wheel_load_n=WHEEL_LOAD_N + 5.0 * k,
                contact_radius_m=CONTACT_R,
                cable_tension_n=20.0 * k,
            )
            tm.step(0.01)
            states.append(
                (st.slip_ratio, st.traction_force, st.sinkage, st.rut_depth)
            )
        final = tm.update_wheel(
            _pos(11.0, 20.0), wheel_angular_vel=4.0, ground_velocity=1.0,
            wheel_load_n=WHEEL_LOAD_N, contact_radius_m=CONTACT_R,
        ).rut_depth
        return states, final

    a_states, a_final = run()
    b_states, b_final = run()
    assert a_states == b_states
    assert a_final == b_final
