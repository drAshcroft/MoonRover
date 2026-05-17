"""Unit tests for GenesisMPMRegolith (System 2.2).

The analytic deformation core is CPU-deterministic and covered here without
markers. The Genesis MPM soil-bed path requires a CUDA backend and is marked
``@pytest.mark.gpu`` (deselect with ``-m 'not gpu'``); it is validated on a
GPU box, not on the CPU reference machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from moon_rover.environment.regolith import (
    GenesisMPMRegolith,
    RegolithConfig,
)
from moon_rover.environment.terrain.generator import TerrainOutput


def _flat_terrain(res: int = 64) -> TerrainOutput:
    """A flat TerrainOutput at z=0 (TerrainOutput carries no extent)."""
    hf = np.zeros((res, res), dtype=np.float32)
    return TerrainOutput(
        height_field=hf,
        slope_map=np.zeros((res, res), dtype=np.float32),
        normal_map=np.tile(np.array([0, 0, 1], np.float32), (res, res, 1)),
        rock_positions=[],
        crater_list=[],
        nav_mesh=np.ones((res, res), dtype=np.uint8),
    )


def _model(size_m: float = 50.0, **kw) -> GenesisMPMRegolith:
    m = GenesisMPMRegolith(terrain_size_m=size_m, **kw)
    m.initialize(RegolithConfig(), _flat_terrain())
    return m


# --------------------------------------------------------------------------- #
# Construction / config validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw", [
    {"terrain_size_m": 0.0},
    {"wheel_radius_m": -1.0},
    {"rut_compaction_gain": -0.1},
    {"cable_diameter_m": 0.0},
])
def test_bad_constructor_args(kw):
    with pytest.raises(ValueError):
        GenesisMPMRegolith(**kw)


@pytest.mark.parametrize("cfg_kw", [
    {"bulk_density_loose": -1.0},
    {"bulk_density_loose": 2000.0, "bulk_density_compacted": 1500.0},
    {"friction_angle_deg": 0.0},
    {"friction_angle_deg": 95.0},
    {"cohesion_kpa": -1.0},
    {"particle_resolution_near": 0.0},
    {"constitutive_model": "bogus"},
])
def test_initialize_rejects_bad_config(cfg_kw):
    m = GenesisMPMRegolith(terrain_size_m=50.0)
    with pytest.raises(ValueError):
        m.initialize(RegolithConfig(**cfg_kw), _flat_terrain())


def test_initialize_rejects_bad_heightfield():
    m = GenesisMPMRegolith(terrain_size_m=50.0)
    bad = _flat_terrain()
    bad.height_field = np.zeros((4, 8), dtype=np.float32)  # non-square
    with pytest.raises(ValueError):
        m.initialize(RegolithConfig(), bad)


def test_methods_before_initialize_raise():
    m = GenesisMPMRegolith(terrain_size_m=50.0)
    with pytest.raises(RuntimeError):
        m.step(0.1)
    with pytest.raises(RuntimeError):
        m.get_sinkage_at(np.array([1.0, 1.0, 0.0]))


# --------------------------------------------------------------------------- #
# Sinkage / rut formation
# --------------------------------------------------------------------------- #
def test_undisturbed_sinkage_is_zero():
    m = _model()
    assert m.get_sinkage_at(np.array([25.0, 25.0, 0.0])) == 0.0


def test_wheel_pass_creates_sinkage():
    m = _model()
    peak = m.apply_wheel_pass(np.array([25.0, 25.0, 0.0]),
                              wheel_load_n=200.0, contact_radius_m=0.15)
    assert peak > 0.0
    assert m.get_sinkage_at(np.array([25.0, 25.0, 0.0])) == pytest.approx(
        peak, rel=0.2
    )


def test_repeated_passes_deepen_rut():
    m = _model()
    p1 = m.apply_wheel_pass(np.array([25.0, 25.0, 0.0]), 200.0, 0.15)
    p5 = p1
    for _ in range(4):
        p5 = m.apply_wheel_pass(np.array([25.0, 25.0, 0.0]), 200.0, 0.15)
    assert p5 > p1  # compaction accumulates with passes


def test_heavier_load_sinks_more():
    light = _model().apply_wheel_pass(np.array([25.0, 25.0, 0.0]), 100.0, 0.15)
    heavy = _model().apply_wheel_pass(np.array([25.0, 25.0, 0.0]), 400.0, 0.15)
    assert heavy > light


def test_sinkage_bounded_by_wheel_radius():
    m = _model(wheel_radius_m=0.3)
    peak = m.apply_wheel_pass(np.array([25.0, 25.0, 0.0]),
                              wheel_load_n=1e7, contact_radius_m=0.15)
    assert peak <= 0.3 + 1e-9


def test_sinkage_localized_to_contact():
    m = _model()
    m.apply_wheel_pass(np.array([25.0, 25.0, 0.0]), 300.0, 0.2)
    assert m.get_sinkage_at(np.array([25.0, 25.0, 0.0])) > 0.0
    assert m.get_sinkage_at(np.array([5.0, 5.0, 0.0])) == 0.0


def test_out_of_bounds_raises():
    m = _model(size_m=50.0)
    with pytest.raises(ValueError):
        m.get_sinkage_at(np.array([60.0, 10.0, 0.0]))
    with pytest.raises(ValueError):
        m.apply_wheel_pass(np.array([-1.0, 10.0, 0.0]), 100.0, 0.1)


def test_determinism():
    def run():
        mm = _model()
        for k in range(3):
            mm.apply_wheel_pass(np.array([10.0 + k, 20.0, 0.0]), 250.0, 0.15)
        return mm.get_sinkage_at(np.array([11.0, 20.0, 0.0]))

    assert run() == run()


def test_mean_particle_speed_requires_mpm_bed():
    m = _model()  # analytic-only, no engine/MPM bed
    with pytest.raises(RuntimeError, match="no MPM bed"):
        m.mean_particle_speed()


def test_step_advances_time_and_validates():
    m = _model()
    m.step(0.01)
    m.step(0.01)
    assert m.get_sim_time() == pytest.approx(0.02)
    with pytest.raises(ValueError):
        m.step(0.0)


# --------------------------------------------------------------------------- #
# Cable drag
# --------------------------------------------------------------------------- #
def test_drag_zero_above_surface():
    m = _model()
    nodes = np.array([[10.0, 10.0, 1.0], [11.0, 10.0, 1.0]], dtype=np.float32)
    f = m.get_drag_force(nodes)
    assert f.shape == (2, 3)
    assert np.allclose(f, 0.0)


def test_drag_nonzero_when_embedded():
    m = _model()
    # Nodes 0.1 m below the flat z=0 surface.
    nodes = np.array(
        [[10.0, 10.0, -0.1], [11.0, 10.0, -0.1], [12.0, 10.0, -0.1]],
        dtype=np.float32,
    )
    f = m.get_drag_force(nodes)
    mags = np.linalg.norm(f, axis=1)
    assert (mags[:] > 0.0).all()
    # Drag opposes the cable tangent (here +X), so Fx should be negative.
    assert f[1, 0] < 0.0


def test_drag_scales_with_depth():
    m = _model()
    shallow = m.get_drag_force(np.array([[10.0, 10.0, -0.05]], np.float32))
    deep = m.get_drag_force(np.array([[10.0, 10.0, -0.50]], np.float32))
    assert np.linalg.norm(deep) > np.linalg.norm(shallow)


def test_drag_shape_validation_and_empty():
    m = _model()
    assert m.get_drag_force(np.zeros((0, 3), np.float32)).shape == (0, 3)
    with pytest.raises(ValueError):
        m.get_drag_force(np.zeros((4, 2), np.float32))
    with pytest.raises(ValueError):
        m.get_drag_force(np.array([[100.0, 10.0, -0.1]], np.float32))  # OOB


def test_feeds_lunar_wheel_terrain_rut_sampler():
    """get_sinkage_at is consumable as LunarRegolithWheelTerrain.rut_sampler."""
    from moon_rover.rover.drive.lunar_wheel_terrain import (
        LunarRegolithWheelTerrain,
    )

    m = _model()
    m.apply_wheel_pass(np.array([25.0, 25.0, 0.0]), 300.0, 0.15)
    wt = LunarRegolithWheelTerrain(rut_sampler=m.get_sinkage_at)
    rut = wt.compute_rut_state(np.array([25.0, 25.0, 0.0], dtype=np.float32))
    assert rut > 0.0


# --------------------------------------------------------------------------- #
# Genesis MPM soil bed — CPU-checkable failure contract
# --------------------------------------------------------------------------- #
# The real CUDA MPM-bed smoke test lives in
# tests/physics_real/test_real_genesis_mpm_regolith.py (subprocess-isolated,
# process-global Genesis lifecycle), consistent with the other real-Genesis
# tests. Here we only assert the off-GPU failure contract.
class _FakeCfg:
    use_gpu = False


class _FakeEngine:
    _scene = object()
    _config = _FakeCfg()


def test_engine_without_gpu_raises_actionable_error():
    """With the bed explicitly enabled, an engine without CUDA must fail loudly."""
    m = GenesisMPMRegolith(engine=_FakeEngine(), terrain_size_m=50.0)
    with pytest.raises(RuntimeError, match="CUDA backend"):
        m.initialize(RegolithConfig(mpm_enabled=True), _flat_terrain())


# --------------------------------------------------------------------------- #
# mpm_enabled wiring (default off — opt-in MPM bed)
# --------------------------------------------------------------------------- #
def test_mpm_disabled_by_default_skips_bed_even_with_engine():
    """Default config => analytic-only, no MPM solver entity, even with engine.

    The engine is never touched for an MPM build, so a CUDA-less fake engine
    must NOT raise and the analytic rut field must still work.
    """
    m = GenesisMPMRegolith(engine=_FakeEngine(), terrain_size_m=50.0)
    m.initialize(RegolithConfig(), _flat_terrain())  # mpm_enabled defaults False
    assert m._mpm_entity is None
    with pytest.raises(RuntimeError, match="no MPM bed"):
        m.mean_particle_speed()
    # Analytic rut field is still fully functional.
    peak = m.apply_wheel_pass(np.array([25.0, 25.0, 0.0]), 200.0, 0.15)
    assert peak > 0.0


def test_mpm_enabled_without_engine_raises():
    """mpm_enabled=True with no engine is a misconfiguration, not a downgrade."""
    m = GenesisMPMRegolith(terrain_size_m=50.0)  # engine=None
    with pytest.raises(RuntimeError, match="mpm_enabled=True but no engine"):
        m.initialize(RegolithConfig(mpm_enabled=True), _flat_terrain())
