"""Unit tests for LunarTerrainGenerator (System 2.1)."""

from __future__ import annotations

import numpy as np
import pytest

from moon_rover.environment.terrain import (
    LunarTerrainGenerator,
    TerrainConfig,
    TerrainOutput,
)


def _config(**overrides) -> TerrainConfig:
    base = dict(
        seed=7,
        size_m=50.0,
        fBm_octaves=5,
        fBm_amplitude=2.0,
        rock_density=0.02,
        rille_enabled=True,
        moonbase_position=(25.0, 25.0, 0.0),
        resolution=64,
    )
    base.update(overrides)
    return TerrainConfig(**base)


# --------------------------------------------------------------------------- #
# Output shape / dtype contract
# --------------------------------------------------------------------------- #
def test_output_shapes_and_dtypes():
    gen = LunarTerrainGenerator()
    out = gen.generate(_config())

    assert isinstance(out, TerrainOutput)
    res = 64
    assert out.height_field.shape == (res, res)
    assert out.height_field.dtype == np.float32
    assert out.slope_map.shape == (res, res)
    assert out.slope_map.dtype == np.float32
    assert out.normal_map.shape == (res, res, 3)
    assert out.normal_map.dtype == np.float32
    assert out.nav_mesh.shape == (res, res)
    assert out.nav_mesh.dtype == np.uint8
    assert set(np.unique(out.nav_mesh)).issubset({0, 1})


def test_no_nans_in_products():
    out = LunarTerrainGenerator().generate(_config())
    assert np.isfinite(out.height_field).all()
    assert np.isfinite(out.slope_map).all()
    assert np.isfinite(out.normal_map).all()


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_seeded_reproducibility():
    gen = LunarTerrainGenerator()
    a = gen.generate(_config(seed=123))
    b = gen.generate(_config(seed=123))
    np.testing.assert_array_equal(a.height_field, b.height_field)
    np.testing.assert_array_equal(a.nav_mesh, b.nav_mesh)
    assert a.rock_positions == b.rock_positions
    assert a.crater_list == b.crater_list


def test_different_seeds_differ():
    gen = LunarTerrainGenerator()
    a = gen.generate(_config(seed=1))
    b = gen.generate(_config(seed=2))
    assert not np.array_equal(a.height_field, b.height_field)


# --------------------------------------------------------------------------- #
# Slope / normal correctness
# --------------------------------------------------------------------------- #
def test_flat_terrain_has_zero_slope_and_up_normals():
    # Zero amplitude, no craters, no rilles, no rocks, no pad => flat plane.
    cfg = _config(
        fBm_amplitude=0.0,
        rille_enabled=False,
        rock_density=0.0,
        crater_params={"count": 0, "min_radius_m": 1.0,
                       "max_radius_m": 2.0, "depth_ratio": 0.3},
    )
    gen = LunarTerrainGenerator(moonbase_pad_radius_m=0.0)
    out = gen.generate(cfg)

    assert np.allclose(out.height_field, out.height_field.flat[0], atol=1e-5)
    assert np.allclose(out.slope_map, 0.0, atol=1e-4)
    assert np.allclose(out.normal_map[..., 2], 1.0, atol=1e-5)
    assert (out.nav_mesh == 1).all()


def test_known_ramp_slope():
    """A constant-gradient ramp should report the analytic slope angle."""
    gen = LunarTerrainGenerator()
    size_m, res = 10.0, 11
    cell = size_m / (res - 1)
    # 1 m rise per metre of x => 45 degrees.
    ramp = np.tile(np.arange(res, dtype=np.float64) * cell, (res, 1))

    slope_deg, normals = gen._compute_slope_normals(ramp, cell)
    interior = slope_deg[2:-2, 2:-2]
    assert np.allclose(interior, 45.0, atol=1e-3)
    # Normal tilts away from +x with unit length.
    assert np.allclose(np.linalg.norm(normals, axis=2), 1.0, atol=1e-6)
    assert np.all(normals[2:-2, 2:-2, 0] < 0.0)


def test_unit_normals_everywhere():
    out = LunarTerrainGenerator().generate(_config())
    mags = np.linalg.norm(out.normal_map.astype(np.float64), axis=2)
    assert np.allclose(mags, 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# Craters / rocks
# --------------------------------------------------------------------------- #
def test_crater_count_matches_config():
    cfg = _config(
        crater_params={"count": 8, "min_radius_m": 1.0,
                       "max_radius_m": 3.0, "depth_ratio": 0.3}
    )
    out = LunarTerrainGenerator().generate(cfg)
    assert len(out.crater_list) == 8
    for c in out.crater_list:
        assert {"position", "radius", "depth"} <= set(c)
        assert c["radius"] > 0.0
        assert c["depth"] > 0.0


def test_craters_lower_terrain():
    """With craters enabled the minimum elevation drops below the no-crater run."""
    no_crater = _config(
        fBm_amplitude=0.0, rille_enabled=False, rock_density=0.0,
        crater_params={"count": 0, "min_radius_m": 1.0,
                       "max_radius_m": 3.0, "depth_ratio": 0.3},
    )
    with_crater = _config(
        fBm_amplitude=0.0, rille_enabled=False, rock_density=0.0,
        crater_params={"count": 10, "min_radius_m": 2.0,
                       "max_radius_m": 4.0, "depth_ratio": 0.3},
    )
    gen = LunarTerrainGenerator(moonbase_pad_radius_m=0.0)
    flat = gen.generate(no_crater)
    cratered = gen.generate(with_crater)
    assert cratered.height_field.min() < flat.height_field.min() - 0.1


def test_rock_density_scales_count():
    sparse = LunarTerrainGenerator().generate(_config(rock_density=0.01))
    dense = LunarTerrainGenerator().generate(_config(rock_density=0.10))
    assert len(dense.rock_positions) > len(sparse.rock_positions)
    for (x, y, z, r) in dense.rock_positions:
        assert 0.0 <= x <= 50.0 and 0.0 <= y <= 50.0
        assert r > 0.0


def test_zero_rock_density_no_rocks():
    out = LunarTerrainGenerator().generate(_config(rock_density=0.0))
    assert out.rock_positions == []


def test_rocks_block_nav_mesh():
    cfg = _config(
        fBm_amplitude=0.0, rille_enabled=False, rock_density=0.05,
        crater_params={"count": 0, "min_radius_m": 1.0,
                       "max_radius_m": 2.0, "depth_ratio": 0.3},
    )
    out = LunarTerrainGenerator(moonbase_pad_radius_m=0.0).generate(cfg)
    # Flat terrain: the only impassable cells must come from rock clearance.
    assert out.rock_positions
    assert (out.nav_mesh == 0).any()


# --------------------------------------------------------------------------- #
# Moonbase pad
# --------------------------------------------------------------------------- #
def test_moonbase_pad_is_flattened():
    cfg = _config(fBm_amplitude=3.0, moonbase_position=(25.0, 25.0, 0.0))
    gen = LunarTerrainGenerator(moonbase_pad_radius_m=6.0)
    out = gen.generate(cfg)

    res, size = 64, 50.0
    cx = cy = int(round(25.0 / size * (res - 1)))
    # 3x3 patch at the pad centre should be near-level.
    patch = out.height_field[cy - 1 : cy + 2, cx - 1 : cx + 2]
    assert patch.std() < 0.05


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "overrides",
    [
        {"resolution": 2},
        {"size_m": 0.0},
        {"fBm_octaves": 0},
        {"fBm_amplitude": -1.0},
        {"rock_density": -0.1},
        {"crater_params": {"count": 1, "min_radius_m": 5.0,
                           "max_radius_m": 1.0, "depth_ratio": 0.3}},
        {"crater_params": {"count": -1, "min_radius_m": 1.0,
                           "max_radius_m": 2.0, "depth_ratio": 0.3}},
    ],
)
def test_invalid_config_raises(overrides):
    with pytest.raises(ValueError):
        LunarTerrainGenerator().generate(_config(**overrides))


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        LunarTerrainGenerator(max_traversable_slope_deg=0.0)
    with pytest.raises(ValueError):
        LunarTerrainGenerator(max_traversable_slope_deg=95.0)
    with pytest.raises(ValueError):
        LunarTerrainGenerator(rock_clearance_m=-1.0)


# --------------------------------------------------------------------------- #
# Genesis export
# --------------------------------------------------------------------------- #
def test_export_genesis_heightfield_requires_genesis(monkeypatch):
    import builtins

    out = LunarTerrainGenerator().generate(_config())
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "genesis":
            raise ImportError("genesis not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="Genesis is not installed"):
        LunarTerrainGenerator().export_genesis_heightfield(out)


def test_export_nav_mesh_returns_defensive_copy():
    gen = LunarTerrainGenerator()
    out = gen.generate(_config())
    nm = gen.export_nav_mesh(out)
    assert nm.dtype == np.uint8
    assert nm.shape == out.nav_mesh.shape
    nm[0, 0] = 9
    assert out.nav_mesh[0, 0] != 9  # original untouched


# --------------------------------------------------------------------------- #
# Composer default-generator wiring
# --------------------------------------------------------------------------- #
def test_terrain_composer_default_generator_resolves():
    from moon_rover.core.scene.terrain_composer import TerrainComposer

    gen = TerrainComposer._default_generator()
    assert isinstance(gen, LunarTerrainGenerator)
