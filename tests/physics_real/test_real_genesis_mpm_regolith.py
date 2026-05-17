"""Opt-in real Genesis smoke test for the MPM regolith soil bed.

Requires Genesis 0.4.4 + a CUDA backend. Run with:
    $env:MOON_ROVER_RUN_GENESIS_SMOKE="1"
    python -m pytest tests/physics_real -q -m genesis

This validates the GPU-only path of GenesisMPMRegolith._build_mpm_bed:
attaching an MPM granular bed to a real Genesis scene and stepping it. The
deterministic analytic deformation core is covered separately (CPU) in
tests/unit/test_genesis_mpm_regolith.py.
"""

from __future__ import annotations

import pytest

from ._helpers import require_real_genesis_smoke, run_smoke_payload


@pytest.mark.genesis
@pytest.mark.gpu
@pytest.mark.slow
def test_real_genesis_mpm_bed_settles() -> None:
    """The MPM regolith bed builds, steps, and SETTLES to near-static.

    Regression for the "sand in constant motion" symptom: a bed seated on
    the solver floor with MPM-appropriate substeps must shed kinetic energy
    over a gravity settle window rather than jitter indefinitely.
    """
    require_real_genesis_smoke()
    payload = run_smoke_payload(
        r"""
import json
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, "src")

from moon_rover.core.physics.engine import GenesisConfig
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine
from moon_rover.environment.regolith import GenesisMPMRegolith, RegolithConfig
from moon_rover.environment.terrain.generator import TerrainOutput

base = GenesisConfig.from_yaml("configs/physics.yaml")
# MPM CFL: substep_dt = timestep/substeps must stay < 2e-2*dx (dx=1/64).
# At timestep 1/240 that needs substeps >= 14; use 20 for transient margin.
cfg = replace(base, use_gpu=True, substeps=20)

engine = GenesisPhysicsEngine()
engine.configure(cfg, show_viewer=False)

res = 32
terrain = TerrainOutput(
    height_field=np.zeros((res, res), dtype=np.float32),
    slope_map=np.zeros((res, res), dtype=np.float32),
    normal_map=np.tile(np.array([0, 0, 1], np.float32), (res, res, 1)),
    rock_positions=[],
    crater_list=[],
    nav_mesh=np.ones((res, res), dtype=np.uint8),
)

sim = GenesisMPMRegolith(engine=engine, terrain_size_m=10.0)
sim.initialize(RegolithConfig(mpm_enabled=True), terrain)  # opt-in: builds the MPM bed entity
mpm_entity_built = sim._mpm_entity is not None

engine.build_scene(n_envs=1)

# A few steps to register initial transient, then a settle window.
for _ in range(10):
    engine.step(cfg.timestep, render=False)
    sim.step(cfg.timestep)
speed_early = sim.mean_particle_speed()

for _ in range(300):
    engine.step(cfg.timestep, render=False)
    sim.step(cfg.timestep)
speed_settled = sim.mean_particle_speed()

payload = {
    "mpm_entity_built": bool(mpm_entity_built),
    "speed_early": round(speed_early, 6),
    "speed_settled": round(speed_settled, 6),
}
print(json.dumps(payload))
""",
        timeout=600,
    )
    assert payload["mpm_entity_built"] is True
    # Settled regolith is near-static and well below the early transient.
    assert payload["speed_settled"] < 0.05, payload
    assert payload["speed_settled"] < 0.5 * payload["speed_early"] + 1e-6, payload
