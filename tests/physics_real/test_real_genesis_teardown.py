"""Opt-in real Genesis smoke tests for teardown safety."""

from __future__ import annotations

import os

from ._helpers import require_real_genesis_smoke, run_smoke_payload

import pytest


@pytest.mark.genesis
@pytest.mark.slow
def test_real_genesis_teardown_returns_promptly_under_safe_policy() -> None:
    """Verify the supported teardown path completes without hanging."""
    require_real_genesis_smoke()
    payload = run_smoke_payload(
        r"""
import json
import os
import sys
from dataclasses import replace

import genesis as gs

sys.path.insert(0, "src")

from moon_rover.core.physics.engine import GenesisConfig
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine

os.environ.pop("MOON_ROVER_GENESIS_DESTROY_POLICY", None)

base = GenesisConfig.from_yaml("configs/physics.yaml")
cfg = replace(base, use_gpu=False, timestep=1.0 / 60.0, substeps=1, contact_iterations=4)
engine = GenesisPhysicsEngine()
engine.configure(cfg, show_viewer=False)
engine.add_entity("ground", gs.morphs.Plane(), gs.materials.Rigid(), entity_type="terrain")
for idx in range(12):
    engine.add_entity(
        f"box_{idx:03d}",
        gs.morphs.Box(size=(0.25, 0.25, 0.25), pos=(0.1 * idx, 0.0, 1.0 + 0.02 * idx)),
        gs.materials.Rigid(rho=700),
        entity_type="rigid",
    )
engine.build_scene(n_envs=1)
for _ in range(5):
    engine.step(cfg.timestep, render=False)
engine.teardown()
payload = {
    "phase": engine.get_phase().name,
    "runtime_initialized_after": GenesisPhysicsEngine._gs_initialized,
}
print(json.dumps(payload))
"""
    )
    assert payload["phase"] == "TEARDOWN"
    if os.name == "nt":
        assert payload["runtime_initialized_after"] is True
    else:
        assert payload["runtime_initialized_after"] is False
