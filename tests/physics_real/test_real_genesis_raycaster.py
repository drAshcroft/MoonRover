"""Opt-in real Genesis smoke tests for raycaster adapter behavior."""

from __future__ import annotations

from ._helpers import require_real_genesis_smoke, run_smoke_payload

import pytest


@pytest.mark.genesis
@pytest.mark.slow
def test_real_genesis_raycaster_registers_and_reads() -> None:
    """Verify register_raycaster()/query_raycaster() against Genesis 0.4.4."""
    require_real_genesis_smoke()
    payload = run_smoke_payload(
        r"""
import json
import sys
from dataclasses import replace

import genesis as gs

sys.path.insert(0, "src")

from moon_rover.core.physics.engine import GenesisConfig
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine

base = GenesisConfig.from_yaml("configs/physics.yaml")
cfg = replace(base, use_gpu=False, timestep=1.0 / 60.0, substeps=1, contact_iterations=4)
engine = GenesisPhysicsEngine()
engine.configure(cfg, show_viewer=False)
engine.add_entity("ground", gs.morphs.Plane(), gs.materials.Rigid(), entity_type="terrain")
engine.add_entity(
    "box",
    gs.morphs.Box(size=(0.25, 0.25, 0.25), pos=(0.0, 0.0, 0.5)),
    gs.materials.Rigid(),
    entity_type="rigid",
)
engine.register_raycaster(
    "lidar",
    "box",
    0,
    {"num_channels": 4, "elevation_range_deg": (-90.0, -10.0), "h_resolution_deg": 20.0},
    10.0,
)
engine.build_scene(n_envs=1)
for _ in range(3):
    engine.step(cfg.timestep, render=False)
data = engine.query_raycaster("lidar")
payload = {
    "distance_count": int(data["distances"].size),
    "position_shape": list(data["positions"].shape),
    "normal_shape": list(data["normals"].shape),
    "hit_count": int((data["distances"] < 10.0).sum()),
}
print(json.dumps(payload))
"""
    )
    assert payload["distance_count"] == 72
    assert payload["position_shape"] == [72, 3]
    assert payload["normal_shape"] == [72, 3]
    assert payload["hit_count"] > 0
