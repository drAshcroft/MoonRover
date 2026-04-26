"""Opt-in real Genesis smoke tests for contact adapter behavior.

Run with:
    $env:MOON_ROVER_RUN_GENESIS_SMOKE="1"
    python -m pytest tests/physics_real -q -m genesis
"""

from __future__ import annotations

from ._helpers import require_real_genesis_smoke, run_smoke_payload

import pytest


@pytest.mark.genesis
@pytest.mark.slow
def test_real_genesis_box_ground_contacts() -> None:
    """Verify get_body_contacts() sees actual Genesis 0.4.4 entity contacts."""
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
    gs.morphs.Box(size=(0.25, 0.25, 0.25), pos=(0.0, 0.0, 0.45)),
    gs.materials.Rigid(rho=400),
    entity_type="rigid",
)
engine.build_scene(n_envs=1)
for _ in range(80):
    engine.step(cfg.timestep, render=False)
contacts = engine.get_body_contacts("box")
payload = {
    "contact_count": len(contacts),
    "body_b": contacts[0]["body_b"] if contacts else None,
    "force_n": contacts[0]["force_n"] if contacts else 0.0,
    "is_in_contact": engine.is_in_contact("box", "ground"),
}
print(json.dumps(payload))
"""
    )
    assert payload["contact_count"] > 0
    assert payload["body_b"] == "ground"
    assert payload["force_n"] > 0.0
    assert payload["is_in_contact"] is True
