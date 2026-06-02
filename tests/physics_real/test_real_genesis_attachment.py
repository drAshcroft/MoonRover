"""Opt-in real Genesis smoke test for rigid attachment fallback."""

from __future__ import annotations

import pytest

from ._helpers import require_real_genesis_smoke, run_smoke_payload


@pytest.mark.genesis
@pytest.mark.slow
def test_real_genesis_attach_follow_detach_fall() -> None:
    require_real_genesis_smoke()
    payload = run_smoke_payload(
        r"""
import json
import sys
from dataclasses import replace

import genesis as gs
import numpy as np

sys.path.insert(0, "src")

from moon_rover.core.physics.engine import GenesisConfig
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine

base = GenesisConfig.from_yaml("configs/physics.yaml")
cfg = replace(base, use_gpu=False, timestep=1.0 / 60.0, substeps=1, contact_iterations=4)
engine = GenesisPhysicsEngine()
engine.configure(cfg, show_viewer=False)
engine.add_entity("ground", gs.morphs.Plane(), gs.materials.Rigid(), entity_type="terrain")
engine.add_entity("parent", gs.morphs.Box(size=(0.25, 0.25, 0.25), pos=(0.0, 0.0, 1.0)), gs.materials.Rigid(rho=600))
engine.add_entity("child", gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 1.5)), gs.materials.Rigid(rho=400))
engine.build_scene()
handle = engine.attach_bodies("parent", "child")
initial_parent, _ = engine.get_body_pose("parent")
initial_child, _ = engine.get_body_pose("child")
initial_offset = initial_child - initial_parent
engine.set_body_velocity("parent", np.array([0.4, 0.0, 0.7], dtype=np.float32), np.zeros(3, dtype=np.float32))
for _ in range(4):
    engine.step(cfg.timestep, render=False)
follow_parent, _ = engine.get_body_pose("parent")
follow_child, _ = engine.get_body_pose("child")
follow_offset = follow_child - follow_parent
snap = engine.save_snapshot()
engine.step(cfg.timestep, render=False)
branch_child, _ = engine.get_body_pose("child")
engine.restore_snapshot(snap)
engine.step(cfg.timestep, render=False)
replay_child, _ = engine.get_body_pose("child")
engine.detach_bodies(handle)
engine.set_body_velocity("child", np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
released_z, _ = engine.get_body_pose("child")
for _ in range(12):
    engine.step(cfg.timestep, render=False)
fallen_z, _ = engine.get_body_pose("child")
print(json.dumps({
    "follow_error": float(np.max(np.abs(follow_offset - initial_offset))),
    "replay_error": float(np.max(np.abs(replay_child - branch_child))),
    "fall_distance": float(released_z[2] - fallen_z[2]),
}))
"""
    )
    assert payload["follow_error"] < 1e-4
    assert payload["replay_error"] < 1e-4
    assert payload["fall_distance"] > 0.01
