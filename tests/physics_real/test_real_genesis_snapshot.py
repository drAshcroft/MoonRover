"""Opt-in real Genesis smoke tests for snapshot determinism."""

from __future__ import annotations

from ._helpers import require_real_genesis_smoke, run_smoke_payload

import pytest


@pytest.mark.genesis
@pytest.mark.slow
def test_real_genesis_snapshot_restore_is_exact_time() -> None:
    """Verify restore_snapshot() preserves exact state and replay trajectory."""
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
engine.add_entity(
    "box",
    gs.morphs.Box(size=(0.25, 0.25, 0.25), pos=(0.0, 0.0, 1.0)),
    gs.materials.Rigid(rho=600),
    entity_type="rigid",
)
engine.build_scene(n_envs=1)
engine.set_body_velocity(
    "box",
    np.array([0.8, -0.35, 2.6], dtype=np.float32),
    np.array([0.4, -0.2, 0.3], dtype=np.float32),
)
for _ in range(3):
    engine.step(cfg.timestep, render=False)

snap = engine.save_snapshot()
saved_pos, saved_quat = engine.get_body_pose("box")
saved_lin, saved_ang = engine.get_body_velocity("box")
saved_time = engine.get_sim_time()
saved_step = engine.get_step_count()

engine.step(cfg.timestep, render=False)
branch_pos, branch_quat = engine.get_body_pose("box")
branch_lin, branch_ang = engine.get_body_velocity("box")

for _ in range(2):
    engine.step(cfg.timestep, render=False)

engine.restore_snapshot(snap)
restored_pos, restored_quat = engine.get_body_pose("box")
restored_lin, restored_ang = engine.get_body_velocity("box")
restored_time = engine.get_sim_time()
restored_step = engine.get_step_count()

engine.step(cfg.timestep, render=False)
replay_pos, replay_quat = engine.get_body_pose("box")
replay_lin, replay_ang = engine.get_body_velocity("box")

payload = {
    "restore_pos_max": float(np.max(np.abs(restored_pos - saved_pos))),
    "restore_quat_max": float(np.max(np.abs(restored_quat - saved_quat))),
    "restore_lin_max": float(np.max(np.abs(restored_lin - saved_lin))),
    "restore_ang_max": float(np.max(np.abs(restored_ang - saved_ang))),
    "replay_pos_max": float(np.max(np.abs(replay_pos - branch_pos))),
    "replay_quat_max": float(np.max(np.abs(replay_quat - branch_quat))),
    "replay_lin_max": float(np.max(np.abs(replay_lin - branch_lin))),
    "replay_ang_max": float(np.max(np.abs(replay_ang - branch_ang))),
    "time_error": float(abs(restored_time - saved_time)),
    "step_error": int(abs(restored_step - saved_step)),
}
print(json.dumps(payload))
"""
    )
    assert payload["restore_pos_max"] < 1e-4
    assert payload["restore_quat_max"] < 1e-4
    assert payload["restore_lin_max"] < 1e-4
    assert payload["restore_ang_max"] < 1e-4
    assert payload["replay_pos_max"] < 1e-4
    assert payload["replay_quat_max"] < 1e-4
    assert payload["replay_lin_max"] < 1e-4
    assert payload["replay_ang_max"] < 1e-4
    assert payload["time_error"] < 1e-9
    assert payload["step_error"] == 0
