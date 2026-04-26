"""Opt-in real Genesis smoke tests for fall timing and seeded replay behavior."""

from __future__ import annotations

from ._helpers import require_real_genesis_smoke, run_smoke_payload

import numpy as np
import pytest


@pytest.mark.genesis
@pytest.mark.slow
def test_real_genesis_falling_body_matches_lunar_gravity() -> None:
    """Verify contact-free vertical motion tracks the configured lunar gravity."""
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
cfg = replace(base, use_gpu=False, timestep=1.0 / 120.0, substeps=1, contact_iterations=4, random_seed=17)
engine = GenesisPhysicsEngine()
engine.configure(cfg, show_viewer=False)
engine.add_entity(
    "box",
    gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 2.0)),
    gs.materials.Rigid(rho=500),
    entity_type="rigid",
)
engine.build_scene(n_envs=1)
steps = 18
for _ in range(steps):
    engine.step(cfg.timestep, render=False)
pos, _ = engine.get_body_pose("box")
lin, _ = engine.get_body_velocity("box")
t = steps * cfg.timestep
g = cfg.gravity_vector[2]
expected_z = 2.0 + 0.5 * g * (t ** 2)
expected_vz = g * t
payload = {
    "z_error": float(abs(pos[2] - expected_z)),
    "vz_error": float(abs(lin[2] - expected_vz)),
}
print(json.dumps(payload))
"""
    )
    assert payload["z_error"] < 0.03
    assert payload["vz_error"] < 0.03


@pytest.mark.genesis
@pytest.mark.slow
def test_real_genesis_seeded_replay_is_stable_for_repeated_cpu_runs() -> None:
    """Verify same-seed CPU runs match and different seeds produce different traces."""
    require_real_genesis_smoke()
    free_payload = run_smoke_payload(
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

def run_case() -> tuple[np.ndarray, np.ndarray]:
    seed = 23
    cfg = replace(base, use_gpu=False, timestep=1.0 / 90.0, substeps=1, contact_iterations=4, random_seed=seed)
    rng = np.random.default_rng(seed)
    engine = GenesisPhysicsEngine()
    engine.configure(cfg, show_viewer=False)
    engine.add_entity(
        "box",
        gs.morphs.Box(size=(0.22, 0.22, 0.22), pos=(0.0, 0.0, 2.0)),
        gs.materials.Rigid(rho=550),
        entity_type="rigid",
    )
    engine.build_scene(n_envs=1)
    lin = np.array(
        [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.3, 1.0)],
        dtype=np.float32,
    )
    ang = rng.uniform(-0.8, 0.8, size=3).astype(np.float32)
    engine.set_body_velocity("box", lin, ang)
    poses = []
    velocities = []
    for _ in range(30):
        engine.step(cfg.timestep, render=False)
        pos, _ = engine.get_body_pose("box")
        vel, _ = engine.get_body_velocity("box")
        poses.append(pos.copy())
        velocities.append(vel.copy())
    return np.asarray(poses, dtype=np.float32), np.asarray(velocities, dtype=np.float32)

free_a_pos, free_a_vel = run_case()
free_b_pos, free_b_vel = run_case()

payload = {
    "free_pos_max": float(np.max(np.abs(free_a_pos - free_b_pos))),
    "free_vel_max": float(np.max(np.abs(free_a_vel - free_b_vel))),
    "free_final_pos": free_a_pos[-1].tolist(),
}
print(json.dumps(payload))
"""
    )
    contact_same_seed_payload = run_smoke_payload(
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

def run_case() -> tuple[np.ndarray, np.ndarray]:
    seed = 29
    cfg = replace(base, use_gpu=False, timestep=1.0 / 90.0, substeps=1, contact_iterations=4, random_seed=seed)
    rng = np.random.default_rng(seed)
    engine = GenesisPhysicsEngine()
    engine.configure(cfg, show_viewer=False)
    engine.add_entity("ground", gs.morphs.Plane(), gs.materials.Rigid(), entity_type="terrain")
    engine.add_entity(
        "box",
        gs.morphs.Box(size=(0.22, 0.22, 0.22), pos=(0.0, 0.0, 0.9)),
        gs.materials.Rigid(rho=550),
        entity_type="rigid",
    )
    engine.build_scene(n_envs=1)
    lin = np.array(
        [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.3, 1.0)],
        dtype=np.float32,
    )
    ang = rng.uniform(-0.8, 0.8, size=3).astype(np.float32)
    engine.set_body_velocity("box", lin, ang)
    poses = []
    velocities = []
    for _ in range(45):
        engine.step(cfg.timestep, render=False)
        pos, _ = engine.get_body_pose("box")
        vel, _ = engine.get_body_velocity("box")
        poses.append(pos.copy())
        velocities.append(vel.copy())
    return np.asarray(poses, dtype=np.float32), np.asarray(velocities, dtype=np.float32)

contact_a_pos, contact_a_vel = run_case()
contact_b_pos, contact_b_vel = run_case()

payload = {
    "contact_pos_max": float(np.max(np.abs(contact_a_pos - contact_b_pos))),
    "contact_vel_max": float(np.max(np.abs(contact_a_vel - contact_b_vel))),
    "contact_final_pos": contact_a_pos[-1].tolist(),
}
print(json.dumps(payload))
"""
    )
    contact_different_seed_payload = run_smoke_payload(
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
seed = 31
cfg = replace(base, use_gpu=False, timestep=1.0 / 90.0, substeps=1, contact_iterations=4, random_seed=seed)
rng = np.random.default_rng(seed)
engine = GenesisPhysicsEngine()
engine.configure(cfg, show_viewer=False)
engine.add_entity("ground", gs.morphs.Plane(), gs.materials.Rigid(), entity_type="terrain")
engine.add_entity(
    "box",
    gs.morphs.Box(size=(0.22, 0.22, 0.22), pos=(0.0, 0.0, 0.9)),
    gs.materials.Rigid(rho=550),
    entity_type="rigid",
)
engine.build_scene(n_envs=1)
lin = np.array(
    [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.3, 1.0)],
    dtype=np.float32,
)
ang = rng.uniform(-0.8, 0.8, size=3).astype(np.float32)
engine.set_body_velocity("box", lin, ang)
poses = []
for _ in range(45):
    engine.step(cfg.timestep, render=False)
    pos, _ = engine.get_body_pose("box")
    poses.append(pos.copy())
payload = {"contact_final_pos": np.asarray(poses, dtype=np.float32)[-1].tolist()}
print(json.dumps(payload))
"""
    )
    free_final_pos = np.asarray(free_payload["free_final_pos"], dtype=np.float32)
    contact_final_pos = np.asarray(contact_same_seed_payload["contact_final_pos"], dtype=np.float32)
    different_seed_contact_final_pos = np.asarray(
        contact_different_seed_payload["contact_final_pos"],
        dtype=np.float32,
    )
    assert free_payload["free_pos_max"] < 1e-5
    assert free_payload["free_vel_max"] < 1e-5
    assert contact_same_seed_payload["contact_pos_max"] < 1e-5
    assert contact_same_seed_payload["contact_vel_max"] < 1e-5
    assert np.max(np.abs(contact_final_pos - different_seed_contact_final_pos)) > 1e-3
    assert np.max(np.abs(free_final_pos - contact_final_pos)) > 1e-2
