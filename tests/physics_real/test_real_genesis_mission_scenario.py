"""Opt-in real Genesis smoke test for the physical mission scenario."""

from __future__ import annotations

from pathlib import Path

import pytest

from ._helpers import require_real_genesis_smoke, run_smoke_payload


def run_real_genesis_mission_smoke(log_dir: Path) -> dict[str, object]:
    """Run the physical mission in an isolated Genesis subprocess."""
    return run_smoke_payload(
        rf"""
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import genesis as gs

sys.path.insert(0, "src")

from moon_rover.antenna.system import AntennaConfig
from moon_rover.core.physics.engine import GenesisConfig
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine
from moon_rover.scenarios.missions import GenesisMissionScenario
from moon_rover.scenarios.runner import MissionScenarioRunner

antenna_config = AntennaConfig(
    base_plate_m=(0.4, 0.4, 0.05),
    base_mass_kg=2.5,
    mast_height_m=1.2,
    mast_radius_m=0.02,
    mast_mass_kg=1.0,
    dish_diameter_m=0.6,
    dish_mass_kg=0.8,
    connector_mass_kg=0.2,
    total_mass_kg=4.5,
)

class SmokeComposer:
    def compose_scene(self, engine):
        engine.add_entity(
            "terrain",
            gs.morphs.Plane(),
            gs.materials.Rigid(),
            entity_type="terrain",
        )
        engine.add_entity(
            "rover_001",
            gs.morphs.Box(size=(0.8, 0.5, 0.3), pos=(0.0, 0.0, 0.8)),
            gs.materials.Rigid(rho=600),
        )
        engine.add_entity(
            "rover_001_antenna",
            gs.morphs.Box(size=(0.4, 0.4, 0.05), pos=(0.0, 0.0, 1.15)),
            gs.materials.Rigid(rho=400),
        )
        engine.build_scene()
        return SimpleNamespace(
            rovers=[SimpleNamespace(rover_id="rover_001")],
            antennas=[SimpleNamespace(rover_id="rover_001", antenna_config=antenna_config)],
        )

base = GenesisConfig.from_yaml("configs/physics.yaml")
physics = replace(
    base,
    use_gpu=False,
    timestep=1.0 / 60.0,
    substeps=1,
    contact_iterations=4,
)
log_dir = Path({str(log_dir)!r})
scenarios = []

def scenario_factory(config):
    scenario = GenesisMissionScenario(config)
    scenarios.append(scenario)
    return scenario

runner = MissionScenarioRunner(scenario_factory=scenario_factory)
episode = runner.run_episode(
    {{
        "engine_factory": GenesisPhysicsEngine,
        "composer_factory": SmokeComposer,
        "physics_config": physics,
        "target_position": [2.0, 0.0, 0.0],
        "carry_steps": 1,
        "release_settle_steps": 20,
    }},
    seed=5,
    log_dir=log_dir,
)
print(json.dumps({{
    "success": episode.success,
    "steps": episode.steps,
    "antennas_deployed": episode.metrics.antennas_deployed,
    "antenna_state": scenarios[0].antenna_states()[0].value,
    "mcap_exists": (log_dir / "log.mcap").exists(),
    "hdf5_exists": (log_dir / "log.h5").exists(),
}}))
"""
    )


@pytest.mark.genesis
@pytest.mark.slow
def test_real_genesis_mission_scenario_returns_episode_and_logs(tmp_path) -> None:
    require_real_genesis_smoke()
    payload = run_real_genesis_mission_smoke(tmp_path)
    assert payload == {
        "success": True,
        "steps": 21,
        "antennas_deployed": 1,
        "antenna_state": "active",
        "mcap_exists": True,
        "hdf5_exists": True,
    }
