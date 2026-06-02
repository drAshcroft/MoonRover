"""Focused tests for the compose-scene-backed Genesis mission scenario."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from moon_rover.antenna.system import AntennaState
from moon_rover.core.physics.engine import AttachmentHandle, GenesisConfig, ScenePhase
from moon_rover.scenarios.missions import GenesisMissionScenario, genesis_mission_factory
from moon_rover.scenarios.runner import EpisodeResult, MissionScenarioRunner


_ANTENNA_CONFIG = SimpleNamespace(
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


def _physics_config() -> GenesisConfig:
    return GenesisConfig(gravity_vector=(0.0, 0.0, -1.622), timestep=0.1)


class _FakeContactEngine:
    def __init__(self) -> None:
        self.phase = ScenePhase.CONSTRUCTION
        self.sim_time = 0.0
        self.step_count = 0
        self.render_values: list[bool] = []
        self.attachment: AttachmentHandle | None = None
        self.detached = False
        self.torn_down = False
        self.poses = {
            "rover_001": (
                np.array([0.0, 0.0, 0.5], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
            "rover_001_antenna": (
                np.array([0.0, 0.0, 0.85], dtype=np.float32),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            ),
        }
        self.velocities = {
            name: (np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
            for name in self.poses
        }

    def configure(self, config: GenesisConfig, *, show_viewer: bool = False) -> None:
        self.config = config
        self.show_viewer = show_viewer

    def build_scene(self) -> None:
        self.phase = ScenePhase.SIMULATION

    def get_phase(self) -> ScenePhase:
        return self.phase

    def attach_bodies(self, parent: str, child: str) -> AttachmentHandle:
        assert parent == "rover_001"
        assert child == "rover_001_antenna"
        self.attachment = AttachmentHandle("attachment_0001")
        return self.attachment

    def detach_bodies(self, handle: AttachmentHandle) -> None:
        assert handle == self.attachment
        self.attachment = None
        self.detached = True

    def step(self, dt: float, *, render: bool = True) -> None:
        assert dt == pytest.approx(self.config.timestep)
        self.sim_time += dt
        self.step_count += 1
        self.render_values.append(render)
        if self.detached:
            antenna_pos, antenna_quat = self.poses["rover_001_antenna"]
            antenna_pos = antenna_pos.copy()
            antenna_pos[2] = 0.025
            self.poses["rover_001_antenna"] = (antenna_pos, antenna_quat)

    def set_body_pose(self, name: str, pos, quat) -> None:
        self.poses[name] = (
            np.asarray(pos, dtype=np.float32),
            np.asarray(quat, dtype=np.float32),
        )

    def get_body_pose(self, name: str):
        return self.poses[name]

    def set_body_velocity(self, name: str, lin_vel, ang_vel) -> None:
        self.velocities[name] = (
            np.asarray(lin_vel, dtype=np.float32),
            np.asarray(ang_vel, dtype=np.float32),
        )

    def get_body_velocity(self, name: str):
        return self.velocities[name]

    def get_terrain_height(self, x: float, y: float) -> float:
        return 0.0

    def get_sim_time(self) -> float:
        return self.sim_time

    def get_step_count(self) -> int:
        return self.step_count

    def teardown(self) -> None:
        self.phase = ScenePhase.TEARDOWN
        self.torn_down = True


class _FakeComposer:
    def __init__(self, engine: _FakeContactEngine) -> None:
        self.engine = engine
        self.compose_calls = 0

    def compose_scene(self, engine: _FakeContactEngine):
        assert engine is self.engine
        self.compose_calls += 1
        engine.build_scene()
        return SimpleNamespace(
            rovers=[SimpleNamespace(rover_id="rover_001")],
            antennas=[
                SimpleNamespace(
                    rover_id="rover_001",
                    antenna_config=_ANTENNA_CONFIG,
                )
            ],
        )


def _scenario_config(engine: _FakeContactEngine) -> dict:
    composer = _FakeComposer(engine)
    return {
        "engine_factory": lambda: engine,
        "composer_factory": lambda: composer,
        "physics_config": _physics_config(),
        "backend": "cpu",
        "target_position": [3.0, 4.0, 0.0],
        "carry_steps": 1,
        "release_settle_steps": 2,
    }


def test_genesis_mission_composes_releases_and_commissions_physical_antenna():
    engine = _FakeContactEngine()
    scenario = GenesisMissionScenario(_scenario_config(engine))
    scenario.setup(seed=7)

    while not scenario.is_complete():
        record = scenario.step()
    scenario.teardown()

    assert engine.detached is True
    assert engine.torn_down is True
    assert engine.step_count == 3
    assert engine.render_values == [False, False, False]
    assert scenario.antenna_states() == [AntennaState.ACTIVE]
    assert scenario.succeeded() is True
    assert record["timestamp"] == pytest.approx(0.3)
    assert record["rover_position"] == pytest.approx([0.0, 0.0, 0.5])
    assert scenario.antenna_placements[0]["antenna_id"] == "element_r00_c00"
    assert scenario.antenna_placements[0]["success"] is True
    assert [event["event_type"] for event in scenario.events] == [
        "mission_start",
        "antenna_picked",
        "antenna_released",
        "antenna_activated",
        "mission_end",
    ]


def test_genesis_mission_runs_through_existing_runner_contract():
    engine = _FakeContactEngine()
    config = _scenario_config(engine)
    runner = MissionScenarioRunner(scenario_factory=genesis_mission_factory)

    result = runner.run_episode(config, seed=3)

    assert isinstance(result, EpisodeResult)
    assert result.success is True
    assert result.steps == 3
    assert result.metrics.antennas_deployed == 1
    assert result.metrics.antennas_failed == 0
    assert result.metrics.cable_coverage_percent == pytest.approx(100.0)
    assert result.log_data["rover_position"].shape == (3, 3)


def test_genesis_mission_setup_failure_tears_down_engine():
    engine = _FakeContactEngine()

    class _EmptyComposer:
        def compose_scene(self, engine: _FakeContactEngine):
            engine.build_scene()
            return SimpleNamespace(rovers=[], antennas=[])

    with pytest.raises(RuntimeError, match="no rover"):
        GenesisMissionScenario(
            {
                "engine_factory": lambda: engine,
                "composer_factory": _EmptyComposer,
                "physics_config": _physics_config(),
            }
        ).setup(seed=0)

    assert engine.torn_down is True
