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

    # -- camera / video capture -------------------------------------------
    def add_camera(self, name, resolution, pos, lookat, fov):
        self.camera = {
            "name": name,
            "resolution": resolution,
            "pos": pos,
            "lookat": lookat,
            "fov": fov,
        }
        self.camera_poses: list = []
        self.camera_frames = 0
        self.camera_started = False
        self.camera_saved: dict | None = None
        return object()

    def set_camera_pose(self, name, pos=None, lookat=None):
        self.camera_poses.append((pos, lookat))

    def start_camera_recording(self, name):
        self.camera_started = True

    def render_camera(self, name):
        self.camera_frames += 1

    def stop_camera_recording(self, name, save_path, fps):
        from pathlib import Path as _Path

        self.camera_saved = {"save_path": save_path, "fps": fps}
        _Path(save_path).write_bytes(b"FAKEMP4")


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


# Surveyed position of the first array element (element_r00_c00) in
# configs/mission.yaml: grid origin [50, 50, 0]. Placing the antenna here lets
# the array-built-to-spec success metric pass.
_DESIGN_TARGET = [50.0, 50.0, 0.0]


def _scenario_config(engine: _FakeContactEngine) -> dict:
    composer = _FakeComposer(engine)
    return {
        "engine_factory": lambda: engine,
        "composer_factory": lambda: composer,
        "physics_config": _physics_config(),
        "backend": "cpu",
        "target_position": list(_DESIGN_TARGET),
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


def test_genesis_mission_success_requires_placement_within_design_tolerance():
    engine = _FakeContactEngine()
    config = _scenario_config(engine)
    # Aim 1 m off the surveyed element target — beyond the 0.25 m design
    # placement tolerance, but still within the 0.5 m connector-engage band so
    # the unit reaches ACTIVE. The array-to-spec gate must still fail the run.
    config["target_position"] = [_DESIGN_TARGET[0] + 1.0, _DESIGN_TARGET[1], 0.0]
    scenario = GenesisMissionScenario(config)
    scenario.setup(seed=1)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()

    assert scenario.antenna_states() == [AntennaState.ACTIVE]
    report = scenario.array_quality_report()
    assert report is not None
    assert report.elements_placed == 1
    assert report.elements_within_tolerance == 0
    assert report.max_position_error_m == pytest.approx(1.0, abs=0.05)
    assert scenario.succeeded() is False


def test_genesis_mission_array_quality_report_scores_to_spec_build():
    engine = _FakeContactEngine()
    scenario = GenesisMissionScenario(_scenario_config(engine))
    scenario.setup(seed=2)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()

    report = scenario.array_quality_report()
    assert report is not None
    assert report.array_id == "lunar_interferometer_001"
    assert report.elements_designed == 20  # 4x5 grid
    assert report.elements_placed == 1
    assert report.elements_within_tolerance == 1
    assert report.completion_fraction == pytest.approx(1.0 / 20.0)
    assert scenario.succeeded() is True


def test_genesis_mission_completion_floor_can_demand_full_array():
    engine = _FakeContactEngine()
    config = _scenario_config(engine)
    # Demand the whole array be built: a single-element slice can no longer pass.
    config["success_min_completion_fraction"] = 1.0
    scenario = GenesisMissionScenario(config)
    scenario.setup(seed=4)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()

    assert scenario.antenna_states() == [AntennaState.ACTIVE]
    assert scenario.succeeded() is False


def test_genesis_mission_array_report_none_before_setup():
    scenario = GenesisMissionScenario(_scenario_config(_FakeContactEngine()))
    assert scenario.array_quality_report() is None


def test_genesis_mission_rejects_out_of_range_completion_floor():
    with pytest.raises(ValueError, match="success_min_completion_fraction"):
        GenesisMissionScenario({"success_min_completion_fraction": 1.5})


def test_genesis_mission_records_tracking_video(tmp_path):
    from moon_rover.visualization.video_export import RecorderConfig

    out = tmp_path / "demos" / "mission.mp4"
    engine = _FakeContactEngine()
    config = _scenario_config(engine)
    config["record_video"] = RecorderConfig(
        output_path=out, track_body="@rover", fps=30
    )
    scenario = GenesisMissionScenario(config)
    scenario.setup(seed=7)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()

    # Camera was added during construction and recording ran end to end.
    assert engine.camera["name"] == "demo_camera"
    assert engine.camera_started is True
    assert engine.camera_frames == engine.step_count  # cadence 1 at dt=0.1, 30 fps
    assert engine.camera_frames > 0
    # The "@rover" sentinel resolved to the bound rover entity.
    assert engine.camera_poses, "tracking shot should reposition the camera"
    # Video was written and surfaced on the scenario.
    assert engine.camera_saved == {"save_path": str(out), "fps": 30}
    assert scenario.video_path == out
    assert out.exists()


def test_genesis_mission_path_string_records_rover_tracking(tmp_path):
    out = tmp_path / "quick.mp4"
    engine = _FakeContactEngine()
    config = _scenario_config(engine)
    config["record_video"] = str(out)
    scenario = GenesisMissionScenario(config)
    scenario.setup(seed=7)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()

    assert scenario.video_path == out
    assert out.exists()
    # Path-string form defaults to a rover-tracking shot (camera repositioned).
    assert engine.camera_poses


def test_genesis_mission_no_recording_by_default():
    engine = _FakeContactEngine()
    scenario = GenesisMissionScenario(_scenario_config(engine))
    scenario.setup(seed=7)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()
    assert scenario.video_path is None
    assert not hasattr(engine, "camera")


def test_genesis_mission_rejects_bad_record_video_type():
    with pytest.raises(TypeError, match="record_video"):
        GenesisMissionScenario({"record_video": 123})


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
