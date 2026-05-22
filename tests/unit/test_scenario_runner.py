"""Unit tests for src/moon_rover/scenarios/runner.py.

Covers:
- run_episode end-to-end with the synthetic scenario: metrics populated,
  step count, success flag, log_data shape.
- run_episode with logging: MCAP + HDF5 artifacts written.
- max_steps cap halts the loop.
- SyntheticDriveScenario determinism across seeds; fault injection.
- Custom Scenario injection via scenario_factory.
- Experiment framework: load_experiment (+errors), run_single -> RunResult,
  run_monte_carlo cartesian product * seeds, check_convergence, generate_report.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from moon_rover.scenarios.runner import (
    EpisodeResult,
    ExperimentConfig,
    MissionScenarioRunner,
    RunResult,
    Scenario,
    SyntheticDriveScenario,
)


# ---------------------------------------------------------------------------
# run_episode
# ---------------------------------------------------------------------------


def test_run_episode_basic():
    runner = MissionScenarioRunner()
    ep = runner.run_episode({"duration_s": 4.0, "dt": 0.02}, seed=7)
    assert isinstance(ep, EpisodeResult)
    assert ep.steps == 200
    assert ep.duration_s == pytest.approx(4.0, abs=0.05)
    assert ep.metrics.total_distance_m > 0
    assert ep.metrics.mission_time_s == pytest.approx(4.0, abs=0.05)
    assert "rover_position" in ep.log_data
    assert ep.log_data["rover_position"].shape == (200, 3)


def test_run_episode_deterministic_for_same_seed():
    runner = MissionScenarioRunner()
    a = runner.run_episode({"duration_s": 3.0}, seed=42)
    b = runner.run_episode({"duration_s": 3.0}, seed=42)
    np.testing.assert_allclose(
        a.log_data["rover_position"], b.log_data["rover_position"]
    )
    assert a.metrics.localization_error_drift_m == pytest.approx(
        b.metrics.localization_error_drift_m
    )


def test_run_episode_max_steps_cap():
    runner = MissionScenarioRunner()
    ep = runner.run_episode({"duration_s": 8.0, "dt": 0.02}, seed=1, max_steps=50)
    assert ep.steps == 50
    # mission did not complete within the cap (8 s would need 400 steps)
    assert ep.metrics.mission_time_s == pytest.approx(50 * 0.02, abs=0.05)


def test_run_episode_writes_logs(tmp_path):
    runner = MissionScenarioRunner()
    ep = runner.run_episode({"duration_s": 2.0}, seed=3, log_dir=tmp_path)
    assert ep.log_dir == tmp_path
    assert (tmp_path / "log.mcap").exists()
    assert (tmp_path / "log.h5").exists()


def test_run_episode_success_requires_accurate_placements():
    runner = MissionScenarioRunner()
    # Zero noise -> all placements accurate -> success.
    good = runner.run_episode({"duration_s": 8.0, "placement_noise_m": 0.0}, seed=1)
    assert good.success is True
    # Huge noise -> placements inaccurate -> failure.
    bad = runner.run_episode({"duration_s": 8.0, "placement_noise_m": 5.0}, seed=1)
    assert bad.success is False
    assert bad.metrics.antennas_failed >= 1


# ---------------------------------------------------------------------------
# SyntheticDriveScenario
# ---------------------------------------------------------------------------


def test_synthetic_scenario_fault_injection():
    scenario = SyntheticDriveScenario({"duration_s": 4.0, "fault_at_s": 2.0, "fault_mode": "stall"})
    scenario.setup(seed=0)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()
    assert len(scenario.faults) == 1
    assert scenario.faults[0]["mode"] == "stall"


def test_synthetic_scenario_emits_lifecycle_events():
    scenario = SyntheticDriveScenario({"duration_s": 2.0})
    scenario.setup(seed=0)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()
    types = [e["event_type"] for e in scenario.events]
    assert types[0] == "mission_start"
    assert types[-1] == "mission_end"
    assert "antenna_placed" in types


def test_synthetic_scenario_step_before_setup_raises():
    scenario = SyntheticDriveScenario()
    with pytest.raises(AssertionError):
        scenario.step()


# ---------------------------------------------------------------------------
# Custom Scenario injection
# ---------------------------------------------------------------------------


class _TwoStepScenario(Scenario):
    rover_type = "test_bot"

    def __init__(self, config):
        self.config = config
        self._i = 0

    def setup(self, seed, *, visualize=False):
        self._i = 0
        self._seed = seed

    def step(self):
        self._i += 1
        return {
            "timestamp": self._i * 0.1,
            "rover_position": [float(self._i), 0.0, 0.0],
            "power_consumed_w": 10.0,
        }

    def is_complete(self):
        return self._i >= 2

    def teardown(self):
        pass

    def succeeded(self):
        return True


def test_custom_scenario_injection():
    runner = MissionScenarioRunner(scenario_factory=lambda cfg: _TwoStepScenario(cfg))
    ep = runner.run_episode({}, seed=5)
    assert ep.steps == 2
    assert ep.success is True
    assert ep.log_data["rover_position"].shape == (2, 3)


def test_scenario_missing_required_keys_raises():
    class _BadScenario(Scenario):
        def __init__(self):
            self._done = False

        def setup(self, seed, *, visualize=False):
            self._done = False

        def step(self):
            return {"timestamp": 0.1}  # missing rover_position

        def is_complete(self):
            done = self._done
            self._done = True  # complete after one step so the loop runs once
            return done

        def teardown(self):
            pass

    runner = MissionScenarioRunner(scenario_factory=lambda cfg: _BadScenario())
    with pytest.raises(ValueError, match="rover_position"):
        runner.run_episode({}, seed=0)


# ---------------------------------------------------------------------------
# Experiment framework
# ---------------------------------------------------------------------------


def _write_experiment_yaml(tmp_path: Path) -> Path:
    content = """
name: antenna_study
variable_params:
  cruise_speed_mps: [0.6, 1.0]
  placement_noise_m: [0.1, 0.3]
fixed_params:
  duration_s: 4.0
num_seeds: 2
success_metrics:
  - total_distance_m
  - placement_success_rate
convergence_threshold: 0.1
"""
    p = tmp_path / "experiment.yaml"
    p.write_text(content)
    return p


def test_load_experiment(tmp_path):
    runner = MissionScenarioRunner()
    exp = runner.load_experiment(str(_write_experiment_yaml(tmp_path)))
    assert exp.name == "antenna_study"
    assert exp.num_seeds == 2
    assert exp.variable_params["cruise_speed_mps"] == [0.6, 1.0]
    assert exp.fixed_params["duration_s"] == 4.0
    assert exp.convergence_threshold == pytest.approx(0.1)


def test_load_experiment_missing_file():
    runner = MissionScenarioRunner()
    with pytest.raises(FileNotFoundError):
        runner.load_experiment("does_not_exist.yaml")


def test_load_experiment_missing_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nvariable_params: {}\n")  # missing num_seeds, success_metrics
    runner = MissionScenarioRunner()
    with pytest.raises(ValueError, match="missing required key"):
        runner.load_experiment(str(p))


def test_run_single_returns_run_result():
    runner = MissionScenarioRunner()
    result = runner.run_single(seed=2, config={"duration_s": 3.0, "rover_type": "diff_drive"})
    assert isinstance(result, RunResult)
    assert result.seed == 2
    assert result.rover_type == "diff_drive"
    assert "total_distance_m" in result.metrics
    assert result.duration_s == pytest.approx(3.0, abs=0.05)


def test_run_monte_carlo_cartesian_product(tmp_path):
    runner = MissionScenarioRunner()
    exp = ExperimentConfig(
        name="mc",
        variable_params={"cruise_speed_mps": [0.5, 1.0], "placement_noise_m": [0.1, 0.4]},
        fixed_params={"duration_s": 2.0},
        num_seeds=3,
        success_metrics=["total_distance_m"],
    )
    results = runner.run_monte_carlo(exp)
    # 2 * 2 combos * 3 seeds = 12 runs.
    assert len(results) == 12
    assert all(isinstance(r, RunResult) for r in results)


def test_check_convergence_true_for_stable_metric():
    runner = MissionScenarioRunner()
    results = [
        RunResult(seed=i, rover_type="x", metrics={"m": 10.0 + 0.01 * i},
                  faults=[], success=True, duration_s=1.0)
        for i in range(10)
    ]
    assert runner.check_convergence(results, "m") is True


def test_check_convergence_false_for_drifting_metric():
    runner = MissionScenarioRunner()
    results = [
        RunResult(seed=i, rover_type="x", metrics={"m": float(i * i)},
                  faults=[], success=True, duration_s=1.0)
        for i in range(10)
    ]
    assert runner.check_convergence(results, "m") is False


def test_check_convergence_too_few_samples_raises():
    runner = MissionScenarioRunner()
    results = [RunResult(seed=0, rover_type="x", metrics={"m": 1.0}, faults=[], success=True, duration_s=1.0)]
    with pytest.raises(ValueError, match="too few samples"):
        runner.check_convergence(results, "m")


def test_generate_report_structure():
    runner = MissionScenarioRunner()
    results = [
        RunResult(seed=0, rover_type="x", metrics={"total_distance_m": 10.0},
                  faults=["stall"], success=True, duration_s=1.0),
        RunResult(seed=1, rover_type="x", metrics={"total_distance_m": 20.0},
                  faults=["stall", "snag"], success=False, duration_s=1.0),
    ]
    report = runner.generate_report(results)
    assert report["n_runs"] == 2
    assert report["success_rate"] == pytest.approx(0.5)
    assert report["failure_modes"] == {"stall": 2, "snag": 1}
    summary = report["metric_summaries"]["total_distance_m"]
    assert summary["mean"] == pytest.approx(15.0)
    assert summary["min"] == 10.0
    assert summary["max"] == 20.0
    assert "p95" in summary


def test_generate_report_empty():
    runner = MissionScenarioRunner()
    report = runner.generate_report([])
    assert report["n_runs"] == 0
    assert report["success_rate"] == 0.0
    assert report["metric_summaries"] == {}
