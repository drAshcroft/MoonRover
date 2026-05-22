"""End-to-end integration tests for the full antenna-placement mission.

Runs the complete scenario through the production ScenarioRunner harness:
spawn rover at moonbase -> pick up antenna -> navigate to grid point -> place,
deploy, and activate the antenna (real DeployableAntennaUnit state machine) ->
return to base. Asserts every antenna reaches AntennaState.ACTIVE and that the
mission KPIs land within tolerance.

These exercise multiple subsystems together (ScenarioRunner + antenna lifecycle
+ MultiStreamLogger + MissionMetricsAnalyzer), so they are marked ``slow``.
The ``genesis``-marked test guards the real-physics variant, which is skipped
until a Genesis-backed MissionPlacementScenario is wired.
"""

from __future__ import annotations

import pytest

from moon_rover.antenna.system import AntennaState
from moon_rover.scenarios.missions import (
    MissionPlacementScenario,
    mission_placement_factory,
)
from moon_rover.scenarios.runner import MissionScenarioRunner

# Standard 3-point test grid (an L-shaped sweep from the moonbase at origin).
_GRID = [[5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0]]


def _runner() -> MissionScenarioRunner:
    return MissionScenarioRunner(scenario_factory=mission_placement_factory)


@pytest.mark.slow
def test_full_mission_all_antennas_active():
    runner = _runner()
    ep = runner.run_episode({"grid_points": _GRID}, seed=1)

    assert ep.success is True
    assert ep.metrics.antennas_deployed == 3
    assert ep.metrics.antennas_failed == 0
    assert ep.metrics.success_rate() == pytest.approx(1.0)
    assert ep.metrics.cable_coverage_percent == pytest.approx(100.0)


@pytest.mark.slow
def test_full_mission_state_machine_reaches_active():
    """Drive the scenario directly to assert the real antenna lifecycle."""
    scenario = MissionPlacementScenario({"grid_points": _GRID})
    scenario.setup(seed=1)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()

    states = scenario.antenna_states()
    assert len(states) == 3
    assert all(s == AntennaState.ACTIVE for s in states)
    assert scenario.returned_to_base is True
    # Every antenna can serve as a beacon once ACTIVE.
    assert all(a.get_beacon_config() is not None for a in scenario.antenna_units)


@pytest.mark.slow
def test_full_mission_kpis_within_tolerance():
    runner = _runner()
    ep = runner.run_episode({"grid_points": _GRID, "drive_speed_mps": 1.0}, seed=1)
    m = ep.metrics

    # Path: base->(5,0)->(5,5)->(0,5)->base = ~20 m, minus arrival tolerances.
    assert 18.0 <= m.total_distance_m <= 21.0
    # Localization estimator is lightly noisy; drift should stay sub-decimeter.
    assert m.localization_error_drift_m < 0.2
    # Energy is strictly positive and finite.
    assert m.energy_consumed_wh > 0.0
    assert m.mission_time_s > 0.0
    # Energy per placed antenna is the total split across 3 successes.
    assert m.energy_per_antenna_wh() == pytest.approx(m.energy_consumed_wh / 3.0)


@pytest.mark.slow
def test_mission_deterministic_for_same_seed():
    runner = _runner()
    a = runner.run_episode({"grid_points": _GRID}, seed=7)
    b = runner.run_episode({"grid_points": _GRID}, seed=7)
    assert a.steps == b.steps
    assert a.metrics.total_distance_m == pytest.approx(b.metrics.total_distance_m)
    assert a.metrics.localization_error_drift_m == pytest.approx(
        b.metrics.localization_error_drift_m
    )


@pytest.mark.slow
def test_mission_failure_path_one_antenna_not_active():
    runner = _runner()
    ep = runner.run_episode({"grid_points": _GRID, "fail_indices": [1]}, seed=1)

    assert ep.success is False
    assert ep.metrics.antennas_deployed == 2
    assert ep.metrics.antennas_failed == 1
    assert "deployment_failed" in ep.metrics.failure_modes

    # Confirm at the state-machine level: index 1 never reaches ACTIVE.
    scenario = MissionPlacementScenario({"grid_points": _GRID, "fail_indices": [1]})
    scenario.setup(seed=1)
    while not scenario.is_complete():
        scenario.step()
    scenario.teardown()
    states = scenario.antenna_states()
    assert states[0] == AntennaState.ACTIVE
    assert states[1] != AntennaState.ACTIVE
    assert states[2] == AntennaState.ACTIVE


@pytest.mark.slow
def test_mission_streams_telemetry_to_logs(tmp_path):
    runner = _runner()
    ep = runner.run_episode({"grid_points": _GRID}, seed=1, log_dir=tmp_path)
    assert (tmp_path / "log.mcap").exists()
    assert (tmp_path / "log.h5").exists()

    # The MCAP should carry the activation events.
    from mcap.reader import make_reader

    activation_events = 0
    with open(tmp_path / "log.mcap", "rb") as f:
        import json

        for _schema, channel, msg in make_reader(f).iter_messages():
            if channel.topic == "/events":
                payload = json.loads(msg.data.decode("utf-8"))
                if payload.get("event_type") == "antenna_activated":
                    activation_events += 1
    assert activation_events == 3


@pytest.mark.slow
def test_run_single_reports_mission_metrics():
    runner = _runner()
    result = runner.run_single(seed=1, config={"grid_points": _GRID})
    assert result.success is True
    assert result.metrics["placement_success_rate"] == pytest.approx(1.0)
    assert result.metrics["cable_coverage_percent"] == pytest.approx(100.0)
    assert result.rover_type == "mission_placement_diff_drive"


@pytest.mark.genesis
@pytest.mark.slow
def test_full_mission_real_genesis_physics():
    """Real-physics variant: run the mission on a Genesis-backed scene.

    Skipped until a Genesis-backed MissionPlacementScenario exists (it would
    spawn the URDF rover + antenna entities and step real contact physics).
    The deterministic mission above validates the orchestration/lifecycle; this
    is the hook for the physics-in-the-loop variant.
    """
    pytest.skip(
        "Genesis-backed MissionPlacementScenario not yet wired; "
        "deterministic mission tests provide end-to-end coverage in the meantime."
    )
