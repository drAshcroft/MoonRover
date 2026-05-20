"""Pure-logic unit tests for the mission management layer.

Covers the concrete :class:`LunarMissionManager` state machine, planning and
fault handling, and the :class:`RelayCoordinator` cable-crossing / comms /
deadlock logic. No Genesis or physics dependency — fast unit lane.
"""
from __future__ import annotations

import numpy as np
import pytest

from moon_rover.mission import (
    FaultType,
    LunarMissionManager,
    MissionConfig,
    MissionPhase,
    RelayCoordinator,
    RoverStatus,
)


@pytest.fixture
def cfg() -> MissionConfig:
    return MissionConfig(
        grid_origin=np.array([10.0, 10.0, 0.0]),
        grid_rows=3,
        grid_cols=4,
        grid_spacing_m=5.0,
        num_rovers=2,
        max_retries=2,
    )


def test_initialize_builds_full_grid(cfg: MissionConfig) -> None:
    mm = LunarMissionManager()
    mm.initialize(cfg)
    grid = mm.get_grid_status()
    assert len(grid) == cfg.grid_rows * cfg.grid_cols
    assert all(gp.status == "unvisited" for gp in grid)


def test_plan_deployment_order_starts_nearest_depot(cfg: MissionConfig) -> None:
    mm = LunarMissionManager(depot_xyz=np.zeros(3))
    mm.initialize(cfg)
    order = mm.plan_deployment_order()
    assert len(order) == 12
    dists = [float(np.linalg.norm(gp.position_xyz)) for gp in mm.get_grid_status()]
    assert float(np.linalg.norm(order[0].position_xyz)) == pytest.approx(min(dists))


def test_assign_rovers_balances_and_partitions(cfg: MissionConfig) -> None:
    mm = LunarMissionManager()
    mm.initialize(cfg)
    asg = mm.assign_rovers(["r1", "r2"])
    assert sorted(len(v) for v in asg.values()) == [6, 6]
    # Every grid point is assigned exactly once.
    assigned = [gp for v in asg.values() for gp in v]
    assert len(assigned) == 12
    assert all(gp.assigned_rover in ("r1", "r2") for gp in assigned)


def test_full_segment_phase_cycle(cfg: MissionConfig) -> None:
    mm = LunarMissionManager()
    mm.initialize(cfg)
    mm.assign_rovers(["r1"])  # 12 points -> 12 segments
    seq = [mm.get_current_phase("r1")]
    for _ in range(7):
        seq.append(mm.advance_phase("r1"))
    assert seq == [
        MissionPhase.PLANNING,
        MissionPhase.DEPOT_PICKUP,
        MissionPhase.TRANSIT,
        MissionPhase.ANTENNA_DEPLOY,
        MissionPhase.CABLE_CONNECT,
        MissionPhase.RETURN_TO_BASE,
        MissionPhase.CHARGING,
        MissionPhase.DEPOT_PICKUP,  # loops to next segment
    ]
    assert mm.get_mission_progress() == pytest.approx(1 / 12)


def test_advance_phase_rejects_illegal_transition(cfg: MissionConfig) -> None:
    mm = LunarMissionManager()
    mm.initialize(cfg)
    mm.assign_rovers(["r1"])
    # Cannot leave PLANNING if the queue was emptied.
    mm._rovers["r1"].queue.clear()
    with pytest.raises(RuntimeError):
        mm.advance_phase("r1")


def test_detect_fault_priority_and_thresholds(cfg: MissionConfig) -> None:
    mm = LunarMissionManager()
    mm.initialize(cfg)
    mm.assign_rovers(["r1"])
    assert (
        mm.detect_fault("r1", {"battery_soc": 0.05, "time_s": 1.0})
        == FaultType.BATTERY_LOW
    )
    assert (
        mm.detect_fault("r1", {"cable_tension_n": 999.0, "time_s": 2.0})
        == FaultType.CABLE_SNAG
    )
    assert mm.detect_fault("r1", {"battery_soc": 0.9, "time_s": 3.0}) is None


def test_gps_outage_integrates_over_time(cfg: MissionConfig) -> None:
    mm = LunarMissionManager()
    mm.initialize(cfg)
    mm.assign_rovers(["r1"])
    f = None
    for t in range(0, 31, 5):
        f = mm.detect_fault("r1", {"gps_fix_quality": 0.05, "time_s": float(t)})
    assert f == FaultType.GPS_LOST


def test_recovery_retry_budget_then_exhausted(cfg: MissionConfig) -> None:
    mm = LunarMissionManager()
    mm.initialize(cfg)
    mm.assign_rovers(["r1"])
    # max_retries=2 -> two successful retries then failure.
    assert mm.execute_recovery("r1", FaultType.ROVER_STUCK) is True
    assert mm.execute_recovery("r1", FaultType.ROVER_STUCK) is True
    assert mm.execute_recovery("r1", FaultType.ROVER_STUCK) is False
    assert mm.get_current_phase("r1") == MissionPhase.FAULT_RECOVERY


def test_battery_low_recovery_diverts_to_base(cfg: MissionConfig) -> None:
    mm = LunarMissionManager()
    mm.initialize(cfg)
    mm.assign_rovers(["r1"])
    mm.advance_phase("r1")  # -> DEPOT_PICKUP
    mm.advance_phase("r1")  # -> TRANSIT
    assert mm.execute_recovery("r1", FaultType.BATTERY_LOW) is True
    resumed = mm.advance_phase("r1")
    assert resumed == MissionPhase.RETURN_TO_BASE


def test_cable_exclusion_zones_track_deployed_points(cfg: MissionConfig) -> None:
    mm = LunarMissionManager(depot_xyz=np.zeros(3))
    mm.initialize(cfg)
    assert mm.get_cable_exclusion_zones() == []
    mm.get_grid_status()[0].status = "complete"
    zones = mm.get_cable_exclusion_zones()
    assert len(zones) == 1 and zones[0].shape[1] == 3


def test_deadlock_detection_by_proximity(cfg: MissionConfig) -> None:
    mm = LunarMissionManager()
    mm.initialize(cfg)
    mm.assign_rovers(["r1", "r2"])
    pairs = mm.check_deadlock(
        {"r1": np.array([1.0, 1.0, 0.0]), "r2": np.array([1.5, 1.0, 0.0])}
    )
    assert pairs == [("r1", "r2")]
    far = mm.check_deadlock(
        {"r1": np.array([0.0, 0.0, 0.0]), "r2": np.array([50.0, 0.0, 0.0])}
    )
    assert far == []


# --------------------------------------------------------------------------- #
# RelayCoordinator
# --------------------------------------------------------------------------- #


def test_cable_crossing_detection() -> None:
    co = RelayCoordinator(depot_xyz=np.zeros(3))
    co.register_rover("r1", "diff_drive")
    co.register_rover("r2", "skid_steer")
    co.report_cable_segment("r1", np.array([20.0, 0.0, 0.0]))
    # Path that runs parallel and clear.
    assert co.check_cable_crossing(
        "r2", [np.array([0.0, 5.0, 0.0]), np.array([20.0, 5.0, 0.0])]
    )
    # Path that cuts across r1's corridor.
    assert not co.check_cable_crossing(
        "r2", [np.array([10.0, -5.0, 0.0]), np.array([10.0, 5.0, 0.0])]
    )
    # A rover may run along its own cable.
    assert co.check_cable_crossing(
        "r1", [np.array([0.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0])]
    )


def test_comm_delay_two_hops_vs_one() -> None:
    co = RelayCoordinator(depot_xyz=np.zeros(3))
    co.register_rover("r1", "skid_steer")
    co.register_rover("r2", "skid_steer")
    co.update_rover_status(
        RoverStatus("r1", np.array([10.0, 0.0, 0.0]), 0.0, "transit", 0.9, 50.0)
    )
    co.update_rover_status(
        RoverStatus("r2", np.array([20.0, 0.0, 0.0]), 0.0, "transit", 0.9, 50.0)
    )
    one_hop = co.get_communication_delay("base", "r1")
    two_hop = co.get_communication_delay("r1", "r2")
    assert two_hop > one_hop
    assert co.get_communication_delay("r1", "r1") == 0.0


def test_resolve_deadlock_assigns_priority() -> None:
    from moon_rover.mission.manager import GridPoint

    co = RelayCoordinator(depot_xyz=np.zeros(3))
    co.register_rover("r1", "skid_steer")
    co.register_rover("r2", "skid_steer")
    g1 = GridPoint(position_xyz=np.array([5.0, 0.0, 0.0]), row=0, col=0)
    g2 = GridPoint(position_xyz=np.array([30.0, 0.0, 0.0]), row=0, col=1)
    co.update_rover_status(
        RoverStatus("r1", np.array([4.0, 0.0, 0.0]), 0.0, "transit", 0.9, 50.0, g1)
    )
    co.update_rover_status(
        RoverStatus("r2", np.array([6.0, 0.0, 0.0]), 0.0, "transit", 0.9, 50.0, g2)
    )
    plan = co.resolve_deadlock("r1", "r2")
    assert plan["priority"] in ("r1", "r2")
    assert set(plan["instruction"].values()) <= {"wait", "reroute"}
