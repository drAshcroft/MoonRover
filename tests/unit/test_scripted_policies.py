"""Unit tests for the scripted baseline policies (placement, cable, arm).

Each policy implements PolicyInterface (via BasePolicy). Tests assert the
deterministic control heuristics behave as specified, and a Monte Carlo test
confirms the scripted baseline clears the >=70% mission success-rate bar on the
standard antenna-placement scenario.
"""

from __future__ import annotations

import numpy as np
import pytest

from moon_rover.rl.arm.policy import CarryStablePolicy, PickupPolicy
from moon_rover.rl.cable_deploy.policy import CableDeploymentPolicy
from moon_rover.rl.common.policy_interface import PolicyMode
from moon_rover.rl.placement.policy import AntennaPlacementPolicy
from moon_rover.scenarios.missions import mission_placement_factory
from moon_rover.scenarios.runner import MissionScenarioRunner


# ---------------------------------------------------------------------------
# Placement policy
# ---------------------------------------------------------------------------


def test_placement_descends_when_above_surface():
    policy = AntennaPlacementPolicy()
    policy.observe(
        {"height_above_surface": [0.5], "ft_force": [0, 0, 0], "surface_normal": [0, 0, 1]}
    )
    policy.act()
    assert policy.last_vertical_rate < 0  # descending


def test_placement_pushes_down_when_under_force():
    policy = AntennaPlacementPolicy(target_force_n=120.0)
    policy.observe(
        {"height_above_surface": [0.0], "ft_force": [0, 0, 0.0], "surface_normal": [0, 0, 1]}
    )
    policy.act()
    assert policy.last_vertical_rate < 0  # under-force -> keep pushing down


def test_placement_retracts_when_over_force():
    policy = AntennaPlacementPolicy(target_force_n=120.0)
    policy.observe(
        {"height_above_surface": [0.0], "ft_force": [0, 0, 400.0], "surface_normal": [0, 0, 1]}
    )
    policy.act()
    assert policy.last_vertical_rate > 0  # over-force -> retract


def test_placement_action_shape_and_bounds():
    policy = AntennaPlacementPolicy()
    policy.observe({"height_above_surface": [0.3], "ft_force": [0, 0, 0], "surface_normal": [0.1, 0, 1]})
    action = policy.act()["joint_velocity_targets"]
    assert action.shape == (4,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert policy.get_mode() == PolicyMode.SCRIPTED
    assert policy.get_confidence() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cable deployment policy
# ---------------------------------------------------------------------------


def test_cable_brakes_on_high_tension():
    policy = CableDeploymentPolicy(high_tension_n=400.0)
    policy.observe({"tension": [500], "rover_speed": [0.3], "rock_proximity": [1, 1, 1, 1]})
    feed = policy.act()["spool_feed_modifier"][0]
    assert feed == pytest.approx(-1.0)


def test_cable_pays_out_with_rover_motion():
    policy = CableDeploymentPolicy(max_feed_mps=0.5)
    policy.observe({"tension": [200], "rover_speed": [0.25], "rock_proximity": [5, 5, 5, 5]})
    feed = policy.act()["spool_feed_modifier"][0]
    assert feed == pytest.approx(0.5, abs=1e-5)  # 0.25 / 0.5


def test_cable_slows_near_obstacle():
    policy = CableDeploymentPolicy(snag_clearance_m=0.3)
    policy.observe({"tension": [150], "rover_speed": [0.4], "rock_proximity": [0.1, 2, 2, 2]})
    feed = policy.act()["spool_feed_modifier"][0]
    assert feed <= 0.0


# ---------------------------------------------------------------------------
# Arm policies
# ---------------------------------------------------------------------------


def test_pickup_keeps_gripper_open_when_far():
    policy = PickupPolicy(grasp_tolerance_m=0.05)
    policy.observe({"antenna_relative_pose": [0.3, 0.0, 0.0, 0, 0, 0]})
    action = policy.act()
    assert action["gripper_command"][0] == pytest.approx(0.0)
    assert not policy.grasp_closed
    # Some reaching motion is commanded.
    assert np.linalg.norm(action["joint_velocity_targets"]) > 0


def test_pickup_latches_grasp_when_close():
    policy = PickupPolicy(grasp_tolerance_m=0.05)
    policy.observe({"antenna_relative_pose": [0.01, 0.0, 0.0, 0, 0, 0]})
    action = policy.act()
    assert action["gripper_command"][0] == pytest.approx(1.0)
    assert policy.grasp_closed
    # Grasp latches: stays closed even if the antenna later appears far.
    policy.observe({"antenna_relative_pose": [1.0, 0.0, 0.0, 0, 0, 0]})
    assert policy.act()["gripper_command"][0] == pytest.approx(1.0)
    policy.reset()
    assert not policy.grasp_closed


def test_carry_stable_damps_motion_within_bounds():
    policy = CarryStablePolicy(max_joint_rate=0.05)
    policy.observe({"joint_vel": [0.2, -0.3, 0.1, 0.0], "chassis_gyro": [0.0, 0.0, 0.0]})
    cmd = policy.act()["joint_velocity_targets"]
    assert cmd.shape == (4,)
    assert np.all(np.abs(cmd) <= 0.05 + 1e-6)
    # Opposes the residual motion (signs flipped where motion is nonzero).
    assert cmd[0] < 0 and cmd[1] > 0


# ---------------------------------------------------------------------------
# Baseline mission success rate (>= 70%)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scripted_baseline_meets_success_rate_threshold():
    """The scripted baseline must clear >=70% mission success on the standard scenario.

    The standard scenario is a 3-point antenna grid driven by the deterministic
    scripted controllers (placement/cable/arm heuristics); we run it across a
    spread of seeds and require the success rate to meet the milestone bar.
    """
    runner = MissionScenarioRunner(scenario_factory=mission_placement_factory)
    grid = [[5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0]]
    successes = 0
    n = 12
    for seed in range(n):
        ep = runner.run_episode({"grid_points": grid}, seed=seed)
        successes += int(ep.success)
    success_rate = successes / n
    assert success_rate >= 0.70, f"baseline success rate {success_rate:.2f} < 0.70"
