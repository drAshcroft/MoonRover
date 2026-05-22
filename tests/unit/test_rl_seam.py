"""Unit tests for the RL seam: PolicyInterface + ScenarioGymEnv.

Covers:
- ScriptedPolicy / RLPolicy: observe->act, mode + confidence, switch_mode and
  get_confidence aliases, mode-validation, act-before-observe error.
- ScenarioGymEnv: spaces, reset/step contract, termination, gymnasium
  check_env conformance.
- The same env loop drives both scripted and RL policies identically, and an
  action-driven scenario actually consumes the RL action (apply_action seam).
"""

from __future__ import annotations

import numpy as np
import pytest

from moon_rover.rl.common.gym_wrapper import (
    GymEnvConfig,
    ScenarioGymEnv,
    make_env,
)
from moon_rover.rl.common.policy_interface import (
    BasePolicy,
    PolicyMode,
    RLPolicy,
    ScriptedPolicy,
)
from moon_rover.scenarios.runner import Scenario


# ---------------------------------------------------------------------------
# PolicyInterface
# ---------------------------------------------------------------------------


def test_scripted_policy_observe_act():
    policy = ScriptedPolicy(lambda obs: {"cmd": obs["x"] * 2.0})
    policy.observe({"x": np.array([1.0, 2.0])})
    action = policy.act()
    np.testing.assert_allclose(action["cmd"], [2.0, 4.0])
    assert policy.get_mode() == PolicyMode.SCRIPTED
    assert policy.get_confidence() == pytest.approx(1.0)


def test_act_before_observe_raises():
    policy = ScriptedPolicy(lambda obs: obs)
    with pytest.raises(RuntimeError, match="before observe"):
        policy.act()


def test_reset_clears_observation():
    policy = ScriptedPolicy(lambda obs: obs)
    policy.observe({"x": np.zeros(2)})
    policy.reset()
    with pytest.raises(RuntimeError):
        policy.act()


def test_switch_mode_alias_and_validation():
    policy = ScriptedPolicy(lambda obs: obs)
    # ScriptedPolicy supports SCRIPTED + FALLBACK.
    policy.switch_mode(PolicyMode.FALLBACK)
    assert policy.get_mode() == PolicyMode.FALLBACK
    with pytest.raises(ValueError, match="not supported"):
        policy.switch_mode(PolicyMode.RL)


def test_rl_policy_predict_and_confidence():
    calls = []

    def predict(obs):
        calls.append(obs)
        return {"cmd": np.array([0.5])}

    policy = RLPolicy(predict, confidence=0.4)
    assert policy.get_mode() == PolicyMode.RL
    assert policy.get_confidence() == pytest.approx(0.4)
    policy.observe({"s": np.zeros(3)})
    out = policy.act()
    np.testing.assert_allclose(out["cmd"], [0.5])
    assert len(calls) == 1
    policy.set_confidence(0.9)
    assert policy.get_confidence() == pytest.approx(0.9)
    # RL policy may fall back to scripted control.
    policy.switch_mode(PolicyMode.FALLBACK)
    assert policy.get_mode() == PolicyMode.FALLBACK


def test_confidence_clamped():
    policy = RLPolicy(lambda o: o, confidence=5.0)
    assert policy.get_confidence() == 1.0
    policy.set_confidence(-1.0)
    assert policy.get_confidence() == 0.0


# ---------------------------------------------------------------------------
# ScenarioGymEnv
# ---------------------------------------------------------------------------


def test_env_spaces():
    env = make_env(scenario_config={"duration_s": 1.0})
    assert env.observation_space.shape == (8,)
    assert env.action_space.shape == (2,)
    assert env.get_action_space()["low"] == -1.0


def test_env_reset_and_step_contract():
    env = make_env(scenario_config={"duration_s": 1.0})
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains(obs)
    assert "sim_time" in info

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "episode_steps" in info


def test_env_terminates_at_mission_end():
    env = make_env(scenario_config={"duration_s": 1.0, "dt": 0.05})
    env.reset(seed=1)
    terminated = False
    steps = 0
    while not terminated and steps < 1000:
        _obs, _r, terminated, truncated, _info = env.step(env.action_space.sample())
        steps += 1
        if truncated:
            break
    assert terminated is True
    assert steps == 20  # 1.0 s / 0.05 s


def test_env_truncates_at_max_steps():
    env = ScenarioGymEnv(
        scenario_config={"duration_s": 100.0, "dt": 0.05},
        config=GymEnvConfig(max_steps=10),
    )
    env.reset(seed=1)
    truncated = False
    for _ in range(10):
        _obs, _r, terminated, truncated, _info = env.step(env.action_space.sample())
    assert truncated is True
    assert terminated is False


def test_env_step_before_reset_raises():
    env = make_env()
    with pytest.raises(RuntimeError, match="before reset"):
        env.step(np.zeros(2, dtype=np.float32))


def test_env_passes_gymnasium_check():
    from gymnasium.utils.env_checker import check_env

    env = ScenarioGymEnv(
        scenario_config={"duration_s": 5.0, "dt": 0.05},
        config=GymEnvConfig(max_steps=50),
    )
    # Raises if the env violates the gymnasium API contract.
    check_env(env, skip_render_check=True)


# ---------------------------------------------------------------------------
# Scripted and RL modes drive the env identically
# ---------------------------------------------------------------------------


class _ActionDriveScenario(Scenario):
    """Action-driven scenario: the rover moves by the applied action vector.

    Proves the apply_action seam — an RL policy's action actually controls the
    rover, using the same env loop a scripted policy uses.
    """

    rover_type = "action_drive"

    def __init__(self, config=None):
        cfg = dict(config or {})
        self.dt = float(cfg.get("dt", 0.1))
        self.n_steps = int(cfg.get("n_steps", 10))
        self._pos = np.zeros(3)
        self._action = np.zeros(2)
        self._i = 0

    def setup(self, seed, *, visualize=False):
        self._pos = np.zeros(3)
        self._action = np.zeros(2)
        self._i = 0

    def apply_action(self, action):
        self._action = np.asarray(action, dtype=np.float64)

    def step(self):
        self._pos[:2] += self._action * self.dt
        self._i += 1
        return {
            "timestamp": self._i * self.dt,
            "rover_position": self._pos.tolist(),
            "velocity": [self._action[0], self._action[1], 0.0],
        }

    def is_complete(self):
        return self._i >= self.n_steps

    def teardown(self):
        pass

    def succeeded(self):
        return True


def _run_policy(env, policy):
    obs, _info = env.reset(seed=0)
    done = False
    while not done:
        policy.observe({"obs": obs})
        action = policy.act()["cmd"]
        obs, _r, terminated, truncated, _info = env.step(action)
        done = terminated or truncated
    return obs


def test_scripted_and_rl_modes_use_same_env_loop():
    # A constant +X command, expressed once as a scripted heuristic and once as
    # an "RL" predict_fn. Same env, same loop, same resulting trajectory.
    factory = lambda cfg: _ActionDriveScenario(cfg)
    cfg = GymEnvConfig(obs_dim=3, action_dim=2, max_steps=10)

    scripted = ScriptedPolicy(lambda o: {"cmd": np.array([1.0, 0.0], dtype=np.float32)})
    rl = RLPolicy(lambda o: {"cmd": np.array([1.0, 0.0], dtype=np.float32)}, confidence=0.8)

    env_s = ScenarioGymEnv(
        factory, {"n_steps": 10, "dt": 0.1}, config=cfg,
        observation_fn=lambda r: np.asarray(r["rover_position"], dtype=np.float32),
    )
    env_r = ScenarioGymEnv(
        factory, {"n_steps": 10, "dt": 0.1}, config=cfg,
        observation_fn=lambda r: np.asarray(r["rover_position"], dtype=np.float32),
    )
    final_s = _run_policy(env_s, scripted)
    final_r = _run_policy(env_r, rl)

    # Rover advanced +1 m/s * 0.1 s * 10 steps = +1 m in X under both modes.
    np.testing.assert_allclose(final_s, final_r)
    assert final_s[0] == pytest.approx(1.0)
