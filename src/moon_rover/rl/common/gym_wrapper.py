"""
System 23.1: Genesis Parallel Environment Wrapper

Gymnasium-compatible environment wrapper for Genesis physics simulator
with support for batched parallel environments (VectorEnv).

All observations and actions remain on GPU during rollout with no CPU
round-trip for maximum throughput during training.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class GenesisGymEnv(ABC):
    """
    Single environment interface compatible with gymnasium.Env.

    Wraps Genesis physics simulator to provide standard gymnasium environment
    interface for single-environment training or testing. Observations and
    actions are numpy arrays that can be transferred to/from GPU as needed.

    Conforms to gymnasium.Env specification:
    - reset(seed) -> (observation, info)
    - step(action) -> (observation, reward, terminated, truncated, info)
    """

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict, dict]:
        """
        Reset environment to initial state.

        Initializes or reinitializes the simulation. Sets random seed if
        provided for reproducible behavior.

        Args:
            seed: Random seed for environment. If None, uses system entropy.

        Returns:
            Tuple of (observation, info) where:
            - observation: Dict mapping string keys to observation arrays
            - info: Dict with auxiliary info (e.g., 'sim_time', 'episode_length')
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        action: npt.NDArray,
    ) -> tuple[dict, float, bool, bool, dict]:
        """
        Execute one step of environment dynamics.

        Applies action to simulator, advances physics by one step, and
        collects new observation and reward.

        Args:
            action: Action array. Shape and semantics depend on action_space.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info):
            - observation: Dict of observation arrays for next state
            - reward: Scalar float reward for this step
            - terminated: Boolean indicating episode termination (goal/failure)
            - truncated: Boolean indicating episode truncation (time limit)
            - info: Dict with auxiliary info (e.g., 'sim_time')

        Raises:
            RuntimeError: If environment not initialized with reset().
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation_space(self) -> dict:
        """
        Get observation space specification.

        Returns:
            Dict describing observation format:
            ```
            {
                'joint_positions': {'shape': (4,), 'dtype': 'float32'},
                'joint_velocities': {'shape': (4,), 'dtype': 'float32'},
                'force_torque': {'shape': (6,), 'dtype': 'float32'},
                ...
            }
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self) -> dict:
        """
        Get action space specification.

        Returns:
            Dict describing action format:
            ```
            {
                'joint_velocity_targets': {'shape': (4,), 'dtype': 'float32',
                                          'low': -0.5, 'high': 0.5}
            }
            ```
        """
        raise NotImplementedError


class GenesisVectorEnv(ABC):
    """
    Batched parallel environment interface for N simultaneous environments.

    Manages N independent Genesis simulations running in parallel on GPU.
    Provides vectorized step() that advances all N environments in one call.
    All observations and actions stay on GPU with no CPU transfers during
    rollout, enabling efficient policy gradient training.

    Conforms to gymnasium.vector.VectorEnv specification (approximate).
    """

    @abstractmethod
    def reset_all(self) -> npt.NDArray:
        """
        Reset all N parallel environments.

        Args:
            None

        Returns:
            Observation array with shape (N,) + obs_shape, containing initial
            observations from all environments.
        """
        raise NotImplementedError

    @abstractmethod
    def step_all(
        self,
        actions: npt.NDArray,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, list[dict]]:
        """
        Execute one step in all N parallel environments.

        Applies actions to all environments simultaneously, advances physics,
        and collects results. All arrays remain on GPU.

        Args:
            actions: Action array with shape (N,) + action_shape.

        Returns:
            Tuple of:
            - observations: (N,) + obs_shape observation array (on GPU)
            - rewards: (N,) float array of rewards (on GPU)
            - terminateds: (N,) boolean array for episode termination (on GPU)
            - truncateds: (N,) boolean array for episode truncation (on GPU)
            - infos: List of N info dicts (on CPU, one per environment)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """
        Get number of parallel environments.

        Returns:
            Number of simultaneous environments (N).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete gymnasium.Env wrapping a ScenarioRunner Scenario
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Callable, Optional  # noqa: E402

try:  # gymnasium is an optional extra: pip install moon-rover[rl]
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    _GYM_AVAILABLE = False

from moon_rover.scenarios.runner import (  # noqa: E402
    Scenario,
    default_scenario_factory,
)

_GYM_INSTALL_HINT = (
    "gymnasium is required for ScenarioGymEnv; install it with "
    "'pip install moon-rover[rl]' (or 'pip install gymnasium')."
)


def default_observation(record: dict) -> "np.ndarray":
    """Default observation vector: [pos(3), vel(3), energy_wh, cable_tension]."""
    pos = np.asarray(record.get("rover_position", [0.0, 0.0, 0.0]), dtype=np.float32)
    vel = np.asarray(record.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float32)
    energy = float(record.get("energy_wh", 0.0))
    tension = float(record.get("cable_tension_n", 0.0))
    return np.concatenate([pos.ravel(), vel.ravel(), [energy, tension]]).astype(np.float32)


def default_reward(record: dict, scenario: Scenario, prev: Optional[dict]) -> float:
    """Default reward: newly-activated antennas this step minus a time penalty.

    Sparse placement bonus (+1 per antenna that became successful this step)
    keeps the seam task-agnostic; replace via ``reward_fn`` for shaped rewards.
    """
    n_success = sum(1 for p in scenario.antenna_placements if p.get("success"))
    prev_success = 0 if prev is None else int(prev.get("_n_success", 0))
    bonus = float(max(0, n_success - prev_success))
    return bonus - 0.001  # small per-step time penalty


@dataclass
class GymEnvConfig:
    """Spaces + episode config for :class:`ScenarioGymEnv`.

    Attributes:
        obs_dim: Dimensionality of the observation vector (Box).
        action_dim: Dimensionality of the continuous action (Box in [-1, 1]).
        max_steps: Truncation horizon in control ticks.
        obs_low / obs_high: Box bounds for the observation space.
    """

    obs_dim: int = 8
    action_dim: int = 2
    max_steps: int = 2000
    obs_low: float = -1e4
    obs_high: float = 1e4


if _GYM_AVAILABLE:

    class ScenarioGymEnv(gym.Env):
        """Wrap a ScenarioRunner :class:`Scenario` as a single ``gymnasium.Env``.

        The same env drives both scripted and RL control identically: each
        ``step(action)`` calls ``scenario.apply_action(action)`` (a no-op for
        self-driving scripted scenarios, consumed by action-driven RL
        scenarios) then advances one control tick. Observation and reward
        extraction are injectable; defaults cover position/velocity/energy obs
        and a sparse antenna-activation reward.

        Args mirror a typical gymnasium.Env constructor; pass ``scenario_config``
        through to the scenario factory.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            scenario_factory: Callable[[dict], Scenario] = default_scenario_factory,
            scenario_config: Optional[dict] = None,
            *,
            config: Optional[GymEnvConfig] = None,
            observation_fn: Callable[[dict], np.ndarray] = default_observation,
            reward_fn: Callable[[dict, Scenario, Optional[dict]], float] = default_reward,
        ) -> None:
            super().__init__()
            self._scenario_factory = scenario_factory
            self._scenario_config = dict(scenario_config or {})
            self._cfg = config or GymEnvConfig()
            self._observation_fn = observation_fn
            self._reward_fn = reward_fn

            self.observation_space = spaces.Box(
                low=self._cfg.obs_low,
                high=self._cfg.obs_high,
                shape=(self._cfg.obs_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self._cfg.action_dim,), dtype=np.float32
            )

            self._scenario: Optional[Scenario] = None
            self._steps = 0
            self._reward_state: dict = {}

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self._scenario = self._scenario_factory(self._scenario_config)
            self._scenario.setup(int(seed or 0), visualize=False)
            self._steps = 0
            self._reward_state = {"_n_success": 0}
            # Prime an observation without advancing: synthesize a zero record.
            obs = self._observation_fn({"rover_position": [0.0, 0.0, 0.0]})
            info = {"sim_time": 0.0, "phase": "reset"}
            return obs, info

        def step(self, action):
            if self._scenario is None:
                raise RuntimeError("step() called before reset()")
            self._scenario.apply_action(action)
            record = self._scenario.step()
            self._steps += 1

            obs = self._observation_fn(record)
            reward = float(self._reward_fn(record, self._scenario, self._reward_state))
            self._reward_state["_n_success"] = sum(
                1 for p in self._scenario.antenna_placements if p.get("success")
            )

            terminated = bool(self._scenario.is_complete())
            truncated = bool(self._steps >= self._cfg.max_steps)
            info = {
                "sim_time": float(record.get("timestamp", 0.0)),
                "episode_steps": self._steps,
                "success": bool(self._scenario.succeeded()) if terminated else False,
            }
            if terminated or truncated:
                self._scenario.teardown()
            return obs, reward, terminated, truncated, info

        def get_observation_space(self) -> dict:
            return {"shape": self.observation_space.shape, "dtype": "float32"}

        def get_action_space(self) -> dict:
            return {
                "shape": self.action_space.shape,
                "dtype": "float32",
                "low": -1.0,
                "high": 1.0,
            }

else:  # pragma: no cover - only without the gymnasium extra

    class ScenarioGymEnv:  # type: ignore[no-redef]
        """Placeholder raising a clear error when gymnasium is missing."""

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(_GYM_INSTALL_HINT)


def make_env(
    scenario_factory: Callable[[dict], Scenario] = default_scenario_factory,
    scenario_config: Optional[dict] = None,
    **kwargs,
) -> "ScenarioGymEnv":
    """Factory for a :class:`ScenarioGymEnv` (raises if gymnasium is absent)."""
    if not _GYM_AVAILABLE:
        raise ImportError(_GYM_INSTALL_HINT)
    return ScenarioGymEnv(scenario_factory, scenario_config, **kwargs)
