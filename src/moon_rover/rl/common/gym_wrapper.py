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

    @abstractmethod
    @property
    def num_envs(self) -> int:
        """
        Get number of parallel environments.

        Returns:
            Number of simultaneous environments (N).
        """
        raise NotImplementedError
