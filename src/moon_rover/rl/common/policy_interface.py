"""
System 18: RL Seam Architecture — Policy Interface

Unified policy interface for both scripted and RL-based controllers.
Frozen at Phase 3 milestone with requirement that both implementations
provide identical interface for seamless fallback and switching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import numpy.typing as npt


class PolicyMode(Enum):
    """
    Operating mode for policy execution.

    Attributes:
        SCRIPTED: Deterministic scripted/heuristic controller.
        RL: Neural network policy (trained via reinforcement learning).
        FALLBACK: Fallback mode when confidence is low (typically scripted).
    """

    SCRIPTED = "scripted"
    RL = "rl"
    FALLBACK = "fallback"


class PolicyInterface(ABC):
    """
    Unified interface for both scripted and RL policies.

    Provides a common API for task controllers (antenna placement, cable
    deployment, arm manipulation) whether implemented as heuristic rules or
    trained neural networks. Enables runtime mode switching and confidence-based
    fallback to scripted control.

    Design frozen at Phase 3 milestone: Both scripted and RL implementations
    must provide identical observe/act interface to ensure interchangeability.

    Key design principle: Observations and actions are numpy arrays with
    fixed shapes and types, allowing seamless swapping of implementations.
    """

    @abstractmethod
    def observe(self, observation: dict[str, npt.NDArray]) -> None:
        """
        Receive observation from environment.

        Called each control step with latest sensor measurements. Policy
        processes and stores observation for use in subsequent act() call.

        Observation dict format depends on task but typically includes:
        - Joint positions/velocities (for manipulation)
        - Force-torque measurements (for force control)
        - Depth images (for visual servoing)
        - Environmental features (distance to obstacles, etc.)

        Args:
            observation: Dict mapping string keys to observation arrays.
                All arrays should be float32 or float64 for numerical stability.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def act(self) -> dict[str, npt.NDArray]:
        """
        Compute and return control action.

        Called after observe() to generate next action based on accumulated
        observation. Must return action dict with consistent structure and
        shape across all calls.

        Action dict format depends on task but typically includes:
        - Joint velocity targets (rad/s)
        - Gripper commands (0/1 for on/off, or continuous 0-1)
        - Tension advisory (tension setpoint for cable control)
        - Spool feed rate (cable deployment speed)

        Returns:
            Dict mapping string keys to action arrays. Actions should be
            normalized to [-1, 1] or [0, 1] range for network policies,
            and then rescaled by task-specific gains.

        Raises:
            RuntimeError: If policy not initialized or observation missing.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset policy to initial state.

        Clears internal state including observation buffers, recurrent hidden
        states (for LSTM policies), and episode counters. Called at start of
        new episode or when episode is terminated.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def get_mode(self) -> PolicyMode:
        """
        Get current policy execution mode.

        Returns:
            Current PolicyMode (SCRIPTED, RL, or FALLBACK).
        """
        raise NotImplementedError

    @abstractmethod
    def set_mode(self, mode: PolicyMode) -> None:
        """
        Set policy execution mode.

        Allows runtime switching between scripted and RL implementations.
        For example, mission controller can switch to FALLBACK mode if
        RL policy confidence drops below threshold.

        Args:
            mode: Target PolicyMode.

        Returns:
            None

        Raises:
            ValueError: If mode not supported by this policy.
        """
        raise NotImplementedError

    @abstractmethod
    def check_confidence(self) -> float:
        """
        Get confidence metric for current policy output.

        Scripted policies typically return 1.0 (fully confident).
        RL policies return confidence based on training state:
        - Early training: 0.0-0.3 (low confidence, use fallback)
        - Mid training: 0.3-0.7 (medium confidence, monitor closely)
        - Converged: 0.7-1.0 (high confidence, use RL policy)

        Mission controller can use this to decide whether to switch to
        FALLBACK mode.

        Returns:
            Confidence score in [0, 1]. 0 = completely unreliable,
            1 = completely reliable.
        """
        raise NotImplementedError

    # -- task-level convenience aliases (frozen Phase-3 API) --------------
    # The descriptive spec lists act()/reset()/get_mode()/switch_mode()/
    # get_confidence(). switch_mode and get_confidence are thin aliases over
    # set_mode and check_confidence so callers can use either name.

    def switch_mode(self, mode: "PolicyMode") -> None:
        """Alias for :meth:`set_mode` (runtime scripted<->RL switching)."""
        self.set_mode(mode)

    def get_confidence(self) -> float:
        """Alias for :meth:`check_confidence`."""
        return self.check_confidence()


from typing import Callable, Optional  # noqa: E402


class BasePolicy(PolicyInterface):
    """Common machinery for concrete policies.

    Stores the last observation, current mode, and a confidence value. Mode
    switching is validated against ``supported_modes``. Subclasses implement
    :meth:`_compute_action`.
    """

    def __init__(
        self,
        *,
        mode: PolicyMode = PolicyMode.SCRIPTED,
        confidence: float = 1.0,
        supported_modes: Optional[set[PolicyMode]] = None,
    ) -> None:
        self._mode = mode
        self._confidence = float(np.clip(confidence, 0.0, 1.0))
        self._supported = supported_modes or {PolicyMode.SCRIPTED, PolicyMode.RL, PolicyMode.FALLBACK}
        self._last_obs: Optional[dict[str, npt.NDArray]] = None

    def observe(self, observation: dict[str, npt.NDArray]) -> None:
        self._last_obs = observation

    def act(self) -> dict[str, npt.NDArray]:
        if self._last_obs is None:
            raise RuntimeError("act() called before observe(); no observation available")
        return self._compute_action(self._last_obs)

    def reset(self) -> None:
        self._last_obs = None

    def get_mode(self) -> PolicyMode:
        return self._mode

    def set_mode(self, mode: PolicyMode) -> None:
        if mode not in self._supported:
            raise ValueError(f"mode {mode} not supported by {type(self).__name__}")
        self._mode = mode

    def check_confidence(self) -> float:
        return self._confidence

    def _compute_action(self, observation: dict[str, npt.NDArray]) -> dict[str, npt.NDArray]:
        raise NotImplementedError


class ScriptedPolicy(BasePolicy):
    """Deterministic policy backed by a heuristic ``act_fn``.

    Always reports full confidence and operates in SCRIPTED mode. Use as a
    baseline controller and as the FALLBACK target for RL policies.
    """

    def __init__(
        self,
        act_fn: Callable[[dict[str, npt.NDArray]], dict[str, npt.NDArray]],
        *,
        confidence: float = 1.0,
    ) -> None:
        super().__init__(
            mode=PolicyMode.SCRIPTED,
            confidence=confidence,
            supported_modes={PolicyMode.SCRIPTED, PolicyMode.FALLBACK},
        )
        self._act_fn = act_fn

    def _compute_action(self, observation: dict[str, npt.NDArray]) -> dict[str, npt.NDArray]:
        return self._act_fn(observation)


class RLPolicy(BasePolicy):
    """Adapter wrapping a trained model's ``predict(obs) -> action``.

    Mirrors the stable-baselines3 ``model.predict`` convention. Confidence is
    supplied by the caller (e.g. derived from training progress or value
    estimates) so a mission controller can fall back to scripted control when
    it drops below a threshold.
    """

    def __init__(
        self,
        predict_fn: Callable[[dict[str, npt.NDArray]], dict[str, npt.NDArray]],
        *,
        confidence: float = 0.5,
    ) -> None:
        super().__init__(
            mode=PolicyMode.RL,
            confidence=confidence,
            supported_modes={PolicyMode.RL, PolicyMode.FALLBACK, PolicyMode.SCRIPTED},
        )
        self._predict_fn = predict_fn

    def set_confidence(self, confidence: float) -> None:
        self._confidence = float(np.clip(confidence, 0.0, 1.0))

    def _compute_action(self, observation: dict[str, npt.NDArray]) -> dict[str, npt.NDArray]:
        return self._predict_fn(observation)
