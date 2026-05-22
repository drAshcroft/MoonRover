"""
System 20: Cable Deployment Policy

Reinforcement learning and scripted policies for cable spool control during
antenna deployment. Learns to manage cable tension, avoid snags, and optimize
deployment rate based on terrain and cable conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from moon_rover.rl.common.policy_interface import BasePolicy, PolicyInterface, PolicyMode


def _field(obs: Any, name: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    arr = np.asarray(value, dtype=np.float64).ravel()
    return float(arr[0]) if arr.size else default


@dataclass
class CableDeployObservation:
    """
    Observation state for cable deployment task.

    Comprehensive sensor state for cable spool control including tension
    history, rover motion, and environmental context.

    Attributes:
        tension: 1-element current cable tension (N).
        tension_rate: 1-element cable tension time derivative (dT/dt) (N/s).
        tension_history: 100-element history buffer of cable tension over
            past ~10 seconds at 10 Hz sampling. Provides temporal context
            for learning tension dynamics.
        spool_velocity: 1-element current spool feed rate (m/s).
        rover_speed: 1-element rover forward velocity (m/s).
        rover_yaw_rate: 1-element rover angular velocity (rad/s).
        deployed_length: 1-element total cable length deployed so far (m).
        remaining_length: 1-element cable length remaining on spool (m).
        rock_proximity: 4-element distance to nearest obstacles in 4 radial
            directions (m). Avoids cables snagging on rocks.
        previous_spool_cmd: 1-element previous spool velocity command (m/s).

    Total dimension: 1 + 1 + 100 + 1 + 1 + 1 + 1 + 1 + 4 + 1 = 112 dimensions.
    """

    tension: npt.NDArray[np.float32]  # shape (1,)
    tension_rate: npt.NDArray[np.float32]  # shape (1,)
    tension_history: npt.NDArray[np.float32]  # shape (100,)
    spool_velocity: npt.NDArray[np.float32]  # shape (1,)
    rover_speed: npt.NDArray[np.float32]  # shape (1,)
    rover_yaw_rate: npt.NDArray[np.float32]  # shape (1,)
    deployed_length: npt.NDArray[np.float32]  # shape (1,)
    remaining_length: npt.NDArray[np.float32]  # shape (1,)
    rock_proximity: npt.NDArray[np.float32]  # shape (4,)
    previous_spool_cmd: npt.NDArray[np.float32]  # shape (1,)


@dataclass
class CableDeployAction:
    """
    Control action for cable deployment spool system.

    Provides two independent control channels for spool feed rate management
    and tension feedback.

    Attributes:
        spool_feed_modifier: 1-element modifier for spool feed rate.
            Normalized to [-1, 1]. Positive values increase deployment rate,
            negative values brake or retract. Rescaled to [-v_max, +v_max]
            by controller (typical range ±0.5 m/s).
        tension_advisory: 1-element target tension setpoint modifier.
            Normalized to [0, 1]. Advises desired tension as fraction of
            nominal range [100, 400] N. Used by closed-loop tension controller.
            10 Hz control rate.

    Total dimension: 1 + 1 = 2 dimensions.
    """

    spool_feed_modifier: npt.NDArray[np.float32]  # shape (1,), [-1, 1]
    tension_advisory: npt.NDArray[np.float32]  # shape (1,), [0, 1]


class CableDeploymentPolicy(BasePolicy):
    """Scripted cable-spool controller (tension feedback + snag avoidance).

    Deterministic baseline. Each tick it pays cable out to match rover motion,
    then corrects for tension and obstacle proximity:

    1. Feed-forward: nominal feed tracks ``rover_speed`` so cable neither piles
       up nor goes taut from the rover simply driving.
    2. Tension regulation: above ``high_tension_n`` it brakes (negative feed)
       to relieve a snag; below ``low_tension_n`` it may pay out faster.
    3. Snag avoidance: when the nearest ``rock_proximity`` drops below
       ``snag_clearance_m`` it slows the feed to avoid wrapping a rock.

    Outputs a feed modifier in [-1, 1] and a tension advisory in [0, 1].
    """

    def __init__(
        self,
        *,
        max_feed_mps: float = 0.5,
        low_tension_n: float = 100.0,
        high_tension_n: float = 400.0,
        snag_clearance_m: float = 0.3,
    ) -> None:
        super().__init__(
            mode=PolicyMode.SCRIPTED,
            confidence=1.0,
            supported_modes={PolicyMode.SCRIPTED, PolicyMode.FALLBACK},
        )
        self.max_feed_mps = float(max_feed_mps)
        self.low_tension_n = float(low_tension_n)
        self.high_tension_n = float(high_tension_n)
        self.snag_clearance_m = float(snag_clearance_m)

    def _compute_action(self, observation: Any) -> dict[str, npt.NDArray]:
        tension = _scalar(_field(observation, "tension"), 0.0)
        rover_speed = _scalar(_field(observation, "rover_speed"), 0.0)
        rock_prox = np.asarray(
            _field(observation, "rock_proximity", [np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float64,
        ).ravel()
        nearest = float(rock_prox.min()) if rock_prox.size else float("inf")

        # Feed-forward: pay out at rover speed (normalized by max feed).
        feed = rover_speed / self.max_feed_mps

        # Tension regulation.
        if tension > self.high_tension_n:
            feed = -1.0  # brake hard to relieve the snag
        elif tension < self.low_tension_n:
            feed += 0.2  # safe to pay out a little faster

        # Snag avoidance near rocks.
        if nearest < self.snag_clearance_m:
            feed = min(feed, 0.0)

        feed = float(np.clip(feed, -1.0, 1.0))

        # Tension advisory: target the mid-band as a fraction of [low, high].
        mid = 0.5 * (self.low_tension_n + self.high_tension_n)
        advisory = float(np.clip(mid / max(self.high_tension_n, 1e-6), 0.0, 1.0))

        return {
            "spool_feed_modifier": np.array([feed], dtype=np.float32),
            "tension_advisory": np.array([advisory], dtype=np.float32),
        }
