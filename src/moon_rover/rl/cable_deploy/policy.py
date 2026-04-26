"""
System 20: Cable Deployment Policy

Reinforcement learning and scripted policies for cable spool control during
antenna deployment. Learns to manage cable tension, avoid snags, and optimize
deployment rate based on terrain and cable conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from moon_rover.rl.common.policy_interface import PolicyInterface


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


class CableDeploymentPolicy(PolicyInterface):
    """
    Policy for cable spool deployment control.

    Implements the PolicyInterface for cable deployment task. Both scripted
    and RL implementations provide identical observe/act interface.

    RL policy learns tension regulation, snag avoidance, and deployment
    optimization through interaction. Scripted policy uses tension feedback
    control with speed limits based on rover motion.

    See PolicyInterface for full method documentation.
    """

    pass
