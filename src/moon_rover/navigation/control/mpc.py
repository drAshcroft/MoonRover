"""
System 11.4: Local Motion Controller — Model Predictive Control

Model Predictive Control implementation for real-time motion planning and
velocity tracking with constraint handling for cable tension, terrain slope,
and speed limits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class MPCConfig:
    """
    Configuration for Model Predictive Control.

    Attributes:
        horizon_s: Prediction horizon in seconds. Default: 2.0 s.
        step_s: Discrete time step for MPC discretization. Default: 0.1 s.
        max_linear_vel: Maximum linear velocity (m/s). Default: 1.5 m/s.
        max_angular_vel: Maximum angular velocity (rad/s). Default: 0.8 rad/s.
        update_rate_hz: Control update frequency in Hz. Default: 20 Hz.
        cable_tension_speed_limit_threshold_n: Tension threshold above which
            speed limits are applied. Default: 200 N.
    """

    horizon_s: float = 2.0
    step_s: float = 0.1
    max_linear_vel: float = 1.5
    max_angular_vel: float = 0.8
    update_rate_hz: int = 20
    cable_tension_speed_limit_threshold_n: float = 200.0


@dataclass
class MPCOutput:
    """
    Control outputs from Model Predictive Controller.

    Attributes:
        linear_velocity: Target forward/backward velocity (m/s).
        angular_velocity: Target rotational velocity (rad/s).
        predicted_trajectory: List of predicted state vectors over horizon
            (one per MPC step).
    """

    linear_velocity: float
    angular_velocity: float
    predicted_trajectory: list[npt.NDArray[np.float64]]


class MotionController(ABC):
    """
    Model Predictive Control for local motion planning and velocity control.

    Computes optimal velocity commands to follow a reference path while
    respecting constraints from terrain slope, cable tension, and mechanical
    limits. Outputs smooth, collision-free motion trajectories at 20 Hz.

    Key features:
    - Integrates terrain slope constraints (max 1.5 m/s on flat, 0.4 m/s at 20°)
    - Cable tension feedback for dynamic speed limiting
    - 2-second predictive horizon with 0.1 s time steps
    - Real-time optimization respecting diff-drive kinematics
    """

    @abstractmethod
    def configure(self, config: MPCConfig) -> None:
        """
        Configure MPC with planning parameters.

        Args:
            config: MPC configuration specifying horizon, update rate, limits.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        current_state: npt.NDArray[np.float64],
        reference_path: list[npt.NDArray[np.float64]],
        cable_tension_n: float,
        terrain_slope_deg: float,
        speed_limit_factor: float,
    ) -> MPCOutput:
        """
        Compute optimal control inputs for current timestep.

        Solves MPC optimization problem to generate velocity commands that
        track the reference path while respecting all constraints. Uses
        iterative optimization (iLQR or QP-based) to find best control inputs.

        Args:
            current_state: Current rover state [x, y, theta, v, omega] (5 elements).
            reference_path: List of goal waypoints [x, y, z] to follow.
            cable_tension_n: Current cable tension in Newtons.
            terrain_slope_deg: Local terrain slope in degrees.
            speed_limit_factor: External speed limit multiplier [0, 1].

        Returns:
            MPCOutput with computed velocities and predicted trajectory.
        """
        raise NotImplementedError

    @abstractmethod
    def get_speed_limit(
        self,
        slope_deg: float,
        cable_tension: float,
    ) -> float:
        """
        Get maximum speed limit based on terrain and cable conditions.

        Implements speed limiting strategy:
        - Flat terrain (slope < 5°): 1.5 m/s
        - Moderate slope (5-15°): linear interpolation to 0.8 m/s
        - Steep slope (15-20°): 0.4 m/s
        - Above 25°: impassable, return 0

        Cable tension modulation:
        - Below 200 N: no additional limit
        - 200-400 N: scale speed by (400 - tension) / 200
        - Above 400 N: disable motion to prevent cable failure

        Args:
            slope_deg: Terrain slope angle in degrees.
            cable_tension: Current cable tension in Newtons.

        Returns:
            Maximum allowable linear velocity in m/s.
        """
        raise NotImplementedError
