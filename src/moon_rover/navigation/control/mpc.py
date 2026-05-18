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


# ---------------------------------------------------------------------------
# Concrete implementation — CasADi-based MPC
# ---------------------------------------------------------------------------

import casadi as ca  # noqa: E402


class MPCController(MotionController):
    """CasADi-based Model Predictive Controller for differential-drive rover.

    Unicycle model: state = [x, y, θ, v], control = [v_cmd, ω_cmd].
    Horizon: horizon_s / step_s steps (default 20 steps at 0.1 s = 2 s).
    Cost: tracking error + control effort + cable tension penalty.
    Constraints: speed limits from terrain slope and cable tension.

    The NLP is built once in configure() using CasADi Opti and reused for
    each compute() call by updating parameters (current state, reference).
    """

    _SLOPE_V_FLAT = 1.5    # m/s below 5°
    _SLOPE_V_MED = 0.8     # m/s at 15°
    _SLOPE_V_STEEP = 0.4   # m/s at 20°
    _SLOPE_IMPASSABLE = 25.0  # degrees

    def __init__(self) -> None:
        self._config: MPCConfig | None = None
        self._opti: ca.Opti | None = None
        self._U: ca.MX | None = None
        self._X: ca.MX | None = None
        self._p_x0: ca.MX | None = None
        self._p_ref: ca.MX | None = None
        self._p_v_max: ca.MX | None = None
        self._p_omega_max: ca.MX | None = None
        self._N: int = 20

    # ------------------------------------------------------------------
    # NLP construction
    # ------------------------------------------------------------------

    def _build_opti(self, cfg: MPCConfig) -> None:
        self._N = max(1, int(round(cfg.horizon_s / cfg.step_s)))
        dt = cfg.step_s
        opti = ca.Opti()

        # Decision variables: states (N+1) x 4, controls N x 2
        X = opti.variable(4, self._N + 1)  # [x, y, theta, v]
        U = opti.variable(2, self._N)       # [v_cmd, omega_cmd]

        # Parameters: initial state, reference trajectory, speed limits
        p_x0 = opti.parameter(4)
        p_ref = opti.parameter(3, self._N)    # reference [x, y, theta] per step
        p_v_max = opti.parameter(1)
        p_omega_max = opti.parameter(1)

        # Cost weights
        w_pos = 10.0
        w_heading = 5.0
        w_v = 0.1
        w_omega = 0.5
        w_dv = 1.0
        w_domega = 1.0

        cost = 0.0
        # Initial state constraint
        opti.subject_to(X[:, 0] == p_x0)

        for k in range(self._N):
            x_k = X[:, k]
            u_k = U[:, k]
            # Unicycle dynamics: x' = v*cos(θ), y' = v*sin(θ), θ' = ω, v' = u_v
            x_next = ca.vertcat(
                x_k[0] + x_k[3] * ca.cos(x_k[2]) * dt,
                x_k[1] + x_k[3] * ca.sin(x_k[2]) * dt,
                x_k[2] + u_k[1] * dt,
                u_k[0],
            )
            opti.subject_to(X[:, k + 1] == x_next)

            # Tracking cost
            ref_k = p_ref[:, k]
            dx = x_k[0] - ref_k[0]
            dy = x_k[1] - ref_k[1]
            dtheta = x_k[2] - ref_k[2]
            cost += w_pos * (dx ** 2 + dy ** 2)
            cost += w_heading * dtheta ** 2
            cost += w_v * u_k[0] ** 2
            cost += w_omega * u_k[1] ** 2

            # Control smoothness penalty
            if k > 0:
                cost += w_dv * (U[0, k] - U[0, k - 1]) ** 2
                cost += w_domega * (U[1, k] - U[1, k - 1]) ** 2

            # Speed constraints (parameterised so they can change per solve)
            opti.subject_to(u_k[0] >= -p_v_max)
            opti.subject_to(u_k[0] <= p_v_max)
            opti.subject_to(u_k[1] >= -p_omega_max)
            opti.subject_to(u_k[1] <= p_omega_max)

        # Terminal cost
        ref_T = p_ref[:, -1]
        cost += 50.0 * ((X[0, -1] - ref_T[0]) ** 2 + (X[1, -1] - ref_T[1]) ** 2)

        opti.minimize(cost)
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 200,
            "print_time": 0,
            "ipopt.tol": 1e-4,
            "ipopt.warm_starting_symbolic": True,
        }
        opti.solver("ipopt", opts)

        self._opti = opti
        self._X = X
        self._U = U
        self._p_x0 = p_x0
        self._p_ref = p_ref
        self._p_v_max = p_v_max
        self._p_omega_max = p_omega_max

    # ------------------------------------------------------------------
    # Reference extraction from path waypoints
    # ------------------------------------------------------------------

    def _extract_reference(
        self,
        current_state: npt.NDArray[np.float64],
        reference_path: list[npt.NDArray[np.float64]],
    ) -> npt.NDArray[np.float64]:
        """Sample N evenly spaced reference poses from the path ahead."""
        if not reference_path:
            xy = current_state[:2]
            return np.tile(np.array([xy[0], xy[1], current_state[2]]), (self._N, 1)).T

        # Find closest point on path
        path_xy = np.array([p[:2] for p in reference_path])
        dists = np.linalg.norm(path_xy - current_state[:2], axis=1)
        closest = int(np.argmin(dists))

        ref = np.zeros((3, self._N), dtype=np.float64)
        for k in range(self._N):
            idx = min(closest + k, len(reference_path) - 1)
            wp = reference_path[idx]
            # Heading: point toward next waypoint
            if idx + 1 < len(reference_path):
                nxt = reference_path[idx + 1]
                heading = float(np.arctan2(nxt[1] - wp[1], nxt[0] - wp[0]))
            else:
                heading = float(current_state[2])
            ref[0, k] = wp[0]
            ref[1, k] = wp[1]
            ref[2, k] = heading
        return ref

    # ------------------------------------------------------------------
    # MotionController interface
    # ------------------------------------------------------------------

    def configure(self, config: MPCConfig) -> None:
        self._config = config
        self._build_opti(config)

    def compute(
        self,
        current_state: npt.NDArray[np.float64],
        reference_path: list[npt.NDArray[np.float64]],
        cable_tension_n: float,
        terrain_slope_deg: float,
        speed_limit_factor: float,
    ) -> MPCOutput:
        if self._opti is None or self._config is None:
            self.configure(MPCConfig())

        v_max = self.get_speed_limit(terrain_slope_deg, cable_tension_n) * speed_limit_factor
        omega_max = self._config.max_angular_vel

        if v_max <= 0.0:
            # Emergency stop
            traj = [np.array(current_state[:5], dtype=np.float64)] * self._N
            return MPCOutput(linear_velocity=0.0, angular_velocity=0.0, predicted_trajectory=traj)

        ref = self._extract_reference(current_state, reference_path)

        # Build 4-element state [x, y, theta, v]
        v0 = current_state[3] if len(current_state) > 3 else 0.0
        x0 = np.array([current_state[0], current_state[1], current_state[2], v0])

        opti = self._opti
        opti.set_value(self._p_x0, x0)
        opti.set_value(self._p_ref, ref)
        opti.set_value(self._p_v_max, v_max)
        opti.set_value(self._p_omega_max, omega_max)

        # Warm-start controls
        opti.set_initial(self._U, np.zeros((2, self._N)))
        opti.set_initial(self._X, np.tile(x0, (self._N + 1, 1)).T)

        try:
            sol = opti.solve()
            u_opt = sol.value(self._U)
            x_opt = sol.value(self._X)
            v_cmd = float(np.clip(u_opt[0, 0], -v_max, v_max))
            omega_cmd = float(np.clip(u_opt[1, 0], -omega_max, omega_max))
            traj = [x_opt[:, k].flatten() for k in range(self._N + 1)]
        except Exception:
            # Solver failed: fall back to pure pursuit toward first waypoint
            v_cmd, omega_cmd = self._pure_pursuit_fallback(current_state, reference_path, v_max, omega_max)
            traj = [np.array(current_state[:5], dtype=np.float64)] * (self._N + 1)

        return MPCOutput(
            linear_velocity=v_cmd,
            angular_velocity=omega_cmd,
            predicted_trajectory=traj,
        )

    def _pure_pursuit_fallback(
        self,
        state: npt.NDArray[np.float64],
        path: list[npt.NDArray[np.float64]],
        v_max: float,
        omega_max: float,
    ) -> tuple[float, float]:
        if not path:
            return 0.0, 0.0
        # Find lookahead point
        L = 1.5  # lookahead distance
        target = path[-1]
        for wp in path:
            if np.linalg.norm(wp[:2] - state[:2]) >= L:
                target = wp
                break
        dx = target[0] - state[0]
        dy = target[1] - state[1]
        desired_heading = float(np.arctan2(dy, dx))
        heading_error = desired_heading - state[2]
        heading_error = float((heading_error + np.pi) % (2 * np.pi) - np.pi)
        omega = float(np.clip(2.0 * heading_error, -omega_max, omega_max))
        v = v_max * max(0.0, 1.0 - abs(heading_error) / np.pi)
        return v, omega

    def get_speed_limit(self, slope_deg: float, cable_tension: float) -> float:
        # Slope-based limit
        s = abs(slope_deg)
        if s >= self._SLOPE_IMPASSABLE:
            v_slope = 0.0
        elif s >= 20.0:
            v_slope = self._SLOPE_V_STEEP
        elif s >= 15.0:
            t = (s - 15.0) / 5.0
            v_slope = self._SLOPE_V_STEEP * t + self._SLOPE_V_MED * (1 - t)
        elif s >= 5.0:
            t = (s - 5.0) / 10.0
            v_slope = self._SLOPE_V_MED * t + self._SLOPE_V_FLAT * (1 - t)
        else:
            v_slope = self._SLOPE_V_FLAT

        # Cable tension modulation
        t_low = self._config.cable_tension_speed_limit_threshold_n if self._config else 200.0
        t_high = t_low * 2.0
        if cable_tension >= t_high:
            v_cable = 0.0
        elif cable_tension >= t_low:
            v_cable = (t_high - cable_tension) / (t_high - t_low) * v_slope
        else:
            v_cable = v_slope

        return float(min(v_slope, v_cable))
