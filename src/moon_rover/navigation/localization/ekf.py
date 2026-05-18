"""
System 11.2: Localisation — Extended Kalman Filter

Extended Kalman Filter implementation for multi-sensor fusion combining IMU,
wheel encoders, GPS, sun sensor, and lidar scan matching for robust localization
in lunar regolith with challenging GPS conditions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


@dataclass
class EKFState:
    """
    Complete state estimate for Extended Kalman Filter.

    Maintains 15-dimensional state vector: position (3), velocity (3), orientation
    (3 Euler angles), and gyroscope bias (3) with associated covariance matrix.

    Attributes:
        position_xyz: 3-element [x, y, z] position in world frame (meters).
        velocity_xyz: 3-element [vx, vy, vz] velocity in world frame (m/s).
        orientation_rpy: 3-element [roll, pitch, yaw] Euler angles (radians).
        gyro_bias_xyz: 3-element gyroscope bias [bx, by, bz] (rad/s).
        covariance: 15x15 symmetric positive-definite covariance matrix.
    """

    position_xyz: npt.NDArray[np.float64]
    velocity_xyz: npt.NDArray[np.float64]
    orientation_rpy: npt.NDArray[np.float64]
    gyro_bias_xyz: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]


@dataclass
class EKFConfig:
    """
    Configuration for Extended Kalman Filter.

    Specifies state dimensions, sensor update rates, and fusion strategy.

    Attributes:
        state_dim: Dimensionality of state vector. Fixed at 15.
        imu_rate_hz: IMU prediction update rate (Hz). Default: 200 Hz.
        encoder_rate_hz: Wheel encoder measurement rate (Hz). Default: 50 Hz.
        gps_rate_hz: GPS measurement rate (Hz). Default: 1 Hz.
        scan_match_rate_hz: Lidar scan matching (ICP) rate (Hz). Default: 5 Hz.
    """

    state_dim: int = 15
    imu_rate_hz: int = 200
    encoder_rate_hz: int = 50
    gps_rate_hz: int = 1
    scan_match_rate_hz: int = 5


class IMUReading(Protocol):
    """Protocol for IMU sensor readings."""

    @property
    def timestamp(self) -> float:
        """Sensor timestamp in seconds."""
        ...

    @property
    def accel_xyz(self) -> npt.NDArray[np.float64]:
        """3-element acceleration vector (m/s^2)."""
        ...

    @property
    def gyro_xyz(self) -> npt.NDArray[np.float64]:
        """3-element angular velocity vector (rad/s)."""
        ...


class EncoderReading(Protocol):
    """Protocol for wheel encoder measurements."""

    @property
    def timestamp(self) -> float:
        """Sensor timestamp in seconds."""
        ...

    @property
    def left_distance_m(self) -> float:
        """Distance traveled by left wheel (meters)."""
        ...

    @property
    def right_distance_m(self) -> float:
        """Distance traveled by right wheel (meters)."""
        ...

    @property
    def track_width_m(self) -> float:
        """Distance between left and right wheels (meters)."""
        ...


class GPSFix(Protocol):
    """Protocol for GPS position fixes."""

    @property
    def timestamp(self) -> float:
        """Sensor timestamp in seconds."""
        ...

    @property
    def position_xyz(self) -> npt.NDArray[np.float64]:
        """3-element [x, y, z] position (meters)."""
        ...

    @property
    def position_covariance(self) -> npt.NDArray[np.float64]:
        """3x3 covariance matrix for position (m^2)."""
        ...

    @property
    def fix_quality(self) -> str:
        """Quality indicator: 'poor', 'good', 'excellent'."""
        ...


class SunReading(Protocol):
    """Protocol for sun sensor heading measurements."""

    @property
    def timestamp(self) -> float:
        """Sensor timestamp in seconds."""
        ...

    @property
    def heading_rad(self) -> float:
        """Estimated heading/yaw angle (radians)."""
        ...

    @property
    def confidence(self) -> float:
        """Confidence in measurement [0, 1]."""
        ...


class LocalizationEKF(ABC):
    """
    Extended Kalman Filter for multi-sensor localization.

    Fuses measurements from IMU (200 Hz), wheel encoders (50 Hz), GPS (1 Hz),
    sun sensor (1 Hz), and lidar scan matching (5 Hz) to maintain a robust
    position and orientation estimate in challenging lunar terrain.

    The filter uses constant-velocity motion model with IMU-derived acceleration
    and angular velocity predictions, corrected by relative motion from encoders
    and absolute references from GPS and sun sensor.
    """

    @abstractmethod
    def initialize(
        self,
        config: EKFConfig,
        initial_state: EKFState,
    ) -> None:
        """
        Initialize EKF with configuration and initial state.

        Args:
            config: EKF configuration specifying rates and dimensions.
            initial_state: Initial estimate of rover state with covariance.

        Returns:
            None
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, imu_reading: IMUReading) -> EKFState:
        """
        Predict step using IMU measurement (200 Hz).

        Propagates state forward using constant-velocity model with IMU
        acceleration and gyro angular velocity. Updates time-evolving
        covariance to account for process noise.

        Args:
            imu_reading: IMU measurement containing acceleration and gyro rates.

        Returns:
            Predicted EKF state after IMU update.
        """
        raise NotImplementedError

    @abstractmethod
    def update_encoder(self, encoder_reading: EncoderReading) -> EKFState:
        """
        Measurement update from wheel encoders (50 Hz).

        Uses differential wheel odometry to provide relative displacement
        measurement. Helps constrain translation between GPS updates and
        corrects accumulated drift from IMU integration.

        Args:
            encoder_reading: Wheel distance measurements from left and right wheels.

        Returns:
            Corrected EKF state after encoder measurement.
        """
        raise NotImplementedError

    @abstractmethod
    def update_gps(self, gps_fix: GPSFix) -> EKFState:
        """
        Measurement update from GPS position fix (1 Hz).

        Incorporates absolute position measurement with dynamic covariance
        based on fix quality. Helps anchor global position and reset drift
        from local odometry.

        Args:
            gps_fix: GPS position and associated uncertainty.

        Returns:
            Corrected EKF state after GPS measurement.
        """
        raise NotImplementedError

    @abstractmethod
    def update_sun_sensor(self, sun_reading: SunReading) -> EKFState:
        """
        Measurement update from sun sensor for heading correction (1 Hz).

        Uses absolute heading measurement from sun position (lunar local time
        known). Provides drift-free yaw estimate to prevent accumulated
        gyro bias from rotating the rover's orientation estimate.

        Args:
            sun_reading: Sun-based heading measurement and confidence.

        Returns:
            Corrected EKF state after sun sensor measurement.
        """
        raise NotImplementedError

    @abstractmethod
    def update_scan_match(
        self,
        transform: npt.NDArray[np.float64],
    ) -> EKFState:
        """
        Measurement update from lidar scan matching (ICP) (5 Hz).

        Registers current lidar scan against map to estimate incremental pose
        change. Provides high-fidelity relative odometry for path-relative
        localization between sparse GPS updates.

        Args:
            transform: 4x4 homogeneous transformation matrix from scan matching.

        Returns:
            Corrected EKF state after scan match measurement.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> EKFState:
        """
        Retrieve current EKF state estimate.

        Returns:
            Current state with position, velocity, orientation, and covariance.
        """
        raise NotImplementedError

    @abstractmethod
    def get_position_accuracy(self) -> float:
        """
        Get 1-sigma position uncertainty from covariance.

        Computes the trace of the 3x3 position block of the covariance matrix.

        Returns:
            1-sigma position error in meters.
        """
        raise NotImplementedError

    @abstractmethod
    def check_mahalanobis_gate(
        self,
        innovation: npt.NDArray[np.float64],
        sensor_cov: npt.NDArray[np.float64],
    ) -> bool:
        """
        Outlier rejection using Mahalanobis distance gating.

        Checks if measurement innovation (residual) is consistent with predicted
        measurement uncertainty. Rejects measurements beyond chi-squared threshold
        to improve robustness to outliers and GPS multipath.

        Args:
            innovation: Measurement residual vector.
            sensor_cov: Sensor measurement covariance matrix.

        Returns:
            True if measurement passes gate (should be incorporated), False otherwise.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete implementation
# ---------------------------------------------------------------------------

from scipy.linalg import block_diag  # noqa: E402
from scipy.stats import chi2  # noqa: E402


class LocalizationEKFImpl(LocalizationEKF):
    """15-state Extended Kalman Filter for lunar rover localization.

    State vector (15-D):
        [0:3]  position xyz (m)
        [3:6]  velocity xyz (m/s)
        [6:9]  orientation rpy (rad)
        [9:12] gyroscope bias (rad/s)
        [12:15] accelerometer bias (m/s^2)

    Sensor fusion:
        - IMU (200 Hz): predict step — integrate accel & gyro
        - Wheel encoders (50 Hz): constrain planar translation
        - GPS / beacon network (1 Hz): absolute position fix
        - Sun sensor (1 Hz): absolute yaw correction
        - LiDAR scan match / ICP (5 Hz): relative pose update
    """

    _N = 15
    _CHI2_95 = {3: 7.815, 1: 3.841, 6: 12.592}  # chi2 thresholds at 95%

    def __init__(self) -> None:
        self._state: npt.NDArray[np.float64] = np.zeros(self._N, dtype=np.float64)
        self._P: npt.NDArray[np.float64] = np.eye(self._N, dtype=np.float64) * 1.0
        self._config: EKFConfig | None = None
        self._t_prev: float | None = None

        # Default process noise covariance (tuned for lunar regolith)
        self._Q = block_diag(
            np.eye(3) * 1e-4,   # position
            np.eye(3) * 1e-3,   # velocity
            np.eye(3) * 1e-4,   # orientation
            np.eye(3) * 1e-6,   # gyro bias
            np.eye(3) * 1e-6,   # accel bias
        ).astype(np.float64)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rpy_to_R(rpy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        r, p, y = rpy
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
        return Rz @ Ry @ Rx

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    def _wrap_state_angles(self) -> None:
        for i in range(6, 9):
            self._state[i] = self._wrap_angle(self._state[i])

    def _ekf_update(
        self,
        H: npt.NDArray[np.float64],
        z: npt.NDArray[np.float64],
        h: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
    ) -> None:
        """Generic EKF measurement update. Modifies self._state and self._P in place."""
        innov = z - h
        # wrap angular innovations
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.solve(S.T, np.eye(S.shape[0])).T
        self._state = self._state + K @ innov
        I_KH = np.eye(self._N) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T  # Joseph form
        self._wrap_state_angles()

    def _build_state(self) -> EKFState:
        return EKFState(
            position_xyz=self._state[0:3].copy(),
            velocity_xyz=self._state[3:6].copy(),
            orientation_rpy=self._state[6:9].copy(),
            gyro_bias_xyz=self._state[9:12].copy(),
            covariance=self._P.copy(),
        )

    # ------------------------------------------------------------------
    # AbstractLocalEKF interface
    # ------------------------------------------------------------------

    def initialize(self, config: EKFConfig, initial_state: EKFState) -> None:
        self._config = config
        self._state[0:3] = initial_state.position_xyz
        self._state[3:6] = initial_state.velocity_xyz
        self._state[6:9] = initial_state.orientation_rpy
        self._state[9:12] = initial_state.gyro_bias_xyz
        self._state[12:15] = 0.0  # accel bias
        self._P = initial_state.covariance.copy() if initial_state.covariance.shape == (self._N, self._N) else np.eye(self._N) * 1.0
        self._t_prev = None

    def predict(self, imu_reading: "IMUReading") -> EKFState:
        t = imu_reading.timestamp
        dt = (t - self._t_prev) if self._t_prev is not None else (1.0 / (self._config.imu_rate_hz if self._config else 200))
        self._t_prev = t
        if dt <= 0 or dt > 1.0:
            dt = 1.0 / 200.0

        rpy = self._state[6:9]
        R = self._rpy_to_R(rpy)
        g_lunar = np.array([0.0, 0.0, -1.62], dtype=np.float64)

        accel_world = R @ (imu_reading.accel_xyz - self._state[12:15]) + g_lunar
        omega_corrected = imu_reading.gyro_xyz - self._state[9:12]

        # Euler integration
        self._state[0:3] += self._state[3:6] * dt + 0.5 * accel_world * dt ** 2
        self._state[3:6] += accel_world * dt
        self._state[6:9] += omega_corrected * dt
        self._wrap_state_angles()

        # Linearised state transition Jacobian F
        F = np.eye(self._N, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 12:15] = -R * dt  # velocity depends on accel bias
        F[6:9, 9:12] = -np.eye(3) * dt  # orientation depends on gyro bias

        self._P = F @ self._P @ F.T + self._Q * dt

        return self._build_state()

    def update_encoder(self, encoder_reading: "EncoderReading") -> EKFState:
        """Odometry update: constrain planar velocity and yaw rate from wheels.

        The encoder gives linear displacement d_centre and angular displacement
        d_theta per period dt.  We form two separate 1-D updates:
          1. vx, vy velocity from d_centre projected through current yaw
          2. yaw rate omega from d_theta/dt — measured against the yaw-rate
             component of the gyro-integrated orientation delta.

        Using separate 1-D updates avoids the large off-diagonal Kalman gain
        cross-terms that arise when H[yaw_rate_row, heading_col] = 1/dt amplifies
        the covariance and inadvertently drives the velocity estimate.
        """
        dl = encoder_reading.left_distance_m
        dr = encoder_reading.right_distance_m
        track = encoder_reading.track_width_m
        d_centre = (dl + dr) / 2.0
        d_theta = (dr - dl) / track
        yaw = self._state[8]
        dt = 1.0 / (self._config.encoder_rate_hz if self._config else 50)

        # --- Update 1: constrain planar velocity (vx, vy) ---
        vx_meas = d_centre * np.cos(yaw) / dt
        vy_meas = d_centre * np.sin(yaw) / dt
        z_v = np.array([vx_meas, vy_meas], dtype=np.float64)
        H_v = np.zeros((2, self._N), dtype=np.float64)
        H_v[0, 3] = 1.0  # vx
        H_v[1, 4] = 1.0  # vy
        R_v = np.diag([0.02 ** 2, 0.02 ** 2])
        h_v = H_v @ self._state
        innov_v = z_v - h_v
        S_v = H_v @ self._P @ H_v.T + R_v
        K_v = self._P @ H_v.T @ np.linalg.solve(S_v.T, np.eye(2)).T
        self._state = self._state + K_v @ innov_v
        self._P = (np.eye(self._N) - K_v @ H_v) @ self._P

        # --- Update 2: constrain yaw rate via heading angle increment ---
        # Measurement: expected heading after one encoder period = yaw + d_theta.
        # Compare to current heading state (after velocity update above).
        z_yaw = np.array([yaw + d_theta], dtype=np.float64)
        H_yaw = np.zeros((1, self._N), dtype=np.float64)
        H_yaw[0, 8] = 1.0
        R_yaw = np.array([[0.005 ** 2]])
        h_yaw = H_yaw @ self._state
        innov_yaw = np.array([self._wrap_angle(float(z_yaw[0] - h_yaw[0]))])
        S_yaw = H_yaw @ self._P @ H_yaw.T + R_yaw
        K_yaw = (self._P @ H_yaw.T) / float(S_yaw[0, 0])
        self._state = self._state + K_yaw.flatten() * innov_yaw[0]
        self._P = (np.eye(self._N) - np.outer(K_yaw.flatten(), H_yaw)) @ self._P

        self._wrap_state_angles()
        return self._build_state()

    def update_gps(self, gps_fix: "GPSFix") -> EKFState:
        """Position fix update (GPS or beacon triangulation)."""
        z = gps_fix.position_xyz.astype(np.float64)
        R_gps = gps_fix.position_covariance.astype(np.float64)
        quality = gps_fix.fix_quality
        if quality == "poor":
            R_gps = R_gps * 9.0
        elif quality == "good":
            R_gps = R_gps * 1.0
        # else "excellent": use as-is
        H = np.zeros((3, self._N), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)
        h = H @ self._state
        innov = z - h
        if self.check_mahalanobis_gate(innov, H @ self._P @ H.T + R_gps):
            self._ekf_update(H, z, h, R_gps)
        return self._build_state()

    def update_sun_sensor(self, sun_reading: "SunReading") -> EKFState:
        """Heading update from sun sensor (absolute yaw)."""
        if sun_reading.confidence < 0.1:
            return self._build_state()
        z = np.array([sun_reading.heading_rad], dtype=np.float64)
        R_sun = np.array([[np.radians(2.0) ** 2 / max(sun_reading.confidence, 0.01)]])
        H = np.zeros((1, self._N), dtype=np.float64)
        H[0, 8] = 1.0  # yaw
        h = H @ self._state
        innov = np.array([self._wrap_angle(float(z[0] - h[0]))])
        if self.check_mahalanobis_gate(innov, H @ self._P @ H.T + R_sun):
            S = H @ self._P @ H.T + R_sun
            K = (self._P @ H.T) / float(S[0, 0])
            self._state += K.flatten() * innov[0]
            self._P = (np.eye(self._N) - np.outer(K.flatten(), H)) @ self._P
            self._wrap_state_angles()
        return self._build_state()

    def update_scan_match(self, transform: npt.NDArray[np.float64]) -> EKFState:
        """ICP relative pose update from 4x4 homogeneous transform."""
        # Extract translation and yaw from transform
        T = np.asarray(transform, dtype=np.float64)
        dxy = T[0:2, 3]
        dyaw = np.arctan2(T[1, 0], T[0, 0])
        z = np.array([dxy[0], dxy[1], dyaw], dtype=np.float64)
        R_icp = np.diag([0.01 ** 2, 0.01 ** 2, np.radians(0.5) ** 2])
        H = np.zeros((3, self._N), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 8] = 1.0
        h = H @ self._state
        innov = z - h
        innov[2] = self._wrap_angle(innov[2])
        if self.check_mahalanobis_gate(innov, H @ self._P @ H.T + R_icp):
            S = H @ self._P @ H.T + R_icp
            K = self._P @ H.T @ np.linalg.solve(S.T, np.eye(3)).T
            self._state += K @ innov
            self._P = (np.eye(self._N) - K @ H) @ self._P
            self._wrap_state_angles()
        return self._build_state()

    def get_state(self) -> EKFState:
        return self._build_state()

    def get_position_accuracy(self) -> float:
        return float(np.sqrt(np.trace(self._P[0:3, 0:3])))

    def check_mahalanobis_gate(
        self,
        innovation: npt.NDArray[np.float64],
        sensor_cov: npt.NDArray[np.float64],
    ) -> bool:
        innov = np.asarray(innovation, dtype=np.float64).flatten()
        S = np.asarray(sensor_cov, dtype=np.float64)
        try:
            md2 = float(innov @ np.linalg.solve(S, innov))
        except np.linalg.LinAlgError:
            return False
        dim = len(innov)
        threshold = self._CHI2_95.get(dim, chi2.ppf(0.95, df=dim))
        return md2 <= threshold
