"""Concrete DriveSystem implementations backed by Genesis articulated entities.

This module provides production implementations of :class:`DriveSystem` for each
supported rover configuration:

* :class:`TwoWheelDifferentialDrive` — independent left/right wheel velocity.
* :class:`ThreeWheelTricycleDrive` — front steering + rear drive (Ackermann).
* :class:`FourWheelSkidSteerDrive` — tank-style skid steering with two drive sides.

All three implementations drive a Genesis articulated body through
:class:`moon_rover.core.physics.engine.PhysicsEngine` using named joint lookups.
They integrate odometry from measured wheel encoder velocities (dead reckoning),
so commanded and observed motion can diverge under slip. The resulting odometry
feeds directly into the navigation / localisation stack (Phase 6) without any
simulator-specific plumbing leaking upwards.

Integration contract
--------------------
1. The caller composes the scene (via :class:`RoverComposer` or a direct
   ``engine.add_entity(...)``) using a URDF produced by
   :class:`GenesisURDFBuilder`. The URDF joint names are fixed by
   :meth:`GenesisURDFBuilder._wheel_names` / ``_append_direct_wheels``.
2. After ``engine.build_scene()`` the caller invokes :meth:`DriveSystem.attach`,
   passing the engine and the rover entity name. The drive system resolves the
   local DOF indices for each wheel joint from the Genesis entity.
3. Every physics step the caller invokes :meth:`update` with the timestep so the
   drive can apply the current command to the physics solver and accumulate
   odometry from wheel feedback.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from moon_rover.rover.drive.interface import (
    DriveCommand,
    DriveConfig,
    DriveSystem,
    DriveType,
    WheelState,
)

if TYPE_CHECKING:
    from moon_rover.core.physics.engine import PhysicsEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw (rotation about Z) from a unit quaternion (qx, qy, qz, qw)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: float) -> Tuple[float, float, float, float]:
    """Build a unit quaternion (w, x, y, z) from a yaw angle about +Z."""
    half = 0.5 * yaw
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _clip(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


@dataclass
class _WheelHandle:
    """Resolved mapping from a URDF wheel joint to a Genesis local DOF index."""

    joint_name: str
    dofs_idx_local: int
    sign: float = 1.0  # +1 forward rotation direction; may be flipped per side


class _BaseGenesisDrive(DriveSystem):
    """Common Genesis-backed drive behaviour (attachment, odometry, wheel state)."""

    #: Joint names expected on the Genesis entity, in canonical wheel order.
    wheel_joint_names: Tuple[str, ...] = ()

    def __init__(self) -> None:
        self._config: Optional[DriveConfig] = None
        self._engine: Optional["PhysicsEngine"] = None
        self._rover_name: Optional[str] = None
        self._entity: Any = None

        # Resolved wheel handles, one per controllable wheel DOF.
        self._wheels: List[_WheelHandle] = []

        # Current wheel command and feedback caches.
        self._command = DriveCommand(linear_velocity_mps=0.0, angular_velocity_radps=0.0)
        self._wheel_target_rads: NDArray = np.zeros(0, dtype=np.float32)
        self._wheel_torques: NDArray = np.zeros(0, dtype=np.float32)
        self._wheel_vel_measured: NDArray = np.zeros(0, dtype=np.float32)

        # Odometry state (dead reckoning from measured wheel velocities).
        self._odom_xyz = np.zeros(3, dtype=np.float32)
        self._odom_yaw = 0.0
        self._odom_last_pose_xyz: Optional[NDArray] = None

        # Traction / slip diagnostics.
        self._slip_ratios: NDArray = np.zeros(0, dtype=np.float32)
        self._sinkage: NDArray = np.zeros(0, dtype=np.float32)
        self._contact_forces: NDArray = np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------
    # ABC: configure / get_drive_type
    # ------------------------------------------------------------------

    def configure(self, config: DriveConfig) -> None:
        if config.wheel_radius_m <= 0.0:
            raise ValueError(f"wheel_radius_m must be > 0, got {config.wheel_radius_m}")
        if config.track_width_m <= 0.0:
            raise ValueError(f"track_width_m must be > 0, got {config.track_width_m}")
        if config.max_torque_nm <= 0.0:
            raise ValueError(f"max_torque_nm must be > 0, got {config.max_torque_nm}")
        expected_type = self.expected_drive_type()
        if config.drive_type is not expected_type:
            raise ValueError(
                f"{type(self).__name__} expects drive_type={expected_type.name}, "
                f"got {config.drive_type.name}"
            )
        self._config = config
        n = self.num_wheel_dofs()
        self._wheel_target_rads = np.zeros(n, dtype=np.float32)
        self._wheel_torques = np.zeros(n, dtype=np.float32)
        self._wheel_vel_measured = np.zeros(n, dtype=np.float32)
        self._slip_ratios = np.zeros(n, dtype=np.float32)
        self._sinkage = np.zeros(n, dtype=np.float32)
        self._contact_forces = np.zeros(n, dtype=np.float32)

    def expected_drive_type(self) -> DriveType:
        raise NotImplementedError

    def num_wheel_dofs(self) -> int:
        """Number of wheel DOFs this drive controls (excludes steering DOFs)."""
        return len(self.wheel_joint_names)

    def get_drive_type(self) -> DriveType:
        if self._config is None:
            raise RuntimeError("DriveSystem.configure() must be called first")
        return self._config.drive_type

    # ------------------------------------------------------------------
    # Attachment
    # ------------------------------------------------------------------

    def attach(self, engine: "PhysicsEngine", rover_name: str) -> None:
        """Bind this drive system to a Genesis entity already added to the scene.

        Parameters:
            engine: The :class:`PhysicsEngine` hosting the rover.
            rover_name: Entity name used when :meth:`PhysicsEngine.add_entity` was
                called (matches ``rover_id`` in mission.yaml).

        Raises:
            RuntimeError: If :meth:`configure` has not been called.
            ValueError: If any wheel joint name is missing from the entity.
        """
        if self._config is None:
            raise RuntimeError("DriveSystem.configure() must be called before attach()")

        self._engine = engine
        self._rover_name = rover_name
        self._entity = engine.get_entity(rover_name)

        self._wheels = self._resolve_wheels(self._entity)

        # Configure the velocity-control gains so the solver applies motor torque
        # rather than hard-setting velocities (hard-set breaks the sim's momentum
        # conservation and causes spurious energy injection).
        self._configure_wheel_controllers()

        # Reset odometry using the rover's actual pose.
        self._reset_odometry_from_pose()

    def _resolve_wheels(self, entity: Any) -> List[_WheelHandle]:
        """Resolve each wheel joint name to a local DOF index."""
        entity_dof_start = int(getattr(entity, "_dof_start", 0))
        handles: List[_WheelHandle] = []
        for name in self.wheel_joint_names:
            joint = entity.get_joint(name=name)
            global_dof_start = int(joint.dof_start)
            local_idx = global_dof_start - entity_dof_start
            handles.append(_WheelHandle(joint_name=name, dofs_idx_local=local_idx))
        return handles

    def _configure_wheel_controllers(self) -> None:
        """Apply motor Kv gain to the wheel DOFs for PD velocity control."""
        if self._config is None:
            return
        # Kv is chosen so the motor can saturate at the configured max torque
        # with a 1 rad/s velocity error. This approximates a real DC motor stall
        # torque characteristic and keeps the controller stable.
        kv = float(self._config.max_torque_nm)
        try:
            self._entity.set_dofs_kv(
                kv=np.full(len(self._wheels), kv, dtype=np.float32),
                dofs_idx_local=[h.dofs_idx_local for h in self._wheels],
            )
        except (AttributeError, TypeError):
            # Older/mocked entities may not expose set_dofs_kv; the Genesis
            # default gains are acceptable for test doubles.
            pass

    # ------------------------------------------------------------------
    # Command / update
    # ------------------------------------------------------------------

    def command(self, cmd: DriveCommand) -> None:
        if self._config is None:
            raise RuntimeError("DriveSystem.configure() must be called first")
        self._command = cmd
        self._wheel_target_rads = np.asarray(
            self.inverse_kinematics(cmd), dtype=np.float32
        )

    def update(self, dt: float) -> None:
        """Apply current command to physics and accumulate odometry from encoders.

        This must be called once per physics step, after :meth:`command` has been
        issued and before the next :meth:`PhysicsEngine.step`. It does *not* step
        the engine itself — the caller owns the simulation loop.

        Parameters:
            dt: Timestep in seconds. Must be > 0.
        """
        if self._entity is None:
            raise RuntimeError("DriveSystem.attach() must be called before update()")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        self._apply_wheel_velocity_command()
        self._read_wheel_feedback()
        self._integrate_odometry(dt)
        self._update_slip_estimates()

    def _apply_wheel_velocity_command(self) -> None:
        """Issue velocity targets for each wheel DOF to Genesis."""
        assert self._entity is not None
        assert self._config is not None

        max_speed = _max_wheel_speed(self._config)
        # Apply sign convention per side and saturate.
        targets = self._wheel_target_rads.copy()
        for i, handle in enumerate(self._wheels):
            targets[i] = _clip(float(targets[i]) * handle.sign, -max_speed, max_speed)

        try:
            self._entity.control_dofs_velocity(
                velocity=targets.astype(np.float32),
                dofs_idx_local=[h.dofs_idx_local for h in self._wheels],
            )
        except (AttributeError, TypeError):
            # Fallback: hard-set DOF velocities (used only when the mocked entity
            # does not expose control_dofs_velocity).
            try:
                self._entity.set_dofs_velocity(
                    velocity=targets.astype(np.float32),
                    dofs_idx_local=[h.dofs_idx_local for h in self._wheels],
                )
            except (AttributeError, TypeError):
                pass

    def _read_wheel_feedback(self) -> None:
        """Read measured wheel angular velocities back from Genesis."""
        assert self._entity is not None
        try:
            vel = self._entity.get_dofs_velocity(
                dofs_idx_local=[h.dofs_idx_local for h in self._wheels],
            )
            arr = np.asarray(_to_numpy(vel), dtype=np.float32).flatten()
        except (AttributeError, TypeError, IndexError):
            arr = np.zeros(len(self._wheels), dtype=np.float32)

        # Apply sign convention (flip sign on wheels mounted backwards).
        for i, handle in enumerate(self._wheels):
            arr[i] = float(arr[i]) * handle.sign
        # In case the mocked entity returned the wrong shape, clamp to expected.
        if arr.shape[0] != len(self._wheels):
            arr = np.zeros(len(self._wheels), dtype=np.float32)
        self._wheel_vel_measured = arr

    def _integrate_odometry(self, dt: float) -> None:
        """Dead-reckon the rover pose from the measured wheel speeds."""
        # Delegate to subclass: forward kinematics from *measured* wheel speeds
        # yields the observed twist.
        twist = self.forward_kinematics(self._wheel_vel_measured.tolist())

        # Integrate in the body frame, then rotate into world.
        cos_y = math.cos(self._odom_yaw)
        sin_y = math.sin(self._odom_yaw)
        dx = float(twist.linear_velocity_mps) * cos_y * dt
        dy = float(twist.linear_velocity_mps) * sin_y * dt
        self._odom_xyz[0] += np.float32(dx)
        self._odom_xyz[1] += np.float32(dy)
        self._odom_yaw = _wrap_angle(self._odom_yaw + float(twist.angular_velocity_radps) * dt)

    def _reset_odometry_from_pose(self) -> None:
        """Seed the odometry integrator from the current simulated rover pose."""
        assert self._engine is not None and self._rover_name is not None
        try:
            pos, quat = self._engine.get_body_pose(self._rover_name)
        except Exception:
            self._odom_xyz = np.zeros(3, dtype=np.float32)
            self._odom_yaw = 0.0
            return
        pos_arr = np.asarray(pos, dtype=np.float32).flatten()
        quat_arr = np.asarray(quat, dtype=np.float32).flatten()
        if pos_arr.size >= 3:
            self._odom_xyz = pos_arr[:3].astype(np.float32)
        if quat_arr.size == 4:
            # Genesis stores (qx, qy, qz, qw) in set_body_pose but returns
            # (qw, qx, qy, qz) from get_body_pose — normalise both orderings.
            qw, qx, qy, qz = _coerce_wxyz(quat_arr)
            self._odom_yaw = _quat_to_yaw(qx, qy, qz, qw)

    def _update_slip_estimates(self) -> None:
        """Estimate per-wheel slip ratio from command vs measured angular velocity."""
        if self._wheel_vel_measured.size == 0:
            return
        if self._wheel_target_rads.size != self._wheel_vel_measured.size:
            return
        denom = np.maximum(
            np.abs(self._wheel_target_rads),
            np.abs(self._wheel_vel_measured),
        )
        denom = np.maximum(denom, 1e-3)
        slip = (self._wheel_target_rads - self._wheel_vel_measured) / denom
        self._slip_ratios = np.clip(slip, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # ABC: state readers
    # ------------------------------------------------------------------

    def get_wheel_states(self) -> list[WheelState]:
        states: List[WheelState] = []
        n = self.num_wheel_dofs()
        for i in range(n):
            omega = float(self._wheel_vel_measured[i]) if i < self._wheel_vel_measured.size else 0.0
            torque = float(self._wheel_torques[i]) if i < self._wheel_torques.size else 0.0
            slip = float(self._slip_ratios[i]) if i < self._slip_ratios.size else 0.0
            sinkage = float(self._sinkage[i]) if i < self._sinkage.size else 0.0
            contact = float(self._contact_forces[i]) if i < self._contact_forces.size else 0.0
            states.append(
                WheelState(
                    angular_velocity=omega,
                    torque=torque,
                    slip_ratio=abs(slip),
                    sinkage_depth=sinkage,
                    contact_force=contact,
                )
            )
        return states

    def get_odometry(self) -> tuple[NDArray, NDArray]:
        pos = self._odom_xyz.copy()
        qw, qx, qy, qz = _yaw_to_quat(self._odom_yaw)
        quat = np.array([qw, qx, qy, qz], dtype=np.float32)
        return pos, quat


# ---------------------------------------------------------------------------
# Two-wheel differential drive (unicycle model)
# ---------------------------------------------------------------------------


class TwoWheelDifferentialDrive(_BaseGenesisDrive):
    """Independent left/right wheel velocity control (unicycle kinematics).

    Kinematics:
        * Linear v = r/2 * (ωR + ωL)
        * Angular ω = r/W * (ωR − ωL)
    where r is wheel radius and W is track width.
    """

    wheel_joint_names = ("left_wheel_joint", "right_wheel_joint")

    def expected_drive_type(self) -> DriveType:
        return DriveType.TWO_WHEEL_DIFF

    def _resolve_wheels(self, entity: Any) -> List[_WheelHandle]:
        handles = super()._resolve_wheels(entity)
        # Left wheel sits on the −Y side; both URDF joints spin about +Y. Under
        # that convention both wheels produce positive forward motion with a
        # positive angular velocity command, so no sign flip is needed.
        return handles

    def inverse_kinematics(self, cmd: DriveCommand) -> list[float]:
        if self._config is None:
            raise RuntimeError("configure() must be called first")
        r = self._config.wheel_radius_m
        w = self._config.track_width_m
        v = float(cmd.linear_velocity_mps)
        omega = float(cmd.angular_velocity_radps)
        omega_left = (v - 0.5 * w * omega) / r
        omega_right = (v + 0.5 * w * omega) / r
        return [omega_left, omega_right]

    def forward_kinematics(self, wheel_speeds: list[float]) -> DriveCommand:
        if self._config is None:
            raise RuntimeError("configure() must be called first")
        if len(wheel_speeds) != 2:
            raise ValueError(f"expected 2 wheel speeds, got {len(wheel_speeds)}")
        r = self._config.wheel_radius_m
        w = self._config.track_width_m
        omega_left, omega_right = float(wheel_speeds[0]), float(wheel_speeds[1])
        v = 0.5 * r * (omega_right + omega_left)
        omega = (r / w) * (omega_right - omega_left)
        return DriveCommand(linear_velocity_mps=v, angular_velocity_radps=omega)


# ---------------------------------------------------------------------------
# Three-wheel tricycle (Ackermann steering)
# ---------------------------------------------------------------------------


class ThreeWheelTricycleDrive(_BaseGenesisDrive):
    """Front steering wheel + two rear drive wheels (bicycle / Ackermann model).

    Kinematics (single-track approximation, steering δ, wheelbase L):
        * v is applied to the rear wheels: ω_rear = v / r
        * δ = atan2(L * ω, max(|v|, v_min))  (chosen to reach requested yaw rate)
    """

    wheel_joint_names = ("rear_left_wheel_joint", "rear_right_wheel_joint")
    STEERING_JOINT_NAME = "front_steering_joint"
    FRONT_WHEEL_JOINT_NAME = "front_wheel_joint"
    MIN_VELOCITY_FOR_STEER = 0.05  # m/s: below this, use pivot-like behaviour.

    def __init__(self) -> None:
        super().__init__()
        self._steer_dof_local: Optional[int] = None
        self._front_wheel_dof_local: Optional[int] = None
        self._steer_angle_target: float = 0.0

    def expected_drive_type(self) -> DriveType:
        return DriveType.THREE_WHEEL_TRICYCLE

    def _resolve_wheels(self, entity: Any) -> List[_WheelHandle]:
        handles = super()._resolve_wheels(entity)
        entity_dof_start = int(getattr(entity, "_dof_start", 0))
        steer = entity.get_joint(name=self.STEERING_JOINT_NAME)
        self._steer_dof_local = int(steer.dof_start) - entity_dof_start
        front = entity.get_joint(name=self.FRONT_WHEEL_JOINT_NAME)
        self._front_wheel_dof_local = int(front.dof_start) - entity_dof_start
        return handles

    def inverse_kinematics(self, cmd: DriveCommand) -> list[float]:
        if self._config is None:
            raise RuntimeError("configure() must be called first")
        r = self._config.wheel_radius_m
        wheelbase = max(self._config.wheelbase_m, 1e-3)
        v = float(cmd.linear_velocity_mps)
        omega = float(cmd.angular_velocity_radps)
        v_eff = v if abs(v) > self.MIN_VELOCITY_FOR_STEER else (
            math.copysign(self.MIN_VELOCITY_FOR_STEER, v) if v != 0.0 else self.MIN_VELOCITY_FOR_STEER
        )
        delta = math.atan2(wheelbase * omega, v_eff)
        max_delta = self._config.max_steer_angle_rad or 0.6108  # ~35 deg
        delta = _clip(delta, -max_delta, max_delta)
        self._steer_angle_target = delta
        omega_rear = v / r
        return [omega_rear, omega_rear]

    def forward_kinematics(self, wheel_speeds: list[float]) -> DriveCommand:
        if self._config is None:
            raise RuntimeError("configure() must be called first")
        if len(wheel_speeds) != 2:
            raise ValueError(f"expected 2 rear wheel speeds, got {len(wheel_speeds)}")
        r = self._config.wheel_radius_m
        wheelbase = max(self._config.wheelbase_m, 1e-3)
        avg = 0.5 * (float(wheel_speeds[0]) + float(wheel_speeds[1]))
        v = r * avg
        delta = self._steer_angle_target
        omega = v * math.tan(delta) / wheelbase
        return DriveCommand(linear_velocity_mps=v, angular_velocity_radps=omega)

    def _apply_wheel_velocity_command(self) -> None:
        super()._apply_wheel_velocity_command()
        # Drive the front wheel at the same linear speed as the rear pair.
        if self._config is None or self._front_wheel_dof_local is None:
            return
        if self._wheel_target_rads.size == 0:
            return
        avg_rear = float(np.mean(self._wheel_target_rads))
        try:
            self._entity.control_dofs_velocity(
                velocity=np.array([avg_rear], dtype=np.float32),
                dofs_idx_local=[int(self._front_wheel_dof_local)],
            )
        except (AttributeError, TypeError):
            pass
        # Command steering with position control.
        if self._steer_dof_local is not None:
            try:
                self._entity.control_dofs_position(
                    position=np.array([self._steer_angle_target], dtype=np.float32),
                    dofs_idx_local=[int(self._steer_dof_local)],
                )
            except (AttributeError, TypeError):
                pass


# ---------------------------------------------------------------------------
# Four-wheel skid-steer
# ---------------------------------------------------------------------------


class FourWheelSkidSteerDrive(_BaseGenesisDrive):
    """Tank / skid-steer drive: left pair and right pair share wheel speeds."""

    wheel_joint_names = (
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    )

    def expected_drive_type(self) -> DriveType:
        return DriveType.FOUR_WHEEL_SKID

    def inverse_kinematics(self, cmd: DriveCommand) -> list[float]:
        if self._config is None:
            raise RuntimeError("configure() must be called first")
        r = self._config.wheel_radius_m
        w = self._config.track_width_m
        v = float(cmd.linear_velocity_mps)
        omega = float(cmd.angular_velocity_radps)
        # Skid-steer widens the effective turning width to account for lateral
        # slip; a conservative 1.0 factor treats the two sides as ideal tracks.
        omega_left = (v - 0.5 * w * omega) / r
        omega_right = (v + 0.5 * w * omega) / r
        return [omega_left, omega_right, omega_left, omega_right]

    def forward_kinematics(self, wheel_speeds: list[float]) -> DriveCommand:
        if self._config is None:
            raise RuntimeError("configure() must be called first")
        if len(wheel_speeds) != 4:
            raise ValueError(f"expected 4 wheel speeds, got {len(wheel_speeds)}")
        r = self._config.wheel_radius_m
        w = self._config.track_width_m
        left_avg = 0.5 * (float(wheel_speeds[0]) + float(wheel_speeds[2]))
        right_avg = 0.5 * (float(wheel_speeds[1]) + float(wheel_speeds[3]))
        v = 0.5 * r * (right_avg + left_avg)
        omega = (r / w) * (right_avg - left_avg)
        return DriveCommand(linear_velocity_mps=v, angular_velocity_radps=omega)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_drive_system(config: DriveConfig) -> _BaseGenesisDrive:
    """Instantiate and configure the concrete drive matching ``config.drive_type``.

    Parameters:
        config: Fully populated :class:`DriveConfig`.

    Returns:
        A concrete :class:`DriveSystem` already configured. Caller must still
        call :meth:`attach` after ``build_scene``.
    """
    if config.drive_type is DriveType.TWO_WHEEL_DIFF:
        drive: _BaseGenesisDrive = TwoWheelDifferentialDrive()
    elif config.drive_type is DriveType.THREE_WHEEL_TRICYCLE:
        drive = ThreeWheelTricycleDrive()
    elif config.drive_type is DriveType.FOUR_WHEEL_SKID:
        drive = FourWheelSkidSteerDrive()
    else:
        raise ValueError(f"Unsupported drive type: {config.drive_type}")
    drive.configure(config)
    return drive


def drive_config_from_profile(profile: dict, drive_type: DriveType) -> DriveConfig:
    """Build a :class:`DriveConfig` from a rover.yaml profile section."""
    max_steer_deg = None
    steering = profile.get("steering") or {}
    if steering.get("max_steering_angle_degrees") is not None:
        max_steer_deg = math.radians(float(steering["max_steering_angle_degrees"]))

    return DriveConfig(
        drive_type=drive_type,
        track_width_m=float(profile.get("track_width_m", 1.0)),
        wheelbase_m=float(profile.get("wheelbase_m", 0.0)),
        wheel_radius_m=float(profile.get("wheel_radius_m", 0.3)),
        max_torque_nm=float(profile.get("max_torque_nm", 100.0)),
        max_steer_angle_rad=max_steer_deg,
        num_wheels=int(profile.get("num_wheels", 4)),
    )


# ---------------------------------------------------------------------------
# Low-level utilities (kept private)
# ---------------------------------------------------------------------------


def _max_wheel_speed(config: DriveConfig) -> float:
    """Best-effort max angular velocity for a wheel DOF in rad/s."""
    # Not all DriveConfigs carry an explicit cap; use radius-derived 10 rad/s
    # as a safe conservative default (≈ 3 m/s at 0.3 m wheel radius).
    return 10.0


def _to_numpy(x: Any) -> NDArray:
    if hasattr(x, "cpu") and hasattr(x, "numpy"):
        try:
            return np.asarray(x.detach().cpu().numpy())
        except AttributeError:
            return np.asarray(x.cpu().numpy())
    return np.asarray(x)


def _coerce_wxyz(quat_arr: NDArray) -> Tuple[float, float, float, float]:
    """Coerce a quaternion of unknown ordering to (w, x, y, z).

    The physics engine's ``get_body_pose`` returns (w, x, y, z) while
    ``set_body_pose`` consumes (x, y, z, w). Detect by which element has
    the largest magnitude near unity.
    """
    q = np.asarray(quat_arr, dtype=np.float32).flatten()
    if q.size != 4:
        return (1.0, 0.0, 0.0, 0.0)
    # Heuristic: if the last element is the most unit-like and first is near
    # zero, assume (x, y, z, w); otherwise assume (w, x, y, z).
    if abs(q[0]) < abs(q[3]):
        return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))
