"""Drivable self-balancing 2-wheel rover demo with visible arm.

A hand-on sanity check that lets a human operator validate the physics, drive
controller, balance controller, and manipulator arm together in a single
interactive session. The rover is a Segway-style inverted pendulum: two lateral
drive wheels, a slim tall chassis with the center of mass above the axle, and
a visible 4-DOF manipulator arm with parallel-jaw gripper. There are no
casters, skids, or stabilizers — upright attitude is held by an active PD
balance controller that biases the wheel velocity targets with the body pitch
error.

Keyboard controls (interactive viewer mode)
-------------------------------------------
    W / S     forward / reverse velocity target  (m/s)
    A / D     spin left / right (yaw rate)       (rad/s)
    Space     zero the drive command
    1..4      select arm joint to jog
    [ / ]     jog selected joint − / +           (rad)
    O / C     open / close gripper
    Z         stow arm to neutral pose
    R         reset rover pose (respawn upright)
    ESC / Q   quit cleanly

Usage
-----
    # Interactive viewer (CPU, default):
    python scripts/demo_rover_drive.py

    # Headless smoke run, no keyboard:
    python scripts/demo_rover_drive.py --no-viewer --steps 600

    # GPU backend (NOTE: first run compiles CUDA kernels — up to ~10 minutes):
    python scripts/demo_rover_drive.py --backend gpu

Physics / viewer pacing
-----------------------
Physics runs at ``--sim-hz`` (default 120). When the viewer is on, Genesis runs
the render loop in its own thread at ``--render-hz`` (default 60). The main
thread uses a single ``time.perf_counter`` sleep to the next physics deadline;
if the sim falls behind wall clock the sleep is simply skipped rather than
bursting catch-up frames, which is what produced the jerky motion in earlier
builds. On the GPU backend the first run spends several minutes compiling
Genesis kernels — this is called out up front, not hidden behind a spinner.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Make the src/ layout importable when the script is launched directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import genesis as gs  # noqa: E402

from moon_rover.core.physics.engine import GenesisConfig  # noqa: E402
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine  # noqa: E402
from moon_rover.core.scene.rover_composer import RoverComposer  # noqa: E402
from moon_rover.core.assets.urdf_builder import GenesisURDFBuilder  # noqa: E402
from moon_rover.rover.drive import (  # noqa: E402
    DriveCommand,
    DriveType,
    create_drive_system,
    drive_config_from_profile,
)
from moon_rover.rover.manipulator.arm import ArmConfig, GripperConfig  # noqa: E402
from moon_rover.rover.manipulator.serial_arm import SerialArm  # noqa: E402
from moon_rover.rover.power import (  # noqa: E402
    RoverPowerSystem,
    power_config_from_yaml,
)

ROVER_NAME = "demo_rover"
DEFAULT_PROFILE = "two_wheel_diff"
DEFAULT_PHYSICS = _PROJECT_ROOT / "configs" / "physics.yaml"
DEFAULT_ROVER = _PROJECT_ROOT / "configs" / "rover.yaml"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drivable self-balancing 2-wheel rover demo")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Run headless (no 3-D window).")
    parser.add_argument("--steps", type=int, default=0,
                        help="Run for N physics steps then exit. 0 = until Ctrl+C / ESC.")
    parser.add_argument("--sim-hz", type=float, default=120.0,
                        help="Physics rate (Hz). Default 120.")
    parser.add_argument("--render-hz", type=int, default=60,
                        help="Viewer refresh cap (Hz). Default 60.")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu",
                        help="Genesis backend. CPU first-run build ~0–5 min, GPU ~5–10 min.")
    parser.add_argument("--physics-config", default=str(DEFAULT_PHYSICS),
                        help="Path to physics.yaml")
    parser.add_argument("--rover-config", default=str(DEFAULT_ROVER),
                        help="Path to rover.yaml")
    parser.add_argument("--profile", default=DEFAULT_PROFILE,
                        help="Rover profile name from rover.yaml (default: two_wheel_diff).")
    parser.add_argument("--sun-elevation-deg", type=float, default=35.0,
                        help="Sun elevation angle (deg). Drives solar generation.")
    parser.add_argument("--no-keyboard", action="store_true",
                        help="Disable keyboard input (useful for recording / CI).")
    parser.add_argument("--destroy", action="store_true",
                        help="Explicitly tear down Genesis on exit.")
    parser.add_argument("--self-test", action="store_true",
                        help="Inject a scripted drive+arm command sequence "
                             "(no keyboard) to verify balance under motion.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------


def load_rover_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_rover_urdf(rover_cfg: dict, profile_name: str) -> str:
    """Build the Segway rover URDF via the production composer pipeline.

    ``_profile_to_urdf_config`` reads wheel positions, chassis dimensions, arm
    geometry, and gripper parameters straight from rover.yaml. No local
    overrides — the profile in rover.yaml is the source of truth.
    """
    profiles = rover_cfg["profiles"]
    if profile_name not in profiles:
        raise KeyError(
            f"Profile '{profile_name}' not in rover.yaml. Available: {list(profiles)}"
        )
    profile = profiles[profile_name]
    config = RoverComposer._profile_to_urdf_config(
        rover_id=f"demo_{profile_name}",
        profile=profile,
        rover_cfg=rover_cfg,
    )
    return GenesisURDFBuilder().build_rover(config)


def spawn_height_for(rover_cfg: dict, profile: dict) -> float:
    """Pick a spawn Z that puts the wheels just above the ground plane.

    The URDF builder subtracts ``chassis_half_height`` from each wheel's local
    position, so the effective wheel axle height below the chassis origin is
    ``half_h - wheel_positions_z``. Spawn the chassis high enough that the
    wheel bottom clears ground by a small margin.
    """
    wheel_radius = float(profile["wheel_radius_m"])
    dims = profile.get("dimensions") or rover_cfg.get("structure", {}).get(
        "dimensions", [2.0, 1.5, 0.8]
    )
    half_h = float(dims[2]) / 2.0
    wheel_positions = profile.get("wheel_positions", [[0.0, 0.0, 0.0]])
    wheel_z = float(wheel_positions[0][2]) if wheel_positions else 0.0
    # Axle-to-chassis distance = half_h - wheel_z (for a wheel that hangs below).
    axle_below_origin = max(0.0, half_h - wheel_z)
    return axle_below_origin + wheel_radius + 0.02


def make_genesis_config(physics_path: Path, sim_hz: float, use_gpu: bool) -> GenesisConfig:
    base = GenesisConfig.from_yaml(str(physics_path))
    timestep = base.timestep if sim_hz <= 0 else 1.0 / sim_hz
    return replace(base, timestep=timestep, use_gpu=use_gpu)


# ---------------------------------------------------------------------------
# Keyboard input (non-blocking, Windows via msvcrt; POSIX via stdin termios)
# ---------------------------------------------------------------------------


class KeyboardPoll:
    """Non-blocking keyboard polling that works on Windows and POSIX terminals."""

    def __init__(self) -> None:
        self._win = sys.platform.startswith("win")
        self._queue: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._posix_state = None  # Saved termios attrs on POSIX.

    def start(self) -> None:
        if self._win:
            self._thread = threading.Thread(
                target=self._win_loop, name="keyboard-poll", daemon=True
            )
        else:
            self._enter_cbreak()
            self._thread = threading.Thread(
                target=self._posix_loop, name="keyboard-poll", daemon=True
            )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if not self._win:
            self._leave_cbreak()

    def pop_all(self) -> list[str]:
        with self._lock:
            out = self._queue[:]
            self._queue.clear()
            return out

    # -- Windows ---------------------------------------------------------
    def _win_loop(self) -> None:
        import msvcrt
        while not self._stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                with self._lock:
                    self._queue.append(ch)
            else:
                time.sleep(0.01)

    # -- POSIX -----------------------------------------------------------
    def _enter_cbreak(self) -> None:
        try:
            import termios
            import tty
            fd = sys.stdin.fileno()
            self._posix_state = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            self._posix_state = None

    def _leave_cbreak(self) -> None:
        if self._posix_state is None:
            return
        try:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._posix_state)
        except Exception:
            pass

    def _posix_loop(self) -> None:
        import select
        while not self._stop.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                ch = sys.stdin.read(1)
                with self._lock:
                    self._queue.append(ch)


# ---------------------------------------------------------------------------
# Balance controller
# ---------------------------------------------------------------------------


class BalanceController:
    """Cascaded wheel-torque self-balancing controller (Segway architecture).

    Torque control — not a velocity setpoint. An inverted pendulum needs a
    *continuous* restoring torque proportional to lean; a velocity-control
    inner loop only produces torque transiently while accelerating and cannot
    hold the steady-state moment of a top-heavy body, which is why the original
    velocity-cascade design drifted and fell.

    Two nested loops:

    * **Inner (fast, stiff)** — drives body pitch to a lean *setpoint* with a
      high-gain PD on wheel torque::

          tau_common = k_theta * (pitch - theta_sp) + k_theta_d * pitch_rate

      +pitch = forward lean; +torque drives the wheels forward, whose reaction
      torque and base motion both reduce the lean, so +(pitch-theta_sp) → +tau.

    * **Outer (slow, gentle)** — picks ``theta_sp`` so the rover regulates
      ground speed to ``v_ref`` and station-keeps. To go faster forward the
      body must lean forward, so a positive speed error commands a positive
      lean. The **velocity integral** is what rejects the constant forward
      moment of the stowed arm (a pure PD leaves a standing drift)::

          theta_sp = kp_v*(v_ref - v) + ki_v*∫(v_ref - v) + kp_x*(x_ref - x)

    A differential yaw torque is layered on top for steering.
    """

    def __init__(
        self,
        *,
        k_theta: float = 35.0,
        k_theta_d: float = 13.0,
        kp_v: float = 0.16,
        ki_v: float = 0.045,
        kp_x: float = 0.018,
        k_yaw: float = 6.0,
        max_lean_rad: float = 0.30,
        v_int_clamp: float = 6.0,
        d_filter_alpha: float = 0.25,
        max_torque_nm: float = 150.0,
    ) -> None:
        # Env overrides for fast headless gain sweeps: MR_K_THETA, MR_K_THETA_D,
        # MR_KP_V, MR_KI_V, MR_KP_X, MR_K_YAW, MR_MAX_LEAN, MR_D_ALPHA.
        def _env(name: str, default: float) -> float:
            try:
                return float(os.environ[name])
            except (KeyError, ValueError):
                return default

        self.k_theta = _env("MR_K_THETA", k_theta)
        self.k_theta_d = _env("MR_K_THETA_D", k_theta_d)
        self.kp_v = _env("MR_KP_V", kp_v)
        self.ki_v = _env("MR_KI_V", ki_v)
        self.kp_x = _env("MR_KP_X", kp_x)
        self.k_yaw = _env("MR_K_YAW", k_yaw)
        self.max_lean = _env("MR_MAX_LEAN", max_lean_rad)
        self.d_alpha = _env("MR_D_ALPHA", d_filter_alpha)
        self.v_int_clamp = v_int_clamp
        self.max_torque = max_torque_nm
        self._prev_pitch: Optional[float] = None
        self._pitch_rate_f = 0.0
        self._v_int = 0.0
        self._x = 0.0
        self._x_ref = 0.0

    def reset(self) -> None:
        self._prev_pitch = None
        self._pitch_rate_f = 0.0
        self._v_int = 0.0
        self._x = 0.0
        self._x_ref = 0.0

    def step(
        self,
        *,
        pitch_rad: float,
        body_v_mps: float,
        yaw_rate_radps: float,
        v_ref_mps: float,
        w_ref_radps: float,
        dt: float,
    ) -> tuple[float, float]:
        """Return (tau_left, tau_right) wheel torques in N·m."""
        if self._prev_pitch is None or dt <= 0.0:
            pitch_rate_raw = 0.0
        else:
            pitch_rate_raw = (pitch_rad - self._prev_pitch) / dt
        self._prev_pitch = pitch_rad
        # Low-pass the finite-difference rate: a raw 120 Hz quaternion
        # derivative is noisy enough that a stiff k_theta_d would chatter the
        # motor torque and ring the inner loop.
        self._pitch_rate_f += self.d_alpha * (pitch_rate_raw - self._pitch_rate_f)
        pitch_rate = self._pitch_rate_f

        # Travelled distance vs. commanded reference (station-keeping when
        # v_ref == 0, rides along while the operator drives).
        self._x += body_v_mps * dt
        self._x_ref += v_ref_mps * dt

        # --- Outer loop: speed/position error -> desired lean ---------------
        v_err = v_ref_mps - body_v_mps
        self._v_int += v_err * dt
        self._v_int = max(-self.v_int_clamp, min(self.v_int_clamp, self._v_int))
        theta_sp = (
            self.kp_v * v_err
            + self.ki_v * self._v_int
            + self.kp_x * (self._x_ref - self._x)
        )
        theta_sp = max(-self.max_lean, min(self.max_lean, theta_sp))

        # --- Inner loop: stiff PD drives pitch to theta_sp -----------------
        tau_common = (
            self.k_theta * (pitch_rad - theta_sp)
            + self.k_theta_d * pitch_rate
        )
        tau_yaw = self.k_yaw * (w_ref_radps - yaw_rate_radps)

        lo, hi = -self.max_torque, self.max_torque
        tau_left = max(lo, min(hi, tau_common - tau_yaw))
        tau_right = max(lo, min(hi, tau_common + tau_yaw))
        return float(tau_left), float(tau_right)


def _euler_from_quat_wxyz(q: np.ndarray) -> tuple[float, float, float]:
    """Extract (roll, pitch, yaw) radians from a Genesis (w, x, y, z) quaternion."""
    qw, qx, qy, qz = (float(q[i]) for i in range(4))
    # roll (x)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y)
    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    # yaw (z)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# Arm bridge: software SerialArm ⇄ Genesis joints
# ---------------------------------------------------------------------------


class ArmBridge:
    """Mirror SerialArm joint targets into Genesis via control_dofs_position."""

    def __init__(
        self,
        entity,
        arm: SerialArm,
        num_dof: int,
        stroke_m: float,
        kp: float = 80.0,
        kv: float = 4.0,
    ) -> None:
        self._entity = entity
        self._arm = arm
        self._num_dof = num_dof
        self._stroke = stroke_m
        self._dof_local = self._resolve_dofs(num_dof)
        self._gripper_dof_local = self._resolve_gripper_dofs()
        self._configure_gains(kp, kv)

    def _resolve_dofs(self, num_dof: int) -> list[int]:
        entity_dof_start = int(getattr(self._entity, "_dof_start", 0))
        out: list[int] = []
        for i in range(1, num_dof + 1):
            j = self._entity.get_joint(name=f"arm_joint_{i}")
            out.append(int(j.dof_start) - entity_dof_start)
        return out

    def _resolve_gripper_dofs(self) -> list[int]:
        entity_dof_start = int(getattr(self._entity, "_dof_start", 0))
        out: list[int] = []
        for name in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
            try:
                j = self._entity.get_joint(name=name)
                out.append(int(j.dof_start) - entity_dof_start)
            except Exception:
                return []
        return out

    def _configure_gains(self, kp: float, kv: float) -> None:
        all_dofs = self._dof_local + self._gripper_dof_local
        if not all_dofs:
            return
        try:
            self._entity.set_dofs_kp(
                kp=np.full(len(all_dofs), kp, dtype=np.float32),
                dofs_idx_local=all_dofs,
            )
            self._entity.set_dofs_kv(
                kv=np.full(len(all_dofs), kv, dtype=np.float32),
                dofs_idx_local=all_dofs,
            )
        except (AttributeError, TypeError):
            pass

    def sync(self) -> None:
        """Push SerialArm targets into Genesis. Call every physics step."""
        targets = np.asarray(self._arm.get_state().joint_positions, dtype=np.float32)
        if targets.size != self._num_dof:
            return
        try:
            self._entity.control_dofs_position(
                position=targets,
                dofs_idx_local=self._dof_local,
            )
        except (AttributeError, TypeError):
            return

        if self._gripper_dof_local:
            # SerialArm's gripper_position is in meters of stroke. Split
            # symmetrically across the two finger prismatic joints.
            grip_m = float(self._arm.get_state().gripper_position)
            half = 0.5 * grip_m
            try:
                self._entity.control_dofs_position(
                    position=np.array([half, half], dtype=np.float32),
                    dofs_idx_local=self._gripper_dof_local,
                )
            except (AttributeError, TypeError):
                pass


class WheelTorqueActuator:
    """Direct-torque actuation of the two wheel DOFs.

    Bypasses the production drive system's velocity controller (which cannot
    balance an inverted pendulum) and instead applies raw motor torque to the
    left/right wheel joints, reading their angular velocity back for state
    feedback. The drive system is still used for odometry/telemetry but its
    ``update()`` (velocity command) is not called for this profile.
    """

    def __init__(self, entity, wheel_dofs_local: list[int], wheel_radius_m: float) -> None:
        self._entity = entity
        self._dofs = list(wheel_dofs_local)  # canonical order: [left, right]
        self._r = float(wheel_radius_m)
        # Pure torque mode: kill any position/velocity servo gains the drive
        # system installed on the wheel DOFs so they free-spin under torque.
        n = len(self._dofs)
        for setter, val in (("set_dofs_kp", 0.0), ("set_dofs_kv", 0.0)):
            try:
                getattr(self._entity, setter)(
                    **{setter.split("_")[-1]: np.zeros(n, dtype=np.float32)},
                    dofs_idx_local=self._dofs,
                )
            except (AttributeError, TypeError):
                pass

    def wheel_omega(self) -> tuple[float, float]:
        try:
            v = self._entity.get_dofs_velocity(dofs_idx_local=self._dofs)
            if hasattr(v, "cpu"):
                v = v.detach().cpu().numpy() if hasattr(v, "detach") else v.cpu().numpy()
            arr = np.asarray(v, dtype=np.float64).flatten()
            if arr.size >= 2:
                return float(arr[0]), float(arr[1])
        except (AttributeError, TypeError, IndexError):
            pass
        return 0.0, 0.0

    def body_velocity(self) -> float:
        """Forward ground velocity (m/s) from mean wheel angular velocity."""
        ol, orr = self.wheel_omega()
        return self._r * 0.5 * (ol + orr)

    def apply(self, tau_left: float, tau_right: float) -> None:
        try:
            self._entity.control_dofs_force(
                force=np.array([tau_left, tau_right], dtype=np.float32),
                dofs_idx_local=self._dofs,
            )
        except (AttributeError, TypeError):
            pass


# ---------------------------------------------------------------------------
# Header / telemetry
# ---------------------------------------------------------------------------


def print_header(args: argparse.Namespace, profile: dict) -> None:
    print("=" * 92)
    print("  Moon Rover - Self-balancing 2-Wheel Demo")
    print("=" * 92)
    print(f"  Profile     : {profile.get('name', args.profile)}")
    print(f"  Wheels      : 2 (lateral axle, track={profile['track_width_m']:.2f} m, "
          f"radius={profile['wheel_radius_m']:.2f} m)")
    print(f"  Chassis     : {profile.get('dimensions', '(from structure)')} m (L,W,H)")
    print(f"  Arm         : visible 4-DOF + parallel-jaw gripper (URDF-emitted)")
    print(f"  Backend     : {args.backend.upper()}  sim @ {args.sim_hz:.0f} Hz")
    print(f"  Viewer      : {'off (headless)' if args.no_viewer else f'on @ {args.render_hz} Hz'}")
    if args.backend == "gpu":
        print("  *** GPU FIRST RUN: Genesis will compile CUDA kernels (~5–10 min). ***")
        print("  *** Subsequent runs are cached and start in seconds.              ***")
    if not args.no_viewer and not args.no_keyboard:
        print("-" * 92)
        print("  Keys: W/S drive  A/D spin  Space stop")
        print("        1..4 select joint   [ / ] jog joint  O/C gripper  Z stow")
        print("        R reset pose   ESC/Q quit")
    print("=" * 92)
    print("  step |  sim_t  |  pose (x, y, yaw_deg) | pitch/roll_deg | v_user/v_meas | tauL/tauR | SoC")
    print("-" * 92)


def print_telemetry(
    engine: GenesisPhysicsEngine,
    step: int,
    sim_t: float,
    v_user: float,
    v_meas: float,
    tau_left: float,
    tau_right: float,
    pitch: float,
    roll: float,
    yaw: float,
    soc: float,
) -> None:
    pos, _ = engine.get_body_pose(ROVER_NAME)
    print(
        f"  {step:5d} | {sim_t:6.2f} | "
        f"({float(pos[0]):+5.2f},{float(pos[1]):+5.2f},{math.degrees(yaw):+6.1f}) | "
        f"{math.degrees(pitch):+6.1f}/{math.degrees(roll):+6.1f} | "
        f"{v_user:+5.2f}/{v_meas:+5.2f} | "
        f"{tau_left:+6.1f}/{tau_right:+6.1f} | "
        f"{soc * 100.0:5.1f}%"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    rover_cfg_path = Path(args.rover_config)
    rover_yaml = load_rover_yaml(rover_cfg_path)
    profile = rover_yaml["profiles"][args.profile]
    power_cfg = rover_yaml.get("power", {})
    arm_cfg_yaml = rover_yaml.get("arm", {})

    if int(profile["num_wheels"]) != 2:
        raise SystemExit(
            f"demo_rover_drive.py expects a 2-wheel profile; '{args.profile}' has "
            f"{profile['num_wheels']} wheels. Use --profile two_wheel_diff."
        )

    urdf_xml = build_rover_urdf(rover_yaml, args.profile)
    urdf_path = _urdf_to_tempfile(urdf_xml)

    cfg = make_genesis_config(
        Path(args.physics_config), args.sim_hz, use_gpu=(args.backend == "gpu")
    )

    viewer_options = None
    if not args.no_viewer:
        viewer_options = gs.options.ViewerOptions(
            max_FPS=args.render_hz,
            refresh_rate=args.render_hz,
            run_in_thread=True,
            camera_pos=(2.5, -2.5, 1.6),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=45,
        )

    print_header(args, profile)

    engine = GenesisPhysicsEngine()
    keyboard: Optional[KeyboardPoll] = None
    completed_cleanly = False
    try:
        engine.configure(cfg, show_viewer=not args.no_viewer, viewer_options=viewer_options)

        ground_friction = 1.2
        engine.add_entity(
            "ground", gs.morphs.Plane(), gs.materials.Rigid(friction=ground_friction)
        )

        spawn_z = spawn_height_for(rover_yaml, profile)
        rover_morph = gs.morphs.URDF(file=urdf_path, fixed=False, pos=(0.0, 0.0, spawn_z))
        # Match the rover's contact friction to the regolith ground so the
        # wheels actually get the grip the balance controller assumes.
        engine.add_entity(
            ROVER_NAME, rover_morph, gs.materials.Rigid(friction=ground_friction)
        )

        engine.build_scene()

        drive_config = drive_config_from_profile(profile, DriveType.TWO_WHEEL_DIFF)
        drive = create_drive_system(drive_config)
        drive.attach(engine, ROVER_NAME)

        # Direct-torque actuator over the resolved wheel DOFs. Balancing an
        # inverted pendulum requires torque control, so we drive the wheels
        # ourselves rather than through the drive system's velocity loop.
        wheel_dofs = [int(h.dofs_idx_local) for h in drive._wheels]
        wheel_actuator = WheelTorqueActuator(
            engine.get_entity(ROVER_NAME),
            wheel_dofs,
            wheel_radius_m=float(drive._config.wheel_radius_m),
        )

        solar_cfg, battery_cfg, budget = power_config_from_yaml(power_cfg)
        power = RoverPowerSystem()
        power.initialize(solar_cfg, battery_cfg, budget)

        # Software arm: kinematics + grasp bookkeeping; the URDF chain mirrors it.
        arm = SerialArm()
        arm.configure(
            ArmConfig(
                num_dof=int(arm_cfg_yaml.get("num_dof", 4)),
                joint_limits=[
                    (
                        float(arm_cfg_yaml["joints"][f"joint_{i}"]["lower_limit_rad"]),
                        float(arm_cfg_yaml["joints"][f"joint_{i}"]["upper_limit_rad"]),
                    )
                    for i in range(1, int(arm_cfg_yaml.get("num_dof", 4)) + 1)
                ],
                reach_m=float(arm_cfg_yaml.get("reach_m", 2.0)),
                payload_kg=float(arm_cfg_yaml.get("payload_kg", 5.0)),
                joint_accuracy_deg=0.5,
            ),
            GripperConfig(
                num_fingers=2,
                max_open_m=float(arm_cfg_yaml.get("gripper", {}).get("stroke_m", 0.1)),
                max_force_n=float(arm_cfg_yaml.get("gripper", {}).get("max_grip_force_n", 500.0)),
                compliance_model="linear",
            ),
        )
        # Stow pose: shoulder (joint 2) lifts link 2 vertical, then the elbow
        # (joint 3) folds link 3+ back *over the base* rather than out in front.
        # The earlier [0,-pi/2,+pi/2,0] pose unfolded the elbow forward, parking
        # ~2 kg of arm ~0.7 m ahead of the axle — a large constant pitching
        # moment that forced the balancer into a permanent ~11° back-lean.
        # Folding the elbow back (joint 3 = -pi/2) keeps the arm COM ~0.2 m of
        # the axle so the rover balances near-upright. (Joints clamp at ±pi/2;
        # 1.55 rad stays just inside the limit to avoid limit chatter.)
        arm.set_joint_positions(
            np.array([0.0, -1.55, -1.55, 0.0], dtype=np.float64)
        )

        arm_bridge = ArmBridge(
            engine.get_entity(ROVER_NAME),
            arm,
            num_dof=int(arm_cfg_yaml.get("num_dof", 4)),
            stroke_m=float(arm_cfg_yaml.get("gripper", {}).get("stroke_m", 0.1)),
        )

        # Traction-limited torque cap. On the moon the rover only weighs
        # m*g ≈ 41 N, so each wheel can transmit at most ~mu*(m*g/2)*r ≈ a few
        # N·m before it slips. Commanding the motor's 150 N·m just spins the
        # wheels and the body falls — so cap the balance torque at the friction
        # cone (with a little headroom) instead of the motor rating.
        try:
            m_total = float(engine.get_entity(ROVER_NAME).get_mass())
        except Exception:
            m_total = 25.55
        g_mag = abs(float(cfg.gravity_vector[2])) or 1.622
        mu_eff = ground_friction
        n_wheels = 2
        tau_slip_per_wheel = mu_eff * (m_total * g_mag / n_wheels) * float(
            drive._config.wheel_radius_m
        )
        # 1.5x headroom: the trailing wheel carries extra normal load under a
        # pitch correction, and brief micro-slip is acceptable.
        tau_cap = min(
            float(drive._config.max_torque_nm), 1.5 * tau_slip_per_wheel
        )
        print(
            f"  Traction    : m={m_total:.1f} kg, g={g_mag:.3f} m/s^2, "
            f"mu={mu_eff:.2f} -> slip ~{tau_slip_per_wheel:.2f} N·m/wheel, "
            f"torque cap {tau_cap:.2f} N·m"
        )
        balance = BalanceController(max_torque_nm=tau_cap)

        keyboard = None
        if not args.no_viewer and not args.no_keyboard:
            keyboard = KeyboardPoll()
            keyboard.start()

        run_loop(
            engine, drive, wheel_actuator, power, arm, arm_bridge, balance,
            keyboard, cfg, args, spawn_z,
        )
        completed_cleanly = True

    except KeyboardInterrupt:
        print("\n  Stopped by user (Ctrl+C).")
        completed_cleanly = True
    finally:
        if keyboard is not None:
            keyboard.stop()
        if args.destroy:
            try:
                engine.teardown()
            except Exception:
                pass
        try:
            os.unlink(urdf_path)
        except OSError:
            pass

    print("\n  Demo complete." if completed_cleanly else "\n  Demo aborted.")
    return 0 if completed_cleanly else 1


def run_loop(
    engine: GenesisPhysicsEngine,
    drive,
    wheel_actuator: WheelTorqueActuator,
    power: RoverPowerSystem,
    arm: SerialArm,
    arm_bridge: ArmBridge,
    balance: BalanceController,
    keyboard: Optional[KeyboardPoll],
    cfg: GenesisConfig,
    args: argparse.Namespace,
    spawn_z: float,
) -> None:
    dt = cfg.timestep
    step_count = 0
    wall_start = time.perf_counter()
    next_report_wall = wall_start
    pace_to_real_time = not args.no_viewer

    v_user = 0.0
    w_user = 0.0
    prev_yaw: Optional[float] = None  # for finite-difference yaw rate
    self_test_arm_phase = 0  # advances as scripted arm moves fire
    selected_joint = 0  # 0..3 index into arm joint positions
    jog_step = 0.05  # radians per keypress
    v_step = 0.2  # m/s per W/S press
    w_step = 0.3  # rad/s per A/D press

    while True:
        t = engine.get_sim_time()

        # Scripted command sequence: settle, drive forward, stop, spin, stop,
        # then perturb the arm (lift) and re-stow — all while balancing.
        if args.self_test:
            if t < 4.0:
                v_user, w_user = 0.0, 0.0
            elif t < 8.0:
                v_user, w_user = 0.4, 0.0
            elif t < 11.0:
                v_user, w_user = 0.0, 0.0
            elif t < 15.0:
                v_user, w_user = 0.0, 0.7
            elif t < 18.0:
                v_user, w_user = 0.0, 0.0
            else:
                v_user, w_user = 0.0, 0.0
            # Gentle wrist/elbow nudge — proves the arm can move under the
            # balancer. A *large* fast slew exceeds the lunar traction budget
            # (~5 N·m/wheel) and will tip the rover; that is a physical limit
            # of balancing under 1/6 g, not a controller fault.
            if self_test_arm_phase == 0 and t >= 18.0:
                arm.set_joint_positions(
                    np.array([0.0, -1.55, -1.30, -0.20], dtype=np.float64)
                )
                self_test_arm_phase = 1
            elif self_test_arm_phase == 1 and t >= 21.0:
                arm.set_joint_positions(
                    np.array([0.0, -1.55, -1.55, 0.0], dtype=np.float64)
                )
                self_test_arm_phase = 2

        # 1. Drain keyboard events.
        if keyboard is not None:
            for ch in keyboard.pop_all():
                lower = ch.lower()
                if ch == "\x1b" or lower == "q":  # ESC or Q
                    print("\n  Quit key pressed.")
                    return
                elif lower == "w":
                    v_user = min(v_user + v_step, 2.0)
                elif lower == "s":
                    v_user = max(v_user - v_step, -2.0)
                elif lower == "a":
                    w_user = min(w_user + w_step, 2.0)
                elif lower == "d":
                    w_user = max(w_user - w_step, -2.0)
                elif ch == " ":
                    v_user = 0.0
                    w_user = 0.0
                elif ch in "1234":
                    selected_joint = int(ch) - 1
                elif ch == "]":
                    q = np.asarray(arm.get_state().joint_positions, dtype=np.float64).copy()
                    if 0 <= selected_joint < q.size:
                        q[selected_joint] += jog_step
                        arm.set_joint_positions(q)
                elif ch == "[":
                    q = np.asarray(arm.get_state().joint_positions, dtype=np.float64).copy()
                    if 0 <= selected_joint < q.size:
                        q[selected_joint] -= jog_step
                        arm.set_joint_positions(q)
                elif lower == "o":
                    arm.command_gripper(1.0)
                elif lower == "c":
                    arm.command_gripper(0.0)
                elif lower == "z":
                    arm.stow()
                elif lower == "r":
                    _reset_pose(engine, spawn_z)
                    balance.reset()
                    prev_yaw = None
                    v_user = 0.0
                    w_user = 0.0

        # 2. Read body attitude and wheel state → balance torque.
        _, quat = engine.get_body_pose(ROVER_NAME)
        quat_arr = np.asarray(quat).flatten()
        roll, pitch, yaw = _euler_from_quat_wxyz(quat_arr)

        # Yaw rate by finite difference (wrapped).
        if prev_yaw is None:
            yaw_rate = 0.0
        else:
            dyaw = math.atan2(math.sin(yaw - prev_yaw), math.cos(yaw - prev_yaw))
            yaw_rate = dyaw / dt if dt > 0.0 else 0.0
        prev_yaw = yaw

        # Forward ground velocity from the wheel encoders.
        body_v = wheel_actuator.body_velocity()

        tau_left, tau_right = balance.step(
            pitch_rad=pitch,
            body_v_mps=body_v,
            yaw_rate_radps=yaw_rate,
            v_ref_mps=v_user,
            w_ref_radps=w_user,
            dt=dt,
        )

        # 3. Arm: update software state, push into Genesis.
        arm.update(dt)
        arm_bridge.sync()

        # 4. Apply wheel torque + step physics.
        wheel_actuator.apply(tau_left, tau_right)
        engine.step(dt, render=not args.no_viewer)

        # 5. Power accounting.
        driving = abs(v_user) > 0.05 or abs(w_user) > 0.05
        power.step(
            dt,
            sun_elevation=args.sun_elevation_deg,
            subsystem_states={
                "mobility": "active" if driving else "idle",
                "compute": "active",
                "communication": "idle",
                "cameras": "idle",
                "lidar": "idle",
                "imu": "idle",
                "manipulator": "active",
            },
        )

        step_count += 1

        # Real-time pacing: sleep to the next physics deadline, but never burst.
        if pace_to_real_time:
            target_wall = wall_start + step_count * dt
            sleep_for = target_wall - time.perf_counter()
            if sleep_for > 0.0005:
                time.sleep(sleep_for)

        now = time.perf_counter()
        if step_count <= 3 or now >= next_report_wall:
            print_telemetry(
                engine,
                step=step_count,
                sim_t=t,
                v_user=v_user,
                v_meas=body_v,
                tau_left=tau_left,
                tau_right=tau_right,
                pitch=pitch,
                roll=roll,
                yaw=yaw,
                soc=power.get_battery_soc(),
            )
            next_report_wall = now + 0.5  # 2 Hz console telemetry

        # Safety: if the rover falls flat, call it out — balance controller failed.
        if abs(pitch) > math.radians(70.0) or abs(roll) > math.radians(70.0):
            print(f"\n  *** Rover fell over (pitch={math.degrees(pitch):.1f} deg, "
                  f"roll={math.degrees(roll):.1f} deg) - press R to reset.")
            # Continue running; let the user recover.

        if args.steps and step_count >= args.steps:
            return


def _reset_pose(engine: GenesisPhysicsEngine, spawn_z: float) -> None:
    """Respawn the rover upright at the origin with zero velocity."""
    try:
        engine.set_body_pose(
            ROVER_NAME,
            position=(0.0, 0.0, spawn_z),
            quaternion=(1.0, 0.0, 0.0, 0.0),  # wxyz: identity
        )
    except Exception as exc:
        print(f"  reset_pose failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _urdf_to_tempfile(urdf_xml: str) -> str:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(urdf_xml)
    tmp.flush()
    tmp.close()
    return tmp.name


if __name__ == "__main__":
    raise SystemExit(main())
