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
    """Self-balancing PD controller that biases drive velocity by body pitch.

    The output is a forward velocity correction (m/s) added on top of the
    user's drive command. The velocity then enters the drive system's inner
    PD loop (Kv = max_torque_nm) which applies wheel torque proportional to
    the velocity error, producing the reaction torque that drives the
    inverted-pendulum body back upright.
    """

    def __init__(
        self,
        kp: float = 18.0,
        kd: float = 2.5,
        k_vel: float = 1.0,
        max_bias_mps: float = 3.0,
    ) -> None:
        self.kp = kp
        self.kd = kd
        self.k_vel = k_vel
        self.max_bias = max_bias_mps
        self._prev_pitch: Optional[float] = None

    def reset(self) -> None:
        self._prev_pitch = None

    def step(self, pitch_rad: float, body_vx_mps: float, dt: float) -> float:
        """Compute velocity bias.

        Positive pitch = nose down (forward lean). Drive the wheels in the
        direction of the fall so the contact point moves under the COM. The
        ``body_vx_mps`` term suppresses drift: when the rover is coasting but
        the user commanded zero, we bleed the bias back toward zero.
        """
        if self._prev_pitch is None or dt <= 0.0:
            pitch_rate = 0.0
        else:
            pitch_rate = (pitch_rad - self._prev_pitch) / dt
        self._prev_pitch = pitch_rad
        bias = (
            self.kp * pitch_rad
            + self.kd * pitch_rate
            - self.k_vel * body_vx_mps
        )
        return float(max(-self.max_bias, min(self.max_bias, bias)))


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
    print("  step |  sim_t  |  pose (x, y, yaw_deg) | pitch/roll_deg | v_user/v_bal | v/w cmd | SoC")
    print("-" * 92)


def print_telemetry(
    engine: GenesisPhysicsEngine,
    step: int,
    sim_t: float,
    v_user: float,
    v_bal: float,
    v_cmd: float,
    w_cmd: float,
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
        f"{v_user:+4.2f}/{v_bal:+4.2f} | "
        f"{v_cmd:+4.2f}/{w_cmd:+4.2f} | "
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

        engine.add_entity("ground", gs.morphs.Plane(), gs.materials.Rigid(friction=1.2))

        spawn_z = spawn_height_for(rover_yaml, profile)
        rover_morph = gs.morphs.URDF(file=urdf_path, fixed=False, pos=(0.0, 0.0, spawn_z))
        engine.add_entity(ROVER_NAME, rover_morph, gs.materials.Rigid())

        engine.build_scene()

        drive_config = drive_config_from_profile(profile, DriveType.TWO_WHEEL_DIFF)
        drive = create_drive_system(drive_config)
        drive.attach(engine, ROVER_NAME)

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
        # Stow pose: arm folded up over the chassis so the mass sits close to
        # the wheel axle. Joint 2 lifts the shoulder, joint 3 folds the elbow
        # back — the tip ends up just above the chassis top, well within the
        # pitch envelope the PD balance loop can correct.
        arm.set_joint_positions(
            np.array([0.0, -math.pi / 2.0, math.pi / 2.0, 0.0], dtype=np.float64)
        )

        arm_bridge = ArmBridge(
            engine.get_entity(ROVER_NAME),
            arm,
            num_dof=int(arm_cfg_yaml.get("num_dof", 4)),
            stroke_m=float(arm_cfg_yaml.get("gripper", {}).get("stroke_m", 0.1)),
        )

        balance = BalanceController(kp=6.0, kd=0.8, max_bias_mps=2.5)

        keyboard = None
        if not args.no_viewer and not args.no_keyboard:
            keyboard = KeyboardPoll()
            keyboard.start()

        run_loop(engine, drive, power, arm, arm_bridge, balance, keyboard, cfg, args, spawn_z)
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
    selected_joint = 0  # 0..3 index into arm joint positions
    jog_step = 0.05  # radians per keypress
    v_step = 0.2  # m/s per W/S press
    w_step = 0.3  # rad/s per A/D press

    while True:
        t = engine.get_sim_time()

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
                    v_user = 0.0
                    w_user = 0.0

        # 2. Read body pose → balance correction.
        _, quat = engine.get_body_pose(ROVER_NAME)
        quat_arr = np.asarray(quat).flatten()
        roll, pitch, yaw = _euler_from_quat_wxyz(quat_arr)

        # Body forward velocity from wheel encoders. Average wheel angular
        # speed times radius, minus the user's commanded velocity so the
        # drift term only sees unintended motion.
        wheel_states = drive.get_wheel_states()
        if len(wheel_states) >= 2:
            wheel_radius = float(drive._config.wheel_radius_m)
            avg_omega = 0.5 * (wheel_states[0].angular_velocity + wheel_states[1].angular_velocity)
            body_vx = wheel_radius * avg_omega - v_user
        else:
            body_vx = 0.0

        v_bias = balance.step(pitch, body_vx, dt)

        v_cmd = v_user + v_bias
        w_cmd = w_user
        drive.command(DriveCommand(linear_velocity_mps=v_cmd, angular_velocity_radps=w_cmd))

        # 3. Arm: update software state, push into Genesis.
        arm.update(dt)
        arm_bridge.sync()

        # 4. Apply drive to physics + step.
        drive.update(dt)
        engine.step(dt, render=not args.no_viewer)

        # 5. Power accounting.
        power.step(
            dt,
            sun_elevation=args.sun_elevation_deg,
            subsystem_states={
                "mobility": "active" if abs(v_cmd) > 0.05 or abs(w_cmd) > 0.05 else "idle",
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
                v_bal=v_bias,
                v_cmd=v_cmd,
                w_cmd=w_cmd,
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
