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
Motion is on the **arrow keys**, not WASD. Keys are captured by the Genesis
viewer window itself (the built-in viewer hotkeys are disabled so nothing
shadows them), so **click the 3-D window once to give it keyboard focus** —
mouse orbit/zoom works regardless, but the drive keys need window focus. A
terminal key poller stays active as a fallback when the console is focused.

    Up / Down    forward / reverse velocity target  (m/s)
    Left / Right spin left / right (yaw rate)       (rad/s)
    Space        zero the drive command
    1..4         select arm joint to jog
    [ / ]        jog selected joint − / +           (rad)
    O / C        open / close gripper
    X            stow arm to neutral pose
    R            reset rover pose (respawn upright)
    ESC / Q      quit cleanly

The viewer scene is a finite lunar heightfield (gentle near the spawn pad so
the balancer stays upright), with scattered rocks, the moonbase, a trailing
power/data cable that follows the rover, and wheel-rut decals whose depth comes
from the analytic terramechanics model. Pass ``--flat-ground`` for the legacy
bare infinite plane.

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
from moon_rover.antenna import (  # noqa: E402
    AntennaConfig,
    AntennaState,
    DeployableAntennaUnit,
)
from moon_rover.cable import CableConfig, RigidLinkCableSystem  # noqa: E402
from moon_rover.environment.lighting import LunarSolarSystem, SolarConfig  # noqa: E402
from moon_rover.environment.terrain import (  # noqa: E402
    LunarTerrainGenerator,
    TerrainConfig,
)
from moon_rover.environment.thermal import (  # noqa: E402
    ComponentThermal,
    LunarThermalModel,
    ThermalConfig,
)
from moon_rover.moonbase import LunarMoonbase, MoonbaseConfig  # noqa: E402
from moon_rover.rover.drive import (  # noqa: E402
    DriveCommand,
    DriveType,
    create_drive_system,
    default_analytic_terramechanics,
    drive_config_from_profile,
)
from moon_rover.rover.manipulator.arm import ArmConfig, GripperConfig  # noqa: E402
from moon_rover.rover.manipulator.serial_arm import SerialArm  # noqa: E402
from moon_rover.rover.power import (  # noqa: E402
    RoverPowerSystem,
    power_config_from_yaml,
)
from moon_rover.sensors import (  # noqa: E402
    BeaconConfig,
    CameraConfig,
    EncoderConfig,
    FTConfig,
    GenesisForceTorqueSensor,
    GenesisIMUSensor,
    GenesisLiDARScanner,
    GenesisSunSensor,
    GenesisWheelEncoder,
    IMUConfig,
    LiDARConfig,
    RaycastStereoCamera,
    SunSensorConfig,
    TrilaterationBeaconNetwork,
)
from moon_rover.navigation.control.mpc import MPCConfig, MPCController  # noqa: E402
from moon_rover.navigation.perception.mapping import (  # noqa: E402
    GridOccupancyMap,
    GridTraversabilityMap,
    SpatialCableMap,
)
from moon_rover.navigation.planning.path_planner import (  # noqa: E402
    AStarDStarPathPlanner,
    PlannedPath,
    PlannerConfig,
)

ROVER_NAME = "demo_rover"

# Visible lunar-world scene (viewer only). Entities cannot be added to a Genesis
# scene after build_scene(), so the cable and rut decals are fixed-size pools
# allocated up front and then repositioned every physics step.
WORLD_SIZE_M = 30.0
WORLD_RESOLUTION = 80
WORLD_FLAT_RADIUS_M = 4.0          # fully flat spawn pad (balancer-safe)
WORLD_BLEND_RADIUS_M = 9.0         # height blends to full terrain by here
WORLD_SEED = 4242
CABLE_LINK_COUNT = 28
RUT_DECAL_COUNT = 80
RUT_DROP_SPACING_M = 0.25          # advance distance between rut stamps
WORLD_VISUAL_HZ = 30.0             # cable re-draping rate (decoupled from sim Hz)

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
    parser.add_argument("--capability-demo", action="store_true",
                        help="Layer current subsystem telemetry onto the "
                             "2-wheel balance bot demo.")
    parser.add_argument("--flat-ground", action="store_true",
                        help="Use the legacy bare infinite ground plane instead "
                             "of the lunar heightfield / rocks / cable scene. "
                             "Headless runs always use the flat plane.")
    parser.add_argument("--navigation", action="store_true",
                        help="Drive the rover autonomously with the full "
                             "navigation pipeline: A* path planner picks a tour "
                             "across the lunar scene, MPC tracks each leg, and "
                             "cable tension + terrain slope feed the MPC speed "
                             "limit so the rover visibly struggles when the "
                             "cable pulls taut. Implies --capability-demo.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rover_yaml(path: Path) -> dict:
    return load_yaml(path)


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

    # Arrow keys are reported as multi-byte sequences that differ per OS.
    # Both loops normalise them to the tokens UP / DOWN / LEFT / RIGHT so the
    # control handler can treat an arrow press as a single logical key.
    _WIN_ARROWS = {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}
    _POSIX_ARROWS = {"A": "UP", "B": "DOWN", "D": "LEFT", "C": "RIGHT"}

    # -- Windows ---------------------------------------------------------
    def _win_loop(self) -> None:
        import msvcrt
        while not self._stop.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                # \x00 / \xe0 prefix a function/arrow key; the next read is the
                # scan code (H/P/K/M for the arrows).
                if ch in ("\x00", "\xe0"):
                    code = msvcrt.getwch()
                    ch = self._WIN_ARROWS.get(code, "")
                    if not ch:
                        continue
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
            if not r:
                continue
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                # Could be a bare ESC (quit) or a CSI arrow sequence
                # ESC '[' 'A'..'D'. Peek without blocking; if nothing
                # follows promptly it was a real ESC.
                r2, _, _ = select.select([sys.stdin], [], [], 0.02)
                if r2 and sys.stdin.read(1) == "[":
                    code = sys.stdin.read(1)
                    ch = self._POSIX_ARROWS.get(code, "")
                    if not ch:
                        continue
            with self._lock:
                self._queue.append(ch)


class ViewerControls:
    """Drive the rover from the Genesis viewer window itself.

    ``KeyboardPoll`` only sees keystrokes when the *terminal* has focus, but
    while flying the camera the user is focused on the 3-D window, so the
    arrows never reached the rover. Genesis exposes a viewer keybinding API;
    registering here means the focused viewer window feeds the same control
    tokens ``run_loop`` already understands ("UP", " ", "1", "[", "r", ...).

    Callbacks fire on the viewer thread, so the queue is mutex-guarded exactly
    like :class:`KeyboardPoll`. Motion keys bind on HOLD (press-and-hold to
    keep accelerating); discrete actions bind on PRESS.
    """

    def __init__(self) -> None:
        self._queue: list[str] = []
        self._lock = threading.Lock()

    def _push(self, token: str) -> None:
        with self._lock:
            self._queue.append(token)

    def pop_all(self) -> list[str]:
        with self._lock:
            out = self._queue[:]
            self._queue.clear()
            return out

    def attach(self, viewer) -> bool:
        """Register keybinds on a Genesis viewer. Returns False if unavailable."""
        if viewer is None or not hasattr(viewer, "register_keybinds"):
            return False
        from genesis.vis.keybindings import Key, KeyAction, Keybind

        hold = KeyAction.HOLD
        press = KeyAction.PRESS
        # (name, Key, token, action)
        spec = [
            ("mr_drive_fwd", Key.UP, "UP", hold),
            ("mr_drive_rev", Key.DOWN, "DOWN", hold),
            ("mr_spin_left", Key.LEFT, "LEFT", hold),
            ("mr_spin_right", Key.RIGHT, "RIGHT", hold),
            ("mr_stop", Key.SPACE, " ", press),
            ("mr_joint_1", Key._1, "1", press),
            ("mr_joint_2", Key._2, "2", press),
            ("mr_joint_3", Key._3, "3", press),
            ("mr_joint_4", Key._4, "4", press),
            ("mr_jog_minus", Key.BRACKETLEFT, "[", press),
            ("mr_jog_plus", Key.BRACKETRIGHT, "]", press),
            ("mr_grip_open", Key.O, "o", press),
            ("mr_grip_close", Key.C, "c", press),
            ("mr_arm_stow", Key.X, "x", press),
            ("mr_reset", Key.R, "r", press),
            ("mr_nav_start", Key.F5, "F5", press),
            ("mr_quit", Key.Q, "q", press),
            ("mr_quit_esc", Key.ESCAPE, "\x1b", press),
        ]
        binds = [
            Keybind(name=n, key=k, key_action=a,
                    callback=self._push, args=(tok,))
            for (n, k, tok, a) in spec
        ]
        try:
            viewer.register_keybinds(*binds, overwrite=True)
            return True
        except Exception as exc:  # noqa: BLE001 - keep terminal fallback
            print(f"  Viewer keybind registration failed ({exc}); "
                  "terminal keys only.")
            return False


def _get_viewer(engine: GenesisPhysicsEngine):
    """Best-effort handle on the live Genesis viewer (None if headless)."""
    scene = getattr(engine, "_scene", None)
    return getattr(scene, "viewer", None) if scene is not None else None


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

    def snap(self) -> None:
        """Hard-set the arm/gripper DOFs to the current target *now*.

        The URDF spawns the arm at its zero pose (links straight out in front
        of the axle). If we only ``sync()`` a position target, the PD servo
        sweeps the ~2 kg arm from straight-out to folded over the first
        fraction of a second — a large moving pitch moment that knocks the
        balancer over before it has even settled. Calling this once after
        ``build_scene()`` teleports the joints (and zeros their velocity) so
        the rover *starts* folded and the balancer only ever sees the stowed
        configuration.
        """
        targets = np.asarray(self._arm.get_state().joint_positions, dtype=np.float32)
        if targets.size == self._num_dof:
            try:
                self._entity.set_dofs_position(
                    position=targets, dofs_idx_local=self._dof_local
                )
                self._entity.set_dofs_velocity(
                    velocity=np.zeros(self._num_dof, dtype=np.float32),
                    dofs_idx_local=self._dof_local,
                )
            except (AttributeError, TypeError):
                pass
        if self._gripper_dof_local:
            half = 0.5 * float(self._arm.get_state().gripper_position)
            try:
                self._entity.set_dofs_position(
                    position=np.array([half, half], dtype=np.float32),
                    dofs_idx_local=self._gripper_dof_local,
                )
                self._entity.set_dofs_velocity(
                    velocity=np.zeros(2, dtype=np.float32),
                    dofs_idx_local=self._gripper_dof_local,
                )
            except (AttributeError, TypeError):
                pass
        self.sync()

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
# Current-capabilities showcase layer
# ---------------------------------------------------------------------------


class CapabilityTerrainScene:
    """Small heightfield scene for the analytic sensor showcase."""

    def __init__(self, terrain_output, terrain_size_m: float) -> None:
        self._terrain = terrain_output
        self._size = float(terrain_size_m)
        self._height = np.asarray(terrain_output.height_field, dtype=np.float64)
        self._normals = np.asarray(terrain_output.normal_map, dtype=np.float64)
        self._res = int(self._height.shape[0])

    def get_terrain_height(self, x: float, y: float) -> float:
        x = float(np.clip(x, 0.0, self._size))
        y = float(np.clip(y, 0.0, self._size))
        gx = x / self._size * (self._res - 1) if self._size > 0.0 else 0.0
        gy = y / self._size * (self._res - 1) if self._size > 0.0 else 0.0
        j0 = int(np.clip(math.floor(gx), 0, self._res - 1))
        i0 = int(np.clip(math.floor(gy), 0, self._res - 1))
        j1 = min(j0 + 1, self._res - 1)
        i1 = min(i0 + 1, self._res - 1)
        tx = gx - j0
        ty = gy - i0
        top = self._height[i0, j0] * (1.0 - tx) + self._height[i0, j1] * tx
        bot = self._height[i1, j0] * (1.0 - tx) + self._height[i1, j1] * tx
        return float(top * (1.0 - ty) + bot * ty)

    def get_terrain_normal(self, x: float, y: float) -> np.ndarray:
        x = float(np.clip(x, 0.0, self._size))
        y = float(np.clip(y, 0.0, self._size))
        j = int(np.clip(round(x / self._size * (self._res - 1)), 0, self._res - 1))
        i = int(np.clip(round(y / self._size * (self._res - 1)), 0, self._res - 1))
        n = self._normals[i, j]
        mag = float(np.linalg.norm(n))
        return n / mag if mag > 1e-9 else np.array([0.0, 0.0, 1.0])

    def raycast_batch(
        self, origins: np.ndarray, directions: np.ndarray, max_range: float
    ) -> dict[str, np.ndarray]:
        """Fast terrain hit test for the demo sensors.

        The production sensors can fall back to exact heightfield marching, but
        that is intentionally conservative and too slow for this telemetry
        overlay. Here we intersect each ray against the local terrain surface,
        then do one height correction at the estimated hit point.
        """
        origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
        directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norms, 1e-12)
        n = origins.shape[0]

        distances = np.full(n, np.inf, dtype=np.float64)
        positions = np.full((n, 3), np.nan, dtype=np.float64)
        normals = np.zeros((n, 3), dtype=np.float64)
        dz = directions[:, 2]
        downward = dz < -1e-6
        if not np.any(downward):
            return {
                "distances": distances,
                "positions": positions,
                "normals": normals,
                "hit": np.zeros(n, dtype=bool),
            }

        idx = np.where(downward)[0]
        h0 = np.array(
            [self.get_terrain_height(origins[i, 0], origins[i, 1]) for i in idx],
            dtype=np.float64,
        )
        t = (h0 - origins[idx, 2]) / dz[idx]
        p = origins[idx] + directions[idx] * t[:, None]
        h1 = np.array(
            [self.get_terrain_height(p[k, 0], p[k, 1]) for k in range(p.shape[0])],
            dtype=np.float64,
        )
        t = (h1 - origins[idx, 2]) / dz[idx]
        hit = (t > 0.0) & (t <= max_range)
        hit_idx = idx[hit]
        if hit_idx.size:
            distances[hit_idx] = t[hit]
            positions[hit_idx] = origins[hit_idx] + directions[hit_idx] * t[hit, None]
            positions[hit_idx, 2] = h1[hit]
            for j in hit_idx:
                normals[j] = self.get_terrain_normal(positions[j, 0], positions[j, 1])
        return {
            "distances": distances,
            "positions": positions,
            "normals": normals,
            "hit": np.isfinite(distances),
        }


class CapabilityShowcase:
    """Low-rate subsystem telemetry layered onto the balance-bot physics run."""

    def __init__(self, args: argparse.Namespace, rover_yaml: dict, profile: dict) -> None:
        self._args = args
        self._profile = profile
        self._terrain_size_m = 32.0
        self._terrain_origin = np.array([6.0, 6.0, 0.0], dtype=np.float64)
        self._last_sensor_t = -1.0e9
        self._last_report_t = -1.0e9
        self._last_soil_t = -1.0e9
        self._sensor_period_s = 1.0
        self._report_period_s = 1.0
        self._soil_period_s = 0.25
        self._path_length_m = 0.0
        self._prev_body_v = 0.0
        self._antenna_phase = 0
        self._antenna_registered = False

        terrain_cfg = TerrainConfig(
            seed=12345,
            size_m=self._terrain_size_m,
            fBm_octaves=4,
            fBm_amplitude=0.35,
            crater_params={
                "count": 6,
                "min_radius_m": 0.5,
                "max_radius_m": 2.0,
                "depth_ratio": 0.25,
            },
            rock_density=0.012,
            rille_enabled=True,
            moonbase_position=(6.0, 6.0, 0.0),
            resolution=64,
        )
        self._terrain = LunarTerrainGenerator(
            max_traversable_slope_deg=25.0,
            rock_clearance_m=0.35,
            moonbase_pad_radius_m=2.0,
        ).generate(terrain_cfg)
        self._scene = CapabilityTerrainScene(self._terrain, self._terrain_size_m)
        self._nav_pct = 100.0 * float(np.mean(self._terrain.nav_mesh > 0))

        self._solar = LunarSolarSystem(
            terrain_size_m=self._terrain_size_m,
            terrain_normal_map=self._terrain.normal_map,
        )
        self._solar.configure(
            SolarConfig(
                elevation_deg=float(np.clip(args.sun_elevation_deg, 0.0, 90.0)),
                azimuth_deg=135.0,
                lunar_day_cycle=False,
            )
        )

        self._thermal = LunarThermalModel()
        self._thermal.initialize(
            ThermalConfig(
                component_models={
                    "battery_main": ComponentThermal(
                        operating_range=(0.0, 45.0),
                        survival_range=(-20.0, 60.0),
                        thermal_mass=14000.0,
                        heat_generation=8.0,
                        radiative_area=0.18,
                        current_temp=20.0,
                    ),
                    "motor_pair": ComponentThermal(
                        operating_range=(-40.0, 80.0),
                        survival_range=(-60.0, 125.0),
                        thermal_mass=4500.0,
                        heat_generation=24.0,
                        radiative_area=0.22,
                        current_temp=20.0,
                    ),
                }
            )
        )

        self._lidar = GenesisLiDARScanner()
        self._lidar.configure(
            LiDARConfig(
                num_channels=6,
                h_resolution_deg=20.0,
                elevation_range_deg=(-25.0, 10.0),
                max_range_m=12.0,
                range_noise_sigma_m=0.01,
                intensity_noise_sigma=0.02,
                rotation_rate_hz=10.0,
                min_range_m=0.25,
                max_returns=1,
                dropout_probability=0.02,
                seed=11,
            )
        )
        self._camera = RaycastStereoCamera()
        self._camera.configure(
            CameraConfig(
                resolution=(40, 30),
                baseline_m=0.12,
                fov_h_deg=60.0,
                fov_v_deg=45.0,
                focal_length_px=38.0,
                frame_rate_hz=10.0,
                depth_range_m=(0.25, 12.0),
                depth_noise_sigma=0.02,
                seed=12,
            )
        )
        self._imu = GenesisIMUSensor()
        self._imu.configure(
            IMUConfig(
                update_rate_hz=max(float(args.sim_hz), 1.0),
                gyro_noise_sigma=0.002,
                accel_noise_sigma=0.03,
                gyro_bias_drift_deg_hr=0.5,
                seed=13,
            )
        )
        self._encoder = GenesisWheelEncoder()
        self._encoder.configure(
            EncoderConfig(
                counts_per_rev=2048,
                update_rate_hz=max(float(args.sim_hz), 1.0),
                seed=14,
            )
        )
        self._sun_sensor = GenesisSunSensor()
        self._sun_sensor.configure(
            SunSensorConfig(accuracy_deg=0.5, update_rate_hz=1.0, seed=15)
        )
        self._force_torque = GenesisForceTorqueSensor()
        self._force_torque.configure(
            FTConfig(
                force_range_n=500.0,
                torque_range_nm=50.0,
                resolution_force=0.1,
                resolution_torque=0.01,
                update_rate_hz=50.0,
                seed=16,
            )
        )

        self._moonbase = LunarMoonbase()
        self._moonbase.initialize(
            MoonbaseConfig(
                habitat_dims_m=(4.0, 3.0, 2.5),
                solar_array_config=None,
                power_bus_voltage=48.0,
                comm_tower_height_m=6.0,
                num_docking_bays=2,
                charge_rate_w=500.0,
                num_cable_reels=3,
                num_antennas=6,
                landing_pad_radius_m=8.0,
            ),
            self._scene,
        )
        self._moonbase.request_cable_reel(ROVER_NAME)
        self._moonbase.request_antenna(ROVER_NAME)
        self._moonbase.request_antenna(ROVER_NAME)

        self._beacons = TrilaterationBeaconNetwork(seed=17)
        self._beacons.add_beacon("moonbase_primary", self._moonbase.get_primary_beacon())
        for beacon_id, xyz in {
            "pad_north": (6.0, 12.0, 1.0),
            "pad_south": (6.0, 0.0, 1.0),
            "pad_east": (12.0, 6.0, 1.0),
            "pad_west": (0.0, 6.0, 1.0),
            "pad_high": (10.0, 10.0, 7.0),
        }.items():
            self._beacons.add_beacon(
                beacon_id,
                BeaconConfig(
                    position_xyz=np.array(xyz, dtype=np.float64),
                    signal_range_m=80.0,
                    power_w=12.0,
                    noise_sigma_m=0.08,
                ),
            )

        mission_cable = (load_yaml(_PROJECT_ROOT / "configs" / "mission.yaml")).get(
            "cable", {}
        )
        cable_len = float(mission_cable.get("length_m", 60.0))
        cable_diam = float(mission_cable.get("diameter_mm", 10.0)) / 1000.0
        self._cable_link_m = 1.0
        self._cable = RigidLinkCableSystem()
        self._cable.initialize(
            CableConfig(
                link_length_m=self._cable_link_m,
                link_diameter_m=cable_diam,
                link_mass_kg=float(mission_cable.get("linear_density_kg_m", 0.15)),
                total_length_m=min(cable_len, 40.0),
                joint_damping=0.4,
                joint_stiffness=80.0,
                terrain_friction=0.55,
                max_tension_n=float(mission_cable.get("max_tension_n", 500.0)),
                bend_radius_min_m=0.12,
                voltage_dc=48.0,
                resistance_per_m=float(
                    mission_cable.get("power", {}).get("resistance_per_m_ohms", 0.005)
                ),
            ),
            self._scene,
        )

        self._antenna_cfg = AntennaConfig(
            base_plate_m=(0.45, 0.45, 0.08),
            base_mass_kg=2.0,
            mast_height_m=2.5,
            mast_radius_m=0.035,
            mast_mass_kg=1.2,
            dish_diameter_m=0.75,
            dish_mass_kg=1.0,
            connector_mass_kg=0.3,
            total_mass_kg=4.5,
        )
        self._antenna = DeployableAntennaUnit(self._antenna_cfg, self._scene)

        self._terramechanics = default_analytic_terramechanics(
            terrain=self._terrain,
            terrain_size_m=self._terrain_size_m,
            wheel_radius_m=float(profile["wheel_radius_m"]),
            wheel_width_m=0.08,
        )

        self._lidar_points = 0
        self._depth_valid_pct = 0.0
        self._gps_error_m = float("inf")
        self._gdop = float("inf")
        self._visible_beacons = 0
        self._rut_depth_m = 0.0
        self._traction_scale = 1.0
        self._imu_yaw_rate = 0.0
        self._encoder_left = 0
        self._ft_force_n = 0.0
        self._sun_valid = False

    def print_initial_status(self) -> None:
        inv = self._moonbase.get_inventory()
        print("-" * 92)
        print("  Capability demo: terrain/regolith, sensors, cable, antenna, moonbase, power/thermal")
        print(
            f"  Terrain     : {self._terrain_size_m:.0f} m field, "
            f"{len(self._terrain.crater_list)} craters, "
            f"{len(self._terrain.rock_positions)} rocks, "
            f"{self._nav_pct:.1f}% nav cells"
        )
        print(
            f"  Moonbase    : primary beacon online, depot assigned "
            f"{len(inv.assigned_items.get(ROVER_NAME, []))} items to rover"
        )
        print("-" * 92)

    @property
    def sun_elevation_deg(self) -> float:
        return float(np.clip(self._solar.get_sun_elevation_deg(), 0.0, 90.0))

    def pre_power_step(self, sim_t: float, dt: float, power: RoverPowerSystem) -> None:
        self._solar.update(max(0.0, sim_t))
        self._thermal.step(dt, self.sun_elevation_deg)
        power.set_battery_temperature(
            self._thermal.get_component_temp("battery_main")
        )

    def subsystem_states(self, driving: bool) -> dict[str, str]:
        active = "active"
        idle = "idle"
        return {
            "drive_motors": active if driving else idle,
            "compute": active,
            "comms": active,
            "cameras": active,
            "lidar": active,
            "imu": active,
            "manipulator": active,
            "heating": idle,
        }

    def post_physics_step(
        self,
        *,
        engine: GenesisPhysicsEngine,
        wheel_actuator: WheelTorqueActuator,
        power_state,
        sim_t: float,
        dt: float,
        body_v: float,
        yaw_rate: float,
        roll: float,
        pitch: float,
        yaw: float,
        driving: bool,
    ) -> None:
        terrain_pos, sensor_pose = self._terrain_pose(engine)
        wheel_omega = wheel_actuator.wheel_omega()

        self._path_length_m += abs(body_v) * max(dt, 0.0)
        self._update_cable(terrain_pos, dt, abs(body_v), power_state)
        self._update_antenna(sim_t, terrain_pos)

        if sim_t - self._last_soil_t >= self._soil_period_s:
            self._last_soil_t = sim_t
            self._update_terramechanics(terrain_pos, body_v, wheel_omega)

        if sim_t - self._last_sensor_t >= self._sensor_period_s:
            self._last_sensor_t = sim_t
            self._sample_sensors(
                terrain_pos,
                sensor_pose,
                wheel_omega,
                body_v,
                yaw_rate,
                roll,
                pitch,
                yaw,
                driving,
                dt,
            )

        if sim_t - self._last_report_t >= self._report_period_s:
            self._last_report_t = sim_t
            self._print_status(power_state)

    def _terrain_pose(
        self, engine: GenesisPhysicsEngine
    ) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = engine.get_body_pose(ROVER_NAME)
        pos_arr = np.asarray(pos, dtype=np.float64).reshape(3)
        quat_arr = np.asarray(quat, dtype=np.float64).reshape(4)
        mapped_xy = self._terrain_origin[:2] + pos_arr[:2]
        ground_z = self._scene.get_terrain_height(mapped_xy[0], mapped_xy[1])
        terrain_pos = np.array([mapped_xy[0], mapped_xy[1], ground_z], dtype=np.float64)
        sensor_pose = np.array(
            [
                terrain_pos[0],
                terrain_pos[1],
                ground_z + 0.75,
                quat_arr[0],
                quat_arr[1],
                quat_arr[2],
                quat_arr[3],
            ],
            dtype=np.float64,
        )
        return terrain_pos, sensor_pose

    def _update_cable(
        self,
        terrain_pos: np.ndarray,
        dt: float,
        speed_mps: float,
        power_state,
    ) -> None:
        states = self._cable.get_link_states()
        active = sum(1 for link in states if link.active)
        target_active = min(
            len(states),
            int(self._path_length_m / self._cable_link_m) + 1,
        )
        while active < target_active:
            if not self._cable.activate_next_link(terrain_pos):
                break
            active += 1
        self._cable.command_spool(speed_mps if speed_mps > 0.02 else 0.0)
        if hasattr(self._cable, "set_electrical_load"):
            current = max(0.0, float(power_state.total_draw_w)) / 48.0
            self._cable.set_electrical_load(current)
        self._cable.step(dt)

    def _update_antenna(self, sim_t: float, terrain_pos: np.ndarray) -> None:
        if self._antenna_phase == 0 and sim_t >= 2.0:
            self._antenna.transition(AntennaState.GRIPPED)
            self._antenna_phase = 1
        elif self._antenna_phase == 1 and sim_t >= 3.0:
            self._antenna.transition(AntennaState.CARRIED)
            self._antenna_phase = 2
        elif self._antenna_phase == 2 and sim_t >= 4.5:
            place_xy = terrain_pos[:2] + np.array([1.0, 0.35])
            self._antenna.set_placement(
                place_xy,
                tilt_deg=2.0,
                base_contact_corners=4,
                position_error_m=0.12,
                connector_engaged=False,
            )
            self._antenna.transition(AntennaState.PLACED)
            self._antenna_phase = 3
        elif self._antenna_phase == 3 and sim_t >= 5.5:
            self._antenna.transition(AntennaState.DEPLOYED)
            self._antenna_phase = 4
        elif self._antenna_phase == 4 and sim_t >= 6.5:
            self._antenna.set_connector_engaged(True)
            self._antenna.transition(AntennaState.ACTIVE)
            self._antenna_phase = 5

        if (
            self._antenna.get_state() is AntennaState.ACTIVE
            and not self._antenna_registered
        ):
            beacon = self._antenna.get_beacon_config()
            if beacon is not None:
                self._beacons.add_beacon("deployed_antenna_1", beacon)
                self._antenna_registered = True

    def _update_terramechanics(
        self,
        terrain_pos: np.ndarray,
        body_v: float,
        wheel_omega: tuple[float, float],
    ) -> None:
        track = float(self._profile["track_width_m"])
        radius = float(self._profile["wheel_radius_m"])
        mass = float(self._profile.get("mass_kg", 20.0)) + self._antenna_cfg.total_mass_kg
        wheel_load = mass * 1.622 / 2.0
        cable_tension = self._cable.get_spool_state().tension_n
        states = []
        for side, omega in ((1.0, wheel_omega[0]), (-1.0, wheel_omega[1])):
            contact = terrain_pos + np.array([0.0, side * track * 0.5, 0.0])
            contact[0] = float(np.clip(contact[0], 0.0, self._terrain_size_m))
            contact[1] = float(np.clip(contact[1], 0.0, self._terrain_size_m))
            states.append(
                self._terramechanics.update_wheel(
                    contact,
                    wheel_angular_vel=omega,
                    ground_velocity=abs(body_v),
                    wheel_load_n=wheel_load,
                    contact_radius_m=max(0.04, radius * 0.35),
                    cable_tension_n=cable_tension,
                )
            )
        self._rut_depth_m = max(s.rut_depth for s in states)
        self._traction_scale = min(s.traction_scale for s in states)

    def _sample_sensors(
        self,
        terrain_pos: np.ndarray,
        sensor_pose: np.ndarray,
        wheel_omega: tuple[float, float],
        body_v: float,
        yaw_rate: float,
        roll: float,
        pitch: float,
        yaw: float,
        driving: bool,
        dt: float,
    ) -> None:
        cloud = self._lidar.scan(self._scene, sensor_pose)
        dust_density = 6.0 if driving else 2.0
        cloud = self._lidar.apply_dust_interference(dust_density, cloud)
        self._lidar_points = int(cloud.points.shape[0])

        frame = self._camera.capture(self._scene, sensor_pose)
        self._depth_valid_pct = 100.0 * float(np.isfinite(frame.depth_map).mean())

        fix = self._beacons.compute_fix(terrain_pos)
        self._visible_beacons = len(self._beacons.get_visible_beacons(terrain_pos))
        if fix is None:
            self._gps_error_m = float("inf")
            self._gdop = float("inf")
        else:
            self._gps_error_m = float(np.linalg.norm(fix.position_xyz - terrain_pos))
            self._gdop = float(fix.gdop)

        accel_x = (body_v - self._prev_body_v) / dt if dt > 0.0 else 0.0
        self._prev_body_v = body_v
        imu = self._imu.read(
            np.array([accel_x, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, yaw_rate], dtype=np.float64),
        )
        self._imu_yaw_rate = float(imu.gyro_xyz[2])
        enc = self._encoder.read([wheel_omega[0], wheel_omega[1]])
        self._encoder_left = int(enc.counts[0]) if enc.counts else 0

        sun = self._sun_sensor.read(
            self._solar.get_sun_azimuth_deg(),
            self.sun_elevation_deg,
            in_shadow=False,
        )
        self._sun_valid = bool(sun.valid)

        held_load = 0.0
        if self._antenna.get_state() in (AntennaState.GRIPPED, AntennaState.CARRIED):
            held_load = self._antenna_cfg.total_mass_kg * 1.622
        ft = self._force_torque.read(
            np.array(
                [
                    0.2 * abs(math.sin(yaw)),
                    0.1 * abs(math.sin(roll)),
                    held_load + 0.2 * abs(math.sin(pitch)),
                    0.02,
                    0.01,
                    0.0,
                ],
                dtype=np.float64,
            )
        )
        self._ft_force_n = float(np.linalg.norm(ft.force_xyz))

    def _print_status(self, power_state) -> None:
        cable_states = self._cable.get_link_states()
        active_links = sum(1 for link in cable_states if link.active)
        deployed_m = active_links * self._cable_link_m
        spool = self._cable.get_spool_state()
        elec = self._cable.get_electrical_state()
        ant_state = self._antenna.get_state().value
        if self._antenna.get_state() in (
            AntennaState.STORED,
            AntennaState.GRIPPED,
            AntennaState.CARRIED,
        ):
            ant_quality = "pending"
        else:
            ant_quality = self._antenna.evaluate_deployment().status
        gps = "inf" if not np.isfinite(self._gps_error_m) else f"{self._gps_error_m:.2f}"
        gdop = "inf" if not np.isfinite(self._gdop) else f"{self._gdop:.1f}"
        batt_t = self._thermal.get_component_temp("battery_main")
        motor_t = self._thermal.get_component_temp("motor_pair")
        print(
            "  cap | "
            f"nav={self._nav_pct:4.1f}% rocks={len(self._terrain.rock_positions):02d} | "
            f"lidar={self._lidar_points:03d} depth={self._depth_valid_pct:4.0f}% "
            f"gps={gps}m/{gdop} b{self._visible_beacons} | "
            f"cable={deployed_m:4.1f}m {spool.tension_n:4.1f}N "
            f"Vdrop={elec['voltage_drop_v']:.2f} | "
            f"rut={self._rut_depth_m:.3f}m tr={self._traction_scale:.2f} | "
            f"ant={ant_state}/{ant_quality} | "
            f"T={batt_t:4.1f}/{motor_t:4.1f}C "
            f"solar={power_state.solar_output_w:5.1f}W "
            f"imuZ={self._imu_yaw_rate:+.2f} encL={self._encoder_left} "
            f"sun={'ok' if self._sun_valid else 'hold'} ft={self._ft_force_n:.1f}N"
        )


def _surface(color: tuple[float, float, float]):
    """Best-effort coloured surface; ``None`` if this Genesis build lacks it."""
    try:
        return gs.surfaces.Default(color=color)
    except Exception:  # noqa: BLE001 - older/headless builds: fall back to default
        return None


def _visual_box(pos, size):
    """A render-only Box (no collider).

    The cable links and rut decals are repositioned every visual tick; if they
    kept colliders, ~100 teleporting kinematic bodies would flood the contact
    solver and shove the balancer over. Visual-only keeps them from perturbing
    the physics at all.
    """
    try:
        return gs.morphs.Box(pos=pos, size=size, collision=False)
    except TypeError:  # very old Genesis without the collision kwarg
        return gs.morphs.Box(pos=pos, size=size)


class LunarWorld:
    """Visible lunar scene: heightfield ground, rocks, moonbase, trailing
    cable, and analytic wheel-rut decals.

    Construction (terrain, props, and the cable/rut entity pools) happens in the
    engine CONSTRUCTION phase via :meth:`construct`. During SIMULATION,
    :meth:`step` slides the pre-allocated cable links along the rover→anchor
    line and stamps recessed rut decals under the wheels, with sinkage taken
    from the analytic terramechanics stack so the grooves track the real soil
    model rather than a cosmetic constant.

    The heightfield is flattened to z=0 inside ``WORLD_FLAT_RADIUS_M`` of the
    spawn pad and blended to full fBm terrain by ``WORLD_BLEND_RADIUS_M`` so the
    flat-tuned balance controller still launches cleanly.
    """

    def __init__(self, profile: dict, sun_elevation_deg: float) -> None:
        self._size = WORLD_SIZE_M
        self._res = WORLD_RESOLUTION
        cx = cy = self._size / 2.0
        self._center = np.array([cx, cy], dtype=np.float64)
        # Anchor the cable at the moonbase, set off the flat pad on a diagonal.
        self._anchor = np.array([cx - 7.0, cy - 7.0], dtype=np.float64)

        crater = {
            "count": 7,
            "min_radius_m": 0.6,
            "max_radius_m": 2.2,
            "depth_ratio": 0.22,
        }
        terrain_cfg = TerrainConfig(
            seed=WORLD_SEED,
            size_m=self._size,
            fBm_octaves=5,
            fBm_amplitude=0.18,           # gentle: a textured surface, not dunes
            crater_params=crater,
            rock_density=0.01,
            rille_enabled=True,
            moonbase_position=(float(self._anchor[0]), float(self._anchor[1]), 0.0),
            resolution=self._res,
        )
        self._terrain = LunarTerrainGenerator(
            max_traversable_slope_deg=25.0,
            rock_clearance_m=0.35,
            moonbase_pad_radius_m=2.0,
        ).generate(terrain_cfg)

        self._height = self._flatten_spawn_pad(
            np.asarray(self._terrain.height_field, dtype=np.float64)
        )
        self._cell = self._size / max(self._res - 1, 1)

        # Analytic soil model — same stack the capability demo uses; here it
        # only drives the visible rut depth.
        self._terramechanics = default_analytic_terramechanics(
            terrain=self._terrain,
            terrain_size_m=self._size,
            wheel_radius_m=float(profile["wheel_radius_m"]),
            wheel_width_m=0.08,
        )
        self._track = float(profile["track_width_m"])
        self._wheel_r = float(profile["wheel_radius_m"])
        self._mass_kg = float(profile.get("mass_kg", 20.0))

        self._rut_idx = 0
        self._last_drop_xy: Optional[np.ndarray] = None
        self._cable_n = CABLE_LINK_COUNT
        self._rut_n = RUT_DECAL_COUNT
        self._link_half_h = 0.025  # half of the cable link box height
        self._vis_accum = 1.0e9    # force a cable redrape on the first step
        self._vis_period = 1.0 / WORLD_VISUAL_HZ

    # -- geometry helpers ------------------------------------------------
    @property
    def center(self) -> tuple[float, float]:
        return float(self._center[0]), float(self._center[1])

    def _flatten_spawn_pad(self, h: np.ndarray) -> np.ndarray:
        res = h.shape[0]
        axis = np.linspace(0.0, self._size, res)
        gx, gy = np.meshgrid(axis, axis)
        dist = np.hypot(gx - self._center[0], gy - self._center[1])
        # 0 inside the flat radius, 1 beyond the blend radius, smooth between.
        t = np.clip(
            (dist - WORLD_FLAT_RADIUS_M)
            / max(WORLD_BLEND_RADIUS_M - WORLD_FLAT_RADIUS_M, 1e-6),
            0.0,
            1.0,
        )
        blend = t * t * (3.0 - 2.0 * t)  # smoothstep
        return h * blend

    def ground_z(self, x: float, y: float) -> float:
        gx = np.clip(x / self._size * (self._res - 1), 0.0, self._res - 1)
        gy = np.clip(y / self._size * (self._res - 1), 0.0, self._res - 1)
        j0, i0 = int(math.floor(gx)), int(math.floor(gy))
        j1, i1 = min(j0 + 1, self._res - 1), min(i0 + 1, self._res - 1)
        tx, ty = gx - j0, gy - i0
        top = self._height[i0, j0] * (1 - tx) + self._height[i0, j1] * tx
        bot = self._height[i1, j0] * (1 - tx) + self._height[i1, j1] * tx
        return float(top * (1 - ty) + bot * ty)

    # -- construction phase ---------------------------------------------
    def construct(self, engine: GenesisPhysicsEngine) -> None:
        engine.add_terrain_entity(
            "lunar_terrain",
            self._height.astype(np.float32),
            [self._size, self._size],
        )

        regolith = _surface((0.62, 0.60, 0.57))
        rock_surf = _surface((0.40, 0.39, 0.37))
        base_surf = _surface((0.75, 0.76, 0.80))

        # Scattered boulders (skip any on/near the flat spawn pad).
        rock_kw = {"entity_type": "fixed"}
        if rock_surf is not None:
            rock_kw["surface"] = rock_surf
        placed = 0
        for k, rock in enumerate(self._terrain.rock_positions):
            rx, ry = float(rock[0]), float(rock[1])
            radius = float(rock[3]) if len(rock) > 3 else 0.25
            if np.hypot(rx - self._center[0], ry - self._center[1]) < (
                WORLD_FLAT_RADIUS_M + 1.0
            ):
                continue
            rz = self.ground_z(rx, ry) + radius
            engine.add_entity(
                f"rock_{k:03d}",
                gs.morphs.Box(pos=(rx, ry, rz),
                              size=(radius * 2, radius * 2, radius * 2)),
                gs.materials.Rigid(friction=1.0),
                **rock_kw,
            )
            placed += 1
        self._rock_count = placed

        # Moonbase: habitat block + comm tower + cable spool post at the anchor.
        ax, ay = float(self._anchor[0]), float(self._anchor[1])
        az = self.ground_z(ax, ay)
        base_kw = {"entity_type": "fixed"}
        if base_surf is not None:
            base_kw["surface"] = base_surf
        engine.add_entity(
            "moonbase_hab",
            gs.morphs.Box(pos=(ax, ay, az + 1.25), size=(4.0, 3.0, 2.5)),
            gs.materials.Rigid(), **base_kw,
        )
        engine.add_entity(
            "moonbase_tower",
            gs.morphs.Box(pos=(ax + 1.6, ay + 1.2, az + 3.0),
                          size=(0.18, 0.18, 6.0)),
            gs.materials.Rigid(), **base_kw,
        )
        self._anchor_z = az + 0.4  # cable leaves the spool ~0.4 m up

        # Cable link pool — thin beads, parked underground until positioned.
        cable_surf = _surface((0.10, 0.10, 0.12))
        cable_kw = {"entity_type": "kinematic"}
        if cable_surf is not None:
            cable_kw["surface"] = cable_surf
        for i in range(self._cable_n):
            engine.add_entity(
                f"cable_link_{i:03d}",
                _visual_box(pos=(0.0, 0.0, -50.0 - i),
                            size=(0.13, 0.05, 0.05)),
                gs.materials.Rigid(), **cable_kw,
            )

        # Rut decal pool — flat dark slabs recessed into the regolith.
        rut_surf = _surface((0.22, 0.20, 0.18))
        rut_kw = {"entity_type": "kinematic"}
        if rut_surf is not None:
            rut_kw["surface"] = rut_surf
        for i in range(self._rut_n):
            engine.add_entity(
                f"rut_{i:03d}",
                _visual_box(pos=(0.0, 0.0, -80.0 - i),
                            size=(0.24, 0.13, 0.04)),
                gs.materials.Rigid(), **rut_kw,
            )

    # -- simulation phase ------------------------------------------------
    @staticmethod
    def _yaw_quat(yaw: float) -> np.ndarray:
        # Genesis base quaternion order is (w, x, y, z).
        return np.array(
            [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)],
            dtype=np.float32,
        )

    def _place(self, engine: GenesisPhysicsEngine, name: str,
               pos: np.ndarray, quat: np.ndarray) -> None:
        try:
            engine.set_body_pose(name, pos.astype(np.float32), quat)
            engine.set_body_velocity(name, np.zeros(3, np.float32),
                                     np.zeros(3, np.float32))
        except Exception:  # noqa: BLE001 - never let a visual abort the sim
            pass

    def _update_cable(self, engine: GenesisPhysicsEngine,
                      rover_pos: np.ndarray, yaw: float) -> None:
        # Cable leaves the chassis rear and runs to the moonbase spool, sagging
        # under 1/6 g between the two ends.
        rear = rover_pos[:2] - 0.35 * np.array([math.cos(yaw), math.sin(yaw)])
        p0 = np.array([rear[0], rear[1], float(rover_pos[2]) - 0.05])
        p1 = np.array([self._anchor[0], self._anchor[1], self._anchor_z])
        span = float(np.linalg.norm(p1[:2] - p0[:2]))
        sag = min(0.45, 0.06 * span)
        seg_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        quat = self._yaw_quat(seg_yaw)
        for i in range(self._cable_n):
            s = i / max(self._cable_n - 1, 1)
            xy = p0[:2] * (1 - s) + p1[:2] * s
            z = p0[2] * (1 - s) + p1[2] * s - sag * math.sin(math.pi * s)
            # The cable rests *on* the regolith: never let a link sink below
            # the local terrain surface, so it drapes over the ground and
            # follows bumps instead of clipping through them.
            floor = self.ground_z(float(xy[0]), float(xy[1])) + self._link_half_h
            z = max(z, floor)
            self._place(engine, f"cable_link_{i:03d}",
                        np.array([xy[0], xy[1], z]), quat)

    def _drop_ruts(self, engine: GenesisPhysicsEngine, rover_pos: np.ndarray,
                   yaw: float, body_v: float,
                   wheel_omega: tuple[float, float]) -> None:
        xy = rover_pos[:2].astype(np.float64)
        if self._last_drop_xy is not None:
            if np.linalg.norm(xy - self._last_drop_xy) < RUT_DROP_SPACING_M:
                return
        if abs(body_v) < 0.03:
            return
        self._last_drop_xy = xy.copy()

        # Lateral (left/right) wheel offset in world frame.
        lat = np.array([-math.sin(yaw), math.cos(yaw)]) * (self._track * 0.5)
        wheel_load = (self._mass_kg * 1.622) / 2.0
        for side, omega in ((1.0, wheel_omega[0]), (-1.0, wheel_omega[1])):
            wxy = xy + side * lat
            cx = float(np.clip(wxy[0], 0.0, self._size))
            cy = float(np.clip(wxy[1], 0.0, self._size))
            state = self._terramechanics.update_wheel(
                np.array([cx, cy, 0.0], dtype=np.float64),
                wheel_angular_vel=omega,
                ground_velocity=abs(body_v),
                wheel_load_n=wheel_load,
                contact_radius_m=max(0.04, self._wheel_r * 0.35),
                cable_tension_n=0.0,
            )
            sink = float(max(state.rut_depth, state.sinkage))
            sink = min(0.06, max(0.005, sink))
            gz = self.ground_z(cx, cy)
            # Recess the slab so its top sits one sinkage depth below grade.
            z = gz - sink - 0.02
            self._place(engine, f"rut_{self._rut_idx:03d}",
                        np.array([cx, cy, z]), self._yaw_quat(yaw))
            self._rut_idx = (self._rut_idx + 1) % self._rut_n

    def step(self, engine: GenesisPhysicsEngine, rover_pos: np.ndarray,
             yaw: float, body_v: float,
             wheel_omega: tuple[float, float], dt: float) -> None:
        # Redrape the cable at WORLD_VISUAL_HZ, not the 120 Hz physics rate —
        # re-posing every link every step is the bulk of the visual cost.
        self._vis_accum += dt
        if self._vis_accum >= self._vis_period:
            self._vis_accum = 0.0
            self._update_cable(engine, rover_pos, yaw)
        # Rut stamps are self-throttled by travel distance.
        self._drop_ruts(engine, rover_pos, yaw, body_v, wheel_omega)

    def print_summary(self) -> None:
        print(
            f"  World       : {self._size:.0f} m lunar heightfield, "
            f"{self._rock_count} rocks, moonbase + trailing cable, "
            f"analytic rut decals"
        )


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
    if args.capability_demo:
        print("  Capabilities: terrain/regolith, sensors, cable, antenna, moonbase, power/thermal")
    print(f"  Backend     : {args.backend.upper()}  sim @ {args.sim_hz:.0f} Hz")
    print(f"  Viewer      : {'off (headless)' if args.no_viewer else f'on @ {args.render_hz} Hz'}")
    if args.backend == "gpu":
        print("  *** GPU FIRST RUN: Genesis will compile CUDA kernels (~5–10 min). ***")
        print("  *** Subsequent runs are cached and start in seconds.              ***")
    if not args.no_viewer and not args.no_keyboard:
        print("-" * 92)
        print("  Keys (click the 3-D window first so it has keyboard focus):")
        print("        Up/Down drive  Left/Right spin (hold to keep going)  Space stop")
        print("        1..4 select joint   [ / ] jog joint  O/C gripper  X stow")
        print("        R reset pose   ESC/Q quit")
        if args.navigation:
            print("        F5 start the autonomous navigation tour (planner+MPC)")
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
# Compile / build progress spinner
# ---------------------------------------------------------------------------


class ProgressSpinner:
    """Background spinner with elapsed-time readout.

    Genesis's first-run CUDA-kernel compile can take five-plus minutes and
    runs entirely inside `engine.configure()` / `engine.build_scene()` with
    no visible output. Without a heartbeat, the F5'd VS Code window looks
    frozen and the user kills the run before the viewer ever appears. The
    spinner writes a single-line tick to stdout every second so the user
    can see the wall clock advancing and the demo is alive.

    Use as a context manager:

        with ProgressSpinner("Compiling Genesis kernels"):
            engine.configure(...)
    """

    _GLYPHS = ("|", "/", "-", "\\")

    def __init__(self, label: str, *, tick_s: float = 1.0) -> None:
        self._label = label
        self._tick_s = float(tick_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._is_tty = sys.stdout.isatty()

    def __enter__(self) -> "ProgressSpinner":
        self._t0 = time.perf_counter()
        # On a TTY we redraw a single line; in piped / debug consoles we
        # fall back to one print per tick so the integratedTerminal in
        # VS Code doesn't end up with a blank screen.
        self._thread = threading.Thread(
            target=self._run, name=f"spinner-{self._label}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        elapsed = time.perf_counter() - self._t0
        # Always print a clean "done" line so the spinner row is replaced.
        suffix = "OK" if exc is None else "FAIL"
        if self._is_tty:
            sys.stdout.write("\r")
        print(f"  {self._label}: {elapsed:6.2f}s  [{suffix}]")
        sys.stdout.flush()

    def _run(self) -> None:
        i = 0
        next_tick = self._t0
        while not self._stop.is_set():
            now = time.perf_counter()
            if now >= next_tick:
                elapsed = now - self._t0
                glyph = self._GLYPHS[i % len(self._GLYPHS)]
                msg = f"  {self._label}: {glyph} {elapsed:6.2f}s elapsed"
                if self._is_tty:
                    sys.stdout.write("\r" + msg)
                else:
                    print(msg)
                sys.stdout.flush()
                i += 1
                next_tick = self._t0 + i * self._tick_s
            self._stop.wait(timeout=0.1)


# ---------------------------------------------------------------------------
# Navigation pipeline driver (planner + MPC + cable/slope feedback)
# ---------------------------------------------------------------------------


class NavigationDriver:
    """Autonomous driver: A* planner picks a path, MPC tracks it, cable
    tension and terrain slope feed back into the MPC speed limit.

    This is what makes the rover *visibly* struggle when the cable pulls
    taut: the cable-distance proxy ramps tension into the 200-400 N band as
    the rover gets further from the moonbase spool, and the MPC's
    :meth:`get_speed_limit` enforces ``v *= (400 - T) / 200`` in that band
    (full stop above 400 N). Combined with the slope penalty above ~5 deg,
    the rover slows down on rough terrain and again when the leash bites.

    A tour of waypoints around the lunar scene is planned at construction
    time so each leg's planned path is precomputed. The rover follows the
    legs in sequence; when the final goal is reached, the tour loops.
    """

    # Cable tension proxy (Newtons) as a function of rover-to-anchor distance:
    # zero while there is slack, then a linear ramp once the leash bites.
    # Tuned so the configured tour visits the 0..350 N band — light slack
    # near the spool, MPC speed-limit kicks in past 200 N, full stop at 400 N.
    _CABLE_TAUT_M = 9.0      # cable goes taut at this distance from spool
    _CABLE_TENSION_PER_M = 50.0   # N gained per metre stretched past taut
    _CABLE_TENSION_MAX_N = 500.0
    # The balance bot tips above ~0.6 m/s under 1/6 g; cap the MPC's
    # effective speed ceiling here. The factor is computed at init so the
    # *shape* of the cable / slope speed curve from get_speed_limit() is
    # preserved — only the absolute ceiling shrinks.
    _BALANCE_SAFE_CEILING_MPS = 0.55

    def __init__(
        self,
        world: "LunarWorld",
        spawn_xy: tuple[float, float],
        sim_hz: float,
    ) -> None:
        self._world = world
        self._spawn = np.array(spawn_xy, dtype=np.float64)
        self._anchor = np.asarray(world._anchor, dtype=np.float64).copy()
        # MPC at 5 Hz: more than that just thrashes the IPOPT warm-starts and
        # the balance controller can't react faster than its own pitch loop.
        self._mpc_period_s = 0.2
        self._last_mpc_t = -1.0e9
        self._last_v_cmd = 0.0
        self._last_w_cmd = 0.0

        # Tour: drive away from the spool to stretch the cable, swing
        # laterally so the leash drags across regolith, then come back.
        # Max distance from the (cx-7, cy-7) anchor is ~15 m, which puts
        # tension at ~300 N — solidly into the MPC's 200..400 N modulation
        # band so the user can SEE the rover slow on its leash.
        cx, cy = world.center
        self._tour: list[np.ndarray] = [
            np.array([cx + 2.0, cy + 0.0, 0.0]),   # ~11 m from anchor
            np.array([cx + 4.0, cy + 2.0, 0.0]),   # ~15 m — leash bites
            np.array([cx + 2.0, cy + 4.0, 0.0]),   # ~15 m — leash bites
            np.array([cx + 0.0, cy + 5.0, 0.0]),   # ~13.9 m
            np.array([cx - 3.0, cy + 2.0, 0.0]),   # ~9.9 m — slack returns
            np.array([cx - 2.0, cy - 1.0, 0.0]),   # ~7.8 m — close to home
        ]
        self._tour_idx = 0
        self._waypoint_reach_m = 0.7

        # ------------------------------------------------------------------
        # Build the navigation stack: occupancy, traversability, cable map,
        # planner, MPC. Origin and grid shape are sized to the lunar world.
        # ------------------------------------------------------------------
        size = float(world._size)
        res = 0.5  # 0.5 m planning grid
        nx = ny = int(round(size / res))
        # Wrap GridOccupancyMap so rocks read as occupied. The default origin
        # is (0,0), matching our world coordinates which run (0..size).
        self._occ = GridOccupancyMap(
            width_m=size,
            height_m=size,
            origin_xy=(0.0, 0.0),
            voxel_resolution_m=res,
        )
        self._mark_rocks_occupied()
        self._cable_map = SpatialCableMap()
        self._trav = GridTraversabilityMap(grid_shape=self._occ.grid_shape[:2])
        self._trav.update(None, self._occ, self._cable_map)

        self._planner = AStarDStarPathPlanner()
        self._planner.configure(
            PlannerConfig(
                grid_resolution_m=res,
                algorithm="a_star",
                heading_change_penalty=0.15,
                cable_clearance_m=1.0,
                replan_trigger_distance_m=4.0,
            )
        )
        # The planner stores its own _grid_shape from a 200x200 default;
        # narrow it to our world so A* doesn't search outside the heightfield.
        self._planner._grid_shape = (nx, ny)
        self._planner._origin = np.zeros(2, dtype=np.float64)
        self._planner._resolution = res

        self._mpc = MPCController()
        self._mpc.configure(
            MPCConfig(
                horizon_s=1.6,
                step_s=0.1,
                # max_linear_vel here is the inner-NLP control limit; the
                # *effective* speed cap is set externally via
                # speed_limit_factor below so cable/slope feedback still
                # modulates the rover. Use a generous internal cap.
                max_linear_vel=1.5,
                max_angular_vel=0.7,
                update_rate_hz=int(round(1.0 / self._mpc_period_s)),
                cable_tension_speed_limit_threshold_n=200.0,
            )
        )
        # Pre-compute the speed factor that maps MPC's idealised 1.5 m/s
        # flat-no-tension ceiling down to the balance-bot-safe ceiling. The
        # factor preserves the SHAPE of get_speed_limit's slope+tension
        # response (so the rover still visibly slows on rough or tense
        # legs), just scaled to a ceiling the inverted pendulum can sustain.
        baseline = self._mpc.get_speed_limit(0.0, 0.0)
        self._speed_factor = self._BALANCE_SAFE_CEILING_MPS / max(baseline, 0.1)

        # Plan the first leg now so the very first MPC call has a path.
        self._current_path: Optional[PlannedPath] = None
        self._plan_to(self._tour[self._tour_idx], from_xy=self._spawn)
        self._mpc_failures = 0

        # Armed but idle. The rover keeps balancing in place until the
        # operator presses N (or types n in the terminal) — that gives
        # everyone a chance to confirm the inverted pendulum is stable
        # before the planner starts demanding motion.
        self._started: bool = False
        self._start_sim_t: float = 0.0

    def is_started(self) -> bool:
        return self._started

    def start(self, sim_t: float) -> None:
        if self._started:
            return
        self._started = True
        self._start_sim_t = sim_t
        # Re-plan from current position the first time motion starts so the
        # MPC reference is anchored at *now*, not at the spawn pose where
        # the demo built the navigator.
        print(f"  [nav] F5 pressed at sim_t={sim_t:.2f}s — "
              "planner+MPC engaged")

    # ------------------------------------------------------------------
    # Map construction
    # ------------------------------------------------------------------

    def _mark_rocks_occupied(self) -> None:
        """Stamp the world's rocks into the occupancy grid as obstacles."""
        for rock in self._world._terrain.rock_positions:
            rx, ry = float(rock[0]), float(rock[1])
            radius = float(rock[3]) if len(rock) > 3 else 0.25
            cx, cy = self._world.center
            if np.hypot(rx - cx, ry - cy) < (WORLD_FLAT_RADIUS_M + 1.0):
                # Rocks inside the spawn flat are skipped by the world too.
                continue
            r_cells = max(1, int(round((radius + 0.25) / self._occ.voxel_resolution_m)))
            gx, gy = self._occ._world_to_grid(np.array([rx, ry]))
            nx_, ny_ = self._occ.grid_shape
            for dx in range(-r_cells, r_cells + 1):
                for dy in range(-r_cells, r_cells + 1):
                    if dx * dx + dy * dy > r_cells * r_cells:
                        continue
                    self._occ._update_cell(gx + dx, gy + dy, 5.0)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_to(self, goal_xyz: np.ndarray, from_xy: np.ndarray) -> bool:
        start = np.array([float(from_xy[0]), float(from_xy[1]), 0.0], dtype=np.float64)
        try:
            self._current_path = self._planner.plan(
                start, goal_xyz, self._trav, self._cable_map
            )
            print(
                f"  [nav] A* leg {self._tour_idx + 1}/{len(self._tour)}: "
                f"({start[0]:.1f},{start[1]:.1f}) -> "
                f"({goal_xyz[0]:.1f},{goal_xyz[1]:.1f})  "
                f"{len(self._current_path.waypoints)} wp, "
                f"{self._current_path.total_distance_m:.2f} m, "
                f"risk={self._current_path.risk_score:.3f}"
            )
            return True
        except ValueError as exc:
            print(f"  [nav] planner failed: {exc}; falling back to direct line")
            self._current_path = PlannedPath(
                waypoints=[start, goal_xyz],
                total_distance_m=float(np.linalg.norm(goal_xyz[:2] - start[:2])),
                estimated_cable_drag_energy=0.0,
                risk_score=0.0,
            )
            return False

    # ------------------------------------------------------------------
    # Feedback proxies
    # ------------------------------------------------------------------

    def cable_tension_n(self, pos_xy: np.ndarray) -> float:
        d = float(np.linalg.norm(pos_xy[:2] - self._anchor))
        over = max(0.0, d - self._CABLE_TAUT_M)
        return float(min(self._CABLE_TENSION_MAX_N, self._CABLE_TENSION_PER_M * over))

    def terrain_slope_deg(self, pos_xy: np.ndarray) -> float:
        eps = 0.4
        z0 = self._world.ground_z(float(pos_xy[0]), float(pos_xy[1]))
        zx = self._world.ground_z(float(pos_xy[0] + eps), float(pos_xy[1]))
        zy = self._world.ground_z(float(pos_xy[0]), float(pos_xy[1] + eps))
        dzdx = (zx - z0) / eps
        dzdy = (zy - z0) / eps
        slope_m = math.hypot(dzdx, dzdy)
        return float(math.degrees(math.atan(slope_m)))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def current_goal(self) -> np.ndarray:
        return self._tour[self._tour_idx].copy()

    def step(
        self,
        sim_t: float,
        rover_pos: np.ndarray,
        yaw: float,
        body_v: float,
    ) -> tuple[float, float, dict]:
        info: dict = {"started": self._started}
        if not self._started:
            # Idle: report telemetry but command zero motion so the
            # balance controller just station-keeps.
            info["tension_n"] = self.cable_tension_n(rover_pos[:2])
            info["slope_deg"] = self.terrain_slope_deg(rover_pos[:2])
            info["goal_xy"] = (float(self._tour[self._tour_idx][0]),
                               float(self._tour[self._tour_idx][1]))
            info["leg_idx"] = self._tour_idx
            info["dist_to_goal_m"] = float(
                np.linalg.norm(rover_pos[:2] - self._tour[self._tour_idx][:2])
            )
            info["v_max_mps"] = 0.0
            info["v_cmd_mps"] = 0.0
            info["w_cmd_radps"] = 0.0
            return 0.0, 0.0, info

        # Advance waypoint if we've reached the current goal.
        goal = self._tour[self._tour_idx]
        dist_to_goal = float(np.linalg.norm(rover_pos[:2] - goal[:2]))
        if dist_to_goal < self._waypoint_reach_m:
            self._tour_idx = (self._tour_idx + 1) % len(self._tour)
            self._plan_to(self._tour[self._tour_idx], from_xy=rover_pos[:2])
            goal = self._tour[self._tour_idx]
            info["leg_advanced"] = True

        tension = self.cable_tension_n(rover_pos[:2])
        slope = self.terrain_slope_deg(rover_pos[:2])
        info["tension_n"] = tension
        info["slope_deg"] = slope
        info["goal_xy"] = (float(goal[0]), float(goal[1]))
        info["leg_idx"] = self._tour_idx
        info["dist_to_goal_m"] = dist_to_goal
        info["v_max_mps"] = self._mpc.get_speed_limit(slope, tension)

        # MPC at its configured period; in between, hold the last command.
        if sim_t - self._last_mpc_t < self._mpc_period_s:
            return self._last_v_cmd, self._last_w_cmd, info
        self._last_mpc_t = sim_t

        # Build [x, y, theta, v, omega] state for MPC.
        current_state = np.array(
            [float(rover_pos[0]), float(rover_pos[1]), float(yaw), float(body_v), 0.0],
            dtype=np.float64,
        )

        ref_waypoints = self._current_path.waypoints if self._current_path is not None else [goal]
        try:
            out = self._mpc.compute(
                current_state,
                ref_waypoints,
                cable_tension_n=tension,
                terrain_slope_deg=slope,
                speed_limit_factor=self._speed_factor,
            )
            v_cmd = float(out.linear_velocity)
            w_cmd = float(out.angular_velocity)
        except Exception as exc:  # noqa: BLE001 - keep the rover moving
            self._mpc_failures += 1
            info["mpc_error"] = str(exc)
            # Fallback: simple proportional heading + speed toward goal,
            # respecting the MPC speed limit so cable behaviour is preserved.
            dx = float(goal[0] - rover_pos[0])
            dy = float(goal[1] - rover_pos[1])
            desired_yaw = math.atan2(dy, dx)
            yaw_err = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))
            v_limit = self._mpc.get_speed_limit(slope, tension) * self._speed_factor
            if abs(yaw_err) > 0.5:
                v_cmd = 0.05
            else:
                v_cmd = min(v_limit, 0.4 * max(0.0, dist_to_goal))
            w_cmd = max(-0.6, min(0.6, 1.4 * yaw_err))

        # Belt-and-braces clamp into a balancer-safe range.
        v_cmd = max(-self._BALANCE_SAFE_CEILING_MPS,
                    min(self._BALANCE_SAFE_CEILING_MPS, v_cmd))
        w_cmd = max(-0.7, min(0.7, w_cmd))
        self._last_v_cmd = v_cmd
        self._last_w_cmd = w_cmd
        info["v_cmd_mps"] = v_cmd
        info["w_cmd_radps"] = w_cmd
        info["mpc_failures"] = self._mpc_failures
        return v_cmd, w_cmd, info


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

    # The rich lunar scene is the default in the viewer; headless runs and
    # --flat-ground keep the legacy bare plane so existing CI / balance-tuning
    # behaviour (and determinism) is unchanged.
    use_world = (not args.no_viewer) and (not args.flat_ground)
    world: Optional[LunarWorld] = None
    if use_world:
        world = LunarWorld(profile, args.sun_elevation_deg)
        spawn_xy = world.center
    else:
        spawn_xy = (0.0, 0.0)

    viewer_options = None
    if not args.no_viewer:
        viewer_options = gs.options.ViewerOptions(
            max_FPS=args.render_hz,
            refresh_rate=args.render_hz,
            run_in_thread=True,
            camera_pos=(spawn_xy[0] + 2.5, spawn_xy[1] - 2.5, 1.6),
            camera_lookat=(spawn_xy[0], spawn_xy[1], 0.5),
            camera_fov=45,
            # Drop the built-in viewer hotkeys (A/D/W/S/Z/R/...). They never
            # reached the rover anyway and they are exactly what shadowed the
            # drive keys; we install our own bindings on the viewer below.
            enable_default_keybinds=False,
        )

    print_header(args, profile)

    engine = GenesisPhysicsEngine()
    keyboard: Optional[KeyboardPoll] = None
    completed_cleanly = False
    try:
        configure_label = (
            "Compiling Genesis CUDA kernels (first GPU run can take 5-10 min)"
            if args.backend == "gpu"
            else "Initialising Genesis (first CPU run can take 0-5 min)"
        )
        with ProgressSpinner(configure_label):
            engine.configure(cfg, show_viewer=not args.no_viewer, viewer_options=viewer_options)

        ground_friction = 1.2
        if world is not None:
            # Terrain uses the engine's regolith-tuned contact (friction 1.2),
            # matching the rover material below.
            world.construct(engine)
        else:
            engine.add_entity(
                "ground", gs.morphs.Plane(),
                gs.materials.Rigid(friction=ground_friction),
            )

        spawn_z = spawn_height_for(rover_yaml, profile)
        rover_morph = gs.morphs.URDF(
            file=urdf_path, fixed=False,
            pos=(spawn_xy[0], spawn_xy[1], spawn_z),
        )
        # Match the rover's contact friction to the regolith ground so the
        # wheels actually get the grip the balance controller assumes.
        engine.add_entity(
            ROVER_NAME, rover_morph, gs.materials.Rigid(friction=ground_friction)
        )

        with ProgressSpinner("Building scene"):
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
        if world is not None:
            world.print_summary()
        # --navigation implies --capability-demo because the capability layer
        # is what publishes cable / sensor / power telemetry the user wants to
        # watch alongside the autonomous drive.
        if args.navigation and not args.capability_demo:
            args.capability_demo = True
        capability_demo = None
        if args.capability_demo:
            capability_demo = CapabilityShowcase(args, rover_yaml, profile)
            capability_demo.print_initial_status()

        navigation: Optional[NavigationDriver] = None
        if args.navigation:
            if world is None:
                print("  [nav] --navigation requires the lunar world scene; "
                      "ignoring (flat-ground or headless).")
            else:
                with ProgressSpinner("Building navigation stack (occupancy + A* + MPC)"):
                    navigation = NavigationDriver(world, spawn_xy, args.sim_hz)
                print("  Navigation  : planner+MPC armed — focus the 3-D window "
                      "and press F5 to start the autonomous tour. Cable tension "
                      "and slope feed the speed limit so the rover slows when "
                      "the leash bites.")

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
        # Spawn already folded over the body — do not let the servo sweep the
        # arm in from straight-out while the balancer is still settling.
        arm_bridge.snap()

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
        viewer_controls = None
        if not args.no_viewer and not args.no_keyboard:
            keyboard = KeyboardPoll()
            keyboard.start()
            # The real fix for "arrows do nothing": bind them on the viewer
            # window the user is actually focused on. Terminal polling stays
            # as a fallback for when the console has focus instead.
            viewer_controls = ViewerControls()
            if viewer_controls.attach(_get_viewer(engine)):
                print("  Controls    : viewer window focused — arrows drive, "
                      "Space stops, R resets, Q/Esc quit")
            else:
                print("  Controls    : focus the TERMINAL window for keys "
                      "(viewer keybind API unavailable)")

        run_loop(
            engine, drive, wheel_actuator, power, arm, arm_bridge, balance,
            keyboard, cfg, args, spawn_z, capability_demo,
            world=world, spawn_xy=spawn_xy, viewer_controls=viewer_controls,
            navigation=navigation,
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
    capability_demo: Optional[CapabilityShowcase] = None,
    *,
    world: Optional[LunarWorld] = None,
    spawn_xy: tuple[float, float] = (0.0, 0.0),
    viewer_controls: Optional[ViewerControls] = None,
    navigation: Optional[NavigationDriver] = None,
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
    next_nav_report_wall = 0.0
    last_nav_info: dict = {}

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

        # 1. Drain keyboard events (terminal poll + viewer-window keybinds).
        events: list[str] = []
        if keyboard is not None:
            events.extend(keyboard.pop_all())
        if viewer_controls is not None:
            events.extend(viewer_controls.pop_all())
        if events:
            for ch in events:
                lower = ch.lower()
                if ch == "\x1b" or lower == "q":  # ESC or Q
                    print("\n  Quit key pressed.")
                    return
                elif ch == "UP":
                    v_user = min(v_user + v_step, 2.0)
                elif ch == "DOWN":
                    v_user = max(v_user - v_step, -2.0)
                elif ch == "LEFT":
                    w_user = min(w_user + w_step, 2.0)
                elif ch == "RIGHT":
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
                elif lower == "x":
                    arm.stow()
                elif lower == "r":
                    _reset_pose(engine, spawn_z, spawn_xy)
                    balance.reset()
                    prev_yaw = None
                    v_user = 0.0
                    w_user = 0.0
                    if navigation is not None:
                        navigation._started = False  # rearm on reset
                elif ch == "F5":
                    if navigation is not None and not navigation.is_started():
                        navigation.start(engine.get_sim_time())

        # 2. Read body attitude and wheel state → balance torque.
        rover_pos, quat = engine.get_body_pose(ROVER_NAME)
        rover_pos = np.asarray(rover_pos, dtype=np.float64).reshape(3)
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

        # Navigation: planner+MPC drive v_user / w_user instead of the
        # keyboard or the scripted self-test sequence above.
        if navigation is not None:
            v_user, w_user, last_nav_info = navigation.step(t, rover_pos, yaw, body_v)

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

        # 4b. Slide the trailing cable and stamp wheel-rut decals.
        if world is not None:
            world.step(
                engine, rover_pos, yaw, body_v,
                wheel_actuator.wheel_omega(), dt,
            )

        # 5. Power accounting.
        driving = abs(v_user) > 0.05 or abs(w_user) > 0.05
        if capability_demo is not None:
            capability_demo.pre_power_step(t, dt, power)
            subsystem_states = capability_demo.subsystem_states(driving)
            sun_elevation = capability_demo.sun_elevation_deg
        else:
            subsystem_states = {
                "drive_motors": "active" if driving else "idle",
                "compute": "active",
                "comms": "idle",
                "cameras": "idle",
                "lidar": "idle",
                "imu": "idle",
                "manipulator": "active",
                "heating": "idle",
            }
            sun_elevation = args.sun_elevation_deg
        power_state = power.step(
            dt,
            sun_elevation=sun_elevation,
            subsystem_states=subsystem_states,
        )
        if capability_demo is not None:
            capability_demo.post_physics_step(
                engine=engine,
                wheel_actuator=wheel_actuator,
                power_state=power_state,
                sim_t=t,
                dt=dt,
                body_v=body_v,
                yaw_rate=yaw_rate,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                driving=driving,
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

        if navigation is not None and last_nav_info and now >= next_nav_report_wall:
            gx, gy = last_nav_info.get("goal_xy", (0.0, 0.0))
            tension_n = last_nav_info.get("tension_n", 0.0)
            slope_deg = last_nav_info.get("slope_deg", 0.0)
            v_max = last_nav_info.get("v_max_mps", 0.0)
            d = last_nav_info.get("dist_to_goal_m", 0.0)
            leg = last_nav_info.get("leg_idx", 0)
            state = "RUN" if last_nav_info.get("started") else "ARMED (press F5)"
            print(
                f"  nav | {state} leg {leg + 1}/{len(navigation._tour)} -> "
                f"({gx:5.2f},{gy:5.2f}) d={d:4.2f}m | "
                f"cable={tension_n:5.1f}N slope={slope_deg:4.1f}deg | "
                f"v_max={v_max:.2f} v_cmd={v_user:+.2f} w_cmd={w_user:+.2f}"
            )
            next_nav_report_wall = now + 1.0  # 1 Hz nav telemetry

        # Safety: if the rover falls flat, call it out — balance controller failed.
        if abs(pitch) > math.radians(70.0) or abs(roll) > math.radians(70.0):
            print(f"\n  *** Rover fell over (pitch={math.degrees(pitch):.1f} deg, "
                  f"roll={math.degrees(roll):.1f} deg) - press R to reset.")
            # Continue running; let the user recover.

        if args.steps and step_count >= args.steps:
            return


def _reset_pose(
    engine: GenesisPhysicsEngine,
    spawn_z: float,
    spawn_xy: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Respawn the rover upright at the spawn pad with zero velocity."""
    try:
        engine.set_body_pose(
            ROVER_NAME,
            np.array([spawn_xy[0], spawn_xy[1], spawn_z], dtype=np.float32),
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),  # wxyz identity
        )
        engine.set_body_velocity(
            ROVER_NAME,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
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
