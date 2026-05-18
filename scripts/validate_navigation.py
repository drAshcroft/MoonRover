"""Navigation subsystem validation with location logging.

Exercises the five navigation modules implemented in phase-6:
  - GridOccupancyMap    (LiDAR point clouds -> 2-D obstacle grid)
  - LocalizationEKFImpl (IMU + encoder + beacon -> pose estimate)
  - AStarDStarPathPlanner (A* and D* Lite on traversability grid)
  - MPCController       (CasADi unicycle MPC -> velocity commands)
  - ArmManipulationSequencer (antenna pickup/place pipeline)

No Genesis physics engine is required.  Sensor readings come from the same
simulation layer used by the demo's CapabilityShowcase (GenesisIMUSensor,
GenesisWheelEncoder, GenesisSunSensor, TrilaterationBeaconNetwork), fed with
synthetic ground-truth kinematics.

Usage
-----
    python scripts/validate_navigation.py
    python scripts/validate_navigation.py --steps 120 --verbose
    python scripts/validate_navigation.py --no-mpc      # skip MPC (faster)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

# Navigation modules under test
from moon_rover.navigation.perception.mapping import (  # noqa: E402
    GridOccupancyMap,
    SpatialCableMap,
    GridTraversabilityMap,
)
from moon_rover.navigation.localization.ekf import (  # noqa: E402
    LocalizationEKFImpl,
    EKFConfig,
    EKFState,
)
from moon_rover.navigation.planning.path_planner import (  # noqa: E402
    AStarDStarPathPlanner,
    PlannerConfig,
)
from moon_rover.navigation.control.mpc import MPCController, MPCConfig  # noqa: E402
from moon_rover.navigation.manipulation.sequencer import (  # noqa: E402
    ArmManipulationSequencer,
    ManipulationTask,
)

# Sensor layer (same modules used by the demo)
from moon_rover.sensors import (  # noqa: E402
    BeaconConfig,
    EncoderConfig,
    GenesisIMUSensor,
    GenesisWheelEncoder,
    GenesisSunSensor,
    IMUConfig,
    SunSensorConfig,
    TrilaterationBeaconNetwork,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Navigation subsystem validation")
    p.add_argument("--steps", type=int, default=60,
                   help="Simulation steps to run (default 60 @ 1 Hz sensor rate = 60 s)")
    p.add_argument("--sim-hz", type=float, default=50.0,
                   help="Simulation rate (Hz). Default 50.")
    p.add_argument("--verbose", action="store_true",
                   help="Print every-step location log instead of 5-step summary.")
    p.add_argument("--no-mpc", action="store_true",
                   help="Skip MPC solve (faster on machines without IPOPT warm-up).")
    p.add_argument("--algorithm", choices=("a_star", "d_star_lite"), default="a_star",
                   help="Path planning algorithm.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Simple protocol adapters so sensor objects satisfy EKF Protocols
# ---------------------------------------------------------------------------

@dataclass
class _IMUReading:
    timestamp: float
    accel_xyz: np.ndarray
    gyro_xyz: np.ndarray

@dataclass
class _EncoderReading:
    timestamp: float
    left_distance_m: float
    right_distance_m: float
    track_width_m: float

@dataclass
class _GPSFix:
    timestamp: float
    position_xyz: np.ndarray
    position_covariance: np.ndarray
    fix_quality: str

@dataclass
class _SunReading:
    timestamp: float
    heading_rad: float
    confidence: float


# ---------------------------------------------------------------------------
# Mock arm for sequencer validation (no real hardware)
# ---------------------------------------------------------------------------

class MockArm:
    """Records trajectory calls for sequencer validation."""

    def __init__(self) -> None:
        self.trajectory_log: list[list[np.ndarray]] = []
        self.gripper_log: list[float] = []
        self._status = "idle"
        self._aborted = False

    def follow_trajectory(self, waypoints: list[np.ndarray]) -> bool:
        self.trajectory_log.append(list(waypoints))
        self._status = "done"
        return True

    def command_gripper(self, open_fraction: float) -> None:
        self.gripper_log.append(open_fraction)

    def get_status(self) -> str:
        return self._status

    def abort(self) -> bool:
        self._aborted = True
        return True


# ---------------------------------------------------------------------------
# Ground-truth rover kinematics
# ---------------------------------------------------------------------------

class GroundTruthKinematics:
    """Simple unicycle for generating synthetic sensor truth."""

    TRACK_M = 0.45     # wheel track width
    WHEEL_R_M = 0.12   # wheel radius

    def __init__(self, x: float = 6.0, y: float = 6.0, yaw: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = 0.0
        self.vy = 0.0
        self.v = 0.0   # forward velocity
        self.omega = 0.0

    def step(self, v_cmd: float, omega_cmd: float, dt: float) -> None:
        self.v = v_cmd
        self.omega = omega_cmd
        self.yaw += omega_cmd * dt
        self.yaw = (self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.x += v_cmd * math.cos(self.yaw) * dt
        self.y += v_cmd * math.sin(self.yaw) * dt
        self.vx = v_cmd * math.cos(self.yaw)
        self.vy = v_cmd * math.sin(self.yaw)

    @property
    def wheel_omega(self) -> tuple[float, float]:
        """Left and right wheel angular velocities."""
        v_left = self.v - self.omega * self.TRACK_M / 2.0
        v_right = self.v + self.omega * self.TRACK_M / 2.0
        return v_left / self.WHEEL_R_M, v_right / self.WHEEL_R_M

    @property
    def pose6(self) -> np.ndarray:
        return np.array([self.x, self.y, 0.0, 0.0, 0.0, self.yaw], dtype=np.float64)


# ---------------------------------------------------------------------------
# Navigation pipeline harness
# ---------------------------------------------------------------------------

def build_beacon_network() -> TrilaterationBeaconNetwork:
    net = TrilaterationBeaconNetwork(seed=42)
    for bid, xyz in {
        "b0": (0.0, 0.0, 2.0),
        "b1": (20.0, 0.0, 2.0),
        "b2": (10.0, 20.0, 2.0),
        "b3": (0.0, 15.0, 2.0),
        "b4": (20.0, 15.0, 2.0),
        "b5": (10.0, 10.0, 8.0),
    }.items():
        net.add_beacon(bid, BeaconConfig(
            position_xyz=np.array(xyz, dtype=np.float64),
            signal_range_m=80.0,
            power_w=10.0,
            noise_sigma_m=0.05,
        ))
    return net


def build_lidar_points(rover_pose: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate a sparse LiDAR scan in rover frame (flat terrain + scattered obstacles)."""
    n = 72  # 72 beams per scan
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ranges = 4.0 + rng.normal(0, 0.1, n)   # ~4 m clear radius
    # Inject a couple of fake obstacles at fixed bearing
    ranges[10:14] = 1.5 + rng.normal(0, 0.05, 4)
    ranges[35:38] = 2.0 + rng.normal(0, 0.05, 3)
    pts = np.stack([
        ranges * np.cos(angles),
        ranges * np.sin(angles),
        np.zeros(n),
    ], axis=1).astype(np.float32)
    return pts


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> bool:
    dt = 1.0 / args.sim_hz
    rng = np.random.default_rng(seed=99)

    # ------------------------------------------------------------------
    # Build navigation stack
    # ------------------------------------------------------------------
    occ_map = GridOccupancyMap(width_m=30.0, height_m=30.0, origin_xy=(-5.0, -5.0))
    cable_map = SpatialCableMap()
    trav_map = GridTraversabilityMap(grid_shape=occ_map.grid_shape[:2])

    ekf = LocalizationEKFImpl()
    ekf.initialize(
        EKFConfig(imu_rate_hz=int(args.sim_hz), encoder_rate_hz=int(args.sim_hz)),
        EKFState(
            position_xyz=np.array([6.0, 6.0, 0.0]),
            velocity_xyz=np.zeros(3),
            orientation_rpy=np.zeros(3),
            gyro_bias_xyz=np.zeros(3),
            covariance=np.eye(15) * 0.1,
        ),
    )

    planner = AStarDStarPathPlanner()
    planner.configure(PlannerConfig(grid_resolution_m=0.5, algorithm=args.algorithm))

    mpc = None
    if not args.no_mpc:
        mpc = MPCController()
        mpc.configure(MPCConfig(horizon_s=2.0, step_s=0.1))

    seq = ArmManipulationSequencer()
    mock_arm = MockArm()

    # Beacon network
    beacons = build_beacon_network()

    # Sensor drivers
    imu_sensor = GenesisIMUSensor()
    imu_sensor.configure(IMUConfig(
        update_rate_hz=args.sim_hz,
        gyro_noise_sigma=0.003,
        accel_noise_sigma=0.05,
        gyro_bias_drift_deg_hr=0.5,
        seed=21,
    ))
    enc_sensor = GenesisWheelEncoder()
    enc_sensor.configure(EncoderConfig(counts_per_rev=2048, update_rate_hz=args.sim_hz, seed=22))
    sun_sensor = GenesisSunSensor()
    sun_sensor.configure(SunSensorConfig(accuracy_deg=0.8, update_rate_hz=1.0, seed=23))

    # Ground truth
    gt = GroundTruthKinematics(x=6.0, y=6.0, yaw=0.0)

    # Scripted drive commands: forward + turn
    def _cmd_for_step(step: int) -> tuple[float, float]:
        if step < 20:
            return 0.4, 0.0         # straight
        elif step < 35:
            return 0.3, 0.25        # left arc
        elif step < 50:
            return 0.4, -0.15       # slight right
        else:
            return 0.2, 0.0         # slow forward

    # Plan a path before the loop starts (static plan)
    goal = np.array([14.0, 14.0, 0.0])
    trav_map.update(None, occ_map, cable_map)  # empty map first pass
    planned_path = planner.plan(
        np.array([gt.x, gt.y, 0.0]), goal, trav_map, cable_map
    )
    print(f"\n[path] A* initial plan: {len(planned_path.waypoints)} waypoints, "
          f"{planned_path.total_distance_m:.2f} m, risk={planned_path.risk_score:.3f}")

    # ------------------------------------------------------------------
    # Sequencer smoke test (before physics loop)
    # ------------------------------------------------------------------
    print("\n[seq]  Testing ManipulationSequencer...")
    for task in (ManipulationTask.ANTENNA_PICKUP, ManipulationTask.ANTENNA_PLACEMENT,
                 ManipulationTask.CABLE_CONNECTION, ManipulationTask.TRANSPORT_STOW):
        ok = seq.execute_task(task, mock_arm)
        n_trajs = len(mock_arm.trajectory_log)
        print(f"         {task.value:<22} -> {'OK' if ok else 'FAIL'}  "
              f"(total trajectory segments so far: {n_trajs})")

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    print(f"\n[loop] Running {args.steps} steps @ {args.sim_hz:.0f} Hz  "
          f"(dt={dt*1000:.1f} ms)  algorithm={args.algorithm}\n")
    print(f"{'step':>5}  {'t':>6}  {'gt_x':>7} {'gt_y':>7} {'gt_yaw':>7}  "
          f"{'ekf_x':>7} {'ekf_y':>7} {'ekf_yaw':>7}  "
          f"{'pos_err':>8}  {'yaw_err':>8}  {'beacons':>7}  {'v_cmd':>6} {'w_cmd':>6}")
    print("-" * 110)

    pos_errors: list[float] = []
    beacon_counts: list[int] = []
    mpc_calls = 0
    mpc_failures = 0
    t0 = time.perf_counter()
    prev_v = 0.0  # for computing body-frame acceleration

    for step in range(args.steps):
        t = step * dt
        v_ref, omega_ref = _cmd_for_step(step)

        # 1. Advance ground truth (capture velocity before the step for accel)
        prev_v = gt.v
        gt.step(v_ref, omega_ref, dt)
        delta_v = gt.v - prev_v  # actual velocity change this step

        # 2. Build synthetic sensor readings
        # For a constant-velocity driving scenario the forward acceleration is
        # near zero; we compute it from the actual velocity change rather than
        # using a fixed fraction of v_ref so the EKF doesn't see spurious
        # forward acceleration and accumulate unbounded position drift.
        accel_body_x = delta_v / dt if dt > 0 else 0.0
        accel = np.array([accel_body_x, 0.0, 0.0], dtype=np.float64)
        gyro = np.array([0.0, 0.0, gt.omega], dtype=np.float64)
        imu_raw = imu_sensor.read(accel, gyro)
        imu_reading = _IMUReading(
            timestamp=t,
            accel_xyz=imu_raw.accel_xyz,
            gyro_xyz=imu_raw.gyro_xyz,
        )

        wl, wr = gt.wheel_omega
        enc_raw = enc_sensor.read([wl, wr])
        # Use quantized angular velocities (rad/s) to get per-step distances.
        # enc_raw.counts is cumulative since start — do NOT divide total counts by
        # counts_per_rev or you get total-distance-as-velocity, growing every step.
        av = enc_raw.angular_velocities if len(enc_raw.angular_velocities) >= 2 else [wl, wr]
        dl_per_step = float(av[0]) * gt.WHEEL_R_M * dt
        dr_per_step = float(av[1]) * gt.WHEEL_R_M * dt
        enc_reading = _EncoderReading(
            timestamp=t,
            left_distance_m=dl_per_step + rng.normal(0, 0.001),
            right_distance_m=dr_per_step + rng.normal(0, 0.001),
            track_width_m=gt.TRACK_M,
        )

        # 3. EKF predict + update
        ekf_state = ekf.predict(imu_reading)
        ekf_state = ekf.update_encoder(enc_reading)

        # Beacon fix every 1 s
        n_beacons = 0
        if step % max(1, int(1.0 / dt)) == 0:
            gt_pos3 = np.array([gt.x, gt.y, 0.0])
            fix = beacons.compute_fix(gt_pos3)
            n_beacons = len(beacons.get_visible_beacons(gt_pos3))
            if fix is not None and n_beacons >= 3:
                gps_fix = _GPSFix(
                    timestamp=t,
                    position_xyz=fix.position_xyz,
                    position_covariance=np.eye(3) * (fix.gdop * 0.1) ** 2,
                    fix_quality="good" if fix.gdop < 3.0 else "poor",
                )
                ekf_state = ekf.update_gps(gps_fix)

            # Sun sensor every 1 s too.
            # The sensor returns the sun's azimuth in degrees (not rover heading).
            # To derive rover heading from sun azimuth we need the sun's world-frame
            # direction.  In this scenario the sun is at 135° (SE) in world frame,
            # so rover_heading = sun_azimuth_in_rover_body - 135°.
            # We use the GT yaw to simulate what the body-frame sun direction would
            # be, then add noise, keeping the update honest.
            SUN_WORLD_AZ_DEG = 135.0
            sun_raw = sun_sensor.read(SUN_WORLD_AZ_DEG, 35.0, False)
            if sun_raw.valid:
                # sun_raw.azimuth_deg is the sun's azimuth in world frame (noisy).
                # Rover heading = sun_azimuth_world - body_frame_sun_offset, but
                # in the demo convention sun_azimuth_world is directly the reference.
                # We subtract the fixed sun world azimuth to recover a heading offset.
                rover_heading_est = math.radians(sun_raw.azimuth_deg - SUN_WORLD_AZ_DEG + 0.0)
                # Add GT yaw as the true base (sun sensor gives heading relative to sun)
                rover_heading_est = gt.yaw + math.radians(sun_raw.azimuth_deg - SUN_WORLD_AZ_DEG)
                sun_reading = _SunReading(
                    timestamp=t,
                    heading_rad=rover_heading_est,
                    confidence=0.7,
                )
                ekf_state = ekf.update_sun_sensor(sun_reading)

        beacon_counts.append(n_beacons)

        # 4. LiDAR -> OccupancyMap (every 5 steps to stay fast)
        if step % 5 == 0:
            pts = build_lidar_points(gt.pose6, rng)
            occ_map.update_from_point_cloud(pts, gt.pose6)

        # 5. MPC (once every 10 steps)
        v_mpc = v_ref
        w_mpc = omega_ref
        if mpc is not None and step % 10 == 0:
            mpc_calls += 1
            curr_state = np.array([
                ekf_state.position_xyz[0],
                ekf_state.position_xyz[1],
                ekf_state.orientation_rpy[2],  # yaw
                float(np.linalg.norm(ekf_state.velocity_xyz[:2])),
                0.0,
            ])
            cable_tension = 80.0 + 20.0 * abs(math.sin(t * 0.3))
            slope_deg = 5.0 * abs(math.sin(t * 0.1))
            try:
                out = mpc.compute(
                    curr_state,
                    planned_path.waypoints,
                    cable_tension_n=cable_tension,
                    terrain_slope_deg=slope_deg,
                    speed_limit_factor=1.0,
                )
                v_mpc = out.linear_velocity
                w_mpc = out.angular_velocity
            except Exception as exc:
                mpc_failures += 1
                if args.verbose:
                    print(f"  [mpc] step {step}: solver raised {exc}")

        # 6. Position error
        ekf_x = float(ekf_state.position_xyz[0])
        ekf_y = float(ekf_state.position_xyz[1])
        ekf_yaw = float(ekf_state.orientation_rpy[2])
        pos_err = math.sqrt((ekf_x - gt.x) ** 2 + (ekf_y - gt.y) ** 2)
        yaw_err = abs((ekf_yaw - gt.yaw + math.pi) % (2 * math.pi) - math.pi)
        pos_errors.append(pos_err)

        # 7. Log
        log_every = 1 if args.verbose else 5
        if step % log_every == 0:
            print(
                f"{step:>5}  {t:>6.2f}  "
                f"{gt.x:>7.3f} {gt.y:>7.3f} {math.degrees(gt.yaw):>7.2f}  "
                f"{ekf_x:>7.3f} {ekf_y:>7.3f} {math.degrees(ekf_yaw):>7.2f}  "
                f"{pos_err:>8.4f}  {math.degrees(yaw_err):>8.3f}°  "
                f"{n_beacons:>7d}  "
                f"{v_mpc:>6.3f} {w_mpc:>6.3f}"
            )

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # D* Lite replanning test (with a new obstacle injected)
    # ------------------------------------------------------------------
    print(f"\n[replan] Testing D* Lite replanning with injected obstacle...")
    obstacle_pts = [np.array([
        [10.0, 10.0, 0.0],
        [10.5, 10.0, 0.0],
        [10.0, 10.5, 0.0],
    ], dtype=np.float64)]
    current_pos = np.array([gt.x, gt.y, 0.0])
    try:
        replanned = planner.replan(current_pos, obstacle_pts)
        print(f"         Replanned: {len(replanned.waypoints)} waypoints, "
              f"{replanned.total_distance_m:.2f} m, risk={replanned.risk_score:.3f}  OK")
        replan_ok = True
    except Exception as exc:
        print(f"         Replanning failed: {exc}")
        replan_ok = False

    # ------------------------------------------------------------------
    # Speed limit table validation
    # ------------------------------------------------------------------
    print("\n[mpc]  Speed limit table (MPCController.get_speed_limit):")
    if mpc is None:
        mpc = MPCController()
        mpc.configure(MPCConfig())
    for slope, tension, label in [
        (0.0, 0.0, "flat / no tension"),
        (10.0, 0.0, "10° slope / no tension"),
        (18.0, 0.0, "18° slope / no tension"),
        (26.0, 0.0, "26° slope (impassable)"),
        (0.0, 150.0, "flat / 150 N (below threshold)"),
        (0.0, 300.0, "flat / 300 N (scaling)"),
        (0.0, 450.0, "flat / 450 N (above 400 N -> stop)"),
    ]:
        v = mpc.get_speed_limit(slope, tension)
        print(f"         {label:<40} -> {v:.3f} m/s")

    # ------------------------------------------------------------------
    # Occupancy map sanity
    # ------------------------------------------------------------------
    print("\n[map]  OccupancyMap sanity checks:")
    occ_at_rover = occ_map.query_occupancy(np.array([gt.x, gt.y, 0.0]))
    # Injected obstacles are at beam index 10-13 (~50° bearing) at 1.5 m range.
    # Query in world frame using rover's final yaw so the bearing is correct.
    obs_bearing = 10 / 72 * 2 * math.pi + gt.yaw
    occ_at_obstacle = occ_map.query_occupancy(np.array([
        gt.x + 1.5 * math.cos(obs_bearing),
        gt.y + 1.5 * math.sin(obs_bearing),
        0.0,
    ]))
    cost_field = occ_map.get_traversability_cost()
    print(f"         Occupancy at rover pos          : {occ_at_rover:.4f}  (expect near 0 = free)")
    print(f"         Occupancy at obstacle bearing   : {occ_at_obstacle:.4f}  (expect > rover)")
    print(f"         Cost field shape           : {cost_field.shape}")
    cable_clearance = cable_map.query_clearance(np.array([gt.x, gt.y, 0.0]))
    print(f"         Cable clearance (no cables): {cable_clearance:.1f}  (expect inf)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    pos_errors_arr = np.array(pos_errors)
    print("\n" + "=" * 72)
    print("NAVIGATION VALIDATION SUMMARY")
    print("=" * 72)
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        mark = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        else:
            failed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {mark} {name}{suffix}")

    check("EKF position error < 1.0 m (mean)",
          pos_errors_arr.mean() < 1.0,
          f"mean={pos_errors_arr.mean():.4f} m")
    check("EKF position error < 2.0 m (max)",
          pos_errors_arr.max() < 2.0,
          f"max={pos_errors_arr.max():.4f} m")
    check("EKF position accuracy reported",
          ekf.get_position_accuracy() < 10.0,
          f"1σ={ekf.get_position_accuracy():.4f} m")
    check("A* path returned valid waypoints",
          len(planned_path.waypoints) >= 2,
          f"{len(planned_path.waypoints)} pts, {planned_path.total_distance_m:.2f} m")
    check("D* Lite replan succeeded", replan_ok)
    if mpc_calls > 0:
        check("MPC produced commands without crash",
              mpc_failures == 0,
              f"{mpc_calls} calls, {mpc_failures} failures")
    else:
        check("MPC skipped (--no-mpc)", True, "n/a")
    check("ManipulationSequencer all tasks completed",
          all(len(r) > 0 for r in mock_arm.trajectory_log),
          f"{len(mock_arm.trajectory_log)} trajectory segments")
    check("OccupancyMap free space near rover",
          occ_at_rover < 0.5,
          f"prob={occ_at_rover:.4f}")
    check("OccupancyMap obstacle > rover free space",
          occ_at_obstacle > occ_at_rover,
          f"obstacle={occ_at_obstacle:.4f} rover={occ_at_rover:.4f}")
    check("Cable map clearance = inf (empty)",
          not np.isfinite(cable_clearance) or cable_clearance > 1000,
          f"d={cable_clearance:.1f}")
    check("Speed limit 0 at 26° slope (impassable)",
          mpc.get_speed_limit(26.0, 0.0) == 0.0)
    check("Speed limit 0 at 450 N cable tension",
          mpc.get_speed_limit(0.0, 450.0) == 0.0)

    print(f"\n  Steps run : {args.steps}  elapsed: {elapsed:.2f} s  "
          f"({args.steps / elapsed:.1f} steps/s)")
    print(f"  Passed    : {passed}")
    print(f"  Failed    : {failed}")
    print("=" * 72)

    return failed == 0


def main() -> int:
    args = parse_args()
    ok = run(args)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

