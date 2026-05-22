"""End-to-end showcase of the phase-8 data layer.

Drives a rover through a deterministic figure-8 mission while exercising the
three production data systems delivered in phase-8:

    1. MultiStreamLogger       — async writes of rover state, IMU, mission
                                 events to MCAP; LIDAR scans, RGB and depth
                                 camera frames, occupancy snapshots to HDF5.
    2. CheckpointStore         — save engine snapshot mid-mission, branch the
                                 saved checkpoint into N parallel engines for
                                 Monte Carlo replay from identical state.
    3. MissionMetricsAnalyzer  — compute mission KPIs (distance, energy,
                                 placement success rate, localization drift,
                                 etc.) and render them as a pandas DataFrame
                                 across all runs.

The demo uses a small synthetic rover engine that satisfies the same
``save_snapshot()``/``restore_snapshot()`` byte contract that
``GenesisPhysicsEngine`` exposes. That keeps the demo fast and deterministic
without paying Genesis's CUDA-kernel-compile startup cost on the first run.

Run::

    C:\\ve\\.genesis\\Scripts\\python.exe scripts/demo_data_pipeline.py

The demo prints clearly labelled sections for each subsystem and writes its
artifacts to a temp directory whose path is reported at the end so you can
crack open the HDF5/MCAP files (e.g. with ``h5py``, ``mcap info``, or Foxglove
Studio) to verify content.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from mcap.reader import make_reader

from moon_rover.data.analysis.metrics import MissionMetricsAnalyzer
from moon_rover.data.logging.streams import (
    LogConfig,
    MultiStreamLogger,
    StreamType,
)
from moon_rover.data.replay.checkpoint import CheckpointStore


# ---------------------------------------------------------------------------
# Synthetic rover engine
# ---------------------------------------------------------------------------


_DT = 1.0 / 50.0  # 50 Hz physics tick


@dataclass
class SyntheticRoverEngine:
    """Tiny deterministic stand-in for ``GenesisPhysicsEngine``.

    Mirrors the snapshot byte contract: ``save_snapshot() -> bytes`` and
    ``restore_snapshot(data: bytes) -> None``. The state is just the kinematic
    pose plus the per-step counter, so two engines restored from the same
    checkpoint will tick forward identically given identical control inputs.
    """

    sim_time: float = 0.0
    step_count: int = 0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heading: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    energy_wh: float = 0.0

    def step(self, command_v: float, command_yaw_rate: float) -> None:
        self.heading += command_yaw_rate * _DT
        vx = command_v * np.cos(self.heading)
        vy = command_v * np.sin(self.heading)
        self.velocity = np.array([vx, vy, 0.0])
        self.position = self.position + self.velocity * _DT
        # ~60 W cruising, scaled by speed and curvature.
        power_w = 60.0 + 80.0 * abs(command_v) + 40.0 * abs(command_yaw_rate)
        self.energy_wh += power_w * (_DT / 3600.0)
        self.sim_time += _DT
        self.step_count += 1

    def save_snapshot(self) -> bytes:
        return pickle.dumps(
            {
                "sim_time": self.sim_time,
                "step_count": self.step_count,
                "position": self.position.copy(),
                "heading": self.heading,
                "velocity": self.velocity.copy(),
                "energy_wh": self.energy_wh,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    def restore_snapshot(self, data: bytes) -> None:
        s = pickle.loads(data)
        self.sim_time = s["sim_time"]
        self.step_count = s["step_count"]
        self.position = np.array(s["position"], dtype=np.float64)
        self.heading = s["heading"]
        self.velocity = np.array(s["velocity"], dtype=np.float64)
        self.energy_wh = s["energy_wh"]


# ---------------------------------------------------------------------------
# Mission trajectory + sensor synthesis
# ---------------------------------------------------------------------------


@dataclass
class MissionPlan:
    """Figure-8 trajectory parameters."""

    radius_m: float = 6.0
    duration_s: float = 8.0
    cruise_speed_mps: float = 0.8
    waypoints: tuple = (  # antenna drop sites along the path
        (1.0, "antenna_01"),
        (3.5, "antenna_02"),
        (6.0, "antenna_03"),
    )

    def command_for_time(self, t: float) -> tuple[float, float]:
        """(forward speed, yaw rate) for a figure-8 traced once per duration."""
        omega = 2.0 * np.pi / self.duration_s
        # +omega for the first half, -omega for the second → smooth figure-8.
        yaw_rate = omega if t < self.duration_s / 2.0 else -omega
        return self.cruise_speed_mps, yaw_rate


def synthetic_lidar_scan(engine: SyntheticRoverEngine, rng: np.random.Generator) -> dict:
    """One frame of fake LIDAR: ~120 random points within 6 m of the rover."""
    n = int(80 + 40 * rng.random())
    azimuths = rng.uniform(-np.pi, np.pi, size=n)
    ranges = rng.uniform(0.5, 6.0, size=n)
    xs = engine.position[0] + ranges * np.cos(azimuths)
    ys = engine.position[1] + ranges * np.sin(azimuths)
    zs = rng.uniform(-0.1, 0.4, size=n)
    points = np.column_stack([xs, ys, zs]).astype(np.float32)
    return {"points": points, "timestamp": engine.sim_time}


def synthetic_rgb_frame(engine: SyntheticRoverEngine) -> np.ndarray:
    """64x96 RGB frame whose mean intensity drifts with rover heading."""
    h, w = 64, 96
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    angle = engine.heading
    field_ = (
        128.0
        + 60.0 * np.cos(0.05 * (xx - cx) + angle)
        + 60.0 * np.sin(0.05 * (yy - cy) - angle)
    )
    img = np.stack([field_, field_ * 0.7, field_ * 0.3], axis=-1)
    return np.clip(img, 0, 255).astype(np.uint8)


def synthetic_imu_reading(engine: SyntheticRoverEngine) -> dict:
    speed = float(np.linalg.norm(engine.velocity))
    return {
        "accel_xyz": [0.0, 0.0, -1.62],  # lunar gravity, body frame approx
        "gyro_xyz": [0.0, 0.0, float(engine.heading)],
        "speed_mps": speed,
        "timestamp": engine.sim_time,
    }


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------


def run_episode(
    *,
    name: str,
    output_dir: Path,
    plan: MissionPlan,
    duration_s: float,
    seed: int,
    engine: Optional[SyntheticRoverEngine] = None,
    sim_time_offset: float = 0.0,
    distance_offset: float = 0.0,
    checkpoint_store: Optional[CheckpointStore] = None,
    checkpoint_at_s: Optional[float] = None,
    placement_noise_m: float = 0.15,
) -> dict:
    """Run a single episode and return the log dict ready for the analyzer.

    A separate output directory under ``output_dir / name`` collects this
    episode's MCAP + HDF5 artifacts. If ``checkpoint_at_s`` is supplied, the
    engine is snapshotted to ``checkpoint_store`` at that sim time. The
    returned dict is shaped for ``MissionMetricsAnalyzer.compute_run_metrics``.
    """
    rng = np.random.default_rng(seed)
    engine = engine or SyntheticRoverEngine()

    episode_dir = output_dir / name
    episode_dir.mkdir(parents=True, exist_ok=True)

    logger = MultiStreamLogger()
    logger.initialize(
        LogConfig(output_dir=str(episode_dir), enable_hdf5=True, enable_mcap=True)
    )
    logger.log_event(
        "mission_start",
        {"run_id": name, "seed": seed, "sim_time": engine.sim_time, "timestamp": engine.sim_time},
    )

    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    headings: list[float] = []
    speeds: list[float] = []
    energies: list[float] = []
    powers: list[float] = []
    cable_tensions: list[float] = []
    cable_coverages: list[float] = []
    estimated_positions: list[np.ndarray] = []
    ground_truth_positions: list[np.ndarray] = []
    placements: list[dict] = []
    faults: list[dict] = []
    lidar_frames: list[dict] = []
    saved_checkpoint_id: Optional[str] = None

    n_steps = int(duration_s / _DT)
    next_waypoint_idx = 0
    last_lidar_t = -1.0
    last_camera_t = -1.0
    prev_energy = engine.energy_wh

    for i in range(n_steps):
        t_episode = i * _DT
        v_cmd, yaw_cmd = plan.command_for_time(t_episode)
        engine.step(v_cmd, yaw_cmd)

        t_sim = engine.sim_time
        timestamps.append(t_sim)
        positions.append(engine.position.copy())
        headings.append(engine.heading)
        speeds.append(float(np.linalg.norm(engine.velocity)))
        energies.append(engine.energy_wh)

        # Per-step power = energy delta / dt.
        power_w = (engine.energy_wh - prev_energy) * 3600.0 / _DT
        prev_energy = engine.energy_wh
        powers.append(power_w)

        # Cable tension grows linearly with distance from base.
        cable_tension = 20.0 + 8.0 * float(np.linalg.norm(engine.position))
        cable_tensions.append(cable_tension)

        # Coverage = fraction of duration completed.
        coverage_frac = (i + 1) / n_steps
        cable_coverages.append(coverage_frac)

        # A slightly noisy estimator for the localization drift KPI.
        gt = engine.position.copy()
        est = gt + rng.normal(scale=0.05, size=3)
        ground_truth_positions.append(gt)
        estimated_positions.append(est)

        # Stream rover state + IMU at every tick.
        logger.log_rover_state(
            "rover_a",
            {
                "position": engine.position.tolist(),
                "heading": engine.heading,
                "velocity": engine.velocity.tolist(),
                "energy_wh": engine.energy_wh,
                "cable_tension_n": cable_tension,
                "timestamp": t_sim,
            },
        )
        logger.log_sensor_reading(StreamType.SENSOR_IMU, synthetic_imu_reading(engine))

        # LIDAR @ 5 Hz, camera @ 10 Hz — independent sub-rates to exercise both backends.
        if t_episode - last_lidar_t >= 0.2:
            scan = synthetic_lidar_scan(engine, rng)
            logger.log_sensor_reading(StreamType.SENSOR_LIDAR, scan)
            lidar_frames.append({"t": t_sim, "points": scan["points"]})
            last_lidar_t = t_episode
        if t_episode - last_camera_t >= 0.1:
            logger.log_camera_frame(synthetic_rgb_frame(engine), StreamType.CAMERA_RGB)
            last_camera_t = t_episode

        # Antenna drop logic — at preset elapsed times.
        if (
            next_waypoint_idx < len(plan.waypoints)
            and t_episode >= plan.waypoints[next_waypoint_idx][0]
        ):
            drop_at_s, antenna_id = plan.waypoints[next_waypoint_idx]
            target = engine.position.copy()
            actual = target + rng.normal(scale=placement_noise_m, size=3)
            placements.append(
                {
                    "antenna_id": antenna_id,
                    "target": target.tolist(),
                    "actual": actual.tolist(),
                    "success": float(np.linalg.norm(actual - target)) <= 0.5,
                    "sim_time": t_sim,
                }
            )
            logger.log_event(
                "antenna_placed",
                {
                    "antenna_id": antenna_id,
                    "sim_time": t_sim,
                    "target": target.tolist(),
                    "actual": actual.tolist(),
                    "timestamp": t_sim,
                },
            )
            next_waypoint_idx += 1

        # Mid-mission checkpoint.
        if (
            checkpoint_store is not None
            and checkpoint_at_s is not None
            and saved_checkpoint_id is None
            and t_episode >= checkpoint_at_s
        ):
            saved_checkpoint_id = checkpoint_store.save_checkpoint(
                engine,
                sim_time=t_sim,
                description=f"{name} mid-mission checkpoint",
            )
            logger.log_event(
                "checkpoint_saved",
                {
                    "checkpoint_id": saved_checkpoint_id,
                    "sim_time": t_sim,
                    "timestamp": t_sim,
                },
            )

    logger.log_event(
        "mission_end",
        {"run_id": name, "sim_time": engine.sim_time, "timestamp": engine.sim_time},
    )
    logger.flush()
    artifact_bytes = logger.get_estimated_size_bytes()
    logger.close()

    log_data = {
        "run_id": name,
        "timestamp": np.array(timestamps, dtype=np.float64),
        "rover_position": np.stack(positions, axis=0),
        "power_consumed_w": np.array(powers, dtype=np.float64),
        "cable_tension_n": np.array(cable_tensions, dtype=np.float64),
        "cable_coverage_fraction": np.array(cable_coverages, dtype=np.float64),
        "estimated_position": np.stack(estimated_positions, axis=0),
        "ground_truth_position": np.stack(ground_truth_positions, axis=0),
        "antenna_placements": placements,
        "faults": faults,
        # Extra series for rendering (ignored by the analyzer).
        "heading": np.array(headings, dtype=np.float64),
        "speed_mps": np.array(speeds, dtype=np.float64),
        "energy_wh": np.array(energies, dtype=np.float64),
        "lidar_frames": lidar_frames,
        "_artifacts": {
            "dir": episode_dir,
            "bytes": artifact_bytes,
            "checkpoint_id": saved_checkpoint_id,
        },
    }
    return log_data


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------


def banner(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {title}\n{bar}")


def describe_hdf5(path: Path) -> None:
    with h5py.File(path, "r") as f:
        for group_name in sorted(f.keys()):
            grp = f[group_name]
            if isinstance(grp, h5py.Group):
                shapes = {k: grp[k].shape for k in grp.keys()}
                print(f"    /{group_name}: {shapes}")


def count_mcap_messages(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with open(path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, _msg in reader.iter_messages():
            counts[channel.topic] = counts.get(channel.topic, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Mission video rendering
# ---------------------------------------------------------------------------


def _rover_polygon(x: float, y: float, heading: float, length=0.6, width=0.4):
    """Corner points of an oriented chassis rectangle in world coordinates."""
    half_l, half_w = length / 2.0, width / 2.0
    body = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]]
    )
    c, s = np.cos(heading), np.sin(heading)
    rot = np.array([[c, -s], [s, c]])
    world = body @ rot.T + np.array([x, y])
    return world


def render_mission_video(
    log_data: dict,
    out_path: Path,
    *,
    fps: int = 20,
    trail_seconds: float = 1.5,
) -> Path:
    """Render a top-down animation of the rover driving, stamped with sim_time.

    Each frame corresponds to a real ``sim_time`` from the logged trajectory so
    the video can be scrubbed alongside the data table / MCAP records. Draws the
    traced path, trailing cable to base, current LIDAR scan points, antenna
    drops, and a HUD (time / speed / energy / cable tension).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import imageio.v2 as imageio

    ts = log_data["timestamp"]
    pos = log_data["rover_position"]
    heading = log_data["heading"]
    speed = log_data["speed_mps"]
    energy = log_data["energy_wh"]
    tension = log_data["cable_tension_n"]
    placements = log_data["antenna_placements"]
    lidar_frames = log_data["lidar_frames"]
    n = len(ts)
    if n < 2:
        raise ValueError("need at least two samples to render")

    sim_dt = float(ts[1] - ts[0])
    stride = max(1, int(round((1.0 / fps) / sim_dt)))
    frame_idxs = list(range(0, n, stride))
    trail_len = max(1, int(trail_seconds / sim_dt))

    # World bounds with a margin, including base station at the origin.
    pad = 1.5
    xs = np.concatenate([pos[:, 0], [0.0]])
    ys = np.concatenate([pos[:, 1], [0.0]])
    xlim = (float(xs.min()) - pad, float(xs.max()) + pad)
    ylim = (float(ys.min()) - pad, float(ys.max()) + pad)

    lidar_t = np.array([lf["t"] for lf in lidar_frames]) if lidar_frames else None

    writer = imageio.get_writer(str(out_path), fps=fps, macro_block_size=None)
    try:
        for k in frame_idxs:
            fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_facecolor("#0b0b14")
            ax.grid(True, color="#222233", linewidth=0.5)
            ax.set_title(f"Moon Rover mission  |  run={log_data['run_id']}", color="w")
            ax.tick_params(colors="#888899")

            # Base station marker.
            ax.scatter([0], [0], marker="s", s=120, color="#4fc3f7", zorder=5)
            ax.annotate("BASE", (0, 0), color="#4fc3f7", fontsize=8,
                        textcoords="offset points", xytext=(6, 6))

            # Current LIDAR scan (most recent scan at or before this time).
            if lidar_t is not None:
                idx = int(np.searchsorted(lidar_t, ts[k], side="right") - 1)
                if 0 <= idx < len(lidar_frames):
                    pts = lidar_frames[idx]["points"]
                    ax.scatter(pts[:, 0], pts[:, 1], s=3, color="#ffd54f",
                               alpha=0.35, zorder=2)

            # Fading path trail.
            lo = max(0, k - trail_len)
            ax.plot(pos[lo : k + 1, 0], pos[lo : k + 1, 1],
                    color="#80cbc4", linewidth=2.0, zorder=3)
            # Full faint path for context.
            ax.plot(pos[:, 0], pos[:, 1], color="#37474f", linewidth=0.8, zorder=1)

            # Trailing cable: straight line from base to rover.
            ax.plot([0, pos[k, 0]], [0, pos[k, 1]],
                    color="#ef5350", linewidth=1.0, alpha=0.6, zorder=2)

            # Antenna drops up to this time.
            for p in placements:
                if p.get("sim_time", 0.0) <= ts[k]:
                    a = p["actual"]
                    color = "#66bb6a" if p["success"] else "#ff7043"
                    ax.scatter([a[0]], [a[1]], marker="^", s=90, color=color, zorder=4)
                    ax.annotate(p["antenna_id"], (a[0], a[1]), color=color, fontsize=7,
                                textcoords="offset points", xytext=(5, 5))

            # Rover chassis + heading arrow.
            poly = _rover_polygon(pos[k, 0], pos[k, 1], heading[k])
            ax.add_patch(Polygon(poly, closed=True, facecolor="#e0e0e0",
                                 edgecolor="w", zorder=6))
            hx, hy = np.cos(heading[k]), np.sin(heading[k])
            ax.arrow(pos[k, 0], pos[k, 1], 0.5 * hx, 0.5 * hy,
                     head_width=0.18, color="#ff1744", zorder=7)

            # HUD.
            hud = (
                f"t = {ts[k]:6.2f} s\n"
                f"speed = {speed[k]:5.2f} m/s\n"
                f"energy = {energy[k]:6.3f} Wh\n"
                f"cable T = {tension[k]:6.1f} N"
            )
            ax.text(0.02, 0.98, hud, transform=ax.transAxes, va="top", ha="left",
                    color="#eeeeee", fontsize=9, family="monospace",
                    bbox=dict(boxstyle="round", facecolor="#1a1a2e", alpha=0.8))

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
            plt.close(fig)
    finally:
        writer.close()
    return out_path


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end showcase of phase-8 data layer (logging, checkpoint, metrics).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write demo artifacts. Default: a fresh temp dir.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the temp output dir on exit (default: keep, but suppress cleanup message).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Baseline RNG seed (branch seeds are derived).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip MP4 rendering of the driving trajectory.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Frame rate for the rendered mission video (default: 20).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="moon_rover_demo_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    banner("Moon Rover -- Phase-8 Data Pipeline Demo")
    print(f"  Output dir:  {output_dir}")
    print(f"  Seed:        {args.seed}")
    print(f"  Sim rate:    {1/_DT:.0f} Hz")
    plan = MissionPlan()
    print(
        f"  Mission:     figure-8, radius={plan.radius_m} m, "
        f"duration={plan.duration_s} s, cruise={plan.cruise_speed_mps} m/s"
    )

    # ------------------------------------------------------------------
    banner("1/4  Baseline episode + multi-stream logging")
    # ------------------------------------------------------------------
    store_dir = output_dir / "checkpoints"
    store = CheckpointStore(store_dir)

    baseline_log = run_episode(
        name="baseline",
        output_dir=output_dir,
        plan=plan,
        duration_s=plan.duration_s,
        seed=args.seed,
        checkpoint_store=store,
        checkpoint_at_s=plan.duration_s / 2.0,
    )
    bart = baseline_log["_artifacts"]
    print(f"  Episode dir: {bart['dir']}")
    print(f"  Disk usage:  {bart['bytes'] / 1024:.1f} KiB across MCAP + HDF5")
    print("  HDF5 groups:")
    describe_hdf5(bart["dir"] / "log.h5")
    print("  MCAP topics:")
    for topic, n in sorted(count_mcap_messages(bart["dir"] / "log.mcap").items()):
        print(f"    {topic:40s} {n:5d} messages")

    video_path: Optional[Path] = None
    if not args.no_video:
        video_path = output_dir / "baseline_drive.mp4"
        print(f"  Rendering driving video at {args.fps} fps ...")
        render_mission_video(baseline_log, video_path, fps=args.fps)
        print(f"  Video:       {video_path}  ({video_path.stat().st_size / 1024:.1f} KiB)")

    # ------------------------------------------------------------------
    banner("2/4  CheckpointStore -- save, list, branch")
    # ------------------------------------------------------------------
    print(f"  Store dir:   {store.store_dir}")
    print(f"  Checkpoint saved during baseline at sim_time={plan.duration_s/2.0:.2f}s:")
    print(f"    id = {bart['checkpoint_id']}")
    print("  Existing checkpoints:")
    for entry in store.list_checkpoints():
        print(
            f"    {entry['checkpoint_id']}  sim_time={entry['sim_time']:.2f}s  "
            f"size={entry['size_bytes'] / 1024:.1f} KiB"
        )
    # Monte Carlo branch: 3 fresh engines all restored from the same checkpoint.
    branch_engines = [SyntheticRoverEngine() for _ in range(3)]
    sim_times = store.branch(bart["checkpoint_id"], branch_engines)
    print(f"  Branched {len(branch_engines)} engines from the same snapshot:")
    for i, (eng, st) in enumerate(zip(branch_engines, sim_times)):
        print(
            f"    branch {i}: pos={tuple(round(float(x), 3) for x in eng.position)}, "
            f"heading={eng.heading:.3f}, restored sim_time={st:.2f}s"
        )

    # ------------------------------------------------------------------
    banner("3/4  Branched episode replays from checkpoint")
    # ------------------------------------------------------------------
    # Continue the first branched engine forward with a different RNG seed —
    # this is the Monte Carlo use case: same starting state, perturbed noise.
    branch_engine = branch_engines[0]
    branch_remaining = plan.duration_s - branch_engine.sim_time
    branch_plan = MissionPlan(
        radius_m=plan.radius_m,
        duration_s=plan.duration_s,
        cruise_speed_mps=plan.cruise_speed_mps,
        waypoints=tuple(
            wp for wp in plan.waypoints if wp[0] > plan.duration_s / 2.0
        ),
    )
    # Shift the branched plan's command timeline so figure-8 phasing matches.
    branch_plan_t0 = branch_engine.sim_time

    class _ShiftedPlan(MissionPlan):
        def command_for_time(self, t: float) -> tuple[float, float]:
            return super().command_for_time(t + branch_plan_t0)

    shifted = _ShiftedPlan(
        radius_m=branch_plan.radius_m,
        duration_s=branch_plan.duration_s,
        cruise_speed_mps=branch_plan.cruise_speed_mps,
        waypoints=tuple((wp[0] - branch_plan_t0, wp[1]) for wp in branch_plan.waypoints),
    )

    branch_log = run_episode(
        name="branch_0",
        output_dir=output_dir,
        plan=shifted,
        duration_s=branch_remaining,
        seed=args.seed + 1,
        engine=branch_engine,
        placement_noise_m=0.6,  # noisier placements in this branch
    )
    bart_b = branch_log["_artifacts"]
    print(f"  Episode dir: {bart_b['dir']}")
    print(f"  Branch duration: {branch_remaining:.2f}s (resumed at {branch_plan_t0:.2f}s)")
    print(f"  Final pos: {tuple(round(float(x), 3) for x in branch_engine.position)}")
    print(f"  Disk usage:  {bart_b['bytes'] / 1024:.1f} KiB")

    # ------------------------------------------------------------------
    banner("4/4  MissionMetricsAnalyzer -- KPIs across both runs")
    # ------------------------------------------------------------------
    analyzer = MissionMetricsAnalyzer()
    baseline_metrics = analyzer.compute_run_metrics(baseline_log)
    branch_metrics = analyzer.compute_run_metrics(branch_log)
    df = analyzer.to_dataframe([baseline_metrics, branch_metrics])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(df.to_string(index=False))

    print("\n  Cross-run statistics:")
    stats = analyzer.compare_across_runs([baseline_metrics, branch_metrics])
    for kpi in (
        "total_distance_m",
        "energy_consumed_wh",
        "placement_success_rate",
        "localization_error_drift_m",
    ):
        s = stats[kpi]
        print(
            f"    {kpi:32s} mean={s['mean']:.3f}  "
            f"std={s['std']:.3f}  min={s['min']:.3f}  max={s['max']:.3f}"
        )

    failure_modes = analyzer.failure_mode_analysis([baseline_metrics, branch_metrics])
    print("\n  Failure mode summary:")
    if failure_modes["failure_modes"]:
        for row in failure_modes["failure_modes"]:
            print(
                f"    {row['mode']:32s} count={row['count']}  "
                f"pct={row['percentage']:.1f}%"
            )
    else:
        print("    (no failures recorded)")

    cable_health = analyzer.cable_health_report(baseline_log["cable_tension_n"])
    print("\n  Baseline cable health:")
    print(json.dumps(cable_health, indent=4, default=float))

    banner("Demo complete")
    print(f"  Artifacts retained at: {output_dir}")
    if video_path is not None:
        print(f"  >> Driving video:   {video_path}")
        print("     Play it alongside the KPI table above to compare visual")
        print("     behaviour against the logged data (frames are sim_time-stamped).")
    print("  Inspect data:")
    print(f"    h5dump  {output_dir / 'baseline' / 'log.h5'}")
    print(f"    mcap info {output_dir / 'baseline' / 'log.mcap'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
