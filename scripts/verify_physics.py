"""Interactive Genesis physics verifier and box-launch demo.

The default mode is tuned for a smooth rigid-body viewer demo on this project:
CPU backend, 60 Hz simulation, and a launcher grid that throws a cloud of boxes
into lunar gravity. Use --backend config --sim-hz 0 for the raw physics.yaml
configuration, or --no-viewer for headless throughput checks.

Usage
-----
    python scripts/verify_physics.py
    python scripts/verify_physics.py --boxes 24
    python scripts/verify_physics.py --backend config --sim-hz 0
    python scripts/verify_physics.py --no-viewer --steps 500

For machine-readable throughput and viewer-pacing gates, use
`python scripts/benchmark_physics.py ...` instead of this interactive verifier.

Viewer controls: left-drag = orbit | right-drag = zoom | scroll = zoom
Genesis hotkeys: A = auto-rotate | D = wireframe cycle | Z = reset camera
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from typing import NamedTuple

import numpy as np

sys.path.insert(0, "src")

from moon_rover.core.physics.engine import GenesisConfig
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine

import genesis as gs


class BoxSpec(NamedTuple):
    name: str
    pos: np.ndarray
    quat: np.ndarray
    lin_vel: np.ndarray
    ang_vel: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Moon Rover physics engine verifier")
    parser.add_argument("--steps", type=int, default=0,
                        help="Sim steps to run then exit (0 = run until Ctrl+C)")
    parser.add_argument("--interactive", action="store_true",
                        help="Pause after each simulation step, press Enter to advance")
    parser.add_argument("--no-viewer", action="store_true",
                        help="Headless: run as fast as possible unless --real-time is set")
    parser.add_argument("--real-time", action="store_true",
                        help="Pace headless mode to simulated real time")
    parser.add_argument("--render-hz", type=int, default=60,
                        help="Viewer refresh cap in Hz (default 60)")
    parser.add_argument("--render-mode",
                        choices=("adaptive", "every-step", "fixed"),
                        default="adaptive",
                        help="Viewer cadence; adaptive renders every step if sim is slow")
    parser.add_argument("--sim-hz", type=float, default=60.0,
                        help="Simulation Hz for the demo (0 = use config timestep)")
    parser.add_argument("--backend",
                        choices=("auto", "config", "cpu", "gpu"),
                        default="auto",
                        help="Backend override; auto uses CPU for this rigid demo")
    parser.add_argument("--boxes", type=int, default=12,
                        help="Number of launched boxes (default 12)")
    parser.add_argument("--box-size", type=float, default=0.25,
                        help="Cube edge length in metres")
    parser.add_argument("--spread", type=float, default=0.55,
                        help="Launch grid spacing in metres")
    parser.add_argument("--launch-speed", type=float, default=3.2,
                        help="Base upward launch speed in m/s")
    parser.add_argument("--launch-jitter", type=float, default=1.0,
                        help="Random upward launch speed variation in m/s")
    parser.add_argument("--side-speed", type=float, default=0.9,
                        help="Maximum lateral launch speed in m/s")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for repeatable box launches")
    parser.add_argument("--substeps", type=int, default=0,
                        help="Override Genesis substeps (0 = config/default)")
    parser.add_argument("--contact-iterations", type=int, default=0,
                        help="Override rigid contact iterations (0 = config/default)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Silent prelaunch warm-up steps, then reset boxes")
    parser.add_argument("--snapshot-check", action="store_true",
                        help="Run snapshot round-trip sanity check before teardown")
    parser.add_argument("--destroy", action="store_true",
                        help="Call engine teardown before exit; global Genesis destroy follows MOON_ROVER_GENESIS_DESTROY_POLICY")
    parser.add_argument("--config", default="configs/physics.yaml",
                        help="Path to physics.yaml")
    return parser.parse_args()


def configure_for_demo(base: GenesisConfig, args: argparse.Namespace) -> tuple[GenesisConfig, str]:
    if args.backend == "auto":
        use_gpu = False
        backend_label = "CPU (auto: rigid viewer demo)"
    elif args.backend == "config":
        use_gpu = base.use_gpu
        backend_label = "GPU (config)" if use_gpu else "CPU (config)"
    elif args.backend == "gpu":
        use_gpu = True
        backend_label = "GPU (override)"
    else:
        use_gpu = False
        backend_label = "CPU (override)"

    timestep = base.timestep if args.sim_hz <= 0 else 1.0 / args.sim_hz
    substeps = base.substeps if args.substeps <= 0 else args.substeps
    contact_iterations = (
        base.contact_iterations if args.contact_iterations <= 0 else args.contact_iterations
    )
    config = replace(
        base,
        timestep=timestep,
        substeps=substeps,
        contact_iterations=contact_iterations,
        use_gpu=use_gpu,
        random_seed=args.seed,
    )
    return config, backend_label


def make_box_specs(args: argparse.Namespace) -> list[BoxSpec]:
    rng = np.random.default_rng(args.seed)
    count = max(1, args.boxes)
    specs: list[BoxSpec] = []

    for idx in range(count):
        x = (idx - (count - 1) * 0.5) * args.spread
        y = 0.35 * math.sin(idx * 0.9)
        z = 1.0 + 0.08 * (idx % 4)

        angle = (2.0 * math.pi * idx / count) + rng.uniform(-0.25, 0.25)
        side_speed = rng.uniform(0.15, args.side_speed)
        upward = args.launch_speed + rng.uniform(-args.launch_jitter, args.launch_jitter)

        lin_vel = np.array(
            [
                math.cos(angle) * side_speed,
                math.sin(angle) * side_speed,
                max(0.5, upward),
            ],
            dtype=np.float32,
        )
        ang_vel = rng.uniform(-6.0, 6.0, size=3).astype(np.float32)
        specs.append(
            BoxSpec(
                name=f"box_{idx:03d}",
                pos=np.array([x, y, z], dtype=np.float32),
                quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                lin_vel=lin_vel,
                ang_vel=ang_vel,
            )
        )
    return specs


def add_box(engine: GenesisPhysicsEngine, spec: BoxSpec, size: float) -> None:
    morph = gs.morphs.Box(size=[size, size, size], pos=tuple(spec.pos))
    material = gs.materials.Rigid(rho=900)
    engine.add_entity(spec.name, morph, material)


def reset_and_launch(
    engine: GenesisPhysicsEngine,
    specs: list[BoxSpec],
    *,
    reset_pose: bool = False,
) -> None:
    for spec in specs:
        if reset_pose:
            engine.set_body_pose(spec.name, spec.pos, spec.quat)
        engine.set_body_velocity(spec.name, spec.lin_vel, spec.ang_vel)


def sample_heights(engine: GenesisPhysicsEngine, names: list[str]) -> tuple[float, float]:
    heights = [float(engine.get_body_pose(name)[0][2]) for name in names]
    return heights[0], max(heights)


def main() -> int:
    args = parse_args()
    base_config = GenesisConfig.from_yaml(args.config)
    config, backend_label = configure_for_demo(base_config, args)
    sim_hz = 1.0 / config.timestep
    nominal_render_interval = max(1, round(sim_hz / max(1, args.render_hz)))

    viewer_options = None
    if not args.no_viewer:
        viewer_options = gs.options.ViewerOptions(
            max_FPS=args.render_hz,
            refresh_rate=args.render_hz,
            run_in_thread=True,
            camera_pos=(4.5, -5.5, 3.2),
            camera_lookat=(0.0, 0.0, 1.1),
            camera_fov=45,
        )

    print("=" * 78)
    print("  Moon Rover - Physics Box Launcher")
    print("=" * 78)
    print(f"  Backend    : {backend_label}")
    print(f"  Gravity    : {config.gravity_vector}")
    print(f"  Sim rate   : {sim_hz:.1f} Hz  (dt = {config.timestep * 1000.0:.3f} ms)")
    print(f"  Solver     : substeps={config.substeps}, contact iters={config.contact_iterations}")
    print(f"  Boxes      : {max(1, args.boxes)} launched rigid bodies")
    print(f"  Viewer     : {'off' if args.no_viewer else f'on, {args.render_hz} Hz cap'}")
    print(
        f"  Teardown   : {'on exit' if args.destroy else 'process-exit cleanup'}"
        f" ({GenesisPhysicsEngine._resolve_destroy_policy()} policy)"
    )
    print("  Controls   : mouse orbit/zoom | A auto-rotate | D wireframe | Z reset camera")
    print("=" * 78)

    engine = GenesisPhysicsEngine()
    completed_cleanly = False

    try:
        t0 = time.perf_counter()
        engine.configure(config, show_viewer=not args.no_viewer, viewer_options=viewer_options)
        print(f"  Engine configured in {time.perf_counter() - t0:.2f}s")

        engine.add_entity("ground", gs.morphs.Plane(), gs.materials.Rigid())

        specs = make_box_specs(args)
        for spec in specs:
            add_box(engine, spec, args.box_size)

        t0 = time.perf_counter()
        engine.build_scene()
        print(f"  Scene built in {time.perf_counter() - t0:.2f}s")

        if args.warmup > 0:
            print(f"  Warm-up: {args.warmup} hidden steps, then reset and launch")
            for _ in range(args.warmup):
                engine.step(config.timestep, render=False)

        reset_and_launch(engine, specs, reset_pose=args.warmup > 0)
        launch_sim_t0 = engine.get_sim_time()

        print("-" * 78)
        print("  Step | Sim time |  RTF  | Steps/s | Render/s | Box0 z | Max z")
        print("-" * 78)

        step_count = 0
        render_count = 0
        recent_rtf = 1.0
        report_step = 0
        report_render = 0
        wall_start = time.perf_counter()
        report_t0 = wall_start
        pace_to_real_time = args.real_time or not args.no_viewer
        sample_names = [spec.name for spec in specs[: min(len(specs), 64)]]

        while True:
            if args.interactive:
                input(f"  [step {step_count}] Press Enter to advance one step ...")

            if args.render_mode == "every-step":
                render_interval = 1
            elif args.render_mode == "fixed":
                render_interval = nominal_render_interval
            else:
                render_interval = 1 if recent_rtf < 0.95 else nominal_render_interval

            render_this = (not args.no_viewer) and (step_count % render_interval == 0)
            engine.step(config.timestep, render=render_this)
            step_count += 1
            if render_this:
                render_count += 1

            if pace_to_real_time:
                target_wall = wall_start + step_count * config.timestep
                sleep_for = target_wall - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)

            now = time.perf_counter()
            should_report = step_count <= 3 or (now - report_t0) >= 0.5
            if should_report:
                elapsed = max(now - report_t0, 1e-9)
                steps_s = (step_count - report_step) / elapsed
                render_s = (render_count - report_render) / elapsed
                recent_rtf = steps_s * config.timestep
                z0, zmax = sample_heights(engine, sample_names)
                sim_time = engine.get_sim_time() - launch_sim_t0
                print(
                    f"  {step_count:5d} | {sim_time:8.3f} | {recent_rtf:5.2f} |"
                    f" {steps_s:7.1f} | {render_s:8.1f} | {z0:+6.3f} | {zmax:+6.3f}"
                )
                report_t0 = now
                report_step = step_count
                report_render = render_count

            if args.steps and step_count >= args.steps:
                completed_cleanly = True
                break

    except KeyboardInterrupt:
        print("\n  Stopped by user")
        completed_cleanly = True
    finally:
        if args.snapshot_check and completed_cleanly:
            print("\n  Snapshot round-trip check...")
            primary_name = "box_000"
            snap = engine.save_snapshot()
            saved_pos, saved_quat = engine.get_body_pose(primary_name)
            saved_lin, saved_ang = engine.get_body_velocity(primary_name)
            saved_time = engine.get_sim_time()
            saved_step = engine.get_step_count()
            engine.step(config.timestep, render=False)
            replay_pos_expected, replay_quat_expected = engine.get_body_pose(primary_name)
            replay_lin_expected, replay_ang_expected = engine.get_body_velocity(primary_name)
            for _ in range(max(0, min(3, nominal_render_interval - 1))):
                engine.step(config.timestep, render=False)
            engine.restore_snapshot(snap)
            restored_pos, restored_quat = engine.get_body_pose(primary_name)
            restored_lin, restored_ang = engine.get_body_velocity(primary_name)
            restore_pose_delta = max(
                float(np.max(np.abs(restored_pos - saved_pos))),
                float(np.max(np.abs(restored_quat - saved_quat))),
            )
            restore_vel_delta = max(
                float(np.max(np.abs(restored_lin - saved_lin))),
                float(np.max(np.abs(restored_ang - saved_ang))),
            )
            restore_time_ok = abs(engine.get_sim_time() - saved_time) < 1e-9
            restore_step_ok = engine.get_step_count() == saved_step

            engine.step(config.timestep, render=False)
            replay_pos, replay_quat = engine.get_body_pose(primary_name)
            replay_lin, replay_ang = engine.get_body_velocity(primary_name)
            replay_pose_delta = max(
                float(np.max(np.abs(replay_pos - replay_pos_expected))),
                float(np.max(np.abs(replay_quat - replay_quat_expected))),
            )
            replay_vel_delta = max(
                float(np.max(np.abs(replay_lin - replay_lin_expected))),
                float(np.max(np.abs(replay_ang - replay_ang_expected))),
            )
            ok = (
                restore_pose_delta < 1e-3
                and restore_vel_delta < 1e-3
                and replay_pose_delta < 1e-3
                and replay_vel_delta < 1e-3
                and restore_time_ok
                and restore_step_ok
            )
            print(f"  Snapshot  : {len(snap)} bytes")
            print(
                "  Restore   : "
                f"pose/quaternion max delta={restore_pose_delta:.3e}, "
                f"velocity max delta={restore_vel_delta:.3e}, "
                f"time={'ok' if restore_time_ok else 'FAIL'}, "
                f"step={'ok' if restore_step_ok else 'FAIL'}"
            )
            print(
                "  Replay    : "
                f"pose/quaternion max delta={replay_pose_delta:.3e}, "
                f"velocity max delta={replay_vel_delta:.3e}  "
                f"({'PASS' if ok else 'FAIL'})"
            )

        if args.destroy:
            try:
                engine.teardown()
            except Exception:
                pass

    print("\n  Physics demo complete.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
