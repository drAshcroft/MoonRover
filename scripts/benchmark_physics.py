"""Machine-readable physics benchmarks for the Moon Rover Genesis adapter.

This script complements `scripts/verify_physics.py`: the verifier stays focused on
human-in-the-loop visualization, while this benchmark emits timing metrics and
threshold checks suitable for production readiness reviews.

Examples
--------
    python scripts/benchmark_physics.py --scene many-boxes --backend cpu --json
    python scripts/benchmark_physics.py --scene many-boxes --backend gpu --assert-thresholds
    python scripts/benchmark_physics.py --scene rover-proxy --mode viewer --steps 240 --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import replace
from typing import Any, Dict, NamedTuple

import numpy as np

sys.path.insert(0, "src")

import genesis as gs

from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine
from moon_rover.core.physics.engine import GenesisConfig


class BodySpec(NamedTuple):
    name: str
    size: tuple[float, float, float]
    pos: tuple[float, float, float]
    lin_vel: tuple[float, float, float]
    ang_vel: tuple[float, float, float]


DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "headless:cpu:many-boxes": {
        "min_steps_per_second": 25.0,
        "min_real_time_factor": 0.20,
        "max_build_seconds": 300.0,
        "max_teardown_seconds": 15.0,
    },
    "headless:gpu:many-boxes": {
        "min_steps_per_second": 25.0,
        "min_real_time_factor": 0.20,
        "max_build_seconds": 600.0,
        "max_teardown_seconds": 15.0,
    },
    "viewer:cpu:many-boxes": {
        "min_render_rate_hz": 15.0,
        "max_frame_interval_jitter_s": 0.20,
        "max_build_seconds": 300.0,
        "max_teardown_seconds": 15.0,
    },
    "viewer:gpu:many-boxes": {
        "min_render_rate_hz": 15.0,
        "max_frame_interval_jitter_s": 0.20,
        "max_build_seconds": 600.0,
        "max_teardown_seconds": 15.0,
    },
    "headless:cpu:rover-proxy": {
        "min_steps_per_second": 20.0,
        "min_real_time_factor": 0.15,
        "max_build_seconds": 300.0,
        "max_teardown_seconds": 15.0,
    },
    "headless:gpu:rover-proxy": {
        "min_steps_per_second": 20.0,
        "min_real_time_factor": 0.15,
        "max_build_seconds": 600.0,
        "max_teardown_seconds": 15.0,
    },
    "viewer:cpu:rover-proxy": {
        "min_render_rate_hz": 15.0,
        "max_frame_interval_jitter_s": 0.20,
        "max_build_seconds": 300.0,
        "max_teardown_seconds": 15.0,
    },
    "viewer:gpu:rover-proxy": {
        "min_render_rate_hz": 15.0,
        "max_frame_interval_jitter_s": 0.20,
        "max_build_seconds": 600.0,
        "max_teardown_seconds": 15.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Moon Rover Genesis adapter")
    parser.add_argument("--scene", choices=("many-boxes", "rover-proxy"), default="many-boxes")
    parser.add_argument("--mode", choices=("headless", "viewer"), default="headless")
    parser.add_argument("--backend", choices=("cpu", "gpu", "config"), default="cpu")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--sim-hz", type=float, default=120.0)
    parser.add_argument("--render-hz", type=int, default=60)
    parser.add_argument("--bodies", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--config", default="configs/physics.yaml")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    parser.add_argument(
        "--assert-thresholds",
        action="store_true",
        help="Exit non-zero when benchmark thresholds are not met",
    )
    parser.add_argument(
        "--skip-teardown",
        action="store_true",
        help="Leave cleanup to process exit instead of timing engine.teardown()",
    )
    return parser.parse_args()


def _resolve_backend(base: GenesisConfig, backend_arg: str) -> tuple[bool, str]:
    if backend_arg == "config":
        return base.use_gpu, "gpu" if base.use_gpu else "cpu"
    if backend_arg == "gpu":
        return True, "gpu"
    return False, "cpu"


def _threshold_profile_key(mode: str, backend: str, scene: str) -> str:
    return f"{mode}:{backend}:{scene}"


def _make_box_grid_specs(count: int, seed: int) -> list[BodySpec]:
    count = max(1, int(count))
    side = max(1, int(math.ceil(math.sqrt(count))))
    rng = np.random.default_rng(seed)
    specs: list[BodySpec] = []
    for idx in range(count):
        gx = idx % side
        gy = idx // side
        x = (gx - (side - 1) * 0.5) * 0.42
        y = (gy - (side - 1) * 0.5) * 0.42
        z = 0.6 + 0.09 * (idx % 5)
        lin = (
            float(rng.uniform(-0.2, 0.2)),
            float(rng.uniform(-0.2, 0.2)),
            float(rng.uniform(0.0, 0.4)),
        )
        ang = tuple(float(v) for v in rng.uniform(-1.0, 1.0, size=3))
        specs.append(
            BodySpec(
                name=f"box_{idx:03d}",
                size=(0.22, 0.22, 0.22),
                pos=(x, y, z),
                lin_vel=lin,
                ang_vel=ang,
            )
        )
    return specs


def _make_rover_proxy_specs(seed: int, extra_bodies: int) -> list[BodySpec]:
    rng = np.random.default_rng(seed)
    specs = [
        BodySpec("chassis", (0.9, 0.6, 0.18), (0.0, 0.0, 0.7), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        BodySpec("wheel_fl", (0.18, 0.18, 0.18), (0.38, 0.28, 0.32), (0.0, 0.0, 0.0), (0.0, 0.0, 4.0)),
        BodySpec("wheel_fr", (0.18, 0.18, 0.18), (0.38, -0.28, 0.32), (0.0, 0.0, 0.0), (0.0, 0.0, 4.0)),
        BodySpec("wheel_rl", (0.18, 0.18, 0.18), (-0.38, 0.28, 0.32), (0.0, 0.0, 0.0), (0.0, 0.0, 4.0)),
        BodySpec("wheel_rr", (0.18, 0.18, 0.18), (-0.38, -0.28, 0.32), (0.0, 0.0, 0.0), (0.0, 0.0, 4.0)),
    ]
    for idx in range(max(0, extra_bodies)):
        specs.append(
            BodySpec(
                name=f"cargo_{idx:03d}",
                size=(0.14, 0.14, 0.14),
                pos=(float(rng.uniform(-1.2, 1.2)), float(rng.uniform(-1.2, 1.2)), 0.45),
                lin_vel=(0.0, 0.0, 0.0),
                ang_vel=(0.0, 0.0, 0.0),
            )
        )
    return specs


def _build_specs(scene: str, bodies: int, seed: int) -> list[BodySpec]:
    if scene == "rover-proxy":
        return _make_rover_proxy_specs(seed, extra_bodies=max(0, int(bodies) - 5))
    return _make_box_grid_specs(bodies, seed)


def _render_interval(sim_hz: float, render_hz: int) -> int:
    return max(1, int(round(max(sim_hz, 1.0) / max(render_hz, 1))))


def _evaluate_thresholds(metrics: Dict[str, Any]) -> tuple[bool, list[str]]:
    thresholds = metrics["thresholds"]
    metric_name_map = {
        "min_steps_per_second": "steps_per_second",
        "min_real_time_factor": "real_time_factor",
        "min_render_rate_hz": "render_rate_hz",
        "max_build_seconds": "build_seconds",
        "max_teardown_seconds": "teardown_seconds",
        "max_frame_interval_jitter_s": "frame_interval_jitter_s",
    }
    violations: list[str] = []
    for key, threshold in thresholds.items():
        value = metrics.get(metric_name_map.get(key, key))
        if value is None:
            continue
        if key.startswith("min_") and float(value) < float(threshold):
            violations.append(f"{key}={value:.6g} < {threshold:.6g}")
        if key.startswith("max_") and float(value) > float(threshold):
            violations.append(f"{key}={value:.6g} > {threshold:.6g}")
    return not violations, violations


def _viewer_options(render_hz: int) -> Any:
    return gs.options.ViewerOptions(
        max_FPS=render_hz,
        refresh_rate=render_hz,
        run_in_thread=True,
        camera_pos=(4.2, -4.4, 2.8),
        camera_lookat=(0.0, 0.0, 0.8),
        camera_fov=45,
    )


def main() -> int:
    args = parse_args()
    base = GenesisConfig.from_yaml(args.config)
    use_gpu, backend_label = _resolve_backend(base, args.backend)
    timestep = base.timestep if args.sim_hz <= 0 else 1.0 / args.sim_hz
    config = replace(base, use_gpu=use_gpu, timestep=timestep, random_seed=args.seed)
    specs = _build_specs(args.scene, args.bodies, args.seed)
    sim_hz = 1.0 / config.timestep
    render_every = _render_interval(sim_hz, args.render_hz)
    show_viewer = args.mode == "viewer"
    threshold_key = _threshold_profile_key(args.mode, backend_label, args.scene)
    thresholds = DEFAULT_THRESHOLDS[threshold_key]

    engine = GenesisPhysicsEngine()

    configure_seconds = 0.0
    build_seconds = 0.0
    loop_seconds = 0.0
    teardown_seconds = 0.0
    teardown_outcome = "skipped"
    runtime_initialized_after = None
    render_timestamps: list[float] = []
    step_durations: list[float] = []

    try:
        t0 = time.perf_counter()
        engine.configure(
            config,
            show_viewer=show_viewer,
            viewer_options=_viewer_options(args.render_hz) if show_viewer else None,
        )
        configure_seconds = time.perf_counter() - t0

        material = gs.materials.Rigid(
            rho=GenesisPhysicsEngine._terrain_density_kg_m3,
            friction=GenesisPhysicsEngine._terrain_contact_friction,
        )
        engine.add_entity("ground", gs.morphs.Plane(), material, entity_type="terrain")
        for spec in specs:
            engine.add_entity(
                spec.name,
                gs.morphs.Box(size=spec.size, pos=spec.pos),
                gs.materials.Rigid(rho=900, friction=0.9),
                entity_type="rigid",
            )

        t0 = time.perf_counter()
        engine.build_scene()
        build_seconds = time.perf_counter() - t0

        for spec in specs:
            engine.set_body_velocity(
                spec.name,
                np.asarray(spec.lin_vel, dtype=np.float32),
                np.asarray(spec.ang_vel, dtype=np.float32),
            )

        for _ in range(max(0, args.warmup)):
            engine.step(config.timestep, render=False)

        loop_t0 = time.perf_counter()
        for step_idx in range(max(1, args.steps)):
            should_render = show_viewer and (step_idx % render_every == 0)
            step_t0 = time.perf_counter()
            engine.step(config.timestep, render=should_render)
            step_t1 = time.perf_counter()
            step_durations.append(step_t1 - step_t0)
            if should_render:
                render_timestamps.append(step_t1)
        loop_seconds = time.perf_counter() - loop_t0

    finally:
        if not args.skip_teardown:
            t0 = time.perf_counter()
            try:
                engine.teardown()
            finally:
                teardown_seconds = time.perf_counter() - t0
                runtime_initialized_after = GenesisPhysicsEngine._gs_initialized
                teardown_outcome = "retained" if runtime_initialized_after else "destroyed"

    rendered_frames = len(render_timestamps)
    render_intervals = [
        b - a for a, b in zip(render_timestamps[:-1], render_timestamps[1:])
    ]
    render_rate_hz = (
        rendered_frames / loop_seconds if loop_seconds > 0.0 and rendered_frames > 0 else 0.0
    )
    frame_interval_jitter_s = (
        statistics.pstdev(render_intervals) if len(render_intervals) >= 2 else 0.0
    )
    steps_per_second = (max(1, args.steps) / loop_seconds) if loop_seconds > 0.0 else 0.0
    simulated_seconds = max(1, args.steps) * config.timestep
    real_time_factor = (simulated_seconds / loop_seconds) if loop_seconds > 0.0 else 0.0

    metrics: Dict[str, Any] = {
        "scene": args.scene,
        "mode": args.mode,
        "backend": backend_label,
        "bodies": len(specs),
        "steps": max(1, args.steps),
        "warmup_steps": max(0, args.warmup),
        "dt_seconds": config.timestep,
        "sim_hz": sim_hz,
        "render_hz_target": args.render_hz if show_viewer else 0,
        "configure_seconds": configure_seconds,
        "build_seconds": build_seconds,
        "loop_seconds": loop_seconds,
        "teardown_seconds": teardown_seconds,
        "simulated_seconds": simulated_seconds,
        "steps_per_second": steps_per_second,
        "real_time_factor": real_time_factor,
        "rendered_frames": rendered_frames,
        "render_rate_hz": render_rate_hz,
        "frame_interval_jitter_s": frame_interval_jitter_s,
        "avg_step_seconds": statistics.fmean(step_durations) if step_durations else 0.0,
        "max_step_seconds": max(step_durations) if step_durations else 0.0,
        "teardown_outcome": teardown_outcome,
        "runtime_initialized_after_teardown": runtime_initialized_after,
        "destroy_policy": GenesisPhysicsEngine._resolve_destroy_policy(),
        "threshold_profile": threshold_key,
        "thresholds": thresholds,
        "python_encoding": os.environ.get("PYTHONIOENCODING", ""),
    }
    passed, violations = _evaluate_thresholds(metrics)
    metrics["thresholds_passed"] = passed
    metrics["threshold_violations"] = violations

    if args.json:
        print(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        print("=" * 78)
        print("Moon Rover physics benchmark")
        print("=" * 78)
        print(f"Scene      : {metrics['scene']} ({metrics['bodies']} bodies)")
        print(f"Mode       : {metrics['mode']} / {metrics['backend']}")
        print(f"Sim rate   : {metrics['sim_hz']:.1f} Hz")
        print(f"Build      : {metrics['build_seconds']:.3f}s")
        print(f"Loop       : {metrics['loop_seconds']:.3f}s")
        print(f"Steps/s    : {metrics['steps_per_second']:.2f}")
        print(f"RTF        : {metrics['real_time_factor']:.2f}")
        if show_viewer:
            print(f"Render/s   : {metrics['render_rate_hz']:.2f}")
            print(f"Jitter     : {metrics['frame_interval_jitter_s']:.4f}s")
        print(
            f"Teardown   : {metrics['teardown_outcome']} in "
            f"{metrics['teardown_seconds']:.3f}s "
            f"(policy={metrics['destroy_policy']})"
        )
        print(f"Thresholds : {'PASS' if passed else 'FAIL'} [{metrics['threshold_profile']}]")
        if violations:
            for violation in violations:
                print(f"  - {violation}")
        print("-" * 78)
        print(json.dumps(metrics, sort_keys=True))

    if args.assert_thresholds and not passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
