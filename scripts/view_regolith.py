"""Visualize lunar regolith deformation (System 2.2).

Two modes:

* ``--mode analytic`` (default, CPU, no GPU): drives the deterministic
  Bekker-Wong rut field with a simulated wheel track and renders the
  resulting sinkage as a heat-map, a 3-D surface, and a cross-section.
  Runnable anywhere.

* ``--mode mpm`` (GPU + Genesis viewer): builds the real Genesis MPM sand
  bed via ``GenesisMPMRegolith``, drops a heavy "wheel" block onto it, and
  opens the Genesis viewer so you can watch the granular material deform.
  Requires a CUDA backend + Genesis 0.4.4.

Examples
--------
Watch ruts form from repeated wheel passes (CPU)::

    C:\\ve\\.genesis\\Scripts\\python.exe scripts/view_regolith.py

Save instead of showing::

    C:\\ve\\.genesis\\Scripts\\python.exe scripts/view_regolith.py --save rut.png

Watch real MPM sand deform in the Genesis viewer (GPU)::

    C:\\ve\\.genesis\\Scripts\\python.exe scripts/view_regolith.py --mode mpm
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from moon_rover.environment.regolith import GenesisMPMRegolith, RegolithConfig
from moon_rover.environment.terrain.generator import TerrainOutput


def _flat_terrain(res: int = 64) -> TerrainOutput:
    return TerrainOutput(
        height_field=np.zeros((res, res), dtype=np.float32),
        slope_map=np.zeros((res, res), dtype=np.float32),
        normal_map=np.tile(np.array([0, 0, 1], np.float32), (res, res, 1)),
        rock_positions=[],
        crater_list=[],
        nav_mesh=np.ones((res, res), dtype=np.uint8),
    )


# --------------------------------------------------------------------------- #
# Analytic rut-field mode (CPU)
# --------------------------------------------------------------------------- #
def run_analytic(args: argparse.Namespace) -> int:
    size = args.size
    sim = GenesisMPMRegolith(terrain_size_m=size, wheel_radius_m=args.wheel_radius)
    sim.initialize(RegolithConfig(), _flat_terrain())

    # Drive a straight wheel track across the middle, several passes.
    y = size * 0.5
    xs = np.linspace(size * 0.15, size * 0.85, args.steps_per_pass)
    for lap in range(args.passes):
        for x in xs:
            sim.apply_wheel_pass(
                np.array([x, y, 0.0], dtype=np.float32),
                wheel_load_n=args.wheel_load,
                contact_radius_m=args.contact_radius,
            )

    # Sample the rut field on a dense grid via the public API.
    n = 220
    axis = np.linspace(0.0, size, n)
    gx, gy = np.meshgrid(axis, axis)
    rut = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            rut[i, j] = sim.get_sinkage_at(
                np.array([gx[i, j], gy[i, j], 0.0], dtype=np.float32)
            )

    peak = rut.max()
    print(
        f"passes={args.passes} load={args.wheel_load:.0f}N "
        f"contact_r={args.contact_radius:.3f}m -> peak sinkage "
        f"{peak * 1000:.1f} mm over a {size:.0f}m bed"
    )
    if peak <= 0.0:
        print("No sinkage produced — try a higher --wheel-load.")

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5))
    extent = (0.0, size, 0.0, size)

    ax1 = fig.add_subplot(1, 3, 1)
    im = ax1.imshow(rut * 1000.0, extent=extent, origin="lower", cmap="magma")
    plt.colorbar(im, ax=ax1, label="sinkage (mm)", fraction=0.046, pad=0.04)
    ax1.set_title("Rut depth (top view)")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    step = max(1, n // 120)
    ax2.plot_surface(
        gx[::step, ::step], gy[::step, ::step],
        -rut[::step, ::step] * 1000.0,
        cmap="magma", linewidth=0, antialiased=True,
    )
    ax2.set_title("Deformed surface")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_zlabel("z (mm)")

    ax3 = fig.add_subplot(1, 3, 3)
    mid = n // 2
    ax3.plot(axis, -rut[mid, :] * 1000.0, color="firebrick")
    ax3.fill_between(axis, -rut[mid, :] * 1000.0, 0.0,
                     color="firebrick", alpha=0.25)
    ax3.set_title(f"Cross-section at y={y:.0f} m")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("surface z (mm)")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Lunar regolith rut — {args.passes} passes, {args.wheel_load:.0f} N",
        fontsize=13,
    )
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"saved -> {args.save}")
    else:
        plt.show()
    return 0


# --------------------------------------------------------------------------- #
# Genesis MPM viewer mode (GPU)
# --------------------------------------------------------------------------- #
def run_mpm(args: argparse.Namespace) -> int:
    from dataclasses import replace

    import genesis as gs

    from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine
    from moon_rover.core.physics.engine import GenesisConfig

    base = GenesisConfig.from_yaml(args.config)
    # MPM CFL: substep_dt = timestep/substeps must stay < 2e-2*dx (dx=1/64).
    # At the config timestep (1/240) that needs substeps >= 14; 20 gives
    # transient margin. substeps=1-4 leaves the sand in perpetual motion.
    config = replace(base, use_gpu=True, substeps=20)

    viewer_options = gs.options.ViewerOptions(
        max_FPS=args.render_hz,
        refresh_rate=args.render_hz,
        run_in_thread=True,
        camera_pos=(2.0, -2.6, 1.7),
        camera_lookat=(0.0, 0.0, 0.2),
        camera_fov=45,
    )

    print("=" * 70)
    print("  Moon Rover - Regolith MPM viewer  (GPU)")
    print("  A heavy block drops onto the MPM sand bed; watch it deform.")
    print("  Controls: mouse orbit/zoom | A auto-rotate | Z reset camera")
    print("=" * 70)

    engine = GenesisPhysicsEngine()
    completed = False
    try:
        engine.configure(config, show_viewer=True, viewer_options=viewer_options)

        sim = GenesisMPMRegolith(engine=engine, terrain_size_m=2.0)
        # This viewer exists to show the MPM bed, so opt in explicitly.
        sim.initialize(RegolithConfig(mpm_enabled=True), _flat_terrain(res=32))
        lo, hi = sim._mpm_bed_bounds  # diagnostics: where the bed sits
        cx = 0.5 * (lo[0] + hi[0])
        cy = 0.5 * (lo[1] + hi[1])
        # Park the block high enough that the bed visibly settles first,
        # then it falls and forms a real depression.
        drop_z = hi[2] + 0.6
        print(f"  MPM bed bounds: {lo} -> {hi}")
        print(f"  Wheel block parked at ({cx:.2f}, {cy:.2f}, {drop_z:.2f})")

        engine.add_entity(
            "wheel_block",
            gs.morphs.Box(pos=(cx, cy, drop_z), size=(0.25, 0.25, 0.25)),
            gs.materials.Rigid(rho=3000.0),
            entity_type="rigid",
        )

        engine.build_scene(n_envs=1)

        settle_steps = int(args.settle_s / config.timestep)
        print(f"  Settling the bed for {args.settle_s:.1f}s "
              f"({settle_steps} steps)...")
        for _ in range(settle_steps):
            engine.step(config.timestep, render=True)
            sim.step(config.timestep)
        print(f"  Settled mean particle speed: "
              f"{sim.mean_particle_speed():.4f} m/s (should be near 0)")

        n_steps = int(args.duration_s / config.timestep)
        print(f"  Running {args.duration_s:.1f}s — block falls onto the bed. "
              "Close the viewer to exit.")
        for _ in range(n_steps):
            engine.step(config.timestep, render=True)
            sim.step(config.timestep)
        completed = True
    finally:
        try:
            engine.teardown()
        except Exception:  # noqa: BLE001 - best-effort viewer teardown
            pass
    return 0 if completed else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Visualize regolith deformation.")
    p.add_argument("--mode", choices=["analytic", "mpm"], default="analytic")
    # analytic options
    p.add_argument("--size", type=float, default=8.0, help="Bed side (m).")
    p.add_argument("--passes", type=int, default=6,
                   help="Number of wheel passes over the track.")
    p.add_argument("--steps-per-pass", type=int, default=140,
                   help="Wheel contacts per pass; higher = continuous rut.")
    p.add_argument("--wheel-load", type=float, default=350.0, help="N.")
    p.add_argument("--contact-radius", type=float, default=0.12, help="m.")
    p.add_argument("--wheel-radius", type=float, default=0.30, help="m.")
    p.add_argument("--save", metavar="PATH", default=None,
                   help="Write the figure to PATH instead of showing it.")
    # mpm options
    p.add_argument("--config", default="configs/physics.yaml")
    p.add_argument("--render-hz", type=int, default=30)
    p.add_argument("--settle-s", type=float, default=2.0,
                   help="Seconds to let the bed settle before the block drops.")
    p.add_argument("--duration-s", type=float, default=4.0)
    args = p.parse_args(argv)

    if args.mode == "mpm":
        try:
            return run_mpm(args)
        except RuntimeError as exc:
            print(f"\nMPM viewer unavailable: {exc}\n"
                  "This mode needs Genesis 0.4.4 + a CUDA GPU. Use the "
                  "default --mode analytic on CPU.", file=sys.stderr)
            return 1
    return run_analytic(args)


if __name__ == "__main__":
    sys.exit(main())
