"""Visualize generated lunar terrain (System 2.1).

Generates terrain with :class:`LunarTerrainGenerator` and renders it with
matplotlib: shaded relief, 3D surface, slope map, and the navigation mesh
with rock obstacles overlaid. No GPU or Genesis required.

Examples
--------
Interactive window with defaults::

    C:\\ve\\.genesis\\Scripts\\python.exe scripts/view_terrain.py

Reproduce a specific scene and save to a PNG instead of showing it::

    C:\\ve\\.genesis\\Scripts\\python.exe scripts/view_terrain.py \\
        --seed 99 --size 60 --resolution 256 --craters 12 \\
        --rock-density 0.02 --save terrain.png

Only the 3D surface::

    C:\\ve\\.genesis\\Scripts\\python.exe scripts/view_terrain.py --mode 3d
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from moon_rover.environment.terrain import LunarTerrainGenerator, TerrainConfig


def _build_config(args: argparse.Namespace) -> TerrainConfig:
    return TerrainConfig(
        seed=args.seed,
        size_m=args.size,
        fBm_octaves=args.octaves,
        fBm_amplitude=args.amplitude,
        crater_params={
            "count": args.craters,
            "min_radius_m": 1.0,
            "max_radius_m": max(5.0, args.size * 0.05),
            "depth_ratio": 0.3,
        },
        rock_density=args.rock_density,
        rille_enabled=not args.no_rilles,
        moonbase_position=(args.size / 2.0, args.size / 2.0, 0.0),
        resolution=args.resolution,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render generated lunar terrain.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=float, default=100.0, help="Terrain side (m).")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--octaves", type=int, default=8)
    p.add_argument("--amplitude", type=float, default=2.0, help="fBm height (m).")
    p.add_argument("--craters", type=int, default=15)
    p.add_argument("--rock-density", type=float, default=0.02,
                   help="Rocks per m^2.")
    p.add_argument("--no-rilles", action="store_true")
    p.add_argument("--mode", choices=["all", "2d", "3d"], default="all")
    p.add_argument("--save", metavar="PATH", default=None,
                   help="Write the figure to PATH instead of showing it.")
    args = p.parse_args(argv)

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    cfg = _build_config(args)
    out = LunarTerrainGenerator().generate(cfg)
    extent = (0.0, cfg.size_m, 0.0, cfg.size_m)

    print(
        f"seed={cfg.seed} size={cfg.size_m}m res={cfg.resolution} "
        f"height=[{out.height_field.min():.2f}, {out.height_field.max():.2f}]m "
        f"slope_max={out.slope_map.max():.1f}deg "
        f"craters={len(out.crater_list)} rocks={len(out.rock_positions)} "
        f"traversable={out.nav_mesh.mean() * 100:.1f}%"
    )

    if args.mode == "3d":
        fig = plt.figure(figsize=(10, 8))
        _plot_surface_3d(fig.add_subplot(111, projection="3d"), out, cfg)
    elif args.mode == "2d":
        fig, ax = plt.subplots(figsize=(9, 8))
        _plot_relief(ax, out, cfg, extent, LightSource, plt)
    else:
        fig = plt.figure(figsize=(14, 11))
        _plot_relief(fig.add_subplot(2, 2, 1), out, cfg, extent, LightSource, plt)
        _plot_surface_3d(fig.add_subplot(2, 2, 2, projection="3d"), out, cfg)
        _plot_slope(fig.add_subplot(2, 2, 3), out, extent, plt)
        _plot_navmesh(fig.add_subplot(2, 2, 4), out, extent, plt)

    fig.suptitle(
        f"Lunar terrain — seed {cfg.seed}, {cfg.size_m:.0f}m, "
        f"res {cfg.resolution}",
        fontsize=13,
    )
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"saved -> {args.save}")
    else:
        plt.show()
    return 0


def _plot_relief(ax, out, cfg, extent, LightSource, plt) -> None:
    ls = LightSource(azdeg=315, altdeg=35)
    shaded = ls.shade(
        out.height_field.astype(float), cmap=plt.cm.gray,
        vert_exag=2.0, blend_mode="overlay",
    )
    ax.imshow(shaded, extent=extent, origin="lower")
    if out.rock_positions:
        rp = np.array(out.rock_positions)
        ax.scatter(rp[:, 0], rp[:, 1], s=8, c="orangered",
                   label=f"{len(rp)} rocks")
        ax.legend(loc="upper right", fontsize=8)
    bx, by = cfg.moonbase_position[0], cfg.moonbase_position[1]
    ax.scatter([bx], [by], marker="*", s=220, c="deepskyblue",
               edgecolors="k", label="moonbase", zorder=5)
    ax.set_title("Shaded relief")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _plot_surface_3d(ax, out, cfg) -> None:
    res = cfg.resolution
    axis = np.linspace(0.0, cfg.size_m, res)
    xx, yy = np.meshgrid(axis, axis)
    # Downsample large grids so the 3D render stays interactive.
    step = max(1, res // 160)
    ax.plot_surface(
        xx[::step, ::step], yy[::step, ::step],
        out.height_field[::step, ::step],
        cmap="terrain", linewidth=0, antialiased=True,
    )
    ax.set_title("3D surface")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")


def _plot_slope(ax, out, extent, plt) -> None:
    im = ax.imshow(out.slope_map, extent=extent, origin="lower",
                   cmap="inferno")
    plt.colorbar(im, ax=ax, label="slope (deg)", fraction=0.046, pad=0.04)
    ax.set_title("Slope map")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def _plot_navmesh(ax, out, extent, plt) -> None:
    ax.imshow(out.nav_mesh, extent=extent, origin="lower",
              cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("Navigation mesh (green=drivable)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


if __name__ == "__main__":
    sys.exit(main())
