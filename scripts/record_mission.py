"""Record an MP4 demo of the Genesis mission scenario.

Runs :class:`~moon_rover.scenarios.missions.GenesisMissionScenario` headless and
captures a video from an offscreen camera. By default the camera tracks the
rover; pass ``--fixed`` for a static wide shot.

Scene size
----------
By default this records a **lightweight demo scene** (ground plane + one rover +
one antenna). The full production mission scene (a 4x5 array of antenna units +
rover URDF + terrain heightfield) currently exceeds Genesis's per-build field
(snode) cap and fails to build on a real backend, so it is opt-in via
``--full-scene`` and only works with a reduced ``configs/mission.yaml`` grid.

Offscreen rendering wants a real rasterizer — use ``--backend gpu`` for smooth,
reliable output. The CPU path may work but is slow and can be flaky on the local
Windows setup (the same reason the smoke suite runs no-viewer).

Usage
-----
    python scripts/record_mission.py
    python scripts/record_mission.py --backend gpu --out demos/mission.mp4
    python scripts/record_mission.py --fixed --settle-steps 240 --fps 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "src")

from moon_rover.scenarios.missions import GenesisMissionScenario, default_antenna_config
from moon_rover.visualization.video_export import RecorderConfig

# Where the demo composer parks the rover, and where the antenna is released so
# it lands right next to the rover and stays in frame for the tracking camera.
_DEMO_RELEASE_TARGET = (2.0, 0.0, 0.0)


class _DemoComposer:
    """Minimal Genesis scene for demos: plane + one rover + one antenna.

    Sidesteps the snode-cap blow-up of the full 20-antenna production scene
    while still exercising the real pick / carry / release / settle loop.
    """

    def __init__(self) -> None:
        self.antenna_config = default_antenna_config()

    def compose_scene(self, engine):
        import genesis as gs

        engine.add_entity(
            "terrain", gs.morphs.Plane(), gs.materials.Rigid(), entity_type="terrain"
        )
        engine.add_entity(
            "rover_001",
            gs.morphs.Box(size=(0.8, 0.5, 0.3), pos=(0.0, 0.0, 0.4)),
            gs.materials.Rigid(rho=600),
        )
        base = self.antenna_config.base_plate_m
        engine.add_entity(
            "rover_001_antenna",
            gs.morphs.Box(size=base, pos=(0.0, 0.0, 0.75)),
            gs.materials.Rigid(rho=400),
        )
        engine.build_scene()
        return SimpleNamespace(
            rovers=[SimpleNamespace(rover_id="rover_001")],
            antennas=[
                SimpleNamespace(rover_id="rover_001", antenna_config=self.antenna_config)
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a Genesis mission demo video")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demos/mission.mp4"),
        help="Output MP4 path (default: demos/mission.mp4)",
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Physics + render backend (default: gpu — recommended for rendering)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Mission RNG seed")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--width", type=int, default=1280, help="Frame width in pixels")
    parser.add_argument("--height", type=int, default=720, help="Frame height in pixels")
    parser.add_argument(
        "--carry-steps",
        type=int,
        default=60,
        help="Steps to carry the antenna before releasing (default: 60)",
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=180,
        help="Steps to settle after release (default: 180)",
    )
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="Use a fixed wide shot instead of tracking the rover",
    )
    parser.add_argument(
        "--full-scene",
        action="store_true",
        help="Use the production scene composer (needs a reduced mission grid; "
        "the default 4x5 grid exceeds the Genesis snode cap)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Frame the shot on the scene the run will actually build.
    lookat = (50.0, 50.0, 0.5) if args.full_scene else (0.0, 0.0, 0.4)
    cam_pos = (58.0, 42.0, 6.0) if args.full_scene else (4.0, -4.0, 3.0)

    if args.fixed:
        recorder = RecorderConfig(
            output_path=args.out,
            resolution=(args.width, args.height),
            fps=args.fps,
            track_body=None,
            camera_position=cam_pos,
            camera_lookat=lookat,
        )
    else:
        # Follow the rover (resolved by the scenario to the bound entity).
        recorder = RecorderConfig(
            output_path=args.out,
            resolution=(args.width, args.height),
            fps=args.fps,
            track_body="@rover",
        )

    config: dict = {
        "backend": args.backend,
        "record_video": recorder,
        "carry_steps": args.carry_steps,
        "release_settle_steps": args.settle_steps,
    }
    if not args.full_scene:
        config["composer_factory"] = _DemoComposer
        config["target_position"] = list(_DEMO_RELEASE_TARGET)

    scene_label = "full" if args.full_scene else "demo"
    scenario = GenesisMissionScenario(config)

    print(f"Recording {scene_label} scene ({args.backend}) -> {args.out} ...")
    scenario.setup(seed=args.seed)
    try:
        steps = 0
        while not scenario.is_complete():
            scenario.step()
            steps += 1
    finally:
        scenario.teardown()

    path = scenario.video_path
    report = scenario.array_quality_report()
    print(f"  steps={steps}")
    if path is not None and Path(path).exists():
        size_kb = Path(path).stat().st_size / 1024.0
        print(f"  wrote {path} ({size_kb:.0f} KiB)")
    else:
        print("  no video written (see warnings above)")
    if report is not None:
        print(
            f"  array: {report.elements_placed}/{report.elements_designed} placed, "
            f"completion={report.completion_fraction:.2f}, success={scenario.succeeded()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
