"""Opt-in real Genesis smoke test for offscreen video recording.

Renders a handful of frames through a real Genesis camera and confirms an MP4 is
encoded to disk. Gated behind MOON_ROVER_RUN_GENESIS_SMOKE=1 and the ``genesis``
marker, like the other real-engine smoke tests.

Note: offscreen rasterizing is most reliable on a GPU backend. On a CPU-only box
this may be slow or unavailable; the test asserts a non-empty file is produced
when rendering works in the subprocess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ._helpers import require_real_genesis_smoke, run_smoke_payload


def run_real_genesis_recording_smoke(out_path: Path) -> dict[str, object]:
    return run_smoke_payload(
        rf"""
import json
import sys
from pathlib import Path

import genesis as gs

sys.path.insert(0, "src")

from moon_rover.core.physics.engine import GenesisConfig
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine

out = Path({str(out_path)!r})
out.parent.mkdir(parents=True, exist_ok=True)

physics = GenesisConfig(
    gravity_vector=(0.0, 0.0, -1.622),
    timestep=1.0 / 60.0,
    substeps=1,
    use_gpu=False,
)

engine = GenesisPhysicsEngine()
engine.configure(physics, show_viewer=False)
engine.add_camera(
    "demo", resolution=(320, 240), pos=(3.0, -3.0, 2.0), lookat=(0.0, 0.0, 0.3), fov=45.0
)
engine.add_entity("ground", gs.morphs.Plane(), gs.materials.Rigid(), entity_type="terrain")
engine.add_entity(
    "box", gs.morphs.Box(size=(0.3, 0.3, 0.3), pos=(0.0, 0.0, 1.0)), gs.materials.Rigid()
)
engine.build_scene()

engine.start_camera_recording("demo")
for _ in range(20):
    engine.step(physics.timestep, render=False)
    engine.render_camera("demo")
engine.stop_camera_recording("demo", save_path=str(out), fps=20)
engine.teardown()

print(json.dumps({{
    "exists": out.exists(),
    "size": out.stat().st_size if out.exists() else 0,
}}))
""",
        timeout=600,
    )


@pytest.mark.genesis
@pytest.mark.slow
def test_real_genesis_records_mp4(tmp_path) -> None:
    require_real_genesis_smoke()
    payload = run_real_genesis_recording_smoke(tmp_path / "smoke.mp4")
    assert payload["exists"] is True
    assert int(payload["size"]) > 0
