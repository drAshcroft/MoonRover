# Agent Instructions

This repository is not a minimal viable product. The goal is a complete,
production-ready moon rover simulation application. Favor robust, maintainable
changes over quick demos.

Keep this file in sync with `CLAUDE.md`; this file exists so Codex and other
agent runners pick up the project rules without needing Claude-specific notes.

## Python Environment

Always use the project virtual environment before running Python or pip
commands:

```powershell
C:\ve\.genesis\Scripts\activate.ps1
```

When installing dependencies or running the package, activate this environment
first:

```bash
source C:/ve/.genesis/Scripts/activate
# or in PowerShell:
# C:\ve\.genesis\Scripts\activate.ps1
```

All `pip install`, `python`, and `moon-rover` CLI commands should run inside
this venv. Prefer direct invocations such as:

```powershell
C:\ve\.genesis\Scripts\python.exe -m pytest tests/physics/test_genesis_engine.py -q
```

## Tasks

The backlog for this project is managed in the WaterFree todo MCP. Use it
heavily for implementation planning and ideation. Suggestions and improvements
that make the product better are welcome.

## Genesis Demo Performance Notes

Genesis is fast when the simulation loop stays on the engine fast path. Avoid
per-step Python/visualization work that forces synchronization or repeated
crossing into Genesis APIs.

- Supported Genesis version: `0.4.4`.
- For viewer demos, do not render every physics step. Call
  `engine.step(dt, render=False)` for intermediate physics steps and render only
  on the desired viewer frames.
- Genesis `Scene.step()` accepts visualizer controls such as
  `update_visualizer` and `refresh_visualizer`; the local adapter exposes this
  as `GenesisPhysicsEngine.step(dt, render=...)`.
- `ViewerOptions.max_FPS` can synchronize or rate-limit the sim when the viewer
  is refreshed too often. Use a render cadence such as 30 Hz instead of the full
  physics cadence.
- Scripted demos should use separate physics, control, and render rates. For
  `scripts/demo_cable_pull.py`, the practical CPU defaults are currently:
  `--sim-hz 60 --control-hz 30 --render-hz 30 --target-rtf 1.0`.
- Headless runs should skip pure visuals. Do not spend headless sim time
  moving cable boxes, decals, beacon-only visuals, or other render-only props.
- Frozen/deployed visuals should be placed once and left alone. Re-teleporting
  visual bodies every tick is expensive.
- Genesis controllers hold targets between steps. For scripted demos, avoid
  reissuing identical wheel/arm commands every physics tick unless behavior
  genuinely requires it.
- For performance checks, add/report real-time factor (RTF) around the sim loop
  separately from Genesis cold-start/build time.
- GPU first runs can spend minutes compiling CUDA kernels. Treat that as build
  latency, not steady-state frame rate.

Useful cable demo checks:

```powershell
C:\ve\.genesis\Scripts\python.exe -m py_compile scripts\demo_cable_pull.py
C:\ve\.genesis\Scripts\python.exe scripts\demo_cable_pull.py --no-viewer --max-sim-seconds 5 --num-reels 1 --backend cpu
C:\ve\.genesis\Scripts\python.exe scripts\demo_cable_pull.py --backend cpu
C:\ve\.genesis\Scripts\python.exe scripts\demo_cable_pull.py --backend gpu
```

## Physics Engine Production Notes

### Supported Runtime

- Supported Genesis version: `0.4.4`.
- The production physics adapter is
  `src/moon_rover/core/physics/_genesis_engine.py`.
- `GenesisPhysicsEngine` currently supports `n_envs=1` only. Any `env_idx`
  other than `0` is rejected explicitly.
- CPU is the reference backend for replay, smoke tests, and determinism checks.
- GPU is supported for performance-oriented runs, but GPU trajectories should be
  treated as approximately equivalent to CPU, not bitwise identical.

### Process-Global Lifecycle

- Genesis runtime init is process-global. Backend, precision, seed, and logging
  level must stay compatible across repeated `configure()` calls in the same
  process.
- On Windows, safe teardown intentionally skips process-global `gs.destroy()` by
  default because multi-entity scenes can hang there.
- Teardown policy is controlled by `MOON_ROVER_GENESIS_DESTROY_POLICY`:
  `safe` = destroy on non-Windows only, `always` = always attempt destroy,
  `never` = never attempt destroy.
- Strict adapter diagnostics are controlled by
  `MOON_ROVER_GENESIS_STRICT_DIAGNOSTICS=1`. In strict mode, compatibility
  warnings become immediate `RuntimeError`s.

### Physics Contract

- Fixed-step stepping is required for deterministic replay. Call `step()` with
  the exact configured `GenesisConfig.timestep`.
- `save_snapshot()` and `restore_snapshot()` are exact-time operations for
  rigid-body state on the CPU reference path.
- Snapshot restore covers pose, quaternion when supported, linear/angular
  velocity, DOF state, `sim_time`, and `step_count`.
- Genesis 0.4.4 cannot serialize MPM particle state. `can_snapshot_mpm` remains
  `False`.
- Genesis 0.4.4 does not expose raycaster normals directly; the adapter returns
  zero-filled normals when the backend does not provide them.
- Contact queries support both real Genesis 0.4.4 entity contact dictionaries
  and the older mock scene-level object-list path used by unit tests.

### Terrain And Contact Semantics

- Terrain normals are derived from world-space height gradients using the cached
  heightfield and terrain size.
- The default terrain material is tuned as a regolith-like rigid contact surface
  using `rho=1800`, `friction=1.2`, and `coup_restitution=0.02`.
- Flat terrain should return a `+Z` normal. Sloped heightfields should return
  normals consistent with the world-space slope, not grid-index slope.

### Verification Commands

- Fast mock/unit physics tests:
  `C:\ve\.genesis\Scripts\python.exe -m pytest tests/physics/test_genesis_engine.py -q`
- Full fast physics test directory:
  `C:\ve\.genesis\Scripts\python.exe -m pytest tests/physics -q`
- Real Genesis smoke suite:
  `$env:MOON_ROVER_RUN_GENESIS_SMOKE="1"`
  `C:\ve\.genesis\Scripts\python.exe -m pytest tests/physics_real -q -m genesis`
- Interactive viewer/demo:
  `C:\ve\.genesis\Scripts\python.exe scripts/verify_physics.py`
- Machine-readable benchmark gates:
  `C:\ve\.genesis\Scripts\python.exe scripts/benchmark_physics.py --scene many-boxes --backend cpu --json`
  `C:\ve\.genesis\Scripts\python.exe scripts/benchmark_physics.py --scene many-boxes --backend gpu --assert-thresholds`
  `C:\ve\.genesis\Scripts\python.exe scripts/benchmark_physics.py --scene rover-proxy --mode viewer --json`

### Benchmark Gate Profiles

These are cold-start floor gates for the local Windows workstation, not
idealized targets. First-run Genesis kernel compilation is included in
`build_seconds`.

- `headless:cpu:many-boxes`: min `steps_per_second=25`, min
  `real_time_factor=0.20`, max `build_seconds=300`, max `teardown_seconds=15`.
- `headless:gpu:many-boxes`: min `steps_per_second=25`, min
  `real_time_factor=0.20`, max `build_seconds=600`, max `teardown_seconds=15`.
- `viewer:cpu:many-boxes`: min `render_rate_hz=15`, max
  `frame_interval_jitter_s=0.20`, max `build_seconds=300`, max
  `teardown_seconds=15`.
- `viewer:gpu:many-boxes`: min `render_rate_hz=15`, max
  `frame_interval_jitter_s=0.20`, max `build_seconds=600`, max
  `teardown_seconds=15`.
- `headless:cpu:rover-proxy`: min `steps_per_second=20`, min
  `real_time_factor=0.15`, max `build_seconds=300`, max `teardown_seconds=15`.
- `headless:gpu:rover-proxy`: min `steps_per_second=20`, min
  `real_time_factor=0.15`, max `build_seconds=600`, max `teardown_seconds=15`.
- `viewer:cpu:rover-proxy`: min `render_rate_hz=15`, max
  `frame_interval_jitter_s=0.20`, max `build_seconds=300`, max
  `teardown_seconds=15`.
- `viewer:gpu:rover-proxy`: min `render_rate_hz=15`, max
  `frame_interval_jitter_s=0.20`, max `build_seconds=600`, max
  `teardown_seconds=15`.

## Regolith Interaction Tiers

There are two wheel/regolith interaction tiers. Tier 1 is the canonical default;
Tier 2 is opt-in.

### Tier 1: Rigid Plus Analytic Terramechanics

This is the default and does not require a GPU.

- Module: `moon_rover.rover.drive.terramechanics`.
- One-call entry point:
  `default_analytic_terramechanics(...)` -> `AnalyticTerramechanics`.
- Composes the engine's regolith-tuned rigid heightfield terrain
  (`rho=1800`, `friction=1.2`, `coup_restitution=0.02`) as the contact surface.
- Uses `LunarRegolithWheelTerrain` for kinematic slip, Pacejka traction,
  Bekker-Wong sinkage, and cable-drag degradation.
- Uses `GenesisMPMRegolith(engine=None)` analytic core for deterministic rut
  accumulation and cable soil drag, wired in as the wheel model's `rut_sampler`.
- No CUDA, no MPM solver entity, normal substeps such as `substeps=2`, and CPU
  bit-deterministic replay.
- Covered by `tests/integration/test_default_terramechanics.py`.
- This is what normal sim, RL, and replay should use.

### Tier 2: Genesis MPM Particle Soil Bed

This is opt-in true granular MPM substrate under the wheels.

- Enable via `RegolithConfig.mpm_enabled=True` or `regolith.mpm_enabled` in
  `configs/physics.yaml`.
- Requires CUDA plus `substeps>=14`.
- Runs about 10x slower than Tier 1.
- Use only for high-fidelity soil-deformation studies such as deep
  sinkage/entrapment, excavation, berms, scour geometry, or particle-scale soil
  failure. Do not use Tier 2 for routine driving, navigation, RL, replay, or
  trafficability runs.

### Regolith MPM Soil Bed Configuration

- The Genesis MPM regolith bed is disabled by default.
- With the default config, `GenesisMPMRegolith` runs analytic-only: no MPM solver
  entity is created even if an engine is passed.
- The deterministic Bekker-Wong rut field still feeds
  `LunarRegolithWheelTerrain` via `get_sinkage_at`; that analytic field is the
  authoritative replay-deterministic data product regardless of `mpm_enabled`.
- When `mpm_enabled=True`, the MPM bed is built and the config must satisfy the
  CFL substep gate: `substep_dt = timestep / substeps` must be
  `< 2e-2 * dx` with `dx=1/64`.
- At the default timestep `1/240`, MPM requires `substeps >= 14`; use about 20
  for transient margin.
- If the CFL gate does not pass, the build must fail fast with an actionable
  `RuntimeError`; never silently downgrade to a smaller substep or analytic-only.
- `mpm_enabled=True` with no engine supplied is a misconfiguration and raises
  `RuntimeError`, not an implicit analytic fallback.
- No regolith scene composer exists yet; when one is added it should read
  `regolith.mpm_enabled` to decide whether to pass `engine=` to
  `GenesisMPMRegolith`.

## Known Limitations

- The real Genesis smoke suite is intentionally CPU/no-viewer so it stays stable
  on the local Windows setup.
- Full project pytest coverage is broader than the physics lane; scene/config
  failures outside `tests/physics*` do not automatically indicate a regression
  in the engine adapter.
- If a physics failure is only reproducible with real Genesis, prefer adding
  coverage under `tests/physics_real` instead of weakening mock-only tests.
