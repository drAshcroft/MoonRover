# Claude Agent Notes
This is not a minimal viable product.  the goal here is a comlete, production ready application

## Python Environment

Always use the project virtual environment before running Python or pip commands:

```powershell
C:\ve\.genesis\Scripts\activate.ps1
```

When installing dependencies or running the package, activate this environment first:

```bash
source C:/ve/.genesis/Scripts/activate
# or in PowerShell:
# C:\ve\.genesis\Scripts\activate.ps1
```

All `pip install`, `python`, and `moon-rover` CLI commands should run inside this venv.



## tasks
The backlog for this project is managed in the waterfree todo mcp.  use this heavily for implimentation and idealation.  we welcome suggestions and improvements that make this product the best it can be 

## Physics Engine Production Notes

### Supported runtime

- Supported Genesis version: `0.4.4`.
- The production physics adapter is `src/moon_rover/core/physics/_genesis_engine.py`.
- `GenesisPhysicsEngine` currently supports `n_envs=1` only. Any `env_idx` other than `0` is rejected explicitly.
- CPU is the reference backend for replay, smoke tests, and determinism checks.
- GPU is supported for performance-oriented runs, but GPU trajectories should be treated as approximately equivalent to CPU, not bitwise identical.

### Process-global lifecycle

- Genesis runtime init is process-global. Backend, precision, seed, and logging level must stay compatible across repeated `configure()` calls in the same process.
- On Windows, safe teardown intentionally skips process-global `gs.destroy()` by default because multi-entity scenes can hang there.
- Teardown policy is controlled by `MOON_ROVER_GENESIS_DESTROY_POLICY`:
  `safe` = destroy on non-Windows only, `always` = always attempt destroy, `never` = never attempt destroy.
- Strict adapter diagnostics are controlled by `MOON_ROVER_GENESIS_STRICT_DIAGNOSTICS=1`. In strict mode, compatibility warnings become immediate `RuntimeError`s.

### Physics contract

- Fixed-step stepping is required for deterministic replay. Call `step()` with the exact configured `GenesisConfig.timestep`.
- `save_snapshot()` / `restore_snapshot()` are exact-time operations for rigid-body state on the CPU reference path.
- Snapshot restore covers pose, quaternion when supported, linear/angular velocity, DOF state, `sim_time`, and `step_count`.
- Genesis 0.4.4 cannot serialize MPM particle state. `can_snapshot_mpm` remains `False`.
- Genesis 0.4.4 does not expose raycaster normals directly; the adapter returns zero-filled normals when the backend does not provide them.
- Contact queries support both real Genesis 0.4.4 entity contact dictionaries and the older mock scene-level object-list path used by unit tests.

### Terrain and contact semantics

- Terrain normals are derived from world-space height gradients using the cached heightfield and terrain size.
- The default terrain material is tuned as a regolith-like rigid contact surface using `rho=1800`, `friction=1.2`, and `coup_restitution=0.02`.
- Flat terrain should return a `+Z` normal. Sloped heightfields should return normals consistent with the world-space slope, not grid-index slope.

### Verification commands

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

### Benchmark gate profiles

- These are cold-start floor gates for the local Windows workstation, not idealized targets. First-run Genesis kernel compilation is included in `build_seconds`.
- `headless:cpu:many-boxes`: min `steps_per_second=25`, min `real_time_factor=0.20`, max `build_seconds=300`, max `teardown_seconds=15`.
- `headless:gpu:many-boxes`: min `steps_per_second=25`, min `real_time_factor=0.20`, max `build_seconds=600`, max `teardown_seconds=15`.
- `viewer:cpu:many-boxes`: min `render_rate_hz=15`, max `frame_interval_jitter_s=0.20`, max `build_seconds=300`, max `teardown_seconds=15`.
- `viewer:gpu:many-boxes`: min `render_rate_hz=15`, max `frame_interval_jitter_s=0.20`, max `build_seconds=600`, max `teardown_seconds=15`.
- `headless:cpu:rover-proxy`: min `steps_per_second=20`, min `real_time_factor=0.15`, max `build_seconds=300`, max `teardown_seconds=15`.
- `headless:gpu:rover-proxy`: min `steps_per_second=20`, min `real_time_factor=0.15`, max `build_seconds=600`, max `teardown_seconds=15`.
- `viewer:cpu:rover-proxy`: min `render_rate_hz=15`, max `frame_interval_jitter_s=0.20`, max `build_seconds=300`, max `teardown_seconds=15`.
- `viewer:gpu:rover-proxy`: min `render_rate_hz=15`, max `frame_interval_jitter_s=0.20`, max `build_seconds=600`, max `teardown_seconds=15`.

### Regolith interaction tiers (two-tier model)

There are two wheel/regolith interaction tiers. **Tier 1 is the canonical default**; Tier 2 is opt-in.

- **Tier 1 — rigid + analytic terramechanics (DEFAULT, GPU-free).** Module: `moon_rover.rover.drive.terramechanics`. One-call entry point: `default_analytic_terramechanics(...)` → `AnalyticTerramechanics`. It composes the engine's regolith-tuned rigid heightfield terrain (rho=1800, friction=1.2, coup_restitution=0.02) as the contact surface, `LunarRegolithWheelTerrain` for kinematic slip + Pacejka traction + Bekker-Wong sinkage + cable-drag degradation, and `GenesisMPMRegolith(engine=None)`'s analytic core for deterministic rut accumulation (repeated-pass compaction) and cable soil drag — wired in as the wheel model's `rut_sampler`. No CUDA, no MPM solver entity, runs at normal substeps (e.g. `substeps=2`), CPU bit-deterministic on replay. Covered by `tests/integration/test_default_terramechanics.py`. This is what normal sim / RL / replay should use.
- **Tier 2 — Genesis MPM particle soil bed (opt-in).** True granular MPM substrate under the wheels. Enable via `RegolithConfig.mpm_enabled=True` (see below). Requires CUDA + `substeps>=14`; ~10x slower than Tier 1.
- **Accuracy vs runtime.** Tier 1 captures slip-traction coupling, load-dependent sinkage, and multi-pass rut compaction analytically (no true granular flow, lateral berm formation, or particle-scale soil failure). It is the right model for trafficability, path-following, energy/odometry, and RL where ~real-time and determinism matter. Tier 2 adds genuine granular deformation at ~10x cost and a CUDA dependency. **Escalate to Tier 2 only** for high-fidelity soil-deformation studies (deep sinkage/entrapment, excavation, berm/scour geometry) where the analytic rut field is insufficient — not for routine driving, navigation, or RL rollouts.

### Regolith MPM soil bed (opt-in, default OFF)

- The Genesis MPM regolith bed is **disabled by default**. The switch is `RegolithConfig.mpm_enabled` (surfaced as `regolith.mpm_enabled` in `configs/physics.yaml`), default `False`.
- With the default config, `GenesisMPMRegolith` runs analytic-only: no MPM solver entity is created even if an engine is passed. The deterministic Bekker-Wong rut field still feeds `LunarRegolithWheelTerrain` via its `rut_sampler` (`get_sinkage_at`) — that analytic field is the authoritative, replay-deterministic data product regardless of this flag.
- When `mpm_enabled=True`, the MPM bed is built and the config MUST satisfy the CFL substep gate: `substep_dt = timestep / substeps` must be `< 2e-2 * dx` (`dx=1/64`). At the default timestep `1/240` this requires `substeps >= 14` (use ~20 for transient margin). If it does not, the build fails fast with an actionable `RuntimeError` — never a silent downgrade to a smaller substep or to analytic-only.
- `mpm_enabled=True` with no engine supplied is a misconfiguration and raises `RuntimeError`, not an implicit analytic fallback.
- Perf/accuracy tradeoff: the MPM bed requires a CUDA backend plus the raised substep count, making MPM scenes ~10x slower than rigid-only scenes. Enable only for high-fidelity soil-deformation studies.
- No regolith scene composer exists yet; when one is added it should read `regolith.mpm_enabled` to decide whether to pass `engine=` to `GenesisMPMRegolith`.

### Known limitations

- The real Genesis smoke suite is intentionally CPU/no-viewer so it stays stable on the local Windows setup.
- Full project pytest coverage is broader than the physics lane; scene/config failures outside `tests/physics*` do not automatically indicate a regression in the engine adapter.
- If a physics failure is only reproducible with real Genesis, prefer adding coverage under `tests/physics_real` instead of weakening the mock-only tests.
