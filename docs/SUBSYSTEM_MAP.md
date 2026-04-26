# Subsystem Integration Map

## How Systems Communicate

This document maps every data flow between subsystems. Each arrow represents a runtime dependency
where one system calls another's interface or consumes its output. Use this to verify that all
interfaces are correctly wired before implementing any subsystem.

---

## Layer 0: Physics Foundation

### System 1 → (everything)
`PhysicsEngine` is the root dependency. Every system that touches Genesis calls through it.

```
PhysicsEngine.build_scene()  →  called by SceneComposer
PhysicsEngine.step(dt)       →  called by main simulation loop
PhysicsEngine.save_snapshot() →  called by ReplaySystem
PhysicsEngine.restore_snapshot() → called by ReplaySystem, ScenarioRunner
```

### System 3 → System 1, System 2
`SceneComposer` reads YAML configs and constructs the Genesis scene.

```
SceneComposer.compose_scene(engine: PhysicsEngine)
  ├── calls TerrainGenerator.generate() → loads terrain into engine
  ├── calls URDFBuilder.build_rover()   → spawns rover entity
  ├── calls URDFBuilder.build_antenna() → spawns antenna entities (STORED state)
  ├── calls URDFBuilder.build_moonbase()→ spawns moonbase structure
  ├── calls CableSystem.initialize()    → pre-allocates all cable links
  └── calls MaterialLibrary.get_material() → assigns contact properties
```

---

## Layer 1: Environment

### System 2 (Environment) ← System 1 (Physics)
All environment subsystems register their entities during scene construction and update each step.

```
TerrainGenerator.export_genesis_heightfield() → gs.morphs.Terrain
RegolithSimulation.step(dt)                   → MPM solver advances
DustSimulation.step(dt, wheel_velocities)     → SPH solver advances
SolarSystem.update(sim_time)                  → directional light updated
ThermalModel.step(dt, sun_elevation)          → component temps updated (1 Hz)
```

### Key data flows FROM environment TO other systems:

| Producer | Consumer | Data | Rate |
|----------|----------|------|------|
| TerrainGenerator | GlobalPathPlanner | nav_mesh (traversability grid) | once at start |
| TerrainGenerator | ElevationMap | height_field, slope_map | once at start |
| RegolithSimulation | WheelTerrainModel | sinkage_at(position) | 240 Hz |
| RegolithSimulation | CableSystem | cable drag resistance | 240 Hz |
| SolarSystem | PowerSystem | sun_elevation → solar output | 1 Hz |
| SolarSystem | StereoCamera | illuminance → exposure adjust | 30 Hz |
| SolarSystem | SunSensor | sun_azimuth, sun_elevation | 1 Hz |
| ThermalModel | PowerSystem | battery_temp → capacity factor | 1 Hz |
| ThermalModel | DriveSystem | motor_temp → torque derating | 1 Hz |
| DustSimulation | PowerSystem | panel_efficiency_factor | 1 Hz |
| DustSimulation | StereoCamera | camera_noise_factor | 30 Hz |
| DustSimulation | LiDARScanner | sph_density → intensity loss | 20 Hz |

---

## Layer 2: Rover Hardware

### System 4 (Drive) ← System 11 (Navigation)
```
DriveSystem.command(DriveCommand)  ← MotionController (MPC) outputs (v, ω)
DriveSystem.get_wheel_states()     → WheelTerrainModel, PowerSystem
DriveSystem.get_odometry()         → LocalizationEKF (encoder update)
```

### System 5 (Power) ← Systems 2, 4, 6, 7
```
PowerSystem.step(dt, sun_elevation, subsystem_states)
  subsystem_states = {
      "drive_motors": active/idle from DriveSystem,
      "manipulator": active/idle from ManipulatorArm,
      "lidar": always active,
      "cameras": always active,
      "imu": always active,
      "compute": active during autonomy,
      "comms": active during relay,
      "heating": active when battery cold (from ThermalModel)
  }
PowerSystem.is_battery_low() → MissionManager triggers RETURN_TO_BASE
```

### System 6 (Manipulator) ← Systems 11, 19, 22 (RL)
```
ManipulatorArm.command_joints(velocities) ← ManipulationSequencer (scripted)
                                          ← AntennaPlacementPolicy (RL, Phase 7)
                                          ← PickupPolicy (RL, Phase 7)
                                          ← CarryStablePolicy (RL, Phase 7)
ManipulatorArm.get_state()                → PolicyInterface.observe()
ManipulatorArm.get_grasp_quality()        → MissionManager (grasp check)
ForceTorqueSensor.read()                  → ManipulatorArm.get_state().ft_reading
```

---

## Layer 3: Sensors

### System 7 → System 11 (Navigation)
Every sensor feeds into the navigation stack. Data flow rates are critical for EKF stability.

```
LiDARScanner.scan()         → OccupancyMap.update()        @ 20 Hz
                            → LocalizationEKF.update_scan() @ 5 Hz (ICP)
StereoCamera.capture()      → ElevationMap.update()         @ 10 Hz
                            → rock_detection → PathPlanner  @ 10 Hz
IMUSensor.read()            → LocalizationEKF.predict()     @ 200 Hz
WheelEncoder.read()         → LocalizationEKF.update_encoder() @ 50 Hz
BeaconNetwork.compute_fix() → LocalizationEKF.update_gps()  @ 1 Hz
SunSensor.read()            → LocalizationEKF.update_sun()  @ 1 Hz
ForceTorqueSensor.read()    → ManipulatorArm, PolicyInterface @ 1000 Hz
```

### GPS Beacon Network dynamic expansion:
```
AntennaUnit (ACTIVE state) → BeaconNetwork.add_beacon()
  → improves GDOP across mission area
  → better LocalizationEKF fixes for all rovers
```

---

## Layer 4: Cable and Antenna

### System 8 (Cable) ← Systems 1, 2, 4, 12, 20
```
CableSystem.step(dt)              ← main loop (physics)
CableSystem.activate_next_link()  ← triggered by rover motion
CableSystem.command_spool()       ← CableDeploymentPolicy (RL) or scripted
CableSystem.get_total_drag_force()→ MotionController (cable tension feedback)
CableSystem.get_spool_state()     → CableDeploymentPolicy.observe()
                                  → CableHealthMonitor.infer()
                                  → MissionManager (remaining cable check)
```

### System 9 (Antenna) ← Systems 6, 7, 12
```
AntennaUnit.transition()          ← MissionManager drives state machine
AntennaUnit.evaluate_deployment() ← called after 3s settle (by MissionManager)
AntennaUnit.get_beacon_config()   → BeaconNetwork (when ACTIVE)
```

State machine transitions triggered by:
- STORED → GRIPPED: F/T sensor confirms grasp > 15 N
- GRIPPED → CARRIED: antenna clear of depot 0.3 m
- CARRIED → PLACED: arm releases, antenna contacts terrain
- PLACED → DEPLOYED: tilt < 15°, base contact verified
- DEPLOYED → ACTIVE: cable connector locked, power confirmed
- any → FAILED: tilt > 15° or position error > 0.5 m

---

## Layer 5: Moonbase

### System 10 ← Systems 4, 8, 9, 12
```
Moonbase.request_cable_reel()  ← MissionManager (depot pickup phase)
Moonbase.request_antenna()     ← MissionManager (depot pickup phase)
Moonbase.dock_rover()          ← MissionManager (charging phase)
Moonbase.get_primary_beacon()  → BeaconNetwork (always active)
Moonbase.get_charge_state()    → MissionManager (wait for charge)
```

---

## Layer 6: Navigation and Autonomy

### System 11 is the most interconnected. All sensor data flows in, all actuator commands flow out.

```
                    ┌──────────────────────────────────────────┐
                    │         NAVIGATION STACK (Sys 11)         │
                    │                                          │
  LiDAR ──────────→│  Perception    ──→  Traversability Map   │
  Stereo Camera ──→│  (OccupancyMap,    (slope + rocks +      │
  Cable positions →│   ElevationMap)     cable exclusion)      │
                    │       │                   │              │
  IMU ────────────→│  Localisation  ←───────────┘              │
  Encoders ───────→│  (EKF)                                    │
  GPS Beacon ─────→│       │                                   │
  Sun Sensor ─────→│       ▼                                   │
                    │  Global Path Planner (A* / D* Lite)      │
                    │       │                                   │
                    │       ▼                                   │
                    │  MPC Controller ←── cable_tension          │
                    │       │          ←── slope                │
                    │       │          ←── speed_limit_factor   │
                    │       ▼             (from RL policy)      │
                    │  DriveCommand (v, ω) → DriveSystem        │
                    │                                          │
                    │  ManipulationSequencer → ManipulatorArm   │
                    └──────────────────────────────────────────┘
```

---

## Layer 7: Mission Management

### System 12 ← (everything above)
The mission manager is the top-level orchestrator.

```
MissionManager.detect_fault() checks:
  ├── antenna tilt     ← AntennaUnit.evaluate_deployment()
  ├── cable snag       ← CableSystem tension > 350 N for > 2s
  ├── cable exhausted  ← CableSystem remaining < 5m, distance > 10m
  ├── rover stuck      ← velocity < 0.05 m/s for > 10s at full torque
  ├── battery low      ← PowerSystem.is_battery_low()
  ├── GPS lost         ← BeaconNetwork GDOP > 10
  └── motor overheat   ← ThermalModel component temp > 80°C

MissionManager.advance_phase() sequences:
  PLANNING → DEPOT_PICKUP → TRANSIT → ANTENNA_DEPLOY → CABLE_CONNECT → RETURN_TO_BASE
```

### Multi-Rover Coordination:
```
MultiRoverCoordinator.get_shared_world_model() ← all rovers contribute LiDAR data
MultiRoverCoordinator.check_cable_crossing()   ← CableMap from each rover
MultiRoverCoordinator.get_communication_delay() = 50ms via moonbase relay
```

---

## Layer 8: Data and Infrastructure

### System 13 (Logging) ← all runtime systems
```
DataLogger.log_rover_state()    ← DriveSystem, PowerSystem        @ 50 Hz
DataLogger.log_sensor_reading() ← all sensors                     @ varies
DataLogger.log_camera_frame()   ← StereoCamera                    @ 30 Hz
DataLogger.log_event()          ← MissionManager (state changes)  @ event
```

### System 14 (Scenarios) ← Systems 1, 3, 12, 13
```
ScenarioRunner.run_single()       → SceneComposer → PhysicsEngine → MissionManager
ScenarioRunner.run_monte_carlo()  → N parallel workers, each a full run_single()
ScenarioRunner.generate_report()  ← AnalysisToolkit.compute_run_metrics()
```

---

## RL Integration (Phase 7+)

### Policy Interface Seam Points

Each RL policy replaces a specific scripted controller at a defined interface boundary.
The mission controller calls the PolicyInterface — it never knows if scripted or RL is running.

```
┌─────────────────────────────────────────────────────┐
│  PolicyInterface (ABC)                               │
│    observe(obs_dict) → act() → action_dict           │
│    mode: SCRIPTED | RL | FALLBACK                    │
├─────────────────────────────────────────────────────┤
│  Sys 19: AntennaPlacementPolicy                      │
│    Seam: after rover positioned, arm in pre-place    │
│    In:  joint_pos, F/T, surface_normal (24 dims)     │
│    Out: joint_velocity_targets (4 dims) @ 20 Hz      │
│    Feeds: ManipulatorArm.command_joints()            │
├─────────────────────────────────────────────────────┤
│  Sys 20: CableDeploymentPolicy                       │
│    Seam: co-controller during all cable-drag transit  │
│    In:  tension, tension_history, spool (112 dims)   │
│    Out: spool_feed_modifier + tension_advisory @ 10Hz│
│    Feeds: CableSystem.command_spool(), MPC.speed_lim │
├─────────────────────────────────────────────────────┤
│  Sys 21: CableHealthMonitor (LSTM, not RL)           │
│    Seam: passive monitor, runs continuously           │
│    In:  30s rolling window (1500 x 6 features)       │
│    Out: anomaly_score, fault_class, time_to_event    │
│    Feeds: MissionManager.detect_fault() advisory     │
├─────────────────────────────────────────────────────┤
│  Sys 22: PickupPolicy + CarryStablePolicy            │
│    Seam: pickup at depot/terrain, stable during drive │
│    In:  joint, F/T, wrist_cam, IMU (1039+ / 22 dims)│
│    Out: joint_vel + gripper (5 dims) / (4 dims)      │
│    Feeds: ManipulatorArm.command_joints/gripper()    │
└─────────────────────────────────────────────────────┘
```

---

## What Still Needs to Be Done

### Per-System Implementation Checklist

| System | Stub File(s) | Status | Critical Dependencies | Implementation Notes |
|--------|-------------|--------|----------------------|---------------------|
| 1. Genesis Config | `core/physics/engine.py` | STUB | Genesis package | Requires real Genesis API — test with mock first |
| 2. Lunar Env | `environment/terrain/generator.py`, `regolith/mpm_model.py`, `lighting/solar.py`, `thermal/model.py`, `dust/sph_model.py` | STUB | System 1 | Terrain generator can be built independently; MPM/SPH need Genesis GPU |
| 3. Asset Pipeline | `core/assets/urdf_builder.py`, `material_library.py`, `core/scene/composer.py` | STUB | YAML configs | URDF builder and material library can be built immediately |
| 4. Drive Systems | `rover/drive/interface.py`, `wheel_terrain.py` | STUB | Systems 1, 2 | Implement 4-wheel first (simplest), then 2-wheel, then 3-wheel |
| 5. Power/Thermal | `rover/power/systems.py` | STUB | System 2 (thermal) | Can prototype with dummy thermal input |
| 6. Manipulator | `rover/manipulator/arm.py` | STUB | System 1 | IK solver is self-contained; URDF generation from System 3 |
| 7. Sensors | `sensors/*/` (6 files) | STUB | System 1 | IMU and encoders are simplest; LiDAR needs Genesis raycast |
| 8. Cable | `cable/chain.py` | STUB | Systems 1, 2 | ALL links must be pre-allocated at build time |
| 9. Antenna | `antenna/system.py` | STUB | Systems 1, 6, 7 | State machine is self-contained; can unit test independently |
| 10. Moonbase | `moonbase/base.py` | STUB | Systems 1, 3 | Inventory system is self-contained |
| 11. Navigation | `navigation/*/` (5 files) | STUB | Systems 7, 8 | EKF can be tested with synthetic sensor data |
| 12. Mission Mgmt | `mission/manager.py`, `coordinator.py` | STUB | System 11 | State machine can be unit tested with mocks |
| 13. Data/Logging | `data/logging/streams.py`, `data/replay/checkpoint.py`, `data/analysis/metrics.py` | STUB | All | Logger can wrap any running system |
| 14. Scenarios | `scenarios/runner.py` | STUB | Systems 1-13 | Monte Carlo orchestration is framework-level |
| 15. Visualization | `visualization/dashboard/app.py` | STUB | Systems 12, 13 | Dashboard is a consumer; can build on mock data |
| 16. Testing | `testing/validators.py` | STUB | All | V1 is the first integration test target |
| 18-23. RL | `rl/*/` (6 files) | STUB | Systems 1-17 complete | Phase 7+ only; do not start until Phase 6 passes |

### Missing Pieces (Not Yet Stubbed)

1. **ROS2 Interface Layer** (System 16.3) — `src/moon_rover/ros2/` — topic publishers/subscribers for HIL. Deferred until Phase 6.
2. **JSON Schema files** for YAML config validation — `configs/schemas/`. Recommended but not blocking.
3. **Video Export Implementation** — `visualization/video_export/recorder.py`. Low priority.
4. **Genesis-specific adapter** — The actual `import genesis as gs` adapter that maps our ABCs to Genesis API calls. This is the first thing to implement.
5. **Scripted controllers** — Concrete implementations of each PolicyInterface for Phases 1-6 (before RL). These go alongside the RL stubs.

### Recommended Implementation Order (matches Phase plan)

**Phase 1** (Foundation): Systems 1, 2 (terrain only), 3, 4 (4-wheel only)
**Phase 2** (Sensors): System 7 (all sensors), System 11.2 (EKF)
**Phase 3** (Cable/Antenna): Systems 8, 9, 6, 11.5
**Phase 4** (Autonomy): Systems 11.1, 11.3, 11.4, 12
**Phase 5** (Multi-Rover): Systems 4 (2/3-wheel), 5, 10, 12.2
**Phase 6** (Infrastructure): Systems 13, 14, 15, 16
**Phase 7** (RL): Systems 18-23
