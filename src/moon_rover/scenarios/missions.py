"""Concrete mission scenarios for the ScenarioRunner harness.

Provides higher-fidelity :class:`~moon_rover.scenarios.runner.Scenario`
implementations that drive real subsystem state machines (e.g. the antenna
deployment lifecycle) end to end. The analytic scenario stays deterministic
and GPU-free for integration tests and RL/dashboard baselines. The Genesis
scenario builds a physical scene and emits the same telemetry contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from moon_rover.antenna.array import ArrayDesign
from moon_rover.antenna.system import AntennaConfig, AntennaState, DeployableAntennaUnit
from moon_rover.core.physics.engine import AttachmentHandle, GenesisConfig
from moon_rover.core.scene.specs import Scene
from moon_rover.data.analysis.metrics import ArrayQualityReport, MissionMetricsAnalyzer
from moon_rover.scenarios.runner import Scenario
from moon_rover.visualization.video_export import RecorderConfig, VideoRecorder


def default_antenna_config() -> AntennaConfig:
    """A representative antenna unit (~12 kg assembled)."""
    return AntennaConfig(
        base_plate_m=(0.4, 0.4, 0.05),
        base_mass_kg=4.0,
        mast_height_m=1.5,
        mast_radius_m=0.03,
        mast_mass_kg=3.0,
        dish_diameter_m=0.6,
        dish_mass_kg=4.0,
        connector_mass_kg=1.0,
        total_mass_kg=12.0,
    )


class MissionPhase(Enum):
    PICKUP = "pickup"
    NAVIGATE = "navigate"
    PLACE = "place"
    RETURN = "return"
    DONE = "done"


@dataclass
class _PlacementProfile:
    """Sensed deployment quality applied at a grid point."""

    tilt_deg: float
    base_contact_corners: int
    position_error_m: float
    connector_engaged: bool


# A clean placement that should reach ACTIVE; a bad one that should fail to.
_GOOD_PLACEMENT = _PlacementProfile(2.0, 4, 0.1, True)
_BAD_PLACEMENT = _PlacementProfile(12.0, 1, 0.8, False)


class MissionPlacementScenario(Scenario):
    """Full antenna-placement mission, driven through the real antenna lifecycle.

    Flow: spawn at moonbase -> for each grid point {pick up antenna, navigate to
    the point, place + deploy + activate the antenna} -> return to base. Each
    antenna is a real :class:`DeployableAntennaUnit`, so the test asserts the
    actual ``STORED -> GRIPPED -> CARRIED -> PLACED -> DEPLOYED -> ACTIVE``
    transitions rather than a mock.

    Config keys (all optional):
        base_position     [x, y, z]            moonbase origin (default [0,0,0])
        grid_points       [[x, y, z], ...]     antenna target sites
        dt                float                control tick (default 0.05 -> 20 Hz)
        drive_speed_mps   float                cruise speed (default 1.0)
        arrive_tol_m      float                arrival threshold (default 0.2)
        pickup_ticks      int                  ticks to grasp/stow (default 5)
        place_ticks       int                  ticks to place/deploy (default 5)
        fail_indices      [int, ...]           grid indices that mis-deploy
        antenna_config    AntennaConfig        override antenna geometry
    """

    rover_type = "mission_placement_diff_drive"

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = dict(config or {})
        self.base_position = np.asarray(cfg.get("base_position", [0.0, 0.0, 0.0]), dtype=np.float64)
        grid = cfg.get("grid_points", [[5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0]])
        self.grid_points = [np.asarray(p, dtype=np.float64) for p in grid]
        self.dt = float(cfg.get("dt", 0.05))
        self.drive_speed_mps = float(cfg.get("drive_speed_mps", 1.0))
        self.arrive_tol_m = float(cfg.get("arrive_tol_m", 0.2))
        self.pickup_ticks = int(cfg.get("pickup_ticks", 5))
        self.place_ticks = int(cfg.get("place_ticks", 5))
        self.fail_indices = set(int(i) for i in cfg.get("fail_indices", []))
        self.antenna_config = cfg.get("antenna_config") or default_antenna_config()

        self._rng: Optional[np.random.Generator] = None
        self._pos = self.base_position.copy()
        self._sim_time = 0.0
        self._energy_wh = 0.0
        self._phase = MissionPhase.DONE
        self._idx = 0
        self._phase_ticks = 0
        self._antennas: list[DeployableAntennaUnit] = []
        self._placements: list[dict] = []
        self._faults: list[dict] = []
        self._events: list[dict] = []
        self._returned = False

    # -- lifecycle ----------------------------------------------------------

    def setup(self, seed: int, *, visualize: bool = False) -> None:
        self._rng = np.random.default_rng(seed)
        self._pos = self.base_position.copy()
        self._sim_time = 0.0
        self._energy_wh = 0.0
        self._idx = 0
        self._phase_ticks = 0
        self._antennas = []
        self._placements = []
        self._faults = []
        self._returned = False
        self._events = [{"event_type": "mission_start", "sim_time": 0.0, "payload": {"seed": seed}}]
        self._phase = MissionPhase.PICKUP if self.grid_points else MissionPhase.RETURN

    def step(self) -> dict:
        assert self._rng is not None, "setup() must be called before step()"
        moving = False

        if self._phase == MissionPhase.PICKUP:
            self._do_pickup()
        elif self._phase == MissionPhase.NAVIGATE:
            moving = self._drive_toward(self.grid_points[self._idx])
            if not moving:
                self._phase = MissionPhase.PLACE
                self._phase_ticks = 0
        elif self._phase == MissionPhase.PLACE:
            self._do_place()
        elif self._phase == MissionPhase.RETURN:
            moving = self._drive_toward(self.base_position)
            if not moving:
                self._returned = True
                self._phase = MissionPhase.DONE
                self._events.append(
                    {"event_type": "returned_to_base", "sim_time": self._sim_time, "payload": {}}
                )

        # Energy/time bookkeeping: idle draw + drive draw.
        power_w = 60.0 + (120.0 if moving else 0.0)
        self._energy_wh += power_w * (self.dt / 3600.0)
        self._sim_time += self.dt

        gt = self._pos.copy()
        est = gt + self._rng.normal(scale=0.04, size=3)
        cable_tension = 20.0 + 8.0 * float(np.linalg.norm(self._pos - self.base_position))
        n_active = sum(1 for a in self._antennas if a.get_state() == AntennaState.ACTIVE)
        coverage = n_active / max(1, len(self.grid_points))

        return {
            "timestamp": self._sim_time,
            "rover_position": self._pos.tolist(),
            "velocity": [0.0, 0.0, 0.0],
            "energy_wh": self._energy_wh,
            "power_consumed_w": power_w,
            "cable_tension_n": cable_tension,
            "cable_coverage_fraction": coverage,
            "estimated_position": est.tolist(),
            "ground_truth_position": gt.tolist(),
            "imu": {
                "accel_xyz": [0.0, 0.0, -1.62],
                "gyro_xyz": [0.0, 0.0, 0.0],
                "timestamp": self._sim_time,
            },
        }

    def is_complete(self) -> bool:
        return self._phase == MissionPhase.DONE

    def teardown(self) -> None:
        self._events.append(
            {"event_type": "mission_end", "sim_time": self._sim_time, "payload": {}}
        )

    # -- phase handlers -----------------------------------------------------

    def _do_pickup(self) -> None:
        if self._phase_ticks == 0:
            antenna = DeployableAntennaUnit(self.antenna_config)
            antenna.transition(AntennaState.GRIPPED)
            antenna.transition(AntennaState.CARRIED)
            self._antennas.append(antenna)
            self._events.append(
                {
                    "event_type": "antenna_picked",
                    "sim_time": self._sim_time,
                    "payload": {"antenna_id": f"antenna_{self._idx:02d}"},
                }
            )
        self._phase_ticks += 1
        if self._phase_ticks >= self.pickup_ticks:
            self._phase = MissionPhase.NAVIGATE
            self._phase_ticks = 0

    def _do_place(self) -> None:
        if self._phase_ticks == 0:
            antenna = self._antennas[self._idx]
            point = self.grid_points[self._idx]
            profile = _BAD_PLACEMENT if self._idx in self.fail_indices else _GOOD_PLACEMENT
            antenna.set_placement(
                position_xy=point[:2],
                tilt_deg=profile.tilt_deg,
                base_contact_corners=profile.base_contact_corners,
                position_error_m=profile.position_error_m,
                connector_engaged=profile.connector_engaged,
            )
            antenna.transition(AntennaState.PLACED)
            antenna.transition(AntennaState.DEPLOYED)
            activated = antenna.transition(AntennaState.ACTIVE)
            success = antenna.get_state() == AntennaState.ACTIVE
            antenna_id = f"antenna_{self._idx:02d}"
            self._placements.append(
                {
                    "antenna_id": antenna_id,
                    "target": point.tolist(),
                    "actual": self._pos.tolist(),
                    "success": success,
                    "sim_time": self._sim_time,
                    "failure_mode": None if success else "deployment_failed",
                }
            )
            if not success:
                self._faults.append(
                    {"mode": "deployment_failed", "time": self._sim_time, "antenna_id": antenna_id}
                )
            self._events.append(
                {
                    "event_type": "antenna_activated" if activated else "antenna_deploy_failed",
                    "sim_time": self._sim_time,
                    "payload": {"antenna_id": antenna_id, "state": antenna.get_state().value},
                }
            )
        self._phase_ticks += 1
        if self._phase_ticks >= self.place_ticks:
            self._idx += 1
            self._phase_ticks = 0
            if self._idx < len(self.grid_points):
                self._phase = MissionPhase.PICKUP
            else:
                self._phase = MissionPhase.RETURN

    def _drive_toward(self, target: np.ndarray) -> bool:
        """Advance toward target. Returns True while still moving."""
        delta = target - self._pos
        dist = float(np.linalg.norm(delta))
        if dist <= self.arrive_tol_m:
            return False
        step = min(self.drive_speed_mps * self.dt, dist)
        self._pos = self._pos + (delta / dist) * step
        return True

    # -- introspection (for assertions) ------------------------------------

    @property
    def antenna_units(self) -> list[DeployableAntennaUnit]:
        return self._antennas

    def antenna_states(self) -> list[AntennaState]:
        return [a.get_state() for a in self._antennas]

    @property
    def returned_to_base(self) -> bool:
        return self._returned

    @property
    def antenna_placements(self) -> list[dict]:
        return self._placements

    @property
    def faults(self) -> list[dict]:
        return self._faults

    @property
    def events(self) -> list[dict]:
        return self._events

    def succeeded(self) -> bool:
        if not self._antennas or len(self._antennas) < len(self.grid_points):
            return False
        all_active = all(a.get_state() == AntennaState.ACTIVE for a in self._antennas)
        return all_active and self._returned


def mission_placement_factory(config: dict) -> Scenario:
    """Scenario factory for :class:`MissionPlacementScenario`."""
    return MissionPlacementScenario(config)


class GenesisMissionPhase(Enum):
    """Physical single-antenna mission phases."""

    CARRIED_SETTLE = "carried_settle"
    RELEASE_SETTLE = "release_settle"
    DONE = "done"


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

# Sentinel for RecorderConfig.track_body meaning "follow this scenario's rover";
# resolved to the bound rover entity name once the scene is composed.
_ROVER_TRACK_SENTINEL = "@rover"


class GenesisMissionScenario(Scenario):
    """Compose and step a real Genesis scene for one physical antenna release.

    This is the first physics-in-the-loop mission slice. It intentionally keeps
    navigation and arm motion scripted until the manipulation loop is wired:
    compose the configured scene, attach the first stored antenna to the first
    rover, release it over the first surveyed array target, settle it through
    rigid contact physics, and commission the resulting placement.

    Inject ``engine_factory`` and ``composer_factory`` for focused tests.
    Production callers normally use the config-file defaults.
    """

    rover_type = "genesis_mission_diff_drive"

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = dict(config or {})
        configs_dir = _PROJECT_ROOT / "configs"
        self.scene_config_path = Path(cfg.get("scene_config_path", configs_dir / "scene.yaml"))
        self.rover_config_path = Path(cfg.get("rover_config_path", configs_dir / "rover.yaml"))
        self.mission_config_path = Path(cfg.get("mission_config_path", configs_dir / "mission.yaml"))
        self.physics_config_path = Path(cfg.get("physics_config_path", configs_dir / "physics.yaml"))
        sensors_path = cfg.get("sensors_config_path", configs_dir / "sensors.yaml")
        self.sensors_config_path = None if sensors_path is None else Path(sensors_path)

        self._engine_factory: Callable[[], Any] = cfg.get("engine_factory") or self._default_engine
        self._composer_factory: Optional[Callable[[], Any]] = cfg.get("composer_factory")
        self._physics_config: Optional[GenesisConfig] = cfg.get("physics_config")
        self._backend = cfg.get("backend")
        self._target_override = cfg.get("target_position")
        self._target_quat_override = cfg.get("target_orientation_quat")
        self.carry_steps = self._positive_int(cfg.get("carry_steps", 2), "carry_steps")
        self.release_settle_steps = self._positive_int(
            cfg.get("release_settle_steps", 30), "release_settle_steps"
        )
        self.render_hz = float(cfg.get("render_hz", 30.0))
        if self.render_hz <= 0.0:
            raise ValueError("render_hz must be > 0")

        # Mission success = array built to spec. A run succeeds only when every
        # commissioned element it placed is within the surveyed ArrayDesign
        # tolerance (position + tilt), every evaluable baseline holds, and the
        # built fraction of the full design reaches this floor. This slice
        # commissions a single element, so the completion floor defaults to 0.0
        # (success := the placed element is to-spec); multi-element array
        # missions should raise it toward 1.0 to demand the whole array.
        self.success_min_completion_fraction = self._unit_fraction(
            cfg.get("success_min_completion_fraction", 0.0),
            "success_min_completion_fraction",
        )

        # Optional MP4 demo recording. Accepts a RecorderConfig, a dict of its
        # fields, or a path-like (records a rover-tracking shot to that file).
        self._recorder = self._build_recorder(cfg.get("record_video"))
        self._video_path: Optional[Path] = None

        self._rng: Optional[np.random.Generator] = None
        self._engine: Any = None
        self._scene: Optional[Scene] = None
        self._physics: Optional[GenesisConfig] = None
        self._visualize = False
        self._phase = GenesisMissionPhase.DONE
        self._phase_ticks = 0
        self._energy_wh = 0.0
        self._attachment: Optional[AttachmentHandle] = None
        self._rover_entity_name = ""
        self._antenna_entity_name = ""
        self._antenna_id = ""
        self._target = np.zeros(3, dtype=np.float64)
        self._target_quat = _IDENTITY_QUAT.copy()
        self._antenna_config: Optional[AntennaConfig] = None
        self._antenna_unit: Optional[DeployableAntennaUnit] = None
        self._array_design: Optional[ArrayDesign] = None
        self._analyzer = MissionMetricsAnalyzer()
        self._placements: list[dict] = []
        self._faults: list[dict] = []
        self._events: list[dict] = []

    def setup(self, seed: int, *, visualize: bool = False) -> None:
        self._rng = np.random.default_rng(seed)
        self._visualize = bool(visualize)
        self._phase = GenesisMissionPhase.DONE
        self._phase_ticks = 0
        self._energy_wh = 0.0
        self._attachment = None
        self._placements = []
        self._faults = []
        self._events = [{"event_type": "mission_start", "sim_time": 0.0, "payload": {"seed": seed}}]

        engine = self._engine_factory()
        self._engine = engine
        try:
            physics = self._resolve_physics_config()
            self._physics = physics
            engine.configure(physics, show_viewer=self._visualize)
            # Camera must be registered during CONSTRUCTION, before the composer
            # builds the scene.
            if self._recorder is not None:
                self._recorder.attach(engine)
            scene = self._make_composer().compose_scene(engine)
            self._bind_first_rover_and_antenna(scene)
            if self._recorder is not None:
                # Resolve the "track the rover" sentinel to the bound entity now
                # that the scene exists; an explicit name or None is left as-is.
                if self._recorder.config.track_body == _ROVER_TRACK_SENTINEL:
                    self._recorder.set_track_body(self._rover_entity_name)
                self._recorder.start()
            self._attachment = engine.attach_bodies(
                self._rover_entity_name, self._antenna_entity_name
            )
            assert self._antenna_unit is not None
            if not self._antenna_unit.transition(AntennaState.GRIPPED):
                raise RuntimeError("physical antenna could not transition to GRIPPED")
            if not self._antenna_unit.transition(AntennaState.CARRIED):
                raise RuntimeError("physical antenna could not transition to CARRIED")
            self._phase = GenesisMissionPhase.CARRIED_SETTLE
            self._events.append(
                {
                    "event_type": "antenna_picked",
                    "sim_time": 0.0,
                    "payload": {
                        "antenna_id": self._antenna_id,
                        "entity_name": self._antenna_entity_name,
                    },
                }
            )
        except Exception:
            self._teardown_engine()
            raise

    def step(self) -> dict:
        if self._engine is None or self._rng is None or self._physics is None:
            raise RuntimeError("setup() must be called before step()")
        if self._phase == GenesisMissionPhase.DONE:
            raise RuntimeError("step() called after Genesis mission completed")

        if (
            self._phase == GenesisMissionPhase.CARRIED_SETTLE
            and self._phase_ticks >= self.carry_steps
        ):
            self._release_antenna()

        self._engine.step(self._physics.timestep, render=self._should_render())
        self._phase_ticks += 1
        if self._recorder is not None and self._should_capture():
            self._recorder.capture()
        if (
            self._phase == GenesisMissionPhase.RELEASE_SETTLE
            and self._phase_ticks >= self.release_settle_steps
        ):
            self._commission_placement()

        return self._telemetry_record()

    def is_complete(self) -> bool:
        return self._phase == GenesisMissionPhase.DONE

    def teardown(self) -> None:
        sim_time = self._sim_time()
        self._events.append({"event_type": "mission_end", "sim_time": sim_time, "payload": {}})
        # Encode the video while the scene is still alive, but never let a
        # recording failure leak the engine — always tear it down, then surface.
        recording_error: Optional[BaseException] = None
        try:
            self._finalize_recording()
        except BaseException as exc:  # noqa: BLE001 - re-raised after teardown
            recording_error = exc
        self._teardown_engine()
        if recording_error is not None:
            raise recording_error

    @property
    def video_path(self) -> Optional[Path]:
        """Path to the recorded MP4 once teardown has written it, else None."""
        return self._video_path

    @property
    def antenna_units(self) -> list[DeployableAntennaUnit]:
        return [] if self._antenna_unit is None else [self._antenna_unit]

    def antenna_states(self) -> list[AntennaState]:
        return [unit.get_state() for unit in self.antenna_units]

    @property
    def antenna_placements(self) -> list[dict]:
        return self._placements

    @property
    def faults(self) -> list[dict]:
        return self._faults

    @property
    def events(self) -> list[dict]:
        return self._events

    def array_quality_report(self, committed_only: bool = True) -> Optional[ArrayQualityReport]:
        """Score the deployed array against the surveyed :class:`ArrayDesign`.

        Args:
            committed_only: When True (default) only elements that reached the
                commissioned (``ACTIVE``) state count as built; geometrically
                placed-but-uncommissioned elements are excluded. When False the
                report reflects every placement record regardless of state.

        Returns:
            The :class:`ArrayQualityReport`, or ``None`` if the design has not
            been bound yet (``setup()`` not called).
        """
        if self._array_design is None:
            return None
        placements = {
            record["antenna_id"]: record["actual"]
            for record in self._placements
            if record.get("antenna_id") is not None
            and record.get("actual") is not None
            and (not committed_only or record.get("success") is True)
        }
        return self._analyzer.array_quality_report(
            self._array_design, placements, run_id=self._antenna_id or None
        )

    def succeeded(self) -> bool:
        """Mission success := the array was built to the design spec.

        True only when at least one element was commissioned, every commissioned
        element sits within the surveyed design tolerance, all evaluable pairwise
        baselines hold, and the built fraction of the full design meets
        ``success_min_completion_fraction``.
        """
        report = self.array_quality_report(committed_only=True)
        if report is None or report.elements_placed == 0:
            return False
        return (
            report.elements_within_tolerance == report.elements_placed
            and report.baselines_within_tolerance == report.baselines_evaluated
            and report.completion_fraction >= self.success_min_completion_fraction
        )

    def _resolve_physics_config(self) -> GenesisConfig:
        physics = self._physics_config or GenesisConfig.from_yaml(str(self.physics_config_path))
        if self._backend is None:
            return physics
        backend = str(self._backend).lower()
        if backend not in {"cpu", "gpu"}:
            raise ValueError("backend must be 'cpu' or 'gpu'")
        return replace(physics, use_gpu=backend == "gpu")

    def _make_composer(self) -> Any:
        if self._composer_factory is not None:
            return self._composer_factory()
        from moon_rover.core.scene.composer import GenesisSceneComposer

        composer = GenesisSceneComposer()
        composer.load_scene_config(str(self.scene_config_path))
        composer.load_rover_config(str(self.rover_config_path))
        composer.load_mission_config(str(self.mission_config_path))
        composer.load_physics_config(str(self.physics_config_path))
        composer.load_sensors_config(
            None if self.sensors_config_path is None else str(self.sensors_config_path)
        )
        return composer

    def _bind_first_rover_and_antenna(self, scene: Scene) -> None:
        if not scene.rovers:
            raise RuntimeError("composed Genesis mission scene has no rover")
        rover = scene.rovers[0]
        try:
            antenna = next(item for item in scene.antennas if item.rover_id == rover.rover_id)
        except StopIteration as exc:
            raise RuntimeError(
                f"composed Genesis mission scene has no antenna for rover {rover.rover_id!r}"
            ) from exc

        self._scene = scene
        self._rover_entity_name = rover.rover_id
        self._antenna_entity_name = f"{rover.rover_id}_antenna"
        self._antenna_config = antenna.antenna_config
        self._antenna_unit = DeployableAntennaUnit(antenna.antenna_config, self._engine)
        design = ArrayDesign.from_yaml(self.mission_config_path)
        self._array_design = design
        target = design.elements[0]
        self._antenna_id = target.element_id
        self._target = np.asarray(
            self._target_override if self._target_override is not None else target.position_xyz,
            dtype=np.float64,
        ).reshape(3)
        self._target_quat = np.asarray(
            self._target_quat_override
            if self._target_quat_override is not None
            else target.orientation_quat_xyzw,
            dtype=np.float32,
        ).reshape(4)

    def _release_antenna(self) -> None:
        assert self._engine is not None
        assert self._antenna_unit is not None
        assert self._antenna_config is not None
        assert self._attachment is not None
        self._engine.detach_bodies(self._attachment)
        self._attachment = None
        terrain_z = self._terrain_height(self._target[0], self._target[1], self._target[2])
        clearance = (self._antenna_config.base_plate_m[2] * 0.5) + 0.005
        release_pos = np.array([self._target[0], self._target[1], terrain_z + clearance])
        self._engine.set_body_pose(self._antenna_entity_name, release_pos, self._target_quat)
        self._engine.set_body_velocity(
            self._antenna_entity_name, np.zeros(3), np.zeros(3)
        )
        self._phase = GenesisMissionPhase.RELEASE_SETTLE
        self._phase_ticks = 0
        self._events.append(
            {
                "event_type": "antenna_released",
                "sim_time": self._sim_time(),
                "payload": {"antenna_id": self._antenna_id, "target": self._target.tolist()},
            }
        )

    def _commission_placement(self) -> None:
        assert self._engine is not None
        assert self._antenna_unit is not None
        assert self._antenna_config is not None
        actual, quat = self._engine.get_body_pose(self._antenna_entity_name)
        actual = np.asarray(actual, dtype=np.float64).reshape(3)
        terrain_z = self._terrain_height(actual[0], actual[1], self._target[2])
        horizontal_error = float(np.linalg.norm(actual[:2] - self._target[:2]))
        tilt_deg = self._tilt_deg(np.asarray(quat, dtype=np.float64))
        base_height = self._antenna_config.base_plate_m[2]
        grounded = float(actual[2]) <= terrain_z + base_height + 0.05
        connector_engaged = grounded and horizontal_error <= 0.5
        self._antenna_unit.set_placement(
            position_xy=actual[:2],
            tilt_deg=tilt_deg,
            base_contact_corners=4 if grounded else 0,
            position_error_m=horizontal_error,
            connector_engaged=connector_engaged,
        )
        placed = self._antenna_unit.transition(AntennaState.PLACED)
        deployed = placed and self._antenna_unit.transition(AntennaState.DEPLOYED)
        activated = deployed and self._antenna_unit.transition(AntennaState.ACTIVE)
        success = self._antenna_unit.get_state() == AntennaState.ACTIVE
        self._placements.append(
            {
                "antenna_id": self._antenna_id,
                "target": self._target.tolist(),
                "actual": actual.tolist(),
                "success": success,
                "sim_time": self._sim_time(),
                "failure_mode": None if success else "deployment_failed",
            }
        )
        if not success:
            self._faults.append(
                {"mode": "deployment_failed", "time": self._sim_time(), "antenna_id": self._antenna_id}
            )
        self._events.append(
            {
                "event_type": "antenna_activated" if activated else "antenna_deploy_failed",
                "sim_time": self._sim_time(),
                "payload": {
                    "antenna_id": self._antenna_id,
                    "entity_name": self._antenna_entity_name,
                    "state": self._antenna_unit.get_state().value,
                },
            }
        )
        self._phase = GenesisMissionPhase.DONE

    def _telemetry_record(self) -> dict:
        assert self._engine is not None
        assert self._rng is not None
        rover_pos, _ = self._engine.get_body_pose(self._rover_entity_name)
        rover_lin, rover_ang = self._engine.get_body_velocity(self._rover_entity_name)
        gt = np.asarray(rover_pos, dtype=np.float64).reshape(3)
        est = gt + self._rng.normal(scale=0.04, size=3)
        power_w = 60.0 + (20.0 if self._phase == GenesisMissionPhase.CARRIED_SETTLE else 10.0)
        dt = self._physics.timestep if self._physics is not None else 0.0
        self._energy_wh += power_w * (dt / 3600.0)
        coverage = 1.0 if self.succeeded() else 0.0
        return {
            "timestamp": self._sim_time(),
            "rover_position": gt.tolist(),
            "velocity": np.asarray(rover_lin, dtype=np.float64).reshape(3).tolist(),
            "energy_wh": self._energy_wh,
            "power_consumed_w": power_w,
            "cable_tension_n": 20.0,
            "cable_coverage_fraction": coverage,
            "estimated_position": est.tolist(),
            "ground_truth_position": gt.tolist(),
            "imu": {
                "accel_xyz": [0.0, 0.0, -1.622],
                "gyro_xyz": np.asarray(rover_ang, dtype=np.float64).reshape(3).tolist(),
                "timestamp": self._sim_time(),
            },
        }

    def _should_render(self) -> bool:
        assert self._engine is not None
        assert self._physics is not None
        if not self._visualize:
            return False
        cadence = max(1, round(1.0 / (self._physics.timestep * self.render_hz)))
        return self._engine.get_step_count() % cadence == 0

    def _should_capture(self) -> bool:
        """Capture a video frame at the recorder's target fps cadence."""
        assert self._engine is not None
        assert self._physics is not None and self._recorder is not None
        fps = self._recorder.config.fps
        cadence = max(1, round(1.0 / (self._physics.timestep * fps)))
        return self._engine.get_step_count() % cadence == 0

    def _sim_time(self) -> float:
        return 0.0 if self._engine is None else float(self._engine.get_sim_time())

    def _terrain_height(self, x: float, y: float, fallback: float) -> float:
        assert self._engine is not None
        try:
            return float(self._engine.get_terrain_height(float(x), float(y)))
        except (AttributeError, RuntimeError):
            return float(fallback)

    @staticmethod
    def _build_recorder(spec: Any) -> Optional[VideoRecorder]:
        """Build a :class:`VideoRecorder` from the ``record_video`` config.

        Accepts:
            * ``None`` / ``False`` — recording disabled.
            * a path-like — convenience form: records a rover-tracking shot to
              that file (``track_body`` defaults to the rover sentinel).
            * a mapping of :class:`RecorderConfig` fields, a
              :class:`RecorderConfig`, or a :class:`VideoRecorder` — honored
              verbatim. Set ``track_body`` to :data:`_ROVER_TRACK_SENTINEL` to
              follow the rover, an entity name to follow that body, or ``None``
              for a fixed wide shot.
        """
        if spec is None or spec is False:
            return None
        if isinstance(spec, VideoRecorder):
            return spec
        if isinstance(spec, RecorderConfig):
            return VideoRecorder(spec)
        if isinstance(spec, dict):
            return VideoRecorder(RecorderConfig(**spec))
        if isinstance(spec, (str, Path)):
            return VideoRecorder(
                RecorderConfig(output_path=Path(spec), track_body=_ROVER_TRACK_SENTINEL)
            )
        raise TypeError(
            "record_video must be a path, dict, RecorderConfig, or VideoRecorder; "
            f"got {type(spec).__name__}"
        )

    def _finalize_recording(self) -> None:
        if self._recorder is None:
            return
        self._video_path = self._recorder.close()

    def _teardown_engine(self) -> None:
        if self._engine is None:
            return
        engine, self._engine = self._engine, None
        engine.teardown()

    @staticmethod
    def _positive_int(value: Any, name: str) -> int:
        number = int(value)
        if number <= 0:
            raise ValueError(f"{name} must be > 0")
        return number

    @staticmethod
    def _unit_fraction(value: Any, name: str) -> float:
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError(f"{name} must be in [0.0, 1.0]")
        return number

    @staticmethod
    def _tilt_deg(quat_xyzw: np.ndarray) -> float:
        quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-12:
            return 180.0
        x, y, z, w = quat / norm
        vertical_z = 1.0 - (2.0 * ((x * x) + (y * y)))
        return math.degrees(math.acos(float(np.clip(vertical_z, -1.0, 1.0))))

    @staticmethod
    def _default_engine() -> Any:
        from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine

        return GenesisPhysicsEngine()


def genesis_mission_factory(config: dict) -> Scenario:
    """Scenario factory for :class:`GenesisMissionScenario`."""
    return GenesisMissionScenario(config)
