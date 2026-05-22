"""
System 14: Scenario Runner and Experiment Framework

Experiment orchestration and statistical analysis framework for running
parametric studies, Monte Carlo simulations, and convergence analysis.

The end-to-end harness is built around three pieces:

    Scenario              — one runnable mission episode: owns the engine,
                            scene, and control loop, and emits a per-tick
                            telemetry record.
    MissionScenarioRunner — orchestrates a Scenario: load configs -> build
                            scene -> run mission loop -> collect metrics ->
                            teardown, with optional MCAP/HDF5 logging, plus
                            parameter sweeps and Monte Carlo analysis.
    SyntheticDriveScenario — a GPU-free default scenario (figure-8 drive with
                            antenna drops) so the harness runs out of the box
                            and is unit-testable without Genesis. Real Genesis
                            scenarios subclass Scenario and plug straight in.
"""

from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import yaml

from moon_rover.data.analysis.metrics import MissionMetricsAnalyzer, RunMetrics
from moon_rover.data.logging.streams import LogConfig, MultiStreamLogger, StreamType


@dataclass
class ExperimentConfig:
    """
    Configuration for experiment run with parametric variations.

    Attributes:
        name: Experiment identifier/name.
        variable_params: Dict of parameter_name -> list of values to sweep.
            Creates cartesian product of all combinations.
        fixed_params: Dict of parameter_name -> fixed value for all runs.
        num_seeds: Number of random seeds per parameter combination for
            Monte Carlo estimation.
        success_metrics: List of metric names to track (e.g., 'mission_time_s',
            'cable_drag_energy_j', 'success_rate').
        min_runs_significance: Minimum number of runs required before
            statistical tests are valid. Default: 30.
        convergence_threshold: Relative change threshold for convergence
            detection. Default: 0.05 (5% relative change).
    """

    name: str
    variable_params: dict[str, list]
    fixed_params: dict[str, Any]
    num_seeds: int
    success_metrics: list[str]
    min_runs_significance: int = 30
    convergence_threshold: float = 0.05


@dataclass
class RunResult:
    """
    Result from single simulation run.

    Attributes:
        seed: Random seed used for this run.
        rover_type: Rover type/configuration used.
        metrics: Dict mapping metric_name -> value (floats).
        faults: List of FaultType enums encountered during run.
        success: Boolean indicating mission success/completion.
        duration_s: Simulation duration in seconds.
    """

    seed: int
    rover_type: str
    metrics: dict[str, float]
    faults: list[str]
    success: bool
    duration_s: float


# ---------------------------------------------------------------------------
# Scenario interface + episode result
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """Outcome of a single ``run_episode`` call."""

    seed: int
    success: bool
    duration_s: float
    steps: int
    metrics: RunMetrics
    log_data: dict
    rover_type: str = "generic"
    log_dir: Optional[Path] = None


class Scenario(ABC):
    """One runnable mission episode.

    A scenario owns its physics engine, scene, and control loop. The runner
    drives it through ``setup -> (step)* -> teardown`` and assembles the
    per-tick telemetry records into the ``log_data`` dict consumed by
    :class:`~moon_rover.data.analysis.metrics.MissionMetricsAnalyzer`.

    Each ``step()`` returns a telemetry record. Recognized keys (all optional
    except ``timestamp`` and ``rover_position``) are aggregated by the runner:

        timestamp                float (seconds)            REQUIRED
        rover_position           [x, y, z]                  REQUIRED
        power_consumed_w         float
        cable_tension_n          float
        cable_coverage_fraction  float (0-1)
        estimated_position       [x, y, z]
        ground_truth_position    [x, y, z]

    Per-tick records may also carry ``camera_rgb`` (HxWx3 uint8 array),
    ``lidar_points`` (Nx3 array), and ``imu`` (dict) which the runner streams to
    the logger but does not aggregate.
    """

    rover_type: str = "generic"

    @abstractmethod
    def setup(self, seed: int, *, visualize: bool = False) -> None:
        """Build the scene, spawn rover(s), and prepare the control loop."""
        raise NotImplementedError

    def apply_action(self, action: Any) -> None:
        """Apply an external control action before the next ``step()``.

        Default is a no-op: self-driving / scripted scenarios ignore actions.
        Action-driven (RL) scenarios override this to consume the action so the
        same env loop serves both scripted and RL modes identically.
        """
        return None

    @abstractmethod
    def step(self) -> dict:
        """Advance one control tick and return a telemetry record."""
        raise NotImplementedError

    @abstractmethod
    def is_complete(self) -> bool:
        """Return True once the mission objective is reached (early stop)."""
        raise NotImplementedError

    @abstractmethod
    def teardown(self) -> None:
        """Release engine/scene resources."""
        raise NotImplementedError

    @property
    def antenna_placements(self) -> list[dict]:
        """Placement events: {antenna_id, target, actual, success, sim_time}."""
        return []

    @property
    def faults(self) -> list[dict]:
        """Fault events: {mode, time, ...}."""
        return []

    @property
    def events(self) -> list[dict]:
        """Mission events for the log: {event_type, payload, sim_time}."""
        return []

    def succeeded(self) -> bool:
        """Whether the run met its success criteria (default: completed)."""
        return self.is_complete()


# ---------------------------------------------------------------------------
# Built-in synthetic scenario (GPU-free, deterministic)
# ---------------------------------------------------------------------------


class SyntheticDriveScenario(Scenario):
    """Deterministic figure-8 drive with timed antenna drops.

    Pure-Python kinematics; no Genesis dependency. Useful as the harness's
    out-of-the-box default and as a fast, deterministic substrate for tests.
    Config keys (all optional):

        duration_s        float   mission length (default 8.0)
        dt                float   control tick (default 0.02 -> 50 Hz)
        radius_m          float   figure-8 scale (default 6.0)
        cruise_speed_mps  float   forward speed (default 0.8)
        placement_noise_m float   antenna placement error scale (default 0.15)
        waypoints         list    [[t_seconds, antenna_id], ...] drop schedule
        fault_at_s        float   optional time to inject a single fault
        fault_mode        str     fault tag (default 'injected_fault')
    """

    rover_type = "synthetic_diff_drive"

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = dict(config or {})
        self.duration_s = float(cfg.get("duration_s", 8.0))
        self.dt = float(cfg.get("dt", 0.02))
        self.cruise_speed_mps = float(cfg.get("cruise_speed_mps", 0.8))
        self.placement_noise_m = float(cfg.get("placement_noise_m", 0.15))
        self.waypoints = list(
            cfg.get("waypoints", [[1.0, "antenna_01"], [3.5, "antenna_02"], [6.0, "antenna_03"]])
        )
        self.fault_at_s = cfg.get("fault_at_s")
        self.fault_mode = str(cfg.get("fault_mode", "injected_fault"))

        self._rng: Optional[np.random.Generator] = None
        self._position = np.zeros(3)
        self._heading = 0.0
        self._energy_wh = 0.0
        self._sim_time = 0.0
        self._step_idx = 0
        self._n_steps = 0
        self._next_wp = 0
        self._fault_done = False
        self._placements: list[dict] = []
        self._faults: list[dict] = []
        self._events: list[dict] = []

    def setup(self, seed: int, *, visualize: bool = False) -> None:
        self._rng = np.random.default_rng(seed)
        self._position = np.zeros(3)
        self._heading = 0.0
        self._energy_wh = 0.0
        self._sim_time = 0.0
        self._step_idx = 0
        self._n_steps = int(self.duration_s / self.dt)
        self._next_wp = 0
        self._fault_done = False
        self._placements = []
        self._faults = []
        self._events = [
            {"event_type": "mission_start", "sim_time": 0.0, "payload": {"seed": seed}}
        ]

    def _command(self, t: float) -> tuple[float, float]:
        omega = 2.0 * math.pi / self.duration_s
        yaw_rate = omega if t < self.duration_s / 2.0 else -omega
        return self.cruise_speed_mps, yaw_rate

    def step(self) -> dict:
        assert self._rng is not None, "setup() must be called before step()"
        t_episode = self._step_idx * self.dt
        v_cmd, yaw_cmd = self._command(t_episode)

        self._heading += yaw_cmd * self.dt
        vx = v_cmd * math.cos(self._heading)
        vy = v_cmd * math.sin(self._heading)
        velocity = np.array([vx, vy, 0.0])
        self._position = self._position + velocity * self.dt
        power_w = 60.0 + 80.0 * abs(v_cmd) + 40.0 * abs(yaw_cmd)
        self._energy_wh += power_w * (self.dt / 3600.0)
        self._sim_time += self.dt
        self._step_idx += 1

        gt = self._position.copy()
        est = gt + self._rng.normal(scale=0.05, size=3)
        cable_tension = 20.0 + 8.0 * float(np.linalg.norm(self._position))
        coverage = self._step_idx / max(1, self._n_steps)

        # Timed antenna drop.
        if self._next_wp < len(self.waypoints) and t_episode >= self.waypoints[self._next_wp][0]:
            antenna_id = str(self.waypoints[self._next_wp][1])
            target = self._position.copy()
            actual = target + self._rng.normal(scale=self.placement_noise_m, size=3)
            success = float(np.linalg.norm(actual - target)) <= 0.5
            self._placements.append(
                {
                    "antenna_id": antenna_id,
                    "target": target.tolist(),
                    "actual": actual.tolist(),
                    "success": success,
                    "sim_time": self._sim_time,
                    "failure_mode": None if success else "placement_inaccurate",
                }
            )
            self._events.append(
                {
                    "event_type": "antenna_placed",
                    "sim_time": self._sim_time,
                    "payload": {"antenna_id": antenna_id, "success": success},
                }
            )
            self._next_wp += 1

        # Optional fault injection.
        if (
            self.fault_at_s is not None
            and not self._fault_done
            and t_episode >= float(self.fault_at_s)
        ):
            self._faults.append({"mode": self.fault_mode, "time": self._sim_time})
            self._events.append(
                {"event_type": "fault", "sim_time": self._sim_time, "payload": {"mode": self.fault_mode}}
            )
            self._fault_done = True

        return {
            "timestamp": self._sim_time,
            "rover_position": self._position.tolist(),
            "heading": self._heading,
            "velocity": velocity.tolist(),
            "energy_wh": self._energy_wh,
            "power_consumed_w": power_w,
            "cable_tension_n": cable_tension,
            "cable_coverage_fraction": coverage,
            "estimated_position": est.tolist(),
            "ground_truth_position": gt.tolist(),
            "imu": {
                "accel_xyz": [0.0, 0.0, -1.62],
                "gyro_xyz": [0.0, 0.0, self._heading],
                "timestamp": self._sim_time,
            },
        }

    def is_complete(self) -> bool:
        return self._step_idx >= self._n_steps

    def teardown(self) -> None:
        self._events.append(
            {"event_type": "mission_end", "sim_time": self._sim_time, "payload": {}}
        )

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
        # Mission is a success if every scheduled antenna was placed accurately.
        if not self._placements:
            return False
        return all(p["success"] for p in self._placements)


def default_scenario_factory(config: dict) -> Scenario:
    """Default factory: a GPU-free synthetic figure-8 drive scenario."""
    return SyntheticDriveScenario(config)


# ---------------------------------------------------------------------------
# ScenarioRunner ABC
# ---------------------------------------------------------------------------


class ScenarioRunner(ABC):
    """
    Experiment runner orchestrating parametric studies and Monte Carlo analysis.

    Handles:
    - Loading experiment configurations from YAML
    - Running single simulations with seed control
    - Parallel execution of Monte Carlo runs
    - Statistical convergence checking
    - Report generation with distributions and failure analysis
    """

    @abstractmethod
    def load_experiment(self, experiment_yaml: str) -> ExperimentConfig:
        """Load experiment configuration from YAML file."""
        raise NotImplementedError

    @abstractmethod
    def run_single(self, seed: int, config: dict) -> RunResult:
        """Execute single simulation run with given parameters."""
        raise NotImplementedError

    @abstractmethod
    def run_monte_carlo(
        self, experiment: ExperimentConfig, num_workers: int = 1
    ) -> list[RunResult]:
        """Execute full Monte Carlo experiment."""
        raise NotImplementedError

    @abstractmethod
    def check_convergence(self, results: list[RunResult], metric: str) -> bool:
        """Check if results have converged for specified metric."""
        raise NotImplementedError

    @abstractmethod
    def generate_report(self, results: list[RunResult]) -> dict:
        """Generate statistical summary report from results."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete MissionScenarioRunner
# ---------------------------------------------------------------------------

# Recognized scalar/vector keys aggregated into log_data for the analyzer.
_VECTOR_KEYS = ("rover_position", "estimated_position", "ground_truth_position")
_SCALAR_KEYS = ("power_consumed_w", "cable_tension_n", "cable_coverage_fraction")


class MissionScenarioRunner(ScenarioRunner):
    """End-to-end simulation harness.

    ``run_episode`` orchestrates one mission: build scene (via the scenario)
    -> run the mission loop -> collect KPIs -> teardown, optionally streaming
    telemetry to MCAP/HDF5. ``run_single`` / ``run_monte_carlo`` build sweeps
    on top of it; ``check_convergence`` and ``generate_report`` provide the
    statistical layer.

    The runner is scenario-agnostic: pass any ``scenario_factory`` that returns
    a :class:`Scenario`. The default factory yields a GPU-free synthetic drive
    so the harness runs and tests without Genesis. A real-Genesis scenario
    subclasses :class:`Scenario` and is injected the same way.
    """

    def __init__(
        self,
        scenario_factory: Callable[[dict], Scenario] = default_scenario_factory,
        analyzer: Optional[MissionMetricsAnalyzer] = None,
        *,
        max_steps: int = 100_000,
    ) -> None:
        self._scenario_factory = scenario_factory
        self._analyzer = analyzer or MissionMetricsAnalyzer()
        self._max_steps = int(max_steps)

    # ------------------------------------------------------------------
    # Core orchestration
    # ------------------------------------------------------------------

    def run_episode(
        self,
        config: Optional[dict] = None,
        *,
        seed: int = 0,
        visualize: bool = False,
        log_dir: Optional[Path | str] = None,
        max_steps: Optional[int] = None,
    ) -> EpisodeResult:
        """Run a single mission episode end to end.

        Args:
            config: Scenario parameters passed to the scenario factory.
            seed: RNG seed for reproducibility.
            visualize: Request the scenario's interactive viewer (real Genesis
                scenarios honor this; the synthetic default ignores it).
            log_dir: If set, telemetry is streamed to ``<log_dir>/log.mcap`` and
                ``<log_dir>/log.h5`` via MultiStreamLogger.
            max_steps: Hard cap on control ticks (defaults to runner's cap).

        Returns:
            EpisodeResult with computed RunMetrics and the assembled log_data.
        """
        config = dict(config or {})
        step_cap = int(max_steps if max_steps is not None else self._max_steps)
        run_id = config.get("run_id", f"episode_seed{seed}")

        scenario = self._scenario_factory(config)
        logger: Optional[MultiStreamLogger] = None
        log_path: Optional[Path] = None
        if log_dir is not None:
            log_path = Path(log_dir)
            logger = MultiStreamLogger()
            logger.initialize(
                LogConfig(output_dir=str(log_path), enable_hdf5=True, enable_mcap=True)
            )

        series: dict[str, list] = {k: [] for k in ("timestamp",) + _VECTOR_KEYS + _SCALAR_KEYS}
        steps = 0
        try:
            scenario.setup(seed, visualize=visualize)
            self._emit_events(logger, scenario, drain=True)

            while not scenario.is_complete() and steps < step_cap:
                record = scenario.step()
                steps += 1
                self._accumulate(series, record)
                self._stream_record(logger, run_id, record)
                self._emit_events(logger, scenario, drain=True)

            scenario.teardown()
            self._emit_events(logger, scenario, drain=True)
        finally:
            if logger is not None:
                logger.flush()
                logger.close()

        log_data = self._assemble_log_data(run_id, series, scenario)
        metrics = self._analyzer.compute_run_metrics(log_data)
        duration_s = float(series["timestamp"][-1]) if series["timestamp"] else 0.0

        return EpisodeResult(
            seed=seed,
            success=bool(scenario.succeeded()),
            duration_s=duration_s,
            steps=steps,
            metrics=metrics,
            log_data=log_data,
            rover_type=getattr(scenario, "rover_type", "generic"),
            log_dir=log_path,
        )

    @staticmethod
    def _accumulate(series: dict[str, list], record: dict) -> None:
        if "timestamp" not in record or "rover_position" not in record:
            raise ValueError("scenario step record must include 'timestamp' and 'rover_position'")
        series["timestamp"].append(float(record["timestamp"]))
        for key in _VECTOR_KEYS:
            if key in record:
                series[key].append(np.asarray(record[key], dtype=np.float64))
        for key in _SCALAR_KEYS:
            if key in record:
                series[key].append(float(record[key]))

    @staticmethod
    def _stream_record(logger: Optional[MultiStreamLogger], run_id: str, record: dict) -> None:
        if logger is None:
            return
        logger.log_rover_state(
            run_id,
            {
                "position": record["rover_position"],
                "heading": record.get("heading"),
                "velocity": record.get("velocity"),
                "energy_wh": record.get("energy_wh"),
                "cable_tension_n": record.get("cable_tension_n"),
                "timestamp": record["timestamp"],
            },
        )
        if "imu" in record:
            logger.log_sensor_reading(StreamType.SENSOR_IMU, dict(record["imu"]))
        if "lidar_points" in record:
            logger.log_sensor_reading(
                StreamType.SENSOR_LIDAR,
                {"points": np.asarray(record["lidar_points"]), "timestamp": record["timestamp"]},
            )
        if "camera_rgb" in record:
            logger.log_camera_frame(np.asarray(record["camera_rgb"]), StreamType.CAMERA_RGB)

    @staticmethod
    def _emit_events(
        logger: Optional[MultiStreamLogger], scenario: Scenario, *, drain: bool
    ) -> None:
        if logger is None:
            return
        # Stream any events not yet flushed. We track via a private cursor on the
        # scenario object so repeated drains don't double-log.
        cursor = getattr(scenario, "_event_cursor", 0)
        events = scenario.events
        for ev in events[cursor:]:
            payload = dict(ev.get("payload", {}))
            payload.setdefault("timestamp", ev.get("sim_time", 0.0))
            logger.log_event(str(ev.get("event_type", "event")), payload)
        scenario._event_cursor = len(events)  # type: ignore[attr-defined]

    def _assemble_log_data(self, run_id: str, series: dict[str, list], scenario: Scenario) -> dict:
        log_data: dict[str, Any] = {"run_id": run_id}
        log_data["timestamp"] = np.asarray(series["timestamp"], dtype=np.float64)
        for key in _VECTOR_KEYS:
            if series[key]:
                log_data[key] = np.stack(series[key], axis=0)
        for key in _SCALAR_KEYS:
            if series[key]:
                log_data[key] = np.asarray(series[key], dtype=np.float64)
        log_data["antenna_placements"] = list(scenario.antenna_placements)
        log_data["faults"] = list(scenario.faults)
        return log_data

    # ------------------------------------------------------------------
    # Experiment framework
    # ------------------------------------------------------------------

    def load_experiment(self, experiment_yaml: str) -> ExperimentConfig:
        path = Path(experiment_yaml)
        if not path.exists():
            raise FileNotFoundError(f"experiment file not found: {experiment_yaml}")
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ValueError("experiment YAML must be a mapping")
        try:
            return ExperimentConfig(
                name=str(raw["name"]),
                variable_params=dict(raw.get("variable_params", {})),
                fixed_params=dict(raw.get("fixed_params", {})),
                num_seeds=int(raw["num_seeds"]),
                success_metrics=list(raw["success_metrics"]),
                min_runs_significance=int(raw.get("min_runs_significance", 30)),
                convergence_threshold=float(raw.get("convergence_threshold", 0.05)),
            )
        except KeyError as exc:
            raise ValueError(f"experiment YAML missing required key: {exc}") from exc

    def run_single(self, seed: int, config: dict) -> RunResult:
        episode = self.run_episode(config, seed=seed)
        return self._to_run_result(episode, config)

    def run_monte_carlo(
        self, experiment: ExperimentConfig, num_workers: int = 1
    ) -> list[RunResult]:
        # num_workers is accepted for API compatibility; execution is serial to
        # keep determinism and avoid multiprocessing hazards with a stateful
        # physics backend. Parallelism, if needed, belongs in a process pool of
        # whole experiments rather than mid-episode.
        results: list[RunResult] = []
        for combo in self._param_combos(experiment):
            for seed in range(experiment.num_seeds):
                cfg = dict(experiment.fixed_params)
                cfg.update(combo)
                results.append(self.run_single(seed, cfg))
        return results

    def check_convergence(self, results: list[RunResult], metric: str) -> bool:
        values = [r.metrics[metric] for r in results if metric in r.metrics]
        if len(values) < 2:
            raise ValueError(f"metric {metric!r} has too few samples to assess convergence")
        # Threshold lives on the runner default unless results carry their own.
        threshold = 0.05
        n = len(values)
        if n < 4:
            return False
        first_half = float(np.mean(values[: n // 2]))
        second_half = float(np.mean(values[n // 2 :]))
        denom = abs(first_half) if abs(first_half) > 1e-12 else 1e-12
        rel_change = abs(second_half - first_half) / denom
        return rel_change <= threshold

    def generate_report(self, results: list[RunResult]) -> dict:
        if not results:
            return {
                "metric_summaries": {},
                "success_rate": 0.0,
                "failure_modes": {},
                "n_runs": 0,
            }

        metric_names: set[str] = set()
        for r in results:
            metric_names.update(r.metrics.keys())

        summaries: dict[str, dict] = {}
        for name in sorted(metric_names):
            vals = np.array(
                [r.metrics[name] for r in results if name in r.metrics], dtype=np.float64
            )
            summaries[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p95": float(np.percentile(vals, 95)),
                "p99": float(np.percentile(vals, 99)),
            }

        failure_modes: dict[str, int] = {}
        for r in results:
            for mode in r.faults:
                failure_modes[mode] = failure_modes.get(mode, 0) + 1

        success_rate = float(np.mean([1.0 if r.success else 0.0 for r in results]))

        return {
            "metric_summaries": summaries,
            "success_rate": success_rate,
            "failure_modes": failure_modes,
            "n_runs": len(results),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _param_combos(experiment: ExperimentConfig) -> list[dict]:
        if not experiment.variable_params:
            return [{}]
        keys = list(experiment.variable_params.keys())
        value_lists = [experiment.variable_params[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    def _to_run_result(self, episode: EpisodeResult, config: dict) -> RunResult:
        m = episode.metrics
        metrics = {
            "total_distance_m": m.total_distance_m,
            "energy_consumed_wh": m.energy_consumed_wh,
            "energy_per_antenna_wh": m.energy_per_antenna_wh(),
            "mission_time_s": m.mission_time_s,
            "cable_drag_energy_j": m.cable_drag_energy_j,
            "cable_coverage_percent": m.cable_coverage_percent,
            "placement_success_rate": m.success_rate(),
            "mean_placement_accuracy_m": m.mean_placement_accuracy_m(),
            "localization_error_drift_m": m.localization_error_drift_m,
            "fault_count": float(m.fault_count),
        }
        return RunResult(
            seed=episode.seed,
            rover_type=str(config.get("rover_type", episode.rover_type)),
            metrics=metrics,
            faults=list(m.failure_modes),
            success=episode.success,
            duration_s=episode.duration_s,
        )
