"""System 13.3: Analysis Toolkit for Mission Performance Metrics.

This module provides data structures and utilities for computing, aggregating,
and analyzing Moon Rover mission performance metrics across single runs and
Monte Carlo experiment sets.

Key Capabilities:
    - Per-run metrics collection (distance, energy, time, faults)
    - Per-antenna placement accuracy tracking
    - Cross-run statistical analysis and comparison
    - Failure mode Pareto analysis
    - Path quality assessment (planned vs actual)
    - Cable health and tension monitoring

Classes:
    RunMetrics (dataclass): Single mission run performance summary
    AnalysisToolkit (ABC): High-level analysis interface
    MissionMetricsAnalyzer: Concrete production analyzer

Typical Usage:
    analyzer = MissionMetricsAnalyzer()
    metrics = analyzer.compute_run_metrics(log_data)
    df = analyzer.to_dataframe([m1, m2, m3])
    stats = analyzer.compare_across_runs([m1, m2, m3])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from moon_rover.antenna.array import ArrayDesign, ArrayElementTarget, BaselineResidual


@dataclass
class RunMetrics:
    """Summary metrics from a single Moon Rover mission run.

    Captures aggregate mission statistics, per-antenna deployment accuracy,
    and fault occurrences. These metrics enable post-mission analysis,
    Monte Carlo statistical inference, and design optimization.

    Attributes:
        total_distance_m (float): Total rover travel distance in meters.
                                 Computed via odometry or path length.
        energy_consumed_wh (float): Total energy consumed in watt-hours.
                                   Includes: propulsion, payload, thermal.
        mission_time_s (float): Total mission elapsed time in seconds.
                               From start to mission completion or timeout.
        cable_drag_energy_j (float): Energy lost to cable drag in joules.
                                    Estimated from tension history.
        placement_accuracy_m (Dict[str, float]): Per-antenna final accuracy.
                                                Key: antenna_id, Value: error in meters.
                                                Computed as distance from target after placement.
        fault_count (int): Total number of faults injected/encountered.
        antennas_deployed (int): Number of antennas successfully deployed.
        antennas_failed (int): Number of antenna deployments that failed.
        failure_modes (List[str]): Sequence of failure-mode tags encountered
                                   during the run, used for Pareto analysis.
        cable_coverage_percent (float): Fraction of the target grid covered by
                                       deployed cable, expressed as 0-100.
        localization_error_drift_m (float): Mean Euclidean error between the
                                           estimator's position and ground
                                           truth over the run, in meters.
        run_id (Optional[str]): Optional human-readable identifier carried
                               into the DataFrame for joins.
    """

    total_distance_m: float = 0.0
    """Total travel distance of rover(s) in meters."""

    energy_consumed_wh: float = 0.0
    """Total energy consumed by all systems in watt-hours."""

    mission_time_s: float = 0.0
    """Total mission duration in seconds."""

    cable_drag_energy_j: float = 0.0
    """Energy dissipated due to cable drag and tension losses in joules."""

    placement_accuracy_m: Dict[str, float] = field(default_factory=dict)
    """Per-antenna placement accuracy.

    Maps antenna IDs to final position error in meters.
    Example: {"antenna_1": 0.3, "antenna_2": 0.15}
    Accuracy computed as Euclidean distance from target after placement.
    """

    fault_count: int = 0
    """Total number of faults injected during the mission."""

    antennas_deployed: int = 0
    """Count of successfully deployed antennas."""

    antennas_failed: int = 0
    """Count of antenna deployments that failed due to faults or constraints."""

    failure_modes: List[str] = field(default_factory=list)
    """Failure-mode tags encountered (one entry per fault)."""

    cable_coverage_percent: float = 0.0
    """Fraction of target grid covered by deployed cable, 0-100."""

    localization_error_drift_m: float = 0.0
    """Mean estimator-vs-truth position error over the run, in meters."""

    run_id: Optional[str] = None
    """Optional run identifier (carried into DataFrame for joins)."""

    def mean_placement_accuracy_m(self) -> float:
        """Compute mean placement accuracy across all antennas.

        Returns:
            float: Mean accuracy in meters. Returns 0.0 if no antennas deployed.
        """
        if not self.placement_accuracy_m:
            return 0.0
        return float(np.mean(list(self.placement_accuracy_m.values())))

    def success_rate(self) -> float:
        """Compute antenna deployment success rate.

        Returns:
            float: Fraction of attempted deployments that succeeded [0.0, 1.0].
                  Returns 0.0 if no deployments attempted.
        """
        total = self.antennas_deployed + self.antennas_failed
        if total == 0:
            return 0.0
        return self.antennas_deployed / total

    def energy_per_antenna_wh(self) -> float:
        """Energy spent per successfully deployed antenna (Wh)."""
        if self.antennas_deployed == 0:
            return 0.0
        return self.energy_consumed_wh / self.antennas_deployed


@dataclass(frozen=True)
class ArrayElementResidual:
    """Per-element geometric accuracy of a deployed array element vs its target.

    Attributes:
        element_id: Design element identifier this record scores.
        placed: True if a deployed position was supplied for the element.
        position_error_m: Euclidean distance (m) between the deployed position
            and the surveyed target, or ``None`` if the element was not placed.
        tilt_error_deg: Angular deviation (deg) between the deployed mast
            up-axis and the target up-axis, or ``None`` if no orientation was
            supplied for the element.
        within_position_tolerance: True if ``position_error_m`` is within the
            element's ``placement_tolerance_m``. False when not placed.
        within_tilt_tolerance: True if ``tilt_error_deg`` is within the
            element's ``tilt_tolerance_deg``. True when no orientation was
            supplied (tilt is not evaluated and does not penalize the element).
        within_tolerance: True iff the element is placed and satisfies both the
            position and tilt tolerances — i.e. it counts toward array quality.
    """

    element_id: str
    placed: bool
    position_error_m: Optional[float]
    tilt_error_deg: Optional[float]
    within_position_tolerance: bool
    within_tilt_tolerance: bool
    within_tolerance: bool


@dataclass(frozen=True)
class ArrayQualityReport:
    """Completion and quality assessment of a deployed interferometric array.

    Scores the deployed product against an :class:`ArrayDesign`: how many of
    the designed elements were placed, how accurately each sits relative to its
    surveyed target, and whether the pairwise baselines that define the
    instrument are within tolerance.

    Attributes:
        array_id: Identifier of the scored :class:`ArrayDesign`.
        elements_designed: Number of elements the design calls for.
        elements_placed: Number of designed elements with a deployed position.
        elements_within_tolerance: Placed elements meeting both position and
            tilt tolerances.
        completion_fraction: ``elements_placed / elements_designed`` in [0, 1].
        quality_fraction: ``elements_within_tolerance / elements_designed`` in
            [0, 1] — the fraction of the design that is built *to spec*.
        mean_position_error_m / max_position_error_m / rms_position_error_m:
            Position-error statistics over placed elements (0.0 if none placed).
        baselines_designed: Number of pairwise baselines in the design.
        baselines_evaluated: Baselines whose both endpoints were placed.
        baselines_within_tolerance: Evaluated baselines within their tolerance.
        mean_baseline_error_m / max_baseline_error_m: Baseline-error statistics
            over evaluated baselines (0.0 if none evaluated).
        array_complete: True iff every designed element is placed within
            tolerance and every designed baseline was evaluated and is within
            tolerance — i.e. the array is built to spec.
        element_residuals: Per-element accuracy records (design order).
        baseline_residuals: Per-baseline residuals for evaluated baselines.
        run_id: Optional run identifier carried for joins/logging.
    """

    array_id: str
    elements_designed: int
    elements_placed: int
    elements_within_tolerance: int
    completion_fraction: float
    quality_fraction: float
    mean_position_error_m: float
    max_position_error_m: float
    rms_position_error_m: float
    baselines_designed: int
    baselines_evaluated: int
    baselines_within_tolerance: int
    mean_baseline_error_m: float
    max_baseline_error_m: float
    array_complete: bool
    element_residuals: tuple[ArrayElementResidual, ...] = field(default_factory=tuple)
    baseline_residuals: tuple[BaselineResidual, ...] = field(default_factory=tuple)
    run_id: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        """Flatten the headline KPIs into a dict for the dashboard / logs."""
        return {
            "array_id": self.array_id,
            "run_id": self.run_id,
            "elements_designed": self.elements_designed,
            "elements_placed": self.elements_placed,
            "elements_within_tolerance": self.elements_within_tolerance,
            "completion_fraction": self.completion_fraction,
            "quality_fraction": self.quality_fraction,
            "mean_position_error_m": self.mean_position_error_m,
            "max_position_error_m": self.max_position_error_m,
            "rms_position_error_m": self.rms_position_error_m,
            "baselines_designed": self.baselines_designed,
            "baselines_evaluated": self.baselines_evaluated,
            "baselines_within_tolerance": self.baselines_within_tolerance,
            "mean_baseline_error_m": self.mean_baseline_error_m,
            "max_baseline_error_m": self.max_baseline_error_m,
            "array_complete": self.array_complete,
        }


class AnalysisToolkit(ABC):
    """Abstract interface for Moon Rover mission performance analysis.

    Provides high-level methods for:
    - Computing aggregate metrics from raw telemetry logs
    - Comparing performance across multiple runs (Monte Carlo)
    - Analyzing failure modes and identifying critical paths
    - Assessing trajectory quality (planned vs actual)
    - Monitoring cable health and integrity
    """

    @abstractmethod
    def compute_run_metrics(self, log_data: Dict) -> RunMetrics:
        """Compute aggregate metrics from a single mission run."""
        raise NotImplementedError

    @abstractmethod
    def compare_across_runs(self, results: List[RunMetrics]) -> Dict:
        """Perform statistical analysis across multiple mission runs."""
        raise NotImplementedError

    @abstractmethod
    def failure_mode_analysis(self, results: List[RunMetrics]) -> Dict:
        """Analyze failure modes and compute Pareto chart data."""
        raise NotImplementedError

    @abstractmethod
    def path_quality_metrics(
        self,
        planned: List[np.ndarray],
        actual: List[np.ndarray],
    ) -> Dict:
        """Evaluate path execution quality (planned vs actual trajectory)."""
        raise NotImplementedError

    @abstractmethod
    def cable_health_report(self, tension_series: np.ndarray) -> Dict:
        """Analyze cable tension history and estimate cable health."""
        raise NotImplementedError

    @abstractmethod
    def array_quality_report(
        self,
        design: ArrayDesign,
        placements: Any,
        run_id: Optional[str] = None,
    ) -> ArrayQualityReport:
        """Score a deployed antenna array against its design specification."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete analyzer
# ---------------------------------------------------------------------------


@dataclass
class MetricsConfig:
    """Tuning knobs for KPI computation.

    Attributes:
        placement_success_tolerance_m: Distance threshold (m) below which a
            placement counts as successful when ``success`` is not supplied.
        cable_max_tension_n: Nominal cable break load (N) used as the
            denominator for ``safety_margin`` and the fatigue estimate.
        cable_overstress_threshold_n: Tension above which an event is logged
            in ``overstress_events`` and excluded from "good" health.
        cable_overstress_health_critical_count: Overstress events above which
            health is rated ``critical``.
        cable_overstress_health_degraded_count: Overstress events above which
            health is rated ``degraded``.
        cable_rated_cycles: Manufacturer-rated load cycles for fatigue %.
    """

    placement_success_tolerance_m: float = 0.5
    cable_max_tension_n: float = 1000.0
    cable_overstress_threshold_n: float = 800.0
    cable_overstress_health_critical_count: int = 50
    cable_overstress_health_degraded_count: int = 5
    cable_rated_cycles: int = 100_000


class MissionMetricsAnalyzer(AnalysisToolkit):
    """Production analyzer for Moon Rover mission KPIs.

    Expected ``log_data`` schema (all keys optional unless noted):

        timestamp                    np.ndarray[N]   seconds, monotonic — REQUIRED
        rover_position               np.ndarray[N, 3]  meters
        power_consumed_w             np.ndarray[N]   instantaneous Watts
        cable_tension_n              np.ndarray[N]   Newtons
        cable_coverage_fraction      np.ndarray[N]   0-1, final value used as % covered
        estimated_position           np.ndarray[N, 3]  estimator output, m
        ground_truth_position        np.ndarray[N, 3]  reference truth, m
        antenna_placements           list[dict]      see _compute_placement_metrics
        faults                       list[dict]      {"mode": str, "time": float, ...}

    Output: a fully populated :class:`RunMetrics`. Aggregating across runs
    is done with :meth:`to_dataframe` and :meth:`compare_across_runs`.
    """

    def __init__(self, config: Optional[MetricsConfig] = None) -> None:
        self.config = config or MetricsConfig()

    # ------------------------------------------------------------------
    # Single-run KPI computation
    # ------------------------------------------------------------------

    def compute_run_metrics(self, log_data: Dict) -> RunMetrics:
        timestamps = _as_array(log_data.get("timestamp"), dtype=np.float64)
        if timestamps is None or timestamps.size == 0:
            raise ValueError("log_data must contain non-empty 'timestamp' array")
        if timestamps.ndim != 1:
            raise ValueError(
                f"'timestamp' must be 1-D; got shape {timestamps.shape}"
            )

        mission_time = float(timestamps[-1] - timestamps[0])

        positions = _as_array(log_data.get("rover_position"))
        total_distance = _path_length(positions)

        power = _as_array(log_data.get("power_consumed_w"), dtype=np.float64)
        energy_wh = _trapezoidal_energy_wh(timestamps, power)

        tension = _as_array(log_data.get("cable_tension_n"), dtype=np.float64)
        cable_drag_j = _trapezoidal_integral(timestamps, tension)

        deployed, failed, accuracies, modes_from_placements = (
            self._compute_placement_metrics(log_data.get("antenna_placements") or [])
        )

        faults = list(log_data.get("faults") or [])
        fault_modes = [str(f.get("mode", "unknown")) for f in faults]
        all_modes = modes_from_placements + fault_modes

        coverage = _final_coverage_percent(log_data.get("cable_coverage_fraction"))

        drift = _localization_drift(
            log_data.get("estimated_position"),
            log_data.get("ground_truth_position"),
        )

        return RunMetrics(
            total_distance_m=total_distance,
            energy_consumed_wh=energy_wh,
            mission_time_s=mission_time,
            cable_drag_energy_j=cable_drag_j,
            placement_accuracy_m=accuracies,
            fault_count=len(faults),
            antennas_deployed=deployed,
            antennas_failed=failed,
            failure_modes=all_modes,
            cable_coverage_percent=coverage,
            localization_error_drift_m=drift,
            run_id=log_data.get("run_id"),
        )

    def _compute_placement_metrics(
        self, placements: List[Dict]
    ) -> tuple[int, int, Dict[str, float], List[str]]:
        deployed = 0
        failed = 0
        accuracies: Dict[str, float] = {}
        failure_modes: List[str] = []
        tol = self.config.placement_success_tolerance_m
        for ev in placements:
            aid = str(ev.get("antenna_id", f"antenna_{len(accuracies)}"))
            target = ev.get("target")
            actual = ev.get("actual")
            err: Optional[float] = None
            if target is not None and actual is not None:
                err = float(np.linalg.norm(np.asarray(actual) - np.asarray(target)))
                accuracies[aid] = err
            success = ev.get("success")
            if success is None and err is not None:
                success = err <= tol
            if success is True:
                deployed += 1
            elif success is False:
                failed += 1
                failure_modes.append(str(ev.get("failure_mode", "placement_failed")))
        return deployed, failed, accuracies, failure_modes

    # ------------------------------------------------------------------
    # Cross-run aggregation
    # ------------------------------------------------------------------

    _DATAFRAME_COLUMNS: tuple[str, ...] = (
        "run_id",
        "total_distance_m",
        "energy_consumed_wh",
        "energy_per_antenna_wh",
        "mission_time_s",
        "cable_drag_energy_j",
        "cable_coverage_percent",
        "antennas_deployed",
        "antennas_failed",
        "placement_success_rate",
        "mean_placement_accuracy_m",
        "localization_error_drift_m",
        "fault_count",
    )

    def to_dataframe(self, results: List[RunMetrics]) -> pd.DataFrame:
        """Render a list of run summaries as a per-run pandas DataFrame.

        One row per run, one column per KPI. Suitable for direct ingestion by
        the dashboard or for groupby comparisons across experiment sweeps.
        """
        rows = [self._as_row(m) for m in results]
        return pd.DataFrame(rows, columns=list(self._DATAFRAME_COLUMNS))

    @staticmethod
    def _as_row(m: RunMetrics) -> Dict[str, Any]:
        return {
            "run_id": m.run_id,
            "total_distance_m": m.total_distance_m,
            "energy_consumed_wh": m.energy_consumed_wh,
            "energy_per_antenna_wh": m.energy_per_antenna_wh(),
            "mission_time_s": m.mission_time_s,
            "cable_drag_energy_j": m.cable_drag_energy_j,
            "cable_coverage_percent": m.cable_coverage_percent,
            "antennas_deployed": m.antennas_deployed,
            "antennas_failed": m.antennas_failed,
            "placement_success_rate": m.success_rate(),
            "mean_placement_accuracy_m": m.mean_placement_accuracy_m(),
            "localization_error_drift_m": m.localization_error_drift_m,
            "fault_count": m.fault_count,
        }

    _SUMMARY_COLUMNS: tuple[str, ...] = (
        "total_distance_m",
        "energy_consumed_wh",
        "energy_per_antenna_wh",
        "mission_time_s",
        "cable_drag_energy_j",
        "cable_coverage_percent",
        "placement_success_rate",
        "mean_placement_accuracy_m",
        "localization_error_drift_m",
        "fault_count",
    )

    def compare_across_runs(self, results: List[RunMetrics]) -> Dict:
        if not results:
            return {col: _empty_stats() for col in self._SUMMARY_COLUMNS}
        df = self.to_dataframe(results)
        out: Dict[str, Dict[str, float]] = {}
        for col in self._SUMMARY_COLUMNS:
            series = df[col].astype(float)
            out[col] = {
                "mean": float(series.mean()),
                "std": float(series.std(ddof=0)),
                "min": float(series.min()),
                "max": float(series.max()),
                "p25": float(series.quantile(0.25)),
                "p50": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "n": int(series.size),
            }
        return out

    # ------------------------------------------------------------------
    # Failure modes
    # ------------------------------------------------------------------

    def failure_mode_analysis(self, results: List[RunMetrics]) -> Dict:
        modes: List[str] = []
        antennas_lost_by_mode: Dict[str, int] = {}
        for run in results:
            modes.extend(run.failure_modes)
            for mode in set(run.failure_modes):
                antennas_lost_by_mode[mode] = (
                    antennas_lost_by_mode.get(mode, 0) + run.antennas_failed
                )

        total = len(modes)
        if total == 0:
            return {
                "failure_modes": [],
                "cumulative_percentage": [],
                "dominant_mode": None,
                "total_failures": 0,
            }

        counter = Counter(modes)
        ranked = counter.most_common()
        cum_pct: List[float] = []
        running = 0
        rows: List[Dict[str, Any]] = []
        for mode, count in ranked:
            running += count
            pct = 100.0 * count / total
            cum_pct.append(100.0 * running / total)
            n_runs_with_mode = sum(1 for r in results if mode in r.failure_modes)
            avg_impact = (
                antennas_lost_by_mode.get(mode, 0) / n_runs_with_mode
                if n_runs_with_mode
                else 0.0
            )
            rows.append(
                {
                    "mode": mode,
                    "count": count,
                    "percentage": pct,
                    "avg_impact": avg_impact,
                }
            )

        return {
            "failure_modes": rows,
            "cumulative_percentage": cum_pct,
            "dominant_mode": ranked[0][0],
            "total_failures": total,
        }

    # ------------------------------------------------------------------
    # Path quality
    # ------------------------------------------------------------------

    def path_quality_metrics(
        self,
        planned: List[np.ndarray],
        actual: List[np.ndarray],
    ) -> Dict:
        planned_arr = _stack_path(planned)
        actual_arr = _stack_path(actual)
        if planned_arr.shape[0] < 2 or actual_arr.shape[0] < 2:
            raise ValueError("planned and actual paths must each have >= 2 points")

        errors = _point_to_polyline_distances(actual_arr, planned_arr)
        path_len_planned = _path_length(planned_arr)
        path_len_actual = _path_length(actual_arr)
        straight = float(np.linalg.norm(planned_arr[-1] - planned_arr[0]))
        directness = straight / path_len_actual if path_len_actual > 0 else 0.0

        smoothness = _path_smoothness(actual_arr)
        tracking_efficiency = float(
            np.clip(1.0 - errors.mean() / max(path_len_planned, 1e-9), 0.0, 1.0)
        )

        return {
            "mean_tracking_error_m": float(np.sqrt(np.mean(errors**2))),
            "max_tracking_error_m": float(errors.max()),
            "path_length_planned_m": path_len_planned,
            "path_length_actual_m": path_len_actual,
            "directness": float(np.clip(directness, 0.0, 1.0)),
            "smoothness": smoothness,
            "tracking_efficiency": tracking_efficiency,
        }

    # ------------------------------------------------------------------
    # Cable health
    # ------------------------------------------------------------------

    def cable_health_report(self, tension_series: np.ndarray) -> Dict:
        tension = np.asarray(tension_series, dtype=np.float64).ravel()
        if tension.size == 0:
            raise ValueError("tension_series must be non-empty")

        mean_t = float(tension.mean())
        max_t = float(tension.max())
        overstress = int(np.sum(tension > self.config.cable_overstress_threshold_n))
        cycles = _count_stress_cycles(tension, self.config.cable_overstress_threshold_n)

        fatigue_pct = (
            100.0 * cycles / max(1, self.config.cable_rated_cycles)
        )
        safety_margin = (
            self.config.cable_max_tension_n / max_t if max_t > 0 else float("inf")
        )

        if overstress >= self.config.cable_overstress_health_critical_count:
            health = "critical"
        elif overstress >= self.config.cable_overstress_health_degraded_count:
            health = "degraded"
        else:
            health = "good"

        return {
            "mean_tension_n": mean_t,
            "max_tension_n": max_t,
            "stress_cycles": int(cycles),
            "fatigue_estimate_percent": float(fatigue_pct),
            "safety_margin": float(safety_margin),
            "overstress_events": overstress,
            "health_status": health,
        }

    # ------------------------------------------------------------------
    # Interferometric array completion & quality
    # ------------------------------------------------------------------

    def array_quality_report(
        self,
        design: ArrayDesign,
        placements: Any,
        run_id: Optional[str] = None,
    ) -> ArrayQualityReport:
        """Score a deployed antenna array against its :class:`ArrayDesign`.

        Args:
            design: The surveyed array build specification.
            placements: Deployed element poses, in any of these forms:

                * a mapping ``{element_id: position_xyz}`` where the value is a
                  length-3 sequence;
                * a mapping ``{element_id: record}`` where ``record`` is a dict
                  with a position under ``position_xyz`` / ``position`` /
                  ``actual`` and, optionally, ``orientation_quat_xyzw`` (xyzw)
                  or an explicit ``tilt_deg``;
                * an iterable of such records, each additionally carrying an
                  ``element_id`` (or ``antenna_id``) key.

                Unknown element IDs (not in the design) are ignored. Designed
                elements with no record are reported as not placed.
            run_id: Optional identifier carried into the report.

        Returns:
            An :class:`ArrayQualityReport` with completion fraction, per-element
            geometric accuracy, baseline residuals, and a build-to-spec flag.
        """
        records = _normalize_array_placements(placements)

        element_residuals: List[ArrayElementResidual] = []
        placed_positions: Dict[str, np.ndarray] = {}
        position_errors: List[float] = []
        placed_count = 0
        within_tol_count = 0

        for element in design.elements:
            record = records.get(element.element_id)
            residual = _score_element(element, record)
            element_residuals.append(residual)
            if residual.placed:
                placed_count += 1
                placed_positions[element.element_id] = record["position"]
                if residual.position_error_m is not None:
                    position_errors.append(residual.position_error_m)
            if residual.within_tolerance:
                within_tol_count += 1

        n_designed = len(design.elements)
        completion = placed_count / n_designed if n_designed else 0.0
        quality = within_tol_count / n_designed if n_designed else 0.0

        mean_pos_err, max_pos_err, rms_pos_err = _error_stats(position_errors)

        baseline_residuals = _evaluate_available_baselines(design, placed_positions)
        baseline_errors = [b.error_m for b in baseline_residuals]
        n_baselines = len(design.baselines)
        baselines_within = sum(1 for b in baseline_residuals if b.within_tolerance)
        mean_bl_err, max_bl_err, _ = _error_stats(baseline_errors)

        array_complete = (
            n_designed > 0
            and placed_count == n_designed
            and within_tol_count == n_designed
            and len(baseline_residuals) == n_baselines
            and baselines_within == n_baselines
        )

        return ArrayQualityReport(
            array_id=design.array_id,
            elements_designed=n_designed,
            elements_placed=placed_count,
            elements_within_tolerance=within_tol_count,
            completion_fraction=float(completion),
            quality_fraction=float(quality),
            mean_position_error_m=mean_pos_err,
            max_position_error_m=max_pos_err,
            rms_position_error_m=rms_pos_err,
            baselines_designed=n_baselines,
            baselines_evaluated=len(baseline_residuals),
            baselines_within_tolerance=baselines_within,
            mean_baseline_error_m=mean_bl_err,
            max_baseline_error_m=max_bl_err,
            array_complete=array_complete,
            element_residuals=tuple(element_residuals),
            baseline_residuals=baseline_residuals,
            run_id=run_id,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _as_array(
    value: Any, dtype: Optional[np.dtype] = None
) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    return arr if arr.size > 0 else arr


def _path_length(points: Optional[np.ndarray]) -> float:
    if points is None or points.size == 0:
        return 0.0
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _trapezoidal_energy_wh(
    timestamps: np.ndarray, power_w: Optional[np.ndarray]
) -> float:
    if power_w is None or power_w.size == 0:
        return 0.0
    if power_w.size != timestamps.size:
        raise ValueError(
            f"power_consumed_w length {power_w.size} != timestamp length {timestamps.size}"
        )
    energy_joules = float(np.trapezoid(power_w, timestamps))
    return energy_joules / 3600.0


def _trapezoidal_integral(
    timestamps: np.ndarray, values: Optional[np.ndarray]
) -> float:
    if values is None or values.size == 0:
        return 0.0
    if values.size != timestamps.size:
        raise ValueError(
            f"values length {values.size} != timestamp length {timestamps.size}"
        )
    return float(np.trapezoid(values, timestamps))


def _final_coverage_percent(value: Any) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.clip(arr[-1], 0.0, 1.0) * 100.0)


def _localization_drift(estimated: Any, truth: Any) -> float:
    if estimated is None or truth is None:
        return 0.0
    est = np.asarray(estimated, dtype=np.float64)
    tru = np.asarray(truth, dtype=np.float64)
    if est.shape != tru.shape or est.size == 0:
        return 0.0
    diffs = est - tru
    if diffs.ndim == 1:
        return float(np.mean(np.abs(diffs)))
    return float(np.mean(np.linalg.norm(diffs, axis=-1)))


def _stack_path(path: Any) -> np.ndarray:
    arr = np.asarray(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _point_to_polyline_distances(
    points: np.ndarray, polyline: np.ndarray
) -> np.ndarray:
    """Min distance from each row of `points` to the line `polyline`."""
    starts = polyline[:-1]
    ends = polyline[1:]
    segs = ends - starts
    seg_len_sq = np.einsum("ij,ij->i", segs, segs)
    seg_len_sq = np.where(seg_len_sq > 0, seg_len_sq, 1e-18)

    out = np.empty(points.shape[0], dtype=np.float64)
    for i, p in enumerate(points):
        rel = p[None, :] - starts
        t = np.clip(np.einsum("ij,ij->i", rel, segs) / seg_len_sq, 0.0, 1.0)
        proj = starts + (t[:, None] * segs)
        d = np.linalg.norm(proj - p[None, :], axis=1)
        out[i] = d.min()
    return out


def _path_smoothness(path: np.ndarray) -> float:
    """Heuristic smoothness metric: 1.0 = perfectly smooth; 0.0 = noisy.

    Computed as ``1 / (1 + var(heading_change))`` where heading change is the
    angular difference between successive path segments.
    """
    if path.shape[0] < 3:
        return 1.0
    segs = np.diff(path, axis=0)
    norms = np.linalg.norm(segs, axis=1)
    valid = norms > 1e-9
    if valid.sum() < 2:
        return 1.0
    unit = segs[valid] / norms[valid, None]
    dots = np.clip(np.einsum("ij,ij->i", unit[:-1], unit[1:]), -1.0, 1.0)
    angle_changes = np.arccos(dots)
    return float(1.0 / (1.0 + np.var(angle_changes)))


def _count_stress_cycles(tension: np.ndarray, threshold: float) -> int:
    """Count rising threshold crossings — a coarse proxy for load cycles."""
    above = tension > threshold
    transitions = np.diff(above.astype(np.int8))
    return int(np.sum(transitions == 1))


def _empty_stats() -> Dict[str, float]:
    return {
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "p25": 0.0,
        "p50": 0.0,
        "p75": 0.0,
        "n": 0,
    }


# ---------------------------------------------------------------------------
# Array completion / quality helpers
# ---------------------------------------------------------------------------

_POSITION_KEYS: tuple[str, ...] = ("position_xyz", "position", "actual", "actual_xyz")
_ELEMENT_ID_KEYS: tuple[str, ...] = ("element_id", "antenna_id", "id")


def _normalize_array_placements(placements: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize the flexible ``placements`` input to ``{id: {position, quat, tilt_deg}}``.

    Accepts a mapping of id -> position/record or an iterable of records.
    Records missing a usable position are dropped (treated as not placed).
    """
    if placements is None:
        return {}

    items: Iterable[tuple[Optional[str], Any]]
    if isinstance(placements, Mapping):
        items = placements.items()
    else:
        items = ((None, record) for record in placements)

    out: Dict[str, Dict[str, Any]] = {}
    for key, value in items:
        record = _parse_placement_record(key, value)
        if record is not None:
            out[record["element_id"]] = record
    return out


def _parse_placement_record(
    key: Optional[str], value: Any
) -> Optional[Dict[str, Any]]:
    position: Optional[np.ndarray] = None
    quat: Optional[np.ndarray] = None
    tilt_deg: Optional[float] = None
    element_id: Optional[str] = str(key) if key is not None else None

    if isinstance(value, Mapping):
        for id_key in _ELEMENT_ID_KEYS:
            if value.get(id_key) is not None:
                element_id = str(value[id_key])
                break
        for pos_key in _POSITION_KEYS:
            if value.get(pos_key) is not None:
                position = _as_xyz(value[pos_key])
                break
        if value.get("orientation_quat_xyzw") is not None:
            quat = _as_quat(value["orientation_quat_xyzw"])
        if value.get("tilt_deg") is not None:
            tilt_deg = float(value["tilt_deg"])
    else:
        # Bare position sequence keyed by element id.
        position = _as_xyz(value)

    if element_id is None or position is None:
        return None
    return {
        "element_id": element_id,
        "position": position,
        "quat": quat,
        "tilt_deg": tilt_deg,
    }


def _as_xyz(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != 3:
        return None
    return arr


def _as_quat(value: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size != 4:
        return None
    return arr


def _score_element(
    element: ArrayElementTarget, record: Optional[Dict[str, Any]]
) -> ArrayElementResidual:
    if record is None:
        return ArrayElementResidual(
            element_id=element.element_id,
            placed=False,
            position_error_m=None,
            tilt_error_deg=None,
            within_position_tolerance=False,
            within_tilt_tolerance=True,
            within_tolerance=False,
        )

    target = np.asarray(element.position_xyz, dtype=np.float64)
    pos_err = float(np.linalg.norm(record["position"] - target))
    within_pos = pos_err <= element.placement_tolerance_m

    tilt_err = _element_tilt_error_deg(element, record)
    within_tilt = True if tilt_err is None else tilt_err <= element.tilt_tolerance_deg

    return ArrayElementResidual(
        element_id=element.element_id,
        placed=True,
        position_error_m=pos_err,
        tilt_error_deg=tilt_err,
        within_position_tolerance=within_pos,
        within_tilt_tolerance=within_tilt,
        within_tolerance=within_pos and within_tilt,
    )


def _element_tilt_error_deg(
    element: ArrayElementTarget, record: Dict[str, Any]
) -> Optional[float]:
    """Angle (deg) between the deployed and target mast up-axes.

    Prefers an explicit ``tilt_deg`` (interpreted as deviation from the target
    orientation). Otherwise derives it from the deployed quaternion relative to
    the design's target orientation. Returns ``None`` when neither is supplied.
    """
    if record.get("tilt_deg") is not None:
        return abs(float(record["tilt_deg"]))
    quat = record.get("quat")
    if quat is None:
        return None
    actual_up = _quat_up_axis(quat)
    target_up = _quat_up_axis(np.asarray(element.orientation_quat_xyzw, dtype=np.float64))
    cos_angle = float(
        np.clip(
            np.dot(actual_up, target_up)
            / (np.linalg.norm(actual_up) * np.linalg.norm(target_up) + 1e-18),
            -1.0,
            1.0,
        )
    )
    return float(np.degrees(np.arccos(cos_angle)))


def _quat_up_axis(quat: np.ndarray) -> np.ndarray:
    """Body +Z axis expressed in world frame for a unit quaternion (x, y, z, w)."""
    x, y, z, w = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    return np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)],
        dtype=np.float64,
    )


def _evaluate_available_baselines(
    design: ArrayDesign, placed_positions: Dict[str, np.ndarray]
) -> tuple[BaselineResidual, ...]:
    """Evaluate only baselines whose both endpoints have deployed positions."""
    evaluable = [
        baseline
        for baseline in design.baselines
        if baseline.element_a in placed_positions
        and baseline.element_b in placed_positions
    ]
    if not evaluable:
        return ()
    residuals: List[BaselineResidual] = []
    for baseline in evaluable:
        pos_a = placed_positions[baseline.element_a]
        pos_b = placed_positions[baseline.element_b]
        actual = float(np.linalg.norm(pos_b - pos_a))
        error = abs(actual - baseline.target_length_m)
        residuals.append(
            BaselineResidual(
                element_a=baseline.element_a,
                element_b=baseline.element_b,
                target_length_m=baseline.target_length_m,
                actual_length_m=actual,
                error_m=error,
                within_tolerance=error <= baseline.tolerance_m,
            )
        )
    return tuple(residuals)


def _error_stats(errors: Sequence[float]) -> tuple[float, float, float]:
    """Return (mean, max, rms) of an error list; (0, 0, 0) when empty."""
    if not errors:
        return 0.0, 0.0, 0.0
    arr = np.asarray(errors, dtype=np.float64)
    return (
        float(arr.mean()),
        float(arr.max()),
        float(np.sqrt(np.mean(arr**2))),
    )
