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

Typical Usage:
    metrics = analyzer.compute_run_metrics(log_data)
    stats = analyzer.compare_across_runs([m1, m2, m3])  # Compare multiple runs
    pareto = analyzer.failure_mode_analysis(results)     # Find critical failures
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


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

    def mean_placement_accuracy_m(self) -> float:
        """Compute mean placement accuracy across all antennas.

        Returns:
            float: Mean accuracy in meters. Returns 0.0 if no antennas deployed.

        Example:
            mean_acc = metrics.mean_placement_accuracy_m()
            print(f"Mean accuracy: {mean_acc:.3f} m")
        """
        if not self.placement_accuracy_m:
            return 0.0
        return float(np.mean(list(self.placement_accuracy_m.values())))

    def success_rate(self) -> float:
        """Compute antenna deployment success rate.

        Returns:
            float: Fraction of attempted deployments that succeeded [0.0, 1.0].
                  Returns 0.0 if no deployments attempted.

        Example:
            if metrics.success_rate() >= 0.95:
                print("Mission highly successful")
        """
        total = self.antennas_deployed + self.antennas_failed
        if total == 0:
            return 0.0
        return self.antennas_deployed / total


class AnalysisToolkit(ABC):
    """Abstract interface for Moon Rover mission performance analysis.

    Provides high-level methods for:
    - Computing aggregate metrics from raw telemetry logs
    - Comparing performance across multiple runs (Monte Carlo)
    - Analyzing failure modes and identifying critical paths
    - Assessing trajectory quality (planned vs actual)
    - Monitoring cable health and integrity

    Abstract Methods:
        compute_run_metrics: Aggregate raw log data into RunMetrics
        compare_across_runs: Statistical analysis of multiple runs
        failure_mode_analysis: Pareto analysis of failure modes
        path_quality_metrics: Evaluate trajectory following
        cable_health_report: Cable tension and degradation analysis
    """

    @abstractmethod
    def compute_run_metrics(self, log_data: Dict) -> RunMetrics:
        """Compute aggregate metrics from a single mission run.

        Processes raw telemetry log data and computes high-level performance
        metrics including distance, energy, time, accuracy, and faults.

        Args:
            log_data (Dict): Telemetry log from a single run.
                            Expected keys:
                              - "timestamp": list of timestamps (float)
                              - "rover_position": list of [x, y, z] positions
                              - "power_consumed": list of instantaneous power (watts)
                              - "cable_tension": list of tension values (N)
                              - "antenna_placements": list of deployment events
                              - "faults": list of fault records
                              - "odometry": wheel-based distance estimates

        Returns:
            RunMetrics: Aggregated performance metrics from the run.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            log = {
                "timestamp": [0.0, 0.1, 0.2, ...],
                "rover_position": [[0, 0, 0], [1, 0, 0], [2, 0, 0], ...],
                "power_consumed": [10, 15, 12, ...],
                ...
            }
            metrics = analyzer.compute_run_metrics(log)
            print(f"Distance: {metrics.total_distance_m:.1f} m")
        """
        raise NotImplementedError("compute_run_metrics implementation pending")

    @abstractmethod
    def compare_across_runs(self, results: List[RunMetrics]) -> Dict:
        """Perform statistical analysis across multiple mission runs.

        Computes mean, std, min, max, and percentile statistics for each
        metric. Useful for Monte Carlo experiments and design space exploration.

        Args:
            results (List[RunMetrics]): List of RunMetrics from multiple runs.

        Returns:
            Dict: Statistical summary with structure:
                {
                    "total_distance_m": {
                        "mean": float,
                        "std": float,
                        "min": float,
                        "max": float,
                        "p25": float,
                        "p50": float,
                        "p75": float
                    },
                    "energy_consumed_wh": {...},
                    "mission_time_s": {...},
                    "success_rate": {...},
                    "mean_accuracy_m": {...},
                    ...
                }

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            stats = analyzer.compare_across_runs([metrics1, metrics2, metrics3])
            print(f"Mean distance: {stats['total_distance_m']['mean']:.1f} +/- "
                  f"{stats['total_distance_m']['std']:.1f} m")
        """
        raise NotImplementedError("compare_across_runs implementation pending")

    @abstractmethod
    def failure_mode_analysis(self, results: List[RunMetrics]) -> Dict:
        """Analyze failure modes and compute Pareto chart data.

        Identifies dominant failure modes (antenna deployment failures,
        energy constraints, cable breakage, etc.) and ranks them by
        frequency and impact.

        Args:
            results (List[RunMetrics]): List of RunMetrics from runs with faults.

        Returns:
            Dict: Failure analysis with structure:
                {
                    "failure_modes": [
                        {
                            "mode": str (e.g., "antenna_placement_failed"),
                            "count": int (occurrences),
                            "percentage": float (0-100),
                            "avg_impact": float (e.g., antennas lost)
                        },
                        ...
                    ],
                    "cumulative_percentage": list (for Pareto curve),
                    "dominant_mode": str (most frequent failure),
                    "total_failures": int
                }

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            pareto = analyzer.failure_mode_analysis(results_with_faults)
            print(f"Dominant failure: {pareto['dominant_mode']}")
            for mode in pareto['failure_modes']:
                print(f"  {mode['mode']}: {mode['percentage']:.1f}%")
        """
        raise NotImplementedError("failure_mode_analysis implementation pending")

    @abstractmethod
    def path_quality_metrics(
        self,
        planned: List[np.ndarray],
        actual: List[np.ndarray]
    ) -> Dict:
        """Evaluate path execution quality (planned vs actual trajectory).

        Compares the planner's reference trajectory to the rover's actual
        motion. Metrics include tracking error, path smoothness, and
        directness of execution.

        Args:
            planned (List[np.ndarray]): Planned waypoints, each [x, y, z] in meters.
            actual (List[np.ndarray]): Actual trajectory points, each [x, y, z] in meters.
                                       May be at higher sampling rate than planned.

        Returns:
            Dict: Path quality metrics:
                {
                    "mean_tracking_error_m": float (RMS error from planned path),
                    "max_tracking_error_m": float (peak deviation),
                    "path_length_planned_m": float,
                    "path_length_actual_m": float,
                    "directness": float (0-1, planned_length/actual_length),
                    "smoothness": float (0-1, inverse of curvature variation),
                    "tracking_efficiency": float (0-1)
                }

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            quality = analyzer.path_quality_metrics(waypoints, odometry_log)
            print(f"Tracking error: {quality['mean_tracking_error_m']:.3f} m")
            print(f"Path efficiency: {quality['directness']:.2%}")
        """
        raise NotImplementedError("path_quality_metrics implementation pending")

    @abstractmethod
    def cable_health_report(self, tension_series: np.ndarray) -> Dict:
        """Analyze cable tension history and estimate cable health/degradation.

        Computes cable stress statistics, identifies peak stress events,
        and estimates remaining cable life based on fatigue models.

        Args:
            tension_series (np.ndarray): Time series of cable tension in Newtons.
                                        Shape: (n_samples,)

        Returns:
            Dict: Cable health analysis:
                {
                    "mean_tension_n": float,
                    "max_tension_n": float,
                    "stress_cycles": int (detected tension/release cycles),
                    "fatigue_estimate_percent": float (% of cable life consumed),
                    "safety_margin": float (max_allowable / max_observed),
                    "overstress_events": int (times tension exceeded threshold),
                    "health_status": str ("good" | "degraded" | "critical")
                }

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            health = analyzer.cable_health_report(tension_log)
            if health['health_status'] == 'critical':
                print("WARNING: Cable replacement needed")
            print(f"Cable life consumed: {health['fatigue_estimate_percent']:.1f}%")
        """
        raise NotImplementedError("cable_health_report implementation pending")
