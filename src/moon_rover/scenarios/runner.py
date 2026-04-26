"""
System 14: Scenario Runner and Experiment Framework

Experiment orchestration and statistical analysis framework for running
parametric studies, Monte Carlo simulations, and convergence analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


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
    fixed_params: dict[str, any]
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
        """
        Load experiment configuration from YAML file.

        YAML format example:
        ```yaml
        name: "antenna_placement_study"
        variable_params:
          rover_type: ["diff_drive", "skid_steer"]
          terrain_roughness: [0.01, 0.05, 0.10]
        fixed_params:
          grid_size: 5
          antenna_mass_kg: 2.5
        num_seeds: 30
        success_metrics:
          - mission_time_s
          - cable_drag_energy_j
          - antenna_tilt_accuracy_deg
        convergence_threshold: 0.05
        ```

        Args:
            experiment_yaml: Path to YAML experiment file.

        Returns:
            Parsed ExperimentConfig object.

        Raises:
            FileNotFoundError: If YAML file not found.
            ValueError: If YAML format invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def run_single(
        self,
        seed: int,
        config: dict,
    ) -> RunResult:
        """
        Execute single simulation run with given parameters.

        Runs simulation once with specified random seed and configuration
        parameters. Collects all metrics and fault events.

        Args:
            seed: Random seed for reproducibility.
            config: Configuration dict with parameter values.

        Returns:
            RunResult with metrics, faults, and success status.
        """
        raise NotImplementedError

    @abstractmethod
    def run_monte_carlo(
        self,
        experiment: ExperimentConfig,
        num_workers: int = 1,
    ) -> list[RunResult]:
        """
        Execute full Monte Carlo experiment with parallel workers.

        Runs all parameter combinations (cartesian product of variable_params)
        with num_seeds trials each. Uses multiprocessing to parallelize
        across num_workers processes.

        Args:
            experiment: Experiment configuration.
            num_workers: Number of parallel worker processes. Default: 1 (serial).

        Returns:
            List of all RunResult objects from all trials.
        """
        raise NotImplementedError

    @abstractmethod
    def check_convergence(
        self,
        results: list[RunResult],
        metric: str,
    ) -> bool:
        """
        Check if results have converged for specified metric.

        Uses running mean and variance estimation to detect convergence.
        Compares relative change in estimated mean over successive batches
        against convergence_threshold.

        Args:
            results: List of RunResult objects from runs.
            metric: Metric name to check (must be in success_metrics).

        Returns:
            True if converged, False if more runs needed.

        Raises:
            ValueError: If too few runs (< min_runs_significance) or invalid metric.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_report(
        self,
        results: list[RunResult],
    ) -> dict:
        """
        Generate statistical summary report from results.

        Computes:
        - Mean, std, min, max, percentiles (25, 50, 75, 95, 99) for each metric
        - Success rate and failure mode frequencies
        - Pairwise metric correlations
        - Failure case analysis (which parameter combos had highest fault rates)

        Returns:
            Dict with keys:
            - 'metric_summaries': dict mapping metric_name -> {mean, std, min, max, p25, p50, p75, p95, p99}
            - 'success_rate': float [0, 1]
            - 'failure_modes': dict mapping FaultType -> count
            - 'failure_by_params': dict mapping parameter combo -> failure count
            - 'correlations': dict mapping (metric1, metric2) -> correlation coefficient
        """
        raise NotImplementedError
