"""Unit tests for src/moon_rover/data/analysis/metrics.py.

Covers:
- compute_run_metrics: distance from positions, energy via trapezoidal power
  integration, mission time, cable drag, placement accuracy, coverage,
  localization drift, fault counts, failure modes.
- Cross-run aggregation: to_dataframe shape/columns, compare_across_runs stats.
- failure_mode_analysis Pareto: ordering, cumulative %, dominant mode.
- path_quality_metrics: tracking error, directness, smoothness on simple paths.
- cable_health_report: thresholds for good/degraded/critical, stress cycles.
- RunMetrics convenience methods: success_rate, mean_placement_accuracy_m,
  energy_per_antenna_wh.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moon_rover.antenna import ArrayDesign
from moon_rover.data.analysis.metrics import (
    ArrayQualityReport,
    MetricsConfig,
    MissionMetricsAnalyzer,
    RunMetrics,
)


# ---------------------------------------------------------------------------
# RunMetrics convenience
# ---------------------------------------------------------------------------


def test_runmetrics_success_rate():
    m = RunMetrics(antennas_deployed=8, antennas_failed=2)
    assert m.success_rate() == pytest.approx(0.8)


def test_runmetrics_success_rate_zero_when_no_attempts():
    assert RunMetrics().success_rate() == 0.0


def test_runmetrics_mean_placement_accuracy_zero_when_empty():
    assert RunMetrics().mean_placement_accuracy_m() == 0.0


def test_runmetrics_mean_placement_accuracy_average():
    m = RunMetrics(placement_accuracy_m={"a1": 0.2, "a2": 0.4})
    assert m.mean_placement_accuracy_m() == pytest.approx(0.3)


def test_runmetrics_energy_per_antenna():
    m = RunMetrics(energy_consumed_wh=100.0, antennas_deployed=4)
    assert m.energy_per_antenna_wh() == pytest.approx(25.0)
    assert RunMetrics(energy_consumed_wh=100.0).energy_per_antenna_wh() == 0.0


# ---------------------------------------------------------------------------
# compute_run_metrics
# ---------------------------------------------------------------------------


def _straight_line_log(n: int = 11) -> dict:
    """Log of a rover driving 10 m along +X at constant 1 m/s, 1 W power."""
    t = np.linspace(0.0, 10.0, n)
    pos = np.column_stack([np.linspace(0.0, 10.0, n), np.zeros(n), np.zeros(n)])
    return {
        "timestamp": t,
        "rover_position": pos,
        "power_consumed_w": np.full(n, 3600.0),  # 3600 W for 10 s = 10 Wh
        "cable_tension_n": np.full(n, 50.0),
    }


def test_compute_run_metrics_distance_and_time():
    analyzer = MissionMetricsAnalyzer()
    log = _straight_line_log()
    m = analyzer.compute_run_metrics(log)
    assert m.total_distance_m == pytest.approx(10.0)
    assert m.mission_time_s == pytest.approx(10.0)


def test_compute_run_metrics_energy_trapezoidal():
    analyzer = MissionMetricsAnalyzer()
    log = _straight_line_log()
    m = analyzer.compute_run_metrics(log)
    assert m.energy_consumed_wh == pytest.approx(10.0)


def test_compute_run_metrics_cable_drag_integral():
    analyzer = MissionMetricsAnalyzer()
    log = _straight_line_log()
    m = analyzer.compute_run_metrics(log)
    assert m.cable_drag_energy_j == pytest.approx(500.0)  # 50 N * 10 s


def test_compute_run_metrics_requires_timestamps():
    analyzer = MissionMetricsAnalyzer()
    with pytest.raises(ValueError, match="non-empty 'timestamp'"):
        analyzer.compute_run_metrics({"timestamp": np.array([])})


def test_compute_run_metrics_power_length_mismatch_raises():
    analyzer = MissionMetricsAnalyzer()
    log = _straight_line_log()
    log["power_consumed_w"] = log["power_consumed_w"][:-1]
    with pytest.raises(ValueError, match="power_consumed_w"):
        analyzer.compute_run_metrics(log)


def test_placement_metrics_count_success_and_failure():
    analyzer = MissionMetricsAnalyzer()
    log = _straight_line_log()
    log["antenna_placements"] = [
        {"antenna_id": "a1", "target": [10, 0, 0], "actual": [10.1, 0, 0], "success": True},
        {"antenna_id": "a2", "target": [20, 0, 0], "actual": [22.0, 0, 0], "success": False,
         "failure_mode": "obstacle"},
    ]
    m = analyzer.compute_run_metrics(log)
    assert m.antennas_deployed == 1
    assert m.antennas_failed == 1
    assert m.placement_accuracy_m["a1"] == pytest.approx(0.1)
    assert m.placement_accuracy_m["a2"] == pytest.approx(2.0)
    assert "obstacle" in m.failure_modes


def test_placement_success_inferred_from_tolerance():
    analyzer = MissionMetricsAnalyzer(MetricsConfig(placement_success_tolerance_m=0.5))
    log = _straight_line_log()
    log["antenna_placements"] = [
        {"antenna_id": "a1", "target": [10, 0, 0], "actual": [10.2, 0, 0]},
        {"antenna_id": "a2", "target": [10, 0, 0], "actual": [11.5, 0, 0]},
    ]
    m = analyzer.compute_run_metrics(log)
    assert m.antennas_deployed == 1
    assert m.antennas_failed == 1


def test_cable_coverage_uses_final_value():
    analyzer = MissionMetricsAnalyzer()
    log = _straight_line_log()
    log["cable_coverage_fraction"] = np.linspace(0.0, 0.75, 11)
    m = analyzer.compute_run_metrics(log)
    assert m.cable_coverage_percent == pytest.approx(75.0)


def test_localization_drift_uses_mean_norm():
    analyzer = MissionMetricsAnalyzer()
    log = _straight_line_log()
    log["estimated_position"] = log["rover_position"].copy()
    log["ground_truth_position"] = log["rover_position"].copy()
    log["estimated_position"][:, 0] += 0.5  # 0.5 m bias in X
    m = analyzer.compute_run_metrics(log)
    assert m.localization_error_drift_m == pytest.approx(0.5)


def test_faults_counted_and_modes_collected():
    analyzer = MissionMetricsAnalyzer()
    log = _straight_line_log()
    log["faults"] = [
        {"mode": "motor_overheat", "time": 3.0},
        {"mode": "motor_overheat", "time": 7.0},
        {"mode": "comm_dropout", "time": 5.0},
    ]
    m = analyzer.compute_run_metrics(log)
    assert m.fault_count == 3
    assert sorted(m.failure_modes) == ["comm_dropout", "motor_overheat", "motor_overheat"]


# ---------------------------------------------------------------------------
# Cross-run aggregation
# ---------------------------------------------------------------------------


def test_to_dataframe_shape_and_columns():
    analyzer = MissionMetricsAnalyzer()
    runs = [
        RunMetrics(total_distance_m=100.0, energy_consumed_wh=50.0, run_id="r1"),
        RunMetrics(total_distance_m=120.0, energy_consumed_wh=60.0, run_id="r2"),
    ]
    df = analyzer.to_dataframe(runs)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 2
    expected_cols = {
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
    }
    assert set(df.columns) == expected_cols
    assert df.loc[df.run_id == "r2", "total_distance_m"].iloc[0] == 120.0


def test_compare_across_runs_basic_stats():
    analyzer = MissionMetricsAnalyzer()
    runs = [
        RunMetrics(total_distance_m=10.0),
        RunMetrics(total_distance_m=20.0),
        RunMetrics(total_distance_m=30.0),
    ]
    stats = analyzer.compare_across_runs(runs)
    s = stats["total_distance_m"]
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(20.0)
    assert s["min"] == 10.0
    assert s["max"] == 30.0
    assert s["p50"] == pytest.approx(20.0)


def test_compare_across_runs_empty_returns_zeroed_stats():
    analyzer = MissionMetricsAnalyzer()
    stats = analyzer.compare_across_runs([])
    assert stats["total_distance_m"]["n"] == 0
    assert stats["total_distance_m"]["mean"] == 0.0


# ---------------------------------------------------------------------------
# Failure mode analysis
# ---------------------------------------------------------------------------


def test_failure_mode_analysis_pareto_ordering():
    analyzer = MissionMetricsAnalyzer()
    runs = [
        RunMetrics(failure_modes=["A", "A", "B"], antennas_failed=3),
        RunMetrics(failure_modes=["A", "C"], antennas_failed=2),
    ]
    result = analyzer.failure_mode_analysis(runs)
    counts = {row["mode"]: row["count"] for row in result["failure_modes"]}
    assert counts == {"A": 3, "B": 1, "C": 1}
    assert result["dominant_mode"] == "A"
    assert result["total_failures"] == 5
    assert result["cumulative_percentage"][-1] == pytest.approx(100.0)


def test_failure_mode_analysis_empty():
    analyzer = MissionMetricsAnalyzer()
    result = analyzer.failure_mode_analysis([RunMetrics(), RunMetrics()])
    assert result["failure_modes"] == []
    assert result["dominant_mode"] is None
    assert result["total_failures"] == 0


# ---------------------------------------------------------------------------
# Path quality
# ---------------------------------------------------------------------------


def test_path_quality_straight_line_is_perfect():
    analyzer = MissionMetricsAnalyzer()
    planned = [np.array([0, 0, 0]), np.array([10, 0, 0])]
    actual = [np.array([i, 0, 0]) for i in np.linspace(0, 10, 11)]
    q = analyzer.path_quality_metrics(planned, actual)
    assert q["mean_tracking_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert q["max_tracking_error_m"] == pytest.approx(0.0, abs=1e-9)
    assert q["directness"] == pytest.approx(1.0, abs=1e-9)
    assert q["smoothness"] == pytest.approx(1.0)


def test_path_quality_lateral_offset_increases_error():
    analyzer = MissionMetricsAnalyzer()
    planned = [np.array([0, 0, 0]), np.array([10, 0, 0])]
    actual = [np.array([i, 0.5, 0]) for i in np.linspace(0, 10, 11)]
    q = analyzer.path_quality_metrics(planned, actual)
    assert q["mean_tracking_error_m"] == pytest.approx(0.5)
    assert q["max_tracking_error_m"] == pytest.approx(0.5)


def test_path_quality_requires_two_points():
    analyzer = MissionMetricsAnalyzer()
    with pytest.raises(ValueError, match=">= 2"):
        analyzer.path_quality_metrics([np.array([0, 0, 0])], [np.array([0, 0, 0])])


# ---------------------------------------------------------------------------
# Cable health
# ---------------------------------------------------------------------------


def test_cable_health_good_under_threshold():
    analyzer = MissionMetricsAnalyzer()
    tension = np.full(100, 200.0)
    report = analyzer.cable_health_report(tension)
    assert report["health_status"] == "good"
    assert report["mean_tension_n"] == pytest.approx(200.0)
    assert report["max_tension_n"] == pytest.approx(200.0)
    assert report["overstress_events"] == 0


def test_cable_health_degraded_threshold():
    analyzer = MissionMetricsAnalyzer(
        MetricsConfig(
            cable_overstress_threshold_n=500.0,
            cable_overstress_health_degraded_count=3,
            cable_overstress_health_critical_count=20,
        )
    )
    # 10 samples above threshold (between degraded and critical)
    tension = np.array([100.0, 600.0, 100.0] * 5 + [600.0] * 5)
    report = analyzer.cable_health_report(tension)
    assert report["health_status"] == "degraded"


def test_cable_health_critical_threshold():
    analyzer = MissionMetricsAnalyzer(
        MetricsConfig(
            cable_overstress_threshold_n=500.0,
            cable_overstress_health_critical_count=5,
        )
    )
    tension = np.array([100.0] * 5 + [900.0] * 10)
    report = analyzer.cable_health_report(tension)
    assert report["health_status"] == "critical"


def test_cable_stress_cycles_counted():
    analyzer = MissionMetricsAnalyzer(
        MetricsConfig(cable_overstress_threshold_n=500.0)
    )
    # rising threshold crossings: low/high/low/high/low/high → 3 cycles
    tension = np.array(
        [100, 600, 100, 600, 100, 600, 100], dtype=np.float64
    )
    report = analyzer.cable_health_report(tension)
    assert report["stress_cycles"] == 3


def test_cable_health_empty_raises():
    analyzer = MissionMetricsAnalyzer()
    with pytest.raises(ValueError, match="non-empty"):
        analyzer.cable_health_report(np.array([]))


# ---------------------------------------------------------------------------
# Array completion / quality
# ---------------------------------------------------------------------------


def _array_design(spacing: float = 5.0) -> ArrayDesign:
    """2x2 surveyed grid, 0.1 m placement / 3 deg tilt / 0.2 m baseline tol."""
    return ArrayDesign.from_mission_config(
        {
            "grid": {
                "origin": [0.0, 0.0, 0.0],
                "num_rows": 2,
                "num_cols": 2,
                "spacing_m": spacing,
                "orientation_degrees": 0.0,
            },
            "array": {
                "array_id": "qa_array",
                "placement_tolerance_m": 0.1,
                "tilt_tolerance_deg": 3.0,
                "baseline_tolerance_m": 0.2,
            },
        }
    )


def _exact_placements(design: ArrayDesign) -> dict:
    return {e.element_id: np.asarray(e.position_xyz) for e in design.elements}


def test_array_quality_perfect_build_is_complete():
    analyzer = MissionMetricsAnalyzer()
    design = _array_design()
    report = analyzer.array_quality_report(design, _exact_placements(design), run_id="r1")

    assert isinstance(report, ArrayQualityReport)
    assert report.array_id == "qa_array"
    assert report.run_id == "r1"
    assert report.elements_designed == 4
    assert report.elements_placed == 4
    assert report.elements_within_tolerance == 4
    assert report.completion_fraction == pytest.approx(1.0)
    assert report.quality_fraction == pytest.approx(1.0)
    assert report.mean_position_error_m == pytest.approx(0.0)
    assert report.baselines_designed == 6
    assert report.baselines_evaluated == 6
    assert report.baselines_within_tolerance == 6
    assert report.array_complete is True


def test_array_quality_partial_completion_fraction():
    analyzer = MissionMetricsAnalyzer()
    design = _array_design()
    placed = _exact_placements(design)
    # Only deploy 2 of the 4 elements.
    keep = [e.element_id for e in design.elements[:2]]
    placements = {eid: placed[eid] for eid in keep}

    report = analyzer.array_quality_report(design, placements)
    assert report.elements_placed == 2
    assert report.completion_fraction == pytest.approx(0.5)
    assert report.quality_fraction == pytest.approx(0.5)
    # Only the baseline between the two placed elements can be evaluated.
    assert report.baselines_evaluated == 1
    assert report.array_complete is False
    # Unplaced elements are reported but flagged not placed.
    not_placed = [r for r in report.element_residuals if not r.placed]
    assert len(not_placed) == 2
    assert all(r.position_error_m is None for r in not_placed)


def test_array_quality_out_of_tolerance_position_lowers_quality():
    analyzer = MissionMetricsAnalyzer()
    design = _array_design()
    placements = _exact_placements(design)
    # Push one element 0.5 m off target (tolerance is 0.1 m).
    bad_id = design.elements[0].element_id
    placements[bad_id] = placements[bad_id] + np.array([0.5, 0.0, 0.0])

    report = analyzer.array_quality_report(design, placements)
    assert report.elements_placed == 4
    assert report.elements_within_tolerance == 3
    assert report.completion_fraction == pytest.approx(1.0)
    assert report.quality_fraction == pytest.approx(0.75)
    assert report.max_position_error_m == pytest.approx(0.5)
    assert report.array_complete is False


def test_array_quality_accepts_record_dicts_with_tilt():
    analyzer = MissionMetricsAnalyzer()
    design = _array_design()
    records = []
    for i, e in enumerate(design.elements):
        records.append(
            {
                "element_id": e.element_id,
                "position_xyz": list(e.position_xyz),
                "tilt_deg": 1.0 if i == 0 else 10.0,  # element 0 ok, rest over 3 deg
            }
        )
    report = analyzer.array_quality_report(design, records)
    assert report.elements_placed == 4
    # Only the first element is within the 3 deg tilt tolerance.
    assert report.elements_within_tolerance == 1
    first = report.element_residuals[0]
    assert first.tilt_error_deg == pytest.approx(1.0)
    assert first.within_tilt_tolerance is True


def test_array_quality_tilt_from_quaternion():
    analyzer = MissionMetricsAnalyzer()
    design = _array_design()
    # 10 deg tilt about X: quat = (sin(5deg), 0, 0, cos(5deg)) -> up-axis 10 deg off.
    half = np.radians(10.0) / 2.0
    tilted_quat = [np.sin(half), 0.0, 0.0, np.cos(half)]
    records = {
        e.element_id: {
            "position_xyz": list(e.position_xyz),
            "orientation_quat_xyzw": tilted_quat,
        }
        for e in design.elements
    }
    report = analyzer.array_quality_report(design, records)
    assert all(
        r.tilt_error_deg == pytest.approx(10.0, abs=1e-6)
        for r in report.element_residuals
    )
    # 10 deg exceeds the 3 deg tolerance: none within tolerance.
    assert report.elements_within_tolerance == 0


def test_array_quality_baseline_out_of_tolerance():
    analyzer = MissionMetricsAnalyzer()
    design = _array_design()
    placements = _exact_placements(design)
    # Push one element 0.3 m off: exceeds both the 0.1 m placement tolerance and
    # (for baselines aligned with the shift) the 0.2 m baseline tolerance.
    bad_id = design.elements[0].element_id
    placements[bad_id] = placements[bad_id] + np.array([0.3, 0.0, 0.0])

    report = analyzer.array_quality_report(design, placements)
    assert report.elements_within_tolerance == 3  # the shifted element fails
    assert report.baselines_evaluated == 6
    # At least one baseline touching the moved element now breaks tolerance.
    assert report.baselines_within_tolerance < 6
    assert report.array_complete is False


def test_array_quality_empty_placements():
    analyzer = MissionMetricsAnalyzer()
    design = _array_design()
    report = analyzer.array_quality_report(design, {})
    assert report.elements_placed == 0
    assert report.completion_fraction == 0.0
    assert report.quality_fraction == 0.0
    assert report.baselines_evaluated == 0
    assert report.array_complete is False
    assert report.mean_position_error_m == 0.0


def test_array_quality_ignores_unknown_element_ids():
    analyzer = MissionMetricsAnalyzer()
    design = _array_design()
    placements = _exact_placements(design)
    placements["not_in_design"] = np.array([99.0, 99.0, 0.0])
    report = analyzer.array_quality_report(design, placements)
    assert report.elements_placed == 4
    assert report.array_complete is True


def test_array_quality_to_summary_keys():
    analyzer = MissionMetricsAnalyzer()
    design = _array_design()
    report = analyzer.array_quality_report(design, _exact_placements(design))
    summary = report.to_summary()
    assert summary["completion_fraction"] == pytest.approx(1.0)
    assert summary["array_complete"] is True
    assert summary["elements_designed"] == 4
