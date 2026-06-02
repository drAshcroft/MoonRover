"""Tests for the surveyed interferometric-array product model."""

from __future__ import annotations

import numpy as np
import pytest

from moon_rover.antenna import ArrayDesign, ArrayTopology


def _mission() -> dict:
    return {
        "grid": {
            "origin": [10.0, 20.0, 0.0],
            "num_rows": 2,
            "num_cols": 2,
            "spacing_m": 5.0,
            "orientation_degrees": 0.0,
        },
        "array": {
            "array_id": "test_array",
            "placement_tolerance_m": 0.1,
            "tilt_tolerance_deg": 3.0,
            "baseline_tolerance_m": 0.2,
            "operating_band": {
                "center_frequency_hz": 30.0e6,
                "bandwidth_hz": 10.0e6,
            },
            "commissioning": {
                "topology": "home_run_umbilical",
                "timing_distribution": "fiber",
                "data_transport": "fiber",
                "power_distribution": "umbilical",
            },
        },
    }


def test_grid_generates_surveyed_elements_and_all_pairwise_baselines() -> None:
    design = ArrayDesign.from_mission_config(_mission())

    assert design.array_id == "test_array"
    assert len(design.elements) == 4
    assert len(design.baselines) == 6
    assert design.elements[0].element_id == "element_r00_c00"
    assert design.elements[-1].position_xyz == pytest.approx((15.0, 25.0, 0.0))
    assert design.commissioning.topology is ArrayTopology.HOME_RUN_UMBILICAL


def test_baseline_evaluation_accepts_positions_within_tolerance() -> None:
    design = ArrayDesign.from_mission_config(_mission())
    actual = {
        element.element_id: np.asarray(element.position_xyz) + np.array([0.02, 0.0, 0.0])
        for element in design.elements
    }

    residuals = design.evaluate_baselines(actual)

    assert residuals
    assert all(residual.within_tolerance for residual in residuals)


def test_baseline_evaluation_rejects_distorted_geometry() -> None:
    design = ArrayDesign.from_mission_config(_mission())
    actual = {element.element_id: np.asarray(element.position_xyz) for element in design.elements}
    actual["element_r01_c01"] = actual["element_r01_c01"] + np.array([1.0, 0.0, 0.0])

    residuals = design.evaluate_baselines(actual)

    assert any(not residual.within_tolerance for residual in residuals)


def test_explicit_elements_and_baseline_override_grid() -> None:
    mission = _mission()
    mission["array"]["elements"] = [
        {"element_id": "a", "position_xyz": [0.0, 0.0, 0.0]},
        {"element_id": "b", "position_xyz": [3.0, 4.0, 0.0]},
    ]
    mission["array"]["baselines"] = [
        {"element_a": "a", "element_b": "b", "tolerance_m": 0.05},
    ]

    design = ArrayDesign.from_mission_config(mission)

    assert len(design.elements) == 2
    assert design.baselines[0].target_length_m == pytest.approx(5.0)


def test_unknown_topology_fails_fast() -> None:
    mission = _mission()
    mission["array"]["commissioning"]["topology"] = "coaxial_magic"

    with pytest.raises(ValueError, match="topology"):
        ArrayDesign.from_mission_config(mission)


def test_project_mission_yaml_loads_interferometric_array() -> None:
    design = ArrayDesign.from_yaml("configs/mission.yaml")
    assert design.array_id == "lunar_interferometer_001"
    assert len(design.elements) == 20
    assert len(design.baselines) == 190
