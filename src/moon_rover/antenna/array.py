"""Interferometric antenna-array product model.

The rover mission builds a surveyed interferometric array, not merely a set of
independent beacons. This module defines the target element geometry, pairwise
baselines, RF band metadata, and the commissioning topology required to decide
whether the deployed product is usable.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


class ArrayTopology(str, Enum):
    """Supported power, timing, and data distribution layouts."""

    HOME_RUN_UMBILICAL = "home_run_umbilical"
    TRUNK_WITH_TAPS = "trunk_with_taps"
    WIRELESS_RELAY = "wireless_relay"


@dataclass(frozen=True)
class OperatingBand:
    """RF operating-band metadata for the interferometric instrument."""

    center_frequency_hz: float
    bandwidth_hz: float
    polarization: str = "dual_linear"


@dataclass(frozen=True)
class CommissioningTopology:
    """Physical and logical continuity required before an element is ACTIVE."""

    topology: ArrayTopology
    timing_distribution: str
    data_transport: str
    power_distribution: str
    require_connector_lock: bool = True
    require_timing_lock: bool = True
    require_data_link: bool = True
    require_power_good: bool = True


@dataclass(frozen=True)
class ArrayElementTarget:
    """Surveyed target pose and tolerance for one antenna element."""

    element_id: str
    position_xyz: tuple[float, float, float]
    orientation_quat_xyzw: tuple[float, float, float, float]
    placement_tolerance_m: float
    tilt_tolerance_deg: float


@dataclass(frozen=True)
class InterferometricBaseline:
    """Required separation between a pair of elements."""

    element_a: str
    element_b: str
    target_length_m: float
    tolerance_m: float


@dataclass(frozen=True)
class BaselineResidual:
    """Measured error for one deployed pairwise baseline."""

    element_a: str
    element_b: str
    target_length_m: float
    actual_length_m: float
    error_m: float
    within_tolerance: bool


@dataclass(frozen=True)
class ArrayDesign:
    """Complete interferometric-array build specification."""

    array_id: str
    reference_origin_xyz: tuple[float, float, float]
    elements: tuple[ArrayElementTarget, ...]
    baselines: tuple[InterferometricBaseline, ...]
    operating_band: OperatingBand
    commissioning: CommissioningTopology

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ArrayDesign":
        """Load an array design from a mission YAML file."""
        with open(path, encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
        return cls.from_mission_config(document)

    @classmethod
    def from_mission_config(cls, document: Mapping[str, Any]) -> "ArrayDesign":
        """Build an interferometric array design from parsed mission config."""
        array_cfg = dict(document.get("array") or {})
        grid_cfg = dict(document.get("grid") or {})
        if not array_cfg:
            raise ValueError("mission config must define an 'array' section")

        array_id = str(array_cfg.get("array_id") or "interferometric_array")
        origin = _tuple_floats(
            array_cfg.get("reference_origin_xyz", grid_cfg.get("origin", [0.0, 0.0, 0.0])),
            3,
            "array.reference_origin_xyz",
        )
        placement_tol = _positive_float(
            array_cfg.get("placement_tolerance_m", 0.25),
            "array.placement_tolerance_m",
        )
        tilt_tol = _positive_float(
            array_cfg.get("tilt_tolerance_deg", 5.0),
            "array.tilt_tolerance_deg",
        )
        baseline_tol = _positive_float(
            array_cfg.get("baseline_tolerance_m", placement_tol * 2.0),
            "array.baseline_tolerance_m",
        )

        raw_elements = list(array_cfg.get("elements") or [])
        elements = (
            _elements_from_config(raw_elements, placement_tol, tilt_tol)
            if raw_elements
            else _elements_from_grid(grid_cfg, origin, placement_tol, tilt_tol)
        )
        _validate_unique_element_ids(elements)

        band_cfg = dict(array_cfg.get("operating_band") or {})
        band = OperatingBand(
            center_frequency_hz=_positive_float(
                band_cfg.get("center_frequency_hz", 30.0e6),
                "array.operating_band.center_frequency_hz",
            ),
            bandwidth_hz=_positive_float(
                band_cfg.get("bandwidth_hz", 10.0e6),
                "array.operating_band.bandwidth_hz",
            ),
            polarization=str(band_cfg.get("polarization", "dual_linear")),
        )

        topology_cfg = dict(array_cfg.get("commissioning") or {})
        try:
            topology = ArrayTopology(
                topology_cfg.get("topology", ArrayTopology.HOME_RUN_UMBILICAL.value)
            )
        except ValueError as exc:
            allowed = ", ".join(item.value for item in ArrayTopology)
            raise ValueError(f"array.commissioning.topology must be one of: {allowed}") from exc
        commissioning = CommissioningTopology(
            topology=topology,
            timing_distribution=str(topology_cfg.get("timing_distribution", "fiber")),
            data_transport=str(topology_cfg.get("data_transport", "fiber")),
            power_distribution=str(topology_cfg.get("power_distribution", "umbilical")),
            require_connector_lock=bool(topology_cfg.get("require_connector_lock", True)),
            require_timing_lock=bool(topology_cfg.get("require_timing_lock", True)),
            require_data_link=bool(topology_cfg.get("require_data_link", True)),
            require_power_good=bool(topology_cfg.get("require_power_good", True)),
        )

        baselines = _baselines_from_config(
            list(array_cfg.get("baselines") or []),
            elements,
            baseline_tol,
        )
        return cls(
            array_id=array_id,
            reference_origin_xyz=origin,
            elements=elements,
            baselines=baselines,
            operating_band=band,
            commissioning=commissioning,
        )

    def evaluate_baselines(
        self,
        actual_positions: Mapping[str, tuple[float, float, float] | np.ndarray],
    ) -> tuple[BaselineResidual, ...]:
        """Measure deployed pairwise baselines against the surveyed design."""
        residuals: list[BaselineResidual] = []
        for baseline in self.baselines:
            if baseline.element_a not in actual_positions:
                raise KeyError(f"Missing deployed position for {baseline.element_a!r}")
            if baseline.element_b not in actual_positions:
                raise KeyError(f"Missing deployed position for {baseline.element_b!r}")
            pos_a = np.asarray(actual_positions[baseline.element_a], dtype=np.float64).reshape(3)
            pos_b = np.asarray(actual_positions[baseline.element_b], dtype=np.float64).reshape(3)
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


def _elements_from_config(
    raw_elements: list[Mapping[str, Any]],
    default_placement_tol: float,
    default_tilt_tol: float,
) -> tuple[ArrayElementTarget, ...]:
    elements: list[ArrayElementTarget] = []
    for index, item in enumerate(raw_elements):
        prefix = f"array.elements[{index}]"
        elements.append(
            ArrayElementTarget(
                element_id=str(item.get("element_id") or f"element_{index:03d}"),
                position_xyz=_tuple_floats(item["position_xyz"], 3, f"{prefix}.position_xyz"),
                orientation_quat_xyzw=_tuple_floats(
                    item.get("orientation_quat_xyzw", [0.0, 0.0, 0.0, 1.0]),
                    4,
                    f"{prefix}.orientation_quat_xyzw",
                ),
                placement_tolerance_m=_positive_float(
                    item.get("placement_tolerance_m", default_placement_tol),
                    f"{prefix}.placement_tolerance_m",
                ),
                tilt_tolerance_deg=_positive_float(
                    item.get("tilt_tolerance_deg", default_tilt_tol),
                    f"{prefix}.tilt_tolerance_deg",
                ),
            )
        )
    if not elements:
        raise ValueError("array.elements must contain at least one element")
    return tuple(elements)


def _elements_from_grid(
    grid_cfg: Mapping[str, Any],
    origin: tuple[float, float, float],
    placement_tol: float,
    tilt_tol: float,
) -> tuple[ArrayElementTarget, ...]:
    rows = int(grid_cfg.get("num_rows", 0))
    cols = int(grid_cfg.get("num_cols", 0))
    spacing = _positive_float(grid_cfg.get("spacing_m", 0.0), "grid.spacing_m")
    if rows <= 0 or cols <= 0:
        raise ValueError("grid.num_rows and grid.num_cols must be positive")
    yaw = math.radians(float(grid_cfg.get("orientation_degrees", 0.0)))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    quat = (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))
    elements: list[ArrayElementTarget] = []
    for row in range(rows):
        for col in range(cols):
            local_x, local_y = col * spacing, row * spacing
            x = origin[0] + (cos_yaw * local_x) - (sin_yaw * local_y)
            y = origin[1] + (sin_yaw * local_x) + (cos_yaw * local_y)
            elements.append(
                ArrayElementTarget(
                    element_id=f"element_r{row:02d}_c{col:02d}",
                    position_xyz=(x, y, origin[2]),
                    orientation_quat_xyzw=quat,
                    placement_tolerance_m=placement_tol,
                    tilt_tolerance_deg=tilt_tol,
                )
            )
    return tuple(elements)


def _baselines_from_config(
    raw_baselines: list[Mapping[str, Any]],
    elements: tuple[ArrayElementTarget, ...],
    default_tolerance: float,
) -> tuple[InterferometricBaseline, ...]:
    positions = {element.element_id: np.asarray(element.position_xyz) for element in elements}
    if raw_baselines:
        baselines: list[InterferometricBaseline] = []
        for index, item in enumerate(raw_baselines):
            prefix = f"array.baselines[{index}]"
            element_a, element_b = str(item["element_a"]), str(item["element_b"])
            if element_a not in positions or element_b not in positions:
                raise ValueError(f"{prefix} references an unknown array element")
            if element_a == element_b:
                raise ValueError(f"{prefix} must reference two different elements")
            target = float(item.get("target_length_m", np.linalg.norm(positions[element_b] - positions[element_a])))
            baselines.append(
                InterferometricBaseline(
                    element_a=element_a,
                    element_b=element_b,
                    target_length_m=_positive_float(target, f"{prefix}.target_length_m"),
                    tolerance_m=_positive_float(
                        item.get("tolerance_m", default_tolerance),
                        f"{prefix}.tolerance_m",
                    ),
                )
            )
        return tuple(baselines)

    return tuple(
        InterferometricBaseline(
            element_a=left.element_id,
            element_b=right.element_id,
            target_length_m=float(
                np.linalg.norm(np.asarray(right.position_xyz) - np.asarray(left.position_xyz))
            ),
            tolerance_m=default_tolerance,
        )
        for left, right in itertools.combinations(elements, 2)
    )


def _tuple_floats(value: Any, length: int, label: str) -> tuple[float, ...]:
    try:
        items = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric sequence of length {length}") from exc
    if len(items) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    return items


def _positive_float(value: Any, label: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{label} must be > 0")
    return number


def _validate_unique_element_ids(elements: tuple[ArrayElementTarget, ...]) -> None:
    ids = [element.element_id for element in elements]
    if len(ids) != len(set(ids)):
        raise ValueError("array element IDs must be unique")
