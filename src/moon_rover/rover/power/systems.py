"""System 5: Power and Thermal Systems.

This module defines the power management and energy distribution subsystem,
including solar generation, battery storage, thermal management, and load budgeting
across all rover subsystems. Handles state-of-charge tracking, thermal derating,
and power distribution constraints.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SolarArrayConfig:
    """Configuration for solar panel array.

    Attributes:
        num_panels: Total number of solar panels (typically 4 for rover-mounted array).
        area_per_panel_m2: Area of each panel in square meters.
        efficiency: Solar-to-electrical conversion efficiency (fraction, e.g., 0.22 = 22%).
        dust_factor: Dust coverage factor (1.0 = clean, <1.0 = dust-covered).
                     Reduces effective efficiency due to lunar dust accumulation.
    """

    num_panels: int
    area_per_panel_m2: float
    efficiency: float
    dust_factor: float


@dataclass
class BatteryConfig:
    """Configuration for energy storage battery pack.

    Attributes:
        capacity_wh: Total battery capacity in Watt-hours.
        soc_range: Tuple (min_soc, max_soc) defining safe state-of-charge operating window.
                   Typical: (0.20, 0.95) to extend battery life (avoid deep discharge/overcharge).
        max_discharge_c: Maximum discharge rate in C-rates (multiples of rated capacity).
                         E.g., 2.0 C means can discharge full capacity in 0.5 hours.
        temp_derating_pct_per_deg: Thermal derating factor in %/°C.
                                   Battery capacity decreases with cold temperature.
                                   E.g., 0.02 = 2% reduction per degree below nominal (20°C).
    """

    capacity_wh: float
    soc_range: tuple[float, float]
    max_discharge_c: float
    temp_derating_pct_per_deg: float


@dataclass
class PowerBudget:
    """Estimated power consumption across all subsystems.

    Each subsystem entry is a dict with "idle", "active", "peak" power levels in Watts.
    Allows power management to predict mission duration and detect overconsumption.

    Attributes:
        drive_motors_w: Drive system power consumption.
        manipulator_w: Arm and gripper power consumption.
        lidar_w: LiDAR scanner power consumption.
        cameras_w: Camera systems (stereo, nav cam) power consumption.
        imu_w: Inertial measurement unit power consumption.
        compute_w: Main compute module power consumption.
        comms_w: Communication system power consumption.
        heating_w: Thermal heater power consumption (active in cold periods).
    """

    drive_motors_w: dict[str, float]  # "idle", "active", "peak"
    manipulator_w: dict[str, float]
    lidar_w: dict[str, float]
    cameras_w: dict[str, float]
    imu_w: dict[str, float]
    compute_w: dict[str, float]
    comms_w: dict[str, float]
    heating_w: dict[str, float]


@dataclass
class PowerState:
    """Real-time power system status snapshot.

    Attributes:
        battery_soc: State of charge (fraction, 0.0-1.0). 0.0 = empty, 1.0 = full.
        solar_output_w: Current solar panel power generation in Watts.
        total_draw_w: Total power consumed by all active subsystems in Watts.
        net_power_w: Net power flow (solar - draw). Positive = charging, negative = discharging.
        battery_temp_c: Battery temperature in Celsius. Used for thermal derating.
    """

    battery_soc: float
    solar_output_w: float
    total_draw_w: float
    net_power_w: float
    battery_temp_c: float


class PowerSystem(ABC):
    """Abstract base class for rover power management system.

    Manages energy generation, storage, and distribution. Primary responsibilities:
    - Track solar panel output as function of sun position and dust coverage
    - Monitor battery state-of-charge and thermal state
    - Enforce power constraints and distribution policies
    - Predict mission duration and energy reserve margins
    - Support thermal management (heating in cold periods)
    """

    @abstractmethod
    def initialize(
        self,
        solar_config: SolarArrayConfig,
        battery_config: BatteryConfig,
        budget: PowerBudget,
    ) -> None:
        """Initialize power system with hardware and operational parameters.

        Args:
            solar_config: Solar array configuration.
            battery_config: Battery pack configuration.
            budget: Expected power consumption budget for mission planning.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        dt: float,
        sun_elevation: float,
        subsystem_states: dict[str, Any],
    ) -> PowerState:
        """Update power system state for one simulation step.

        Computes solar output based on sun elevation and dust, integrates
        battery charge/discharge, updates thermal state, and applies
        derating factors.

        Args:
            dt: Simulation time step in seconds.
            sun_elevation: Sun elevation angle above horizon in degrees.
                           Affects solar panel output (cos law).
            subsystem_states: Dictionary mapping subsystem names to their current
                              operating state (e.g., "idle", "active", "peak").

        Returns:
            Current PowerState snapshot after integration.
        """
        raise NotImplementedError

    @abstractmethod
    def get_battery_soc(self) -> float:
        """Return current battery state of charge.

        Returns:
            State of charge as fraction [0.0, 1.0]. 0.0 = empty, 1.0 = full.
        """
        raise NotImplementedError

    @abstractmethod
    def get_remaining_energy_wh(self) -> float:
        """Calculate remaining stored energy in battery.

        Returns:
            Remaining energy in Watt-hours, accounting for thermal derating
            and safe SoC limits.
        """
        raise NotImplementedError

    @abstractmethod
    def is_battery_low(self) -> bool:
        """Check if battery SoC has fallen below safe operating threshold.

        Returns:
            True if battery SoC < 25%, False otherwise.
            When True, rover should prioritize returning to base for charging.
        """
        raise NotImplementedError

    @abstractmethod
    def get_charging_time_hours(self, target_soc: float) -> float:
        """Estimate time to charge battery to a target state of charge.

        Assumes rover is parked under optimal sun exposure with solar charging.

        Args:
            target_soc: Target SoC as fraction [0.0, 1.0].

        Returns:
            Estimated charging time in hours. Accounts for thermal derating
            and actual panel efficiency.
        """
        raise NotImplementedError
