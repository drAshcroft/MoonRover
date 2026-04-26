"""System 2.4: Thermal Environment Model.

This module provides interfaces for simulating thermal conditions on the lunar surface,
including temperature extremes, component-specific thermal behavior, and thermal events.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class ComponentThermal:
    """Thermal properties and state of a system component.

    Parameters:
        operating_range: Temperature range for normal operation (min_C, max_C).
        survival_range: Temperature range component can survive (min_C, max_C).
            Component fails if outside this range.
        thermal_mass: Thermal mass in J/K (higher = slower to heat/cool).
        heat_generation: Steady-state heat generation in watts.
        radiative_area: Effective radiative area in m^2.
        current_temp: Current component temperature in Celsius. Initial state.
    """
    operating_range: Tuple[float, float]
    survival_range: Tuple[float, float]
    thermal_mass: float
    heat_generation: float
    radiative_area: float
    current_temp: float


@dataclass
class ThermalConfig:
    """Configuration for thermal environment simulation.

    Parameters:
        surface_temp_range: Lunar surface temperature range (-173 to 127 C typical).
        component_models: Dictionary mapping component names to ComponentThermal objects.
        update_rate_hz: Thermal update frequency in Hz. Typically 1 Hz (coarse updates).
    """
    surface_temp_range: Tuple[float, float] = (-173.0, 127.0)
    component_models: Dict[str, ComponentThermal] = None
    update_rate_hz: float = 1.0

    def __post_init__(self) -> None:
        """Set default component models if not provided."""
        if self.component_models is None:
            self.component_models = {}


class ThermalModel(ABC):
    """Abstract interface for lunar thermal environment simulation.

    Models surface and component temperatures, including solar heating,
    radiative cooling, component heat generation, and thermal failures.
    Runs at low frequency (1 Hz) separate from physics stepping.
    """

    @abstractmethod
    def initialize(self, config: ThermalConfig) -> None:
        """Initialize thermal model with components and parameters.

        Sets up thermal mass matrix, radiative properties, and initial
        component temperatures.

        Parameters:
            config: ThermalConfig with surface range and component models.

        Raises:
            ValueError: If temperature ranges are invalid (min > max).
            KeyError: If component models have invalid structures.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float, sun_elevation: float) -> None:
        """Advance thermal simulation by dt seconds.

        Executes thermal equations:
        1. Compute solar heating from sun elevation
        2. Compute radiative cooling to space
        3. Update component temperatures via thermal mass
        4. Apply insulation and conduction between components
        5. Detect thermal events (overheating, freezing)

        Parameters:
            dt: Timestep in seconds.
            sun_elevation: Current sun elevation angle in degrees (0-90).
                Used to compute solar heating intensity.

        Raises:
            ValueError: If sun_elevation is outside 0-90 degrees.
        """
        raise NotImplementedError

    @abstractmethod
    def get_component_temp(self, component_name: str) -> float:
        """Get current temperature of a component.

        Parameters:
            component_name: Name of component (e.g., "motor_left", "battery_main").

        Returns:
            Temperature in Celsius.

        Raises:
            KeyError: If component is not in model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_motor_efficiency_factor(self, motor_temp: float) -> float:
        """Get motor efficiency scaling factor based on temperature.

        Efficiency decreases at temperature extremes:
        - At operating range: factor = 1.0 (100% efficiency)
        - Outside operating range: factor decreases linearly
        - At survival limit: factor = 0.0 (motor non-functional)

        Parameters:
            motor_temp: Motor temperature in Celsius.

        Returns:
            Efficiency factor (0-1). 1.0 = full efficiency, 0.0 = no torque output.

        Example:
            motor_torque_actual = motor_torque_nominal * get_motor_efficiency_factor(temp)
        """
        raise NotImplementedError

    @abstractmethod
    def get_battery_capacity_factor(self, battery_temp: float) -> float:
        """Get battery capacity scaling factor based on temperature.

        Battery capacity is reduced at temperature extremes:
        - At operating range: factor = 1.0 (100% capacity)
        - Outside operating range: factor decreases
        - At survival limit: factor = 0.0 (battery non-functional)

        Parameters:
            battery_temp: Battery temperature in Celsius.

        Returns:
            Capacity factor (0-1). 1.0 = full capacity, 0.0 = no energy available.

        Example:
            available_energy = battery_max_energy * get_battery_capacity_factor(temp)
        """
        raise NotImplementedError

    @abstractmethod
    def check_thermal_events(self) -> List[str]:
        """Check for and report thermal failure events.

        Returns list of all thermal events that occurred in the last step:
        - "motor_LEFT_overheat": Motor exceeded operating range
        - "battery_MAIN_thermal_cutoff": Battery disabled due to temperature
        - "antenna_freeze": Antenna mechanism failed from cold
        - etc.

        Returns:
            List of event strings. Empty if no events.

        Example:
            events = check_thermal_events()
            for event in events:
                log.warning(f"Thermal event: {event}")
        """
        raise NotImplementedError
