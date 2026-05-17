"""Concrete lunar thermal environment model (System 2.4).

`LunarThermalModel` implements the
:class:`~moon_rover.environment.thermal.model.ThermalModel` interface with a
per-component lumped-capacitance energy balance:

    C dT/dt = Q_solar + Q_internal - Q_radiative + Q_environment

* ``Q_solar``      absorbed sunlight reaching the (shielded) component,
                   proportional to ``sin(sun_elevation)``.
* ``Q_internal``   steady-state electronics/mechanical heat generation.
* ``Q_radiative``  Stefan-Boltzmann radiation to deep space (2.7 K) — the
                   dominant driver of the lunar-night cold soak.
* ``Q_environment`` conductive/insulated coupling to the surrounding surface
                   (chassis) temperature, which itself swings between the
                   configured day/night extremes with sun elevation.

The radiative term is stiff, so :meth:`step` integrates with internal
adaptive sub-steps for numerical stability while exposing a single coarse
(1 Hz) public step to callers. The model is fully deterministic.

Consumers:

* ``PowerSystem`` reads :meth:`get_component_temp` (battery) and applies
  :meth:`get_battery_capacity_factor` for cold-night derating.
* Drive control scales torque by :meth:`get_motor_efficiency_factor`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from moon_rover.environment.thermal.model import (
    ComponentThermal,
    ThermalConfig,
    ThermalModel,
)

__all__ = ["LunarThermalModel"]

# Physical constants.
STEFAN_BOLTZMANN = 5.670374419e-8  # W / (m^2 K^4)
DEEP_SPACE_TEMP_K = 2.7
SOLAR_CONSTANT_W_M2 = 1361.0
KELVIN_OFFSET = 273.15

# Fallback operating/survival envelopes (Celsius) when a component's name does
# not let us identify it and no explicit ranges are registered.
_DEFAULT_MOTOR_OPERATING = (-40.0, 80.0)
_DEFAULT_MOTOR_SURVIVAL = (-60.0, 125.0)
_DEFAULT_BATTERY_OPERATING = (0.0, 45.0)
_DEFAULT_BATTERY_SURVIVAL = (-20.0, 60.0)


class LunarThermalModel(ThermalModel):
    """Lumped-capacitance thermal model for lunar surface operations.

    Parameters:
        solar_absorptivity: Fraction of incident solar flux that reaches an
            internal component through shielding/MLI (0-1).
        emissivity: Effective infrared emissivity for radiation to space (0-1).
        env_conductance_w_per_k: Conductive coupling (W/K) between a component
            and the surrounding chassis/surface temperature.
        max_substep_s: Largest internal integration sub-step. Smaller values
            improve stability of the stiff radiative term at higher dt.
    """

    def __init__(
        self,
        solar_absorptivity: float = 0.20,
        emissivity: float = 0.85,
        env_conductance_w_per_k: float = 0.5,
        max_substep_s: float = 0.5,
    ) -> None:
        for label, val in (
            ("solar_absorptivity", solar_absorptivity),
            ("emissivity", emissivity),
        ):
            if not 0.0 <= val <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {val}")
        if env_conductance_w_per_k < 0.0:
            raise ValueError("env_conductance_w_per_k must be >= 0")
        if max_substep_s <= 0.0:
            raise ValueError("max_substep_s must be > 0")

        self._alpha = float(solar_absorptivity)
        self._emissivity = float(emissivity)
        self._k_env = float(env_conductance_w_per_k)
        self._max_substep_s = float(max_substep_s)

        self._config: Optional[ThermalConfig] = None
        self._components: Dict[str, ComponentThermal] = {}
        self._temps: Dict[str, float] = {}
        self._events: List[str] = []
        self._surface_min = -173.0
        self._surface_max = 127.0

    # ------------------------------------------------------------------ #
    # ThermalModel interface
    # ------------------------------------------------------------------ #
    def initialize(self, config: ThermalConfig) -> None:
        """Validate config and set initial component temperatures.

        Raises:
            ValueError: If any temperature range has min > max, or
                thermal_mass / radiative_area is non-positive.
            KeyError: If a component entry is not a ComponentThermal-like
                object exposing the required attributes.
        """
        s_min, s_max = config.surface_temp_range
        if s_min > s_max:
            raise ValueError(
                f"surface_temp_range min ({s_min}) > max ({s_max})"
            )
        if config.update_rate_hz <= 0.0:
            raise ValueError(
                f"update_rate_hz must be > 0, got {config.update_rate_hz}"
            )

        required = (
            "operating_range",
            "survival_range",
            "thermal_mass",
            "heat_generation",
            "radiative_area",
            "current_temp",
        )
        components: Dict[str, ComponentThermal] = {}
        for name, comp in (config.component_models or {}).items():
            for attr in required:
                if not hasattr(comp, attr):
                    raise KeyError(
                        f"component '{name}' missing required attribute "
                        f"'{attr}'"
                    )
            op_min, op_max = comp.operating_range
            sv_min, sv_max = comp.survival_range
            if op_min > op_max:
                raise ValueError(
                    f"component '{name}' operating_range min > max"
                )
            if sv_min > sv_max:
                raise ValueError(
                    f"component '{name}' survival_range min > max"
                )
            if comp.thermal_mass <= 0.0:
                raise ValueError(
                    f"component '{name}' thermal_mass must be > 0"
                )
            if comp.radiative_area < 0.0:
                raise ValueError(
                    f"component '{name}' radiative_area must be >= 0"
                )
            components[name] = comp

        self._config = config
        self._components = components
        self._temps = {n: float(c.current_temp) for n, c in components.items()}
        self._events = []
        self._surface_min = float(s_min)
        self._surface_max = float(s_max)

    def step(self, dt: float, sun_elevation: float) -> None:
        """Advance every component temperature by ``dt`` seconds.

        Raises:
            RuntimeError: If called before :meth:`initialize`.
            ValueError: If ``dt`` <= 0 or ``sun_elevation`` outside [0, 90].
        """
        if self._config is None:
            raise RuntimeError("step() called before initialize()")
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        if not 0.0 <= sun_elevation <= 90.0:
            raise ValueError(
                f"sun_elevation must be in [0, 90], got {sun_elevation}"
            )

        sin_elev = math.sin(math.radians(sun_elevation))
        # Surrounding surface/chassis temperature tracks the day/night swing.
        t_env_c = self._surface_min + (self._surface_max - self._surface_min) * sin_elev
        t_env_k = t_env_c + KELVIN_OFFSET

        n_sub = max(1, int(math.ceil(dt / self._max_substep_s)))
        h = dt / n_sub

        for name, comp in self._components.items():
            t_c = self._temps[name]
            q_solar = (
                self._alpha
                * SOLAR_CONSTANT_W_M2
                * comp.radiative_area
                * sin_elev
            )
            for _ in range(n_sub):
                t_k = t_c + KELVIN_OFFSET
                q_rad = (
                    self._emissivity
                    * STEFAN_BOLTZMANN
                    * comp.radiative_area
                    * (t_k ** 4 - DEEP_SPACE_TEMP_K ** 4)
                )
                q_env = self._k_env * (t_env_c - t_c)
                dT = (
                    (q_solar + comp.heat_generation - q_rad + q_env)
                    / comp.thermal_mass
                ) * h
                t_c += dT
            self._temps[name] = t_c

        self._events = self._detect_events()

    def get_component_temp(self, component_name: str) -> float:
        """Current component temperature in Celsius.

        Raises:
            KeyError: If the component is not registered.
        """
        if component_name not in self._temps:
            raise KeyError(f"unknown component '{component_name}'")
        return self._temps[component_name]

    def get_motor_efficiency_factor(self, motor_temp: float) -> float:
        """Torque scaling (0-1) for a motor at ``motor_temp`` Celsius."""
        op, sv = self._ranges_for("motor")
        return self._derate_factor(motor_temp, op, sv)

    def get_battery_capacity_factor(self, battery_temp: float) -> float:
        """Usable-capacity scaling (0-1) for a battery at ``battery_temp`` C."""
        op, sv = self._ranges_for("battery")
        return self._derate_factor(battery_temp, op, sv)

    def check_thermal_events(self) -> List[str]:
        """Thermal events detected during the most recent :meth:`step`."""
        return list(self._events)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _derate_factor(
        temp: float,
        operating: Tuple[float, float],
        survival: Tuple[float, float],
    ) -> float:
        """1.0 inside operating range, linear to 0.0 at the survival limits."""
        op_min, op_max = operating
        sv_min, sv_max = survival
        if op_min <= temp <= op_max:
            return 1.0
        if temp < op_min:
            if temp <= sv_min:
                return 0.0
            span = op_min - sv_min
            return 0.0 if span <= 0.0 else (temp - sv_min) / span
        # temp > op_max
        if temp >= sv_max:
            return 0.0
        span = sv_max - op_max
        return 0.0 if span <= 0.0 else (sv_max - temp) / span

    def _ranges_for(
        self, kind: str
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Resolve operating/survival ranges for 'motor' or 'battery'.

        Uses the first registered component whose name contains ``kind``;
        falls back to documented lunar-hardware defaults otherwise.
        """
        for name, comp in self._components.items():
            if kind in name.lower():
                return comp.operating_range, comp.survival_range
        if kind == "battery":
            return _DEFAULT_BATTERY_OPERATING, _DEFAULT_BATTERY_SURVIVAL
        return _DEFAULT_MOTOR_OPERATING, _DEFAULT_MOTOR_SURVIVAL

    def _detect_events(self) -> List[str]:
        """Compare current temperatures against each component's envelopes."""
        events: List[str] = []
        for name, comp in self._components.items():
            t = self._temps[name]
            op_min, op_max = comp.operating_range
            sv_min, sv_max = comp.survival_range
            if t >= sv_max:
                events.append(f"{name}_thermal_cutoff")
            elif t > op_max:
                events.append(f"{name}_overheat")
            if t <= sv_min:
                events.append(f"{name}_survival_freeze")
            elif t < op_min:
                events.append(f"{name}_freeze")
        return events
