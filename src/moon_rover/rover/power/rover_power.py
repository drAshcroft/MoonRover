"""Concrete PowerSystem implementation: solar + battery + load budgeting.

This module implements the :class:`PowerSystem` ABC for production rover use.
It integrates solar generation (as a function of sun elevation and dust), tracks
battery state of charge via Coulomb counting, applies thermal derating, and
reports a power-flow snapshot (:class:`PowerState`) each simulation step.

Design choices
--------------
* State of charge is a fraction in ``[0.0, 1.0]``; the battery's safe operating
  window ``soc_range`` is enforced as a soft clamp, so discharge above the lower
  bound returns real energy and discharge below it returns zero (the battery is
  effectively in protective shutoff).
* Thermal derating reduces effective battery capacity at low temperatures using
  ``temp_derating_pct_per_deg`` per degree below 20 °C nominal.
* Solar output follows a cosine law over sun elevation, capped by the panel
  nominal max power.
* ``is_battery_low()`` returns ``True`` when SoC drops below 25%, matching the
  contract documented in the ABC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from moon_rover.rover.power.systems import (
    BatteryConfig,
    PowerBudget,
    PowerState,
    PowerSystem,
    SolarArrayConfig,
)


#: Reference battery temperature for derating (Celsius).
_NOMINAL_BATTERY_TEMP_C = 20.0

#: Mean lunar surface irradiance at normal incidence (W/m^2).
_LUNAR_IRRADIANCE_W_M2 = 1361.0

#: SoC threshold below which ``is_battery_low`` returns True.
_LOW_BATTERY_THRESHOLD = 0.25

#: How aggressively consumption scales with operating regime.
_REGIME_POWER_SCALE: Dict[str, float] = {
    "idle": 0.2,
    "active": 1.0,
    "peak": 1.5,
}


class RoverPowerSystem(PowerSystem):
    """Genesis-agnostic power system managing solar, battery, and load budget.

    The implementation is intentionally decoupled from the physics engine: it
    advances its own internal state when :meth:`step` is called with the time
    delta, sun elevation, and a dict of active subsystem regimes.
    """

    def __init__(self) -> None:
        self._solar: Optional[SolarArrayConfig] = None
        self._battery: Optional[BatteryConfig] = None
        self._budget: Optional[PowerBudget] = None

        self._initialized = False
        self._soc: float = 0.0
        self._battery_temp_c: float = _NOMINAL_BATTERY_TEMP_C
        self._last_solar_w: float = 0.0
        self._last_draw_w: float = 0.0

    # ------------------------------------------------------------------
    # ABC: initialize
    # ------------------------------------------------------------------

    def initialize(
        self,
        solar_config: SolarArrayConfig,
        battery_config: BatteryConfig,
        budget: PowerBudget,
    ) -> None:
        _validate_solar(solar_config)
        _validate_battery(battery_config)
        self._solar = solar_config
        self._battery = battery_config
        self._budget = budget

        # Seed SoC at the safe upper limit — a fully-charged rover starts the
        # mission at the top of the configured operating range.
        self._soc = float(battery_config.soc_range[1])
        self._battery_temp_c = _NOMINAL_BATTERY_TEMP_C
        self._initialized = True

    # ------------------------------------------------------------------
    # ABC: step
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float,
        sun_elevation: float,
        subsystem_states: dict[str, Any],
    ) -> PowerState:
        if not self._initialized:
            raise RuntimeError("PowerSystem.initialize() must be called first")
        if dt < 0.0:
            raise ValueError(f"dt must be >= 0, got {dt}")

        assert self._solar is not None and self._battery is not None and self._budget is not None

        solar_w = self._compute_solar_output(sun_elevation)
        draw_w = self._compute_total_draw(subsystem_states)
        net_w = solar_w - draw_w

        # Integrate battery energy over dt. Capacity is thermally derated, so
        # 1% SoC corresponds to (0.01 · capacity_derated_wh) of energy.
        capacity_wh = self._effective_capacity_wh()
        if capacity_wh > 1e-6 and dt > 0.0:
            delta_wh = (net_w * dt) / 3600.0
            self._soc = _clamp(
                self._soc + (delta_wh / capacity_wh),
                self._battery.soc_range[0] * 0.5,  # allow brief dip below
                1.0,
            )

        self._last_solar_w = solar_w
        self._last_draw_w = draw_w

        return PowerState(
            battery_soc=self._soc,
            solar_output_w=solar_w,
            total_draw_w=draw_w,
            net_power_w=net_w,
            battery_temp_c=self._battery_temp_c,
        )

    # ------------------------------------------------------------------
    # ABC: readers
    # ------------------------------------------------------------------

    def get_battery_soc(self) -> float:
        self._require_initialized()
        return self._soc

    def get_remaining_energy_wh(self) -> float:
        self._require_initialized()
        assert self._battery is not None
        usable_fraction = max(0.0, self._soc - self._battery.soc_range[0])
        return usable_fraction * self._effective_capacity_wh()

    def is_battery_low(self) -> bool:
        self._require_initialized()
        return self._soc < _LOW_BATTERY_THRESHOLD

    def get_charging_time_hours(self, target_soc: float) -> float:
        self._require_initialized()
        assert self._solar is not None and self._battery is not None

        target = _clamp(float(target_soc), 0.0, 1.0)
        if target <= self._soc:
            return 0.0

        deficit_wh = (target - self._soc) * self._effective_capacity_wh()
        # Optimal charging: assume panels face the sun at 45° elevation, which
        # the cosine law reduces to ~0.71. Real charging depends on rover
        # orientation; this estimator is conservative.
        optimal_solar_w = self._panel_peak_power_w() * 0.71
        if optimal_solar_w < 1e-3:
            return float("inf")
        return deficit_wh / optimal_solar_w

    # ------------------------------------------------------------------
    # Extended API (non-ABC): thermal telemetry
    # ------------------------------------------------------------------

    def set_battery_temperature(self, temp_c: float) -> None:
        """Update the battery temperature used for capacity derating.

        Thermal plant simulations / :class:`ThermalModel` feed this each step so
        cold-night discharge reduces effective stored energy.
        """
        self._battery_temp_c = float(temp_c)

    def get_last_solar_w(self) -> float:
        return self._last_solar_w

    def get_last_draw_w(self) -> float:
        return self._last_draw_w

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_solar_output(self, sun_elevation_deg: float) -> float:
        """Solar panel power as a function of sun elevation and dust coverage."""
        assert self._solar is not None
        elev = max(0.0, min(90.0, float(sun_elevation_deg)))
        cos_law = _sin_deg(elev)  # elev of 90° → full panel, 0° → zero
        irradiance = _LUNAR_IRRADIANCE_W_M2 * cos_law
        panel_area = self._solar.num_panels * self._solar.area_per_panel_m2
        raw_w = irradiance * panel_area * self._solar.efficiency
        return max(0.0, raw_w * self._solar.dust_factor)

    def _compute_total_draw(self, subsystem_states: Mapping[str, Any]) -> float:
        """Sum per-subsystem draws based on their current operating regime."""
        assert self._budget is not None
        total = 0.0
        subsystem_map = {
            "drive_motors": self._budget.drive_motors_w,
            "manipulator": self._budget.manipulator_w,
            "lidar": self._budget.lidar_w,
            "cameras": self._budget.cameras_w,
            "imu": self._budget.imu_w,
            "compute": self._budget.compute_w,
            "comms": self._budget.comms_w,
            "heating": self._budget.heating_w,
        }
        for name, levels in subsystem_map.items():
            regime = str(subsystem_states.get(name, "idle")).lower()
            # Accept either a dict of levels or a scalar float for backwards
            # compatibility with simpler budget configs.
            if isinstance(levels, dict):
                base = float(levels.get(regime, levels.get("idle", 0.0)))
                total += base
            else:
                scale = _REGIME_POWER_SCALE.get(regime, 1.0)
                total += float(levels) * scale
        return total

    def _effective_capacity_wh(self) -> float:
        """Battery capacity after applying cold-temperature derating."""
        assert self._battery is not None
        delta_c = _NOMINAL_BATTERY_TEMP_C - self._battery_temp_c
        if delta_c <= 0.0:
            derate_fraction = 0.0
        else:
            derate_fraction = min(0.8, self._battery.temp_derating_pct_per_deg * delta_c)
        return self._battery.capacity_wh * (1.0 - derate_fraction)

    def _panel_peak_power_w(self) -> float:
        assert self._solar is not None
        return (
            _LUNAR_IRRADIANCE_W_M2
            * self._solar.num_panels
            * self._solar.area_per_panel_m2
            * self._solar.efficiency
            * self._solar.dust_factor
        )

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("PowerSystem.initialize() must be called first")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_solar(cfg: SolarArrayConfig) -> None:
    if cfg.num_panels <= 0:
        raise ValueError(f"num_panels must be positive, got {cfg.num_panels}")
    if cfg.area_per_panel_m2 <= 0:
        raise ValueError(f"area_per_panel_m2 must be positive, got {cfg.area_per_panel_m2}")
    if not 0.0 < cfg.efficiency <= 1.0:
        raise ValueError(f"efficiency must be in (0, 1], got {cfg.efficiency}")
    if not 0.0 <= cfg.dust_factor <= 1.0:
        raise ValueError(f"dust_factor must be in [0, 1], got {cfg.dust_factor}")


def _validate_battery(cfg: BatteryConfig) -> None:
    if cfg.capacity_wh <= 0:
        raise ValueError(f"capacity_wh must be positive, got {cfg.capacity_wh}")
    lo, hi = cfg.soc_range
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"soc_range must satisfy 0 <= lo < hi <= 1, got {cfg.soc_range}")
    if cfg.max_discharge_c <= 0:
        raise ValueError(f"max_discharge_c must be positive, got {cfg.max_discharge_c}")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sin_deg(deg: float) -> float:
    import math

    return math.sin(math.radians(deg))


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def power_config_from_yaml(power_cfg: Mapping[str, Any]) -> tuple[SolarArrayConfig, BatteryConfig, PowerBudget]:
    """Build :class:`SolarArrayConfig`, :class:`BatteryConfig`, and
    :class:`PowerBudget` from a parsed ``rover.yaml`` ``power`` section.

    The three objects together are the arguments for :meth:`PowerSystem.initialize`.
    """
    solar_cfg = power_cfg.get("solar", {})
    battery_cfg = power_cfg.get("battery", {})
    consumers = power_cfg.get("power_budget", {}).get("consumers", {})

    solar = SolarArrayConfig(
        num_panels=int(solar_cfg.get("num_panels", 4)),
        area_per_panel_m2=float(solar_cfg.get("area_per_panel_m2", 0.5)),
        efficiency=float(solar_cfg.get("efficiency", 0.2)),
        dust_factor=float(solar_cfg.get("dust_factor", 1.0)),
    )
    battery = BatteryConfig(
        capacity_wh=float(battery_cfg.get("capacity_wh", 2000.0)),
        soc_range=(
            float(battery_cfg.get("min_soc", 0.20)),
            float(battery_cfg.get("max_soc", 0.95)),
        ),
        max_discharge_c=float(battery_cfg.get("max_discharge_c", 2.0)),
        temp_derating_pct_per_deg=float(
            battery_cfg.get("temperature", {}).get("cold_derating_percent_per_c", 1.0) / 100.0
        ),
    )

    def _consumer_levels(name: str) -> dict[str, float]:
        entry = consumers.get(name, {})
        if isinstance(entry, dict) and "power_w" in entry:
            watts = float(entry["power_w"])
            return {
                "idle": watts * _REGIME_POWER_SCALE["idle"],
                "active": watts * _REGIME_POWER_SCALE["active"],
                "peak": watts * _REGIME_POWER_SCALE["peak"],
            }
        return {"idle": 0.0, "active": 0.0, "peak": 0.0}

    budget = PowerBudget(
        drive_motors_w=_consumer_levels("mobility"),
        manipulator_w=_consumer_levels("manipulator"),
        lidar_w=_consumer_levels("lidar"),
        cameras_w=_consumer_levels("cameras"),
        imu_w=_consumer_levels("imu"),
        compute_w=_consumer_levels("compute"),
        comms_w=_consumer_levels("communication"),
        heating_w={"idle": 5.0, "active": 30.0, "peak": 80.0},
    )
    return solar, battery, budget
