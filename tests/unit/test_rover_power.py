"""Unit tests for RoverPowerSystem (solar + battery + load accounting)."""
from __future__ import annotations

import math

import pytest

from moon_rover.rover.power.rover_power import RoverPowerSystem, power_config_from_yaml
from moon_rover.rover.power.systems import (
    BatteryConfig,
    PowerBudget,
    SolarArrayConfig,
)


def _solar(num_panels: int = 4, area: float = 0.5, eff: float = 0.2, dust: float = 1.0) -> SolarArrayConfig:
    return SolarArrayConfig(num_panels=num_panels, area_per_panel_m2=area, efficiency=eff, dust_factor=dust)


def _battery(capacity: float = 1000.0, lo: float = 0.2, hi: float = 0.95) -> BatteryConfig:
    return BatteryConfig(
        capacity_wh=capacity,
        soc_range=(lo, hi),
        max_discharge_c=2.0,
        temp_derating_pct_per_deg=0.01,
    )


def _budget(all_idle_w: float = 0.0) -> PowerBudget:
    levels = {"idle": all_idle_w, "active": all_idle_w, "peak": all_idle_w}
    return PowerBudget(
        drive_motors_w=dict(levels),
        manipulator_w=dict(levels),
        lidar_w=dict(levels),
        cameras_w=dict(levels),
        imu_w=dict(levels),
        compute_w=dict(levels),
        comms_w=dict(levels),
        heating_w=dict(levels),
    )


def _initialized_system(**overrides) -> RoverPowerSystem:
    sys_ = RoverPowerSystem()
    sys_.initialize(_solar(**overrides.get("solar", {})),
                    _battery(**overrides.get("battery", {})),
                    overrides.get("budget", _budget()))
    return sys_


# ---------------------------------------------------------------------------
# Initialization / validation
# ---------------------------------------------------------------------------


def test_initialize_seeds_soc_to_upper_limit():
    sys_ = _initialized_system()
    assert sys_.get_battery_soc() == pytest.approx(0.95)


def test_initialize_rejects_bad_efficiency():
    sys_ = RoverPowerSystem()
    with pytest.raises(ValueError):
        sys_.initialize(_solar(eff=0.0), _battery(), _budget())


def test_initialize_rejects_inverted_soc_range():
    sys_ = RoverPowerSystem()
    with pytest.raises(ValueError):
        sys_.initialize(_solar(), _battery(lo=0.9, hi=0.5), _budget())


def test_step_before_initialize_raises():
    sys_ = RoverPowerSystem()
    with pytest.raises(RuntimeError):
        sys_.step(1.0, 45.0, {})


def test_get_battery_soc_before_initialize_raises():
    sys_ = RoverPowerSystem()
    with pytest.raises(RuntimeError):
        sys_.get_battery_soc()


# ---------------------------------------------------------------------------
# Solar output
# ---------------------------------------------------------------------------


def test_solar_zero_at_horizon():
    sys_ = _initialized_system()
    state = sys_.step(0.0, 0.0, {})
    assert state.solar_output_w == pytest.approx(0.0)


def test_solar_peak_at_zenith():
    sys_ = _initialized_system()
    state = sys_.step(0.0, 90.0, {})
    # 4 panels × 0.5 m² × 0.2 eff × 1361 W/m² ≈ 544.4 W.
    assert state.solar_output_w == pytest.approx(0.2 * 4 * 0.5 * 1361.0, rel=1e-6)


def test_solar_follows_sin_law_of_elevation():
    sys_ = _initialized_system()
    at30 = sys_.step(0.0, 30.0, {}).solar_output_w
    at90 = sys_.step(0.0, 90.0, {}).solar_output_w
    assert at30 == pytest.approx(at90 * math.sin(math.radians(30.0)), rel=1e-6)


def test_solar_reduced_by_dust_factor():
    clean = _initialized_system(solar={"dust": 1.0}).step(0.0, 90.0, {}).solar_output_w
    dusty = _initialized_system(solar={"dust": 0.5}).step(0.0, 90.0, {}).solar_output_w
    assert dusty == pytest.approx(0.5 * clean, rel=1e-6)


# ---------------------------------------------------------------------------
# Battery SoC integration
# ---------------------------------------------------------------------------


def test_soc_decreases_when_draw_exceeds_solar():
    budget = _budget(all_idle_w=100.0)
    sys_ = RoverPowerSystem()
    sys_.initialize(_solar(), _battery(capacity=500.0), budget)
    before = sys_.get_battery_soc()
    # 8 consumers × 100 W = 800 W draw, no sun → discharge.
    sys_.step(60.0, 0.0, {"drive_motors": "active", "manipulator": "idle",
                          "lidar": "active", "cameras": "active", "imu": "active",
                          "compute": "active", "comms": "active", "heating": "active"})
    assert sys_.get_battery_soc() < before


def test_soc_increases_when_solar_exceeds_draw():
    sys_ = _initialized_system(battery={"capacity": 5_000.0, "lo": 0.2, "hi": 0.95},
                               budget=_budget(all_idle_w=0.0))
    # Drop SoC first so upper-bound clamp doesn't mask the gain.
    sys_._soc = 0.5
    before = sys_.get_battery_soc()
    sys_.step(60.0, 90.0, {})
    assert sys_.get_battery_soc() > before


def test_soc_clamped_to_upper_bound():
    sys_ = _initialized_system()
    # Already at 0.95; massive solar should not push above 1.0.
    for _ in range(100):
        sys_.step(3600.0, 90.0, {})
    assert sys_.get_battery_soc() <= 1.0


# ---------------------------------------------------------------------------
# Low-battery threshold / remaining energy
# ---------------------------------------------------------------------------


def test_is_battery_low_flips_at_twenty_five_percent():
    sys_ = _initialized_system()
    sys_._soc = 0.30
    assert not sys_.is_battery_low()
    sys_._soc = 0.24
    assert sys_.is_battery_low()


def test_remaining_energy_accounts_for_min_soc():
    sys_ = _initialized_system(battery={"capacity": 1000.0, "lo": 0.2, "hi": 0.95})
    sys_._soc = 0.5
    # usable = (0.5 - 0.2) * 1000 Wh = 300 Wh when no thermal derating.
    assert sys_.get_remaining_energy_wh() == pytest.approx(300.0, rel=1e-6)


def test_remaining_energy_never_negative_below_min_soc():
    sys_ = _initialized_system()
    sys_._soc = 0.05
    assert sys_.get_remaining_energy_wh() >= 0.0


# ---------------------------------------------------------------------------
# Thermal derating
# ---------------------------------------------------------------------------


def test_cold_battery_reduces_effective_capacity():
    sys_ = _initialized_system()
    nominal = sys_._effective_capacity_wh()
    sys_.set_battery_temperature(-20.0)  # 40 °C below 20 °C nominal
    cold = sys_._effective_capacity_wh()
    assert cold < nominal


def test_warm_battery_no_capacity_penalty():
    sys_ = _initialized_system()
    nominal = sys_._effective_capacity_wh()
    sys_.set_battery_temperature(25.0)
    assert sys_._effective_capacity_wh() == pytest.approx(nominal, rel=1e-9)


# ---------------------------------------------------------------------------
# Charging time estimate
# ---------------------------------------------------------------------------


def test_charging_time_zero_when_already_at_or_above_target():
    sys_ = _initialized_system()
    assert sys_.get_charging_time_hours(0.5) == 0.0
    assert sys_.get_charging_time_hours(0.95) == 0.0


def test_charging_time_positive_when_below_target():
    sys_ = _initialized_system()
    sys_._soc = 0.3
    t = sys_.get_charging_time_hours(0.9)
    assert t > 0.0 and math.isfinite(t)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


def test_power_config_from_yaml_round_trip():
    raw = {
        "solar": {"num_panels": 4, "area_per_panel_m2": 0.5,
                  "efficiency": 0.22, "dust_factor": 0.9},
        "battery": {"capacity_wh": 1800.0, "min_soc": 0.25, "max_soc": 0.9,
                    "max_discharge_c": 2.0,
                    "temperature": {"cold_derating_percent_per_c": 2.0}},
        "power_budget": {
            "consumers": {
                "mobility": {"power_w": 100.0},
                "manipulator": {"power_w": 20.0},
                "lidar": {"power_w": 10.0},
                "cameras": {"power_w": 5.0},
                "imu": {"power_w": 1.0},
                "compute": {"power_w": 40.0},
                "communication": {"power_w": 8.0},
            }
        },
    }
    solar, battery, budget = power_config_from_yaml(raw)
    assert solar.num_panels == 4
    assert solar.dust_factor == pytest.approx(0.9)
    assert battery.soc_range == (0.25, 0.9)
    assert battery.temp_derating_pct_per_deg == pytest.approx(0.02, rel=1e-6)
    assert budget.drive_motors_w["active"] == pytest.approx(100.0)
    assert budget.drive_motors_w["peak"] == pytest.approx(150.0)
