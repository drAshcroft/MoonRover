"""Unit tests for LunarThermalModel (System 2.4)."""

from __future__ import annotations

import pytest

from moon_rover.environment.thermal import (
    ComponentThermal,
    LunarThermalModel,
    ThermalConfig,
)


def _battery(current_temp: float = 20.0) -> ComponentThermal:
    return ComponentThermal(
        operating_range=(0.0, 45.0),
        survival_range=(-20.0, 60.0),
        thermal_mass=800.0,       # J/K
        heat_generation=2.0,      # W
        radiative_area=0.05,      # m^2
        current_temp=current_temp,
    )


def _motor(current_temp: float = 20.0) -> ComponentThermal:
    return ComponentThermal(
        operating_range=(-40.0, 80.0),
        survival_range=(-60.0, 125.0),
        thermal_mass=300.0,
        heat_generation=5.0,
        radiative_area=0.03,
        current_temp=current_temp,
    )


def _model(**components) -> LunarThermalModel:
    m = LunarThermalModel()
    m.initialize(ThermalConfig(component_models=components or {"battery_main": _battery()}))
    return m


# --------------------------------------------------------------------------- #
# Construction / initialize validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [
    {"solar_absorptivity": -0.1},
    {"solar_absorptivity": 1.1},
    {"emissivity": 2.0},
    {"env_conductance_w_per_k": -1.0},
    {"max_substep_s": 0.0},
])
def test_bad_constructor_args(kwargs):
    with pytest.raises(ValueError):
        LunarThermalModel(**kwargs)


def test_initialize_rejects_inverted_surface_range():
    m = LunarThermalModel()
    with pytest.raises(ValueError):
        m.initialize(ThermalConfig(surface_temp_range=(100.0, -50.0)))


def test_initialize_rejects_inverted_component_range():
    m = LunarThermalModel()
    bad = ComponentThermal((50.0, 10.0), (-20.0, 60.0), 100.0, 1.0, 0.01, 20.0)
    with pytest.raises(ValueError):
        m.initialize(ThermalConfig(component_models={"x": bad}))


def test_initialize_rejects_nonpositive_thermal_mass():
    m = LunarThermalModel()
    bad = ComponentThermal((0.0, 45.0), (-20.0, 60.0), 0.0, 1.0, 0.01, 20.0)
    with pytest.raises(ValueError):
        m.initialize(ThermalConfig(component_models={"x": bad}))


def test_initialize_rejects_malformed_component():
    m = LunarThermalModel()
    with pytest.raises(KeyError):
        m.initialize(ThermalConfig(component_models={"x": object()}))


# --------------------------------------------------------------------------- #
# step validation
# --------------------------------------------------------------------------- #
def test_step_before_initialize_raises():
    with pytest.raises(RuntimeError):
        LunarThermalModel().step(1.0, 45.0)


def test_step_rejects_bad_dt_and_elevation():
    m = _model()
    with pytest.raises(ValueError):
        m.step(0.0, 45.0)
    with pytest.raises(ValueError):
        m.step(1.0, -1.0)
    with pytest.raises(ValueError):
        m.step(1.0, 91.0)


def test_get_component_temp_unknown_raises():
    m = _model()
    with pytest.raises(KeyError):
        m.get_component_temp("nope")


# --------------------------------------------------------------------------- #
# Thermal physics behavior
# --------------------------------------------------------------------------- #
def test_initial_temp_reflects_config():
    m = _model(battery_main=_battery(current_temp=15.0))
    assert m.get_component_temp("battery_main") == pytest.approx(15.0)


def test_night_cold_soak_cools_component():
    m = _model(battery_main=_battery(current_temp=20.0))
    # Sun on the horizon (elevation 0): no solar input, radiative loss to space.
    for _ in range(6):
        m.step(600.0, 0.0)
    assert m.get_component_temp("battery_main") < 20.0


def test_daytime_heats_component_above_night():
    night = _model(battery_main=_battery(current_temp=-30.0))
    day = _model(battery_main=_battery(current_temp=-30.0))
    for _ in range(6):
        night.step(600.0, 0.0)
        day.step(600.0, 85.0)
    assert day.get_component_temp("battery_main") > night.get_component_temp(
        "battery_main"
    )


def test_temperature_stays_physically_bounded():
    m = _model(battery_main=_battery(current_temp=20.0))
    for _ in range(40):
        m.step(900.0, 0.0)
    t = m.get_component_temp("battery_main")
    # Cannot fall below deep-space-driven floor; well above absolute zero.
    assert -273.15 < t < 130.0


def test_multiple_components_tracked_independently():
    m = _model(battery_main=_battery(20.0), motor_left=_motor(20.0))
    m.step(600.0, 60.0)
    tb = m.get_component_temp("battery_main")
    tm = m.get_component_temp("motor_left")
    assert tb != tm  # different mass/area/heat => diverge


def test_determinism():
    def run():
        m = _model(battery_main=_battery(20.0), motor_left=_motor(10.0))
        for _ in range(5):
            m.step(300.0, 30.0)
        return (m.get_component_temp("battery_main"),
                m.get_component_temp("motor_left"))

    assert run() == run()


def test_substep_count_does_not_change_result_meaningfully():
    a = LunarThermalModel(max_substep_s=0.25)
    b = LunarThermalModel(max_substep_s=1.0)
    cfg = lambda: ThermalConfig(component_models={"battery_main": _battery(20.0)})
    a.initialize(cfg())
    b.initialize(cfg())
    for _ in range(4):
        a.step(120.0, 20.0)
        b.step(120.0, 20.0)
    assert a.get_component_temp("battery_main") == pytest.approx(
        b.get_component_temp("battery_main"), abs=1.0
    )


# --------------------------------------------------------------------------- #
# Derating factors
# --------------------------------------------------------------------------- #
def test_motor_efficiency_within_operating_is_unity():
    m = _model(motor_left=_motor())
    assert m.get_motor_efficiency_factor(20.0) == 1.0
    assert m.get_motor_efficiency_factor(-40.0) == 1.0
    assert m.get_motor_efficiency_factor(80.0) == 1.0


def test_motor_efficiency_zero_beyond_survival():
    m = _model(motor_left=_motor())
    assert m.get_motor_efficiency_factor(200.0) == 0.0
    assert m.get_motor_efficiency_factor(-100.0) == 0.0


def test_motor_efficiency_linear_midpoint():
    m = _model(motor_left=_motor())
    # Hot side: operating max 80, survival max 125 -> midpoint 102.5 => 0.5.
    assert m.get_motor_efficiency_factor(102.5) == pytest.approx(0.5, abs=1e-6)
    # Cold side: operating min -40, survival min -60 -> -50 => 0.5.
    assert m.get_motor_efficiency_factor(-50.0) == pytest.approx(0.5, abs=1e-6)


def test_battery_capacity_factor_uses_registered_ranges():
    m = _model(battery_main=_battery())
    assert m.get_battery_capacity_factor(20.0) == 1.0
    assert m.get_battery_capacity_factor(-20.0) == 0.0  # survival min
    assert m.get_battery_capacity_factor(60.0) == 0.0   # survival max


def test_derate_uses_defaults_when_no_named_component():
    m = _model(widget=_battery())  # name has neither 'motor' nor 'battery'
    # Falls back to default battery envelope (0..45 op, -20..60 surv).
    assert m.get_battery_capacity_factor(22.0) == 1.0
    assert m.get_battery_capacity_factor(-20.0) == 0.0
    # Default motor envelope (-40..80 op).
    assert m.get_motor_efficiency_factor(0.0) == 1.0


# --------------------------------------------------------------------------- #
# Thermal events
# --------------------------------------------------------------------------- #
def test_no_events_when_nominal():
    m = _model(battery_main=_battery(20.0))
    m.step(10.0, 45.0)
    assert m.check_thermal_events() == []


def test_overheat_and_cutoff_events():
    m = _model(battery_main=_battery(current_temp=70.0))  # > survival max 60
    m.step(1.0, 45.0)
    events = m.check_thermal_events()
    assert any("battery_main" in e and "cutoff" in e for e in events)


def test_freeze_event_on_cold_component():
    m = _model(motor_left=_motor(current_temp=-70.0))  # < survival min -60
    m.step(1.0, 0.0)
    events = m.check_thermal_events()
    assert any("motor_left" in e and "freeze" in e for e in events)


def test_events_reflect_latest_step_only():
    m = _model(battery_main=_battery(current_temp=70.0))
    m.step(1.0, 45.0)
    assert m.check_thermal_events()  # overheated initially
    # Reset to a nominal component and confirm events clear next step.
    m.initialize(ThermalConfig(component_models={"battery_main": _battery(20.0)}))
    m.step(1.0, 45.0)
    assert m.check_thermal_events() == []
