"""Unit tests for phase-5 subsystems: cable, antenna, moonbase.

All deterministic and engine-free (the optional ``engine`` is either None or a
tiny flat-terrain double), mirroring the physics mock-test convention.
"""
from __future__ import annotations

import numpy as np
import pytest

from moon_rover.antenna import (
    AntennaConfig,
    AntennaState,
    DeployableAntennaUnit,
)
from moon_rover.cable import CableConfig, RigidLinkCableSystem
from moon_rover.moonbase import LunarMoonbase, MoonbaseConfig


class FlatEngine:
    def get_terrain_height(self, x: float, y: float) -> float:
        return 0.0


# --------------------------------------------------------------------------
# Cable system
# --------------------------------------------------------------------------


def _cable_cfg(**kw) -> CableConfig:
    base = dict(
        link_length_m=1.0,
        link_diameter_m=0.02,
        link_mass_kg=0.5,
        total_length_m=10.0,
        joint_damping=0.1,
        joint_stiffness=50.0,
        terrain_friction=0.6,
        max_tension_n=200.0,
        bend_radius_min_m=0.3,
        voltage_dc=48.0,
        resistance_per_m=0.01,
    )
    base.update(kw)
    return CableConfig(**base)


def test_cable_preallocates_links():
    c = RigidLinkCableSystem()
    c.initialize(_cable_cfg(), FlatEngine())
    links = c.get_link_states()
    assert len(links) == 10
    assert all(not l.active for l in links)
    assert c.get_spool_state().remaining_length_m == pytest.approx(10.0)


def test_cable_initialize_validation():
    c = RigidLinkCableSystem()
    with pytest.raises(ValueError):
        c.initialize(_cable_cfg(total_length_m=0.0), None)


def test_cable_activate_and_drag():
    c = RigidLinkCableSystem()
    c.initialize(_cable_cfg(), FlatEngine())
    for i in range(5):
        assert c.activate_next_link(np.array([float(i), 0.0, 0.5]))
    c.step(0.01)
    states = c.get_link_states()
    assert sum(1 for l in states if l.active) == 5
    # 5 grounded links, friction = mu*m*g per link, drag opposes +x motion.
    drag = c.get_total_drag_force()
    expected = 0.6 * 0.5 * 1.62 * 5
    assert np.linalg.norm(drag) == pytest.approx(expected, rel=1e-6)
    assert drag[0] < 0.0  # opposes forward (+x) motion
    # Tension grows from rover end toward spool.
    tens = [l.tension_n for l in states if l.active]
    assert tens[0] >= tens[-1]


def test_cable_exhaustion_and_spool():
    c = RigidLinkCableSystem()
    c.initialize(_cable_cfg(total_length_m=3.0), FlatEngine())
    n_ok = sum(
        1 for i in range(10) if c.activate_next_link(np.array([float(i), 0.0, 0.0]))
    )
    assert n_ok == 3  # only 3 links exist
    assert not c.activate_next_link(np.array([20.0, 0.0, 0.0]))


def test_cable_brake_and_tension_fault():
    c = RigidLinkCableSystem()
    c.initialize(_cable_cfg(max_tension_n=1.0), FlatEngine())
    c.command_spool(0.5)
    c.engage_brake()
    assert c.get_spool_state().brake_engaged
    c.command_spool(0.5)  # ignored while braked
    c.step(1.0)
    assert c.get_spool_state().angular_velocity == 0.0
    for i in range(4):
        c.activate_next_link(np.array([float(i), 0.0, 0.0]))
    c.step(0.01)
    assert c.check_tension_fault()  # tiny max_tension -> fault


def test_cable_electrical_state():
    c = RigidLinkCableSystem()
    c.initialize(_cable_cfg(), FlatEngine())
    for i in range(4):
        c.activate_next_link(np.array([float(i), 0.0, 0.0]))
    c.set_electrical_load(10.0)
    es = c.get_electrical_state()
    assert es["voltage_dc"] == pytest.approx(48.0)
    assert es["current_a"] == pytest.approx(10.0)
    # R = 0.01 * (4 links * 1 m) * 2 (round trip) = 0.08 ohm
    assert es["voltage_drop_v"] == pytest.approx(10.0 * 0.08, rel=1e-6)
    assert es["power_w"] == pytest.approx(100.0 * 0.08, rel=1e-6)


def test_cable_bend_radius_fault():
    c = RigidLinkCableSystem()
    c.initialize(_cable_cfg(bend_radius_min_m=5.0), FlatEngine())
    # Sharp right-angle path -> small bend radius -> fault flagged.
    c.activate_next_link(np.array([0.0, 0.0, 0.0]))
    c.activate_next_link(np.array([1.0, 0.0, 0.0]))
    c.activate_next_link(np.array([1.0, 1.0, 0.0]))
    c.activate_next_link(np.array([2.0, 1.0, 0.0]))
    assert len(c.check_bend_radius_fault()) >= 1


# --------------------------------------------------------------------------
# Antenna unit
# --------------------------------------------------------------------------


def _ant_cfg() -> AntennaConfig:
    return AntennaConfig(
        base_plate_m=(0.6, 0.6, 0.05),
        base_mass_kg=4.0,
        mast_height_m=2.0,
        mast_radius_m=0.03,
        mast_mass_kg=3.0,
        dish_diameter_m=0.8,
        dish_mass_kg=2.0,
        connector_mass_kg=0.5,
        total_mass_kg=9.5,
    )


def test_antenna_happy_path_to_active():
    a = DeployableAntennaUnit(_ant_cfg(), FlatEngine())
    assert a.get_state() == AntennaState.STORED
    assert a.transition(AntennaState.GRIPPED)
    assert a.transition(AntennaState.CARRIED)
    a.set_placement(
        position_xy=np.array([5.0, 5.0]),
        tilt_deg=2.0,
        base_contact_corners=4,
        position_error_m=0.1,
        connector_engaged=True,
    )
    assert a.transition(AntennaState.PLACED)
    assert a.transition(AntennaState.DEPLOYED)
    props = a.get_physical_properties()
    assert props["mast_length_m"] == pytest.approx(2.0)
    assert a.transition(AntennaState.ACTIVE)
    assert a.get_state() == AntennaState.ACTIVE
    q = a.evaluate_deployment()
    assert q.status == "full"


def test_antenna_invalid_transition_rejected():
    a = DeployableAntennaUnit(_ant_cfg())
    assert not a.transition(AntennaState.ACTIVE)  # cannot skip from STORED
    assert a.get_state() == AntennaState.STORED


def test_antenna_place_requires_pose():
    a = DeployableAntennaUnit(_ant_cfg())
    a.transition(AntennaState.GRIPPED)
    a.transition(AntennaState.CARRIED)
    assert not a.transition(AntennaState.PLACED)  # no placement set


def test_antenna_failed_blocks_activation():
    a = DeployableAntennaUnit(_ant_cfg())
    a.transition(AntennaState.GRIPPED)
    a.transition(AntennaState.CARRIED)
    a.set_placement(np.array([0.0, 0.0]), tilt_deg=20.0, base_contact_corners=1,
                    position_error_m=0.0, connector_engaged=False)
    a.transition(AntennaState.PLACED)
    a.transition(AntennaState.DEPLOYED)
    assert not a.transition(AntennaState.ACTIVE)  # deployment "failed"
    assert a.evaluate_deployment().status == "failed"


def test_antenna_beacon_only_when_active():
    a = DeployableAntennaUnit(_ant_cfg(), FlatEngine())
    assert a.get_beacon_config() is None
    a.transition(AntennaState.GRIPPED)
    a.transition(AntennaState.CARRIED)
    a.set_placement(np.array([10.0, -3.0]), 1.0, 4, 0.05, connector_engaged=True)
    a.transition(AntennaState.PLACED)
    a.transition(AntennaState.DEPLOYED)
    a.transition(AntennaState.ACTIVE)
    bc = a.get_beacon_config()
    assert bc is not None
    np.testing.assert_allclose(bc.position_xyz, [10.0, -3.0, 2.0])


def test_antenna_fail_from_any_state():
    a = DeployableAntennaUnit(_ant_cfg())
    a.transition(AntennaState.GRIPPED)
    assert a.transition(AntennaState.FAILED)
    assert a.get_state() == AntennaState.FAILED
    assert not a.transition(AntennaState.GRIPPED)  # terminal


# --------------------------------------------------------------------------
# Moonbase
# --------------------------------------------------------------------------


def _mb_cfg(**kw) -> MoonbaseConfig:
    base = dict(
        habitat_dims_m=(6.0, 4.0, 3.0),
        solar_array_config=None,
        power_bus_voltage=48.0,
        comm_tower_height_m=10.0,
        num_docking_bays=2,
        charge_rate_w=500.0,
        num_cable_reels=3,
        num_antennas=5,
        landing_pad_radius_m=20.0,
    )
    base.update(kw)
    return MoonbaseConfig(**base)


def test_moonbase_depot_assignment():
    mb = LunarMoonbase()
    mb.initialize(_mb_cfg(), FlatEngine())
    assert mb.request_cable_reel("rover_1")
    assert not mb.request_cable_reel("rover_1")  # one reel per rover
    assert mb.request_antenna("rover_1")
    assert mb.request_antenna("rover_1")  # multiple antennas allowed
    inv = mb.get_inventory()
    assert inv.cable_reels_available == 2
    assert inv.antennas_available == 3
    assert len(inv.assigned_items["rover_1"]) == 3


def test_moonbase_depot_exhaustion():
    mb = LunarMoonbase()
    mb.initialize(_mb_cfg(num_antennas=1), None)
    assert mb.request_antenna("r1")
    assert not mb.request_antenna("r2")  # depot empty


def test_moonbase_primary_beacon():
    mb = LunarMoonbase()
    mb.initialize(_mb_cfg(), FlatEngine())
    bc = mb.get_primary_beacon()
    np.testing.assert_allclose(bc.position_xyz, [0.0, 0.0, 10.0])
    assert bc.signal_range_m >= 2000.0


def test_moonbase_docking_alignment():
    mb = LunarMoonbase()
    mb.initialize(_mb_cfg(), FlatEngine())
    # Bay 0 is on the +X face; query its exact pose for a clean dock.
    bay = mb._bays[0]
    mb.set_rover_pose("rover_1", bay.position_xyz, bay.heading_deg)
    assert mb.dock_rover("rover_1")
    assert mb.get_charge_state("rover_1") == pytest.approx(0.0)

    # Misaligned rover cannot dock.
    mb.set_rover_pose("rover_2", bay.position_xyz + np.array([2.0, 0.0, 0.0]), 0.0)
    assert not mb.dock_rover("rover_2")
    assert mb.get_charge_state("rover_2") == -1.0


def test_moonbase_charging_and_undock():
    mb = LunarMoonbase()
    mb.initialize(_mb_cfg(charge_rate_w=3600.0), FlatEngine())
    bay = mb._bays[0]
    mb.set_rover_pose("r", bay.position_xyz, bay.heading_deg)
    mb.set_rover_battery_capacity("r", 1.0)  # 1 Wh -> fast charge for test
    assert mb.dock_rover("r")
    mb.step(0.5)  # 3600 W * 0.5 s / (1 Wh*3600) = 0.5
    assert mb.get_charge_state("r") == pytest.approx(0.5, rel=1e-6)
    mb.step(1.0)
    assert mb.get_charge_state("r") == pytest.approx(1.0)  # clamped
    mb.undock_rover("r")
    assert mb.get_charge_state("r") == -1.0
    with pytest.raises(ValueError):
        mb.undock_rover("r")


def test_moonbase_bays_capacity():
    mb = LunarMoonbase()
    mb.initialize(_mb_cfg(num_docking_bays=1), FlatEngine())
    b = mb._bays[0]
    mb.set_rover_pose("a", b.position_xyz, b.heading_deg)
    mb.set_rover_pose("b", b.position_xyz, b.heading_deg)
    assert mb.dock_rover("a")
    assert not mb.dock_rover("b")  # only bay occupied
