"""Concrete mission scenarios for the ScenarioRunner harness.

Provides higher-fidelity, GPU-free :class:`~moon_rover.scenarios.runner.Scenario`
implementations that drive real subsystem state machines (e.g. the antenna
deployment lifecycle) end to end. These are deterministic and run without
Genesis so they can serve as integration-test substrates and RL/dashboard
baselines, while a future real-physics scenario can subclass ``Scenario`` and
reuse the same runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from moon_rover.antenna.system import AntennaConfig, AntennaState, DeployableAntennaUnit
from moon_rover.scenarios.runner import Scenario


def default_antenna_config() -> AntennaConfig:
    """A representative antenna unit (~12 kg assembled)."""
    return AntennaConfig(
        base_plate_m=(0.4, 0.4, 0.05),
        base_mass_kg=4.0,
        mast_height_m=1.5,
        mast_radius_m=0.03,
        mast_mass_kg=3.0,
        dish_diameter_m=0.6,
        dish_mass_kg=4.0,
        connector_mass_kg=1.0,
        total_mass_kg=12.0,
    )


class MissionPhase(Enum):
    PICKUP = "pickup"
    NAVIGATE = "navigate"
    PLACE = "place"
    RETURN = "return"
    DONE = "done"


@dataclass
class _PlacementProfile:
    """Sensed deployment quality applied at a grid point."""

    tilt_deg: float
    base_contact_corners: int
    position_error_m: float
    connector_engaged: bool


# A clean placement that should reach ACTIVE; a bad one that should fail to.
_GOOD_PLACEMENT = _PlacementProfile(2.0, 4, 0.1, True)
_BAD_PLACEMENT = _PlacementProfile(12.0, 1, 0.8, False)


class MissionPlacementScenario(Scenario):
    """Full antenna-placement mission, driven through the real antenna lifecycle.

    Flow: spawn at moonbase -> for each grid point {pick up antenna, navigate to
    the point, place + deploy + activate the antenna} -> return to base. Each
    antenna is a real :class:`DeployableAntennaUnit`, so the test asserts the
    actual ``STORED -> GRIPPED -> CARRIED -> PLACED -> DEPLOYED -> ACTIVE``
    transitions rather than a mock.

    Config keys (all optional):
        base_position     [x, y, z]            moonbase origin (default [0,0,0])
        grid_points       [[x, y, z], ...]     antenna target sites
        dt                float                control tick (default 0.05 -> 20 Hz)
        drive_speed_mps   float                cruise speed (default 1.0)
        arrive_tol_m      float                arrival threshold (default 0.2)
        pickup_ticks      int                  ticks to grasp/stow (default 5)
        place_ticks       int                  ticks to place/deploy (default 5)
        fail_indices      [int, ...]           grid indices that mis-deploy
        antenna_config    AntennaConfig        override antenna geometry
    """

    rover_type = "mission_placement_diff_drive"

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = dict(config or {})
        self.base_position = np.asarray(cfg.get("base_position", [0.0, 0.0, 0.0]), dtype=np.float64)
        grid = cfg.get("grid_points", [[5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [0.0, 5.0, 0.0]])
        self.grid_points = [np.asarray(p, dtype=np.float64) for p in grid]
        self.dt = float(cfg.get("dt", 0.05))
        self.drive_speed_mps = float(cfg.get("drive_speed_mps", 1.0))
        self.arrive_tol_m = float(cfg.get("arrive_tol_m", 0.2))
        self.pickup_ticks = int(cfg.get("pickup_ticks", 5))
        self.place_ticks = int(cfg.get("place_ticks", 5))
        self.fail_indices = set(int(i) for i in cfg.get("fail_indices", []))
        self.antenna_config = cfg.get("antenna_config") or default_antenna_config()

        self._rng: Optional[np.random.Generator] = None
        self._pos = self.base_position.copy()
        self._sim_time = 0.0
        self._energy_wh = 0.0
        self._phase = MissionPhase.DONE
        self._idx = 0
        self._phase_ticks = 0
        self._antennas: list[DeployableAntennaUnit] = []
        self._placements: list[dict] = []
        self._faults: list[dict] = []
        self._events: list[dict] = []
        self._returned = False

    # -- lifecycle ----------------------------------------------------------

    def setup(self, seed: int, *, visualize: bool = False) -> None:
        self._rng = np.random.default_rng(seed)
        self._pos = self.base_position.copy()
        self._sim_time = 0.0
        self._energy_wh = 0.0
        self._idx = 0
        self._phase_ticks = 0
        self._antennas = []
        self._placements = []
        self._faults = []
        self._returned = False
        self._events = [{"event_type": "mission_start", "sim_time": 0.0, "payload": {"seed": seed}}]
        self._phase = MissionPhase.PICKUP if self.grid_points else MissionPhase.RETURN

    def step(self) -> dict:
        assert self._rng is not None, "setup() must be called before step()"
        moving = False

        if self._phase == MissionPhase.PICKUP:
            self._do_pickup()
        elif self._phase == MissionPhase.NAVIGATE:
            moving = self._drive_toward(self.grid_points[self._idx])
            if not moving:
                self._phase = MissionPhase.PLACE
                self._phase_ticks = 0
        elif self._phase == MissionPhase.PLACE:
            self._do_place()
        elif self._phase == MissionPhase.RETURN:
            moving = self._drive_toward(self.base_position)
            if not moving:
                self._returned = True
                self._phase = MissionPhase.DONE
                self._events.append(
                    {"event_type": "returned_to_base", "sim_time": self._sim_time, "payload": {}}
                )

        # Energy/time bookkeeping: idle draw + drive draw.
        power_w = 60.0 + (120.0 if moving else 0.0)
        self._energy_wh += power_w * (self.dt / 3600.0)
        self._sim_time += self.dt

        gt = self._pos.copy()
        est = gt + self._rng.normal(scale=0.04, size=3)
        cable_tension = 20.0 + 8.0 * float(np.linalg.norm(self._pos - self.base_position))
        n_active = sum(1 for a in self._antennas if a.get_state() == AntennaState.ACTIVE)
        coverage = n_active / max(1, len(self.grid_points))

        return {
            "timestamp": self._sim_time,
            "rover_position": self._pos.tolist(),
            "velocity": [0.0, 0.0, 0.0],
            "energy_wh": self._energy_wh,
            "power_consumed_w": power_w,
            "cable_tension_n": cable_tension,
            "cable_coverage_fraction": coverage,
            "estimated_position": est.tolist(),
            "ground_truth_position": gt.tolist(),
            "imu": {
                "accel_xyz": [0.0, 0.0, -1.62],
                "gyro_xyz": [0.0, 0.0, 0.0],
                "timestamp": self._sim_time,
            },
        }

    def is_complete(self) -> bool:
        return self._phase == MissionPhase.DONE

    def teardown(self) -> None:
        self._events.append(
            {"event_type": "mission_end", "sim_time": self._sim_time, "payload": {}}
        )

    # -- phase handlers -----------------------------------------------------

    def _do_pickup(self) -> None:
        if self._phase_ticks == 0:
            antenna = DeployableAntennaUnit(self.antenna_config)
            antenna.transition(AntennaState.GRIPPED)
            antenna.transition(AntennaState.CARRIED)
            self._antennas.append(antenna)
            self._events.append(
                {
                    "event_type": "antenna_picked",
                    "sim_time": self._sim_time,
                    "payload": {"antenna_id": f"antenna_{self._idx:02d}"},
                }
            )
        self._phase_ticks += 1
        if self._phase_ticks >= self.pickup_ticks:
            self._phase = MissionPhase.NAVIGATE
            self._phase_ticks = 0

    def _do_place(self) -> None:
        if self._phase_ticks == 0:
            antenna = self._antennas[self._idx]
            point = self.grid_points[self._idx]
            profile = _BAD_PLACEMENT if self._idx in self.fail_indices else _GOOD_PLACEMENT
            antenna.set_placement(
                position_xy=point[:2],
                tilt_deg=profile.tilt_deg,
                base_contact_corners=profile.base_contact_corners,
                position_error_m=profile.position_error_m,
                connector_engaged=profile.connector_engaged,
            )
            antenna.transition(AntennaState.PLACED)
            antenna.transition(AntennaState.DEPLOYED)
            activated = antenna.transition(AntennaState.ACTIVE)
            success = antenna.get_state() == AntennaState.ACTIVE
            antenna_id = f"antenna_{self._idx:02d}"
            self._placements.append(
                {
                    "antenna_id": antenna_id,
                    "target": point.tolist(),
                    "actual": self._pos.tolist(),
                    "success": success,
                    "sim_time": self._sim_time,
                    "failure_mode": None if success else "deployment_failed",
                }
            )
            if not success:
                self._faults.append(
                    {"mode": "deployment_failed", "time": self._sim_time, "antenna_id": antenna_id}
                )
            self._events.append(
                {
                    "event_type": "antenna_activated" if activated else "antenna_deploy_failed",
                    "sim_time": self._sim_time,
                    "payload": {"antenna_id": antenna_id, "state": antenna.get_state().value},
                }
            )
        self._phase_ticks += 1
        if self._phase_ticks >= self.place_ticks:
            self._idx += 1
            self._phase_ticks = 0
            if self._idx < len(self.grid_points):
                self._phase = MissionPhase.PICKUP
            else:
                self._phase = MissionPhase.RETURN

    def _drive_toward(self, target: np.ndarray) -> bool:
        """Advance toward target. Returns True while still moving."""
        delta = target - self._pos
        dist = float(np.linalg.norm(delta))
        if dist <= self.arrive_tol_m:
            return False
        step = min(self.drive_speed_mps * self.dt, dist)
        self._pos = self._pos + (delta / dist) * step
        return True

    # -- introspection (for assertions) ------------------------------------

    @property
    def antenna_units(self) -> list[DeployableAntennaUnit]:
        return self._antennas

    def antenna_states(self) -> list[AntennaState]:
        return [a.get_state() for a in self._antennas]

    @property
    def returned_to_base(self) -> bool:
        return self._returned

    @property
    def antenna_placements(self) -> list[dict]:
        return self._placements

    @property
    def faults(self) -> list[dict]:
        return self._faults

    @property
    def events(self) -> list[dict]:
        return self._events

    def succeeded(self) -> bool:
        if not self._antennas or len(self._antennas) < len(self.grid_points):
            return False
        all_active = all(a.get_state() == AntennaState.ACTIVE for a in self._antennas)
        return all_active and self._returned


def mission_placement_factory(config: dict) -> Scenario:
    """Scenario factory for :class:`MissionPlacementScenario`."""
    return MissionPlacementScenario(config)
