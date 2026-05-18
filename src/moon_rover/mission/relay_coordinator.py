"""Concrete MultiRoverCoordinator: shared world model + cable-aware coordination.

Implements :class:`MultiRoverCoordinator` for missions where every rover talks
to its peers through the moonbase relay. Responsibilities:

* maintain a coarse shared occupancy grid fed by every rover's reported pose,
* track each rover's deployed cable corridor (depot -> placed antenna) and
  validate that a proposed path keeps the documented 1.5 m clearance,
* model realistic relay-mediated communication delay (rover -> base -> rover),
* and produce a concrete deadlock-resolution plan (yield priority + target
  reassignment) for a stuck pair.

Design choices
--------------
* The shared world model is a 2-D occupancy grid (XY, fixed cell size). It is
  intentionally lightweight: it records "a rover was here" and "a cable runs
  here", which is what path-clearance and progress queries actually need. Dense
  lidar fusion belongs in :mod:`moon_rover.navigation.perception`, not here.
* Communication delay is modelled as two relay hops (sender -> base -> receiver)
  each with a base latency plus a small distance term, and an asymmetric penalty
  when a rover's reported antenna orientation is poor. ``'base'`` as sender or
  receiver collapses to a single hop.
* :meth:`resolve_deadlock` is deterministic: the rover with fewer remaining
  targets (then the lexicographically smaller id) is told to yield, and the
  pair's *next* targets are swapped only if that strictly reduces the summed
  distance-to-next — otherwise a wait instruction is issued instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from moon_rover.mission.coordinator import MultiRoverCoordinator, RoverStatus

#: One-way base relay latency (s) for a single hop, before distance term.
_BASE_RELAY_LATENCY_S = 0.025
#: Extra one-way delay per metre of link distance (s/m), ~speed-of-light + proc.
_DELAY_PER_METER_S = 5e-7
#: Multiplicative penalty applied to a hop when the sender's antenna is poorly
#: oriented (degraded link budget).
_BAD_ANTENNA_PENALTY = 2.5
#: Clearance (m) every planned path must keep from any deployed cable.
_CABLE_CLEARANCE_M = 1.5
#: Default occupancy-grid cell size (m).
_DEFAULT_CELL_M = 1.0


@dataclass
class _SharedWorldModel:
    """Coarse XY occupancy grid shared across all rovers.

    Cells store an integer code: 0 free, 1 rover-observed, 2 cable corridor.
    Unbounded in practice via a dict keyed by integer cell coordinates so the
    grid never needs a fixed extent.
    """

    cell_size_m: float = _DEFAULT_CELL_M
    cells: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def _key(self, xy: npt.NDArray[np.float64]) -> Tuple[int, int]:
        return (
            int(math.floor(xy[0] / self.cell_size_m)),
            int(math.floor(xy[1] / self.cell_size_m)),
        )

    def mark(self, xy: npt.NDArray[np.float64], code: int) -> None:
        key = self._key(xy)
        # Cable corridor (2) outranks rover-observed (1).
        self.cells[key] = max(self.cells.get(key, 0), code)

    def occupancy_at(self, xy: npt.NDArray[np.float64]) -> int:
        return self.cells.get(self._key(xy), 0)


@dataclass
class _RoverRecord:
    drive_type: str
    status: Optional[RoverStatus] = None
    #: Deployed cable corridors as (start_xyz, end_xyz) segments from the depot.
    cables: List[Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]] = field(
        default_factory=list
    )
    #: Most recent reported antenna-orientation quality [0, 1] (1 = ideal).
    antenna_quality: float = 1.0


class RelayCoordinator(MultiRoverCoordinator):
    """Production multi-rover coordinator using the moonbase as comms relay.

    Args:
        depot_xyz: Moonbase/relay position ``[x, y, z]``. Cable corridors run
            from here; relay hops are computed through here. Defaults to origin.
        cell_size_m: Shared occupancy-grid cell size in metres.
    """

    def __init__(
        self,
        depot_xyz: Optional[npt.NDArray[np.float64]] = None,
        cell_size_m: float = _DEFAULT_CELL_M,
    ) -> None:
        self._depot = (
            np.asarray(depot_xyz, dtype=np.float64).reshape(3)
            if depot_xyz is not None
            else np.zeros(3, dtype=np.float64)
        )
        self._rovers: Dict[str, _RoverRecord] = {}
        self._world = _SharedWorldModel(cell_size_m=cell_size_m)

    # ------------------------------------------------------------------ #
    # Registration & status
    # ------------------------------------------------------------------ #

    def register_rover(self, rover_id: str, drive_type: str) -> None:
        if drive_type not in ("diff_drive", "skid_steer"):
            # Accept but normalise unknown types to skid_steer behaviour.
            drive_type = "skid_steer"
        self._rovers[rover_id] = _RoverRecord(drive_type=drive_type)

    def update_rover_status(self, status: RoverStatus) -> None:
        rec = self._rovers.get(status.rover_id)
        if rec is None:
            # Auto-register on first sighting so the coordinator is robust to
            # ordering of register/update calls.
            rec = _RoverRecord(drive_type="skid_steer")
            self._rovers[status.rover_id] = rec
        rec.status = status
        pos = np.asarray(status.position, dtype=np.float64).reshape(3)
        self._world.mark(pos[:2], code=1)

    def report_cable_segment(
        self,
        rover_id: str,
        antenna_xyz: npt.NDArray[np.float64],
    ) -> None:
        """Record a deployed cable corridor (depot -> ``antenna_xyz``).

        Called by the mission layer when a rover finishes ``CABLE_CONNECT`` so
        peers can route around the new corridor. Stamps the corridor into the
        shared world model.
        """
        rec = self._rovers.setdefault(rover_id, _RoverRecord(drive_type="skid_steer"))
        end = np.asarray(antenna_xyz, dtype=np.float64).reshape(3)
        rec.cables.append((self._depot.copy(), end))
        # Stamp the corridor into the occupancy grid at sub-cell resolution.
        length = float(np.linalg.norm(end - self._depot))
        n = max(2, int(length / max(self._world.cell_size_m, 1e-6)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            p = self._depot + t * (end - self._depot)
            self._world.mark(p[:2], code=2)

    def set_antenna_quality(self, rover_id: str, quality: float) -> None:
        """Update a rover's reported antenna-orientation quality [0, 1]."""
        rec = self._rovers.setdefault(rover_id, _RoverRecord(drive_type="skid_steer"))
        rec.antenna_quality = float(np.clip(quality, 0.0, 1.0))

    def get_shared_world_model(self) -> object:
        """Return the shared occupancy model (XY grid of rover/cable cells)."""
        return self._world

    # ------------------------------------------------------------------ #
    # Cable-crossing check
    # ------------------------------------------------------------------ #

    def check_cable_crossing(
        self,
        rover_id: str,
        planned_path: List[npt.NDArray[np.float64]],
    ) -> bool:
        """Return ``True`` if the path stays clear of *other* rovers' cables.

        Each consecutive pair of waypoints is a segment; the path is rejected if
        any segment comes within :data:`_CABLE_CLEARANCE_M` of any cable
        corridor deployed by a *different* rover (a rover may run alongside its
        own cable).
        """
        if len(planned_path) < 2:
            return True
        path = [np.asarray(p, dtype=np.float64).reshape(3)[:2] for p in planned_path]

        for other_id, rec in self._rovers.items():
            if other_id == rover_id:
                continue
            for c_start, c_end in rec.cables:
                cs, ce = c_start[:2], c_end[:2]
                for a, b in zip(path[:-1], path[1:]):
                    if (
                        _segment_segment_distance(a, b, cs, ce)
                        < _CABLE_CLEARANCE_M
                    ):
                        return False
        return True

    # ------------------------------------------------------------------ #
    # Communication delay model
    # ------------------------------------------------------------------ #

    def get_communication_delay(self, sender: str, receiver: str) -> float:
        """One-way delay (s) for ``sender -> receiver`` via the base relay.

        Rover-to-rover traffic is two hops (sender -> base -> receiver); traffic
        to or from ``'base'`` is a single hop. Each hop is a base latency plus a
        small per-metre term, multiplied by an antenna penalty when the sending
        endpoint reports poor antenna orientation.
        """
        if sender == receiver:
            return 0.0

        def endpoint_xy(name: str) -> npt.NDArray[np.float64]:
            if name == "base":
                return self._depot[:2]
            rec = self._rovers.get(name)
            if rec is None or rec.status is None:
                return self._depot[:2]
            return np.asarray(rec.status.position, dtype=np.float64).reshape(3)[:2]

        def hop(a: str, b: str) -> float:
            dist = float(np.linalg.norm(endpoint_xy(a) - endpoint_xy(b)))
            base = _BASE_RELAY_LATENCY_S + dist * _DELAY_PER_METER_S
            rec = self._rovers.get(a)
            if rec is not None and rec.antenna_quality < 0.5:
                base *= _BAD_ANTENNA_PENALTY
            return base

        if sender == "base" or receiver == "base":
            return hop(sender, receiver)
        # Two relay hops through the moonbase.
        return hop(sender, "base") + hop("base", receiver)

    # ------------------------------------------------------------------ #
    # Deadlock resolution
    # ------------------------------------------------------------------ #

    def resolve_deadlock(self, rover_a: str, rover_b: str) -> dict:
        """Deterministic resolution plan for a deadlocked pair.

        The rover with fewer remaining targets (tie broken by smaller id) is
        assigned ``priority`` to move first; the other yields. If swapping the
        two rovers' *current* targets strictly reduces the combined
        distance-to-next-target, the swap is included; otherwise the yielding
        rover is given an explicit wait instruction and targets are unchanged.

        Returns:
            ``{'rover_a_targets': [...], 'rover_b_targets': [...],
               'priority': <rover_id that moves first>,
               'instruction': {'<yield_id>': 'wait'|'reroute'}}``
        """
        ra = self._rovers.get(rover_a)
        rb = self._rovers.get(rover_b)
        if ra is None or rb is None or ra.status is None or rb.status is None:
            raise KeyError("both rovers must be registered with known status")

        tgt_a = ra.status.current_target
        tgt_b = rb.status.current_target
        pos_a = np.asarray(ra.status.position, dtype=np.float64).reshape(3)
        pos_b = np.asarray(rb.status.position, dtype=np.float64).reshape(3)

        # Yield priority: the rover that is "further along" (closer to its
        # target) keeps moving; deterministic tie-break on id.
        da = _target_distance(pos_a, tgt_a)
        db = _target_distance(pos_b, tgt_b)
        if da < db or (da == db and rover_a < rover_b):
            mover, yielder = rover_a, rover_b
        else:
            mover, yielder = rover_b, rover_a

        result_a = [tgt_a] if tgt_a is not None else []
        result_b = [tgt_b] if tgt_b is not None else []
        instruction = {yielder: "wait"}

        # Consider swapping current targets if it strictly shortens total travel.
        if tgt_a is not None and tgt_b is not None:
            cur_cost = _target_distance(pos_a, tgt_a) + _target_distance(
                pos_b, tgt_b
            )
            swap_cost = _target_distance(pos_a, tgt_b) + _target_distance(
                pos_b, tgt_a
            )
            if swap_cost + 1e-6 < cur_cost:
                result_a, result_b = [tgt_b], [tgt_a]
                instruction = {yielder: "reroute"}

        return {
            "rover_a_targets": result_a,
            "rover_b_targets": result_b,
            "priority": mover,
            "instruction": instruction,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _target_distance(pos: npt.NDArray[np.float64], target: object) -> float:
    """Euclidean XY distance from ``pos`` to a GridPoint-like ``target``."""
    if target is None:
        return math.inf
    tp = getattr(target, "position_xyz", None)
    if tp is None:
        return math.inf
    tp = np.asarray(tp, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(pos[:2] - tp[:2]))


def _segment_segment_distance(
    p1: npt.NDArray[np.float64],
    p2: npt.NDArray[np.float64],
    q1: npt.NDArray[np.float64],
    q2: npt.NDArray[np.float64],
) -> float:
    """Minimum distance between 2-D segments ``p1p2`` and ``q1q2``."""
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(p1 - q1))
    if a <= 1e-12:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= 1e-12:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            s = (
                float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
                if denom > 1e-12
                else 0.0
            )
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    closest_p = p1 + s * d1
    closest_q = q1 + t * d2
    return float(np.linalg.norm(closest_p - closest_q))
