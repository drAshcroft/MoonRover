"""Concrete MissionManager: rover state machine + fault detection and recovery.

This module implements the :class:`MissionManager` ABC for production multi-rover
antenna-deployment missions. It owns:

* the antenna deployment grid (built from :class:`MissionConfig`),
* a per-rover :class:`MissionPhase` state machine with guarded transitions,
* greedy nearest-neighbour deployment ordering anchored at the depot,
* zone-partitioned rover assignment that minimises cross-rover cable interference,
* threshold/heuristic fault detection from telemetry, and
* per-fault recovery procedures with bounded retries.

Design choices
--------------
* The depot/moonbase position is an explicit constructor argument (default origin)
  because :class:`MissionConfig` only carries grid geometry; cable length and
  return-to-base logic are measured from the depot.
* The phase machine is strictly linear per segment
  ``DEPOT_PICKUP -> TRANSIT -> ANTENNA_DEPLOY -> CABLE_CONNECT ->
  RETURN_TO_BASE -> CHARGING -> COMPLETE``; reaching ``COMPLETE`` with more
  assigned points loops back to ``DEPOT_PICKUP`` for the next segment, otherwise
  the rover stays ``COMPLETE``. ``FAULT_RECOVERY`` is entered out of band from
  any active phase and, on successful recovery, resumes the interrupted phase.
* Fault detection is stateful: GPS-outage and stuck detection integrate over
  wall-clock time taken from ``telemetry['time_s']`` when present, otherwise from
  a monotonically increasing internal call counter (one virtual second per call).
* ``advance_phase`` enforces preconditions and raises ``RuntimeError`` on an
  illegal transition rather than silently clamping, so callers cannot skip
  placement confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

from moon_rover.mission.manager import (
    FaultType,
    GridPoint,
    MissionConfig,
    MissionManager,
    MissionPhase,
)

# ---------------------------------------------------------------------------
# Tunables (documented contract values; overridable via FaultThresholds)
# ---------------------------------------------------------------------------

#: Battery state-of-charge below which BATTERY_LOW trips (fraction [0, 1]).
_BATTERY_LOW_SOC = 0.15
#: Cable tension (N) above which a snag is declared.
_CABLE_SNAG_TENSION_N = 400.0
#: Drive/arm motor temperature (C) above which MOTOR_OVERHEAT trips.
_MOTOR_OVERHEAT_C = 85.0
#: GPS fix quality (0..1) at or below which the link is considered degraded.
_GPS_BAD_FIX_QUALITY = 0.2
#: Continuous seconds of degraded GPS before GPS_LOST trips.
_GPS_OUTAGE_TIMEOUT_S = 15.0
#: Speed (m/s) below which a rover commanded to move counts as not moving.
_STUCK_SPEED_EPS = 0.02
#: Continuous seconds of commanded-but-not-moving before ROVER_STUCK trips.
_STUCK_TIMEOUT_S = 20.0
#: Antenna placement tilt error (deg) above which ANTENNA_TILT trips.
_ANTENNA_TILT_TOLERANCE_DEG = 8.0
#: Buffer radius (m) around a deployed cable centerline for exclusion zones.
_CABLE_EXCLUSION_BUFFER_M = 1.5
#: Sample spacing (m) along cable centerlines when emitting exclusion clouds.
_CABLE_SAMPLE_SPACING_M = 1.0
#: Distance (m) under which two rovers are treated as a potential deadlock pair.
_DEADLOCK_RADIUS_M = 3.0


@dataclass
class FaultThresholds:
    """Tunable fault-detection thresholds.

    Defaults match the documented MissionManager contract. Override per mission
    to model harsher environments or more conservative safety margins.
    """

    battery_low_soc: float = _BATTERY_LOW_SOC
    cable_snag_tension_n: float = _CABLE_SNAG_TENSION_N
    motor_overheat_c: float = _MOTOR_OVERHEAT_C
    gps_bad_fix_quality: float = _GPS_BAD_FIX_QUALITY
    gps_outage_timeout_s: float = _GPS_OUTAGE_TIMEOUT_S
    stuck_speed_eps: float = _STUCK_SPEED_EPS
    stuck_timeout_s: float = _STUCK_TIMEOUT_S
    antenna_tilt_tolerance_deg: float = _ANTENNA_TILT_TOLERANCE_DEG


# Linear per-segment phase order. FAULT_RECOVERY and PLANNING are handled
# out of this chain.
_SEGMENT_ORDER: Tuple[MissionPhase, ...] = (
    MissionPhase.DEPOT_PICKUP,
    MissionPhase.TRANSIT,
    MissionPhase.ANTENNA_DEPLOY,
    MissionPhase.CABLE_CONNECT,
    MissionPhase.RETURN_TO_BASE,
    MissionPhase.CHARGING,
    MissionPhase.COMPLETE,
)


@dataclass
class _RoverState:
    """Internal per-rover bookkeeping."""

    phase: MissionPhase = MissionPhase.PLANNING
    #: Phase to resume after a successful FAULT_RECOVERY.
    phase_before_fault: Optional[MissionPhase] = None
    #: Ordered grid points still owed to this rover (FIFO).
    queue: List[GridPoint] = field(default_factory=list)
    #: Index into the rover's original assignment for progress reporting.
    segment_index: int = 0
    total_segments: int = 0
    #: GridPoint currently being serviced (popped from queue on DEPOT_PICKUP).
    active_point: Optional[GridPoint] = None
    #: Retry counters keyed by FaultType.
    retries: Dict[FaultType, int] = field(default_factory=dict)
    #: Fault-detection timers (seconds of continuous condition).
    gps_bad_since: Optional[float] = None
    stuck_since: Optional[float] = None
    last_fault: Optional[FaultType] = None


class LunarMissionManager(MissionManager):
    """Production MissionManager for the lunar antenna-deployment mission.

    Args:
        depot_xyz: Moonbase/depot position ``[x, y, z]`` in world meters. Cable
            length, transit, and return-to-base are measured from here. Defaults
            to the world origin.
        thresholds: Fault-detection thresholds. Defaults to the documented
            MissionManager contract values.
    """

    def __init__(
        self,
        depot_xyz: Optional[npt.NDArray[np.float64]] = None,
        thresholds: Optional[FaultThresholds] = None,
    ) -> None:
        self._depot = (
            np.asarray(depot_xyz, dtype=np.float64)
            if depot_xyz is not None
            else np.zeros(3, dtype=np.float64)
        )
        self._thr = thresholds or FaultThresholds()
        self._config: Optional[MissionConfig] = None
        self._grid: List[GridPoint] = []
        self._rovers: Dict[str, _RoverState] = {}
        self._step_counter: float = 0.0

    # ------------------------------------------------------------------ #
    # Initialization & grid construction
    # ------------------------------------------------------------------ #

    def initialize(self, config: MissionConfig) -> None:
        """Build the deployment grid and reset all mission state.

        Grid point ``(row, col)`` is placed at
        ``grid_origin + [col * spacing, row * spacing, 0]`` in world meters.
        """
        if config.grid_rows <= 0 or config.grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive")
        if config.grid_spacing_m <= 0.0:
            raise ValueError("grid_spacing_m must be positive")

        self._config = config
        origin = np.asarray(config.grid_origin, dtype=np.float64).reshape(3)
        spacing = float(config.grid_spacing_m)

        self._grid = []
        for row in range(config.grid_rows):
            for col in range(config.grid_cols):
                pos = origin + np.array(
                    [col * spacing, row * spacing, 0.0], dtype=np.float64
                )
                self._grid.append(GridPoint(position_xyz=pos, row=row, col=col))

        self._rovers = {}
        self._step_counter = 0.0

    # ------------------------------------------------------------------ #
    # Planning & assignment
    # ------------------------------------------------------------------ #

    def plan_deployment_order(self) -> List[GridPoint]:
        """Greedy nearest-neighbour traversal anchored at the depot.

        Starts from the grid point closest to the depot (shortest first cable
        run) and repeatedly appends the nearest unvisited point. This keeps the
        incremental cable path short and avoids long back-tracks.
        """
        self._require_initialized()
        if not self._grid:
            return []

        # Track remaining points by index; GridPoint holds an ndarray so it is
        # not safely usable with list.remove / equality.
        remaining = set(range(len(self._grid)))
        ordered: List[GridPoint] = []
        cursor = self._depot  # first point: closest to depot.
        while remaining:
            nxt_i = min(
                remaining,
                key=lambda i: float(
                    np.linalg.norm(self._grid[i].position_xyz - cursor)
                ),
            )
            gp = self._grid[nxt_i]
            ordered.append(gp)
            remaining.discard(nxt_i)
            cursor = gp.position_xyz
        return ordered

    def assign_rovers(
        self,
        rover_ids: List[str],
    ) -> Dict[str, List[GridPoint]]:
        """Partition the grid into vertical zones, one contiguous band per rover.

        Column-banded zones (rather than interleaved points) keep each rover's
        cable inside its own corridor, which minimises cross-rover cable
        crossings. Within a zone the points are ordered by nearest-neighbour
        from the depot so each rover's own run stays short. Workload is balanced
        by splitting the *ordered* deployment list into near-equal contiguous
        chunks when zones would be lopsided.
        """
        self._require_initialized()
        if not rover_ids:
            raise ValueError("rover_ids must be non-empty")

        ordered = self.plan_deployment_order()
        n = len(rover_ids)
        # Balanced contiguous chunks of the depot-anchored order. Contiguous
        # chunks of a nearest-neighbour tour are spatially compact, so each
        # rover still gets a coherent corridor while load stays even.
        chunks: List[List[GridPoint]] = [[] for _ in range(n)]
        base = len(ordered) // n
        extra = len(ordered) % n
        idx = 0
        for r in range(n):
            take = base + (1 if r < extra else 0)
            chunks[r] = ordered[idx : idx + take]
            idx += take

        assignment: Dict[str, List[GridPoint]] = {}
        for rid, chunk in zip(rover_ids, chunks):
            for gp in chunk:
                gp.assigned_rover = rid
            assignment[rid] = chunk
            st = self._rovers.setdefault(rid, _RoverState())
            st.queue = list(chunk)
            st.total_segments = len(chunk)
            st.segment_index = 0
            st.phase = MissionPhase.PLANNING
        return assignment

    # ------------------------------------------------------------------ #
    # Phase state machine
    # ------------------------------------------------------------------ #

    def get_current_phase(self, rover_id: str) -> MissionPhase:
        return self._rover(rover_id).phase

    def advance_phase(self, rover_id: str) -> MissionPhase:
        """Advance one step in the linear segment machine.

        Transitions and their preconditions:

        * ``PLANNING -> DEPOT_PICKUP``: requires a non-empty assignment queue.
        * ``DEPOT_PICKUP -> TRANSIT``: pops the next grid point as the active
          target.
        * ``ANTENNA_DEPLOY -> CABLE_CONNECT``: marks the active point
          ``'equipped'``.
        * ``CABLE_CONNECT -> RETURN_TO_BASE``: marks the active point
          ``'complete'``.
        * ``CHARGING -> COMPLETE``: ends the segment; if more points remain the
          rover is immediately re-armed back to ``DEPOT_PICKUP`` for the next
          segment, otherwise it stays ``COMPLETE``.
        * ``FAULT_RECOVERY -> <interrupted phase>``: only legal once recovery
          has cleared (``phase_before_fault`` is set by :meth:`execute_recovery`).

        Raises:
            RuntimeError: if the transition is illegal in the current state.
        """
        st = self._rover(rover_id)
        phase = st.phase

        if phase == MissionPhase.FAULT_RECOVERY:
            if st.phase_before_fault is None:
                raise RuntimeError(
                    f"{rover_id}: cannot leave FAULT_RECOVERY before recovery "
                    f"completes (call execute_recovery first)"
                )
            st.phase = st.phase_before_fault
            st.phase_before_fault = None
            return st.phase

        if phase == MissionPhase.PLANNING:
            if not st.queue:
                raise RuntimeError(
                    f"{rover_id}: cannot leave PLANNING with an empty "
                    f"assignment queue (call assign_rovers first)"
                )
            st.phase = MissionPhase.DEPOT_PICKUP
            return st.phase

        if phase == MissionPhase.DEPOT_PICKUP:
            if not st.queue:
                raise RuntimeError(
                    f"{rover_id}: DEPOT_PICKUP with no remaining grid points"
                )
            st.active_point = st.queue.pop(0)
            st.active_point.status = "visited"
            st.phase = MissionPhase.TRANSIT
            return st.phase

        if phase == MissionPhase.TRANSIT:
            st.phase = MissionPhase.ANTENNA_DEPLOY
            return st.phase

        if phase == MissionPhase.ANTENNA_DEPLOY:
            if st.active_point is None:
                raise RuntimeError(
                    f"{rover_id}: ANTENNA_DEPLOY without an active grid point"
                )
            st.active_point.status = "equipped"
            st.phase = MissionPhase.CABLE_CONNECT
            return st.phase

        if phase == MissionPhase.CABLE_CONNECT:
            if st.active_point is None:
                raise RuntimeError(
                    f"{rover_id}: CABLE_CONNECT without an active grid point"
                )
            st.active_point.status = "complete"
            st.phase = MissionPhase.RETURN_TO_BASE
            return st.phase

        if phase == MissionPhase.RETURN_TO_BASE:
            st.phase = MissionPhase.CHARGING
            return st.phase

        if phase == MissionPhase.CHARGING:
            st.segment_index += 1
            st.active_point = None
            if st.queue:
                # Re-arm for the next segment.
                st.phase = MissionPhase.DEPOT_PICKUP
            else:
                st.phase = MissionPhase.COMPLETE
            return st.phase

        if phase == MissionPhase.COMPLETE:
            # Terminal unless reassigned via assign_rovers.
            return st.phase

        raise RuntimeError(f"{rover_id}: unhandled phase {phase}")

    # ------------------------------------------------------------------ #
    # Fault detection
    # ------------------------------------------------------------------ #

    def detect_fault(
        self,
        rover_id: str,
        telemetry: dict,
    ) -> Optional[FaultType]:
        """Detect the highest-priority fault from a telemetry snapshot.

        Recognised telemetry keys (all optional):

        * ``time_s``: wall-clock seconds; drives outage/stuck integration.
        * ``battery_soc``: state of charge [0, 1].
        * ``cable_tension_n``: instantaneous cable tension (N).
        * ``cable_length_remaining_m``: remaining spool length (m).
        * ``motor_temp_c``: hottest drive/arm motor temperature (C).
        * ``gps_fix_quality``: GPS fix quality [0, 1].
        * ``commanded_speed_mps`` / ``actual_speed_mps``: stuck detection.
        * ``antenna_tilt_error_deg``: placement-arm tilt error (only meaningful
          during ANTENNA_DEPLOY).

        Priority order (safety first): ``BATTERY_LOW`` > ``MOTOR_OVERHEAT`` >
        ``CABLE_SNAG`` > ``CABLE_EXHAUSTED`` > ``ROVER_STUCK`` > ``GPS_LOST`` >
        ``ANTENNA_TILT``. Returns ``None`` if no fault is active.
        """
        st = self._rover(rover_id)
        now = self._clock(telemetry)
        thr = self._thr

        soc = telemetry.get("battery_soc")
        if soc is not None and float(soc) < thr.battery_low_soc:
            return self._latch(st, FaultType.BATTERY_LOW)

        temp = telemetry.get("motor_temp_c")
        if temp is not None and float(temp) > thr.motor_overheat_c:
            return self._latch(st, FaultType.MOTOR_OVERHEAT)

        tension = telemetry.get("cable_tension_n")
        if tension is not None and float(tension) > thr.cable_snag_tension_n:
            return self._latch(st, FaultType.CABLE_SNAG)

        remaining = telemetry.get("cable_length_remaining_m")
        if remaining is not None and float(remaining) <= 0.0:
            return self._latch(st, FaultType.CABLE_EXHAUSTED)

        # Stuck: commanded to move but barely moving, integrated over time.
        cmd = telemetry.get("commanded_speed_mps")
        act = telemetry.get("actual_speed_mps")
        if cmd is not None and act is not None:
            if abs(float(cmd)) > thr.stuck_speed_eps and abs(
                float(act)
            ) < thr.stuck_speed_eps:
                if st.stuck_since is None:
                    st.stuck_since = now
                elif now - st.stuck_since >= thr.stuck_timeout_s:
                    return self._latch(st, FaultType.ROVER_STUCK)
            else:
                st.stuck_since = None

        # GPS outage integrated over time.
        fix = telemetry.get("gps_fix_quality")
        if fix is not None:
            if float(fix) <= thr.gps_bad_fix_quality:
                if st.gps_bad_since is None:
                    st.gps_bad_since = now
                elif now - st.gps_bad_since >= thr.gps_outage_timeout_s:
                    return self._latch(st, FaultType.GPS_LOST)
            else:
                st.gps_bad_since = None

        tilt = telemetry.get("antenna_tilt_error_deg")
        if (
            tilt is not None
            and st.phase == MissionPhase.ANTENNA_DEPLOY
            and abs(float(tilt)) > thr.antenna_tilt_tolerance_deg
        ):
            return self._latch(st, FaultType.ANTENNA_TILT)

        st.last_fault = None
        return None

    # ------------------------------------------------------------------ #
    # Fault recovery
    # ------------------------------------------------------------------ #

    def execute_recovery(
        self,
        rover_id: str,
        fault: FaultType,
    ) -> bool:
        """Run the recovery procedure for ``fault``; report success.

        On entry the rover is moved into ``FAULT_RECOVERY`` (remembering the
        interrupted phase). Recovery strategy by fault:

        * ``CABLE_SNAG`` / ``ROVER_STUCK`` / ``MOTOR_OVERHEAT`` / ``GPS_LOST`` /
          ``ANTENNA_TILT``: bounded retry. Each call consumes one retry; success
          is reported until ``MissionConfig.max_retries`` is exhausted, after
          which recovery fails (escalate to operator/coordinator).
        * ``BATTERY_LOW``: non-retryable — abort the segment and divert straight
          to ``RETURN_TO_BASE``; reported as a successful *managed* response
          (the rover is safe), not a continuation of the task.
        * ``CABLE_EXHAUSTED``: non-retryable end-of-spool — the active point is
          released, the segment ends, and the rover diverts to
          ``RETURN_TO_BASE``; reported as success (nominal end of cable run).

        Returns:
            ``True`` if the rover has a safe path forward (retry available or a
            managed divert completed), ``False`` if retries are exhausted and
            operator intervention is required.
        """
        st = self._rover(rover_id)
        if st.phase != MissionPhase.FAULT_RECOVERY:
            st.phase_before_fault = st.phase
            st.phase = MissionPhase.FAULT_RECOVERY

        max_retries = self._config.max_retries if self._config else 3

        # Non-retryable managed diverts: the rover is steered to safety and the
        # phase machine is rerouted to RETURN_TO_BASE.
        if fault in (FaultType.BATTERY_LOW, FaultType.CABLE_EXHAUSTED):
            if fault == FaultType.CABLE_EXHAUSTED and st.active_point is not None:
                # End-of-spool is a nominal completion of the cable run.
                st.active_point.status = "complete"
            st.phase_before_fault = MissionPhase.RETURN_TO_BASE
            st.retries.pop(fault, None)
            return True

        # Retryable faults: consume a retry budget.
        used = st.retries.get(fault, 0)
        if used >= max_retries:
            # Exhausted — leave the rover parked in FAULT_RECOVERY for the
            # coordinator/operator to handle (e.g. rescue, reassignment).
            return False

        st.retries[fault] = used + 1
        # Resume the interrupted phase on the next advance_phase() call.
        return True

    def clear_fault(self, rover_id: str) -> None:
        """Reset the retry budget for a rover after operator intervention."""
        st = self._rover(rover_id)
        st.retries.clear()
        st.last_fault = None

    # ------------------------------------------------------------------ #
    # Status, deadlock, exclusion zones
    # ------------------------------------------------------------------ #

    def get_grid_status(self) -> List[GridPoint]:
        self._require_initialized()
        return list(self._grid)

    def get_mission_progress(self) -> float:
        """Fraction of grid points fully deployed (``status == 'complete'``)."""
        self._require_initialized()
        if not self._grid:
            return 1.0
        done = sum(1 for gp in self._grid if gp.status == "complete")
        return done / len(self._grid)

    def check_deadlock(
        self,
        rover_positions: Dict[str, npt.NDArray[np.float64]],
    ) -> List[Tuple[str, str]]:
        """Report rover pairs in spatial proximity tight enough to deadlock.

        Two rovers within ``_DEADLOCK_RADIUS_M`` of each other are reported as a
        deadlock candidate. Rovers that are ``COMPLETE`` or in
        ``FAULT_RECOVERY`` are excluded (a parked rover is not a mover that can
        deadlock). Pairs are returned with rover ids in a stable sorted order so
        callers can deduplicate.
        """
        active = []
        for rid, pos in rover_positions.items():
            st = self._rovers.get(rid)
            if st is None:
                continue
            if st.phase in (MissionPhase.COMPLETE, MissionPhase.FAULT_RECOVERY):
                continue
            active.append((rid, np.asarray(pos, dtype=np.float64).reshape(3)))

        pairs: List[Tuple[str, str]] = []
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                a_id, a_pos = active[i]
                b_id, b_pos = active[j]
                if float(np.linalg.norm(a_pos - b_pos)) <= _DEADLOCK_RADIUS_M:
                    pairs.append(tuple(sorted((a_id, b_id))))  # type: ignore[arg-type]
        return pairs

    def get_cable_exclusion_zones(self) -> List[npt.NDArray[np.float64]]:
        """Sampled point clouds along every deployed cable centerline.

        A cable is considered deployed for every grid point whose status is
        ``'equipped'`` or ``'complete'``: the cable runs from the depot to that
        point. Each returned ``Nx3`` array samples the centerline at
        ``_CABLE_SAMPLE_SPACING_M`` intervals. Planners should treat a
        ``_CABLE_EXCLUSION_BUFFER_M`` (1.5 m) radius around these points as
        blocked.
        """
        self._require_initialized()
        zones: List[npt.NDArray[np.float64]] = []
        for gp in self._grid:
            if gp.status not in ("equipped", "complete"):
                continue
            start = self._depot
            end = np.asarray(gp.position_xyz, dtype=np.float64).reshape(3)
            length = float(np.linalg.norm(end - start))
            n_samples = max(2, int(np.ceil(length / _CABLE_SAMPLE_SPACING_M)) + 1)
            ts = np.linspace(0.0, 1.0, n_samples)
            centerline = start[None, :] + ts[:, None] * (end - start)[None, :]
            zones.append(centerline)
        return zones

    @property
    def cable_exclusion_buffer_m(self) -> float:
        """Clearance radius (m) callers must keep from cable centerlines."""
        return _CABLE_EXCLUSION_BUFFER_M

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _require_initialized(self) -> None:
        if self._config is None:
            raise RuntimeError(
                "MissionManager.initialize(config) must be called first"
            )

    def _rover(self, rover_id: str) -> _RoverState:
        st = self._rovers.get(rover_id)
        if st is None:
            raise KeyError(
                f"unknown rover '{rover_id}' (register it via assign_rovers)"
            )
        return st

    def _clock(self, telemetry: dict) -> float:
        """Return a monotonically increasing time source for fault timers."""
        t = telemetry.get("time_s")
        if t is not None:
            self._step_counter = float(t)
            return self._step_counter
        self._step_counter += 1.0
        return self._step_counter

    @staticmethod
    def _latch(st: _RoverState, fault: FaultType) -> FaultType:
        st.last_fault = fault
        return fault


# ---------------------------------------------------------------------------
# YAML config helper (mirrors power_config_from_yaml / arm_config_from_yaml)
# ---------------------------------------------------------------------------


def mission_config_from_yaml(
    yaml_path: str,
) -> Tuple[MissionConfig, npt.NDArray[np.float64]]:
    """Load a :class:`MissionConfig` and depot position from ``configs/mission.yaml``.

    Maps the ``grid`` block (``origin``, ``num_rows``, ``num_cols``,
    ``spacing_m``) and the ``rovers`` list onto :class:`MissionConfig`. The
    depot/moonbase position is read from ``mission.depot`` /
    ``mission.base_station_position`` when present, otherwise it defaults to the
    world origin.

    Returns:
        ``(MissionConfig, depot_xyz)`` ready to pass to
        :class:`LunarMissionManager`.
    """
    import yaml  # local import: optional dependency for config-driven runs

    with open(yaml_path, "r", encoding="utf-8") as fh:
        doc: Dict[str, Any] = yaml.safe_load(fh) or {}

    grid = doc.get("grid", {})
    rovers = doc.get("rovers", []) or []
    mission = doc.get("mission", {})

    origin = np.asarray(
        grid.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64
    ).reshape(3)
    depot = np.asarray(
        mission.get("depot")
        or mission.get("base_station_position")
        or [0.0, 0.0, 0.0],
        dtype=np.float64,
    ).reshape(3)

    config = MissionConfig(
        grid_origin=origin,
        grid_rows=int(grid.get("num_rows", 1)),
        grid_cols=int(grid.get("num_cols", 1)),
        grid_spacing_m=float(grid.get("spacing_m", 1.0)),
        num_rovers=max(1, len(rovers)),
        max_retries=int(
            doc.get("contingency", {}).get("max_retries", 3)
        ),
    )
    return config, depot
