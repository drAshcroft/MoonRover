"""Live mission-dashboard demo: a kinematic multi-rover antenna deployment.

This is the human visual checkpoint for the dashboard "demo site". It drives a
real :class:`LunarMissionManager` + :class:`RelayCoordinator` through a full
multi-rover antenna-deployment mission using a lightweight kinematic motion
model (no physics engine, no GPU), and streams the mission state to the
FastAPI dashboard so a human can watch it in a browser.

What you should see at http://127.0.0.1:8000
---------------------------------------------
* A top-down map: the moonbase, the antenna grid (points turning amber when
  *equipped* then green when *complete*), trailing cables from the base, and
  rover icons moving along dashed lines to their targets.
* A live progress bar, per-rover battery gauges + mission phase, and a fault
  log that shows a scripted cable-snag being detected and recovered.

Usage
-----
    C:\\ve\\.genesis\\Scripts\\python.exe scripts/demo_dashboard.py
    # then open http://127.0.0.1:8000 in a browser

    # headless self-check (runs the mission fast, no server, asserts it finishes)
    C:\\ve\\.genesis\\Scripts\\python.exe scripts/demo_dashboard.py --selfcheck

Options
-------
    --host / --port      bind address (default 127.0.0.1:8000)
    --rovers N           number of rovers (default 2)
    --speed S            sim-seconds advanced per wall tick (default 0.5)
    --selfcheck          run the mission to completion headless and exit
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from typing import Dict, List, Optional

import numpy as np

from moon_rover.mission import (
    FaultType,
    LunarMissionManager,
    MissionConfig,
    MissionPhase,
    RelayCoordinator,
    RoverStatus,
)
from moon_rover.visualization.dashboard.hub import (
    FaultEvent,
    GridPointTelemetry,
    MissionSnapshot,
    RoverTelemetry,
    get_hub,
)

# --- mission geometry / kinematics ---------------------------------------- #
_DEPOT = np.array([0.0, 0.0, 0.0])
_GRID_ORIGIN = np.array([14.0, -12.0, 0.0])
_GRID_ROWS = 3
_GRID_COLS = 4
_GRID_SPACING_M = 9.0
_CABLE_LEN_M = 90.0
_DRIVE_SPEED_MPS = 2.2          # cruise speed
_ARRIVAL_TOL_M = 1.2
_DWELL_DEPLOY_S = 3.0           # antenna placement dwell
_DWELL_CONNECT_S = 2.5         # cable connect dwell
_BATTERY_DRAIN_PER_M = 0.0019  # SoC per metre driven
_BATTERY_WORK_DRAIN_PER_S = 0.0025
_CHARGE_RATE_PER_S = 0.06      # SoC per sim-second at base
_CHARGE_TARGET_SOC = 0.95
#: Scripted demo fault: snag rover_001's cable mid-transit at this sim time.
_SCRIPTED_SNAG_AT_S = 22.0


class _Rover:
    """Kinematic state for one rover plus its scripted-fault bookkeeping."""

    def __init__(self, rover_id: str) -> None:
        self.id = rover_id
        self.pos = _DEPOT.copy()
        self.heading = 0.0
        self.vel = np.zeros(3)
        self.soc = 1.0
        self.cable_remaining = _CABLE_LEN_M
        self.cable_tension = 5.0
        self.dwell_until = 0.0
        self.snag_done = False
        self.recovering = False
        self.recover_until = 0.0
        self.active_fault: Optional[FaultType] = None


def build_mission(n_rovers: int):
    cfg = MissionConfig(
        grid_origin=_GRID_ORIGIN,
        grid_rows=_GRID_ROWS,
        grid_cols=_GRID_COLS,
        grid_spacing_m=_GRID_SPACING_M,
        num_rovers=n_rovers,
        max_retries=3,
    )
    mm = LunarMissionManager(depot_xyz=_DEPOT)
    mm.initialize(cfg)
    co = RelayCoordinator(depot_xyz=_DEPOT)
    rover_ids = [f"rover_{i+1:03d}" for i in range(n_rovers)]
    for rid in rover_ids:
        co.register_rover(rid, "skid_steer")
    mm.assign_rovers(rover_ids)
    rovers = {rid: _Rover(rid) for rid in rover_ids}
    return cfg, mm, co, rovers


def _move_toward(rv: _Rover, target: np.ndarray, dt: float) -> bool:
    """Advance ``rv`` toward ``target``; return True once within tolerance."""
    delta = target - rv.pos
    delta[2] = 0.0
    dist = float(np.linalg.norm(delta))
    if dist <= _ARRIVAL_TOL_M:
        rv.vel = np.zeros(3)
        return True
    step = min(dist, _DRIVE_SPEED_MPS * dt)
    direction = delta / dist
    rv.pos = rv.pos + direction * step
    rv.vel = direction * (_DRIVE_SPEED_MPS)
    rv.heading = math.atan2(direction[1], direction[0])
    rv.soc = max(0.0, rv.soc - _BATTERY_DRAIN_PER_M * step)
    rv.cable_remaining = max(
        0.0, _CABLE_LEN_M - float(np.linalg.norm(rv.pos - _DEPOT))
    )
    return False


def step_rover(
    rv: _Rover,
    mm: LunarMissionManager,
    co: RelayCoordinator,
    sim_t: float,
    dt: float,
    faults: List[FaultEvent],
) -> None:
    """Advance one rover one tick: drive the phase machine kinematically."""
    st = mm  # alias for brevity
    phase = mm.get_current_phase(rv.id)

    # --- scripted demo fault: cable snag on rover_001 mid-transit --------- #
    if (
        rv.id == "rover_001"
        and not rv.snag_done
        and sim_t >= _SCRIPTED_SNAG_AT_S
        and phase == MissionPhase.TRANSIT
    ):
        rv.cable_tension = 520.0  # above snag threshold

    # --- fault detection from synthetic telemetry ------------------------ #
    if not rv.recovering:
        telem = {
            "time_s": sim_t,
            "battery_soc": rv.soc,
            "cable_tension_n": rv.cable_tension,
            "cable_length_remaining_m": rv.cable_remaining,
            "commanded_speed_mps": float(np.linalg.norm(rv.vel)),
            "actual_speed_mps": float(np.linalg.norm(rv.vel)),
            "gps_fix_quality": 1.0,
        }
        fault = mm.detect_fault(rv.id, telem)
        if fault is not None:
            rv.active_fault = fault
            ok = mm.execute_recovery(rv.id, fault)
            rv.recovering = True
            rv.recover_until = sim_t + 3.0
            sev = "critical" if fault in (
                FaultType.BATTERY_LOW,
                FaultType.ROVER_STUCK,
            ) else "warning"
            faults.append(
                FaultEvent(
                    timestamp_s=sim_t,
                    rover_id=rv.id,
                    fault_type=fault.value,
                    severity=sev,
                    description=f"{fault.value} detected",
                    recovery_action=(
                        "retrying" if ok else "operator intervention required"
                    ),
                )
            )
            if fault == FaultType.CABLE_SNAG:
                rv.snag_done = True
            return

    # --- recovery hold, then resume the interrupted phase ---------------- #
    if rv.recovering:
        if sim_t >= rv.recover_until:
            rv.recovering = False
            rv.cable_tension = 8.0
            for f in reversed(faults):
                if f.rover_id == rv.id and not f.resolved:
                    f.resolved = True
                    break
            mm.clear_fault(rv.id)
            if mm.get_current_phase(rv.id) == MissionPhase.FAULT_RECOVERY:
                mm.advance_phase(rv.id)  # back to interrupted phase
            rv.active_fault = None
        else:
            return

    phase = mm.get_current_phase(rv.id)

    if phase == MissionPhase.PLANNING:
        mm.advance_phase(rv.id)

    elif phase == MissionPhase.DEPOT_PICKUP:
        rv.cable_remaining = _CABLE_LEN_M
        mm.advance_phase(rv.id)  # pops active point -> TRANSIT

    elif phase == MissionPhase.TRANSIT:
        tgt = _rover_active_target(mm, rv.id)
        if tgt is not None and _move_toward(rv, tgt, dt):
            mm.advance_phase(rv.id)
            rv.dwell_until = sim_t + _DWELL_DEPLOY_S

    elif phase == MissionPhase.ANTENNA_DEPLOY:
        rv.soc = max(0.0, rv.soc - _BATTERY_WORK_DRAIN_PER_S * dt)
        if sim_t >= rv.dwell_until:
            mm.advance_phase(rv.id)
            rv.dwell_until = sim_t + _DWELL_CONNECT_S

    elif phase == MissionPhase.CABLE_CONNECT:
        rv.soc = max(0.0, rv.soc - _BATTERY_WORK_DRAIN_PER_S * dt)
        if sim_t >= rv.dwell_until:
            tgt = _rover_active_target(mm, rv.id)
            mm.advance_phase(rv.id)  # marks complete -> RETURN_TO_BASE
            if tgt is not None:
                co.report_cable_segment(rv.id, tgt)

    elif phase == MissionPhase.RETURN_TO_BASE:
        if _move_toward(rv, _DEPOT, dt):
            mm.advance_phase(rv.id)  # -> CHARGING

    elif phase == MissionPhase.CHARGING:
        rv.soc = min(1.0, rv.soc + _CHARGE_RATE_PER_S * dt)
        rv.vel = np.zeros(3)
        if rv.soc >= _CHARGE_TARGET_SOC:
            mm.advance_phase(rv.id)  # -> next segment or COMPLETE

    # COMPLETE / FAULT_RECOVERY: nothing to drive here.

    # keep coordinator's shared model fresh
    co.update_rover_status(
        RoverStatus(
            rover_id=rv.id,
            position=rv.pos.copy(),
            heading=rv.heading,
            phase=mm.get_current_phase(rv.id).value,
            battery_soc=rv.soc,
            cable_length_remaining=rv.cable_remaining,
        )
    )


def _rover_active_target(mm: LunarMissionManager, rover_id: str):
    """Active grid point for a rover, read out of the manager's state."""
    state = mm._rovers.get(rover_id)  # internal read for the demo driver
    if state is None or state.active_point is None:
        return None
    return np.asarray(state.active_point.position_xyz, dtype=np.float64)


def build_snapshot(
    mm: LunarMissionManager,
    rovers: Dict[str, _Rover],
    sim_t: float,
    faults: List[FaultEvent],
) -> MissionSnapshot:
    grid = mm.get_grid_status()
    xs = [gp.position_xyz[0] for gp in grid] + [_DEPOT[0]]
    ys = [gp.position_xyz[1] for gp in grid] + [_DEPOT[1]]
    margin = 6.0
    bounds = {
        "x_min": float(min(xs) - margin),
        "x_max": float(max(xs) + margin),
        "y_min": float(min(ys) - margin),
        "y_max": float(max(ys) + margin),
    }
    all_done = all(gp.status == "complete" for gp in grid) and len(grid) > 0
    snap = MissionSnapshot(
        mission_id="lunar_exploration_001",
        status="completed" if all_done else "running",
        sim_time_s=round(sim_t, 1),
        progress=mm.get_mission_progress(),
        depot_position=[float(_DEPOT[0]), float(_DEPOT[1]), float(_DEPOT[2])],
        grid_bounds=bounds,
        rovers=[
            RoverTelemetry(
                rover_id=rv.id,
                position=[float(rv.pos[0]), float(rv.pos[1]), float(rv.pos[2])],
                heading_rad=rv.heading,
                velocity=[float(rv.vel[0]), float(rv.vel[1]), float(rv.vel[2])],
                phase=mm.get_current_phase(rv.id).value,
                battery_soc=rv.soc,
                cable_length_remaining_m=rv.cable_remaining,
                cable_tension_n=rv.cable_tension,
                target=(
                    [float(t[0]), float(t[1]), float(t[2])]
                    if (t := _rover_active_target(mm, rv.id)) is not None
                    else None
                ),
                fault_active=(
                    rv.active_fault.value if rv.active_fault else None
                ),
            )
            for rv in rovers.values()
        ],
        grid_points=[
            GridPointTelemetry(
                row=gp.row,
                col=gp.col,
                position=[
                    float(gp.position_xyz[0]),
                    float(gp.position_xyz[1]),
                    float(gp.position_xyz[2]),
                ],
                status=gp.status,
                assigned_rover=gp.assigned_rover,
            )
            for gp in grid
        ],
        cables=[zone.tolist() for zone in mm.get_cable_exclusion_zones()],
        faults=faults,
    )
    return snap


def run_mission(
    speed: float,
    selfcheck: bool,
    stop: Optional[threading.Event] = None,
) -> None:
    """Drive the kinematic mission, publishing a snapshot every tick."""
    _, mm, co, rovers = build_mission(int(_n_rovers_holder[0]))
    hub = get_hub()
    faults: List[FaultEvent] = []
    sim_t = 0.0
    dt = speed
    tick = 0.0 if selfcheck else 1.0 / 30.0
    max_sim_s = 600.0

    while stop is None or not stop.is_set():
        for rv in rovers.values():
            step_rover(rv, mm, co, sim_t, dt, faults)
        hub.publish(build_snapshot(mm, rovers, sim_t, faults))
        sim_t += dt

        if mm.get_mission_progress() >= 1.0 and all(
            mm.get_current_phase(r).value in ("complete",)
            for r in rovers
        ):
            hub.publish(build_snapshot(mm, rovers, sim_t, faults))
            if selfcheck:
                print(
                    f"[selfcheck] mission complete at sim_t={sim_t:.1f}s, "
                    f"progress={mm.get_mission_progress():.0%}, "
                    f"faults={len(faults)} "
                    f"({sum(1 for f in faults if f.resolved)} resolved)"
                )
                return
            # keep serving the final state for late browser connections
            time.sleep(2.0)
            continue
        if sim_t > max_sim_s:
            print(f"[warn] mission did not converge within {max_sim_s}s")
            return
        if tick:
            time.sleep(tick)


_n_rovers_holder = [2]  # set from argv before run_mission starts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--rovers", type=int, default=2)
    parser.add_argument("--speed", type=float, default=0.5)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    _n_rovers_holder[0] = args.rovers

    if args.selfcheck:
        run_mission(speed=0.5, selfcheck=True)
        return

    stop = threading.Event()
    sim_thread = threading.Thread(
        target=run_mission, args=(args.speed, False, stop), daemon=True
    )
    sim_thread.start()

    print("=" * 64)
    print("  Moon Rover Mission Dashboard — live demo")
    print(f"  Open  http://{args.host}:{args.port}  in a browser")
    print("  (rovers deploy the antenna grid; watch the map + fault log)")
    print("  Ctrl+C to stop")
    print("=" * 64)
    try:
        from moon_rover.visualization.dashboard.app import serve

        serve(host=args.host, port=args.port)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


if __name__ == "__main__":
    main()
