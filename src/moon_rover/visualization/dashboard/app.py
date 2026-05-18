"""System 15.2: Mission Dashboard — FastAPI + WebSocket live telemetry backend.

Serves a single-page mission dashboard plus a REST/WebSocket API backed by the
process-global :class:`~moon_rover.visualization.dashboard.hub.TelemetryHub`. A
running simulation (see ``scripts/demo_dashboard.py``) publishes a
:class:`MissionSnapshot` each tick; this app streams it to every connected
browser at a fixed rate and answers point REST queries from the same snapshot.

Routes
------
* ``GET  /``                       — interactive dashboard page (canvas map +
  gauges + fault log).
* ``WS   /ws/telemetry``           — full :class:`MissionSnapshot` JSON pushed
  at ``STREAM_HZ``.
* ``GET  /api/mission/status``     — mission grid layout + deployment status.
* ``GET  /api/rover/{rover_id}/state`` — single rover pose/velocity/phase.
* ``GET  /api/power/{rover_id}``   — battery / power summary for one rover.
* ``GET  /api/faults``             — active + resolved fault log with stats.

All payloads derive from the live hub snapshot, so the API and the streamed
view never disagree.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

from moon_rover.visualization.dashboard.hub import (
    MissionSnapshot,
    TelemetryHub,
    get_hub,
)

#: WebSocket broadcast rate (Hz).
STREAM_HZ = 15.0

_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Moon Rover Mission Dashboard")


def _hub() -> TelemetryHub:
    return get_hub()


def _rover_record(snap: MissionSnapshot, rover_id: str) -> Dict[str, Any] | None:
    for rover in snap.rovers:
        if rover.rover_id == rover_id:
            return rover.__dict__
    return None


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #


@app.get("/")
async def root() -> HTMLResponse:
    """Serve the interactive dashboard page."""
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="<html><body><h1>Dashboard static assets missing</h1></body></html>",
        status_code=500,
    )


# --------------------------------------------------------------------------- #
# WebSocket telemetry stream
# --------------------------------------------------------------------------- #


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket) -> None:
    """Stream the live mission snapshot as JSON at ``STREAM_HZ``.

    Only re-serialises and sends when the hub revision changes, so an idle
    simulation does not spam the socket; a heartbeat is still sent every second
    so the client can detect a dead connection.
    """
    await websocket.accept()
    hub = _hub()
    period = 1.0 / STREAM_HZ
    last_rev = -1
    idle_ticks = 0
    try:
        while True:
            rev = hub.revision
            if rev != last_rev:
                last_rev = rev
                idle_ticks = 0
                await websocket.send_text(json.dumps(hub.snapshot().to_json()))
            else:
                idle_ticks += 1
                if idle_ticks >= int(STREAM_HZ):  # ~1 s heartbeat
                    idle_ticks = 0
                    await websocket.send_text(json.dumps({"heartbeat": True}))
            await asyncio.sleep(period)
    except WebSocketDisconnect:
        return
    except Exception:
        # Client vanished mid-send; close quietly.
        return


# --------------------------------------------------------------------------- #
# REST API (all derived from the live snapshot)
# --------------------------------------------------------------------------- #


@app.get("/api/mission/status")
async def get_mission_status() -> JSONResponse:
    """Mission grid status and antenna deployment positions."""
    snap = _hub().snapshot()
    antennas = [
        {
            "id": f"grid_{gp.row}_{gp.col}",
            "target_position": gp.position,
            "deployed": gp.status in ("equipped", "complete"),
            "status": gp.status,
            "assigned_rover": gp.assigned_rover,
        }
        for gp in snap.grid_points
    ]
    return JSONResponse(
        {
            "mission_id": snap.mission_id,
            "status": snap.status,
            "sim_time_s": snap.sim_time_s,
            "progress": snap.progress,
            "antennas": antennas,
            "grid_bounds": snap.grid_bounds,
            "base_station_position": snap.depot_position,
        }
    )


@app.get("/api/rover/{rover_id}/state")
async def get_rover_state(rover_id: str) -> JSONResponse:
    """Pose, velocity, phase and cable state for one rover."""
    snap = _hub().snapshot()
    rec = _rover_record(snap, rover_id)
    if rec is None:
        return JSONResponse(
            {"error": f"unknown rover '{rover_id}'"}, status_code=404
        )
    return JSONResponse(
        {
            "rover_id": rover_id,
            "timestamp": snap.sim_time_s,
            "position": rec["position"],
            "heading_rad": rec["heading_rad"],
            "velocity": rec["velocity"],
            "phase": rec["phase"],
            "cable_length_remaining_m": rec["cable_length_remaining_m"],
            "cable_tension_n": rec["cable_tension_n"],
            "target": rec["target"],
            "fault_active": rec["fault_active"],
        }
    )


@app.get("/api/power/{rover_id}")
async def get_power_state(rover_id: str) -> JSONResponse:
    """Battery / thermal summary for one rover."""
    snap = _hub().snapshot()
    rec = _rover_record(snap, rover_id)
    if rec is None:
        return JSONResponse(
            {"error": f"unknown rover '{rover_id}'"}, status_code=404
        )
    soc = float(rec["battery_soc"])
    return JSONResponse(
        {
            "rover_id": rover_id,
            "timestamp": snap.sim_time_s,
            "battery": {
                "level_percent": round(soc * 100.0, 1),
                "soc": soc,
                "temperature_celsius": rec["battery_temp_c"],
            },
            "charging_status": (
                "charging" if rec["phase"] == "charging" else "discharging"
            ),
        }
    )


@app.get("/api/faults")
async def get_faults() -> JSONResponse:
    """Fault log with active/resolved split and summary statistics."""
    snap = _hub().snapshot()
    active = [f.__dict__ for f in snap.faults if not f.resolved]
    resolved = [f.__dict__ for f in snap.faults if f.resolved]
    crit = sum(1 for f in snap.faults if f.severity == "critical")
    warn = sum(1 for f in snap.faults if f.severity == "warning")
    return JSONResponse(
        {
            "active_faults": active,
            "resolved_faults": resolved,
            "fault_statistics": {
                "total_faults": len(snap.faults),
                "critical_count": crit,
                "warning_count": warn,
                "active_count": len(active),
            },
        }
    )


# --------------------------------------------------------------------------- #
# Server helper
# --------------------------------------------------------------------------- #


def serve(host: str = "127.0.0.1", port: int = 8000, log_level: str = "warning") -> None:
    """Run the dashboard with uvicorn (blocking).

    The simulation should publish to :func:`get_hub` from another thread before
    or while this is running.
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level=log_level)
