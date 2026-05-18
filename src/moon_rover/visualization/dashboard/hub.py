"""Thread-safe telemetry hub shared between a running simulation and the dashboard.

The simulation thread calls :meth:`TelemetryHub.publish` with the latest
:class:`MissionSnapshot`; the FastAPI request handlers and the WebSocket pump
read the most recent snapshot via :meth:`TelemetryHub.snapshot`. A single
process-global hub (:func:`get_hub`) lets ``scripts/demo_dashboard.py`` (or any
scenario runner) feed the same dashboard the browser is connected to without
threading the object through every call site.

The snapshot is a plain JSON-serialisable structure so the WebSocket pump can
forward it with ``json.dumps`` and the REST endpoints can slice it directly.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoverTelemetry:
    """Per-rover state as displayed on the dashboard."""

    rover_id: str
    position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    heading_rad: float = 0.0
    velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    phase: str = "planning"
    battery_soc: float = 1.0
    battery_temp_c: float = 20.0
    cable_length_remaining_m: float = 0.0
    cable_tension_n: float = 0.0
    target: Optional[List[float]] = None
    fault_active: Optional[str] = None


@dataclass
class GridPointTelemetry:
    """One antenna grid point and its deployment status."""

    row: int
    col: int
    position: List[float]
    status: str  # unvisited | visited | equipped | complete
    assigned_rover: Optional[str] = None


@dataclass
class FaultEvent:
    """A single fault-log entry."""

    timestamp_s: float
    rover_id: str
    fault_type: str
    severity: str  # critical | warning | info
    description: str
    recovery_action: str
    resolved: bool = False


@dataclass
class MissionSnapshot:
    """Complete dashboard state at one instant."""

    timestamp_s: float = 0.0
    mission_id: str = "mission_001"
    status: str = "idle"  # idle | running | paused | completed | failed
    sim_time_s: float = 0.0
    progress: float = 0.0  # fraction of grid points complete [0, 1]
    depot_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    grid_bounds: Dict[str, float] = field(
        default_factory=lambda: {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
        }
    )
    rovers: List[RoverTelemetry] = field(default_factory=list)
    grid_points: List[GridPointTelemetry] = field(default_factory=list)
    cables: List[List[List[float]]] = field(default_factory=list)
    faults: List[FaultEvent] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryHub:
    """Holds the latest :class:`MissionSnapshot` behind a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = MissionSnapshot()
        self._rev = 0

    def publish(self, snapshot: MissionSnapshot) -> None:
        """Replace the current snapshot (called from the simulation thread)."""
        if not snapshot.timestamp_s:
            snapshot.timestamp_s = time.time()
        with self._lock:
            self._snapshot = snapshot
            self._rev += 1

    def snapshot(self) -> MissionSnapshot:
        """Return the most recently published snapshot."""
        with self._lock:
            return self._snapshot

    @property
    def revision(self) -> int:
        """Monotonic counter incremented on every publish (change detection)."""
        with self._lock:
            return self._rev


_HUB: Optional[TelemetryHub] = None
_HUB_LOCK = threading.Lock()


def get_hub() -> TelemetryHub:
    """Return the process-global telemetry hub, creating it on first use."""
    global _HUB
    with _HUB_LOCK:
        if _HUB is None:
            _HUB = TelemetryHub()
        return _HUB
