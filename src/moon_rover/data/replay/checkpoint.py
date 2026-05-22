"""
System 13.2: Replay System

Checkpoint and replay functionality for simulation state saving, restoration,
and deterministic replay with parameter sensitivity analysis.
"""

from __future__ import annotations

import json
import secrets
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import h5py
import numpy as np


class ReplaySystem(ABC):
    """
    Checkpoint and replay system for simulation reproducibility and analysis.

    Enables:
    - Saving full simulation state at arbitrary moments
    - Restoring to prior state for deterministic replay
    - Replaying with modified control inputs for testing
    - Parameter sensitivity analysis via delta replay
    - Determinism verification (bit-exact reproducibility)

    Useful for:
    - Debugging rover behavior
    - Analyzing failure scenarios
    - Validating RL policies
    - Benchmarking algorithm improvements
    """

    @abstractmethod
    def save_checkpoint(
        self,
        engine: Any,
        sim_time: float,
    ) -> str:
        """
        Save complete simulation state to checkpoint.

        Captures all relevant simulation state including rover poses,
        velocities, sensor histories, cable state, terrain modifications,
        and random number generator state for deterministic replay.

        Args:
            engine: Simulation engine object to checkpoint.
            sim_time: Current simulation time in seconds (for labeling).

        Returns:
            Checkpoint ID (typically timestamp or uuid) for later retrieval.
        """
        raise NotImplementedError

    @abstractmethod
    def restore_checkpoint(
        self,
        checkpoint_id: str,
        engine: Any,
    ) -> float:
        """
        Restore simulation to previously saved checkpoint state.

        Resets all simulation state to match saved checkpoint, including
        rover kinematics, sensor state, and random number generator.
        Subsequent simulation evolution is deterministic.

        Args:
            checkpoint_id: ID of checkpoint to restore (from save_checkpoint).
            engine: Simulation engine to populate with checkpoint state.

        Returns:
            Simulation time at checkpoint in seconds.

        Raises:
            FileNotFoundError: If checkpoint_id not found.
        """
        raise NotImplementedError

    @abstractmethod
    def list_checkpoints(self) -> list[dict]:
        """
        List all available checkpoints with metadata.

        Returns:
            List of dicts with keys: 'checkpoint_id', 'sim_time', 'timestamp',
            'size_bytes', 'description'.
        """
        raise NotImplementedError

    @abstractmethod
    def replay(
        self,
        checkpoint_id: str,
        speed: float,
        control_inputs: Any,
    ) -> None:
        """
        Replay simulation from checkpoint with specified control inputs.

        Restores checkpoint and replays simulation forward with provided
        control sequence. Speed parameter allows time-accelerated or
        time-slowed playback for visualization or testing.

        Args:
            checkpoint_id: Starting checkpoint ID.
            speed: Replay speed multiplier. 1.0 = real-time, 0.1 = 10x slower,
                100 = 100x faster. Range: [0.1, 100].
            control_inputs: Control input sequence (format depends on
                simulation backend).

        Returns:
            None

        Raises:
            FileNotFoundError: If checkpoint not found.
            ValueError: If speed outside valid range.
        """
        raise NotImplementedError

    @abstractmethod
    def delta_replay(
        self,
        checkpoint_id: str,
        modified_params: dict,
    ) -> None:
        """
        Replay with modified simulation parameters for sensitivity analysis.

        Restores checkpoint and reruns simulation with specified parameter
        changes. Useful for understanding how parameter variations affect
        rover behavior and mission outcomes.

        Example params:
        - 'gravity_m_s2': 1.62 (lunar gravity)
        - 'friction_coefficient': 0.6
        - 'motor_max_torque_nm': 5.0
        - 'cable_stiffness_n_m': 100.0

        Args:
            checkpoint_id: Starting checkpoint ID.
            modified_params: Dict of parameter_name -> new_value to override.

        Returns:
            None

        Raises:
            FileNotFoundError: If checkpoint not found.
            ValueError: If modified params are invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def verify_determinism(
        self,
        checkpoint_id: str,
        control_inputs: Any,
    ) -> bool:
        """
        Verify simulation is deterministic by replaying twice.

        Performs two identical replays from checkpoint with same control
        inputs and compares outputs for bit-exact equality. Useful for
        validating that floating-point operations are deterministic and
        RNG is properly seeded.

        Args:
            checkpoint_id: Checkpoint to replay from.
            control_inputs: Control sequence for replay.

        Returns:
            True if both replays produce identical results, False otherwise.
            If False, likely indicates non-deterministic elements
            (e.g., uncontrolled random numbers, floating-point order changes).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete CheckpointStore
# ---------------------------------------------------------------------------


_FILE_EXT = ".h5"
_FILE_VERSION = 1
_SCHEMA_NAME = "moon_rover.checkpoint"


class CheckpointStore(ReplaySystem):
    """HDF5-backed checkpoint store wrapping ``PhysicsEngine.save_snapshot()``.

    Layout (one file per checkpoint at ``<store_dir>/<checkpoint_id>.h5``):

        / (root attrs)
            schema           = "moon_rover.checkpoint"
            file_version     = 1
            checkpoint_id    = str
            sim_time         = float64 (seconds)
            wall_clock_ns    = int64
            description      = str
            engine_class     = str  (e.g. "GenesisPhysicsEngine")
            extra_keys_json  = str  (JSON list of subsystem state keys)
        /snapshot            uint8[N]   raw bytes from engine.save_snapshot()
        /subsystem_states/<name>   uint8[M]   per-subsystem opaque blob

    Monte Carlo branching is supported via ``branch(checkpoint_id, engines)``:
    the same saved snapshot is restored into each of the supplied engines so
    that N parallel branches start from identical state. ``replay``,
    ``delta_replay``, and ``verify_determinism`` require a scenario runner and
    are out of scope for the data layer — they raise ``NotImplementedError``.
    """

    def __init__(self, store_dir: str | Path) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def store_dir(self) -> Path:
        return self._dir

    # ---- save / restore ---------------------------------------------------

    def save_checkpoint(
        self,
        engine: Any,
        sim_time: float,
        *,
        description: str = "",
        extra_state: Optional[dict[str, bytes]] = None,
        checkpoint_id: Optional[str] = None,
    ) -> str:
        """Persist the engine's snapshot bytes to an HDF5 file.

        Args:
            engine: Object exposing ``save_snapshot() -> bytes``.
            sim_time: Simulation time (seconds) recorded as metadata.
            description: Free-text label stored in metadata.
            extra_state: Optional mapping of subsystem name to opaque bytes
                (RNG state, mission state, etc.) persisted alongside.
            checkpoint_id: Override the auto-generated id. Useful for tests
                and deterministic naming.

        Returns:
            The checkpoint id used as filename stem.
        """
        snapshot = engine.save_snapshot()
        if not isinstance(snapshot, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"engine.save_snapshot() must return bytes; got {type(snapshot).__name__}"
            )
        snapshot_bytes = bytes(snapshot)

        cid = _make_checkpoint_id() if checkpoint_id is None else checkpoint_id
        path = self._path_for(cid)
        if path.exists():
            raise FileExistsError(f"checkpoint {cid!r} already exists at {path}")

        extra_state = extra_state or {}
        for name, blob in extra_state.items():
            if not isinstance(blob, (bytes, bytearray, memoryview)):
                raise TypeError(
                    f"extra_state[{name!r}] must be bytes; got {type(blob).__name__}"
                )

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with h5py.File(tmp_path, "w") as f:
                f.attrs["schema"] = _SCHEMA_NAME
                f.attrs["file_version"] = _FILE_VERSION
                f.attrs["checkpoint_id"] = cid
                f.attrs["sim_time"] = float(sim_time)
                f.attrs["wall_clock_ns"] = np.int64(time.time_ns())
                f.attrs["description"] = description
                f.attrs["engine_class"] = type(engine).__name__
                f.attrs["extra_keys_json"] = json.dumps(sorted(extra_state.keys()))
                f.create_dataset(
                    "snapshot",
                    data=np.frombuffer(snapshot_bytes, dtype=np.uint8),
                )
                if extra_state:
                    grp = f.create_group("subsystem_states")
                    for name, blob in extra_state.items():
                        grp.create_dataset(
                            name, data=np.frombuffer(bytes(blob), dtype=np.uint8)
                        )
            tmp_path.replace(path)
        except Exception:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
            raise
        return cid

    def restore_checkpoint(self, checkpoint_id: str, engine: Any) -> float:
        path = self._require_path(checkpoint_id)
        with h5py.File(path, "r") as f:
            self._validate_schema(f, checkpoint_id)
            sim_time = float(f.attrs["sim_time"])
            snapshot_bytes = bytes(f["snapshot"][:].tobytes())
        engine.restore_snapshot(snapshot_bytes)
        return sim_time

    def get_extra_state(self, checkpoint_id: str) -> dict[str, bytes]:
        """Return per-subsystem opaque blobs saved with the checkpoint."""
        path = self._require_path(checkpoint_id)
        out: dict[str, bytes] = {}
        with h5py.File(path, "r") as f:
            self._validate_schema(f, checkpoint_id)
            grp = f.get("subsystem_states")
            if grp is None:
                return out
            for name in grp.keys():
                out[name] = bytes(grp[name][:].tobytes())
        return out

    def get_metadata(self, checkpoint_id: str) -> dict:
        path = self._require_path(checkpoint_id)
        with h5py.File(path, "r") as f:
            self._validate_schema(f, checkpoint_id)
            extra_keys = json.loads(str(f.attrs.get("extra_keys_json", "[]")))
            return {
                "checkpoint_id": str(f.attrs["checkpoint_id"]),
                "sim_time": float(f.attrs["sim_time"]),
                "wall_clock_ns": int(f.attrs["wall_clock_ns"]),
                "description": str(f.attrs.get("description", "")),
                "engine_class": str(f.attrs.get("engine_class", "")),
                "extra_keys": extra_keys,
                "size_bytes": path.stat().st_size,
            }

    # ---- listing / deletion ----------------------------------------------

    def list_checkpoints(self) -> list[dict]:
        entries: list[dict] = []
        for p in sorted(self._dir.glob(f"*{_FILE_EXT}")):
            try:
                with h5py.File(p, "r") as f:
                    if str(f.attrs.get("schema", "")) != _SCHEMA_NAME:
                        continue
                    entries.append(
                        {
                            "checkpoint_id": str(f.attrs["checkpoint_id"]),
                            "sim_time": float(f.attrs["sim_time"]),
                            "timestamp": _iso_from_ns(int(f.attrs["wall_clock_ns"])),
                            "size_bytes": p.stat().st_size,
                            "description": str(f.attrs.get("description", "")),
                        }
                    )
            except OSError:
                continue
        entries.sort(key=lambda e: e["timestamp"])
        return entries

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        path = self._require_path(checkpoint_id)
        path.unlink()

    # ---- Monte Carlo branching -------------------------------------------

    def branch(
        self,
        checkpoint_id: str,
        engines: Iterable[Any],
    ) -> list[float]:
        """Restore one checkpoint into each of N engines for parallel branches.

        Each engine receives the same snapshot bytes so that all branches start
        from identical state. The caller is responsible for advancing each
        branch independently afterwards (different control inputs, RNG seeds,
        or parameter perturbations).

        Args:
            checkpoint_id: Checkpoint to restore from.
            engines: Iterable of engine instances. Each engine must expose
                ``restore_snapshot(bytes) -> None``.

        Returns:
            List of ``sim_time`` values, one per engine (all equal).
        """
        path = self._require_path(checkpoint_id)
        with h5py.File(path, "r") as f:
            self._validate_schema(f, checkpoint_id)
            sim_time = float(f.attrs["sim_time"])
            snapshot_bytes = bytes(f["snapshot"][:].tobytes())

        sim_times: list[float] = []
        for engine in engines:
            engine.restore_snapshot(snapshot_bytes)
            sim_times.append(sim_time)
        if not sim_times:
            raise ValueError("branch() requires at least one engine")
        return sim_times

    # ---- deferred to ScenarioRunner --------------------------------------

    def replay(
        self,
        checkpoint_id: str,
        speed: float,
        control_inputs: Any,
    ) -> None:
        raise NotImplementedError(
            "CheckpointStore.replay() requires a scenario runner; "
            "use scenarios/runner.py once available (task b7148202)."
        )

    def delta_replay(
        self,
        checkpoint_id: str,
        modified_params: dict,
    ) -> None:
        raise NotImplementedError(
            "CheckpointStore.delta_replay() requires a scenario runner; "
            "use scenarios/runner.py once available (task b7148202)."
        )

    def verify_determinism(
        self,
        checkpoint_id: str,
        control_inputs: Any,
    ) -> bool:
        raise NotImplementedError(
            "CheckpointStore.verify_determinism() requires a scenario runner; "
            "use scenarios/runner.py once available (task b7148202)."
        )

    # ---- internals --------------------------------------------------------

    def _path_for(self, checkpoint_id: str) -> Path:
        if not checkpoint_id or "/" in checkpoint_id or "\\" in checkpoint_id:
            raise ValueError(f"invalid checkpoint_id: {checkpoint_id!r}")
        return self._dir / f"{checkpoint_id}{_FILE_EXT}"

    def _require_path(self, checkpoint_id: str) -> Path:
        path = self._path_for(checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint {checkpoint_id!r} not found at {path}")
        return path

    @staticmethod
    def _validate_schema(f: h5py.File, checkpoint_id: str) -> None:
        schema = str(f.attrs.get("schema", ""))
        if schema != _SCHEMA_NAME:
            raise ValueError(
                f"checkpoint {checkpoint_id!r} has unknown schema {schema!r}; "
                f"expected {_SCHEMA_NAME!r}"
            )
        version = int(f.attrs.get("file_version", 0))
        if version != _FILE_VERSION:
            raise ValueError(
                f"checkpoint {checkpoint_id!r} has file_version={version}; "
                f"expected {_FILE_VERSION}"
            )


def _make_checkpoint_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = secrets.token_hex(3)
    return f"ckpt_{stamp}_{suffix}"


def _iso_from_ns(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat()
