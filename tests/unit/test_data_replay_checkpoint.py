"""Unit tests for src/moon_rover/data/replay/checkpoint.py.

Covers:
- Save / restore round-trip via a fake engine exposing save_snapshot/restore_snapshot.
- Metadata persisted: sim_time, description, engine_class, wall_clock_ns.
- Monte Carlo branching: same checkpoint restored into N independent engines.
- list_checkpoints surfaces all entries with stable ordering.
- delete_checkpoint removes the file; later restore raises FileNotFoundError.
- extra_state (subsystem opaque blobs) round-trips with the checkpoint.
- replay/delta_replay/verify_determinism raise NotImplementedError (deferred).
- Bad inputs (non-bytes snapshot, missing checkpoint, invalid id) raise cleanly.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest

from moon_rover.data.replay.checkpoint import CheckpointStore


class _FakeEngine:
    """Minimal engine satisfying the snapshot contract."""

    def __init__(self, state: bytes = b"") -> None:
        self.state = state
        self.restored_with: bytes | None = None

    def save_snapshot(self) -> bytes:
        return self.state

    def restore_snapshot(self, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("restore_snapshot requires bytes")
        self.state = bytes(data)
        self.restored_with = bytes(data)


# ---------------------------------------------------------------------------
# Save / restore
# ---------------------------------------------------------------------------


def test_save_then_restore_round_trip(tmp_path):
    store = CheckpointStore(tmp_path)
    payload = secrets.token_bytes(2048)
    engine = _FakeEngine(payload)

    cid = store.save_checkpoint(engine, sim_time=12.5, description="t=12.5")
    assert (tmp_path / f"{cid}.h5").exists()

    fresh = _FakeEngine()
    restored_sim_time = store.restore_checkpoint(cid, fresh)
    assert restored_sim_time == pytest.approx(12.5)
    assert fresh.state == payload
    assert fresh.restored_with == payload


def test_metadata_round_trip(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"abc")
    cid = store.save_checkpoint(engine, sim_time=3.0, description="phase_start")

    meta = store.get_metadata(cid)
    assert meta["checkpoint_id"] == cid
    assert meta["sim_time"] == pytest.approx(3.0)
    assert meta["description"] == "phase_start"
    assert meta["engine_class"] == "_FakeEngine"
    assert meta["size_bytes"] > 0
    assert meta["wall_clock_ns"] > 0


def test_save_with_explicit_checkpoint_id(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"x")
    cid = store.save_checkpoint(
        engine, sim_time=0.0, checkpoint_id="ckpt_test_explicit"
    )
    assert cid == "ckpt_test_explicit"
    assert (tmp_path / "ckpt_test_explicit.h5").exists()


def test_save_rejects_duplicate_checkpoint_id(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"x")
    store.save_checkpoint(engine, sim_time=0.0, checkpoint_id="dup")
    with pytest.raises(FileExistsError):
        store.save_checkpoint(engine, sim_time=1.0, checkpoint_id="dup")


def test_save_rejects_non_bytes_snapshot(tmp_path):
    class _BadEngine:
        def save_snapshot(self) -> str:  # type: ignore[override]
            return "not bytes"

    store = CheckpointStore(tmp_path)
    with pytest.raises(TypeError, match="must return bytes"):
        store.save_checkpoint(_BadEngine(), sim_time=0.0)


def test_restore_missing_checkpoint_raises(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine()
    with pytest.raises(FileNotFoundError):
        store.restore_checkpoint("nope", engine)


def test_invalid_checkpoint_id_rejected(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"x")
    for bad in ("", "with/slash", "with\\bslash"):
        with pytest.raises(ValueError, match="invalid checkpoint_id"):
            store.save_checkpoint(engine, sim_time=0.0, checkpoint_id=bad)


# ---------------------------------------------------------------------------
# Extra subsystem state
# ---------------------------------------------------------------------------


def test_extra_state_round_trip(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"snap")
    extra = {"rng": b"\x01\x02\x03", "mission": b"phase=drive"}
    cid = store.save_checkpoint(engine, sim_time=1.0, extra_state=extra)

    out = store.get_extra_state(cid)
    assert out == extra
    meta = store.get_metadata(cid)
    assert sorted(meta["extra_keys"]) == ["mission", "rng"]


def test_extra_state_empty_returns_empty_dict(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"snap")
    cid = store.save_checkpoint(engine, sim_time=1.0)
    assert store.get_extra_state(cid) == {}


def test_extra_state_rejects_non_bytes(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"snap")
    with pytest.raises(TypeError):
        store.save_checkpoint(
            engine, sim_time=1.0, extra_state={"bad": "not bytes"}  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# Listing and deletion
# ---------------------------------------------------------------------------


def test_list_checkpoints_returns_all_entries(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"x")
    ids = []
    for i, t in enumerate([1.0, 2.0, 3.0]):
        ids.append(
            store.save_checkpoint(
                engine, sim_time=t, checkpoint_id=f"ckpt_test_{i:02d}"
            )
        )

    entries = store.list_checkpoints()
    listed_ids = [e["checkpoint_id"] for e in entries]
    assert set(listed_ids) == set(ids)
    for e in entries:
        assert {"checkpoint_id", "sim_time", "timestamp", "size_bytes", "description"} <= e.keys()


def test_list_checkpoints_ignores_unrelated_files(tmp_path):
    store = CheckpointStore(tmp_path)
    (tmp_path / "stray.h5").write_bytes(b"not an hdf5 file")
    (tmp_path / "note.txt").write_text("ignore me")
    engine = _FakeEngine(b"x")
    cid = store.save_checkpoint(engine, sim_time=0.0)

    entries = store.list_checkpoints()
    assert [e["checkpoint_id"] for e in entries] == [cid]


def test_delete_checkpoint_removes_file(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"x")
    cid = store.save_checkpoint(engine, sim_time=0.0)
    store.delete_checkpoint(cid)
    assert not (tmp_path / f"{cid}.h5").exists()
    with pytest.raises(FileNotFoundError):
        store.restore_checkpoint(cid, _FakeEngine())


def test_delete_missing_checkpoint_raises(tmp_path):
    store = CheckpointStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.delete_checkpoint("never_existed")


# ---------------------------------------------------------------------------
# Monte Carlo branching
# ---------------------------------------------------------------------------


def test_branch_restores_into_n_engines_identically(tmp_path):
    store = CheckpointStore(tmp_path)
    payload = secrets.token_bytes(64)
    src = _FakeEngine(payload)
    cid = store.save_checkpoint(src, sim_time=7.5)

    branches = [_FakeEngine(b"junk_a"), _FakeEngine(b"junk_b"), _FakeEngine(b"junk_c")]
    sim_times = store.branch(cid, branches)

    assert sim_times == [7.5, 7.5, 7.5]
    for engine in branches:
        assert engine.state == payload


def test_branch_with_no_engines_raises(tmp_path):
    store = CheckpointStore(tmp_path)
    engine = _FakeEngine(b"x")
    cid = store.save_checkpoint(engine, sim_time=0.0)
    with pytest.raises(ValueError, match="at least one engine"):
        store.branch(cid, [])


# ---------------------------------------------------------------------------
# Deferred operations
# ---------------------------------------------------------------------------


def test_replay_is_deferred(tmp_path):
    store = CheckpointStore(tmp_path)
    with pytest.raises(NotImplementedError, match="scenario runner"):
        store.replay("any", speed=1.0, control_inputs=None)


def test_delta_replay_is_deferred(tmp_path):
    store = CheckpointStore(tmp_path)
    with pytest.raises(NotImplementedError, match="scenario runner"):
        store.delta_replay("any", modified_params={})


def test_verify_determinism_is_deferred(tmp_path):
    store = CheckpointStore(tmp_path)
    with pytest.raises(NotImplementedError, match="scenario runner"):
        store.verify_determinism("any", control_inputs=None)


# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------


def test_restoring_foreign_hdf5_raises(tmp_path):
    import h5py

    store = CheckpointStore(tmp_path)
    bad = tmp_path / "foreign.h5"
    with h5py.File(bad, "w") as f:
        f.attrs["schema"] = "not_a_checkpoint"
        f.create_dataset("snapshot", data=[0, 1, 2])
    with pytest.raises(ValueError, match="unknown schema"):
        store.restore_checkpoint("foreign", _FakeEngine())
