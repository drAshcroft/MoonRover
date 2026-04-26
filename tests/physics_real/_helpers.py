from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def require_real_genesis_smoke() -> None:
    if os.environ.get("MOON_ROVER_RUN_GENESIS_SMOKE") != "1":
        pytest.skip("set MOON_ROVER_RUN_GENESIS_SMOKE=1 to run real Genesis smoke tests")


def run_smoke_payload(code: str, *, timeout: int = 180) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return json.loads(completed.stdout.strip().splitlines()[-1])
