"""Scripted cable-deployment demo: rover pulls cables out one by one, no RL.

A 4-wheel skid-steer rover sits at the moonbase next to a bank of three cable
reels. A fully scripted state machine drives the rover and choreographs the
manipulator arm to:

    1. Drive up to reel #i and square up to it.
    2. Reach the arm forward and down toward the reel handle.
    3. Close the gripper on the cable end.
    4. Raise the arm to a carry pose.
    5. Drive away from the base ~6 m, pulling the cable out behind it.
       (The cable visually runs from the reel post to the gripper.)
    6. Lower the arm to set the cable end on the regolith.
    7. Release the gripper — the deployed cable freezes along the rover's path.
    8. Back up, stow the arm, rotate to face the next reel.
    9. Repeat until all three cables are laid out radially from the moonbase.

There is no learned policy, no MPC, no balance controller. Everything is
explicit kinematic waypoints driven by a stopwatch, which is the point of the
demo: it shows the underlying physics, manipulator, cable, terrain, and drive
systems working *together* through a complete multi-step task.

The arm is kept "alive" the whole time — even while driving between objectives
it slowly sweeps joint 1 and bobs joint 3 so the manipulator never looks
parked. On each return trip the controller also takes a small detour to "tap"
an inspection beacon with the arm, exercising IK in the middle of the routine.

Usage
-----
    # Interactive viewer (default):
    python scripts/demo_cable_pull.py

    # Headless smoke (no window, exits when the third cable is deployed):
    python scripts/demo_cable_pull.py --no-viewer

    # GPU backend:
    python scripts/demo_cable_pull.py --backend gpu
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import yaml

# Make the src/ layout importable when the script is launched directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import genesis as gs  # noqa: E402

from moon_rover.core.physics.engine import GenesisConfig  # noqa: E402
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine  # noqa: E402
from moon_rover.core.scene.rover_composer import RoverComposer  # noqa: E402
from moon_rover.core.assets.urdf_builder import GenesisURDFBuilder  # noqa: E402
from moon_rover.environment.terrain import (  # noqa: E402
    LunarTerrainGenerator,
    TerrainConfig,
)
from moon_rover.rover.drive import (  # noqa: E402
    DriveCommand,
    DriveType,
    create_drive_system,
    drive_config_from_profile,
)
from moon_rover.rover.manipulator.arm import ArmConfig, GripperConfig  # noqa: E402
from moon_rover.rover.manipulator.serial_arm import SerialArm  # noqa: E402


ROVER_NAME = "demo_rover"
PROFILE_NAME = "four_wheel_skid"

DEFAULT_PHYSICS = _PROJECT_ROOT / "configs" / "physics.yaml"
DEFAULT_ROVER = _PROJECT_ROOT / "configs" / "rover.yaml"

WORLD_SIZE_M = 28.0
WORLD_RES = 64
WORLD_SEED = 1773
WORLD_FLAT_RADIUS_M = 5.0
WORLD_BLEND_RADIUS_M = 9.0

CABLE_LINKS_PER_REEL = 16          # visual link boxes pre-allocated per cable
# Beacon arm-tap detours are off by default — skid-steer rotation on lunar
# friction is slow, so the extra ~180° pivot they would require between cable
# jobs balloons the run time. Set > 0 to re-enable for a single beacon tap
# between deployments.
INSPECTION_BEACON_COUNT = 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scripted cable-deployment demo")
    p.add_argument("--no-viewer", action="store_true",
                   help="Headless mode (no 3-D window).")
    p.add_argument("--sim-hz", type=float, default=120.0,
                   help="Physics rate (Hz).")
    p.add_argument("--render-hz", type=int, default=60,
                   help="Viewer refresh cap (Hz).")
    p.add_argument("--backend", choices=("cpu", "gpu"), default="cpu",
                   help="Genesis backend.")
    p.add_argument("--max-sim-seconds", type=float, default=0.0,
                   help="Stop after this much sim time (0 = run until task done).")
    p.add_argument("--num-reels", type=int, default=3,
                   help="How many cables to deploy. Capped to 4.")
    p.add_argument("--destroy", action="store_true",
                   help="Explicitly tear down Genesis on exit.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Scene helpers (mirrors demo_rover_drive.LunarWorld minus the rut decals)
# ---------------------------------------------------------------------------


def _surface(color):
    try:
        return gs.surfaces.Default(color=color)
    except Exception:  # noqa: BLE001 - older Genesis falls back to default
        return None


def _visual_box(pos, size):
    """A render-only Box (no collider).

    The cable links and inspection beacons are kinematic visuals — they are
    repositioned every frame, so giving them colliders would flood the contact
    solver with ~100 teleporting bodies.
    """
    try:
        return gs.morphs.Box(pos=pos, size=size, collision=False)
    except TypeError:
        return gs.morphs.Box(pos=pos, size=size)


def _yaw_quat(yaw: float) -> np.ndarray:
    """Genesis base quaternion order is (w, x, y, z)."""
    return np.array(
        [math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)],
        dtype=np.float32,
    )


def _euler_from_quat_wxyz(q: np.ndarray) -> tuple[float, float, float]:
    qw, qx, qy, qz = (float(q[i]) for i in range(4))
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _wrap_pi(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# ---------------------------------------------------------------------------
# Cable reel — visual cable that spans reel → carry point or reel → drop point
# ---------------------------------------------------------------------------


class CableReelState(Enum):
    STORED = "stored"            # cable wound on the reel, links hidden
    BEING_PULLED = "pulling"     # rover is carrying the end; links bridge reel→gripper
    DEPLOYED = "deployed"        # cable laid on the ground, links frozen along path


@dataclass
class CableReel:
    """One reel post + a pool of pre-allocated visual cable links."""

    index: int
    post_xy: np.ndarray              # world XY of the reel post (cable comes out here)
    post_top_z: float                # z of cable handle on top of the reel
    color_rgb: tuple[float, float, float]
    state: CableReelState = CableReelState.STORED
    link_count: int = CABLE_LINKS_PER_REEL
    # Set when the cable is being pulled or deployed:
    deployed_path: Optional[list[np.ndarray]] = None  # (x, y) breadcrumb along ground
    deploy_distance_m: float = 0.0                    # cumulative path length
    target_deploy_m: float = 5.5                      # how far we want this cable laid

    @property
    def handle_world(self) -> np.ndarray:
        return np.array(
            [float(self.post_xy[0]), float(self.post_xy[1]), self.post_top_z],
            dtype=np.float64,
        )

    def link_name(self, k: int) -> str:
        return f"cable_{self.index:02d}_link_{k:03d}"


# ---------------------------------------------------------------------------
# Lunar world: heightfield, moonbase, reel posts, cable pools, beacons
# ---------------------------------------------------------------------------


class CableWorld:
    """The visible scene the scripted controller plays out in.

    Built in the engine CONSTRUCTION phase via :meth:`construct`. During
    SIMULATION the cable link pools and inspection-beacon visuals are
    repositioned in :meth:`step`.
    """

    def __init__(self, num_reels: int) -> None:
        self._size = WORLD_SIZE_M
        self._res = WORLD_RES
        cx = cy = self._size / 2.0
        self._center = np.array([cx, cy], dtype=np.float64)
        self._spawn_xy = (cx, cy)

        # Moonbase sits a few meters off the spawn pad on a diagonal.
        self._base_xy = np.array([cx - 6.0, cy - 6.0], dtype=np.float64)

        terrain_cfg = TerrainConfig(
            seed=WORLD_SEED,
            size_m=self._size,
            fBm_octaves=5,
            fBm_amplitude=0.22,
            crater_params={
                "count": 6,
                "min_radius_m": 0.6,
                "max_radius_m": 2.0,
                "depth_ratio": 0.20,
            },
            rock_density=0.012,
            rille_enabled=True,
            moonbase_position=(float(self._base_xy[0]),
                               float(self._base_xy[1]), 0.0),
            resolution=self._res,
        )
        self._terrain_gen_out = LunarTerrainGenerator(
            max_traversable_slope_deg=25.0,
            rock_clearance_m=0.35,
            moonbase_pad_radius_m=2.5,
        ).generate(terrain_cfg)

        self._height = self._flatten_pad(
            np.asarray(self._terrain_gen_out.height_field, dtype=np.float64)
        )

        # Lay the reels in a fan in front of the moonbase. Limit to 4 to stay
        # within the pre-allocated palette.
        n = max(1, min(num_reels, 4))
        self._reels: list[CableReel] = []
        colors = [
            (0.95, 0.30, 0.20),    # red
            (0.20, 0.80, 0.40),    # green
            (0.20, 0.45, 0.95),    # blue
            (0.95, 0.85, 0.20),    # yellow
        ]
        # Reel posts are arranged along a ~3.5 m arc 2.0 m forward of the base.
        anchor = self._base_xy + np.array([2.2, 2.2])
        # np.linspace(a, b, 1) returns [a], not the midpoint — short-circuit so
        # a single reel sits exactly at the anchor.
        spread = np.array([0.0]) if n == 1 else np.linspace(-1.8, 1.8, n)
        # Unit vector along the moonbase frontage (perpendicular to base→spawn).
        front_perp = np.array([1.0, -1.0]) / math.sqrt(2.0)
        for i in range(n):
            xy = anchor + front_perp * spread[i]
            self._reels.append(
                CableReel(
                    index=i,
                    post_xy=xy,
                    post_top_z=self.ground_z(float(xy[0]), float(xy[1])) + 0.75,
                    color_rgb=colors[i % len(colors)],
                )
            )

        # Inspection beacons the arm taps with its gripper on the return leg.
        self._beacons_xy = [
            self._base_xy + np.array([4.5, -1.2]),
            self._base_xy + np.array([1.5, -3.5]),
        ][:INSPECTION_BEACON_COUNT]

    # ----- terrain helpers --------------------------------------------------
    @property
    def spawn_xy(self) -> tuple[float, float]:
        return self._spawn_xy

    @property
    def reels(self) -> list[CableReel]:
        return self._reels

    @property
    def base_xy(self) -> np.ndarray:
        return self._base_xy.copy()

    @property
    def beacons_xy(self) -> list[np.ndarray]:
        return [b.copy() for b in self._beacons_xy]

    def _flatten_pad(self, h: np.ndarray) -> np.ndarray:
        res = h.shape[0]
        axis = np.linspace(0.0, self._size, res)
        gx, gy = np.meshgrid(axis, axis)
        dist = np.hypot(gx - self._center[0], gy - self._center[1])
        # 0 inside the flat radius, 1 beyond blend; smoothstep between.
        t = np.clip(
            (dist - WORLD_FLAT_RADIUS_M)
            / max(WORLD_BLEND_RADIUS_M - WORLD_FLAT_RADIUS_M, 1e-6),
            0.0, 1.0,
        )
        blend = t * t * (3.0 - 2.0 * t)
        return h * blend

    def ground_z(self, x: float, y: float) -> float:
        gx = np.clip(x / self._size * (self._res - 1), 0.0, self._res - 1)
        gy = np.clip(y / self._size * (self._res - 1), 0.0, self._res - 1)
        j0, i0 = int(math.floor(gx)), int(math.floor(gy))
        j1, i1 = min(j0 + 1, self._res - 1), min(i0 + 1, self._res - 1)
        tx, ty = gx - j0, gy - i0
        top = self._height[i0, j0] * (1 - tx) + self._height[i0, j1] * tx
        bot = self._height[i1, j0] * (1 - tx) + self._height[i1, j1] * tx
        return float(top * (1 - ty) + bot * ty)

    # ----- construction phase ---------------------------------------------
    def construct(self, engine: GenesisPhysicsEngine) -> None:
        engine.add_terrain_entity(
            "lunar_terrain",
            self._height.astype(np.float32),
            [self._size, self._size],
        )

        # Scattered boulders away from the spawn pad.
        rock_surf = _surface((0.40, 0.39, 0.37))
        rock_kw = {"entity_type": "fixed"}
        if rock_surf is not None:
            rock_kw["surface"] = rock_surf
        for k, rock in enumerate(self._terrain_gen_out.rock_positions):
            rx, ry = float(rock[0]), float(rock[1])
            radius = float(rock[3]) if len(rock) > 3 else 0.25
            # Keep the pad and the moonbase frontage clear so the arm has room.
            if np.hypot(rx - self._center[0], ry - self._center[1]) < (
                WORLD_FLAT_RADIUS_M + 1.5
            ):
                continue
            if np.linalg.norm(np.array([rx, ry]) - self._base_xy) < 4.5:
                continue
            rz = self.ground_z(rx, ry) + radius
            engine.add_entity(
                f"rock_{k:03d}",
                gs.morphs.Box(pos=(rx, ry, rz),
                              size=(radius * 2, radius * 2, radius * 2)),
                gs.materials.Rigid(friction=1.0), **rock_kw,
            )

        # Moonbase: a habitat slab + comm tower.
        bx, by = float(self._base_xy[0]), float(self._base_xy[1])
        bz = self.ground_z(bx, by)
        base_surf = _surface((0.75, 0.76, 0.80))
        base_kw = {"entity_type": "fixed"}
        if base_surf is not None:
            base_kw["surface"] = base_surf
        engine.add_entity(
            "moonbase_hab",
            gs.morphs.Box(pos=(bx, by, bz + 1.25), size=(4.0, 3.0, 2.5)),
            gs.materials.Rigid(), **base_kw,
        )
        engine.add_entity(
            "moonbase_tower",
            gs.morphs.Box(pos=(bx + 1.6, by + 1.2, bz + 3.0),
                          size=(0.18, 0.18, 6.0)),
            gs.materials.Rigid(), **base_kw,
        )

        # Reel posts (one fixed coloured pillar per cable) + the pre-allocated
        # cable link pool for that reel.
        for reel in self._reels:
            px, py = float(reel.post_xy[0]), float(reel.post_xy[1])
            pz = self.ground_z(px, py)
            post_surf = _surface(reel.color_rgb)
            post_kw = {"entity_type": "fixed"}
            if post_surf is not None:
                post_kw["surface"] = post_surf
            engine.add_entity(
                f"reel_post_{reel.index:02d}",
                gs.morphs.Box(pos=(px, py, pz + 0.4),
                              size=(0.30, 0.30, 0.80)),
                gs.materials.Rigid(), **post_kw,
            )
            # Slightly darker shade for the cable links so they read against
            # the post.
            r, g, b = reel.color_rgb
            link_surf = _surface((r * 0.5, g * 0.5, b * 0.5))
            link_kw = {"entity_type": "kinematic"}
            if link_surf is not None:
                link_kw["surface"] = link_surf
            for k in range(reel.link_count):
                engine.add_entity(
                    reel.link_name(k),
                    _visual_box(pos=(0.0, 0.0, -50.0 - k),
                                size=(0.18, 0.05, 0.05)),
                    gs.materials.Rigid(), **link_kw,
                )

        # Inspection beacons — tall slim posts the arm taps on the return leg.
        beacon_surf = _surface((0.95, 0.95, 0.30))
        beacon_kw = {"entity_type": "fixed"}
        if beacon_surf is not None:
            beacon_kw["surface"] = beacon_surf
        for i, bxy in enumerate(self._beacons_xy):
            x = float(bxy[0]); y = float(bxy[1])
            engine.add_entity(
                f"beacon_{i:02d}",
                gs.morphs.Box(pos=(x, y, self.ground_z(x, y) + 0.45),
                              size=(0.10, 0.10, 0.90)),
                gs.materials.Rigid(), **beacon_kw,
            )

    # ----- simulation phase ------------------------------------------------
    def _place(self, engine: GenesisPhysicsEngine, name: str,
               pos: np.ndarray, quat: np.ndarray) -> None:
        try:
            engine.set_body_pose(name, pos.astype(np.float32), quat)
            engine.set_body_velocity(name, np.zeros(3, np.float32),
                                     np.zeros(3, np.float32))
        except Exception:  # noqa: BLE001
            pass

    def update_cable_visual(
        self,
        engine: GenesisPhysicsEngine,
        reel: CableReel,
        gripper_world: Optional[np.ndarray],
    ) -> None:
        """Redraw one reel's cable.

        - STORED: park links far below the floor (hidden).
        - BEING_PULLED: drape links from the reel handle to the gripper.
        - DEPLOYED: drape links along the recorded path; let any leftover hide.
        """
        if reel.state is CableReelState.STORED:
            # Tuck links underground so they're not visible.
            for k in range(reel.link_count):
                self._place(
                    engine,
                    reel.link_name(k),
                    np.array([reel.post_xy[0], reel.post_xy[1], -50.0 - k]),
                    _yaw_quat(0.0),
                )
            return

        if reel.state is CableReelState.BEING_PULLED:
            if gripper_world is None:
                return
            p0 = reel.handle_world
            p1 = np.asarray(gripper_world, dtype=np.float64)
            self._drape_links_straight(engine, reel, p0, p1)
            return

        if reel.state is CableReelState.DEPLOYED:
            path = reel.deployed_path or []
            if len(path) < 2:
                # Degenerate: nothing to lay down.
                return
            self._drape_links_along_path(engine, reel, path)
            return

    def _drape_links_straight(self, engine: GenesisPhysicsEngine,
                              reel: CableReel,
                              p0: np.ndarray, p1: np.ndarray) -> None:
        """Slide links uniformly along the segment p0 -> p1, sagging in the middle."""
        span = float(np.linalg.norm(p1[:2] - p0[:2]))
        sag = min(0.35, 0.05 * span)
        seg_yaw = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        quat = _yaw_quat(seg_yaw)
        n = reel.link_count
        for i in range(n):
            s = i / max(n - 1, 1)
            xy = p0[:2] * (1 - s) + p1[:2] * s
            z = p0[2] * (1 - s) + p1[2] * s - sag * math.sin(math.pi * s)
            floor = self.ground_z(float(xy[0]), float(xy[1])) + 0.03
            z = max(z, floor)
            self._place(engine, reel.link_name(i),
                        np.array([xy[0], xy[1], z]), quat)

    def _drape_links_along_path(self, engine: GenesisPhysicsEngine,
                                reel: CableReel,
                                path: list[np.ndarray]) -> None:
        """Lay links along a polyline (resampled to one link per slot)."""
        # Build cumulative arc length.
        pts = np.asarray(path, dtype=np.float64)  # (N, 2)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(np.sum(seg))
        if total < 1e-6:
            return
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        n = reel.link_count
        for i in range(n):
            s = i / max(n - 1, 1) * total
            # Find segment.
            k = int(np.searchsorted(cum, s, side="right") - 1)
            k = max(0, min(k, len(seg) - 1))
            t = 0.0 if seg[k] <= 1e-9 else float((s - cum[k]) / seg[k])
            xy = pts[k] * (1 - t) + pts[k + 1] * t
            yaw = math.atan2(pts[k + 1][1] - pts[k][1],
                             pts[k + 1][0] - pts[k][0])
            z = self.ground_z(float(xy[0]), float(xy[1])) + 0.025
            self._place(engine, reel.link_name(i),
                        np.array([xy[0], xy[1], z]), _yaw_quat(yaw))


# ---------------------------------------------------------------------------
# Arm bridge (lifted, simplified, from demo_rover_drive.ArmBridge)
# ---------------------------------------------------------------------------


class ArmBridge:
    """Mirror SerialArm joint targets into Genesis via PD position control."""

    def __init__(self, entity, arm: SerialArm, num_dof: int,
                 kp: float = 80.0, kv: float = 4.0) -> None:
        self._entity = entity
        self._arm = arm
        self._num_dof = num_dof
        self._dof_local = self._resolve_dofs(num_dof)
        self._gripper_dof_local = self._resolve_gripper_dofs()
        self._configure_gains(kp, kv)

    def _resolve_dofs(self, num_dof: int) -> list[int]:
        entity_dof_start = int(getattr(self._entity, "_dof_start", 0))
        return [
            int(self._entity.get_joint(name=f"arm_joint_{i}").dof_start)
            - entity_dof_start
            for i in range(1, num_dof + 1)
        ]

    def _resolve_gripper_dofs(self) -> list[int]:
        entity_dof_start = int(getattr(self._entity, "_dof_start", 0))
        out: list[int] = []
        for name in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
            try:
                j = self._entity.get_joint(name=name)
                out.append(int(j.dof_start) - entity_dof_start)
            except Exception:
                return []
        return out

    def _configure_gains(self, kp: float, kv: float) -> None:
        all_dofs = self._dof_local + self._gripper_dof_local
        if not all_dofs:
            return
        try:
            self._entity.set_dofs_kp(
                kp=np.full(len(all_dofs), kp, dtype=np.float32),
                dofs_idx_local=all_dofs,
            )
            self._entity.set_dofs_kv(
                kv=np.full(len(all_dofs), kv, dtype=np.float32),
                dofs_idx_local=all_dofs,
            )
        except (AttributeError, TypeError):
            pass

    def snap(self) -> None:
        """Teleport joints (and zero velocity) to the current SerialArm targets."""
        targets = np.asarray(self._arm.get_state().joint_positions, dtype=np.float32)
        if targets.size == self._num_dof:
            try:
                self._entity.set_dofs_position(
                    position=targets, dofs_idx_local=self._dof_local,
                )
                self._entity.set_dofs_velocity(
                    velocity=np.zeros(self._num_dof, dtype=np.float32),
                    dofs_idx_local=self._dof_local,
                )
            except (AttributeError, TypeError):
                pass
        if self._gripper_dof_local:
            half = 0.5 * float(self._arm.get_state().gripper_position)
            try:
                self._entity.set_dofs_position(
                    position=np.array([half, half], dtype=np.float32),
                    dofs_idx_local=self._gripper_dof_local,
                )
                self._entity.set_dofs_velocity(
                    velocity=np.zeros(2, dtype=np.float32),
                    dofs_idx_local=self._gripper_dof_local,
                )
            except (AttributeError, TypeError):
                pass
        self.sync()

    def sync(self) -> None:
        targets = np.asarray(self._arm.get_state().joint_positions, dtype=np.float32)
        if targets.size != self._num_dof:
            return
        try:
            self._entity.control_dofs_position(
                position=targets, dofs_idx_local=self._dof_local,
            )
        except (AttributeError, TypeError):
            return
        if self._gripper_dof_local:
            half = 0.5 * float(self._arm.get_state().gripper_position)
            try:
                self._entity.control_dofs_position(
                    position=np.array([half, half], dtype=np.float32),
                    dofs_idx_local=self._gripper_dof_local,
                )
            except (AttributeError, TypeError):
                pass


# ---------------------------------------------------------------------------
# Arm pose library — joint-angle waypoints (more reliable than IK for a demo)
# ---------------------------------------------------------------------------
#
# The 4-DOF arm in rover.yaml is wired as:
#   joint 1: base yaw      (axis Z, ±π)
#   joint 2: shoulder      (axis Y, ±π/2)   -π/2 lifts vertical
#   joint 3: elbow         (axis Y, ±π/2)
#   joint 4: wrist pitch   (axis Y, ±π/2)
# Joint-2 negative = arm rotates *up* (since it lies along +X out of the base).


# Compact stow: shoulder + elbow folded over the chassis (well clear of wheels).
ARM_STOW = np.array([0.0, -1.45, -1.45, 0.0])

# A high "carry" pose held above the chassis. Used while driving with cable.
ARM_CARRY = np.array([0.0, -1.10, -1.10, 0.20])

# Reach forward and down to the reel handle. Joint 2 = -0.55 brings the end-
# effector forward and low; joint 3 = -0.35 extends; wrist down to grasp.
ARM_REACH_FORWARD_LOW = np.array([0.0, -0.55, -0.35, -0.50])

# Reach forward and DOWN further to lay the cable on regolith.
ARM_LAY_DOWN = np.array([0.0, -0.20, -0.15, -0.80])

# Side-sweep pose for tapping beacons to the rover's left / right.
ARM_TAP_LEFT = np.array([+1.10, -0.50, -0.40, -0.30])
ARM_TAP_RIGHT = np.array([-1.10, -0.50, -0.40, -0.30])

# "Wave" pose used as a victory gesture between cable jobs.
ARM_WAVE_HIGH = np.array([0.0, -1.40, -0.10, 0.60])


def interp_joints(a: np.ndarray, b: np.ndarray, s: float) -> np.ndarray:
    s = float(np.clip(s, 0.0, 1.0))
    # Smoothstep for nicer motion than linear.
    s = s * s * (3.0 - 2.0 * s)
    return (1.0 - s) * a + s * b


def add_alive_bob(q: np.ndarray, sim_t: float, gain: float = 0.06) -> np.ndarray:
    """Add a small breathing motion so the arm never looks frozen."""
    out = q.copy()
    out[0] += gain * 0.6 * math.sin(0.7 * sim_t)
    out[3] += gain * math.sin(1.1 * sim_t + 0.7)
    return out


# ---------------------------------------------------------------------------
# Scripted controller (state machine)
# ---------------------------------------------------------------------------


class Phase(Enum):
    SETTLE = "settle"                    # let the rover drop onto its wheels
    DRIVE_TO_REEL = "drive_to_reel"      # head toward standoff pose in front of reel
    ALIGN_REEL = "align_reel"            # square up yaw
    ARM_REACH = "arm_reach"              # slew arm to grasp pose
    ARM_CLOSE = "arm_close"              # close gripper
    ARM_LIFT = "arm_lift"                # lift to carry pose
    PULL_AWAY = "pull_away"              # drive away from base, cable trails
    ARM_LOWER = "arm_lower"              # set the cable end on the ground
    ARM_OPEN = "arm_open"                # release gripper, cable freezes deployed
    RETREAT = "retreat"                  # back off so cable isn't under the wheels
    ARM_STOW = "arm_stow"                # back to neutral
    DRIVE_TO_BEACON = "drive_to_beacon"  # detour to an inspection beacon
    TAP_BEACON = "tap_beacon"            # sweep arm to touch beacon
    ROTATE_TO_NEXT = "rotate_to_next"    # spin toward the next reel
    NEXT_REEL = "next_reel"              # bookkeeping step
    DONE = "done"


@dataclass
class _PoseSlew:
    """Track an in-progress arm pose interpolation."""

    src: np.ndarray
    dst: np.ndarray
    duration: float
    started_at: float

    def value(self, sim_t: float) -> tuple[np.ndarray, bool]:
        if self.duration <= 0.0:
            return self.dst.copy(), True
        s = (sim_t - self.started_at) / self.duration
        done = s >= 1.0
        return interp_joints(self.src, self.dst, s), done


class CableDeployController:
    """Drives the rover and arm through the full cable-deploy task."""

    # Tunables: kept conservative so the rover never lifts a wheel under torque.
    DRIVE_SPEED_MPS = 0.60
    TURN_SPEED_RADPS = 0.90
    STOP_TOL_M = 0.25
    YAW_TOL_RAD = math.radians(6.0)
    PULL_DISTANCE_M = 3.5            # how far away from the base we lay each cable
    RETREAT_DISTANCE_M = 0.6
    SETTLE_SECONDS = 1.5

    # Standoff distance from the reel post when grasping the cable. The arm
    # base is +0.5 m forward of chassis center, the reach is ~2 m, and we want
    # the handle (0.75 m off ground) inside a comfortable forward-low pose,
    # so park the chassis ~0.9 m from the reel.
    REEL_STANDOFF_M = 1.10

    def __init__(self, world: CableWorld, arm: SerialArm) -> None:
        self._world = world
        self._arm = arm
        self._reels = world.reels
        self._reel_idx = 0
        self._beacon_idx = 0
        self._phase = Phase.SETTLE
        self._phase_t0 = 0.0
        self._slew: Optional[_PoseSlew] = None
        # Where to drive to in the current phase, and what yaw to face there.
        self._target_xy: Optional[np.ndarray] = None
        self._target_yaw: Optional[float] = None
        self._retreat_anchor: Optional[np.ndarray] = None
        self._pull_anchor: Optional[np.ndarray] = None

    # ----- public --------------------------------------------------------
    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def status_line(self) -> str:
        return f"phase={self._phase.value:<16s} reel={self._reel_idx}"

    def update(
        self,
        sim_t: float,
        rover_pos: np.ndarray,
        yaw: float,
        gripper_world: np.ndarray,
    ) -> DriveCommand:
        """Run one tick of the state machine.

        Side-effects: updates the SerialArm joint targets and any active reel
        states. Returns the DriveCommand to issue to the drive system this
        step.
        """
        # Update cable state for the *currently active* reel so its visual
        # tracks the gripper.
        if self._reel_idx < len(self._reels):
            cur = self._reels[self._reel_idx]
            if cur.state is CableReelState.BEING_PULLED:
                # Record the rover position into the breadcrumb every ~0.25 m
                # so the deployed cable matches the actual path.
                path = cur.deployed_path or []
                if not path:
                    path = [cur.post_xy.copy(), rover_pos[:2].copy()]
                else:
                    last = path[-1]
                    if float(np.linalg.norm(rover_pos[:2] - last)) > 0.25:
                        path.append(rover_pos[:2].copy())
                cur.deployed_path = path
                cur.deploy_distance_m = sum(
                    float(np.linalg.norm(path[k + 1] - path[k]))
                    for k in range(len(path) - 1)
                )

        return self._dispatch(sim_t, rover_pos, yaw, gripper_world)

    # ----- dispatch ------------------------------------------------------
    def _dispatch(self, sim_t: float, rover_pos: np.ndarray,
                  yaw: float, gripper_world: np.ndarray) -> DriveCommand:
        fn = {
            Phase.SETTLE: self._do_settle,
            Phase.DRIVE_TO_REEL: self._do_drive_to_reel,
            Phase.ALIGN_REEL: self._do_align_reel,
            Phase.ARM_REACH: self._do_arm_reach,
            Phase.ARM_CLOSE: self._do_arm_close,
            Phase.ARM_LIFT: self._do_arm_lift,
            Phase.PULL_AWAY: self._do_pull_away,
            Phase.ARM_LOWER: self._do_arm_lower,
            Phase.ARM_OPEN: self._do_arm_open,
            Phase.RETREAT: self._do_retreat,
            Phase.ARM_STOW: self._do_arm_stow,
            Phase.DRIVE_TO_BEACON: self._do_drive_to_beacon,
            Phase.TAP_BEACON: self._do_tap_beacon,
            Phase.ROTATE_TO_NEXT: self._do_rotate_to_next,
            Phase.NEXT_REEL: self._do_next_reel,
            Phase.DONE: self._do_done,
        }[self._phase]
        return fn(sim_t, rover_pos, yaw)

    # ----- phase helpers -------------------------------------------------
    def _enter(self, phase: Phase, sim_t: float) -> None:
        self._phase = phase
        self._phase_t0 = sim_t

    def _begin_slew(self, target: np.ndarray, sim_t: float,
                    duration: float) -> None:
        cur = np.asarray(self._arm.get_state().joint_positions,
                         dtype=np.float64).copy()
        self._slew = _PoseSlew(src=cur, dst=target.astype(np.float64),
                               duration=duration, started_at=sim_t)

    def _advance_slew(self, sim_t: float, alive: bool = True) -> bool:
        if self._slew is None:
            self._arm.set_joint_positions(np.zeros(4))
            return True
        q, done = self._slew.value(sim_t)
        if alive:
            q = add_alive_bob(q, sim_t)
        self._arm.set_joint_positions(q)
        if done:
            self._slew = None
        return done

    def _hold_arm_with_bob(self, base: np.ndarray, sim_t: float) -> None:
        self._arm.set_joint_positions(add_alive_bob(base, sim_t))

    # ----- drive primitives ---------------------------------------------
    def _drive_toward(self, rover_xy: np.ndarray, yaw: float,
                      target_xy: np.ndarray) -> DriveCommand:
        delta = target_xy - rover_xy
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return DriveCommand(linear_velocity_mps=0.0,
                                angular_velocity_radps=0.0)
        desired_yaw = math.atan2(delta[1], delta[0])
        yaw_err = _wrap_pi(desired_yaw - yaw)
        # Allow forward translation even with moderate yaw error — skid-steer
        # rotation on lunar friction is slow, so stalling v to 0 turns "drive
        # to point" into "spin in place for an eternity". A floor of 0.25 means
        # the rover always creeps toward its goal while it corrects yaw.
        v_scale = max(0.25, math.cos(yaw_err))
        v = self.DRIVE_SPEED_MPS * v_scale * min(1.0, dist / 0.7)
        w = float(np.clip(2.0 * yaw_err, -self.TURN_SPEED_RADPS,
                          self.TURN_SPEED_RADPS))
        return DriveCommand(linear_velocity_mps=v, angular_velocity_radps=w)

    def _drive_backward_until(self, anchor_xy: np.ndarray, rover_xy: np.ndarray,
                              distance: float) -> tuple[DriveCommand, bool]:
        travelled = float(np.linalg.norm(rover_xy - anchor_xy))
        done = travelled >= distance
        if done:
            return DriveCommand(0.0, 0.0), True
        return DriveCommand(-0.35, 0.0), False

    def _rotate_to(self, yaw: float, target_yaw: float) -> tuple[DriveCommand, bool]:
        err = _wrap_pi(target_yaw - yaw)
        if abs(err) < self.YAW_TOL_RAD:
            return DriveCommand(0.0, 0.0), True
        w = float(np.clip(1.4 * err, -self.TURN_SPEED_RADPS,
                          self.TURN_SPEED_RADPS))
        return DriveCommand(0.0, w), False

    # ----- per-phase logic ----------------------------------------------
    def _do_settle(self, sim_t: float, rover_pos: np.ndarray,
                   yaw: float) -> DriveCommand:
        # Hold stow and wait for the rover to drop onto its suspension.
        self._hold_arm_with_bob(ARM_STOW, sim_t)
        if sim_t - self._phase_t0 >= self.SETTLE_SECONDS:
            self._enter(Phase.DRIVE_TO_REEL, sim_t)
            self._setup_reel_target()
        return DriveCommand(0.0, 0.0)

    def _setup_reel_target(self) -> None:
        reel = self._reels[self._reel_idx]
        # Stand off in front of the reel along the line from the moonbase →
        # reel (so the rover approaches from the open side, not through the
        # base). The yaw target points at the reel.
        from_base = reel.post_xy - self._world.base_xy
        dir_base = from_base / max(float(np.linalg.norm(from_base)), 1e-6)
        self._target_xy = reel.post_xy + dir_base * self.REEL_STANDOFF_M
        self._target_yaw = math.atan2(-dir_base[1], -dir_base[0])

    def _do_drive_to_reel(self, sim_t: float, rover_pos: np.ndarray,
                          yaw: float) -> DriveCommand:
        assert self._target_xy is not None
        self._hold_arm_with_bob(ARM_STOW, sim_t)
        dist = float(np.linalg.norm(rover_pos[:2] - self._target_xy))
        if dist < self.STOP_TOL_M:
            self._enter(Phase.ALIGN_REEL, sim_t)
            return DriveCommand(0.0, 0.0)
        return self._drive_toward(rover_pos[:2], yaw, self._target_xy)

    def _do_align_reel(self, sim_t: float, rover_pos: np.ndarray,
                       yaw: float) -> DriveCommand:
        self._hold_arm_with_bob(ARM_STOW, sim_t)
        assert self._target_yaw is not None
        cmd, done = self._rotate_to(yaw, self._target_yaw)
        if done:
            self._begin_slew(ARM_REACH_FORWARD_LOW, sim_t, duration=2.2)
            self._enter(Phase.ARM_REACH, sim_t)
        return cmd

    def _do_arm_reach(self, sim_t: float, rover_pos: np.ndarray,
                      yaw: float) -> DriveCommand:
        if self._advance_slew(sim_t, alive=False):
            self._enter(Phase.ARM_CLOSE, sim_t)
        return DriveCommand(0.0, 0.0)

    def _do_arm_close(self, sim_t: float, rover_pos: np.ndarray,
                      yaw: float) -> DriveCommand:
        self._hold_arm_with_bob(ARM_REACH_FORWARD_LOW, sim_t)
        # Drive the gripper closed over ~0.8 s.
        t = (sim_t - self._phase_t0) / 0.8
        self._arm.command_gripper(float(np.clip(1.0 - t, 0.0, 1.0)))
        if sim_t - self._phase_t0 >= 1.0:
            # Mark the cable as being pulled — visuals will start tracking gripper.
            reel = self._reels[self._reel_idx]
            reel.state = CableReelState.BEING_PULLED
            reel.deployed_path = [reel.post_xy.copy(), rover_pos[:2].copy()]
            self._begin_slew(ARM_CARRY, sim_t, duration=1.6)
            self._enter(Phase.ARM_LIFT, sim_t)
        return DriveCommand(0.0, 0.0)

    def _do_arm_lift(self, sim_t: float, rover_pos: np.ndarray,
                     yaw: float) -> DriveCommand:
        if self._advance_slew(sim_t, alive=True):
            # Capture the anchor used to measure pull distance.
            self._pull_anchor = rover_pos[:2].copy()
            self._enter(Phase.PULL_AWAY, sim_t)
        return DriveCommand(0.0, 0.0)

    def _do_pull_away(self, sim_t: float, rover_pos: np.ndarray,
                      yaw: float) -> DriveCommand:
        assert self._pull_anchor is not None
        # Keep the carry pose but let the alive bob run so the arm visibly moves.
        self._hold_arm_with_bob(ARM_CARRY, sim_t)
        travelled = float(np.linalg.norm(rover_pos[:2] - self._pull_anchor))
        if travelled >= self.PULL_DISTANCE_M:
            self._begin_slew(ARM_LAY_DOWN, sim_t, duration=2.0)
            self._enter(Phase.ARM_LOWER, sim_t)
            return DriveCommand(0.0, 0.0)
        # Back away while still facing the reel — same way a person lays
        # cable from a reel. A 180° pivot would take ~minute under lunar
        # skid-steer friction; reversing is instantaneous and keeps the arm
        # pointed at the cable origin so the visual stays clean.
        return DriveCommand(linear_velocity_mps=-self.DRIVE_SPEED_MPS,
                            angular_velocity_radps=0.0)

    def _do_arm_lower(self, sim_t: float, rover_pos: np.ndarray,
                      yaw: float) -> DriveCommand:
        if self._advance_slew(sim_t, alive=False):
            self._enter(Phase.ARM_OPEN, sim_t)
        return DriveCommand(0.0, 0.0)

    def _do_arm_open(self, sim_t: float, rover_pos: np.ndarray,
                     yaw: float) -> DriveCommand:
        self._hold_arm_with_bob(ARM_LAY_DOWN, sim_t)
        t = (sim_t - self._phase_t0) / 0.8
        self._arm.command_gripper(float(np.clip(t, 0.0, 1.0)))
        if sim_t - self._phase_t0 >= 1.0:
            # Cable released — freeze the path as deployed.
            reel = self._reels[self._reel_idx]
            # Add the final drop point and any intermediate point so the
            # deployed cable cleanly terminates at the gripper position.
            if reel.deployed_path is None:
                reel.deployed_path = [reel.post_xy.copy(), rover_pos[:2].copy()]
            else:
                reel.deployed_path.append(rover_pos[:2].copy())
            reel.state = CableReelState.DEPLOYED
            self._retreat_anchor = rover_pos[:2].copy()
            self._enter(Phase.RETREAT, sim_t)
        return DriveCommand(0.0, 0.0)

    def _do_retreat(self, sim_t: float, rover_pos: np.ndarray,
                    yaw: float) -> DriveCommand:
        assert self._retreat_anchor is not None
        self._hold_arm_with_bob(ARM_LAY_DOWN, sim_t)
        cmd, done = self._drive_backward_until(
            self._retreat_anchor, rover_pos[:2], self.RETREAT_DISTANCE_M
        )
        if done:
            self._begin_slew(ARM_STOW, sim_t, duration=1.6)
            self._enter(Phase.ARM_STOW, sim_t)
        return cmd

    def _do_arm_stow(self, sim_t: float, rover_pos: np.ndarray,
                     yaw: float) -> DriveCommand:
        if self._advance_slew(sim_t, alive=False):
            # If a beacon is queued, take a detour; otherwise rotate to next reel.
            if self._beacon_idx < len(self._world.beacons_xy):
                bxy = self._world.beacons_xy[self._beacon_idx]
                # Stand off 1.2 m from beacon on the rover side.
                to_b = bxy - rover_pos[:2]
                d = float(np.linalg.norm(to_b))
                if d > 1e-6:
                    to_b /= d
                    self._target_xy = bxy - to_b * 1.2
                    self._target_yaw = math.atan2(to_b[1], to_b[0])
                    self._enter(Phase.DRIVE_TO_BEACON, sim_t)
                    return DriveCommand(0.0, 0.0)
            self._enter(Phase.ROTATE_TO_NEXT, sim_t)
        return DriveCommand(0.0, 0.0)

    def _do_drive_to_beacon(self, sim_t: float, rover_pos: np.ndarray,
                            yaw: float) -> DriveCommand:
        assert self._target_xy is not None
        self._hold_arm_with_bob(ARM_STOW, sim_t)
        dist = float(np.linalg.norm(rover_pos[:2] - self._target_xy))
        if dist < self.STOP_TOL_M:
            # Decide which side the beacon is on relative to the rover yaw so
            # we pick a LEFT or RIGHT tap pose.
            assert self._target_yaw is not None
            bxy = self._world.beacons_xy[self._beacon_idx]
            to_b = bxy - rover_pos[:2]
            yaw_to_b = math.atan2(to_b[1], to_b[0])
            side = _wrap_pi(yaw_to_b - yaw)
            target_pose = ARM_TAP_LEFT if side > 0 else ARM_TAP_RIGHT
            self._begin_slew(target_pose, sim_t, duration=1.6)
            self._enter(Phase.TAP_BEACON, sim_t)
            return DriveCommand(0.0, 0.0)
        return self._drive_toward(rover_pos[:2], yaw, self._target_xy)

    def _do_tap_beacon(self, sim_t: float, rover_pos: np.ndarray,
                       yaw: float) -> DriveCommand:
        # Slew out → hold a moment → slew back to stow.
        elapsed = sim_t - self._phase_t0
        if elapsed < 1.8:
            self._advance_slew(sim_t, alive=False)
        elif elapsed < 2.6:
            # Hold the tap pose.
            pass
        else:
            # Slew back to stow.
            if self._slew is None:
                self._begin_slew(ARM_STOW, sim_t, duration=1.4)
            if self._advance_slew(sim_t, alive=False):
                self._beacon_idx += 1
                self._enter(Phase.ROTATE_TO_NEXT, sim_t)
        return DriveCommand(0.0, 0.0)

    def _do_rotate_to_next(self, sim_t: float, rover_pos: np.ndarray,
                           yaw: float) -> DriveCommand:
        next_idx = self._reel_idx + 1
        # Victory wave while spinning between cables.
        elapsed = sim_t - self._phase_t0
        if elapsed < 1.2:
            if self._slew is None:
                self._begin_slew(ARM_WAVE_HIGH, sim_t, duration=1.2)
            self._advance_slew(sim_t, alive=True)
        elif elapsed < 2.4:
            if self._slew is not None:
                self._advance_slew(sim_t, alive=True)
            else:
                # second half of wave - back to stow
                self._begin_slew(ARM_STOW, sim_t, duration=1.0)
        else:
            self._hold_arm_with_bob(ARM_STOW, sim_t)

        if next_idx >= len(self._reels):
            self._enter(Phase.DONE, sim_t)
            return DriveCommand(0.0, 0.0)

        # Aim toward the next reel's standoff pose so the rotation puts us in
        # a useful heading.
        target_reel = self._reels[next_idx]
        from_base = target_reel.post_xy - self._world.base_xy
        dir_base = from_base / max(float(np.linalg.norm(from_base)), 1e-6)
        standoff = target_reel.post_xy + dir_base * self.REEL_STANDOFF_M
        delta = standoff - rover_pos[:2]
        desired_yaw = math.atan2(delta[1], delta[0])
        cmd, done = self._rotate_to(yaw, desired_yaw)
        if done and elapsed > 2.0:
            self._enter(Phase.NEXT_REEL, sim_t)
        return cmd

    def _do_next_reel(self, sim_t: float, rover_pos: np.ndarray,
                      yaw: float) -> DriveCommand:
        self._reel_idx += 1
        if self._reel_idx >= len(self._reels):
            self._enter(Phase.DONE, sim_t)
            return DriveCommand(0.0, 0.0)
        self._setup_reel_target()
        self._enter(Phase.DRIVE_TO_REEL, sim_t)
        return DriveCommand(0.0, 0.0)

    def _do_done(self, sim_t: float, rover_pos: np.ndarray,
                 yaw: float) -> DriveCommand:
        # Slow gentle wave on loop so the demo doesn't look frozen.
        wave = ARM_WAVE_HIGH.copy()
        wave[0] += 0.4 * math.sin(0.8 * sim_t)
        self._arm.set_joint_positions(wave)
        return DriveCommand(0.0, 0.0)


# ---------------------------------------------------------------------------
# URDF / config builder helpers (reused from demo_rover_drive style)
# ---------------------------------------------------------------------------


def load_rover_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_rover_urdf(rover_cfg: dict, profile_name: str) -> str:
    profile = rover_cfg["profiles"][profile_name]
    config = RoverComposer._profile_to_urdf_config(
        rover_id=f"demo_{profile_name}",
        profile=profile, rover_cfg=rover_cfg,
    )
    return GenesisURDFBuilder().build_rover(config)


def _urdf_to_tempfile(urdf_xml: str) -> str:
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        suffix=".urdf", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(urdf_xml)
    tmp.flush()
    tmp.close()
    return tmp.name


def spawn_height_for(rover_cfg: dict, profile: dict) -> float:
    """Spawn height that puts wheels just above ground."""
    wheel_radius = float(profile["wheel_radius_m"])
    dims = profile.get("dimensions") or rover_cfg.get("structure", {}).get(
        "dimensions", [2.0, 1.5, 0.8]
    )
    half_h = float(dims[2]) / 2.0
    wheel_z = float(profile.get("wheel_positions", [[0.0, 0.0, 0.0]])[0][2])
    axle_below_origin = max(0.0, half_h - wheel_z)
    return axle_below_origin + wheel_radius + 0.05


def make_genesis_config(physics_path: Path, sim_hz: float, use_gpu: bool) -> GenesisConfig:
    base = GenesisConfig.from_yaml(str(physics_path))
    timestep = base.timestep if sim_hz <= 0 else 1.0 / sim_hz
    return replace(base, timestep=timestep, use_gpu=use_gpu)


# ---------------------------------------------------------------------------
# Progress spinner (compile-time heartbeat)
# ---------------------------------------------------------------------------


class ProgressSpinner:
    _GLYPHS = ("|", "/", "-", "\\")

    def __init__(self, label: str, tick_s: float = 1.0) -> None:
        self._label = label
        self._tick_s = float(tick_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0
        self._is_tty = sys.stdout.isatty()

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, name=f"spinner-{self._label}", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        elapsed = time.perf_counter() - self._t0
        if self._is_tty:
            sys.stdout.write("\r")
        suffix = "OK" if exc is None else "FAIL"
        print(f"  {self._label}: {elapsed:6.2f}s  [{suffix}]")
        sys.stdout.flush()

    def _run(self) -> None:
        i = 0
        next_tick = self._t0
        while not self._stop.is_set():
            now = time.perf_counter()
            if now >= next_tick:
                elapsed = now - self._t0
                glyph = self._GLYPHS[i % len(self._GLYPHS)]
                msg = f"  {self._label}: {glyph} {elapsed:6.2f}s elapsed"
                if self._is_tty:
                    sys.stdout.write("\r" + msg)
                else:
                    print(msg)
                sys.stdout.flush()
                i += 1
                next_tick = self._t0 + i * self._tick_s
            self._stop.wait(timeout=0.1)


# ---------------------------------------------------------------------------
# Gripper world position helper
# ---------------------------------------------------------------------------


def gripper_world_xyz(arm: SerialArm, rover_pos: np.ndarray,
                      yaw: float, base_offset_xyz=(0.5, 0.0, 0.3)) -> np.ndarray:
    """Estimate the gripper end-effector position in the world frame.

    Uses the arm's analytic FK and the rover pose; assumes the rover is upright
    (small roll/pitch) which is true for the 4-wheel skid platform.
    """
    q = np.asarray(arm.get_state().joint_positions, dtype=np.float64).flatten()
    T = arm.forward_kinematics(q)
    ee_body = T[:3, 3] + np.asarray(base_offset_xyz, dtype=np.float64)
    cy = math.cos(yaw); sy = math.sin(yaw)
    R = np.array([[cy, -sy, 0.0],
                  [sy,  cy, 0.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return np.asarray(rover_pos, dtype=np.float64).reshape(3) + R @ ee_body


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_header(args: argparse.Namespace, num_reels: int) -> None:
    print("=" * 92)
    print("  Moon Rover - Scripted Cable Deployment Demo")
    print("=" * 92)
    print(f"  Profile     : {PROFILE_NAME} (4-wheel skid steer, stable platform)")
    print(f"  Backend     : {args.backend.upper()}  sim @ {args.sim_hz:.0f} Hz")
    print(f"  Cables      : {num_reels} reels to deploy from the moonbase")
    print("  Controller  : scripted state machine (no RL, no MPC)")
    if args.backend == "gpu":
        print("  *** GPU FIRST RUN: Genesis will compile CUDA kernels (~5-10 min). ***")
    print("=" * 92)


def main() -> int:
    args = parse_args()

    rover_yaml = load_rover_yaml(DEFAULT_ROVER)
    profile = rover_yaml["profiles"][PROFILE_NAME]
    arm_cfg_yaml = rover_yaml.get("arm", {})

    urdf_xml = build_rover_urdf(rover_yaml, PROFILE_NAME)
    urdf_path = _urdf_to_tempfile(urdf_xml)

    cfg = make_genesis_config(DEFAULT_PHYSICS, args.sim_hz,
                              use_gpu=(args.backend == "gpu"))

    num_reels = max(1, min(args.num_reels, 4))
    world = CableWorld(num_reels=num_reels)
    spawn_xy = world.spawn_xy

    viewer_options = None
    if not args.no_viewer:
        viewer_options = gs.options.ViewerOptions(
            max_FPS=args.render_hz,
            refresh_rate=args.render_hz,
            run_in_thread=True,
            camera_pos=(spawn_xy[0] + 6.0, spawn_xy[1] - 6.0, 5.0),
            camera_lookat=(spawn_xy[0] - 2.0, spawn_xy[1] - 2.0, 0.5),
            camera_fov=50,
            enable_default_keybinds=True,
        )

    print_header(args, num_reels)

    engine = GenesisPhysicsEngine()
    completed_cleanly = False
    try:
        configure_label = (
            "Compiling Genesis CUDA kernels (first GPU run can take 5-10 min)"
            if args.backend == "gpu"
            else "Initialising Genesis (first CPU run can take 0-5 min)"
        )
        with ProgressSpinner(configure_label):
            engine.configure(cfg, show_viewer=not args.no_viewer,
                             viewer_options=viewer_options)

        world.construct(engine)

        spawn_z = spawn_height_for(rover_yaml, profile)
        # Initial yaw is set via set_body_pose after build_scene below; the
        # URDF morph itself doesn't accept an `euler` kwarg on this Genesis
        # build.
        base_xy = world.base_xy
        spawn_yaw_rad = math.atan2(
            base_xy[1] - spawn_xy[1], base_xy[0] - spawn_xy[0]
        )
        rover_morph = gs.morphs.URDF(
            file=urdf_path, fixed=False,
            pos=(spawn_xy[0], spawn_xy[1], spawn_z),
        )
        engine.add_entity(
            ROVER_NAME, rover_morph,
            gs.materials.Rigid(friction=1.2),
        )

        with ProgressSpinner("Building scene"):
            engine.build_scene()

        # Orient the rover toward the moonbase reels. set_body_pose uses
        # (w, x, y, z) quaternion ordering (see _reset_pose in
        # demo_rover_drive.py for the convention).
        spawn_quat = np.array(
            [math.cos(spawn_yaw_rad * 0.5), 0.0, 0.0,
             math.sin(spawn_yaw_rad * 0.5)],
            dtype=np.float32,
        )
        try:
            engine.set_body_pose(
                ROVER_NAME,
                np.array([spawn_xy[0], spawn_xy[1], spawn_z], dtype=np.float32),
                spawn_quat,
            )
            engine.set_body_velocity(
                ROVER_NAME,
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: could not set initial rover yaw ({exc}); "
                  "rover will start at default heading")

        drive_config = drive_config_from_profile(profile, DriveType.FOUR_WHEEL_SKID)
        drive = create_drive_system(drive_config)
        drive.attach(engine, ROVER_NAME)

        # 4-DOF arm matching configs/rover.yaml.
        arm = SerialArm()
        num_dof = int(arm_cfg_yaml.get("num_dof", 4))
        arm.configure(
            ArmConfig(
                num_dof=num_dof,
                joint_limits=[
                    (float(arm_cfg_yaml["joints"][f"joint_{i}"]["lower_limit_rad"]),
                     float(arm_cfg_yaml["joints"][f"joint_{i}"]["upper_limit_rad"]))
                    for i in range(1, num_dof + 1)
                ],
                reach_m=float(arm_cfg_yaml.get("reach_m", 2.0)),
                payload_kg=float(arm_cfg_yaml.get("payload_kg", 5.0)),
                joint_accuracy_deg=0.5,
            ),
            GripperConfig(
                num_fingers=2,
                max_open_m=float(arm_cfg_yaml.get("gripper", {}).get("stroke_m", 0.1)),
                max_force_n=float(arm_cfg_yaml.get("gripper", {})
                                  .get("max_grip_force_n", 500.0)),
                compliance_model="linear",
            ),
        )
        arm.set_joint_positions(ARM_STOW)
        arm.command_gripper(1.0)  # start open

        arm_bridge = ArmBridge(
            engine.get_entity(ROVER_NAME), arm, num_dof=num_dof,
        )
        arm_bridge.snap()

        controller = CableDeployController(world, arm)

        # ------------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------------
        dt = cfg.timestep
        step_count = 0
        wall_start = time.perf_counter()
        next_report_wall = wall_start
        pace_to_real_time = not args.no_viewer
        done_at_sim_t: Optional[float] = None
        last_logged_phase: Optional[Phase] = None

        while True:
            t = engine.get_sim_time()

            # Read rover pose.
            pos, quat = engine.get_body_pose(ROVER_NAME)
            rover_pos = np.asarray(pos, dtype=np.float64).reshape(3)
            quat_arr = np.asarray(quat).flatten()
            _, _, yaw = _euler_from_quat_wxyz(quat_arr)

            # Estimate the gripper position in the world.
            grip_world = gripper_world_xyz(arm, rover_pos, yaw)

            # Controller step (updates arm joint targets and reel state).
            drive_cmd = controller.update(t, rover_pos, yaw, grip_world)

            # Apply commands.
            drive.command(drive_cmd)
            drive.update(dt)
            arm.update(dt)
            arm_bridge.sync()

            # Step physics.
            engine.step(dt, render=not args.no_viewer)

            # Refresh cable visuals for all reels (cheap; ~108 boxes total).
            for reel in world.reels:
                controller_grip = grip_world if (
                    reel.state is CableReelState.BEING_PULLED
                    and reel is world.reels[controller._reel_idx
                                             if controller._reel_idx < len(world.reels)
                                             else 0]
                ) else None
                world.update_cable_visual(engine, reel, controller_grip)

            step_count += 1

            # Per-phase one-time log line.
            if controller.phase is not last_logged_phase:
                print(f"  t={t:6.2f}s  -> {controller.status_line}  "
                      f"pos=({rover_pos[0]:+5.2f},{rover_pos[1]:+5.2f})  "
                      f"yaw={math.degrees(yaw):+6.1f} deg")
                last_logged_phase = controller.phase

            # Low-frequency heartbeat (every 3 wall seconds) so long phases
            # like DRIVE_TO_REEL show progress without flooding stdout.
            now = time.perf_counter()
            if now >= next_report_wall:
                print(f"  t={t:6.2f}s  step={step_count:5d}  "
                      f"pos=({rover_pos[0]:+5.2f},{rover_pos[1]:+5.2f})  "
                      f"yaw={math.degrees(yaw):+6.1f} deg  "
                      f"{controller.status_line}  "
                      f"v={drive_cmd.linear_velocity_mps:+.2f} m/s  "
                      f"w={drive_cmd.angular_velocity_radps:+.2f} rad/s")
                next_report_wall = now + 3.0

            # End conditions:
            if controller.phase is Phase.DONE and done_at_sim_t is None:
                done_at_sim_t = t
                print("=" * 92)
                print(f"  All {num_reels} cables deployed at sim_t={t:.2f}s.")
                print("=" * 92)
                if args.no_viewer:
                    completed_cleanly = True
                    break

            # In viewer mode, idle in DONE for ~10 s of victory wave, then exit.
            if done_at_sim_t is not None and not args.no_viewer:
                if t - done_at_sim_t > 10.0:
                    completed_cleanly = True
                    break

            if args.max_sim_seconds > 0.0 and t >= args.max_sim_seconds:
                completed_cleanly = True
                break

            # Real-time pacing for the viewer.
            if pace_to_real_time:
                target_wall = wall_start + step_count * dt
                sleep_for = target_wall - time.perf_counter()
                if sleep_for > 0.0005:
                    time.sleep(sleep_for)

    except KeyboardInterrupt:
        print("\n  Stopped by user (Ctrl+C).")
        completed_cleanly = True
    finally:
        if args.destroy:
            try:
                engine.teardown()
            except Exception:
                pass
        try:
            os.unlink(urdf_path)
        except OSError:
            pass

    print("\n  Demo complete." if completed_cleanly else "\n  Demo aborted.")
    return 0 if completed_cleanly else 1


if __name__ == "__main__":
    raise SystemExit(main())
