"""Concrete Genesis 0.4.4 physics engine implementation for the Moon Rover project.

This module provides GenesisPhysicsEngine, which wraps the Genesis physics simulator
and fulfills the PhysicsEngine ABC contract. It also exposes an extended API surface
required by downstream systems (DriveSystem, IMUSensor, LiDARScanner, CableSystem,
AntennaUnit, RL gym wrapper, checkpoint/replay).

Threading: Genesis uses a single global CUDA context. All Genesis API calls
(scene.step, entity.get_pos, etc.) must be serialized to a single thread by the
caller. This engine does not internally lock Genesis calls.

Known limitations:
  - MPM particle state cannot be serialized in Genesis 0.4.4. can_snapshot_mpm = False.
  - Genesis 0.4.4 has no entity.set_quat(). Orientation resets use DOF positions for
    articulated bodies; free rigid bodies cannot have their orientation restored precisely.
  - Genesis contact query API shape must be verified at runtime; isolated in
    _get_raw_contacts() for easy swapping if the API differs.
  - Genesis runtime init is process-global. Repeated configure() calls must reuse the
    same backend/precision/seed/logging settings or fail fast with diagnostics.
  - Set MOON_ROVER_GENESIS_STRICT_DIAGNOSTICS=1 to turn compatibility warnings into
    immediate RuntimeError failures while auditing Genesis API drift.
  - On Windows, safe teardown skips process-global gs.destroy() by default to avoid
    known hang risk. Override with MOON_ROVER_GENESIS_DESTROY_POLICY=always.
"""

from __future__ import annotations

import functools
import logging
import os
import pickle
import tempfile
import threading
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

import genesis as gs


def _to_numpy(tensor_or_array, dtype=np.float32) -> np.ndarray:
    """Convert a Genesis/PyTorch tensor or array-like to a numpy array.

    Handles CUDA tensors by moving them to CPU first.
    """
    try:
        # PyTorch tensor (may be on CUDA)
        return tensor_or_array.detach().cpu().numpy().astype(dtype)
    except AttributeError:
        return np.array(tensor_or_array, dtype=dtype)

from moon_rover.core.physics.engine import GenesisConfig, PhysicsEngine, ScenePhase

logger = logging.getLogger(__name__)

NDArray = npt.NDArray[np.float32]


# ---------------------------------------------------------------------------
# Internal record types
# ---------------------------------------------------------------------------

@dataclass
class _EntityRecord:
    """Tracks a single Genesis entity registered with the engine."""
    name: str
    genesis_entity: Any           # gs.RigidEntity, gs.MPMEntity, etc.
    entity_type: str              # "rigid" | "mpm" | "kinematic" | "terrain"
    prev_lin_vel: NDArray         # (3,) float32 — cached for tracked acceleration telemetry
    prev_ang_vel: NDArray         # (3,) float32 — cached for tracked acceleration telemetry
    accel_tracking_enabled: bool  # Whether this body is subscribed for acceleration telemetry
    accel_prev_step: int          # step_count that prev_* corresponds to; -1 when unprimed


@dataclass
class _RaycasterRecord:
    """Tracks a registered Genesis raycaster sensor."""
    name: str
    genesis_sensor: Any           # gs.sensors.Raycaster instance
    link_entity_name: str
    link_idx: int


@dataclass
class _TerrainRecord:
    """Stores terrain data for fast Python-level height/normal queries."""
    genesis_entity: Any
    height_field: NDArray         # (H, W) float32
    normal_map: NDArray           # (H, W, 3) float32 — precomputed surface normals
    size_x: float
    size_y: float
    resolution_x: int             # W
    resolution_y: int             # H


@dataclass(frozen=True)
class _GenesisRuntimeConfig:
    """Tracks process-global Genesis settings that must stay compatible."""

    backend_name: str
    precision: str
    seed: int
    logging_level: str


# ---------------------------------------------------------------------------
# Phase enforcement decorator
# ---------------------------------------------------------------------------

def _require_phase(*allowed: ScenePhase):
    """Decorator that raises RuntimeError if the engine is not in an allowed phase."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self: "GenesisPhysicsEngine", *args, **kwargs):
            with self._phase_lock:
                if self._phase not in allowed:
                    allowed_names = [p.value for p in allowed]
                    raise RuntimeError(
                        f"{fn.__name__} requires phase in {allowed_names}, "
                        f"but engine is currently in '{self._phase.value}'"
                    )
            return fn(self, *args, **kwargs)
        return wrapper
    return decorator


def _call_first(obj: Any, names: Tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    """Call the first available method from a Genesis API compatibility list."""
    for name in names:
        method = getattr(obj, name, None)
        if method is not None:
            return method(*args, **kwargs)
    raise AttributeError(f"{type(obj).__name__} has none of: {', '.join(names)}")


# ---------------------------------------------------------------------------
# GenesisPhysicsEngine
# ---------------------------------------------------------------------------

class GenesisPhysicsEngine(PhysicsEngine):
    """Full Genesis 0.4.4 implementation of the PhysicsEngine ABC.

    Wraps a gs.Scene and exposes:
      - The 7-method ABC contract (configure, build_scene, step, teardown,
        save_snapshot, restore_snapshot, get_phase, solver_backends)
      - Extended API for entity registration, body state queries/setters,
        terrain queries, raycaster management, and contact queries.

    Usage::

        engine = GenesisPhysicsEngine()
        config = GenesisConfig.from_yaml("configs/physics.yaml")
        engine.configure(config)

        import genesis as gs
        engine.add_entity("ground", gs.morphs.Plane(), gs.materials.Rigid())
        rover = engine.add_entity("rover", gs.morphs.URDF("rover.urdf"), gs.materials.Rigid())
        engine.build_scene()

        for _ in range(1000):
            engine.step(config.timestep)
            pos, quat = engine.get_body_pose("rover")

        engine.teardown()
    """

    # ------------------------------------------------------------------
    # Class-level Genesis singleton guard
    # ------------------------------------------------------------------
    _gs_initialized: ClassVar[bool] = False
    _gs_init_lock: ClassVar[threading.Lock] = threading.Lock()
    _destroy_policy_env: ClassVar[str] = "MOON_ROVER_GENESIS_DESTROY_POLICY"
    _strict_diagnostics_env: ClassVar[str] = "MOON_ROVER_GENESIS_STRICT_DIAGNOSTICS"
    _gs_runtime_config: ClassVar[Optional[_GenesisRuntimeConfig]] = None
    _gs_runtime_owner_count: ClassVar[int] = 0
    _terrain_contact_friction: ClassVar[float] = 1.2
    _terrain_contact_restitution: ClassVar[float] = 0.02
    _terrain_density_kg_m3: ClassVar[float] = 1800.0

    # MPM particle state cannot be serialized in Genesis 0.4.4
    can_snapshot_mpm: ClassVar[bool] = False

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._phase: ScenePhase = ScenePhase.CONSTRUCTION
        self._phase_lock: threading.RLock = threading.RLock()

        self._config: Optional[GenesisConfig] = None
        self._scene: Optional[Any] = None          # gs.Scene
        self._n_envs: int = 1

        self._entities: Dict[str, _EntityRecord] = {}
        self._entity_lock: threading.RLock = threading.RLock()

        self._raycasters: Dict[str, _RaycasterRecord] = {}
        self._terrain: Optional[_TerrainRecord] = None

        self._sim_time: float = 0.0
        self._step_count: int = 0
        self._dt: float = 1.0 / 240.0             # overwritten by configure()
        self._owns_gs_runtime_init: bool = False
        self._claims_gs_runtime: bool = False

        self._solver_backends_cache: Dict[str, str] = {}

    @classmethod
    def _resolve_destroy_policy(cls) -> str:
        """Return the normalized Genesis runtime destroy policy."""
        raw_policy = os.environ.get(cls._destroy_policy_env, "safe").strip().lower()
        if raw_policy in {"safe", "always", "never"}:
            return raw_policy
        logger.warning(
            "Unknown %s=%r; defaulting to safe teardown policy.",
            cls._destroy_policy_env,
            raw_policy,
        )
        return "safe"

    @classmethod
    def _strict_diagnostics_enabled(cls) -> bool:
        """Return whether adapter compatibility issues should raise immediately."""

        raw_value = os.environ.get(cls._strict_diagnostics_env, "").strip().lower()
        if raw_value in {"", "0", "false", "no", "off"}:
            return False
        if raw_value in {"1", "true", "yes", "on"}:
            return True
        logger.warning(
            "Unknown %s=%r; defaulting to non-strict diagnostics.",
            cls._strict_diagnostics_env,
            raw_value,
        )
        return False

    @staticmethod
    def _make_runtime_config(config: GenesisConfig) -> _GenesisRuntimeConfig:
        """Build the process-global Genesis runtime settings for a config."""

        return _GenesisRuntimeConfig(
            backend_name="cuda" if config.use_gpu else "cpu",
            precision="32",
            seed=int(config.random_seed),
            logging_level="warning",
        )

    @staticmethod
    def _describe_runtime_config(runtime_config: _GenesisRuntimeConfig) -> str:
        """Format runtime settings for logs and exceptions."""

        return (
            f"backend={runtime_config.backend_name}, "
            f"precision={runtime_config.precision}, "
            f"seed={runtime_config.seed}, "
            f"logging_level={runtime_config.logging_level}"
        )

    @classmethod
    def _ensure_runtime_compatible(cls, requested: _GenesisRuntimeConfig) -> None:
        """Validate that a configure() request matches the live Genesis runtime."""

        current = cls._gs_runtime_config
        if current is None or current == requested:
            return

        mismatches: List[str] = []
        if current.backend_name != requested.backend_name:
            mismatches.append(
                f"backend {current.backend_name!r} vs {requested.backend_name!r}"
            )
        if current.precision != requested.precision:
            mismatches.append(
                f"precision {current.precision!r} vs {requested.precision!r}"
            )
        if current.seed != requested.seed:
            mismatches.append(f"seed {current.seed!r} vs {requested.seed!r}")
        if current.logging_level != requested.logging_level:
            mismatches.append(
                f"logging_level {current.logging_level!r} vs {requested.logging_level!r}"
            )

        raise RuntimeError(
            "Genesis runtime is already initialized with "
            f"{cls._describe_runtime_config(current)}; configure() requested "
            f"{cls._describe_runtime_config(requested)}. Genesis is process-global, "
            "so incompatible configure() calls are not allowed in the same process "
            f"({', '.join(mismatches)}). Reuse the existing backend/seed settings "
            "or start a fresh process after a supported runtime destroy."
        )

    @classmethod
    def _reset_runtime_tracking(cls) -> None:
        """Clear local bookkeeping for the process-global Genesis runtime."""

        cls._gs_initialized = False
        cls._gs_runtime_config = None
        cls._gs_runtime_owner_count = 0

    @classmethod
    def _diagnostic_message(
        cls,
        operation: str,
        *,
        entity_name: Optional[str] = None,
        detail: str,
        exc: Optional[BaseException] = None,
    ) -> str:
        """Build a structured diagnostics string."""

        subject = operation
        if entity_name is not None:
            subject = f"{operation} for entity '{entity_name}'"
        message = f"{subject}: {detail}"
        if exc is not None:
            message = f"{message} ({type(exc).__name__}: {exc})"
        return message

    @classmethod
    def _warn_or_raise_diagnostic(
        cls,
        operation: str,
        *,
        entity_name: Optional[str] = None,
        detail: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Warn by default, or raise under strict diagnostics mode."""

        message = cls._diagnostic_message(
            operation,
            entity_name=entity_name,
            detail=detail,
            exc=exc,
        )
        if cls._strict_diagnostics_enabled():
            raise RuntimeError(message) from exc
        logger.warning(message)

    @classmethod
    def _raise_diagnostic(
        cls,
        operation: str,
        *,
        entity_name: Optional[str] = None,
        detail: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Raise a structured runtime error."""

        message = cls._diagnostic_message(
            operation,
            entity_name=entity_name,
            detail=detail,
            exc=exc,
        )
        raise RuntimeError(message) from exc

    # ------------------------------------------------------------------
    # ABC: configure
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.CONSTRUCTION)
    def configure(
        self,
        config: GenesisConfig,
        show_viewer: bool = False,
        viewer_options: Optional[Any] = None,
    ) -> None:
        """Configure the physics engine and create the Genesis scene shell.

        Initialises the Genesis global singleton if not already done (thread-safe).
        Genesis runtime configuration is process-global, so repeated configure()
        calls must request the same backend/precision/seed/logging settings or
        this method raises with actionable diagnostics instead of silently
        reusing an incompatible runtime. Creates a gs.Scene with SimOptions,
        RigidOptions, and MPMOptions derived from the provided GenesisConfig.
        Entities are added after this call and before build_scene().

        Parameters:
            config: GenesisConfig (use GenesisConfig.from_yaml() for YAML loading).
            show_viewer: Open the interactive 3-D viewer window. Default False.
                         When True, call step(render=False) for all sim-only steps
                         and step(render=True) once per desired render frame to
                         avoid the viewer's per-step rate-limiter blocking the sim.
            viewer_options: gs.options.ViewerOptions instance. When None a sensible
                            default is used (60 Hz cap, run_in_thread=True).

        Raises:
            RuntimeError: If called outside CONSTRUCTION phase or if the process
                already hosts an incompatible Genesis runtime configuration.
        """
        runtime_config = self._make_runtime_config(config)

        # --- Genesis global singleton init ---
        with GenesisPhysicsEngine._gs_init_lock:
            if not GenesisPhysicsEngine._gs_initialized:
                backend = gs.cuda if config.use_gpu else gs.cpu
                gs.init(
                    backend=backend,
                    precision=runtime_config.precision,
                    seed=runtime_config.seed,
                    logging_level=runtime_config.logging_level,
                )
                GenesisPhysicsEngine._gs_initialized = True
                GenesisPhysicsEngine._gs_runtime_config = runtime_config
                self._owns_gs_runtime_init = True
            else:
                GenesisPhysicsEngine._ensure_runtime_compatible(runtime_config)

        self._config = config
        self._dt = config.timestep
        self._show_viewer = show_viewer

        # --- Solver backends cache ---
        _default_backends: Dict[str, str] = {
            "rigid_body": "rigid_body_solver",
            "mpm":        "material_point_method_solver",
            "sph":        "sph_solver",
            "fem":        "finite_element_solver",
            "pbd":        "position_based_dynamics_solver",
        }
        self._solver_backends_cache = {**_default_backends, **config.solver_map}

        # --- Viewer options ---
        if viewer_options is None and show_viewer:
            viewer_options = gs.options.ViewerOptions(
                max_FPS=60,
                run_in_thread=True,   # async render — does NOT block scene.step()
                camera_pos=(3.5, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            )

        # --- Build Genesis scene ---
        gravity = config.gravity_vector
        scene_kwargs: Dict[str, Any] = dict(
            sim_options=gs.options.SimOptions(
                gravity=gravity,
                dt=config.timestep,
                substeps=config.substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=config.timestep,
                gravity=gravity,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False,
                iterations=config.contact_iterations,
                tolerance=1e-6,
            ),
            mpm_options=gs.options.MPMOptions(
                dt=config.timestep,
                gravity=gravity,
                grid_density=64,
            ),
            vis_options=gs.options.VisOptions(),
            show_viewer=show_viewer,
        )
        if viewer_options is not None:
            scene_kwargs["viewer_options"] = viewer_options
        self._scene = gs.Scene(**scene_kwargs)
        if not self._claims_gs_runtime:
            with GenesisPhysicsEngine._gs_init_lock:
                GenesisPhysicsEngine._gs_runtime_owner_count += 1
                self._claims_gs_runtime = True

    # ------------------------------------------------------------------
    # Entity registration (CONSTRUCTION only)
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_morph_and_material(
        morph: Any,
        material: Any,
        entity_type: str,
    ) -> tuple:
        """Coerce morph/material to Genesis-native objects.

        Sub-composers pass URDF XML strings and ``None`` materials for
        portability. This method converts them to proper Genesis objects so
        callers do not need to import genesis directly.

        String morphs that look like XML (``<?xml`` or ``<robot``) are written
        to a temporary file and loaded via ``gs.morphs.URDF()``. A temporary
        directory is used; the file is NOT automatically deleted so Genesis can
        finish parsing it before GC collects any reference — it will be cleaned
        up when the OS reclaims the temp dir.

        Material defaults by entity_type:
            - "fixed"      → gs.materials.Rigid(); morph.fixed = True
            - "kinematic"  → gs.materials.Rigid(); morph.fixed = True
            - "articulated"→ gs.materials.Rigid()
            - "rigid"      → gs.materials.Rigid()
            - "mpm"        → gs.materials.MPM.Sand()

        In Genesis 0.4.4 the ``fixed`` flag lives on the morph (e.g.
        ``gs.morphs.URDF(fixed=True)``), not on the material.
        """
        # ── morph coercion ────────────────────────────────────────────
        is_fixed = entity_type in ("fixed", "kinematic")
        if isinstance(morph, str):
            stripped = morph.lstrip()
            if stripped.startswith("<"):
                # Inline URDF XML — write to temp file
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".urdf", delete=False, mode="w", encoding="utf-8"
                )
                tmp.write(morph)
                tmp.flush()
                tmp.close()
                morph = gs.morphs.URDF(file=tmp.name, fixed=is_fixed)
            else:
                # Treat as a file path
                morph = gs.morphs.URDF(file=morph, fixed=is_fixed)
        elif is_fixed and hasattr(morph, "fixed"):
            # Already a genesis morph; set fixed flag if not already set
            morph = morph.model_copy(update={"fixed": True})

        # ── material coercion ─────────────────────────────────────────
        if material is None:
            if entity_type == "mpm":
                material = gs.materials.MPM.Sand()
            else:
                material = gs.materials.Rigid()

        return morph, material

    @_require_phase(ScenePhase.CONSTRUCTION)
    def add_entity(
        self,
        name: str,
        morph: Any,
        material: Any,
        entity_type: str = "rigid",
        **kwargs: Any,
    ) -> Any:
        """Register a Genesis entity with the engine during scene construction.

        Parameters:
            name: Unique name for this entity (used for all subsequent queries).
            morph: Genesis morph descriptor (gs.morphs.Box, gs.morphs.URDF, etc.)
                   OR a URDF XML string / file path (auto-coerced to gs.morphs.URDF).
            material: Genesis material (gs.materials.Rigid, gs.materials.MPM.Sand, etc.)
                      or None (auto-selected by entity_type).
            entity_type: One of "rigid", "articulated", "fixed", "kinematic", "mpm",
                         "terrain". Default "rigid".
            **kwargs: Additional kwargs forwarded to scene.add_entity().

        Returns:
            The Genesis entity object (gs.RigidEntity, etc.).

        Raises:
            RuntimeError: If called outside CONSTRUCTION phase.
            ValueError: If name is already registered.
        """
        morph, material = self._coerce_morph_and_material(morph, material, entity_type)

        with self._entity_lock:
            if name in self._entities:
                raise ValueError(
                    f"Entity '{name}' is already registered. "
                    "Each entity must have a unique name."
                )
            genesis_entity = self._scene.add_entity(
                morph=morph,
                material=material,
                **kwargs,
            )
            record = _EntityRecord(
                name=name,
                genesis_entity=genesis_entity,
                entity_type=entity_type,
                prev_lin_vel=np.zeros(3, dtype=np.float32),
                prev_ang_vel=np.zeros(3, dtype=np.float32),
                accel_tracking_enabled=False,
                accel_prev_step=-1,
            )
            self._entities[name] = record
        return genesis_entity

    @_require_phase(ScenePhase.CONSTRUCTION)
    def add_terrain_entity(
        self,
        name: str,
        height_field: NDArray,
        size: List[float],
    ) -> Any:
        """Register a terrain heightfield entity.

        Precomputes and caches surface normals for fast get_terrain_normal() queries.
        Also stores the raw heightfield for get_terrain_height() bilinear interpolation.

        Parameters:
            name: Unique entity name.
            height_field: (H, W) float32 numpy array of terrain heights in metres.
            size: [size_x, size_y] world dimensions in metres.

        Returns:
            The Genesis terrain entity object.

        Raises:
            RuntimeError: If called outside CONSTRUCTION phase.
            ValueError: If name already registered.
        """
        height_field = np.asarray(height_field, dtype=np.float32)
        res_y, res_x = height_field.shape
        size_x, size_y = float(size[0]), float(size[1])
        cell_size_x = size_x / max(res_x - 1, 1)
        cell_size_y = size_y / max(res_y - 1, 1)

        # Genesis 0.4.4 gs.morphs.Terrain uses horizontal_scale / vertical_scale,
        # n_subterrains and subterrain_size rather than the older hfield/size kwargs.
        horizontal_scale = cell_size_x           # metres per cell (X)
        vertical_scale   = 1.0                   # height values are already in metres
        morph = gs.morphs.Terrain(
            height_field=height_field,
            horizontal_scale=horizontal_scale,
            vertical_scale=vertical_scale,
            n_subterrains=(1, 1),
            subterrain_size=(float(res_x), float(res_y)),
        )
        material = gs.materials.Rigid(
            rho=self._terrain_density_kg_m3,
            friction=self._terrain_contact_friction,
            coup_restitution=self._terrain_contact_restitution,
        )
        genesis_entity = self.add_entity(name, morph, material, entity_type="terrain")

        # Convert height-per-cell gradients into world-space slope components.
        dz_dx = np.gradient(height_field, axis=1) / max(cell_size_x, 1e-8)
        dz_dy = np.gradient(height_field, axis=0) / max(cell_size_y, 1e-8)
        normals = np.stack([-dz_dx, -dz_dy, np.ones_like(dz_dx)], axis=-1)
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normal_map = normals / np.maximum(norms, 1e-8)
        normal_map = normal_map.astype(np.float32)

        self._terrain = _TerrainRecord(
            genesis_entity=genesis_entity,
            height_field=height_field,
            normal_map=normal_map,
            size_x=size_x,
            size_y=size_y,
            resolution_x=res_x,
            resolution_y=res_y,
        )
        return genesis_entity

    @_require_phase(ScenePhase.CONSTRUCTION)
    def register_raycaster(
        self,
        name: str,
        link_entity: str,
        link_idx: int,
        pattern_config: Dict[str, Any],
        max_range: float,
    ) -> None:
        """Register a Genesis raycaster sensor (used by LiDARScanner).

        Must be called during CONSTRUCTION, before build_scene(). The sensor is
        attached to a specific link of a named entity.

        Parameters:
            name: Unique raycaster name.
            link_entity: Name of the entity the sensor is mounted on.
            link_idx: Index of the entity link to attach the sensor to.
            pattern_config: Dict with keys:
                - num_channels (int): Number of vertical beams.
                - elevation_range_deg (tuple[float,float]): (min_deg, max_deg).
                - h_resolution_deg (float): Horizontal angular step in degrees.
            max_range: Maximum ray range in metres.

        Raises:
            RuntimeError: If called outside CONSTRUCTION phase.
            KeyError: If link_entity is not registered.
        """
        if name in self._raycasters:
            raise ValueError(f"Raycaster '{name}' is already registered")
        if link_idx < 0:
            raise IndexError(f"Raycaster link_idx must be non-negative, got {link_idx}")

        with self._entity_lock:
            if link_entity not in self._entities:
                raise KeyError(
                    f"Cannot register raycaster on unknown entity '{link_entity}'"
                )
            entity_record = self._entities[link_entity]

        try:
            entity_record.genesis_entity.links[link_idx]
        except IndexError as exc:
            raise IndexError(
                f"Entity '{link_entity}' has no link at index {link_idx}"
            ) from exc

        entity_idx = self._safe_int(getattr(entity_record.genesis_entity, "idx", None))
        if entity_idx is None:
            entity_idx = -1

        num_channels = int(pattern_config["num_channels"])
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        try:
            elev_min, elev_max = pattern_config["elevation_range_deg"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "elevation_range_deg must be a (min_deg, max_deg) pair"
            ) from exc
        elev_range = (float(elev_min), float(elev_max))
        if elev_range[1] <= elev_range[0]:
            raise ValueError(
                "elevation_range_deg max must be greater than min, "
                f"got {elev_range}"
            )
        if elev_range[1] - elev_range[0] > 360.0:
            raise ValueError(f"elevation range cannot exceed 360 degrees: {elev_range}")
        h_res_deg = float(pattern_config["h_resolution_deg"])
        if h_res_deg <= 0.0:
            raise ValueError(f"h_resolution_deg must be positive, got {h_res_deg}")
        h_count = max(1, int(round(360.0 / h_res_deg)))

        try:
            pattern_cls = gs.sensors.SphericalPattern
        except AttributeError:
            pattern_cls = gs.options.sensors.SphericalPattern
        pattern = pattern_cls(
            fov=(360.0, tuple(elev_range)),
            n_points=(h_count, num_channels),
        )

        try:
            raycaster_cls = gs.sensors.Raycaster
        except AttributeError:
            raycaster_cls = gs.options.sensors.Raycaster
        sensor_options = raycaster_cls(
            entity_idx=entity_idx,
            link_idx_local=int(link_idx),
            pattern=pattern,
            max_range=float(max_range),
            no_hit_value=float(max_range),
            return_world_frame=True,
        )

        genesis_sensor = self._scene.add_sensor(sensor_options)

        self._raycasters[name] = _RaycasterRecord(
            name=name,
            genesis_sensor=genesis_sensor,
            link_entity_name=link_entity,
            link_idx=link_idx,
        )

    # ------------------------------------------------------------------
    # ABC: build_scene
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.CONSTRUCTION)
    def build_scene(self, n_envs: int = 1) -> None:
        """Finalise scene construction and transition to SIMULATION phase.

        Calls scene.build() and transitions the adapter into SIMULATION phase.
        Acceleration telemetry is tracked lazily on demand so we avoid per-step
        velocity reads for bodies that do not need acceleration queries.

        Parameters:
            n_envs: Number of parallel environments. Only n_envs=1 is currently
                    supported by this adapter. Batched environments must use a
                    dedicated implementation instead of silently reusing env 0.

        Raises:
            RuntimeError: If called outside CONSTRUCTION phase or no entities registered.
            NotImplementedError: If n_envs != 1.
        """
        if n_envs != 1:
            raise NotImplementedError(
                "GenesisPhysicsEngine currently supports only n_envs=1. "
                "Multi-environment batching is not implemented in this adapter."
            )
        with self._entity_lock:
            if not self._entities:
                raise RuntimeError(
                    "build_scene() called with no entities registered. "
                    "Add at least one entity before building."
                )

        self._scene.build(n_envs=n_envs)
        self._n_envs = n_envs

        with self._phase_lock:
            self._phase = ScenePhase.SIMULATION

    # ------------------------------------------------------------------
    # ABC: step
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.SIMULATION)
    def step(self, dt: float, render: bool = True) -> None:
        """Advance the simulation by one fixed timestep.

        Genesis uses the dt fixed at scene construction — the dt parameter here
        is validated for consistency but does not change the step size.

        Parameters:
            dt: Expected timestep in seconds. Must match config.timestep.
            render: Whether to update the viewer this step. Set False for RL
                    training or intermediate substeps to skip the viewer's
                    blocking rate-limiter (max_FPS sleep). Only the last step
                    of a render batch should pass render=True.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            ValueError: If dt does not match the configured timestep.
        """
        if abs(dt - self._dt) > 1e-8:
            raise ValueError(
                f"step() called with dt={dt:.8f} but engine configured with "
                f"dt={self._dt:.8f}. Genesis uses a fixed timestep — pass the "
                "same value as GenesisConfig.timestep."
            )

        # Cache velocities only for bodies that opted into acceleration telemetry.
        with self._entity_lock:
            for record in self._entities.values():
                if record.entity_type not in ("rigid", "kinematic"):
                    continue
                if not record.accel_tracking_enabled:
                    continue
                try:
                    vel = record.genesis_entity.get_vel()
                    ang = record.genesis_entity.get_ang()
                    record.prev_lin_vel = _to_numpy(vel).flatten()[:3]
                    record.prev_ang_vel = _to_numpy(ang).flatten()[:3]
                    record.accel_prev_step = self._step_count + 1
                except Exception as exc:
                    record.accel_prev_step = -1
                    self._warn_or_raise_diagnostic(
                        "step() acceleration cache update",
                        entity_name=record.name,
                        detail=(
                            "failed to read pre-step velocities; acceleration telemetry "
                            "will be re-primed on the next query"
                        ),
                        exc=exc,
                    )

        try:
            self._scene.step(update_visualizer=render, refresh_visualizer=render)
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
            self._scene.step()
        self._sim_time += self._dt
        self._step_count += 1

    # ------------------------------------------------------------------
    # ABC: teardown
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.CONSTRUCTION, ScenePhase.SIMULATION)
    def teardown(self) -> None:
        """Destroy the scene and flush all VRAM.

        Transitions to TEARDOWN phase (terminal — no recovery). Clears all internal
        registries. Genesis runtime ownership is reference-counted across
        configured engine instances; only the last owner may attempt process-
        global teardown. Global Genesis runtime destroy then follows
        MOON_ROVER_GENESIS_DESTROY_POLICY:
          - safe (default): skip gs.destroy() on Windows to avoid hanging
            multi-entity scenes, but still destroy on other platforms.
          - always: always attempt gs.destroy().
          - never: never attempt gs.destroy().

        Raises:
            RuntimeError: If already in TEARDOWN phase.
        """
        with self._phase_lock:
            self._phase = ScenePhase.TEARDOWN

        # Release Python references
        with self._entity_lock:
            self._entities.clear()
        self._raycasters.clear()
        self._terrain = None
        self._scene = None

        # Destroy Genesis global state
        with GenesisPhysicsEngine._gs_init_lock:
            if self._claims_gs_runtime and GenesisPhysicsEngine._gs_runtime_owner_count > 0:
                GenesisPhysicsEngine._gs_runtime_owner_count -= 1
            self._claims_gs_runtime = False
            self._owns_gs_runtime_init = False

            if not GenesisPhysicsEngine._gs_initialized:
                return

            if GenesisPhysicsEngine._gs_runtime_owner_count > 0:
                logger.info(
                    "Skipping gs.destroy() because %d Genesis engine instance(s) still "
                    "own the shared runtime.",
                    GenesisPhysicsEngine._gs_runtime_owner_count,
                )
                return

            destroy_policy = self._resolve_destroy_policy()
            should_destroy = (
                destroy_policy == "always"
                or (destroy_policy == "safe" and os.name != "nt")
            )
            if not should_destroy:
                logger.info(
                    "Skipping gs.destroy() under %s=%s on platform %s; "
                    "Genesis runtime stays initialized until process exit.",
                    GenesisPhysicsEngine._destroy_policy_env,
                    destroy_policy,
                    os.name,
                )
                return

            try:
                gs.destroy()
            except Exception as exc:
                logger.warning("gs.destroy() raised an exception: %s", exc)
            finally:
                GenesisPhysicsEngine._reset_runtime_tracking()

    # ------------------------------------------------------------------
    # ABC: save_snapshot
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.SIMULATION)
    def save_snapshot(self) -> bytes:
        """Serialise full rigid-body scene state to bytes.

        Captures positions, quaternions, velocities, angular velocities, and DOF
        states for all rigid entities. MPM particle state is NOT included
        (can_snapshot_mpm = False).

        Returns:
            Pickle bytes (protocol 5) encoding the complete scene state.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
        """
        entity_states: Dict[str, Any] = {}
        with self._entity_lock:
            for name, rec in self._entities.items():
                if rec.entity_type not in ("rigid", "kinematic"):
                    continue
                try:
                    dofs_pos = _to_numpy(
                        _call_first(rec.genesis_entity, ("get_dofs_pos", "get_dofs_position"))
                    )
                    dofs_vel = _to_numpy(
                        _call_first(rec.genesis_entity, ("get_dofs_vel", "get_dofs_velocity"))
                    )
                except AttributeError:
                    dofs_pos = np.zeros(0, dtype=np.float32)
                    dofs_vel = np.zeros(0, dtype=np.float32)
                try:
                    pos = _to_numpy(rec.genesis_entity.get_pos()).flatten()[:3]
                    quat = _to_numpy(rec.genesis_entity.get_quat()).flatten()[:4]
                    vel = _to_numpy(rec.genesis_entity.get_vel()).flatten()[:3]
                    ang = _to_numpy(rec.genesis_entity.get_ang()).flatten()[:3]
                except Exception as exc:
                    self._raise_diagnostic(
                        "save_snapshot() state capture",
                        entity_name=name,
                        detail=(
                            "failed to read pose or velocity state; snapshot capture "
                            "cannot safely continue"
                        ),
                        exc=exc,
                    )
                entity_states[name] = {
                    "pos":          pos,
                    "quat":         quat,
                    "vel":          vel,
                    "ang":          ang,
                    "dofs_pos":     dofs_pos,
                    "dofs_vel":     dofs_vel,
                    "prev_lin_vel": rec.prev_lin_vel.copy(),
                    "prev_ang_vel": rec.prev_ang_vel.copy(),
                    "accel_tracking_enabled": bool(rec.accel_tracking_enabled),
                    "accel_prev_step": int(rec.accel_prev_step),
                }

        snapshot = {
            "version":    1,
            "sim_time":   self._sim_time,
            "step_count": self._step_count,
            "dt":         self._dt,
            "n_envs":     self._n_envs,
            "entities":   entity_states,
        }
        return pickle.dumps(snapshot, protocol=5)

    # ------------------------------------------------------------------
    # ABC: restore_snapshot
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.SIMULATION)
    def restore_snapshot(self, data: bytes) -> None:
        """Restore scene state from bytes produced by save_snapshot().

        Restoration is an exact-time operation: after this call, physical state,
        sim_time, and step_count match the saved snapshot. The method must not
        advance the Genesis scene internally because snapshots are used for replay
        and Monte Carlo branching.

        Parameters:
            data: Bytes from a previous save_snapshot() call.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            ValueError: If data is corrupt, wrong version, or entity mismatch.
        """
        try:
            snapshot: Dict[str, Any] = pickle.loads(data)
        except (pickle.UnpicklingError, EOFError, ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"Snapshot data is corrupt or unreadable: {exc}") from exc

        if snapshot.get("version") != 1:
            raise ValueError(
                f"Unsupported snapshot version {snapshot.get('version')!r}. "
                "Expected version 1."
            )
        if abs(snapshot.get("dt", 0.0) - self._dt) > 1e-8:
            raise ValueError(
                f"Snapshot dt={snapshot['dt']} does not match engine dt={self._dt}. "
                "Cannot restore across different timestep configurations."
            )

        snap_names = set(snapshot.get("entities", {}).keys())
        with self._entity_lock:
            engine_names = {
                n for n, r in self._entities.items()
                if r.entity_type in ("rigid", "kinematic")
            }
        missing = snap_names - engine_names
        if missing:
            raise ValueError(
                f"Snapshot references entities not present in this scene: {missing}"
            )

        # Restore entity states
        with self._entity_lock:
            for name, state in snapshot["entities"].items():
                if name not in self._entities:
                    continue
                rec = self._entities[name]
                pos = np.asarray(state["pos"], dtype=np.float32)
                quat = np.asarray(state["quat"], dtype=np.float32)
                vel = np.asarray(state["vel"], dtype=np.float32)
                ang = np.asarray(state["ang"], dtype=np.float32)
                try:
                    rec.genesis_entity.set_pos(pos)
                except Exception as exc:
                    self._raise_diagnostic(
                        "restore_snapshot() position restore",
                        entity_name=name,
                        detail="failed to restore body position",
                        exc=exc,
                    )
                quat_restored = False
                try:
                    _call_first(rec.genesis_entity, ("set_quat",), quat)
                    quat_restored = True
                except AttributeError:
                    quat_restored = False
                except Exception as exc:
                    self._raise_diagnostic(
                        "restore_snapshot() orientation restore",
                        entity_name=name,
                        detail="set_quat() raised unexpectedly",
                        exc=exc,
                    )
                if not quat_restored:
                    try:
                        n_dofs = int(getattr(rec.genesis_entity, "n_dofs"))
                    except (AttributeError, TypeError, ValueError):
                        n_dofs = 0
                    if n_dofs >= 7:
                        try:
                            dofs_pos = _call_first(
                                rec.genesis_entity,
                                ("get_dofs_pos", "get_dofs_position"),
                            )
                            dofs_arr = _to_numpy(dofs_pos).flatten()
                            dofs_arr[0:3] = pos
                            dofs_arr[3:7] = quat
                            _call_first(
                                rec.genesis_entity,
                                ("set_dofs_pos", "set_dofs_position"),
                                dofs_arr,
                            )
                            quat_restored = True
                        except Exception as exc:
                            self._raise_diagnostic(
                                "restore_snapshot() orientation restore",
                                entity_name=name,
                                detail="floating-base DOF fallback failed",
                                exc=exc,
                            )
                    if not quat_restored:
                        self._warn_or_raise_diagnostic(
                            "restore_snapshot() orientation restore",
                            entity_name=name,
                            detail=(
                                "entity exposes neither set_quat() nor a usable "
                                "floating-base DOF pose fallback; orientation restore "
                                "was skipped"
                            ),
                        )
                try:
                    _call_first(rec.genesis_entity, ("set_vel",), vel)
                    _call_first(rec.genesis_entity, ("set_ang", "set_ang_vel"), ang)
                except AttributeError:
                    try:
                        _call_first(
                            rec.genesis_entity,
                            ("set_dofs_velocity", "set_dofs_vel"),
                            np.concatenate([vel, ang]).astype(np.float32),
                        )
                    except Exception as exc:
                        self._raise_diagnostic(
                            "restore_snapshot() velocity restore",
                            entity_name=name,
                            detail=(
                                "entity exposes neither direct velocity setters nor "
                                "a usable DOF velocity fallback"
                            ),
                            exc=exc,
                        )
                except Exception as exc:
                    self._raise_diagnostic(
                        "restore_snapshot() velocity restore",
                        entity_name=name,
                        detail="direct velocity setter raised unexpectedly",
                        exc=exc,
                    )
                dofs_pos = state.get("dofs_pos", np.zeros(0))
                dofs_vel = state.get("dofs_vel", np.zeros(0))
                if dofs_pos.size > 0:
                    try:
                        _call_first(
                            rec.genesis_entity,
                            ("set_dofs_pos", "set_dofs_position"),
                            dofs_pos,
                        )
                    except Exception as exc:
                        self._raise_diagnostic(
                            "restore_snapshot() DOF position restore",
                            entity_name=name,
                            detail="failed to restore saved DOF positions",
                            exc=exc,
                        )
                if dofs_vel.size > 0:
                    try:
                        _call_first(
                            rec.genesis_entity,
                            ("set_dofs_velocity", "set_dofs_vel"),
                            dofs_vel,
                        )
                    except Exception as exc:
                        self._raise_diagnostic(
                            "restore_snapshot() DOF velocity restore",
                            entity_name=name,
                            detail="failed to restore saved DOF velocities",
                            exc=exc,
                        )
                rec.prev_lin_vel = state["prev_lin_vel"].copy()
                rec.prev_ang_vel = state["prev_ang_vel"].copy()
                rec.accel_tracking_enabled = bool(state.get("accel_tracking_enabled", False))
                rec.accel_prev_step = int(state.get("accel_prev_step", -1))

        self._sim_time = float(snapshot["sim_time"])
        self._step_count = int(snapshot["step_count"])

    # ------------------------------------------------------------------
    # ABC: get_phase
    # ------------------------------------------------------------------

    def get_phase(self) -> ScenePhase:
        """Return the current scene lifecycle phase (thread-safe)."""
        with self._phase_lock:
            return self._phase

    # ------------------------------------------------------------------
    # ABC: solver_backends property
    # ------------------------------------------------------------------

    @property
    def solver_backends(self) -> Dict[str, str]:
        """Return a copy of the solver backend mapping."""
        return dict(self._solver_backends_cache)

    # ------------------------------------------------------------------
    # Extended API: entity access
    # ------------------------------------------------------------------

    def get_entity(self, name: str) -> Any:
        """Return the raw Genesis entity object for a registered entity.

        Parameters:
            name: Registered entity name.

        Returns:
            Genesis entity object (gs.RigidEntity, gs.MPMEntity, etc.).

        Raises:
            KeyError: If name is not registered.
        """
        with self._entity_lock:
            if name not in self._entities:
                raise KeyError(
                    f"Entity '{name}' is not registered. "
                    f"Registered entities: {list(self._entities.keys())}"
                )
            return self._entities[name].genesis_entity

    def list_entities(self) -> List[str]:
        """Return names of all registered entities."""
        with self._entity_lock:
            return list(self._entities.keys())

    # ------------------------------------------------------------------
    # Extended API: body state queries (SIMULATION only)
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.SIMULATION)
    def get_body_pose(
        self, entity_name: str, env_idx: int = 0
    ) -> Tuple[NDArray, NDArray]:
        """Get the world-frame pose of a rigid entity.

        Parameters:
            entity_name: Registered entity name.
            env_idx: Environment index. Only env_idx=0 is currently supported.

        Returns:
            (pos, quat): pos is (3,) float32 [x,y,z] metres;
                         quat is (4,) float32 [x,y,z,w].

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "get_body_pose()")
        rec = self._get_record(entity_name)
        pos = _to_numpy(rec.genesis_entity.get_pos()).flatten()[:3]
        quat = _to_numpy(rec.genesis_entity.get_quat()).flatten()[:4]
        return pos, quat

    @_require_phase(ScenePhase.SIMULATION)
    def get_body_velocity(
        self, entity_name: str, env_idx: int = 0
    ) -> Tuple[NDArray, NDArray]:
        """Get the world-frame velocity of a rigid entity.

        Parameters:
            entity_name: Registered entity name.
            env_idx: Environment index. Only env_idx=0 is currently supported.

        Returns:
            (lin_vel, ang_vel): both (3,) float32 in m/s and rad/s respectively.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "get_body_velocity()")
        rec = self._get_record(entity_name)
        lin = _to_numpy(rec.genesis_entity.get_vel()).flatten()[:3]
        ang = _to_numpy(rec.genesis_entity.get_ang()).flatten()[:3]
        return lin, ang

    @_require_phase(ScenePhase.SIMULATION)
    def get_body_acceleration(
        self, entity_name: str, env_idx: int = 0
    ) -> Tuple[NDArray, NDArray]:
        """Compute body acceleration from velocity delta across the last tracked step.

        Acceleration telemetry is tracked lazily per body. Calling this method
        subscribes the body for future per-step velocity snapshots, but the first
        call after construction or after an untracked run only primes telemetry and
        returns zeros. Once primed, calls after each step return
        ``(current_velocity - previous_velocity) / dt`` for the most recent step.

        Parameters:
            entity_name: Registered entity name.
            env_idx: Environment index. Only env_idx=0 is currently supported.

        Returns:
            (lin_accel, ang_accel): both (3,) float32 in m/s² and rad/s².

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "get_body_acceleration()")
        rec = self._get_record(entity_name)
        if not rec.accel_tracking_enabled or rec.accel_prev_step != self._step_count:
            try:
                rec.prev_lin_vel = _to_numpy(rec.genesis_entity.get_vel()).flatten()[:3]
                rec.prev_ang_vel = _to_numpy(rec.genesis_entity.get_ang()).flatten()[:3]
            except Exception as exc:
                rec.prev_lin_vel = np.zeros(3, dtype=np.float32)
                rec.prev_ang_vel = np.zeros(3, dtype=np.float32)
                self._warn_or_raise_diagnostic(
                    "get_body_acceleration() telemetry prime",
                    entity_name=entity_name,
                    detail=(
                        "failed to read current velocity state; returning zero "
                        "acceleration until telemetry can be primed"
                    ),
                    exc=exc,
                )
            rec.accel_tracking_enabled = True
            rec.accel_prev_step = self._step_count
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        curr_lin = _to_numpy(rec.genesis_entity.get_vel()).flatten()[:3]
        curr_ang = _to_numpy(rec.genesis_entity.get_ang()).flatten()[:3]
        lin_accel = (curr_lin - rec.prev_lin_vel) / self._dt
        ang_accel = (curr_ang - rec.prev_ang_vel) / self._dt
        return lin_accel, ang_accel

    @_require_phase(ScenePhase.SIMULATION)
    def get_link_poses(
        self, entity_name: str, env_idx: int = 0
    ) -> List[Tuple[NDArray, NDArray]]:
        """Get world-frame pose of every link in an articulated entity.

        Parameters:
            entity_name: Registered entity name.
            env_idx: Environment index. Only env_idx=0 is currently supported.

        Returns:
            List of (pos, quat) tuples, one per link.
            pos is (3,) float32; quat is (4,) float32 [x,y,z,w].

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "get_link_poses()")
        rec = self._get_record(entity_name)
        positions = _to_numpy(rec.genesis_entity.get_links_pos())
        quats = _to_numpy(rec.genesis_entity.get_links_quat())
        # Shape: (n_links, 3) and (n_links, 4)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if quats.ndim == 1:
            quats = quats.reshape(1, -1)
        return [(positions[i], quats[i]) for i in range(positions.shape[0])]

    @_require_phase(ScenePhase.SIMULATION)
    def get_link_velocities(
        self, entity_name: str, env_idx: int = 0
    ) -> List[Tuple[NDArray, NDArray]]:
        """Get world-frame velocity of every link in an articulated entity.

        Returns:
            List of (lin_vel, ang_vel) tuples, one per link. Both (3,) float32.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "get_link_velocities()")
        rec = self._get_record(entity_name)
        vels = _to_numpy(rec.genesis_entity.get_links_vel())
        angs = _to_numpy(rec.genesis_entity.get_links_ang())
        if vels.ndim == 1:
            vels = vels.reshape(1, -1)
        if angs.ndim == 1:
            angs = angs.reshape(1, -1)
        return [(vels[i], angs[i]) for i in range(vels.shape[0])]

    @_require_phase(ScenePhase.SIMULATION)
    def get_dof_positions(self, entity_name: str, env_idx: int = 0) -> NDArray:
        """Get joint DOF positions for an articulated entity.

        Returns:
            (n_dofs,) float32 array of joint angles / positions.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "get_dof_positions()")
        rec = self._get_record(entity_name)
        return _to_numpy(
            _call_first(rec.genesis_entity, ("get_dofs_pos", "get_dofs_position"))
        ).flatten()

    @_require_phase(ScenePhase.SIMULATION)
    def get_dof_velocities(self, entity_name: str, env_idx: int = 0) -> NDArray:
        """Get joint DOF velocities for an articulated entity.

        Returns:
            (n_dofs,) float32 array of joint velocities.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "get_dof_velocities()")
        rec = self._get_record(entity_name)
        return _to_numpy(
            _call_first(rec.genesis_entity, ("get_dofs_vel", "get_dofs_velocity"))
        ).flatten()

    # ------------------------------------------------------------------
    # Extended API: body state setters (SIMULATION only)
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.SIMULATION)
    def set_body_pose(
        self,
        entity_name: str,
        pos: NDArray,
        quat: NDArray,
        env_idx: int = 0,
    ) -> None:
        """Set the world-frame pose of a rigid entity.

        Uses Genesis' direct base-link setters when available. For older Genesis
        builds without set_quat(), articulated floating-base bodies fall back to
        DOF positions.

        Parameters:
            entity_name: Registered entity name.
            pos: (3,) float32 world position [x, y, z] in metres.
            quat: (4,) float32 quaternion [x, y, z, w].
            env_idx: Environment index. Only env_idx=0 is currently supported.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "set_body_pose()")
        rec = self._get_record(entity_name)
        pos_arr = np.asarray(pos, dtype=np.float32)
        quat_arr = np.asarray(quat, dtype=np.float32)
        rec.genesis_entity.set_pos(pos_arr)

        quat_method_available = True
        try:
            _call_first(rec.genesis_entity, ("set_quat",), quat_arr)
        except AttributeError:
            quat_method_available = False

        if quat_method_available:
            return

        # Attempt quaternion via DOF positions (articulated bodies with floating base)
        try:
            n_dofs = rec.genesis_entity.n_dofs
            if n_dofs >= 7:
                dofs_pos = _call_first(
                    rec.genesis_entity,
                    ("get_dofs_pos", "get_dofs_position"),
                )
                dofs_arr = _to_numpy(dofs_pos).flatten()
                # Floating base: first 7 DOFs are [x, y, z, qx, qy, qz, qw]
                dofs_arr[0:3] = pos_arr
                dofs_arr[3:7] = quat_arr
                _call_first(
                    rec.genesis_entity,
                    ("set_dofs_pos", "set_dofs_position"),
                    dofs_arr,
                )
                return
        except Exception as exc:
            self._raise_diagnostic(
                "set_body_pose() orientation restore",
                entity_name=entity_name,
                detail="floating-base DOF fallback failed",
                exc=exc,
            )

        self._warn_or_raise_diagnostic(
            "set_body_pose() orientation restore",
            entity_name=entity_name,
            detail=(
                "entity exposes neither set_quat() nor a usable floating-base DOF "
                "pose fallback; quaternion update was skipped"
            ),
        )

    @_require_phase(ScenePhase.SIMULATION)
    def set_body_velocity(
        self,
        entity_name: str,
        lin_vel: NDArray,
        ang_vel: NDArray,
        env_idx: int = 0,
    ) -> None:
        """Set the world-frame velocity of a rigid entity.

        Parameters:
            entity_name: Registered entity name.
            lin_vel: (3,) float32 linear velocity [vx, vy, vz] in m/s.
            ang_vel: (3,) float32 angular velocity [wx, wy, wz] in rad/s.
            env_idx: Environment index. Only env_idx=0 is currently supported.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "set_body_velocity()")
        rec = self._get_record(entity_name)
        lin_arr = np.asarray(lin_vel, dtype=np.float32)
        ang_arr = np.asarray(ang_vel, dtype=np.float32)
        try:
            _call_first(rec.genesis_entity, ("set_vel",), lin_arr)
            _call_first(rec.genesis_entity, ("set_ang", "set_ang_vel"), ang_arr)
        except AttributeError:
            # Genesis 0.4.4 free rigid bodies expose a 6-DOF velocity setter.
            dof_velocity = np.concatenate([lin_arr, ang_arr]).astype(np.float32)
            _call_first(
                rec.genesis_entity,
                ("set_dofs_velocity", "set_dofs_vel"),
                dof_velocity,
            )

    @_require_phase(ScenePhase.SIMULATION)
    def set_dof_positions(
        self,
        entity_name: str,
        positions: NDArray,
        env_idx: int = 0,
    ) -> None:
        """Set joint DOF positions for an articulated entity.

        Parameters:
            entity_name: Registered entity name.
            positions: (n_dofs,) float32 array of joint target positions.
            env_idx: Environment index. Only env_idx=0 is currently supported.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "set_dof_positions()")
        rec = self._get_record(entity_name)
        _call_first(
            rec.genesis_entity,
            ("set_dofs_pos", "set_dofs_position"),
            np.asarray(positions, dtype=np.float32),
        )

    @_require_phase(ScenePhase.SIMULATION)
    def set_dof_velocities(
        self,
        entity_name: str,
        velocities: NDArray,
        env_idx: int = 0,
    ) -> None:
        """Set joint DOF velocities for an articulated entity.

        Parameters:
            entity_name: Registered entity name.
            velocities: (n_dofs,) float32 array of joint target velocities.
            env_idx: Environment index. Only env_idx=0 is currently supported.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "set_dof_velocities()")
        rec = self._get_record(entity_name)
        _call_first(
            rec.genesis_entity,
            ("set_dofs_velocity", "set_dofs_vel"),
            np.asarray(velocities, dtype=np.float32),
        )

    @_require_phase(ScenePhase.SIMULATION)
    def apply_dof_forces(
        self,
        entity_name: str,
        forces: NDArray,
        env_idx: int = 0,
    ) -> None:
        """Apply forces/torques to joint DOFs of an articulated entity.

        Parameters:
            entity_name: Registered entity name.
            forces: (n_dofs,) float32 array of forces/torques to apply.
            env_idx: Environment index (default 0).

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        self._require_env_idx_supported(env_idx, "apply_dof_forces()")
        rec = self._get_record(entity_name)
        rec.genesis_entity.set_dofs_force(np.asarray(forces, dtype=np.float32))

    # ------------------------------------------------------------------
    # Extended API: terrain queries (pure numpy, no Genesis call)
    # ------------------------------------------------------------------

    def get_terrain_height(self, x: float, y: float) -> float:
        """Get terrain surface height at world position (x, y) via bilinear interpolation.

        Parameters:
            x: World x-coordinate in metres. Clamped to terrain bounds.
            y: World y-coordinate in metres. Clamped to terrain bounds.

        Returns:
            Height in metres (float).

        Raises:
            RuntimeError: If no terrain entity has been registered.
        """
        if self._terrain is None:
            raise RuntimeError(
                "get_terrain_height() called but no terrain entity has been registered. "
                "Call add_terrain_entity() before build_scene()."
            )
        t = self._terrain
        x = float(np.clip(x, 0.0, t.size_x))
        y = float(np.clip(y, 0.0, t.size_y))
        px = (x / t.size_x) * (t.resolution_x - 1)
        py = (y / t.size_y) * (t.resolution_y - 1)
        x0, y0 = int(px), int(py)
        x1 = min(x0 + 1, t.resolution_x - 1)
        y1 = min(y0 + 1, t.resolution_y - 1)
        fx, fy = px - x0, py - y0
        h = (t.height_field[y0, x0] * (1 - fx) * (1 - fy)
             + t.height_field[y0, x1] * fx * (1 - fy)
             + t.height_field[y1, x0] * (1 - fx) * fy
             + t.height_field[y1, x1] * fx * fy)
        return float(h)

    def get_terrain_normal(self, x: float, y: float) -> NDArray:
        """Get terrain surface normal at world position (x, y) via bilinear interpolation.

        Parameters:
            x: World x-coordinate in metres. Clamped to terrain bounds.
            y: World y-coordinate in metres. Clamped to terrain bounds.

        Returns:
            (3,) float32 unit normal vector.

        Raises:
            RuntimeError: If no terrain entity has been registered.
        """
        if self._terrain is None:
            raise RuntimeError(
                "get_terrain_normal() called but no terrain entity has been registered."
            )
        t = self._terrain
        x = float(np.clip(x, 0.0, t.size_x))
        y = float(np.clip(y, 0.0, t.size_y))
        px = (x / t.size_x) * (t.resolution_x - 1)
        py = (y / t.size_y) * (t.resolution_y - 1)
        x0, y0 = int(px), int(py)
        x1 = min(x0 + 1, t.resolution_x - 1)
        y1 = min(y0 + 1, t.resolution_y - 1)
        fx, fy = px - x0, py - y0
        n = (t.normal_map[y0, x0] * (1 - fx) * (1 - fy)
             + t.normal_map[y0, x1] * fx * (1 - fy)
             + t.normal_map[y1, x0] * (1 - fx) * fy
             + t.normal_map[y1, x1] * fx * fy)
        norm = np.linalg.norm(n)
        return (n / norm).astype(np.float32) if norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # Extended API: raycaster queries (SIMULATION only)
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.SIMULATION)
    def query_raycaster(self, name: str) -> Dict[str, NDArray]:
        """Query the latest data from a registered raycaster sensor.

        Call after scene.step() to get the current frame's ray hits.

        Parameters:
            name: Raycaster name as passed to register_raycaster().

        Returns:
            Dict with keys:
            "distances": (N,) float32 — range per ray in metres.
            "positions": (N, 3) float32 — hit position in world frame.
            "normals":   (N, 3) float32 — surface normal at hit point.
                         Genesis 0.4.4 RaycasterSensor.read() does not expose
                         normals, so this field is zero-filled when unavailable.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If name is not a registered raycaster.
        """
        if name not in self._raycasters:
            raise KeyError(
                f"Raycaster '{name}' is not registered. "
                f"Registered raycasters: {list(self._raycasters.keys())}"
            )
        record = self._raycasters[name]
        sensor = record.genesis_sensor
        if hasattr(sensor, "get_data"):
            data = sensor.get_data()
        elif hasattr(sensor, "read"):
            data = sensor.read()
        else:
            raise RuntimeError(
                f"Raycaster '{name}' exposes neither get_data() nor read()"
            )

        if hasattr(data, "positions"):
            positions_raw = data.positions
        elif hasattr(data, "points"):
            positions_raw = data.points
        else:
            positions_raw = data[0]

        if hasattr(data, "distances"):
            distances_raw = data.distances
        else:
            distances_raw = data[1]

        positions = _to_numpy(positions_raw).reshape(-1, 3)
        distances = _to_numpy(distances_raw).flatten()
        if hasattr(data, "normals"):
            normals = _to_numpy(data.normals).reshape(-1, 3)
        else:
            normals = np.zeros_like(positions, dtype=np.float32)

        return {
            "distances": distances,
            "positions": positions,
            "normals":   normals,
        }

    # ------------------------------------------------------------------
    # Extended API: contact queries (SIMULATION only)
    # ------------------------------------------------------------------

    @_require_phase(ScenePhase.SIMULATION)
    def get_body_contacts(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get all active contacts involving a registered entity.

        Wraps the Genesis contact list through the private _get_raw_contacts()
        helper, which isolates the Genesis contact API surface for easy patching
        if the API shape changes between Genesis versions.

        Parameters:
            entity_name: Registered entity name.

        Returns:
            List of contact dicts, each with:
                "body_b":  str — name of the other entity (or "unknown").
                "pos":     (3,) float32 — contact point world position.
                "normal":  (3,) float32 — contact normal (pointing away from entity_name).
                "force_n": float — normal contact force magnitude in Newtons.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If entity_name not registered.
        """
        rec = self._get_record(entity_name)
        raw_contacts = self._get_raw_contacts(rec)
        if isinstance(raw_contacts, dict):
            return self._contacts_from_genesis_dict(rec, raw_contacts)

        result: List[Dict[str, Any]] = []
        for c in raw_contacts:
            entity_a = getattr(c, "entity_a", None)
            entity_b = getattr(c, "entity_b", None)
            if entity_a is rec.genesis_entity or entity_b is rec.genesis_entity:
                other = entity_b if entity_a is rec.genesis_entity else entity_a
                other_name = self._reverse_entity_lookup(other)
                try:
                    pos = np.array(c.position, dtype=np.float32).flatten()[:3]
                    if pos.size < 3:
                        raise ValueError("contact position has fewer than 3 values")
                except (AttributeError, TypeError, ValueError) as exc:
                    pos = np.zeros(3, dtype=np.float32)
                    self._warn_or_raise_diagnostic(
                        "get_body_contacts() position decode",
                        entity_name=rec.name,
                        detail=(
                            f"contact with '{other_name}' is missing a usable position; "
                            "returning zeros for that field"
                        ),
                        exc=exc,
                    )
                try:
                    normal = np.array(c.normal, dtype=np.float32).flatten()[:3]
                    if normal.size < 3:
                        raise ValueError("contact normal has fewer than 3 values")
                except (AttributeError, TypeError, ValueError) as exc:
                    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    self._warn_or_raise_diagnostic(
                        "get_body_contacts() normal decode",
                        entity_name=rec.name,
                        detail=(
                            f"contact with '{other_name}' is missing a usable normal; "
                            "returning +Z as a compatibility default"
                        ),
                        exc=exc,
                    )
                try:
                    force_n = float(c.force_normal)
                except (AttributeError, TypeError, ValueError) as exc:
                    force_n = 0.0
                    self._warn_or_raise_diagnostic(
                        "get_body_contacts() force decode",
                        entity_name=rec.name,
                        detail=(
                            f"contact with '{other_name}' is missing a usable normal force; "
                            "returning 0.0 for that field"
                        ),
                        exc=exc,
                    )
                result.append({
                    "body_b":  other_name,
                    "pos":     pos,
                    "normal":  normal,
                    "force_n": force_n,
                })
        return result

    @_require_phase(ScenePhase.SIMULATION)
    def is_in_contact(self, entity_a: str, entity_b: str) -> bool:
        """Return True if entity_a is currently in contact with entity_b.

        Parameters:
            entity_a: Name of the first entity.
            entity_b: Name of the second entity.

        Returns:
            True if at least one contact exists between the two entities.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            KeyError: If either entity is not registered.
        """
        contacts = self.get_body_contacts(entity_a)
        return any(c["body_b"] == entity_b for c in contacts)

    # ------------------------------------------------------------------
    # Extended API: simulation time
    # ------------------------------------------------------------------

    def get_sim_time(self) -> float:
        """Return total simulated time elapsed in seconds."""
        return self._sim_time

    def get_step_count(self) -> int:
        """Return total number of simulation steps completed."""
        return self._step_count

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_env_idx_supported(self, env_idx: int, operation: str) -> None:
        """Reject unsupported multi-environment access paths explicitly."""
        if self._n_envs != 1:
            raise NotImplementedError(
                f"{operation} does not support n_envs={self._n_envs}. "
                "GenesisPhysicsEngine currently supports only single-environment scenes."
            )
        if env_idx != 0:
            raise ValueError(
                f"{operation} received env_idx={env_idx}, but this adapter currently "
                "supports only env_idx=0 with n_envs=1."
            )

    def _get_record(self, name: str) -> _EntityRecord:
        """Thread-safe entity record lookup."""
        with self._entity_lock:
            if name not in self._entities:
                raise KeyError(
                    f"Entity '{name}' is not registered. "
                    f"Registered entities: {list(self._entities.keys())}"
                )
            return self._entities[name]

    def _get_raw_contacts(self, rec: _EntityRecord) -> Any:
        """Return raw Genesis contact data for a registered entity.

        Genesis 0.4.4 exposes contacts on RigidEntity.get_contacts() as a dict of
        batched tensors. The unit-test mocks still expose the older scene-level
        object list, so this method accepts both forms.
        """
        entity_get_contacts = getattr(rec.genesis_entity, "get_contacts", None)
        if callable(entity_get_contacts):
            try:
                entity_contacts = entity_get_contacts()
            except Exception as exc:
                self._warn_or_raise_diagnostic(
                    "get_body_contacts() entity contact query",
                    entity_name=rec.name,
                    detail=(
                        "entity-level get_contacts() raised while probing the Genesis "
                        "compatibility path; falling back to scene-level contacts"
                    ),
                    exc=exc,
                )
            else:
                normalized = self._normalize_raw_contacts(entity_contacts)
                if normalized is not None:
                    return normalized

        scene_get_contacts = getattr(self._scene, "get_contacts", None)
        if callable(scene_get_contacts):
            try:
                scene_contacts = scene_get_contacts()
            except Exception as exc:
                self._warn_or_raise_diagnostic(
                    "get_body_contacts() scene contact query",
                    entity_name=rec.name,
                    detail=(
                        "scene-level get_contacts() raised while probing the Genesis "
                        "compatibility path"
                    ),
                    exc=exc,
                )
            else:
                normalized = self._normalize_raw_contacts(scene_contacts)
                if normalized is not None:
                    return normalized

        logger.warning(
            "No usable Genesis contact API found for entity '%s'; returning no contacts",
            rec.name,
        )
        return []

    @staticmethod
    def _normalize_raw_contacts(raw_contacts: Any) -> Optional[Any]:
        """Normalize supported Genesis contact containers.

        Returns None for unsupported mock artifacts such as an unconfigured
        MagicMock so we can fall through to the next compatibility path.
        """
        if raw_contacts is None:
            return None
        if raw_contacts.__class__.__module__.startswith("unittest.mock"):
            return None
        if isinstance(raw_contacts, dict):
            return raw_contacts
        if isinstance(raw_contacts, (list, tuple)):
            return list(raw_contacts)
        try:
            return list(raw_contacts)
        except TypeError:
            return None

    def _contacts_from_genesis_dict(
        self,
        rec: _EntityRecord,
        raw: Dict[str, Any],
        env_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """Convert Genesis 0.4.4 entity.get_contacts() dicts to our API."""
        valid = self._contact_scalar_array(raw, "valid_mask", env_idx, np.bool_)
        contact_count = self._infer_contact_count(raw, valid, env_idx)
        if contact_count == 0:
            return []
        if valid is None:
            valid = np.ones(contact_count, dtype=np.bool_)

        link_a = self._contact_scalar_array(raw, "link_a", env_idx, np.int64)
        link_b = self._contact_scalar_array(raw, "link_b", env_idx, np.int64)
        geom_a = self._contact_scalar_array(raw, "geom_a", env_idx, np.int64)
        geom_b = self._contact_scalar_array(raw, "geom_b", env_idx, np.int64)
        penetration = self._contact_scalar_array(raw, "penetration", env_idx, np.float32)
        positions = self._contact_vector_array(raw, "position", env_idx)
        normals = self._contact_vector_array(raw, "normal", env_idx)
        forces_a = self._contact_vector_array(raw, "force_a", env_idx)
        forces_b = self._contact_vector_array(raw, "force_b", env_idx)

        target_links = self._entity_link_indices(rec)
        target_geoms = self._entity_geom_indices(rec)
        result: List[Dict[str, Any]] = []

        for i in range(contact_count):
            if i >= valid.size or not bool(valid[i]):
                continue

            a_is_target = self._contact_side_matches(
                self._value_at(link_a, i),
                self._value_at(geom_a, i),
                target_links,
                target_geoms,
            )
            b_is_target = self._contact_side_matches(
                self._value_at(link_b, i),
                self._value_at(geom_b, i),
                target_links,
                target_geoms,
            )
            if not (a_is_target or b_is_target):
                continue

            target_is_a = bool(a_is_target and not b_is_target)
            other_link = self._value_at(link_b if target_is_a else link_a, i)
            other_geom = self._value_at(geom_b if target_is_a else geom_a, i)
            other_name = self._entity_name_from_contact_indices(other_link, other_geom)

            raw_normal = self._vector_at(
                normals,
                i,
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )
            normal = -raw_normal if target_is_a else raw_normal
            force = self._vector_at(
                forces_a if target_is_a else forces_b,
                i,
                np.zeros(3, dtype=np.float32),
            )

            result.append({
                "body_b": other_name,
                "pos": self._vector_at(
                    positions,
                    i,
                    np.zeros(3, dtype=np.float32),
                ),
                "normal": normal.astype(np.float32),
                "force_n": float(np.linalg.norm(force)),
                "penetration": float(self._value_at(penetration, i, default=0.0)),
            })

        return result

    @staticmethod
    def _contact_scalar_array(
        raw: Dict[str, Any],
        key: str,
        env_idx: int,
        dtype: Any,
    ) -> Optional[np.ndarray]:
        value = raw.get(key)
        if value is None:
            return None
        arr = _to_numpy(value, dtype=dtype)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim >= 2:
            arr = arr[env_idx]
        return arr.reshape(-1)

    @staticmethod
    def _contact_vector_array(
        raw: Dict[str, Any],
        key: str,
        env_idx: int,
    ) -> Optional[np.ndarray]:
        value = raw.get(key)
        if value is None:
            return None
        arr = _to_numpy(value, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(1, -1)[:, :3]
        if arr.ndim >= 3:
            arr = arr[env_idx]
        return arr.reshape(-1, 3)

    def _infer_contact_count(
        self,
        raw: Dict[str, Any],
        valid: Optional[np.ndarray],
        env_idx: int,
    ) -> int:
        if valid is not None:
            return int(valid.size)
        for key in ("position", "normal", "force_a", "force_b"):
            vectors = self._contact_vector_array(raw, key, env_idx)
            if vectors is not None:
                return int(vectors.shape[0])
        for key in ("link_a", "link_b", "geom_a", "geom_b", "penetration"):
            scalars = self._contact_scalar_array(raw, key, env_idx, np.float32)
            if scalars is not None:
                return int(scalars.size)
        return 0

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if value.__class__.__module__.startswith("unittest.mock"):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _entity_link_indices(self, rec: _EntityRecord) -> set[int]:
        indices: set[int] = set()
        for link in getattr(rec.genesis_entity, "links", []) or []:
            idx = self._safe_int(getattr(link, "idx", None))
            if idx is not None:
                indices.add(idx)
        entity_idx = self._safe_int(getattr(rec.genesis_entity, "idx", None))
        if entity_idx is not None:
            indices.add(entity_idx)
        return indices

    def _entity_geom_indices(self, rec: _EntityRecord) -> set[int]:
        indices: set[int] = set()
        for geom in getattr(rec.genesis_entity, "geoms", []) or []:
            idx = self._safe_int(getattr(geom, "idx", None))
            if idx is not None:
                indices.add(idx)
        return indices

    def _entity_name_from_contact_indices(
        self,
        link_idx: Any,
        geom_idx: Any,
    ) -> str:
        link_int = self._safe_int(link_idx)
        geom_int = self._safe_int(geom_idx)
        with self._entity_lock:
            for name, rec in self._entities.items():
                if link_int is not None and link_int in self._entity_link_indices(rec):
                    return name
                if geom_int is not None and geom_int in self._entity_geom_indices(rec):
                    return name
        return "unknown"

    @staticmethod
    def _contact_side_matches(
        link_idx: Any,
        geom_idx: Any,
        target_links: set[int],
        target_geoms: set[int],
    ) -> bool:
        link_int = GenesisPhysicsEngine._safe_int(link_idx)
        geom_int = GenesisPhysicsEngine._safe_int(geom_idx)
        return (
            (link_int is not None and link_int in target_links)
            or (geom_int is not None and geom_int in target_geoms)
        )

    @staticmethod
    def _value_at(
        values: Optional[np.ndarray],
        index: int,
        default: Any = None,
    ) -> Any:
        if values is None or index >= values.size:
            return default
        return values[index]

    @staticmethod
    def _vector_at(
        values: Optional[np.ndarray],
        index: int,
        default: np.ndarray,
    ) -> np.ndarray:
        if values is None or index >= values.shape[0]:
            return default.copy()
        return np.asarray(values[index], dtype=np.float32).reshape(-1)[:3]

    def _reverse_entity_lookup(self, genesis_entity: Any) -> str:
        """Find the registered name for a given Genesis entity object."""
        with self._entity_lock:
            for name, rec in self._entities.items():
                if rec.genesis_entity is genesis_entity:
                    return name
        return "unknown"
