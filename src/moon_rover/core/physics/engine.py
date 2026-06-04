"""System 1: Genesis Physics Engine configuration, contracts, and interface.

This module defines the core physics engine abstraction for the Moon Rover simulation,
built on the Genesis physics simulator. It manages scene construction, fixed-timestep
simulation stepping, and state persistence for Monte Carlo branching and checkpointing.

Determinism contract
--------------------
- CPU rigid-body scenes are expected to be replayable at fixed timestep when the
  scene setup, solver settings, backend choice, and seed are held constant.
- Snapshot restore is an exact-time operation: restoring then replaying from the
  same step must reproduce the same branch trajectory within tight CPU tolerances.
- GPU backends may diverge slightly from CPU due to backend-specific execution and
  floating-point behavior. Treat CPU replay as the reference path and document any
  tolerated GPU drift rather than assuming bitwise identity.
- Seed changes are only meaningful when Genesis internals or caller-side setup use
  randomized behavior; repeated runs with the same seed must keep those choices
  stable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import yaml

NDArray = npt.NDArray[np.float32]


class ScenePhase(Enum):
    """Lifecycle phases of a physics scene.

    Attributes:
        CONSTRUCTION: Scene is being built; entities can be added.
        SIMULATION: Scene is actively running; no structural changes allowed.
        TEARDOWN: Scene is being destroyed; VRAM is being flushed.
    """
    CONSTRUCTION = "construction"
    SIMULATION = "simulation"
    TEARDOWN = "teardown"


@dataclass(frozen=True)
class AttachmentHandle:
    """Opaque handle for a runtime rigid-body attachment."""

    attachment_id: str


@dataclass
class GenesisConfig:
    """Configuration for Genesis physics engine.

    Parameters:
        gravity_vector: 3D gravity acceleration vector (m/s^2), typically [0, 0, -1.62].
        timestep: Fixed simulation timestep in seconds. Default 1/240 Hz (4.17 ms).
        contact_iterations: Number of contact solver iterations per step. Default 30.
        friction_model: Friction model name (e.g., "coulomb", "drucker_prager").
        collision_margin: Collision margin for broadphase in meters.
        broadphase: Broadphase algorithm ("aabb", "grid", etc.).
        substeps: Number of substeps per simulation step. Default 4.
        use_gpu: Whether to use GPU acceleration via CUDA.
        cuda_version: CUDA version string if use_gpu is True (e.g., "12.1").
        random_seed: Seed for deterministic simulation. Default 42.
        solver_map: Mapping of material type names to solver backend names.
        enable_sleeping: Allow inactive bodies to sleep for performance. Default True.
        sleep_velocity_threshold: Speed below which a body may sleep (m/s). Default 0.01.
        linear_damping: Global linear velocity damping coefficient. Default 0.04.
        angular_damping: Global angular velocity damping coefficient. Default 0.04.
    """
    gravity_vector: tuple[float, float, float]
    timestep: float = 1.0 / 240.0
    contact_iterations: int = 30
    friction_model: str = "coulomb"
    collision_margin: float = 0.001
    broadphase: str = "aabb"
    substeps: int = 4
    use_gpu: bool = False
    cuda_version: Optional[str] = None
    random_seed: int = 42
    solver_map: Dict[str, str] = field(default_factory=dict)
    enable_sleeping: bool = True
    sleep_velocity_threshold: float = 0.01
    linear_damping: float = 0.04
    angular_damping: float = 0.04

    @classmethod
    def from_yaml(cls, path: str) -> "GenesisConfig":
        """Load GenesisConfig from a physics YAML configuration file.

        Parameters:
            path: Path to physics.yaml (e.g., "configs/physics.yaml").

        Returns:
            Fully populated GenesisConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            KeyError: If required keys are missing from the YAML.
        """
        with open(path) as f:
            cfg = yaml.safe_load(f)
        grav = cfg["gravity"]["value"]
        return cls(
            gravity_vector=(float(grav[0]), float(grav[1]), float(grav[2])),
            timestep=float(cfg["timestep"]["seconds"]),
            contact_iterations=int(cfg["contact_solver"]["solver_iterations"]),
            friction_model=str(cfg["contact_solver"]["friction_model"]),
            collision_margin=float(cfg["contact_solver"]["collision_margin"]),
            broadphase=str(cfg["broadphase"]["algorithm"]),
            substeps=int(cfg["substeps"]["count"]),
            use_gpu=bool(cfg["gpu_backend"]["enabled"]),
            random_seed=int(cfg["determinism"]["random_seed"]),
            solver_map=dict(cfg.get("solver_map", {})),
            enable_sleeping=bool(cfg["performance"]["enable_sleeping"]),
            sleep_velocity_threshold=float(cfg["performance"]["sleep_velocity_threshold"]),
            linear_damping=float(cfg["damping"]["linear"]),
            angular_damping=float(cfg["damping"]["angular"]),
        )


class PhysicsEngine(ABC):
    """Abstract interface for a Genesis-based physics engine.

    This class defines the contract for physics simulation in the Moon Rover project.
    Implementers must handle scene lifecycle (construction, simulation, teardown),
    physics stepping, and snapshot persistence for Monte Carlo branching.

    The engine manages scene phases and ensures proper VRAM cleanup on teardown.
    Implementations must also document their determinism boundaries so downstream
    systems know whether replay guarantees are exact, approximate, or unsupported.
    """

    @abstractmethod
    def configure(
        self,
        config: GenesisConfig,
        show_viewer: bool = False,
        viewer_options: Optional[Any] = None,
        vis_options: Optional[Any] = None,
    ) -> None:
        """Configure the physics engine with simulation parameters.

        Parameters:
            config: GenesisConfig object with all simulation parameters.
            show_viewer: Open an interactive 3-D viewer window if supported.
            viewer_options: Backend-specific viewer options, or None for defaults.
            vis_options: Backend-specific visualization options (e.g. shadows,
                ambient light), or None for defaults.

        Raises:
            RuntimeError: If called when scene is not in CONSTRUCTION phase.
        """
        raise NotImplementedError

    @abstractmethod
    def build_scene(self, n_envs: int = 1) -> None:
        """Build the scene and transition from CONSTRUCTION to SIMULATION phase.

        This method finalizes scene construction and prepares the engine for stepping.
        After this call, no additional entities can be added.

        Parameters:
            n_envs: Number of parallel environments. Implementations may support
                only ``n_envs=1``.

        Raises:
            RuntimeError: If called outside CONSTRUCTION phase or if scene is incomplete.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float, render: bool = True) -> None:
        """Advance simulation by dt seconds.

        Parameters:
            dt: Timestep in seconds. Should match config.timestep exactly for
                deterministic replay and fixed-step accounting.
            render: Whether to update the viewer/visualizer this step. Set False
                for headless throughput or intermediate substeps.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
        """
        raise NotImplementedError

    @abstractmethod
    def teardown(self) -> None:
        """Explicitly destroy scene and flush VRAM.

        Transitions to TEARDOWN phase. Implementations may distinguish between
        scene-local cleanup and process-global runtime destruction when the
        backend owns singleton state. Any such policy must be explicit and must
        not allow teardown to hang indefinitely on the supported path.

        Raises:
            RuntimeError: If called in TEARDOWN phase.
        """
        raise NotImplementedError

    @abstractmethod
    def save_snapshot(self) -> bytes:
        """Save full scene state (all rigid bodies, constraints, forces) to bytes.

        Returns:
            Serialized scene state suitable for restoration or checkpointing.

        Raises:
            RuntimeError: If called outside SIMULATION phase.
        """
        raise NotImplementedError

    @abstractmethod
    def restore_snapshot(self, data: bytes) -> None:
        """Restore scene from a previously saved snapshot.

        Used for Monte Carlo branching: a checkpoint is saved, then multiple
        trajectories are explored from that point. Exact CPU replay assumes the
        same backend, timestep, solver settings, and compatible process-global
        Genesis runtime configuration are preserved across the branch.

        Restoration is an exact-time operation. After this call returns, the
        engine state must match the saved snapshot without performing any hidden
        physics advancement. Pose, orientation, linear velocity, angular
        velocity, DOF state, sim_time, and step_count must all be consistent
        with the saved checkpoint so replayed branches start from the exact same
        physical state.

        Parameters:
            data: Snapshot bytes from save_snapshot().

        Raises:
            RuntimeError: If called outside SIMULATION phase.
            ValueError: If snapshot data is corrupted or incompatible.
        """
        raise NotImplementedError

    @abstractmethod
    def attach_bodies(self, parent: str, child: str) -> AttachmentHandle:
        """Rigidly attach a child body to a parent body at its current pose.

        The child retains its current world-frame pose at attach time and then
        follows the parent's relative transform until detached.
        """
        raise NotImplementedError

    @abstractmethod
    def detach_bodies(self, handle: AttachmentHandle) -> None:
        """Release a previously attached child body."""
        raise NotImplementedError

    @abstractmethod
    def get_phase(self) -> ScenePhase:
        """Return the current scene lifecycle phase.

        Returns:
            Current ScenePhase (CONSTRUCTION, SIMULATION, or TEARDOWN).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def solver_backends(self) -> Dict[str, str]:
        """Map material types to their solver backends.

        Returns:
            Dictionary mapping material_type (e.g., "regolith", "metal") to
            solver_name (e.g., "drucker_prager", "coulomb_friction").

        Example:
            {
                "regolith": "drucker_prager",
                "metal": "coulomb_friction",
                "rubber": "coulomb_friction"
            }
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Entity registration (CONSTRUCTION phase)
    # ------------------------------------------------------------------

    @abstractmethod
    def add_entity(
        self,
        name: str,
        morph: Any,
        material: Any,
        entity_type: str = "rigid",
        **kwargs: Any,
    ) -> Any:
        """Register a simulation entity during scene construction.

        Parameters:
            name: Unique entity name used for all subsequent queries.
            morph: Backend morph/geometry descriptor (or a portable URDF string).
            material: Backend material, or None to auto-select by ``entity_type``.
            entity_type: One of "rigid", "articulated", "fixed", "kinematic",
                "mpm", "terrain".
            **kwargs: Additional backend-specific entity options.

        Returns:
            The backend entity handle.

        Raises:
            RuntimeError: If called outside CONSTRUCTION phase.
            ValueError: If ``name`` is already registered.
        """
        raise NotImplementedError

    @abstractmethod
    def add_terrain_entity(
        self,
        name: str,
        height_field: NDArray,
        size: List[float],
        collision: bool = True,
        visualization: bool = True,
    ) -> Any:
        """Register a terrain heightfield entity during construction.

        Parameters:
            name: Unique entity name.
            height_field: (H, W) array of terrain heights in metres.
            size: [size_x, size_y] world dimensions in metres.
            collision: Whether the terrain participates in collision.
            visualization: Whether the terrain mesh is rendered.

        Returns:
            The backend terrain entity handle.
        """
        raise NotImplementedError

    @abstractmethod
    def get_entity(self, name: str) -> Any:
        """Return the backend entity handle registered under ``name``."""
        raise NotImplementedError

    @abstractmethod
    def list_entities(self) -> List[str]:
        """Return the names of all registered entities."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Body state queries
    # ------------------------------------------------------------------

    @abstractmethod
    def get_body_pose(
        self, entity_name: str, env_idx: int = 0
    ) -> Tuple[NDArray, NDArray]:
        """Return the world-frame ``(position, quaternion)`` of a rigid entity."""
        raise NotImplementedError

    @abstractmethod
    def get_body_velocity(
        self, entity_name: str, env_idx: int = 0
    ) -> Tuple[NDArray, NDArray]:
        """Return the world-frame ``(linear, angular)`` velocity of an entity."""
        raise NotImplementedError

    @abstractmethod
    def get_body_acceleration(
        self, entity_name: str, env_idx: int = 0
    ) -> Tuple[NDArray, NDArray]:
        """Return the body's ``(linear, angular)`` acceleration from velocity deltas.

        Acceleration may be tracked lazily; the first query after subscribing a
        body can return zeros until a subsequent step provides a delta.
        """
        raise NotImplementedError

    @abstractmethod
    def get_link_poses(
        self, entity_name: str, env_idx: int = 0
    ) -> List[Tuple[NDArray, NDArray]]:
        """Return the world-frame pose of every link in an articulated entity."""
        raise NotImplementedError

    @abstractmethod
    def get_link_velocities(
        self, entity_name: str, env_idx: int = 0
    ) -> List[Tuple[NDArray, NDArray]]:
        """Return the world-frame velocity of every link in an articulated entity."""
        raise NotImplementedError

    @abstractmethod
    def get_dof_positions(self, entity_name: str, env_idx: int = 0) -> NDArray:
        """Return the joint DOF positions of an articulated entity."""
        raise NotImplementedError

    @abstractmethod
    def get_dof_velocities(self, entity_name: str, env_idx: int = 0) -> NDArray:
        """Return the joint DOF velocities of an articulated entity."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Body state setters / actuation (SIMULATION phase)
    # ------------------------------------------------------------------

    @abstractmethod
    def set_body_pose(
        self, entity_name: str, pos: NDArray, quat: NDArray, env_idx: int = 0
    ) -> None:
        """Set the world-frame pose of a rigid entity."""
        raise NotImplementedError

    @abstractmethod
    def set_body_velocity(
        self, entity_name: str, lin_vel: NDArray, ang_vel: NDArray, env_idx: int = 0
    ) -> None:
        """Set the world-frame velocity of a rigid entity."""
        raise NotImplementedError

    @abstractmethod
    def set_dof_positions(
        self, entity_name: str, positions: NDArray, env_idx: int = 0
    ) -> None:
        """Set the joint DOF positions of an articulated entity."""
        raise NotImplementedError

    @abstractmethod
    def set_dof_velocities(
        self, entity_name: str, velocities: NDArray, env_idx: int = 0
    ) -> None:
        """Set the joint DOF velocities of an articulated entity."""
        raise NotImplementedError

    @abstractmethod
    def apply_dof_forces(
        self, entity_name: str, forces: NDArray, env_idx: int = 0
    ) -> None:
        """Apply forces/torques to the joint DOFs of an articulated entity."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Terrain queries
    # ------------------------------------------------------------------

    @abstractmethod
    def get_terrain_height(self, x: float, y: float) -> float:
        """Return the terrain surface height (metres) at world ``(x, y)``."""
        raise NotImplementedError

    @abstractmethod
    def get_terrain_normal(self, x: float, y: float) -> NDArray:
        """Return the unit terrain surface normal at world ``(x, y)``."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Contact queries
    # ------------------------------------------------------------------

    @abstractmethod
    def get_body_contacts(self, entity_name: str) -> List[Dict[str, Any]]:
        """Return the active contacts involving ``entity_name``.

        Each contact is a dict with at least ``other``, ``position``,
        ``normal``, and ``force`` keys.
        """
        raise NotImplementedError

    @abstractmethod
    def is_in_contact(self, entity_a: str, entity_b: str) -> bool:
        """Return whether two named entities are currently in contact."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Simulation clock
    # ------------------------------------------------------------------

    @abstractmethod
    def get_sim_time(self) -> float:
        """Return the elapsed simulation time in seconds."""
        raise NotImplementedError

    @abstractmethod
    def get_step_count(self) -> int:
        """Return the number of simulation steps taken since build."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Optional capabilities
    #
    # Declared on the contract so callers can target the abstraction, but given
    # default implementations that raise NotImplementedError: a backend without
    # a rasterizer (video capture) or ray sensor support is still a valid
    # PhysicsEngine. Backends that support them override these.
    # ------------------------------------------------------------------

    def register_raycaster(
        self,
        name: str,
        link_entity: str,
        link_idx: int,
        pattern_config: Dict[str, Any],
        max_range: float,
    ) -> None:
        """Register a ray-cast sensor (e.g. LiDAR) during construction.

        Optional capability. Raises NotImplementedError if unsupported.
        """
        raise NotImplementedError("this backend does not support ray-cast sensors")

    def query_raycaster(self, name: str) -> Dict[str, NDArray]:
        """Return the latest hits for a registered ray-cast sensor.

        Optional capability. Raises NotImplementedError if unsupported.
        """
        raise NotImplementedError("this backend does not support ray-cast sensors")

    def add_camera(
        self,
        name: str,
        resolution: Tuple[int, int] = (1280, 720),
        pos: Tuple[float, float, float] = (6.0, -6.0, 4.0),
        lookat: Tuple[float, float, float] = (0.0, 0.0, 0.5),
        fov: float = 45.0,
        gui: bool = False,
    ) -> Any:
        """Register an offscreen camera for rendering / video capture (CONSTRUCTION).

        Optional capability. Raises NotImplementedError if unsupported.
        """
        raise NotImplementedError("this backend does not support cameras")

    def set_camera_pose(
        self,
        name: str,
        pos: Optional[Tuple[float, float, float]] = None,
        lookat: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Reposition a registered camera mid-simulation.

        Optional capability. Raises NotImplementedError if unsupported.
        """
        raise NotImplementedError("this backend does not support cameras")

    def start_camera_recording(self, name: str) -> None:
        """Begin accumulating rendered frames for a camera into a video buffer.

        Optional capability. Raises NotImplementedError if unsupported.
        """
        raise NotImplementedError("this backend does not support cameras")

    def render_camera(self, name: str) -> Any:
        """Render one frame from a camera; appends to the video when recording.

        Optional capability. Raises NotImplementedError if unsupported.
        """
        raise NotImplementedError("this backend does not support cameras")

    def stop_camera_recording(self, name: str, save_path: str, fps: int = 30) -> None:
        """Finalize a camera recording and encode it to ``save_path``.

        Optional capability. Raises NotImplementedError if unsupported.
        """
        raise NotImplementedError("this backend does not support cameras")


# Import the concrete implementation so callers can do:
#   from moon_rover.core.physics.engine import GenesisPhysicsEngine
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine  # noqa: E402, F401
