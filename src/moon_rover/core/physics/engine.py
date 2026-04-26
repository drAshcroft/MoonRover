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
from typing import Any, Dict, Optional

import yaml


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
    def configure(self, config: GenesisConfig) -> None:
        """Configure the physics engine with simulation parameters.

        Parameters:
            config: GenesisConfig object with all simulation parameters.

        Raises:
            RuntimeError: If called when scene is not in CONSTRUCTION phase.
        """
        raise NotImplementedError

    @abstractmethod
    def build_scene(self) -> None:
        """Build the scene and transition from CONSTRUCTION to SIMULATION phase.

        This method finalizes scene construction and prepares the engine for stepping.
        After this call, no additional entities can be added.

        Raises:
            RuntimeError: If called outside CONSTRUCTION phase or if scene is incomplete.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance simulation by dt seconds.

        Parameters:
            dt: Timestep in seconds. Should match config.timestep exactly for
                deterministic replay and fixed-step accounting.

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


# Import the concrete implementation so callers can do:
#   from moon_rover.core.physics.engine import GenesisPhysicsEngine
from moon_rover.core.physics._genesis_engine import GenesisPhysicsEngine  # noqa: E402, F401
