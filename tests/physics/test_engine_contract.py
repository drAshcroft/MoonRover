"""Contract tests for the PhysicsEngine ABC.

Locks in that the abstraction declares the full engine surface that downstream
systems (scene composers, scenarios, sensors, rover subsystems) depend on, so
those callers can target ``PhysicsEngine`` instead of the concrete Genesis class.
"""

from __future__ import annotations

import inspect

import pytest

from moon_rover.core.physics.engine import GenesisPhysicsEngine, PhysicsEngine


# Core simulation surface every backend must implement (abstract on the ABC).
REQUIRED_ABSTRACT = {
    # lifecycle
    "configure", "build_scene", "step", "teardown",
    # persistence / branching
    "save_snapshot", "restore_snapshot",
    # attachments
    "attach_bodies", "detach_bodies",
    # phase / solvers
    "get_phase",
    # entity registration & access
    "add_entity", "add_terrain_entity", "get_entity", "list_entities",
    # body state queries
    "get_body_pose", "get_body_velocity", "get_body_acceleration",
    "get_link_poses", "get_link_velocities",
    "get_dof_positions", "get_dof_velocities",
    # body state setters / actuation
    "set_body_pose", "set_body_velocity",
    "set_dof_positions", "set_dof_velocities", "apply_dof_forces",
    # terrain
    "get_terrain_height", "get_terrain_normal",
    # contacts
    "get_body_contacts", "is_in_contact",
    # clock
    "get_sim_time", "get_step_count",
}

# Optional capabilities: declared on the ABC (callers can target it) but with a
# default that raises NotImplementedError so a backend may omit them.
OPTIONAL_CAPABILITIES = {
    "register_raycaster", "query_raycaster",
    "add_camera", "set_camera_pose", "start_camera_recording",
    "render_camera", "stop_camera_recording",
}


def test_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        PhysicsEngine()  # type: ignore[abstract]


def test_required_methods_are_abstract_on_abc():
    abstract = set(PhysicsEngine.__abstractmethods__)
    # solver_backends is an abstract property, handled separately.
    missing = REQUIRED_ABSTRACT - abstract
    assert not missing, f"these should be abstract on PhysicsEngine: {sorted(missing)}"


def test_solver_backends_is_abstract_property():
    assert "solver_backends" in PhysicsEngine.__abstractmethods__
    assert isinstance(
        inspect.getattr_static(PhysicsEngine, "solver_backends"), property
    )


def test_optional_capabilities_declared_but_not_abstract():
    abstract = set(PhysicsEngine.__abstractmethods__)
    for name in OPTIONAL_CAPABILITIES:
        assert hasattr(PhysicsEngine, name), f"ABC missing capability {name!r}"
        assert name not in abstract, f"{name!r} should be optional, not abstract"


def test_optional_capabilities_default_to_not_implemented():
    # A trivial subclass implementing only the required methods must still be
    # constructible, and its optional capabilities raise NotImplementedError.
    methods = {n: (lambda self, *a, **k: None) for n in REQUIRED_ABSTRACT}
    methods["solver_backends"] = property(lambda self: {})
    Stub = type("Stub", (PhysicsEngine,), methods)
    engine = Stub()
    calls = {
        "register_raycaster": ("ray", "rover", 0, {}, 10.0),
        "query_raycaster": ("ray",),
        "add_camera": ("cam",),
        "set_camera_pose": ("cam",),
        "start_camera_recording": ("cam",),
        "render_camera": ("cam",),
        "stop_camera_recording": ("cam", "out.mp4"),
    }
    assert set(calls) == OPTIONAL_CAPABILITIES
    for name, args in calls.items():
        with pytest.raises(NotImplementedError):
            getattr(engine, name)(*args)


def test_concrete_engine_implements_full_contract():
    assert issubclass(GenesisPhysicsEngine, PhysicsEngine)
    assert not GenesisPhysicsEngine.__abstractmethods__, (
        "GenesisPhysicsEngine must implement the full ABC to be instantiable"
    )


def test_external_callers_only_use_declared_methods():
    """Every public engine method used outside core/physics must be on the ABC."""
    declared = {
        n for n in dir(PhysicsEngine)
        if not n.startswith("_")
    }
    # The union of required + optional + the property is the public contract.
    contract = REQUIRED_ABSTRACT | OPTIONAL_CAPABILITIES | {"solver_backends"}
    assert contract <= declared
