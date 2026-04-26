"""System 3: Programmatic URDF Generation for Lunar Assets.

This module provides interfaces for building URDF (Unified Robot Description Format)
XML files programmatically for the rover, antenna, and moonbase. It includes validation
at multiple stages: XML schema, physics sanity checks, and Genesis loader compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, List, Mapping, Sequence
from xml.etree import ElementTree as ET


@dataclass
class MaterialProperties:
    """Material properties for physics objects.

    Parameters:
        friction: Coefficient of friction (0-1). Typical lunar regolith ~0.6.
        density: Material density in kg/m^3. Lunar regolith ~1500 kg/m^3 (compacted).
        restitution: Coefficient of restitution (0-1). 0 = perfectly inelastic.
        name: Unique identifier for this material (e.g., "lunar_regolith", "aluminum").
    """
    friction: float
    density: float
    restitution: float
    name: str


class URDFValidationStage(Enum):
    """URDF validation stages with increasing strictness.

    Attributes:
        XML_SCHEMA: Basic XML schema validation (well-formed URDF structure).
        PHYSICS_SANITY: Physics parameter sanity checks (mass > 0, inertias valid, etc.).
        GENESIS_LOAD: Genesis loader compatibility (Genesis-specific constraints).
    """
    XML_SCHEMA = "xml_schema"
    PHYSICS_SANITY = "physics_sanity"
    GENESIS_LOAD = "genesis_load"


class URDFBuilder(ABC):
    """Abstract interface for programmatic URDF generation.

    Builds complete URDF XML definitions for rover, antenna, and moonbase.
    All generated URDFs are validated at configurable stages before use.
    """

    @abstractmethod
    def build_rover(self, config: dict) -> str:
        """Generate complete rover URDF from configuration dictionary.

        The rover includes:
        - Body (chassis, wheels, transmission)
        - Sensors (cameras, IMU, thermal, spectrometer)
        - Actuators (motors, wheels, steering)
        - Cable attachment points

        Parameters:
            config: Dictionary with keys like:
                - "mass_kg": Total rover mass
                - "wheel_radius": Wheel radius in meters
                - "wheel_count": Number of wheels
                - "track_width": Distance between wheel pairs
                - "material_name": Primary material (e.g., "aluminum")
                - "sensor_config": Dict of sensor specifications
                - "motor_specs": Dict of motor parameters

        Returns:
            Complete URDF XML as string, ready for file I/O or direct loading.

        Raises:
            ValueError: If config is missing required keys or has invalid values.
            KeyError: If referenced materials are not in the library.
        """
        raise NotImplementedError

    @abstractmethod
    def build_antenna(self) -> str:
        """Generate antenna URDF with cable attachment points.

        The antenna includes:
        - Boom (main structural element)
        - Antenna array
        - Cable connection points for winch integration

        Returns:
            Complete antenna URDF XML as string.

        Raises:
            RuntimeError: If antenna template is not initialized.
        """
        raise NotImplementedError

    @abstractmethod
    def build_moonbase(self) -> str:
        """Generate moonbase structure URDF (fixed anchor point).

        The moonbase includes:
        - Fixed foundation (immovable)
        - Cable anchors
        - Sensor mounting points

        Returns:
            Complete moonbase URDF XML as string.

        Raises:
            RuntimeError: If moonbase template is not initialized.
        """
        raise NotImplementedError

    @abstractmethod
    def validate(self, urdf_xml: str, stage: URDFValidationStage) -> List[str]:
        """Validate URDF XML at specified strictness level.

        Validation is progressive: GENESIS_LOAD includes all checks from earlier stages.

        Parameters:
            urdf_xml: URDF XML string to validate.
            stage: Validation stage (XML_SCHEMA, PHYSICS_SANITY, or GENESIS_LOAD).

        Returns:
            List of error messages. Empty list indicates all checks passed.
            Each error is a human-readable description of the validation failure.

        Example:
            errors = builder.validate(urdf_str, URDFValidationStage.PHYSICS_SANITY)
            if errors:
                for err in errors:
                    print(f"Validation error: {err}")
        """
        raise NotImplementedError


def _format_float(value: float) -> str:
    """Render floats compactly while keeping deterministic XML output."""
    return f"{float(value):.6g}"


def _vec(values: Iterable[float]) -> str:
    """Render a numeric vector for URDF attributes."""
    return " ".join(_format_float(value) for value in values)


def _sub(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    """Create an XML child element and return it."""
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attrs.items()})


def _inertia_diagonal(mass_kg: float, size_xyz: Sequence[float]) -> tuple[float, float, float]:
    """Return the diagonal box inertia tensor for the supplied dimensions."""
    lx, ly, lz = (float(value) for value in size_xyz)
    ixx = (mass_kg / 12.0) * ((ly * ly) + (lz * lz))
    iyy = (mass_kg / 12.0) * ((lx * lx) + (lz * lz))
    izz = (mass_kg / 12.0) * ((lx * lx) + (ly * ly))
    return (ixx, iyy, izz)


def _append_inertial(
    link: ET.Element,
    *,
    mass_kg: float,
    inertia_xyz: Sequence[float],
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    origin_rpy: Sequence[float] = (0.0, 0.0, 0.0),
) -> None:
    """Attach a simple inertial block to a URDF link."""
    inertial = _sub(link, "inertial")
    _sub(inertial, "origin", xyz=_vec(origin_xyz), rpy=_vec(origin_rpy))
    _sub(inertial, "mass", value=_format_float(mass_kg))
    ixx, iyy, izz = (float(value) for value in inertia_xyz)
    _sub(
        inertial,
        "inertia",
        ixx=_format_float(ixx),
        ixy="0",
        ixz="0",
        iyy=_format_float(iyy),
        iyz="0",
        izz=_format_float(izz),
    )


def _append_geometry(
    link: ET.Element,
    *,
    shape: str,
    size: Sequence[float] | None = None,
    radius: float | None = None,
    length: float | None = None,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    origin_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    material_name: str | None = None,
) -> None:
    """Attach matching visual and collision geometry to a link."""
    for tag in ("visual", "collision"):
        container = _sub(link, tag)
        _sub(container, "origin", xyz=_vec(origin_xyz), rpy=_vec(origin_rpy))
        geometry = _sub(container, "geometry")
        if shape == "box":
            if size is None:
                raise ValueError("box geometry requires size")
            _sub(geometry, "box", size=_vec(size))
        elif shape == "cylinder":
            if radius is None or length is None:
                raise ValueError("cylinder geometry requires radius and length")
            _sub(
                geometry,
                "cylinder",
                radius=_format_float(radius),
                length=_format_float(length),
            )
        elif shape == "sphere":
            if radius is None:
                raise ValueError("sphere geometry requires radius")
            _sub(geometry, "sphere", radius=_format_float(radius))
        else:
            raise ValueError(f"Unsupported geometry shape {shape!r}")

        if tag == "visual" and material_name:
            _sub(container, "material", name=material_name)


def _append_link(
    robot: ET.Element,
    *,
    name: str,
    mass_kg: float,
    inertia_xyz: Sequence[float],
    shape: str,
    size: Sequence[float] | None = None,
    radius: float | None = None,
    length: float | None = None,
    inertial_origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    geometry_origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    geometry_origin_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    material_name: str | None = None,
) -> ET.Element:
    """Create a URDF link with inertial, visual, and collision blocks."""
    link = _sub(robot, "link", name=name)
    _append_inertial(
        link,
        mass_kg=mass_kg,
        inertia_xyz=inertia_xyz,
        origin_xyz=inertial_origin_xyz,
    )
    _append_geometry(
        link,
        shape=shape,
        size=size,
        radius=radius,
        length=length,
        origin_xyz=geometry_origin_xyz,
        origin_rpy=geometry_origin_rpy,
        material_name=material_name,
    )
    return link


def _append_joint(
    robot: ET.Element,
    *,
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    origin_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    origin_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    axis: Sequence[float] | None = None,
    limit: tuple[float, float] | None = None,
) -> ET.Element:
    """Create a URDF joint linking two named links."""
    joint = _sub(robot, "joint", name=name, type=joint_type)
    _sub(joint, "parent", link=parent)
    _sub(joint, "child", link=child)
    _sub(joint, "origin", xyz=_vec(origin_xyz), rpy=_vec(origin_rpy))
    if axis is not None:
        _sub(joint, "axis", xyz=_vec(axis))
    if limit is not None:
        lower, upper = limit
        _sub(
            joint,
            "limit",
            lower=_format_float(lower),
            upper=_format_float(upper),
            effort="100",
            velocity="10",
        )
    return joint


class GenesisURDFBuilder(URDFBuilder):
    """Concrete URDF builder for the Genesis scene pipeline."""

    _ROVER_REQUIRED_KEYS = (
        "rover_id",
        "mass_kg",
        "wheel_radius",
        "wheel_count",
        "wheel_positions",
        "dimensions",
        "inertia",
        "com_offset",
        "arm_base_position",
    )
    _SUPPORTED_GENESIS_JOINT_TYPES = {"fixed", "continuous", "revolute", "prismatic"}

    def build_rover(self, config: Mapping[str, Any]) -> str:
        """Generate a rover URDF with wheel, mount, and sensor attachment links."""
        self._validate_rover_config(config)

        rover_id = str(config["rover_id"])
        mass_kg = float(config["mass_kg"])
        wheel_radius = float(config["wheel_radius"])
        wheel_count = int(config["wheel_count"])
        wheel_positions = [tuple(float(v) for v in pos) for pos in config["wheel_positions"]]
        dimensions = tuple(float(v) for v in config["dimensions"])
        inertia = tuple(float(v) for v in config["inertia"])
        com_offset = tuple(float(v) for v in config["com_offset"])
        arm_base_position = tuple(float(v) for v in config["arm_base_position"])
        material_name = str(config.get("material_name", "aluminum"))

        robot = ET.Element("robot", {"name": rover_id})
        chassis_half_height = dimensions[2] / 2.0
        chassis_link = "base_link"
        _append_link(
            robot,
            name=chassis_link,
            mass_kg=mass_kg,
            inertia_xyz=inertia,
            shape="box",
            size=dimensions,
            inertial_origin_xyz=com_offset,
            material_name=material_name,
        )

        self._append_rover_mounts(
            robot,
            chassis_link=chassis_link,
            dimensions=dimensions,
            arm_base_position=arm_base_position,
            material_name=material_name,
        )

        if wheel_count == 3:
            self._append_tricycle_wheels(
                robot,
                chassis_link=chassis_link,
                wheel_radius=wheel_radius,
                wheel_positions=wheel_positions,
                chassis_half_height=chassis_half_height,
                material_name=material_name,
            )
        else:
            self._append_direct_wheels(
                robot,
                chassis_link=chassis_link,
                wheel_radius=wheel_radius,
                wheel_positions=wheel_positions,
                chassis_half_height=chassis_half_height,
                material_name=material_name,
            )

        arm_dof = int(config.get("arm_dof", 0) or 0)
        if arm_dof > 0:
            self._append_arm(
                robot,
                parent_link="arm_mount",
                num_dof=arm_dof,
                reach_m=float(config.get("arm_reach_m", 2.0)),
                link_lengths=config.get("arm_link_lengths"),
                joint_axes=config.get("arm_joint_axes"),
                joint_limits=config.get("arm_joint_limits"),
                gripper_type=str(config.get("gripper_type", "parallel_jaw")),
                gripper_stroke_m=float(config.get("gripper_stroke_m", 0.1)),
                gripper_finger_length_m=float(
                    config.get("gripper_finger_length_m", 0.1)
                ),
                gripper_finger_width_m=float(
                    config.get("gripper_finger_width_m", 0.03)
                ),
                material_name=material_name,
            )

        return self._to_xml(robot)

    def build_antenna(self) -> str:
        """Generate an antenna URDF using the default antenna dimensions."""
        base_plate_m = (0.4, 0.4, 0.05)
        base_mass_kg = 2.5
        mast_height_m = 1.2
        mast_radius_m = 0.02
        mast_mass_kg = 1.0
        dish_diameter_m = 0.6
        dish_mass_kg = 0.8
        connector_mass_kg = 0.2

        robot = ET.Element("robot", {"name": "antenna_unit"})
        _append_link(
            robot,
            name="base_link",
            mass_kg=base_mass_kg,
            inertia_xyz=_inertia_diagonal(base_mass_kg, base_plate_m),
            shape="box",
            size=base_plate_m,
            material_name="metal",
        )
        _append_link(
            robot,
            name="mast_link",
            mass_kg=mast_mass_kg,
            inertia_xyz=_inertia_diagonal(
                mast_mass_kg,
                (mast_radius_m * 2.0, mast_radius_m * 2.0, mast_height_m),
            ),
            shape="cylinder",
            radius=mast_radius_m,
            length=mast_height_m,
            material_name="metal",
        )
        _append_joint(
            robot,
            name="base_to_mast",
            joint_type="fixed",
            parent="base_link",
            child="mast_link",
            origin_xyz=(0.0, 0.0, (base_plate_m[2] / 2.0) + (mast_height_m / 2.0)),
        )
        dish_radius = dish_diameter_m / 2.0
        _append_link(
            robot,
            name="dish_link",
            mass_kg=dish_mass_kg,
            inertia_xyz=_inertia_diagonal(
                dish_mass_kg,
                (dish_diameter_m, dish_diameter_m, max(0.04, dish_radius / 4.0)),
            ),
            shape="sphere",
            radius=dish_radius,
            material_name="metal",
        )
        _append_joint(
            robot,
            name="mast_to_dish",
            joint_type="fixed",
            parent="mast_link",
            child="dish_link",
            origin_xyz=(0.0, 0.0, mast_height_m / 2.0),
        )
        _append_link(
            robot,
            name="connector_link",
            mass_kg=connector_mass_kg,
            inertia_xyz=_inertia_diagonal(connector_mass_kg, (0.12, 0.08, 0.05)),
            shape="box",
            size=(0.12, 0.08, 0.05),
            material_name="metal",
        )
        _append_joint(
            robot,
            name="base_to_connector",
            joint_type="fixed",
            parent="base_link",
            child="connector_link",
            origin_xyz=(0.0, 0.0, -base_plate_m[2] / 2.0),
        )
        return self._to_xml(robot)

    def build_moonbase(self) -> str:
        """Generate a fixed moonbase URDF with docking and cable anchor links."""
        robot = ET.Element("robot", {"name": "moonbase"})
        _append_link(
            robot,
            name="world",
            mass_kg=0.0,
            inertia_xyz=(0.0, 0.0, 0.0),
            shape="box",
            size=(0.01, 0.01, 0.01),
            material_name="metal",
        )
        _append_link(
            robot,
            name="moonbase_base",
            mass_kg=5000.0,
            inertia_xyz=_inertia_diagonal(5000.0, (10.0, 8.0, 4.0)),
            shape="box",
            size=(10.0, 8.0, 4.0),
            material_name="metal",
        )
        _append_joint(
            robot,
            name="world_to_moonbase",
            joint_type="fixed",
            parent="world",
            child="moonbase_base",
            origin_xyz=(0.0, 0.0, 2.0),
        )
        fixed_offsets = {
            "docking_port_front": (2.0, 0.0, 0.0),
            "docking_port_rear": (-2.0, 0.0, 0.0),
            "cable_anchor_left": (0.0, 3.5, 1.0),
            "cable_anchor_right": (0.0, -3.5, 1.0),
            "comm_tower": (0.0, 0.0, 4.5),
        }
        for name, offset in fixed_offsets.items():
            size = (0.25, 0.25, 1.0) if name == "comm_tower" else (0.2, 0.2, 0.2)
            mass = 25.0 if name == "comm_tower" else 5.0
            _append_link(
                robot,
                name=name,
                mass_kg=mass,
                inertia_xyz=_inertia_diagonal(mass, size),
                shape="box",
                size=size,
                material_name="metal",
            )
            _append_joint(
                robot,
                name=f"moonbase_to_{name}",
                joint_type="fixed",
                parent="moonbase_base",
                child=name,
                origin_xyz=offset,
            )
        return self._to_xml(robot)

    def validate(self, urdf_xml: str, stage: URDFValidationStage) -> List[str]:
        """Validate URDF XML progressively up to the requested stage."""
        errors: List[str] = []
        try:
            root = ET.fromstring(urdf_xml)
        except ET.ParseError as exc:
            return [f"URDF XML is not well-formed: {exc}"]

        if root.tag != "robot":
            errors.append(f"URDF root element must be <robot>, got <{root.tag}>.")
            return errors

        if stage is URDFValidationStage.XML_SCHEMA:
            return errors

        errors.extend(self._validate_physics_sanity(root))
        if stage is URDFValidationStage.PHYSICS_SANITY:
            return errors

        errors.extend(self._validate_genesis_load(root))
        return errors

    @classmethod
    def _validate_rover_config(cls, config: Mapping[str, Any]) -> None:
        """Raise ValueError when the rover config is incomplete or invalid."""
        missing = [key for key in cls._ROVER_REQUIRED_KEYS if key not in config]
        if missing:
            raise ValueError(
                "Rover URDF config missing required keys: " + ", ".join(sorted(missing))
            )

        wheel_count = int(config["wheel_count"])
        if wheel_count not in (2, 3, 4):
            raise ValueError(f"wheel_count must be one of 2, 3, or 4; got {wheel_count}")

        wheel_positions = list(config["wheel_positions"])
        if len(wheel_positions) != wheel_count:
            raise ValueError(
                f"wheel_positions length {len(wheel_positions)} does not match "
                f"wheel_count {wheel_count}"
            )

        mass_kg = float(config["mass_kg"])
        if mass_kg <= 0.0:
            raise ValueError(f"mass_kg must be > 0, got {mass_kg}")

        wheel_radius = float(config["wheel_radius"])
        if wheel_radius <= 0.0:
            raise ValueError(f"wheel_radius must be > 0, got {wheel_radius}")

        dimensions = tuple(float(v) for v in config["dimensions"])
        if len(dimensions) != 3 or any(value <= 0.0 for value in dimensions):
            raise ValueError("dimensions must contain three positive values")

        inertia = tuple(float(v) for v in config["inertia"])
        if len(inertia) != 3 or any(value <= 0.0 for value in inertia):
            raise ValueError("inertia must contain three positive diagonal values")

        com_offset = tuple(float(v) for v in config["com_offset"])
        if len(com_offset) != 3:
            raise ValueError("com_offset must contain exactly three values")

        arm_base = tuple(float(v) for v in config["arm_base_position"])
        if len(arm_base) != 3:
            raise ValueError("arm_base_position must contain exactly three values")

    @staticmethod
    def _append_rover_mounts(
        robot: ET.Element,
        *,
        chassis_link: str,
        dimensions: Sequence[float],
        arm_base_position: Sequence[float],
        material_name: str,
    ) -> None:
        """Add attachment links for the arm, cable spool, and sensors."""
        mount_defs = {
            "arm_mount": tuple(float(value) for value in arm_base_position),
            "cable_spool_mount": (-dimensions[0] / 2.0 + 0.2, 0.0, 0.0),
            "sensor_mount_front": (dimensions[0] / 2.0 - 0.2, 0.0, dimensions[2] / 2.0),
            "sensor_mount_top": (0.0, 0.0, dimensions[2] / 2.0 + 0.1),
            "sensor_mount_rear": (-dimensions[0] / 2.0 + 0.1, 0.0, dimensions[2] / 2.0),
        }
        for name, offset in mount_defs.items():
            _append_link(
                robot,
                name=name,
                mass_kg=0.05,
                inertia_xyz=_inertia_diagonal(0.05, (0.1, 0.1, 0.1)),
                shape="box",
                size=(0.1, 0.1, 0.1),
                material_name=material_name,
            )
            _append_joint(
                robot,
                name=f"base_to_{name}",
                joint_type="fixed",
                parent=chassis_link,
                child=name,
                origin_xyz=offset,
            )

    @staticmethod
    def _wheel_names(wheel_positions: Sequence[Sequence[float]]) -> List[str]:
        """Return deterministic wheel link names for the supplied wheel count."""
        count = len(wheel_positions)
        if count == 2:
            return ["left_wheel", "right_wheel"]
        if count == 3:
            return ["front_wheel", "rear_left_wheel", "rear_right_wheel"]
        return [
            "front_left_wheel",
            "front_right_wheel",
            "rear_left_wheel",
            "rear_right_wheel",
        ]

    def _append_direct_wheels(
        self,
        robot: ET.Element,
        *,
        chassis_link: str,
        wheel_radius: float,
        wheel_positions: Sequence[Sequence[float]],
        chassis_half_height: float,
        material_name: str,
    ) -> None:
        """Attach wheel links directly to the chassis for 2- and 4-wheel rovers."""
        wheel_width = max(wheel_radius * 0.5, 0.12)
        wheel_mass = max(1.0, wheel_radius * 8.0)
        wheel_size = (wheel_radius * 2.0, wheel_width, wheel_radius * 2.0)

        for name, position in zip(self._wheel_names(wheel_positions), wheel_positions):
            _append_link(
                robot,
                name=name,
                mass_kg=wheel_mass,
                inertia_xyz=_inertia_diagonal(wheel_mass, wheel_size),
                shape="cylinder",
                radius=wheel_radius,
                length=wheel_width,
                geometry_origin_rpy=(math.pi / 2.0, 0.0, 0.0),
                material_name="rubber",
            )
            px, py, pz = (float(value) for value in position)
            _append_joint(
                robot,
                name=f"{name}_joint",
                joint_type="continuous",
                parent=chassis_link,
                child=name,
                origin_xyz=(px, py, pz - chassis_half_height),
                axis=(0.0, 1.0, 0.0),
            )

    def _append_tricycle_wheels(
        self,
        robot: ET.Element,
        *,
        chassis_link: str,
        wheel_radius: float,
        wheel_positions: Sequence[Sequence[float]],
        chassis_half_height: float,
        material_name: str,
    ) -> None:
        """Attach a steerable front wheel and two driven rear wheels."""
        wheel_width = max(wheel_radius * 0.5, 0.12)
        wheel_mass = max(1.0, wheel_radius * 8.0)
        wheel_size = (wheel_radius * 2.0, wheel_width, wheel_radius * 2.0)

        steering_position = tuple(float(value) for value in wheel_positions[0])
        _append_link(
            robot,
            name="front_steering_knuckle",
            mass_kg=0.75,
            inertia_xyz=_inertia_diagonal(0.75, (0.15, 0.15, 0.15)),
            shape="box",
            size=(0.15, 0.15, 0.15),
            material_name=material_name,
        )
        _append_joint(
            robot,
            name="front_steering_joint",
            joint_type="revolute",
            parent=chassis_link,
            child="front_steering_knuckle",
            origin_xyz=(
                steering_position[0],
                steering_position[1],
                steering_position[2] - chassis_half_height,
            ),
            axis=(0.0, 0.0, 1.0),
            limit=(-0.610865, 0.610865),
        )

        front_name, rear_left_name, rear_right_name = self._wheel_names(wheel_positions)
        _append_link(
            robot,
            name=front_name,
            mass_kg=wheel_mass,
            inertia_xyz=_inertia_diagonal(wheel_mass, wheel_size),
            shape="cylinder",
            radius=wheel_radius,
            length=wheel_width,
            geometry_origin_rpy=(math.pi / 2.0, 0.0, 0.0),
            material_name="rubber",
        )
        _append_joint(
            robot,
            name=f"{front_name}_joint",
            joint_type="continuous",
            parent="front_steering_knuckle",
            child=front_name,
            axis=(0.0, 1.0, 0.0),
        )

        for name, position in zip((rear_left_name, rear_right_name), wheel_positions[1:]):
            _append_link(
                robot,
                name=name,
                mass_kg=wheel_mass,
                inertia_xyz=_inertia_diagonal(wheel_mass, wheel_size),
                shape="cylinder",
                radius=wheel_radius,
                length=wheel_width,
                geometry_origin_rpy=(math.pi / 2.0, 0.0, 0.0),
                material_name="rubber",
            )
            px, py, pz = (float(value) for value in position)
            _append_joint(
                robot,
                name=f"{name}_joint",
                joint_type="continuous",
                parent=chassis_link,
                child=name,
                origin_xyz=(px, py, pz - chassis_half_height),
                axis=(0.0, 1.0, 0.0),
            )

    def _append_arm(
        self,
        robot: ET.Element,
        *,
        parent_link: str,
        num_dof: int,
        reach_m: float,
        link_lengths: Sequence[float] | None,
        joint_axes: Sequence[Sequence[float]] | None,
        joint_limits: Sequence[Sequence[float]] | None,
        gripper_type: str,
        gripper_stroke_m: float,
        gripper_finger_length_m: float,
        gripper_finger_width_m: float,
        material_name: str,
    ) -> None:
        """Emit a visible serial-kinematic arm chain with optional gripper.

        Joint 1 yaws about +Z; joints 2..N pitch about +Y. At zero angles the
        chain extends along +X for a total of ``reach_m``. The gripper is
        attached to the distal link tip as two prismatic fingers that open
        along ±Y.
        """
        if num_dof <= 0:
            return

        if link_lengths is None:
            per = float(reach_m) / float(num_dof)
            link_lengths = [per] * num_dof
        else:
            link_lengths = [float(v) for v in link_lengths]
            if len(link_lengths) != num_dof:
                raise ValueError(
                    f"arm_link_lengths must contain {num_dof} entries, got {len(link_lengths)}"
                )

        default_axes = [(0.0, 0.0, 1.0)] + [(0.0, 1.0, 0.0)] * (num_dof - 1)
        if joint_axes is None:
            axes = default_axes
        else:
            axes = [tuple(float(v) for v in axis) for axis in joint_axes]
            if len(axes) != num_dof:
                raise ValueError(
                    f"arm_joint_axes must contain {num_dof} entries, got {len(axes)}"
                )

        default_limits = [(-math.pi, math.pi)] + [
            (-math.pi / 2.0, math.pi / 2.0)
        ] * (num_dof - 1)
        if joint_limits is None:
            limits = default_limits
        else:
            limits = [(float(lo), float(hi)) for lo, hi in joint_limits]
            if len(limits) != num_dof:
                raise ValueError(
                    f"arm_joint_limits must contain {num_dof} entries, got {len(limits)}"
                )

        link_radius = 0.04
        link_mass = 0.5
        for i, length in enumerate(link_lengths, start=1):
            link_name = f"arm_link_{i}"
            # Cylinder is default oriented along its local Z. Rotate by π/2 about
            # Y so it lies along +X, and shift the geometry origin to (L/2,0,0)
            # so the link runs from its origin to (L,0,0).
            _append_link(
                robot,
                name=link_name,
                mass_kg=link_mass,
                inertia_xyz=_inertia_diagonal(link_mass, (length, 2 * link_radius, 2 * link_radius)),
                shape="cylinder",
                radius=link_radius,
                length=length,
                inertial_origin_xyz=(length / 2.0, 0.0, 0.0),
                geometry_origin_xyz=(length / 2.0, 0.0, 0.0),
                geometry_origin_rpy=(0.0, math.pi / 2.0, 0.0),
                material_name=material_name,
            )

            if i == 1:
                joint_parent = parent_link
                # The arm_mount cube is 0.1 m; attach at its top face so the base
                # revolute joint is visible sitting on the mount, not inside it.
                joint_origin = (0.0, 0.0, 0.05)
            else:
                joint_parent = f"arm_link_{i - 1}"
                joint_origin = (float(link_lengths[i - 2]), 0.0, 0.0)

            _append_joint(
                robot,
                name=f"arm_joint_{i}",
                joint_type="revolute",
                parent=joint_parent,
                child=link_name,
                origin_xyz=joint_origin,
                axis=axes[i - 1],
                limit=limits[i - 1],
            )

        if gripper_type and gripper_type.lower() != "none":
            self._append_parallel_jaw_gripper(
                robot,
                parent_link=f"arm_link_{num_dof}",
                attach_x=float(link_lengths[-1]),
                stroke_m=gripper_stroke_m,
                finger_length_m=gripper_finger_length_m,
                finger_width_m=gripper_finger_width_m,
                material_name=material_name,
            )

    @staticmethod
    def _append_parallel_jaw_gripper(
        robot: ET.Element,
        *,
        parent_link: str,
        attach_x: float,
        stroke_m: float,
        finger_length_m: float,
        finger_width_m: float,
        material_name: str,
    ) -> None:
        """Emit a two-finger parallel-jaw gripper at the distal link tip."""
        finger_mass = 0.05
        finger_size = (finger_length_m, finger_width_m, finger_width_m)
        half_stroke = stroke_m / 2.0

        specs = (
            ("gripper_left_finger", +1.0, (0.0, +half_stroke), +half_stroke),
            ("gripper_right_finger", -1.0, (-half_stroke, 0.0), -half_stroke),
        )
        for name, sign, limit, _initial in specs:
            _append_link(
                robot,
                name=name,
                mass_kg=finger_mass,
                inertia_xyz=_inertia_diagonal(finger_mass, finger_size),
                shape="box",
                size=finger_size,
                inertial_origin_xyz=(finger_length_m / 2.0, 0.0, 0.0),
                geometry_origin_xyz=(finger_length_m / 2.0, 0.0, 0.0),
                material_name=material_name,
            )
            _append_joint(
                robot,
                name=f"{name}_joint",
                joint_type="prismatic",
                parent=parent_link,
                child=name,
                origin_xyz=(attach_x, 0.0, 0.0),
                axis=(0.0, sign, 0.0),
                limit=(0.0, half_stroke),
            )

    def _validate_physics_sanity(self, root: ET.Element) -> List[str]:
        """Check inertial values and joint limits for obvious physics issues."""
        errors: List[str] = []
        for link in root.findall("link"):
            name = link.get("name", "<unnamed>")
            inertial = link.find("inertial")
            if inertial is None:
                if name != "world":
                    errors.append(f"link {name!r} is missing an inertial block")
                continue

            mass_node = inertial.find("mass")
            inertia_node = inertial.find("inertia")
            if mass_node is None or inertia_node is None:
                errors.append(f"link {name!r} inertial block is incomplete")
                continue

            mass_value = float(mass_node.get("value", "0"))
            if name == "world":
                if mass_value != 0.0:
                    errors.append("world link mass must be 0.0")
                continue

            if mass_value <= 0.0:
                errors.append(f"link {name!r} mass must be > 0, got {mass_value}")

            diagonals = [
                float(inertia_node.get(attr, "0"))
                for attr in ("ixx", "iyy", "izz")
            ]
            if any(value <= 0.0 for value in diagonals):
                errors.append(
                    f"link {name!r} inertia diagonal must be positive, got {diagonals}"
                )

        for joint in root.findall("joint"):
            name = joint.get("name", "<unnamed>")
            joint_type = joint.get("type", "")
            if joint_type in {"revolute", "prismatic"}:
                limit = joint.find("limit")
                if limit is None:
                    errors.append(f"joint {name!r} requires a <limit> block")
                    continue
                lower = float(limit.get("lower", "0"))
                upper = float(limit.get("upper", "0"))
                if lower > upper:
                    errors.append(
                        f"joint {name!r} lower limit {lower} exceeds upper limit {upper}"
                    )

        return errors

    def _validate_genesis_load(self, root: ET.Element) -> List[str]:
        """Check constraints we rely on when handing the URDF to Genesis."""
        errors: List[str] = []
        link_names = [link.get("name", "") for link in root.findall("link")]
        joint_names = [joint.get("name", "") for joint in root.findall("joint")]

        if len(link_names) != len(set(link_names)):
            errors.append("URDF contains duplicate link names")
        if len(joint_names) != len(set(joint_names)):
            errors.append("URDF contains duplicate joint names")

        link_name_set = set(link_names)
        for joint in root.findall("joint"):
            joint_name = joint.get("name", "<unnamed>")
            joint_type = joint.get("type", "")
            if joint_type not in self._SUPPORTED_GENESIS_JOINT_TYPES:
                errors.append(
                    f"joint {joint_name!r} has unsupported Genesis type {joint_type!r}"
                )

            parent = joint.find("parent")
            child = joint.find("child")
            parent_name = parent.get("link", "") if parent is not None else ""
            child_name = child.get("link", "") if child is not None else ""
            if parent_name not in link_name_set:
                errors.append(
                    f"joint {joint_name!r} references missing parent link {parent_name!r}"
                )
            if child_name not in link_name_set:
                errors.append(
                    f"joint {joint_name!r} references missing child link {child_name!r}"
                )

        return errors

    @staticmethod
    def _to_xml(root: ET.Element) -> str:
        """Serialize a URDF XML tree into a deterministic string."""
        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")
