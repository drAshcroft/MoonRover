"""Focused tests for the concrete Genesis URDF builder."""

from __future__ import annotations

import io
import os
from xml.etree import ElementTree as ET

import pytest
import yaml
import yourdfpy

from moon_rover.core.assets.genesis_urdf_builder import GenesisURDFBuilder
from moon_rover.core.assets.urdf_builder import URDFValidationStage
from moon_rover.core.scene.rover_composer import RoverComposer


_CONFIGS = os.path.join(os.path.dirname(__file__), "..", "..", "configs")
_ROVER_YAML = os.path.join(_CONFIGS, "rover.yaml")


@pytest.fixture(scope="module")
def rover_cfg() -> dict:
    with open(_ROVER_YAML, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _rover_urdf_config(rover_cfg: dict, profile_key: str) -> dict:
    profile = rover_cfg["profiles"][profile_key]
    return RoverComposer._profile_to_urdf_config(
        rover_id=f"{profile_key}_rover",
        profile=profile,
        rover_cfg=rover_cfg,
    )


@pytest.mark.parametrize(
    ("profile_key", "stage"),
    [
        ("two_wheel_diff", URDFValidationStage.XML_SCHEMA),
        ("two_wheel_diff", URDFValidationStage.PHYSICS_SANITY),
        ("two_wheel_diff", URDFValidationStage.GENESIS_LOAD),
        ("three_wheel_tricycle", URDFValidationStage.XML_SCHEMA),
        ("three_wheel_tricycle", URDFValidationStage.PHYSICS_SANITY),
        ("three_wheel_tricycle", URDFValidationStage.GENESIS_LOAD),
        ("four_wheel_skid", URDFValidationStage.XML_SCHEMA),
        ("four_wheel_skid", URDFValidationStage.PHYSICS_SANITY),
        ("four_wheel_skid", URDFValidationStage.GENESIS_LOAD),
    ],
)
def test_all_drive_profiles_validate_cleanly(rover_cfg: dict, profile_key, stage):
    builder = GenesisURDFBuilder()
    urdf_xml = builder.build_rover(_rover_urdf_config(rover_cfg, profile_key))
    assert builder.validate(urdf_xml, stage) == []
    yourdfpy.URDF.load(io.StringIO(urdf_xml))


@pytest.mark.parametrize(
    ("profile_key", "expected_wheels"),
    [
        ("two_wheel_diff", 2),
        ("three_wheel_tricycle", 3),
        ("four_wheel_skid", 4),
    ],
)
def test_rover_urdf_contains_expected_number_of_wheel_links(
    rover_cfg: dict,
    profile_key,
    expected_wheels,
):
    builder = GenesisURDFBuilder()
    urdf_xml = builder.build_rover(_rover_urdf_config(rover_cfg, profile_key))
    root = ET.fromstring(urdf_xml)
    wheel_links = [
        link.get("name")
        for link in root.findall("link")
        if (link.get("name") or "").endswith("_wheel")
    ]
    assert len(wheel_links) == expected_wheels


def test_rover_urdf_includes_required_mount_links(rover_cfg: dict):
    builder = GenesisURDFBuilder()
    urdf_xml = builder.build_rover(_rover_urdf_config(rover_cfg, "four_wheel_skid"))
    root = ET.fromstring(urdf_xml)
    link_names = {link.get("name") for link in root.findall("link")}
    assert "arm_mount" in link_names
    assert "cable_spool_mount" in link_names
    assert "sensor_mount_front" in link_names
    assert "sensor_mount_top" in link_names


def test_build_antenna_includes_mast_and_dish_with_expected_mass():
    builder = GenesisURDFBuilder()
    urdf_xml = builder.build_antenna()
    root = ET.fromstring(urdf_xml)
    mass_by_link = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        assert inertial is not None
        mass = inertial.find("mass")
        assert mass is not None
        mass_by_link[link.get("name")] = float(mass.get("value"))

    assert mass_by_link["mast_link"] == pytest.approx(1.0)
    assert mass_by_link["dish_link"] == pytest.approx(0.8)


def test_build_moonbase_has_fixed_joint_to_world():
    builder = GenesisURDFBuilder()
    urdf_xml = builder.build_moonbase()
    root = ET.fromstring(urdf_xml)

    world_joint = root.find("./joint[@name='world_to_moonbase']")
    assert world_joint is not None
    assert world_joint.get("type") == "fixed"
    assert world_joint.find("parent").get("link") == "world"
    assert world_joint.find("child").get("link") == "moonbase_base"

    world_link = root.find("./link[@name='world']")
    assert world_link is not None
    world_mass = world_link.find("./inertial/mass")
    assert world_mass is not None
    assert float(world_mass.get("value")) == pytest.approx(0.0)


def test_missing_mass_kg_raises_value_error(rover_cfg: dict):
    config = _rover_urdf_config(rover_cfg, "four_wheel_skid")
    config.pop("mass_kg")
    with pytest.raises(ValueError, match="mass_kg"):
        GenesisURDFBuilder().build_rover(config)


# ---------------------------------------------------------------------------
# Arm chain emission
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "profile_key",
    ["two_wheel_diff", "three_wheel_tricycle", "four_wheel_skid"],
)
def test_rover_urdf_emits_full_arm_chain(rover_cfg: dict, profile_key):
    builder = GenesisURDFBuilder()
    urdf_xml = builder.build_rover(_rover_urdf_config(rover_cfg, profile_key))
    root = ET.fromstring(urdf_xml)

    link_names = {link.get("name") for link in root.findall("link")}
    for i in range(1, 5):
        assert f"arm_link_{i}" in link_names, (
            f"profile {profile_key!r} missing arm_link_{i}"
        )
    assert "gripper_left_finger" in link_names
    assert "gripper_right_finger" in link_names


def test_arm_joints_are_revolute_with_limits(rover_cfg: dict):
    builder = GenesisURDFBuilder()
    urdf_xml = builder.build_rover(_rover_urdf_config(rover_cfg, "two_wheel_diff"))
    root = ET.fromstring(urdf_xml)

    for i in range(1, 5):
        joint = root.find(f"./joint[@name='arm_joint_{i}']")
        assert joint is not None, f"missing arm_joint_{i}"
        assert joint.get("type") == "revolute"
        assert joint.find("limit") is not None


def test_gripper_fingers_are_prismatic_with_limits(rover_cfg: dict):
    builder = GenesisURDFBuilder()
    urdf_xml = builder.build_rover(_rover_urdf_config(rover_cfg, "two_wheel_diff"))
    root = ET.fromstring(urdf_xml)

    for name in ("gripper_left_finger_joint", "gripper_right_finger_joint"):
        joint = root.find(f"./joint[@name='{name}']")
        assert joint is not None, f"missing gripper joint {name}"
        assert joint.get("type") == "prismatic"
        assert joint.find("limit") is not None


def test_arm_chain_is_anchored_to_arm_mount(rover_cfg: dict):
    builder = GenesisURDFBuilder()
    urdf_xml = builder.build_rover(_rover_urdf_config(rover_cfg, "four_wheel_skid"))
    root = ET.fromstring(urdf_xml)

    j1 = root.find("./joint[@name='arm_joint_1']")
    assert j1 is not None
    assert j1.find("parent").get("link") == "arm_mount"
    # Each downstream arm joint parents the previous arm link.
    for i in range(2, 5):
        ji = root.find(f"./joint[@name='arm_joint_{i}']")
        assert ji is not None
        assert ji.find("parent").get("link") == f"arm_link_{i - 1}"


def test_arm_dof_zero_skips_arm_emission():
    builder = GenesisURDFBuilder()
    config = {
        "rover_id": "no_arm_rover",
        "mass_kg": 60.0,
        "wheel_radius": 0.3,
        "wheel_count": 2,
        "wheel_positions": [[0.0, 0.3, 0.0], [0.0, -0.3, 0.0]],
        "dimensions": [0.3, 0.4, 0.8],
        "inertia": [0.6, 1.5, 1.2],
        "com_offset": [0.0, 0.0, 0.3],
        "arm_base_position": [0.0, 0.0, 0.4],
        "arm_dof": 0,
    }
    urdf_xml = builder.build_rover(config)
    root = ET.fromstring(urdf_xml)
    link_names = {link.get("name") for link in root.findall("link")}
    assert not any(name.startswith("arm_link_") for name in link_names)
    assert not any(name.startswith("gripper_") for name in link_names)
    # arm_mount is always emitted — it's a sensor/attachment anchor.
    assert "arm_mount" in link_names
