"""Pytest configuration and fixtures for Moon Rover testing.

This module provides shared fixtures for unit tests, integration tests, and
validation scenarios. Fixtures create mock/stub versions of the Genesis engine,
sample terrain, and default rover/mission configurations.

Fixtures:
    genesis_engine: Mock Genesis physics engine for GPU-free testing
    sample_terrain: Pre-generated flat terrain mesh
    sample_rover_config: Default 4-wheel rover configuration
    sample_mission_config: Minimal 1-antenna mission setup

All fixtures return placeholder objects suitable for unit testing without
requiring GPU hardware or full physics simulation.
"""

import pytest


@pytest.fixture
def genesis_engine():
    """Provide a mock Genesis physics engine for testing without GPU.

    The Genesis physics engine is computationally expensive to instantiate
    and requires GPU hardware. This fixture provides a minimal mock that
    allows tests to run without GPU availability while still exposing the
    expected Genesis API surface.

    Attributes:
        is_mock (bool): True, indicating this is a mock engine
        scene_created (bool): False initially, set to True after create_scene()
        bodies (dict): Mapping of body names to Body objects (empty initially)
        gravity (list): Gravitational acceleration [0, 0, -1.62] m/s^2 (lunar)

    Methods (stubbed):
        create_scene() -> Scene: Creates a new physics scene
        close(): Shuts down the engine

    Yields:
        Mock: Genesis-like engine object for test usage

    Example:
        @pytest.mark.skipif(not has_genesis, reason="Genesis not installed")
        def test_rover_dynamics(genesis_engine):
            scene = genesis_engine.create_scene()
            assert scene is not None
    """
    class MockGenesisEngine:
        """Minimal mock of Genesis physics engine."""
        def __init__(self):
            self.is_mock = True
            self.scene_created = False
            self.bodies = {}
            self.gravity = [0, 0, -1.62]  # Lunar gravity

        def create_scene(self):
            """Create and return a mock physics scene."""
            self.scene_created = True
            return MockScene()

        def close(self):
            """Clean up engine resources."""
            pass

    class MockScene:
        """Minimal mock of Genesis physics scene."""
        def __init__(self):
            self.bodies = {}
            self.time = 0.0
            self.dt = 0.01

        def add_body(self, name, **kwargs):
            """Add a rigid body to the scene."""
            self.bodies[name] = {"name": name, "config": kwargs}
            return name

        def step(self):
            """Advance physics simulation by dt."""
            self.time += self.dt

    return MockGenesisEngine()


@pytest.fixture
def sample_terrain():
    """Provide a pre-generated flat terrain mesh for unit tests.

    Terrain Specification:
        - Type: Flat lunar regolith (coefficient of friction ~0.7)
        - Dimensions: 200m x 200m
        - Height: Z=0 (baseline)
        - Roughness: Minimal (RMS height variation < 1 cm)
        - Material: Lunar soil properties (density ~1600 kg/m^3)
        - Friction model: Coulomb with mu_s=0.7, mu_k=0.6

    This fixture is suitable for testing basic locomotion, path planning,
    and control without terrain complexity.

    Yields:
        dict: Terrain configuration with vertices, faces, and material properties

    Example:
        def test_basic_locomotion(sample_terrain):
            env = MoonRoverEnv(terrain=sample_terrain)
            assert env.terrain_size == 200.0
    """
    return {
        "type": "flat",
        "size_x": 200.0,
        "size_y": 200.0,
        "height": 0.0,
        "friction_static": 0.7,
        "friction_kinetic": 0.6,
        "material": "lunar_regolith",
        "vertices": [],  # Placeholder; would contain mesh data
        "faces": [],     # Placeholder; would contain face indices
    }


@pytest.fixture
def sample_rover_config():
    """Provide a default 4-wheel rover configuration for testing.

    Rover Specification (MoonRover-A class):
        - Wheels: 4 independently actuated (FL, FR, RL, RR)
        - Motor: DC motor, 50 W peak, 200 rpm max
        - Thruster: 6-DOF attitude control (3-axis thrust system)
        - Mass: 20 kg
        - Wheel radius: 0.15 m
        - Wheelbase: 0.8 m (front-to-rear)
        - Track width: 0.6 m (side-to-side)
        - Max speed: ~2 m/s on flat terrain
        - Payload: Antenna (0.5 kg) + cable reel (2 kg)
        - Battery: 120 Wh nominal (TBD)
        - Sensors:
          * LiDAR: 180° FOV, 30 m range, 180 beams @ 10 Hz
          * IMU: 9-DOF (accel, gyro, magnetometer)
          * Encoder: Wheel odometry
          * Camera: 640x480, 30 FPS (optional)

    Yields:
        dict: Rover configuration with mechanical, electrical, and sensor specs

    Example:
        def test_rover_kinematics(sample_rover_config):
            rover = MoonRover(config=sample_rover_config)
            assert rover.mass == 20.0  # kg
    """
    return {
        "name": "MoonRover-A",
        "mass_kg": 20.0,
        "wheels": {
            "count": 4,
            "radius_m": 0.15,
            "motor_max_rpm": 200,
            "motor_max_w": 50,
        },
        "chassis": {
            "wheelbase_m": 0.8,
            "track_width_m": 0.6,
        },
        "thruster": {
            "dof": 6,
            "max_force_n": [10, 10, 10],  # X, Y, Z axes
            "max_torque_nm": [5, 5, 5],   # Roll, Pitch, Yaw
        },
        "payload": {
            "antenna_mass_kg": 0.5,
            "cable_reel_mass_kg": 2.0,
            "cable_length_m": 50.0,
        },
        "battery": {
            "capacity_wh": 120.0,
            "nominal_voltage_v": 12.0,
        },
        "sensors": {
            "lidar": {
                "fov_deg": 180,
                "range_max_m": 30.0,
                "beam_count": 180,
                "frequency_hz": 10,
            },
            "imu": {
                "type": "9dof",
                "frequency_hz": 100,
            },
            "encoder": {
                "frequency_hz": 50,
            },
            "camera": {
                "resolution": [640, 480],
                "frequency_hz": 30,
            },
        },
    }


@pytest.fixture
def sample_mission_config():
    """Provide a minimal 1-antenna mission configuration for testing.

    Mission Specification:
        - Start: [0, 0, 0] (base station origin)
        - Duration: 300 seconds nominal
        - Antennas: 1 target at [50, 0, 0] (50 m east)
        - Deployment strategy: Direct waypoint following
        - Safety: No energy margin (minimal battery buffer)

    This configuration is suitable for quick unit tests. For more complex
    multi-antenna scenarios, extend this config or use dedicated fixtures.

    Yields:
        dict: Mission configuration with start, goal, and objectives

    Example:
        def test_single_antenna_deployment(sample_mission_config):
            mission = Mission(config=sample_mission_config)
            assert len(mission.antennas) == 1
    """
    return {
        "mission_id": "test_001",
        "start_position": [0.0, 0.0, 0.0],
        "start_orientation": [0.0, 0.0, 0.0, 1.0],
        "base_station_position": [0.0, 0.0, 0.0],
        "antennas": [
            {
                "id": "antenna_01",
                "target_position": [50.0, 0.0, 0.0],
                "surface_normal": [0.0, 0.0, 1.0],
                "priority": 1,
                "deployment_time_s": 10.0,
            }
        ],
        "max_duration_s": 300.0,
        "energy_margin_percent": 10.0,
        "speed_profile": "conservative",  # May be "aggressive" or "balanced"
    }
