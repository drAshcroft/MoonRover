"""Moon Rover - Lunar Exploration Vehicle Simulation.

A comprehensive simulation framework for a multi-system lunar rover including:
- Drive systems (differential, tricycle, skid-steer)
- Power and thermal management
- Robotic manipulator arm with gripper
- Multi-sensor perception (LiDAR, stereo cameras, IMU, beacon network)
- Tethered cable deployment system
- Deployable antenna for extended communication
- Moonbase logistics and charging infrastructure

Package Structure:
  rover.drive: Drive kinematics and wheel-terrain interaction
  rover.power: Power generation, storage, and distribution
  rover.manipulator: Arm kinematics and end-effector control
  sensors.lidar: 3D range scanning
  sensors.stereo_camera: Depth imaging and obstacle detection
  sensors.imu: Inertial measurement and wheel encoders
  sensors.gps_beacon: Pseudo-GNSS localization network
  sensors.force_torque: Wrist force-torque feedback
  sensors.sun_sensor: Solar tracking
  cable: Tethered cable deployment system
  antenna: Deployable high-gain antenna
  moonbase: Base facility and logistics
"""

__version__ = "0.1.0"

__all__ = [
    "rover",
    "sensors",
    "cable",
    "antenna",
    "moonbase",
]
