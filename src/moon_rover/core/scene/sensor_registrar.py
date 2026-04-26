"""SensorRegistrar — registers sensors with the physics engine during CONSTRUCTION phase.

Maps sensors.yaml config to concrete physics engine sensor registrations for each rover.
Populates the sensor_handles dict in each SceneRoverSpec so downstream subsystems
(LiDAR scanner, telemetry, power monitor) can retrieve engine-specific handles without
knowing engine internals.

Supported sensor types and their registration paths:
  - lidar       → engine.register_raycaster()
  - imu         → body-fixed attachment recorded as link name
  - force_torque → gripper joint attachment; engine handle stored
  - sun_sensor  → body-fixed attachment recorded as link name
  - gps_beacon  → pseudo-GNSS receiver; engine handle if engine supports it
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from moon_rover.core.scene.specs import SceneRoverSpec
    from moon_rover.core.physics.engine import PhysicsEngine


# Sentinel for engine registrations that return no handle (e.g., link-attached sensors)
_LINK_ATTACHED = "link_attached"

# Known sensor keys in sensors.yaml, mapped to their handle key in sensor_handles dict
_SENSOR_KEYS = {
    "lidar": "lidar",
    "imu": "imu",
    "force_torque_sensor": "force_torque",
    "sun_sensor": "sun_sensor",
    "gps_beacon": "gps_beacon",
    # Camera sensors — registered as config-only handles (engine camera API pending)
    "stereo_camera_front": "stereo_camera_front",
    "navcam": "navcam",
}


class SensorRegistrar:
    """Registers all sensors from sensors.yaml with the physics engine.

    Iterates over enabled sensors and, for each rover:
      1. LiDAR — calls engine.register_raycaster() (if the method exists) and
         stores the returned handle under 'lidar'.
      2. IMU — records the rover body entity name so the simulation loop can
         call engine.get_body_pose() / get_body_velocity() by name.
      3. Force-torque — records attachment at arm gripper entity name.
      4. Sun sensor — records rover body entity name for body-fixed queries.
      5. GPS beacon — calls engine.register_sensor() if available, else records
         rover entity name for pseudo-GNSS polling.

    Unknown sensor types raise ValueError so misconfigured sensors.yaml is
    caught at construction time rather than during simulation.
    """

    def register(
        self,
        sensors_cfg: Dict[str, Any],
        rover_specs: List["SceneRoverSpec"],
        engine: "PhysicsEngine",
    ) -> None:
        """Register all enabled sensors for all rovers.

        Mutates each SceneRoverSpec.sensor_handles in-place.

        Parameters:
            sensors_cfg: Parsed sensors.yaml dict. Pass {} if sensors config
                         was not loaded (all sensors will be skipped).
            rover_specs: List of SceneRoverSpec to populate. Modified in-place.
            engine: PhysicsEngine in CONSTRUCTION phase.

        Raises:
            ValueError: If sensors.yaml contains an unrecognised top-level sensor
                        key (not in the known set and not a metadata/non-sensor key).
        """
        if not sensors_cfg:
            return

        # Keys in sensors.yaml that are not sensor definitions
        _meta_keys = {
            "sensor_suite", "sensor_fusion", "recording", "debug",
        }

        # Validate that all top-level keys are known
        unknown = (
            set(sensors_cfg.keys()) - set(_SENSOR_KEYS.keys()) - _meta_keys
        )
        if unknown:
            raise ValueError(
                f"sensors.yaml contains unrecognised sensor keys: {sorted(unknown)}. "
                f"Known sensor types: {sorted(_SENSOR_KEYS.keys())}."
            )

        for rover_spec in rover_specs:
            rover_id = rover_spec.rover_id
            handles: Dict[str, Any] = {}

            # ── LiDAR ─────────────────────────────────────────────────────────
            lidar_cfg = sensors_cfg.get("lidar", {})
            if lidar_cfg.get("enabled", False):
                fov_cfg = lidar_cfg.get("fov", {})
                range_cfg = lidar_cfg.get("range", {})
                elev_lower = float(fov_cfg.get("vertical_lower_deg", -25.33))
                elev_upper = float(fov_cfg.get("vertical_upper_deg", 15.67))
                h_fov_deg = float(fov_cfg.get("horizontal_deg", 360.0))
                num_channels = int(lidar_cfg.get("num_channels", 32))
                max_range_m = float(range_cfg.get("max_range_m", 100.0))

                # h_resolution_deg = horizontal FOV / horizontal scan points.
                # For a 360° scanner we typically use the angular resolution.
                h_resolution_deg = float(
                    lidar_cfg.get("accuracy", {}).get("angular_resolution_deg", 0.2)
                )

                # Pattern config in the format GenesisPhysicsEngine.register_raycaster expects
                pattern_config = {
                    "num_channels": num_channels,
                    "elevation_range_deg": (elev_lower, elev_upper),
                    "h_resolution_deg": h_resolution_deg,
                }

                register_raycaster = getattr(engine, "register_raycaster", None)
                if register_raycaster is not None:
                    handles["lidar"] = register_raycaster(
                        name=f"{rover_id}_lidar",
                        link_entity=rover_id,
                        link_idx=0,
                        pattern_config=pattern_config,
                        max_range=max_range_m,
                    )
                else:
                    # Engine does not expose register_raycaster; store config
                    # for the sensor subsystem to handle directly.
                    handles["lidar"] = {
                        "entity_name": rover_id,
                        "pattern_config": pattern_config,
                        "max_range_m": max_range_m,
                    }

            # ── IMU ───────────────────────────────────────────────────────────
            imu_cfg = sensors_cfg.get("imu", {})
            if imu_cfg.get("enabled", False):
                # IMU is body-fixed; the simulation loop queries via entity name.
                handles["imu"] = {
                    "entity_name": rover_id,
                    "mount_position": imu_cfg.get("mount_position", [0.0, 0.0, 0.0]),
                    "frequency_hz": imu_cfg.get("frequency_hz", 100.0),
                    "sensor_id": imu_cfg.get("sensor_id", f"{rover_id}_imu"),
                }

            # ── Force-torque ──────────────────────────────────────────────────
            ft_cfg = sensors_cfg.get("force_torque_sensor", {})
            if ft_cfg.get("enabled", False):
                # Attached at arm gripper (last link of arm chain).
                gripper_entity = f"{rover_id}_gripper"
                register_sensor = getattr(engine, "register_sensor", None)
                if register_sensor is not None:
                    handles["force_torque"] = register_sensor(
                        name=f"{rover_id}_force_torque",
                        entity=gripper_entity,
                        sensor_type="force_torque",
                        config={
                            "frequency_hz": ft_cfg.get("frequency_hz", 50.0),
                            "full_scale_n": ft_cfg.get("force", {}).get("full_scale_n", 500.0),
                            "full_scale_nm": ft_cfg.get("torque", {}).get("full_scale_nm", 50.0),
                        },
                    )
                else:
                    handles["force_torque"] = {
                        "entity_name": gripper_entity,
                        "frequency_hz": ft_cfg.get("frequency_hz", 50.0),
                        "sensor_id": ft_cfg.get("sensor_id", f"{rover_id}_ft"),
                    }

            # ── Sun sensor ────────────────────────────────────────────────────
            sun_cfg = sensors_cfg.get("sun_sensor", {})
            if sun_cfg.get("enabled", False):
                handles["sun_sensor"] = {
                    "entity_name": rover_id,
                    "mount_position": sun_cfg.get("mount_position", [0.0, 0.0, 0.8]),
                    "fov_deg": sun_cfg.get("sensor", {}).get("fov_deg", 60.0),
                    "frequency_hz": sun_cfg.get("frequency_hz", 1.0),
                    "sensor_id": sun_cfg.get("sensor_id", f"{rover_id}_sun"),
                }

            # ── GPS beacon receiver ───────────────────────────────────────────
            gps_cfg = sensors_cfg.get("gps_beacon", {})
            if gps_cfg.get("enabled", False):
                register_sensor = getattr(engine, "register_sensor", None)
                if register_sensor is not None:
                    handles["gps_beacon"] = register_sensor(
                        name=f"{rover_id}_gps_beacon",
                        entity=rover_id,
                        sensor_type="gps_beacon",
                        config={
                            "frequency_hz": gps_cfg.get("frequency_hz", 1.0),
                            "max_range_m": gps_cfg.get("range", {}).get("max_range_m", 1000.0),
                            "mount_position": gps_cfg.get("mount_position", [0.0, 0.0, 0.6]),
                        },
                    )
                else:
                    handles["gps_beacon"] = {
                        "entity_name": rover_id,
                        "frequency_hz": gps_cfg.get("frequency_hz", 1.0),
                        "sensor_id": gps_cfg.get("sensor_id", f"{rover_id}_gps"),
                    }

            # ── Camera sensors (config-only; engine camera API pending) ───────
            for cam_key in ("stereo_camera_front", "navcam"):
                cam_cfg = sensors_cfg.get(cam_key, {})
                if cam_cfg.get("enabled", False):
                    handles[cam_key] = {
                        "entity_name": rover_id,
                        "mount_position": cam_cfg.get("mount_position", [0.0, 0.0, 0.3]),
                        "frequency_hz": cam_cfg.get("frequency_hz", 30.0),
                        "sensor_id": cam_cfg.get("sensor_id", f"{rover_id}_{cam_key}"),
                    }

            rover_spec.sensor_handles.update(handles)
