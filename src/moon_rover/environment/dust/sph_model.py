"""System 2.5: Vacuum and Dust — SPH-based Dust Simulation.

This module provides a smoothed particle hydrodynamics (SPH) model for
simulating lunar dust behavior, including dust settling on solar panels,
accumulation in sensors, and wind/motion effects on dust clouds.

The dust simulation models:
- Dust particle transport via wheel action and rover motion
- Progressive deposition on solar panels (reducing efficiency)
- Lens/camera contamination (increasing noise/blur)
- Particle density field for visualization and effects

Classes:
    DustConfig (dataclass): Dust simulation configuration
    DustSimulation (ABC): Abstract interface for dust physics

Typical Usage:
    config = DustConfig(wheel_threshold_angular_vel=50, ...)
    dust_sim = DustSimulation(config)
    dust_sim.step(dt=0.01, wheel_velocities=[10, 10, 10, 10])
    efficiency = dust_sim.get_solar_panel_efficiency_factor()  # 0.8 (80%)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class DustConfig:
    """Configuration parameters for dust simulation.

    Controls dust generation, transport, and effects on rover sensors/panels.

    Attributes:
        wheel_threshold_angular_vel (float): Wheel angular velocity threshold (rad/s)
                                            above which dust is kicked up.
                                            Typical: 20-50 rad/s.
        panel_deposition_rate (float): Rate at which dust settles on solar panels.
                                      Units: fraction of panel area per second per unit density.
                                      Typical: 0.001-0.01 1/s.
        lens_noise_factor (float): Noise multiplication factor for camera/lens due to dust.
                                  1.0 = clean lens, >1.0 = degraded.
                                  Typical: 1.0-3.0 for dust coverage 0%-50%.
    """

    wheel_threshold_angular_vel: float = 30.0
    """Wheel angular velocity threshold (rad/s) for dust generation.

    When wheel angular velocity exceeds this threshold, dust is kicked up
    into the air and becomes part of the simulation. Below this threshold,
    no dust generation occurs (wheels simply roll cleanly).
    """

    panel_deposition_rate: float = 0.005
    """Rate of dust deposition on solar panels (1/seconds).

    Controls how quickly dust settles onto solar panels, reducing their
    efficiency. Higher values = faster panel degradation.
    Typical range: 0.001 - 0.01
    """

    lens_noise_factor: float = 1.0
    """Camera/lens noise multiplier due to dust contamination (unitless).

    Scales the noise standard deviation added to camera images as dust
    accumulates. 1.0 = clean (baseline noise), 2.0 = 2x noise.
    Typical range: 1.0 - 3.0 for realistic dust levels.
    """


class DustSimulation(ABC):
    """Abstract interface for SPH-based dust simulation.

    Manages dust particle generation, transport, settling, and effects on
    rover systems (panels, cameras, sensors). Uses SPH for particle dynamics
    without explicit particle tracking.

    Initialization:
        Call initialize(config) before stepping the simulation.

    Typical Loop:
        dust_sim = DustSimulation(...)
        dust_sim.initialize(config)
        for dt in timesteps:
            dust_sim.step(dt, wheel_velocities)
            panel_eff = dust_sim.get_solar_panel_efficiency_factor()
            camera_noise = dust_sim.get_camera_noise_factor()

    Abstract Methods:
        initialize: Set up dust simulation with configuration
        step: Advance simulation by one timestep
        get_solar_panel_efficiency_factor: Query panel efficiency degradation
        get_camera_noise_factor: Query camera noise due to dust
        get_sph_density_at: Query dust density field at a location
    """

    @abstractmethod
    def initialize(self, config: DustConfig) -> None:
        """Initialize dust simulation with configuration parameters.

        Sets up internal SPH particle field, boundary conditions, and
        sensor contamination models. Must be called before step().

        Args:
            config (DustConfig): Dust simulation configuration.
                                Contains wheel threshold, deposition rates, etc.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            config = DustConfig(wheel_threshold_angular_vel=30.0)
            dust_sim.initialize(config)
        """
        raise NotImplementedError("initialize implementation pending")

    @abstractmethod
    def step(self, dt: float, wheel_velocities: list) -> None:
        """Advance dust simulation by one timestep.

        Updates SPH particle positions, densities, and sensor contamination.
        Dust generation based on wheel velocities, settling based on gravity
        and SPH kernels.

        Args:
            dt (float): Simulation timestep in seconds (typically 0.01 s).
            wheel_velocities (list): Angular velocities of 4 wheels [rad/s].
                                    Order: [front_left, front_right, rear_left, rear_right]
                                    Typical range: [-50, +50] rad/s.

        Physics Model:
            - Dust generation: If |wheel_vel| > threshold, dust particles
              created in air near wheel contact.
            - Dust transport: Particles advected via SPH velocity field,
              influenced by rover motion and brownian motion.
            - Dust settling: Particles settle under gravity with terminal
              velocity. Heavier particles settle faster.
            - Panel deposition: Particles in contact with solar panel area
              are removed and accumulate as contaminant layer.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            dust_sim.step(0.01, wheel_velocities=[10, 10, 10, 10])
            # Internal state updated: particle positions, densities,
            # contamination layers.
        """
        raise NotImplementedError("step implementation pending")

    @abstractmethod
    def get_solar_panel_efficiency_factor(self) -> float:
        """Get current solar panel efficiency due to dust coverage.

        Returns:
            float: Efficiency multiplier [0.0, 1.0].
                  1.0 = clean panels (100% efficiency)
                  0.5 = 50% dust coverage (50% efficiency)
                  0.0 = completely covered (0% efficiency)

        Physics Model:
            Efficiency = 1.0 - (dust_coverage_fraction * dust_absorption)
            where dust_coverage_fraction is the fraction of panel area covered,
            and dust_absorption ~0.8 (absorbs 80% of light when dense).

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            efficiency = dust_sim.get_solar_panel_efficiency_factor()
            power_output_w = nominal_power_w * efficiency
            if efficiency < 0.5:
                print("Warning: Low power due to dust")
        """
        raise NotImplementedError("get_solar_panel_efficiency_factor implementation pending")

    @abstractmethod
    def get_camera_noise_factor(self) -> float:
        """Get camera/lens noise multiplication factor due to dust.

        Returns:
            float: Noise scaling factor [1.0, inf].
                  1.0 = clean lens (baseline noise only)
                  2.0 = 2x noise standard deviation (degraded image quality)
                  Higher values = more severe lens contamination.

        Physics Model:
            Noise_std = baseline_std * (1.0 + dust_density_on_lens)
            Dust density estimated from local SPH density field near camera.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            noise_factor = dust_sim.get_camera_noise_factor()
            image_noise_std = 2.0 * noise_factor  # baseline std=2.0
            # Apply Gaussian noise to camera image with this std
        """
        raise NotImplementedError("get_camera_noise_factor implementation pending")

    @abstractmethod
    def get_sph_density_at(self, position: np.ndarray) -> float:
        """Query SPH particle density field at a specific location.

        Returns the dust density at the given position in the SPH field.
        Useful for visualization, sensor contamination modeling, and effects.

        Args:
            position (np.ndarray): Query position [x, y, z] in world frame (meters).

        Returns:
            float: Dust particle density at the position.
                  Units: number density (particles/m^3) or kg/m^3 (implementation-dependent).
                  Typical range: [0, 100] particles/m^3 for lunar dust.

        Raises:
            NotImplementedError: Implementation pending.

        Example:
            pos = np.array([1.0, 2.0, 0.5])
            density = dust_sim.get_sph_density_at(pos)
            if density > 10.0:
                print(f"High dust at {pos}: {density:.1f} particles/m³")
        """
        raise NotImplementedError("get_sph_density_at implementation pending")
