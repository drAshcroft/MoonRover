"""Concrete wheel-terrain model tuned for lunar regolith.

Implements :class:`WheelTerrainModel` using the Pacejka Magic Formula for
longitudinal traction and the Bekker-Wong sinkage law for soft-soil penetration.
The physical constants default to values consistent with Apollo-era regolith
bearing-capacity measurements (e.g. Mitchell et al., Lunar Source Book).

The model is intentionally self-contained: rut state is read from an optional
MPM scalar field; in the absence of a registered MPM field the method returns
zero, which matches the flat-terrain Phase-3 integration path.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from moon_rover.rover.drive.wheel_terrain import WheelTerrainConfig, WheelTerrainModel


# Typical lunar regolith soil parameters (Mitchell et al., Bekker-Wong
# formulation). Values are rounded for production use; the model is not
# intended to be a research-grade terramechanics solver.
_DEFAULT_BEKKER = {
    "cohesion_pa": 1.7e3,          # 1.7 kPa cohesion (dry loose regolith)
    "friction_angle_deg": 35.0,    # internal friction angle
    "sinkage_exponent": 1.0,       # power-law exponent n
    "k_phi": 820.0e3,              # friction modulus kφ in Pa/m^n
    "k_c": 14.0e3,                 # cohesion modulus kc in Pa/m^(n-1)
}

# Pacejka coefficients tuned for a rigid wheel on loose regolith (Ding et al.,
# 2011 — rover wheel traction curves).
_DEFAULT_PACEJKA = {
    "B": 10.0,
    "C": 1.6,
    "D": 0.55,     # peak traction coefficient relative to normal force
    "E": 0.97,
}


def default_lunar_regolith_config() -> WheelTerrainConfig:
    """Return a WheelTerrainConfig tuned for lunar regolith (production defaults)."""
    return WheelTerrainConfig(
        contact_model="hertz_coulomb",
        friction_anisotropic=False,
        bekker_params=dict(_DEFAULT_BEKKER),
        pacejka_params=dict(_DEFAULT_PACEJKA),
    )


class LunarRegolithWheelTerrain(WheelTerrainModel):
    """Wheel-terrain physics implementation for lunar regolith.

    Parameters:
        config: Configuration; if ``None`` the lunar regolith defaults are used.
        wheel_width_m: Effective wheel contact width in metres. Used by the
            Bekker-Wong sinkage law. Default 0.15 m (typical rover tyre).
        wheel_radius_m: Wheel radius in metres (Bekker-Wong b parameter is the
            smaller of wheel_width and wheel_radius). Default 0.3 m.
        rut_sampler: Optional callable that returns rut depth in metres for a
            world position (x, y, z). When ``None`` all positions return 0.
    """

    def __init__(
        self,
        config: Optional[WheelTerrainConfig] = None,
        *,
        wheel_width_m: float = 0.15,
        wheel_radius_m: float = 0.30,
        rut_sampler: Optional[Callable[[NDArray], float]] = None,
    ) -> None:
        self._config = config or default_lunar_regolith_config()
        self._wheel_width_m = float(wheel_width_m)
        self._wheel_radius_m = float(wheel_radius_m)
        self._rut_sampler = rut_sampler

    # ------------------------------------------------------------------
    # ABC
    # ------------------------------------------------------------------

    def compute_slip_ratio(self, wheel_angular_vel: float, ground_velocity: float) -> float:
        wheel_speed = float(wheel_angular_vel) * self._wheel_radius_m
        denom = max(abs(wheel_speed), abs(float(ground_velocity)))
        if denom < 1e-6:
            return 0.0
        slip = (wheel_speed - float(ground_velocity)) / denom
        return float(np.clip(slip, 0.0, 1.0)) if slip >= 0.0 else 0.0

    def compute_traction_force(self, slip_ratio: float, normal_force: float) -> float:
        s = float(np.clip(slip_ratio, 0.0, 1.0))
        fz = max(float(normal_force), 0.0)
        p = self._config.pacejka_params
        B = float(p.get("B", _DEFAULT_PACEJKA["B"]))
        C = float(p.get("C", _DEFAULT_PACEJKA["C"]))
        D = float(p.get("D", _DEFAULT_PACEJKA["D"]))
        E = float(p.get("E", _DEFAULT_PACEJKA["E"]))
        # Pacejka Magic Formula, longitudinal variant:
        #   Fx = D·Fz · sin(C · atan(B·s − E·(B·s − atan(B·s))))
        bs = B * s
        inner = bs - E * (bs - math.atan(bs))
        return D * fz * math.sin(C * math.atan(inner))

    def compute_sinkage(self, wheel_load: float, soil_params: dict[str, float]) -> float:
        params = {**self._config.bekker_params, **(soil_params or {})}
        n = float(params.get("sinkage_exponent", 1.0))
        k_phi = float(params.get("k_phi", _DEFAULT_BEKKER["k_phi"]))
        k_c = float(params.get("k_c", _DEFAULT_BEKKER["k_c"]))

        fz = max(float(wheel_load), 0.0)
        b = min(self._wheel_width_m, self._wheel_radius_m)  # Bekker smaller dim
        # Bekker-Wong sinkage law:
        #   p = (kc/b + k_phi) · z^n,    p = Fz / (b · l), l ≈ sqrt(2 R z)
        # Combine to z^(n + 0.5) = Fz / (b · sqrt(2R) · (kc/b + k_phi))
        coeff = (k_c / max(b, 1e-6)) + k_phi
        if coeff < 1e-6:
            return 0.0
        denom = b * math.sqrt(max(2.0 * self._wheel_radius_m, 1e-6)) * coeff
        if denom < 1e-6:
            return 0.0
        # Solve for sinkage z:
        #   z ^ (n + 0.5) = Fz / denom
        exponent = 1.0 / max(n + 0.5, 1e-3)
        z = (fz / denom) ** exponent if fz > 0.0 else 0.0
        # Cap sinkage to physically plausible wheel penetration.
        return float(min(z, 0.5 * self._wheel_radius_m))

    def compute_rut_state(self, position: NDArray) -> float:
        if self._rut_sampler is None:
            return 0.0
        pos = np.asarray(position, dtype=np.float32).flatten()
        if pos.size < 2:
            return 0.0
        try:
            return float(self._rut_sampler(pos[:3] if pos.size >= 3 else np.append(pos, 0.0)))
        except Exception:
            return 0.0

    def compute_cable_drag_effect(self, cable_tension: float, normal_force: float) -> float:
        tension = max(float(cable_tension), 0.0)
        fz = max(float(normal_force), 1e-3)
        # Empirical: traction fraction degrades from 1.0 towards 0.0 as cable
        # tension grows relative to wheel load. 0.4 friction coefficient tracks
        # the terrain_friction used by CableComposer.
        ratio = 0.4 * tension / fz
        factor = 1.0 - ratio
        return float(np.clip(factor, 0.0, 1.0))
