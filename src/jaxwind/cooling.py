"""Conservative grid-scale cooling sources for far-wake LES.

The physical spray nozzle and droplets are deliberately not resolved here.
Instead, their net thermodynamic effect is deposited through a smooth kernel
whose integral is the specified cooling power. This keeps the mesh and time
step tied to the turbine wake rather than millimetre-scale injection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

import jax.numpy as jnp

from jaxwind.domain.grid import Grid

from .state import StaggeredVelocity


@dataclass(frozen=True, slots=True)
class SubgridCooling:
    """A Gaussian temperature sink calibrated by its total cooling power."""

    cooling_power_w: float
    air_density_kg_m3: float
    air_heat_capacity_j_kg_k: float
    center_m: tuple[float, float, float]
    standard_deviation_m: tuple[float, float, float]
    ramp_time_s: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.cooling_power_w,
            self.air_density_kg_m3,
            self.air_heat_capacity_j_kg_k,
            *self.standard_deviation_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError(
                "cooling power, air properties, and widths must be positive"
            )
        if not all(math.isfinite(value) for value in self.center_m):
            raise ValueError("the cooling-source centre must be finite")
        if not math.isfinite(self.ramp_time_s) or self.ramp_time_s < 0.0:
            raise ValueError("the cooling-source ramp time must be nonnegative")

    @property
    def integrated_temperature_sink_k_m3_s(self) -> float:
        """Volume-integrated temperature tendency implied by the power."""

        return -self.cooling_power_w / (
            self.air_density_kg_m3 * self.air_heat_capacity_j_kg_k
        )


def build_subgrid_cooling_source(
    grid: Grid,
    model: SubgridCooling,
    *,
    dtype: str = "float32",
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return S_T(x,t) with exact discrete integral -Q/(rho cp)."""

    x0, y0, z0 = model.center_m
    if not (
        0.0 <= x0 <= grid.lx
        and 0.0 <= y0 <= grid.ly
        and 0.0 <= z0 <= grid.lz
    ):
        raise ValueError("the cooling-source centre lies outside the mesh")
    sx, sy, sz = model.standard_deviation_m
    x = jnp.asarray(grid.x_centers, dtype)
    y = jnp.asarray(grid.y_centers, dtype)
    z = jnp.asarray(grid.z_centers, dtype)
    squared_radius = (
        ((x[None, None, :] - x0) / sx) ** 2
        + ((y[None, :, None] - y0) / sy) ** 2
        + ((z[:, None, None] - z0) / sz) ** 2
    )
    unnormalised = jnp.exp(-0.5 * squared_radius)
    volumes = jnp.asarray(grid.cell_volumes, dtype)
    kernel = unnormalised / jnp.sum(unnormalised * volumes)
    base = jnp.asarray(
        model.integrated_temperature_sink_k_m3_s, dtype
    ) * kernel

    def source(time: jnp.ndarray) -> jnp.ndarray:
        if model.ramp_time_s == 0.0:
            return base
        phase = jnp.clip(
            jnp.asarray(time, base.dtype) / model.ramp_time_s,
            0.0,
            1.0,
        )
        ramp = 0.5 * (1.0 - jnp.cos(jnp.pi * phase))
        return ramp * base

    return source


@dataclass(frozen=True, slots=True)
class SubgridSpray:
    """Conservative unresolved conical LN2 spray for far-wake LES.

    The millimetre nozzle and individual droplets are not resolved.  The
    source instead deposits the specified integral cooling power and axial
    momentum over a grid-scale cone downstream of the physical nozzle.
    """

    cooling_power_w: float
    mass_flow_rate_kg_s: float
    injection_speed_m_s: float
    air_density_kg_m3: float
    air_heat_capacity_j_kg_k: float
    nozzle_m: tuple[float, float, float]
    nozzle_diameter_m: float
    cone_half_angle_degrees: float
    axial_standard_deviation_m: float
    minimum_radial_standard_deviation_m: tuple[float, float]
    ramp_time_s: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.cooling_power_w,
            self.mass_flow_rate_kg_s,
            self.injection_speed_m_s,
            self.air_density_kg_m3,
            self.air_heat_capacity_j_kg_k,
            self.nozzle_diameter_m,
            self.axial_standard_deviation_m,
            *self.minimum_radial_standard_deviation_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("spray properties and numerical widths must be positive")
        if not all(math.isfinite(value) for value in self.nozzle_m):
            raise ValueError("spray nozzle coordinates must be finite")
        if not (
            math.isfinite(self.cone_half_angle_degrees)
            and 0.0 <= self.cone_half_angle_degrees < 90.0
        ):
            raise ValueError("spray cone half-angle must lie in [0, 90) degrees")
        if not math.isfinite(self.ramp_time_s) or self.ramp_time_s < 0.0:
            raise ValueError("spray ramp time must be nonnegative")

    @property
    def integrated_temperature_sink_k_m3_s(self) -> float:
        return -self.cooling_power_w / (
            self.air_density_kg_m3 * self.air_heat_capacity_j_kg_k
        )

    @property
    def integrated_axial_acceleration_m4_s2(self) -> float:
        """Carrier-volume integral of the injected axial acceleration."""

        return (
            self.mass_flow_rate_kg_s
            * self.injection_speed_m_s
            / self.air_density_kg_m3
        )


def _open_x_faces(values: jnp.ndarray, grid: Grid) -> jnp.ndarray:
    """Width-weight cell forcing onto the open-domain x faces."""

    widths = jnp.asarray(grid.x_widths, dtype=values.dtype)
    total = widths[:-1] + widths[1:]
    interior = (
        values[..., :-1] * widths[:-1][None, None, :]
        + values[..., 1:] * widths[1:][None, None, :]
    ) / total[None, None, :]
    return jnp.concatenate(
        (values[..., :1], interior, values[..., -1:]), axis=2
    )


def build_subgrid_spray_sources(
    grid: Grid,
    model: SubgridSpray,
    *,
    dtype: str = "float32",
) -> tuple[
    Callable[[jnp.ndarray], jnp.ndarray],
    Callable[[StaggeredVelocity, jnp.ndarray], StaggeredVelocity],
]:
    """Return conservative temperature and momentum sources for a spray cone."""

    x0, y0, z0 = model.nozzle_m
    if not (
        0.0 <= x0 < grid.lx
        and 0.0 <= y0 <= grid.ly
        and 0.0 <= z0 <= grid.lz
    ):
        raise ValueError("the spray nozzle lies outside the mesh")
    x = jnp.asarray(grid.x_centers, dtype)
    y = jnp.asarray(grid.y_centers, dtype)
    z = jnp.asarray(grid.z_centers, dtype)
    downstream = x - jnp.asarray(x0, dtype)
    positive_distance = jnp.maximum(downstream, 0.0)
    expansion = jnp.tan(
        jnp.deg2rad(jnp.asarray(model.cone_half_angle_degrees, dtype))
    )
    physical_radius = (
        0.5 * jnp.asarray(model.nozzle_diameter_m, dtype)
        + positive_distance * expansion
    )
    minimum_y, minimum_z = model.minimum_radial_standard_deviation_m
    sigma_y = jnp.maximum(physical_radius, jnp.asarray(minimum_y, dtype))
    sigma_z = jnp.maximum(physical_radius, jnp.asarray(minimum_z, dtype))
    exponent = (
        (downstream[None, None, :] / model.axial_standard_deviation_m) ** 2
        + ((y[None, :, None] - y0) / sigma_y[None, None, :]) ** 2
        + ((z[:, None, None] - z0) / sigma_z[None, None, :]) ** 2
    )
    unnormalised = jnp.where(
        downstream[None, None, :] >= 0.0,
        jnp.exp(-0.5 * exponent),
        0.0,
    )
    volumes = jnp.asarray(grid.cell_volumes, dtype)
    kernel = unnormalised / jnp.sum(unnormalised * volumes)
    temperature_base = (
        jnp.asarray(model.integrated_temperature_sink_k_m3_s, dtype)
        * kernel
    )
    acceleration_base = (
        jnp.asarray(model.integrated_axial_acceleration_m4_s2, dtype)
        * kernel
    )

    def ramp(time: jnp.ndarray) -> jnp.ndarray:
        if model.ramp_time_s == 0.0:
            return jnp.asarray(1.0, temperature_base.dtype)
        phase = jnp.clip(
            jnp.asarray(time, temperature_base.dtype) / model.ramp_time_s,
            0.0,
            1.0,
        )
        return 0.5 * (1.0 - jnp.cos(jnp.pi * phase))

    def temperature_source(time: jnp.ndarray) -> jnp.ndarray:
        return ramp(time) * temperature_base

    def momentum_source(
        velocity: StaggeredVelocity,
        time: jnp.ndarray,
    ) -> StaggeredVelocity:
        axial = _open_x_faces(ramp(time) * acceleration_base, grid)
        return StaggeredVelocity(
            axial,
            jnp.zeros_like(velocity.y),
            jnp.zeros_like(velocity.z),
        )

    return temperature_source, momentum_source


__all__ = [
    "SubgridCooling",
    "SubgridSpray",
    "build_subgrid_cooling_source",
    "build_subgrid_spray_sources",
]
