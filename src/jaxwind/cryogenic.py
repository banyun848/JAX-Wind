"""Liquid-nitrogen parcels and conservative FV carrier-phase exchange.

The carrier fields use the finite-volume solver's native ``(z, y, x)`` cell
layout.  Parcels are a fixed-capacity JAX pytree, injected deterministically,
sampled and deposited with clipped cloud-in-cell weights, and advanced with
analytic drag plus the repository's Ranz--Marshall LN2 evaporation model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import numpy as np
import jax.numpy as jnp

from jaxwind.domain.grid import Grid
from jaxwind.physics.cryogenic import (
    CryogenicMicrophysicsConfig,
    advance_nitrogen_droplet,
)

from .discretization import _cells_to_faces, cell_velocity
from .state import StaggeredVelocity


@dataclass(frozen=True, slots=True)
class LN2Jet:
    """Physical and numerical definition of a flashing LN2 nozzle."""

    x: float
    y: float
    z: float
    radius: float
    speed: float
    mass_flow_rate: float
    vapor_quality: float
    initial_temperature: float = 77.34
    initial_diameter: float = 150.0e-6
    minimum_diameter: float = 50.0e-6
    maximum_diameter: float = 300.0e-6
    rosin_rammler_spread: float = 3.0
    parcels_per_step: int = 8
    maximum_parcels: int = 16_384
    substeps: int = 4
    ramp_time: float = 0.05
    random_seed: int = 2024
    liquid_density: float = 806.11
    gravity_x: float = 0.0
    gravity_y: float = 0.0
    gravity_z: float = -9.81
    cone_half_angle_degrees: float = 0.0
    edge_speed_ratio: float = 1.0
    edge_diameter_ratio: float = 1.0
    profile_radius_m: tuple[float, ...] = ()
    profile_axial_velocity_m_s: tuple[float, ...] = ()
    profile_radial_velocity_m_s: tuple[float, ...] = ()
    profile_d10_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        positive = (
            self.radius,
            self.speed,
            self.mass_flow_rate,
            self.initial_temperature,
            self.initial_diameter,
            self.minimum_diameter,
            self.maximum_diameter,
            self.rosin_rammler_spread,
            self.liquid_density,
            self.edge_speed_ratio,
            self.edge_diameter_ratio,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("LN2 jet dimensions, flow, and material values must be positive")
        if not all(
            math.isfinite(value)
            for value in (self.gravity_x, self.gravity_y, self.gravity_z)
        ):
            raise ValueError("LN2 gravity components must be finite")
        if not 0.0 <= self.vapor_quality < 1.0:
            raise ValueError("LN2 vapor quality must lie in [0, 1)")
        if self.minimum_diameter >= self.maximum_diameter:
            raise ValueError("minimum droplet diameter must be below the maximum")
        if self.parcels_per_step <= 0 or self.maximum_parcels < self.parcels_per_step:
            raise ValueError("the parcel capacity must accommodate one injection")
        if self.substeps <= 0 or self.ramp_time < 0.0:
            raise ValueError("parcel substeps must be positive and ramp time nonnegative")
        if not 0.0 <= self.cone_half_angle_degrees < 90.0:
            raise ValueError("cone half-angle must lie in [0, 90) degrees")
        profiles = (
            self.profile_radius_m,
            self.profile_axial_velocity_m_s,
            self.profile_radial_velocity_m_s,
            self.profile_d10_m,
        )
        if any(profiles):
            if not all(
                len(values) == len(self.profile_radius_m)
                for values in profiles
            ):
                raise ValueError(
                    "empirical LN2 source profiles must have equal lengths"
                )
            if len(self.profile_radius_m) < 2:
                raise ValueError(
                    "empirical LN2 source profiles need at least two radii"
                )
            if self.profile_radius_m[0] != 0.0 or any(
                right <= left
                for left, right in zip(
                    self.profile_radius_m[:-1], self.profile_radius_m[1:]
                )
            ):
                raise ValueError(
                    "empirical source radii must start at zero and increase"
                )
            if self.profile_radius_m[-1] > self.radius:
                raise ValueError(
                    "empirical source radii must lie inside the jet radius"
                )
            if not all(
                math.isfinite(value)
                for values in profiles
                for value in values
            ):
                raise ValueError("empirical LN2 source profiles must be finite")
            if any(value <= 0.0 for value in self.profile_d10_m):
                raise ValueError("empirical D10 values must be positive")

    @property
    def vapor_mass_flow_rate(self) -> float:
        return self.mass_flow_rate * self.vapor_quality

    @property
    def liquid_mass_flow_rate(self) -> float:
        return self.mass_flow_rate * (1.0 - self.vapor_quality)


class LN2InletControl(NamedTuple):
    """Dynamic continuous controls for differentiable LN2 injection."""

    initial_diameter: jax.Array
    speed_scale: jax.Array
    edge_speed_ratio: jax.Array
    edge_diameter_ratio: jax.Array


class LN2Parcels(NamedTuple):
    x: jax.Array
    y: jax.Array
    z: jax.Array
    u: jax.Array
    v: jax.Array
    w: jax.Array
    mass: jax.Array
    diameter: jax.Array
    temperature: jax.Array
    multiplicity: jax.Array
    parcel_id: jax.Array
    active: jax.Array


class ParcelExchange(NamedTuple):
    parcels: LN2Parcels
    acceleration_x: jax.Array
    acceleration_y: jax.Array
    acceleration_z: jax.Array
    nitrogen_source: jax.Array
    temperature_source: jax.Array
    gas_mass_source: jax.Array
    volume_expansion: jax.Array
    evaporated_mass_rate: jax.Array


def initial_ln2_parcels(jet: LN2Jet, dtype=jnp.float32) -> LN2Parcels:
    """Return an empty fixed-capacity parcel buffer."""

    count = jet.maximum_parcels
    zeros = jnp.zeros((count,), dtype)
    return LN2Parcels(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        jnp.full((count,), jet.initial_temperature, dtype),
        zeros,
        jnp.arange(count, dtype=jnp.uint32),
        jnp.zeros((count,), dtype=jnp.bool_),
    )


def _ramp(time: jax.Array, jet: LN2Jet) -> jax.Array:
    if jet.ramp_time == 0.0:
        return jnp.asarray(1.0, time.dtype)
    phase = jnp.clip(time / jet.ramp_time, 0.0, 1.0)
    return 0.5 * (1.0 - jnp.cos(jnp.pi * phase))


def inject_ln2_parcels(
    parcels: LN2Parcels,
    step: jax.Array,
    dt: float,
    jet: LN2Jet,
    control: LN2InletControl | None = None,
) -> LN2Parcels:
    """Fill inactive slots with a differentiably controlled deterministic batch."""

    count = jet.parcels_per_step
    slots = jnp.argsort(parcels.active.astype(jnp.int32))[:count]
    valid = jnp.arange(count) < jnp.sum(~parcels.active)
    key = jax.random.fold_in(jax.random.PRNGKey(jet.random_seed), step)
    radial_key, angle_key, diameter_key = jax.random.split(key, 3)
    dtype = parcels.x.dtype
    if control is None:
        control = LN2InletControl(
            jnp.asarray(jet.initial_diameter, dtype),
            jnp.asarray(1.0, dtype),
            jnp.asarray(jet.edge_speed_ratio, dtype),
            jnp.asarray(jet.edge_diameter_ratio, dtype),
        )
    radial = jet.radius * jnp.sqrt(jax.random.uniform(radial_key, (count,), dtype=dtype))
    angle = 2.0 * jnp.pi * jax.random.uniform(angle_key, (count,), dtype=dtype)
    uniform = jax.random.uniform(diameter_key, (count,), dtype=dtype)
    spread = jnp.asarray(jet.rosin_rammler_spread, dtype)
    radial_fraction = radial / jnp.asarray(jet.radius, dtype)
    radial_shape = radial_fraction**2
    if jet.profile_radius_m:
        profile_radius = jnp.asarray(jet.profile_radius_m, dtype)
        # The configured empirical diameter is D10.  A Rosin--Rammler scale
        # lambda has arithmetic mean lambda*Gamma(1 + 1/k) before truncation.
        mean_diameter = jnp.interp(
            radial, profile_radius, jnp.asarray(jet.profile_d10_m, dtype)
        )
        scale = mean_diameter / jnp.asarray(
            math.gamma(1.0 + 1.0 / jet.rosin_rammler_spread), dtype
        )
    else:
        scale = jnp.asarray(control.initial_diameter, dtype) * (
            1.0
            + (jnp.asarray(control.edge_diameter_ratio, dtype) - 1.0)
            * radial_shape
        )
    dmin = jnp.asarray(jet.minimum_diameter, dtype)
    dmax = jnp.asarray(jet.maximum_diameter, dtype)
    cdf_min = 1.0 - jnp.exp(-((dmin / scale) ** spread))
    cdf_max = 1.0 - jnp.exp(-((dmax / scale) ** spread))
    probability = cdf_min + uniform * (cdf_max - cdf_min)
    diameter = scale * (-jnp.log1p(-probability)) ** (1.0 / spread)
    density = jnp.asarray(jet.liquid_density, dtype)
    mass = (jnp.pi / 6.0) * density * diameter**3
    time = (step.astype(dtype) + 0.5) * jnp.asarray(dt, dtype)
    injected_mass = jet.liquid_mass_flow_rate * jnp.asarray(dt, dtype) * _ramp(time, jet)
    multiplicity = injected_mass / jnp.maximum(jnp.sum(mass), jnp.finfo(dtype).tiny)

    def assign(field, values):
        old = field[slots]
        return field.at[slots].set(jnp.where(valid, values, old))

    if jet.profile_radius_m:
        profile_radius = jnp.asarray(jet.profile_radius_m, dtype)
        axial_velocity = jnp.interp(
            radial,
            profile_radius,
            jnp.asarray(jet.profile_axial_velocity_m_s, dtype),
        )
        radial_velocity = jnp.interp(
            radial,
            profile_radius,
            jnp.asarray(jet.profile_radial_velocity_m_s, dtype),
        )
    else:
        local_speed = (
            jnp.asarray(jet.speed, dtype)
            * jnp.asarray(control.speed_scale, dtype)
            * (
                1.0
                + (jnp.asarray(control.edge_speed_ratio, dtype) - 1.0)
                * radial_shape
            )
        )
        inclination = (
            jnp.deg2rad(jnp.asarray(jet.cone_half_angle_degrees, dtype))
            * radial_fraction
        )
        axial_velocity = local_speed * jnp.cos(inclination)
        radial_velocity = local_speed * jnp.sin(inclination)

    return LN2Parcels(
        assign(parcels.x, jnp.full((count,), jet.x, dtype)),
        assign(parcels.y, jet.y + radial * jnp.cos(angle)),
        assign(parcels.z, jet.z + radial * jnp.sin(angle)),
        assign(parcels.u, axial_velocity),
        assign(parcels.v, radial_velocity * jnp.cos(angle)),
        assign(parcels.w, radial_velocity * jnp.sin(angle)),
        assign(parcels.mass, mass),
        assign(parcels.diameter, diameter),
        assign(parcels.temperature, jnp.full((count,), jet.initial_temperature, dtype)),
        assign(parcels.multiplicity, jnp.full((count,), multiplicity, dtype)),
        assign(
            parcels.parcel_id,
            step.astype(jnp.uint32) * jnp.uint32(count) + jnp.arange(count, dtype=jnp.uint32),
        ),
        parcels.active.at[slots].set(parcels.active[slots] | valid),
    )


def _cic_coordinates(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    grid: Grid,
) -> tuple[jax.Array, ...]:
    """Clipped physical-space CIC coordinates for a mapped open box."""

    def coordinate(values, centers):
        centers = jnp.asarray(centers, values.dtype)
        insertion = jnp.searchsorted(centers, values, side="right")
        lower = jnp.clip(insertion - 1, 0, centers.size - 1)
        upper = jnp.clip(insertion, 0, centers.size - 1)
        distance = centers[upper] - centers[lower]
        fraction = jnp.where(
            upper > lower,
            (values - centers[lower]) / jnp.maximum(distance, 1.0e-30),
            0.0,
        )
        return lower.astype(jnp.int32), upper.astype(jnp.int32), jnp.clip(
            fraction, 0.0, 1.0
        )

    ix0, ix1, fx = coordinate(x, grid.x_centers)
    iy0, iy1, fy = coordinate(y, grid.y_centers)
    iz0, iz1, fz = coordinate(z, grid.z_centers)
    return ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz


def _cic_sample(field: jax.Array, coordinates: tuple[jax.Array, ...]) -> jax.Array:
    ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz = coordinates
    result = jnp.zeros_like(fx, dtype=field.dtype)
    for ix, wx in ((ix0, 1.0 - fx), (ix1, fx)):
        for iy, wy in ((iy0, 1.0 - fy), (iy1, fy)):
            for iz, wz in ((iz0, 1.0 - fz), (iz1, fz)):
                result = result + wx * wy * wz * field[iz, iy, ix]
    return result


def _cic_sample_many(
    fields: jax.Array,
    coordinates: tuple[jax.Array, ...],
) -> jax.Array:
    """Sample a leading batch of cell fields with shared CIC coordinates."""

    ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz = coordinates
    result = jnp.zeros((fields.shape[0], fx.shape[0]), fields.dtype)
    for ix, wx in ((ix0, 1.0 - fx), (ix1, fx)):
        for iy, wy in ((iy0, 1.0 - fy), (iy1, fy)):
            for iz, wz in ((iz0, 1.0 - fz), (iz1, fz)):
                result = result + fields[:, iz, iy, ix] * (wx * wy * wz)
    return result


def _cic_deposit(
    values: jax.Array,
    coordinates: tuple[jax.Array, ...],
    shape: tuple[int, int, int],
) -> jax.Array:
    ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz = coordinates
    result = jnp.zeros(shape, values.dtype)
    for ix, wx in ((ix0, 1.0 - fx), (ix1, fx)):
        for iy, wy in ((iy0, 1.0 - fy), (iy1, fy)):
            for iz, wz in ((iz0, 1.0 - fz), (iz1, fz)):
                result = result.at[iz, iy, ix].add(values * wx * wy * wz)
    return result


def _cic_deposit_many(
    values: jax.Array,
    coordinates: tuple[jax.Array, ...],
    shape: tuple[int, int, int],
) -> jax.Array:
    """Deposit a leading batch of parcel values in one scatter operation."""

    ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz = coordinates
    result = jnp.zeros((values.shape[0], *shape), values.dtype)
    corners = (
        (ix0, iy0, iz0, (1.0 - fx) * (1.0 - fy) * (1.0 - fz)),
        (ix0, iy0, iz1, (1.0 - fx) * (1.0 - fy) * fz),
        (ix0, iy1, iz0, (1.0 - fx) * fy * (1.0 - fz)),
        (ix0, iy1, iz1, (1.0 - fx) * fy * fz),
        (ix1, iy0, iz0, fx * (1.0 - fy) * (1.0 - fz)),
        (ix1, iy0, iz1, fx * (1.0 - fy) * fz),
        (ix1, iy1, iz0, fx * fy * (1.0 - fz)),
        (ix1, iy1, iz1, fx * fy * fz),
    )
    ix = jnp.concatenate(tuple(corner[0] for corner in corners))
    iy = jnp.concatenate(tuple(corner[1] for corner in corners))
    iz = jnp.concatenate(tuple(corner[2] for corner in corners))
    updates = jnp.concatenate(
        tuple(values * corner[3][None, :] for corner in corners), axis=1
    )
    field = jnp.arange(values.shape[0], dtype=jnp.int32)[:, None]
    return result.at[
        field,
        iz[None, :],
        iy[None, :],
        ix[None, :],
    ].add(updates)


def _cell_to_velocity_faces(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    grid: Grid,
) -> StaggeredVelocity:
    return StaggeredVelocity(
        _cells_to_faces(x, grid, 2, periodic=False, boundary="copy"),
        _cells_to_faces(y, grid, 1, periodic=False, boundary="zero"),
        _cells_to_faces(z, grid, 0, periodic=False, boundary="zero"),
    )


def advance_ln2_parcels(
    parcels: LN2Parcels,
    velocity: StaggeredVelocity,
    temperature: jax.Array,
    grid: Grid,
    step: jax.Array,
    dt: float,
    jet: LN2Jet,
    microphysics: CryogenicMicrophysicsConfig,
    gas_density: jax.Array | None = None,
    inlet_control: LN2InletControl | None = None,
) -> ParcelExchange:
    """Inject and advance parcels, returning conservative carrier sources."""

    parcels = inject_ln2_parcels(
        parcels, step, dt, jet, inlet_control
    )
    gas_u, gas_v, gas_w = cell_velocity(velocity)
    shape = temperature.shape
    parcel_sources = jnp.zeros((5, parcels.x.shape[0]), temperature.dtype)
    initial_x, initial_y, initial_z = parcels.x, parcels.y, parcels.z
    carrier_density = (
        jnp.full(shape, microphysics.dry_air_density, temperature.dtype)
        if gas_density is None
        else jnp.asarray(gas_density, temperature.dtype)
    )
    if carrier_density.shape != shape:
        raise ValueError("gas density must be cell centred")
    substep = dt / jet.substeps

    carrier_fields = jnp.stack(
        (gas_u, gas_v, gas_w, temperature, carrier_density)
    )

    def subcycle(_, state):
        current, accumulated_sources = state
        coordinates = _cic_coordinates(current.x, current.y, current.z, grid)
        sampled = _cic_sample_many(carrier_fields, coordinates)
        sampled_u, sampled_v, sampled_w, sampled_temperature, sampled_density = (
            sampled
        )
        rel_u = sampled_u - current.u
        rel_v = sampled_v - current.v
        rel_w = sampled_w - current.w
        relative_speed = jnp.sqrt(
            rel_u**2 + rel_v**2 + rel_w**2 + 1.0e-20
        )
        diameter = jnp.maximum(current.diameter, 1.0e-8)
        reynolds = (
            sampled_density
            * relative_speed
            * diameter
            / microphysics.air_dynamic_viscosity
        )
        correction = 1.0 + 0.15 * reynolds**0.687 + (
            0.42 * reynolds / 24.0
        ) / (1.0 + 42_500.0 / jnp.maximum(reynolds, 1.0e-12) ** 1.16)
        drag_rate = (
            18.0
            * microphysics.air_dynamic_viscosity
            * correction
            / (microphysics.liquid_nitrogen_density * diameter**2)
        )
        active = current.active.astype(temperature.dtype)
        relaxation = active * (-jnp.expm1(-drag_rate * substep))
        gravity_factor = (
            1.0
            - sampled_density
            / microphysics.liquid_nitrogen_density
        )
        gravity_x = jet.gravity_x * gravity_factor
        gravity_y = jet.gravity_y * gravity_factor
        gravity_z = jet.gravity_z * gravity_factor
        response = jnp.where(
            drag_rate > 0.0, relaxation / drag_rate, substep * active
        )
        new_u = (
            current.u + relaxation * rel_u + gravity_x * response
        )
        new_v = (
            current.v + relaxation * rel_v + gravity_y * response
        )
        new_w = (
            current.w + relaxation * rel_w + gravity_z * response
        )
        evaluation_mass = jnp.where(
            current.active,
            current.mass,
            jnp.asarray(1.0e-18, temperature.dtype),
        )
        evaluation_temperature = jnp.where(
            current.active,
            current.temperature,
            jnp.asarray(
                microphysics.nitrogen_boiling_temperature,
                temperature.dtype,
            ),
        )
        update = advance_nitrogen_droplet(
            evaluation_mass,
            diameter,
            evaluation_temperature,
            sampled_temperature,
            relative_speed,
            substep,
            microphysics,
        )
        physical_mass = current.mass * current.multiplicity * active
        parcel_impulse_x = -physical_mass * (
            new_u - current.u - gravity_x * response
        )
        parcel_impulse_y = -physical_mass * (
            new_v - current.v - gravity_y * response
        )
        parcel_impulse_z = -physical_mass * (
            new_w - current.w - gravity_z * response
        )
        evaporated = update.evaporated_mass * current.multiplicity * active
        heat = update.gas_energy_loss * current.multiplicity * active
        # Evaporated mass joins the carrier at the parcel velocity. The
        # relative part belongs in the velocity equation; the carrier part is
        # already represented by the conservative gas-mass flux.
        parcel_impulse_x = parcel_impulse_x + evaporated * (new_u - sampled_u)
        parcel_impulse_y = parcel_impulse_y + evaporated * (new_v - sampled_v)
        parcel_impulse_z = parcel_impulse_z + evaporated * (new_w - sampled_w)
        accumulated_sources = accumulated_sources + jnp.stack(
            (
                parcel_impulse_x,
                parcel_impulse_y,
                parcel_impulse_z,
                evaporated,
                heat,
            )
        )
        next_x = current.x + substep * new_u
        next_y = current.y + substep * new_v
        next_z = current.z + substep * new_w
        alive = (
            current.active
            & (update.mass > 1.0e-18)
            & (next_x >= 0.0)
            & (next_x <= grid.lx)
            & (next_y >= 0.0)
            & (next_y <= grid.ly)
            & (next_z >= 0.0)
            & (next_z <= grid.lz)
        )
        next_parcels = LN2Parcels(
            next_x,
            next_y,
            next_z,
            new_u,
            new_v,
            new_w,
            jnp.where(alive, update.mass, 0.0),
            jnp.where(alive, update.diameter, 0.0),
            jnp.where(
                alive,
                update.temperature,
                microphysics.nitrogen_boiling_temperature,
            ),
            current.multiplicity,
            current.parcel_id,
            alive,
        )
        return next_parcels, accumulated_sources

    parcels, parcel_sources = jax.lax.fori_loop(
        0,
        jet.substeps,
        subcycle,
        (parcels, parcel_sources),
    )
    # The carrier fields are frozen while the parcels subcycle. Deposit the
    # accumulated conservative exchange once at the path midpoint, reducing
    # four full-grid CIC deposits to one without biasing it upstream.
    deposition_coordinates = _cic_coordinates(
        0.5 * (initial_x + parcels.x),
        0.5 * (initial_y + parcels.y),
        0.5 * (initial_z + parcels.z),
        grid,
    )
    sources = _cic_deposit_many(
        parcel_sources,
        deposition_coordinates,
        shape,
    )
    impulse_x, impulse_y, impulse_z, vapor_mass, energy = sources
    cell_volume = jnp.asarray(grid.cell_volumes, temperature.dtype)
    rate_scale = 1.0 / (cell_volume * dt)
    acceleration = _cell_to_velocity_faces(
        impulse_x * rate_scale / carrier_density,
        impulse_y * rate_scale / carrier_density,
        impulse_z * rate_scale / carrier_density,
        grid,
    )
    evaporation_rate = vapor_mass * rate_scale
    nitrogen_source = evaporation_rate / carrier_density
    temperature_source = -energy * rate_scale / (
        carrier_density * microphysics.dry_air_heat_capacity
    )
    nitrogen_density = microphysics.pressure / (
        microphysics.nitrogen_gas_constant * jnp.maximum(temperature, 50.0)
    )
    return ParcelExchange(
        parcels,
        acceleration.x,
        acceleration.y,
        acceleration.z,
        nitrogen_source,
        temperature_source,
        evaporation_rate,
        evaporation_rate / nitrogen_density,
        jnp.sum(vapor_mass) / dt,
    )


def vapor_nozzle_source(
    grid: Grid,
    jet: LN2Jet,
    dtype=jnp.float32,
) -> jax.Array:
    """Cell-normalized under-resolved source kernel at the physical nozzle."""

    def local_width(faces: np.ndarray, coordinate: float) -> float:
        index = int(np.clip(np.searchsorted(faces, coordinate) - 1, 0, len(faces) - 2))
        return float(faces[index + 1] - faces[index])

    x = jnp.asarray(grid.x_centers, dtype)
    y = jnp.asarray(grid.y_centers, dtype)
    z = jnp.asarray(grid.z_centers, dtype)
    dx = local_width(grid.x_faces, jet.x)
    dy = local_width(grid.y_faces, jet.y)
    dz = local_width(grid.z_faces, jet.z)
    # An empirical radial profile defines a measured source plane, whereas the
    # analytic nozzle model represents an unresolved finite source volume.
    sigma_x = dx if jet.profile_radius_m else max(dx, 2.0 * jet.radius)
    sigma_y = max(0.5 * dy, jet.radius)
    sigma_z = max(0.5 * dz, jet.radius)
    shape = jnp.exp(
        -0.5
        * (
            ((x[None, None, :] - jet.x) / sigma_x) ** 2
            + ((y[None, :, None] - jet.y) / sigma_y) ** 2
            + ((z[:, None, None] - jet.z) / sigma_z) ** 2
        )
    )
    # Convert a sampled continuous shape to a fraction of the injected mass in
    # each unequal cell. The caller divides this mass fraction by local volume.
    kernel = shape * jnp.asarray(grid.cell_volumes, dtype)
    return kernel / jnp.sum(kernel)


__all__ = [
    "LN2Jet",
    "LN2InletControl",
    "LN2Parcels",
    "ParcelExchange",
    "advance_ln2_parcels",
    "initial_ln2_parcels",
    "inject_ln2_parcels",
    "vapor_nozzle_source",
]
