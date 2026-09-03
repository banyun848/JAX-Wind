from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from applications.fv_ln2_jet.run import _conservative_nonnegative

from jaxwind.domain import AnalyticalGrid, TanhMapping, UniformGrid
from jaxwind.fv import (
    IdealGasMixture,
    PassiveScalar,
    StaggeredVelocity,
    build_pressure_poisson,
    conservative_specific_tendency,
    continuity_residual,
    face_density,
    mass_flux,
    project_low_mach,
)


def random_velocity(grid: UniformGrid, seed: int = 0) -> StaggeredVelocity:
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    cells = (grid.nz, grid.ny, grid.nx)
    vertical = jax.random.normal(keys[2], (grid.nz + 1, grid.ny, grid.nx))
    return StaggeredVelocity(
        jax.random.normal(keys[0], cells),
        jax.random.normal(keys[1], cells),
        vertical.at[0].set(0.0).at[-1].set(0.0),
    )


def test_ideal_gas_mixture_supports_species_inventory_and_base_pressure() -> None:
    temperature = jnp.linspace(180.0, 310.0, 24).reshape(2, 3, 4)
    pressure = jnp.linspace(90_000.0, 75_000.0, 2)[:, None, None]
    model = IdealGasMixture(
        pressure=pressure,
        background_gas_constant=287.05,
        species_gas_constants=(296.8, 461.5),
    )
    nitrogen = jnp.full_like(temperature, 0.3)
    water = jnp.full_like(temperature, 0.01)
    density = model.density(temperature, (nitrogen, water))
    recovered = model.density_from_partial_densities(
        temperature, (density * nitrogen, density * water)
    )
    pure_air = model.density(temperature, (jnp.zeros_like(temperature),) * 2)
    pure_nitrogen = model.density(
        temperature, (jnp.ones_like(temperature), jnp.zeros_like(temperature))
    )

    assert density.shape == temperature.shape
    assert jnp.allclose(recovered, density, rtol=1.0e-12, atol=1.0e-12)
    assert bool(jnp.all(pure_nitrogen < pure_air))
    assert bool(jnp.all(density[0] > density[1]))


def test_variable_density_projection_enforces_discrete_continuity() -> None:
    grids = (
        UniformGrid(8, 8, 6, 1.0, 1.0, 0.75),
        AnalyticalGrid(
            8,
            8,
            6,
            1.0,
            1.0,
            0.75,
            TanhMapping(1.2, focus=0.0),
            TanhMapping(1.0),
            TanhMapping(0.8),
        ),
    )
    for grid in grids:
        velocity = random_velocity(grid, 12)
        previous = jnp.ones((grid.nz, grid.ny, grid.nx))
        x = jnp.asarray(grid.x_centers)
        change = 0.02 * jnp.sin(2.0 * jnp.pi * x / grid.lx)
        # Remove the volume-weighted constant mode on a mapped mesh.
        change = change - jnp.sum(change * jnp.asarray(grid.x_widths)) / grid.lx
        density = previous + jnp.broadcast_to(change, previous.shape)
        source = 0.3 * jnp.broadcast_to(change, previous.shape)
        faces = face_density(density, velocity, grid)
        backend = "fft" if grid.is_uniform else "gmg"
        corrected, _ = project_low_mach(
            velocity,
            previous,
            density,
            build_pressure_poisson(grid, backend=backend),
            0.02,
            mass_source=source,
        )
        before = continuity_residual(
            velocity, previous, density, grid, 0.02, source
        )
        after = continuity_residual(
            corrected, previous, density, grid, 0.02, source
        )

        assert faces.x.shape == velocity.x.shape
        assert faces.y.shape == velocity.y.shape
        assert faces.z.shape == velocity.z.shape
        assert float(jnp.linalg.norm(after)) < 1.0e-8 * float(
            jnp.linalg.norm(before)
        )


def test_conservative_nonnegative_preserves_inventory() -> None:
    field = jnp.asarray([[-1.0, 3.0]])
    volumes = jnp.asarray([[1.0, 2.0]])

    corrected = _conservative_nonnegative(field, volumes)

    assert bool(jnp.allclose(corrected, jnp.asarray([[0.0, 2.5]])))
    assert float(jnp.sum(corrected * volumes)) == 5.0


def test_specific_transport_preserves_periodic_conservative_inventory() -> None:
    grids = (
        UniformGrid(12, 10, 8, 1.5, 1.25, 1.0),
        AnalyticalGrid(
            12,
            10,
            8,
            1.5,
            1.25,
            1.0,
            TanhMapping(1.2, focus=0.0),
            TanhMapping(1.0),
            TanhMapping(0.8),
        ),
    )
    for grid in grids:
        velocity = random_velocity(grid, 19)
        density = 1.0 + 0.1 * jax.random.uniform(
            jax.random.PRNGKey(20), (grid.nz, grid.ny, grid.nx)
        )
        scalar = jax.random.uniform(
            jax.random.PRNGKey(21), (grid.nz, grid.ny, grid.nx)
        )
        tendency = conservative_specific_tendency(
            scalar,
            density,
            velocity,
            grid,
            PassiveScalar(diffusivity=2.0e-5),
        )
        flux = mass_flux(density, velocity, grid)
        inventory_rate = jnp.sum(
            tendency * jnp.asarray(grid.cell_volumes)
        )

        assert abs(float(inventory_rate)) < 1.0e-12 * float(
            jnp.sum(jnp.abs(tendency) * jnp.asarray(grid.cell_volumes))
        )
        assert flux.x.shape == velocity.x.shape
