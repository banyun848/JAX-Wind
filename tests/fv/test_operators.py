from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import AnalyticalGrid, TanhMapping, UniformGrid
from jaxwind.fv import (
    FREE_SLIP,
    Boundaries,
    StaggeredVelocity,
    Wall,
    advection,
    build_pressure_poisson,
    diffusion,
    divergence,
    kinetic_energy,
    project,
    stable_timestep,
)
from jaxwind.fv.metrics import shaped_center_distances, shaped_widths


class AdvectionTest(unittest.TestCase):
    grid = UniformGrid(10, 8, 6, 1.0, 0.8, 0.6)

    def solenoidal(self, seed: int, grid=None) -> StaggeredVelocity:
        grid = self.grid if grid is None else grid
        keys = jax.random.split(jax.random.PRNGKey(seed), 3)
        cells = (grid.nz, grid.ny, grid.nx)
        candidate = StaggeredVelocity(
            jax.random.normal(keys[0], cells),
            jax.random.normal(keys[1], cells),
            jax.random.normal(keys[2], (grid.nz + 1, grid.ny, grid.nx))
            .at[0]
            .set(0.0)
            .at[-1]
            .set(0.0),
        )
        backend = "fft" if grid.is_uniform else "gmg"
        poisson = build_pressure_poisson(grid, backend=backend)
        projected, _ = project(candidate, poisson, 0.1)
        return projected

    def test_transport_conserves_the_periodic_momentum_components(self) -> None:
        """Flux form must telescope exactly across the periodic axes."""
        tendency = advection(self.solenoidal(1), self.grid)
        for component in (tendency.x, tendency.y):
            self.assertLess(
                float(jnp.abs(jnp.sum(component))),
                1.0e-12 * float(jnp.sum(jnp.abs(component))),
            )

    def test_wall_normal_momentum_changes_only_by_the_wall_flux(self) -> None:
        """The walls carry a normal momentum flux; nothing else may leak."""
        velocity = self.solenoidal(1)
        tendency = advection(velocity, self.grid)
        normal_flux = (0.5 * (velocity.z[:-1] + velocity.z[1:])) ** 2
        wall_flux = (
            jnp.sum(normal_flux[-1]) - jnp.sum(normal_flux[0])
        ) / self.grid.dz
        self.assertLess(
            float(jnp.abs(jnp.sum(tendency.z) + wall_flux)),
            1.0e-12 * float(jnp.sum(jnp.abs(tendency.z))),
        )

    def test_transport_conserves_kinetic_energy(self) -> None:
        """Mapped contravariant fluxes must retain discrete skew symmetry."""
        mapped = AnalyticalGrid(
            10,
            8,
            6,
            1.0,
            0.8,
            0.6,
            TanhMapping(1.2, focus=0.0),
            TanhMapping(1.0),
            TanhMapping(0.8),
        )
        for grid in (self.grid, mapped):
            with self.subTest(mapped=not grid.is_uniform):
                velocity = self.solenoidal(2, grid)
                tendency = advection(velocity, grid)
                u_volume = (
                    shaped_widths(grid, 0)
                    * shaped_widths(grid, 1)
                    * shaped_center_distances(grid, 2, periodic=True)
                )
                v_volume = (
                    shaped_widths(grid, 0)
                    * shaped_center_distances(grid, 1, periodic=True)
                    * shaped_widths(grid, 2)
                )
                w_volume = (
                    shaped_center_distances(grid, 0, periodic=False)
                    * shaped_widths(grid, 1)
                    * shaped_widths(grid, 2)
                )
                production = (
                    jnp.sum(u_volume * velocity.x * tendency.x)
                    + jnp.sum(v_volume * velocity.y * tendency.y)
                    + jnp.sum(w_volume * velocity.z * tendency.z)
                )
                energy = kinetic_energy(velocity, grid)
                self.assertLess(
                    float(jnp.abs(production)), 1.0e-8 * float(energy)
                )

    def test_uniform_flow_is_not_advected(self) -> None:
        grids = (
            self.grid,
            AnalyticalGrid(
                10,
                8,
                6,
                1.0,
                0.8,
                0.6,
                TanhMapping(1.4, focus=0.0),
                TanhMapping(1.2),
                TanhMapping(1.0),
            ),
        )
        for grid in grids:
            with self.subTest(mapped=not grid.is_uniform):
                cells = (grid.nz, grid.ny, grid.nx)
                uniform = StaggeredVelocity(
                    jnp.full(cells, 3.0),
                    jnp.full(cells, -1.0),
                    jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
                )
                tendency = advection(uniform, grid)
                self.assertLess(
                    float(jnp.max(jnp.abs(divergence(uniform, grid)))),
                    1.0e-12,
                )
                for component in tendency:
                    self.assertLess(float(jnp.max(jnp.abs(component))), 1.0e-12)
        mapped = grids[1]
        self.assertLess(mapped.x_widths[0], mapped.x_widths[-1])
        self.assertLess(mapped.y_widths[mapped.ny // 2], mapped.y_widths[0])
        self.assertLess(mapped.z_widths[mapped.nz // 2], mapped.z_widths[0])


class DiffusionTest(unittest.TestCase):
    grid = UniformGrid(6, 6, 4, 1.0, 1.0, 0.8)

    def uniform(self) -> StaggeredVelocity:
        cells = (self.grid.nz, self.grid.ny, self.grid.nx)
        return StaggeredVelocity(
            jnp.full(cells, 2.0),
            jnp.zeros(cells),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )

    def test_no_slip_walls_brake_a_uniform_flow(self) -> None:
        tendency = diffusion(self.uniform(), self.grid, Boundaries(), 0.5)
        # Quadratic wall closure: ghost = (8 * 0 - 6 * 2 + 2) / 3.
        expected = 0.5 * (2.0 - 4.0 - 10.0 / 3.0) / self.grid.dz**2
        self.assertAlmostEqual(float(tendency.x[0, 0, 0]), expected, places=10)
        self.assertAlmostEqual(float(tendency.x[-1, 0, 0]), expected, places=10)
        self.assertLess(float(jnp.max(jnp.abs(tendency.x[1:-1]))), 1.0e-12)

    def test_moving_walls_drive_the_flow(self) -> None:
        boundaries = Boundaries(Wall(x_velocity=2.0), Wall(x_velocity=2.0))
        tendency = diffusion(self.uniform(), self.grid, boundaries, 0.5)
        self.assertLess(float(jnp.max(jnp.abs(tendency.x))), 1.0e-12)

    def test_free_slip_walls_leave_a_uniform_flow_alone(self) -> None:
        boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))
        tendency = diffusion(self.uniform(), self.grid, boundaries, 0.5)
        self.assertLess(float(jnp.max(jnp.abs(tendency.x))), 1.0e-12)

    def test_a_free_slip_wall_cannot_move(self) -> None:
        with self.assertRaises(ValueError):
            Wall(FREE_SLIP, x_velocity=1.0)

    def test_diffusion_dissipates_energy(self) -> None:
        keys = jax.random.split(jax.random.PRNGKey(4), 2)
        cells = (self.grid.nz, self.grid.ny, self.grid.nx)
        velocity = StaggeredVelocity(
            jax.random.normal(keys[0], cells),
            jax.random.normal(keys[1], cells),
            jnp.zeros((self.grid.nz + 1, self.grid.ny, self.grid.nx)),
        )
        tendency = diffusion(velocity, self.grid, Boundaries(), 0.25)
        production = jnp.sum(velocity.x * tendency.x) + jnp.sum(
            velocity.y * tendency.y
        )
        self.assertLess(float(production), 0.0)


class TimestepTest(unittest.TestCase):
    def test_the_viscous_limit_binds_a_still_fluid(self) -> None:
        grid = UniformGrid(8, 8, 8, 1.0, 1.0, 1.0)
        cells = (grid.nz, grid.ny, grid.nx)
        still = StaggeredVelocity(
            jnp.zeros(cells),
            jnp.zeros(cells),
            jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
        )
        limit = float(stable_timestep(still, grid, 1.0, diffusion_number=0.25))
        self.assertAlmostEqual(limit, 0.25 / (4.0 * 64.0), places=12)

    def test_an_inviscid_still_fluid_has_no_limit(self) -> None:
        grid = UniformGrid(4, 4, 4, 1.0, 1.0, 1.0)
        cells = (grid.nz, grid.ny, grid.nx)
        still = StaggeredVelocity(
            jnp.zeros(cells),
            jnp.zeros(cells),
            jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
        )
        self.assertTrue(jnp.isinf(stable_timestep(still, grid, 0.0)))


if __name__ == "__main__":
    unittest.main()
