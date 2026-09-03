from __future__ import annotations

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind import (
    FREE_SLIP,
    OPEN,
    Boundaries,
    FlowModel,
    LinearBoussinesqBuoyancy,
    PassiveScalar,
    SubgridCooling,
    SubgridSpray,
    Wall,
    boussinesq_tendency,
    build_atmospheric_step,
    build_pressure_poisson,
    build_subgrid_cooling_source,
    build_subgrid_spray_sources,
    divergence,
    initial_atmospheric_solution,
    zeros,
)


class BoussinesqTendencyTest(unittest.TestCase):
    grid = UniformGrid(6, 4, 5, 3.0, 2.0, 2.5)

    def test_tendency_is_hydrostatic_free_and_impermeable(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        horizontal = jnp.arange(self.grid.nx, dtype=jnp.float64)[None, :]
        scalar = 300.0 + jnp.broadcast_to(horizontal, shape)
        coefficient = 9.81 / 300.0
        tendency = boussinesq_tendency(
            scalar,
            LinearBoussinesqBuoyancy(coefficient),
        )

        self.assertTrue(bool(jnp.all(tendency.x == 0.0)))
        self.assertTrue(bool(jnp.all(tendency.y == 0.0)))
        self.assertTrue(bool(jnp.all(tendency.z[0] == 0.0)))
        self.assertTrue(bool(jnp.all(tendency.z[-1] == 0.0)))
        self.assertTrue(
            bool(
                jnp.allclose(
                    jnp.mean(tendency.z[1:-1], axis=(-2, -1)),
                    0.0,
                    atol=1.0e-14,
                )
            )
        )
        expected = coefficient * (horizontal - jnp.mean(horizontal))
        self.assertTrue(bool(jnp.allclose(tendency.z[1], expected)))

    def test_coupled_step_projects_the_buoyant_acceleration(self) -> None:
        shape = (self.grid.nz, self.grid.ny, self.grid.nx)
        scalar = jnp.broadcast_to(
            jnp.sin(2.0 * jnp.pi * jnp.arange(self.grid.nx) / self.grid.nx),
            shape,
        )
        boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))
        poisson = build_pressure_poisson(self.grid, backend="fft")
        step = build_atmospheric_step(
            self.grid,
            boundaries,
            poisson,
            FlowModel(),
            PassiveScalar(),
            LinearBoussinesqBuoyancy(0.1),
        )
        initial = initial_atmospheric_solution(
            self.grid,
            scalar=scalar,
            dtype="float64",
        )
        final = step(initial, 0.1)

        self.assertGreater(float(jnp.max(jnp.abs(final.velocity.z))), 0.0)
        self.assertLess(
            float(jnp.max(jnp.abs(divergence(final.velocity, self.grid)))),
            1.0e-12,
        )

    def test_subgrid_cooling_conserves_its_thermodynamic_power(self) -> None:
        model = SubgridCooling(
            cooling_power_w=4_842.570359,
            air_density_kg_m3=1.225,
            air_heat_capacity_j_kg_k=1_005.0,
            center_m=(1.4, 1.0, 1.2),
            standard_deviation_m=(0.3, 0.2, 0.15),
            ramp_time_s=1.0,
        )
        source = build_subgrid_cooling_source(
            self.grid, model, dtype="float64"
        )
        full = source(jnp.asarray(1.0))
        integrated = jnp.sum(
            full * jnp.asarray(self.grid.cell_volumes)
        )

        self.assertAlmostEqual(
            float(integrated),
            model.integrated_temperature_sink_k_m3_s,
            places=12,
        )
        self.assertTrue(bool(jnp.all(source(jnp.asarray(0.0)) == 0.0)))
        self.assertTrue(
            bool(jnp.allclose(source(jnp.asarray(0.5)), 0.5 * full))
        )

        spray = SubgridSpray(
            cooling_power_w=model.cooling_power_w,
            mass_flow_rate_kg_s=0.0125,
            injection_speed_m_s=4.0,
            air_density_kg_m3=model.air_density_kg_m3,
            air_heat_capacity_j_kg_k=model.air_heat_capacity_j_kg_k,
            nozzle_m=(0.5, 1.0, 1.2),
            nozzle_diameter_m=0.005,
            cone_half_angle_degrees=0.0,
            axial_standard_deviation_m=0.3,
            minimum_radial_standard_deviation_m=(0.2, 0.15),
            ramp_time_s=1.0,
        )
        temperature, forcing = build_subgrid_spray_sources(
            self.grid, spray, dtype="float64"
        )
        boundaries = Boundaries(
            Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN
        )
        velocity = zeros(self.grid, "float64", boundaries)
        momentum = forcing(velocity, jnp.asarray(1.0))
        dual_x_widths = jnp.concatenate(
            (
                0.5 * jnp.asarray(self.grid.x_widths[:1]),
                0.5
                * (
                    jnp.asarray(self.grid.x_widths[:-1])
                    + jnp.asarray(self.grid.x_widths[1:])
                ),
                0.5 * jnp.asarray(self.grid.x_widths[-1:]),
            )
        )
        face_volumes = (
            jnp.asarray(self.grid.z_widths)[:, None, None]
            * jnp.asarray(self.grid.y_widths)[None, :, None]
            * dual_x_widths[None, None, :]
        )

        self.assertAlmostEqual(
            float(jnp.sum(temperature(jnp.asarray(1.0)) * self.grid.cell_volumes)),
            spray.integrated_temperature_sink_k_m3_s,
            places=12,
        )
        self.assertAlmostEqual(
            float(jnp.sum(momentum.x * face_volumes)),
            spray.integrated_axial_acceleration_m4_s2,
            places=12,
        )
        self.assertTrue(bool(jnp.all(momentum.y == 0.0)))
        self.assertTrue(bool(jnp.all(momentum.z == 0.0)))
        self.assertTrue(
            bool(jnp.all(forcing(velocity, jnp.asarray(0.0)).x == 0.0))
        )


if __name__ == "__main__":
    unittest.main()
