from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.physics.cryogenic import (
    CryogenicMicrophysicsConfig,
    advance_nitrogen_droplet,
)
from jaxwind.cryogenic import (
    LN2InletControl,
    LN2Jet,
    initial_ln2_parcels,
    inject_ln2_parcels,
    vapor_nozzle_source,
)


def jet(**changes) -> LN2Jet:
    values = dict(
        x=3.0,
        y=3.0,
        z=0.876,
        radius=0.005,
        speed=8.0,
        mass_flow_rate=0.0125,
        vapor_quality=0.217596,
        maximum_parcels=64,
        parcels_per_step=8,
    )
    values.update(changes)
    return LN2Jet(**values)


def test_vapor_kernel_and_parcel_injection_conserve_total_source_mass() -> None:
    grid = UniformGrid(16, 16, 16, 6.0, 6.0, 3.6)
    model = jet(ramp_time=0.0)
    source = vapor_nozzle_source(grid, model, jnp.float64)
    dt = 5.0e-4
    parcels = inject_ln2_parcels(
        initial_ln2_parcels(model, jnp.float64),
        jnp.asarray(0, jnp.int32),
        dt,
        model,
    )
    liquid_mass = jnp.sum(
        parcels.mass * parcels.multiplicity * parcels.active
    )

    assert source.shape == (16, 16, 16)
    assert abs(float(jnp.sum(source)) - 1.0) < 1.0e-14
    assert abs(float(liquid_mass) - model.liquid_mass_flow_rate * dt) < 1.0e-12
    assert abs(
        model.vapor_mass_flow_rate
        + model.liquid_mass_flow_rate
        - model.mass_flow_rate
    ) < 1.0e-15


def test_analytic_and_empirical_sources_set_bounded_radial_profiles() -> None:
    analytic = jet(
        ramp_time=0.0,
        cone_half_angle_degrees=13.5,
        edge_speed_ratio=0.9,
        edge_diameter_ratio=1.2,
        minimum_diameter=1.0e-6,
        maximum_diameter=1.0e-3,
    )
    analytic_parcels = inject_ln2_parcels(
        initial_ln2_parcels(analytic, jnp.float64),
        jnp.asarray(0, jnp.int32),
        1.0e-5,
        analytic,
    )
    analytic_radius = jnp.sqrt(
        (analytic_parcels.y - analytic.y) ** 2
        + (analytic_parcels.z - analytic.z) ** 2
    )
    assert float(jnp.max(analytic_radius[analytic_parcels.active])) <= analytic.radius
    assert float(jnp.max(jnp.hypot(analytic_parcels.v, analytic_parcels.w))) > 0.0

    empirical = jet(
        ramp_time=0.0,
        radius=0.02,
        minimum_diameter=1.0e-6,
        maximum_diameter=40.0e-6,
        profile_radius_m=(0.0, 0.01, 0.02),
        profile_axial_velocity_m_s=(30.0, 20.0, 10.0),
        profile_radial_velocity_m_s=(0.0, 60.0, 5.0),
        profile_d10_m=(10.0e-6, 15.0e-6, 8.0e-6),
    )
    parcels = inject_ln2_parcels(
        initial_ln2_parcels(empirical, jnp.float64),
        jnp.asarray(0, jnp.int32),
        1.0e-5,
        empirical,
    )
    active = parcels.active
    radius = jnp.sqrt(
        (parcels.y - empirical.y) ** 2 + (parcels.z - empirical.z) ** 2
    )
    expected_u = jnp.interp(
        radius,
        jnp.asarray(empirical.profile_radius_m),
        jnp.asarray(empirical.profile_axial_velocity_m_s),
    )
    expected_vr = jnp.interp(
        radius,
        jnp.asarray(empirical.profile_radius_m),
        jnp.asarray(empirical.profile_radial_velocity_m_s),
    )

    assert bool(jnp.allclose(parcels.u[active], expected_u[active]))
    assert bool(
        jnp.allclose(
            jnp.hypot(parcels.v, parcels.w)[active], expected_vr[active]
        )
    )
    assert bool(jnp.all(parcels.diameter[active] >= empirical.minimum_diameter))
    assert bool(jnp.all(parcels.diameter[active] <= empirical.maximum_diameter))


def test_inlet_controls_have_finite_nonzero_jax_gradients() -> None:
    model = jet(
        ramp_time=0.0,
        minimum_diameter=1.0e-6,
        maximum_diameter=1.0e-3,
    )
    empty = initial_ln2_parcels(model, jnp.float64)

    def mean_diameter(value):
        control = LN2InletControl(
            value,
            jnp.asarray(1.0),
            jnp.asarray(0.9),
            jnp.asarray(1.1),
        )
        injected = inject_ln2_parcels(
            empty, jnp.asarray(0, jnp.int32), 1.0e-5, model, control
        )
        return jnp.mean(injected.diameter[: model.parcels_per_step])

    derivative = jax.grad(mean_diameter)(jnp.asarray(50.0e-6))

    assert bool(jnp.isfinite(derivative))
    assert float(derivative) > 0.0


def test_fully_evaporated_droplet_has_finite_tangents() -> None:
    config = CryogenicMicrophysicsConfig()

    def update(temperature):
        return advance_nitrogen_droplet(
            jnp.asarray(0.0, jnp.float32),
            jnp.asarray(1.0e-8, jnp.float32),
            temperature,
            jnp.asarray(295.0, jnp.float32),
            jnp.asarray(1.0e-10, jnp.float32),
            1.0e-5,
            config,
        )

    _, tangent = jax.jvp(
        update,
        (jnp.asarray(0.0, jnp.float32),),
        (jnp.asarray(1.0, jnp.float32),),
    )

    assert all(
        bool(jnp.all(jnp.isfinite(leaf)))
        for leaf in jax.tree.leaves(tangent)
    )
