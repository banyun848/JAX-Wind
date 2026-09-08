"""Periodic low-Mach assembly using explicit state and numerical functions."""
from __future__ import annotations
import math
import numpy as np
from jaxwind.config.low_mach import ExtensionCase
from jaxwind.formulations.low_mach_abl import LowMachABLState, _add_velocity, _cell_vector_to_faces

def build_simulation(case: ExtensionCase):
    import jax
    import jax.numpy as jnp

    from jaxwind.simulation.abl import initialize_periodic, build_models
    from jaxwind.config.stages import load_workflow
    from jaxwind import (
        IdealGasMixture,
        StaggeredVelocity,
        build_pressure_poisson,
        build_tendency,
        continuity_residual,
        courant_number,
        dilatation_correction,
        divergence,
        enforce_impermeability,
        face_density,
        pressure_gradient,
        project_low_mach,
        validate,
    )

    workflow = load_workflow(case.source_workflow)
    configured = workflow.case
    physical = configured.physical
    grid = physical.physical_grid
    if not grid.is_uniform:
        raise ValueError("the low-Mach FFT extension requires a uniform mesh")
    boundaries, momentum, _scalar, _buoyancy, _surface = build_models(
        configured,
        periodic_x=True,
        evolve_scalar=False,
    )
    momentum_rhs = build_tendency(grid, boundaries, momentum)
    poisson = build_pressure_poisson(
        grid,
        backend=case.pressure_backend,
        dtype=physical.dtype,
    )
    equation_of_state = IdealGasMixture(
        pressure=case.pressure,
        background_gas_constant=case.air_gas_constant,
        species_gas_constants=(
            case.nitrogen_gas_constant,
            case.water_vapor_gas_constant,
        ),
    )
    shape = (grid.nz, grid.ny, grid.nx)
    if case.restart_formulation == "configured":
        source_solution = initialize_periodic(workflow.case, jax, jnp)
        velocity = source_solution.velocity
        initial_time = source_solution.time
        initial_step = source_solution.step
        kinematic_pressure = source_solution.pressure
    else:
        assert case.source_checkpoint is not None
        from jaxwind.io.checkpoint import checkpoint_metadata
        header = checkpoint_metadata(case.source_checkpoint)
        expected = "low-mach-abl" if case.restart_formulation == "low-mach" else "boussinesq"
        if header.get("formulation") != expected:
            raise ValueError("initialization artifact does not match the declared conversion")
        for axis in ("x", "y", "z"):
            if not np.array_equal(header["mesh"][axis + "_faces"], np.asarray(getattr(grid, axis + "_faces"))):
                raise ValueError("initialization mesh differs; use an explicit transformation")
        from jaxwind.io.field_archive import open_fields
        source = open_fields(case.source_checkpoint)
        velocity = StaggeredVelocity(
            jnp.asarray(source["velocity_x"]),
            jnp.asarray(source["velocity_y"]),
            jnp.asarray(source["velocity_z"]),
        )
        initial_time = jnp.asarray(source["time"])
        initial_step = jnp.asarray(source["step"], jnp.int32)
        if case.restart_formulation == "low-mach":
            pressure = jnp.asarray(source["pressure_dynamic_pa"])
            density = jnp.asarray(source["density"])
            momentum_tendency = StaggeredVelocity(
                jnp.asarray(source["momentum_tendency_x"]),
                jnp.asarray(source["momentum_tendency_y"]),
                jnp.asarray(source["momentum_tendency_z"]),
            )
            temperature = jnp.asarray(source["temperature"])
            water_vapor = jnp.asarray(source["water_vapor_mass_fraction"])
            nitrogen = jnp.asarray(source["nitrogen_mass_fraction"])
            continuity_error = jnp.asarray(
                source["continuity_error_kg_m3_s"]
            )
            checkpoint_pressure = float(source["thermodynamic_pressure_pa"])
            if not math.isclose(
                checkpoint_pressure, case.pressure, rel_tol=1.0e-7
            ):
                raise ValueError(
                    "low-Mach checkpoint thermodynamic pressure does not "
                    "match the continuation configuration"
                )
        else:
            kinematic_pressure = jnp.asarray(source["pressure"])
        source.close()
    validate(velocity, grid, boundaries)
    dtype = velocity.x.dtype
    if case.restart_formulation != "low-mach":
        temperature = jnp.full(shape, case.temperature, dtype)
        nitrogen = jnp.zeros(shape, dtype)
        water_fraction = case.water_vapor_mixing_ratio / (
            1.0 + case.water_vapor_mixing_ratio
        )
        water_vapor = jnp.full(shape, water_fraction, dtype)
        density = equation_of_state.density(
            temperature,
            (nitrogen, water_vapor),
        )
        reference_density = jnp.asarray(density[0, 0, 0], dtype)
        # Incompressible pressure is kinematic (p/rho). Low-Mach pressure is
        # dynamic, so scaling preserves the lagged pressure acceleration.
        pressure = kinematic_pressure * reference_density
        momentum_tendency = StaggeredVelocity(
            jnp.zeros_like(velocity.x),
            jnp.zeros_like(velocity.y),
            jnp.zeros_like(velocity.z),
        )
        continuity_error = jnp.asarray(0.0, dtype)
    else:
        expected = equation_of_state.density(
            temperature,
            (nitrogen, water_vapor),
        )
        maximum_eos_error = float(
            jnp.max(
                jnp.abs(density - expected) / jnp.maximum(expected, 1.0e-6)
            )
        )
        if maximum_eos_error > 1.0e-5:
            raise ValueError(
                "low-Mach checkpoint violates the configured mixture EOS: "
                f"relative error {maximum_eos_error:.3e}"
            )
    reference_density = jnp.asarray(density[0, 0, 0], dtype)
    initial = LowMachABLState(
        enforce_impermeability(velocity),
        pressure,
        density,
        momentum_tendency,
        temperature,
        water_vapor,
        nitrogen,
        continuity_error,
        initial_time,
        initial_step,
    )
    weights = (
        (8.0 / 15.0, 0.0),
        (5.0 / 12.0, -17.0 / 60.0),
        (3.0 / 4.0, -5.0 / 12.0),
    )

    def tendencies(velocity, density, execution_time):
        current = _add_velocity(
            momentum_rhs(velocity, execution_time),
            dilatation_correction(velocity, grid),
        )
        density_anomaly = (density - reference_density) / jnp.maximum(
            density, 1.0e-6
        )
        return _add_velocity(
            current,
            _cell_vector_to_faces(
                case.gravity[0] * density_anomaly,
                case.gravity[1] * density_anomaly,
                case.gravity[2] * density_anomaly,
                grid,
            ),
        )

    def step(state):
        velocity = state.velocity
        pressure = state.pressure
        density = state.density
        previous = state.momentum_tendency
        execution_time = state.time
        lagged = pressure_gradient(
            pressure,
            grid,
            periodic_x=True,
            periodic_y=True,
        )
        continuity_error = state.continuity_error
        current = previous
        for stage, (current_weight, previous_weight) in enumerate(weights):
            current = tendencies(velocity, density, execution_time)
            candidate = StaggeredVelocity(
                velocity.x
                + case.dt
                * (
                    current_weight * current.x
                    + previous_weight * previous.x
                ),
                velocity.y
                + case.dt
                * (
                    current_weight * current.y
                    + previous_weight * previous.y
                ),
                velocity.z
                + case.dt
                * (
                    current_weight * current.z
                    + previous_weight * previous.z
                ),
            )
            substep = case.dt * (current_weight + previous_weight)
            rho_face = face_density(density, candidate, grid)
            candidate = StaggeredVelocity(
                candidate.x - substep * lagged.x / rho_face.x,
                candidate.y - substep * lagged.y / rho_face.y,
                candidate.z - substep * lagged.z / rho_face.z,
            )
            candidate = enforce_impermeability(candidate)
            if stage == len(weights) - 1:
                velocity, correction = project_low_mach(
                    candidate,
                    state.density,
                    density,
                    poisson,
                    substep,
                    continuity_dt=case.dt,
                )
                pressure = pressure + correction * (substep / case.dt)
                continuity_error = jnp.max(
                    jnp.abs(
                        continuity_residual(
                            velocity,
                            state.density,
                            density,
                            grid,
                            case.dt,
                        )
                    )
                )
            else:
                velocity = candidate
            previous = current
            execution_time = execution_time + substep
        return LowMachABLState(
            velocity,
            pressure,
            density,
            current,
            state.temperature,
            state.water_vapor,
            state.nitrogen,
            continuity_error,
            initial_time
            + (state.step + 1 - initial_step).astype(dtype) * case.dt,
            state.step + 1,
        )

    def advance(state, count):
        return jax.lax.fori_loop(0, count, lambda _, value: step(value), state)

    return (
        workflow,
        grid,
        initial,
        jax.jit(advance, static_argnums=1),
        jax.jit(lambda state: courant_number(state.velocity, grid, case.dt)),
    )
