"""Run or continue a periodic ABL with low-Mach projection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import NamedTuple

import numpy as np

import tomllib


@dataclass(frozen=True, slots=True)
class ExtensionCase:
    source_workflow: Path
    source_checkpoint: Path | None
    restart_formulation: str
    dt: float
    steps: int
    chunk_steps: int
    frame_count: int
    statistics_window_seconds: float | None
    temperature: float
    pressure: float
    water_vapor_mixing_ratio: float
    air_gas_constant: float
    nitrogen_gas_constant: float
    water_vapor_gas_constant: float
    gravity: tuple[float, float, float]
    pressure_backend: str
    time_integration: str
    output: Path


class LowMachABLState(NamedTuple):
    velocity: object
    pressure: object
    density: object
    momentum_tendency: object
    temperature: object
    water_vapor: object
    nitrogen: object
    continuity_error: object
    time: object
    step: object


def _path(value: object, *, base: Path, workspace_relative: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("configured paths must be non-empty strings")
    path = Path(value)
    if path.is_absolute() or workspace_relative:
        return path
    return base / path


def load_case(path: str | Path) -> ExtensionCase:
    source = Path(path)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    restart = document["restart"]
    time_table = document["time"]
    thermodynamics = document["thermodynamics"]
    numerics = document["numerics"]
    output = document["output"]
    checkpoint_keys = tuple(
        key
        for key in ("incompressible_checkpoint", "low_mach_checkpoint")
        if key in restart
    )
    initial_condition = restart.get("initial_condition")
    if len(checkpoint_keys) + (initial_condition is not None) != 1:
        raise ValueError(
            "restart must define exactly one of incompressible_checkpoint, "
            "low_mach_checkpoint, or initial_condition"
        )
    if initial_condition is not None and initial_condition != "configured":
        raise ValueError("restart initial_condition must be configured")
    checkpoint_key = checkpoint_keys[0] if checkpoint_keys else None
    case = ExtensionCase(
        source_workflow=_path(
            restart["source_workflow"], base=source.parent
        ),
        source_checkpoint=(
            None
            if checkpoint_key is None
            else _path(
                restart[checkpoint_key],
                base=source.parent,
                workspace_relative=True,
            )
        ),
        restart_formulation=(
            "configured"
            if checkpoint_key is None
            else (
                "incompressible"
                if checkpoint_key == "incompressible_checkpoint"
                else "low-mach"
            )
        ),
        dt=float(time_table["dt_seconds"]),
        steps=int(time_table["steps"]),
        chunk_steps=int(time_table["chunk_steps"]),
        frame_count=int(time_table["frame_count"]),
        statistics_window_seconds=(
            None
            if "statistics_window_seconds" not in time_table
            else float(time_table["statistics_window_seconds"])
        ),
        temperature=float(thermodynamics["temperature_k"]),
        pressure=float(thermodynamics["pressure_pa"]),
        water_vapor_mixing_ratio=float(
            thermodynamics["water_vapor_mixing_ratio"]
        ),
        air_gas_constant=float(
            thermodynamics.get("air_gas_constant_j_kg_k", 287.05)
        ),
        nitrogen_gas_constant=float(
            thermodynamics.get("nitrogen_gas_constant_j_kg_k", 296.8)
        ),
        water_vapor_gas_constant=float(
            thermodynamics.get("water_vapor_gas_constant_j_kg_k", 461.5)
        ),
        gravity=tuple(
            float(value)
            for value in thermodynamics.get(
                "gravity_m_s2", (0.0, 0.0, -9.81)
            )
        ),
        pressure_backend=str(numerics["pressure_backend"]),
        time_integration=str(numerics["time_integration"]),
        output=_path(
            output["directory"], base=source.parent, workspace_relative=True
        ),
    )
    positive = (
        case.dt,
        case.temperature,
        case.pressure,
        case.air_gas_constant,
        case.nitrogen_gas_constant,
        case.water_vapor_gas_constant,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
        raise ValueError("timestep and thermodynamic constants must be positive")
    if case.steps <= 0 or case.chunk_steps <= 0:
        raise ValueError("steps and chunk_steps must be positive")
    if not 0 < case.frame_count <= case.steps:
        raise ValueError("frame_count must lie between one and steps")
    if case.statistics_window_seconds is not None and not (
        0.0 < case.statistics_window_seconds <= case.steps * case.dt
    ):
        raise ValueError(
            "statistics_window_seconds must be positive and no longer "
            "than the configured run"
        )
    if (
        not math.isfinite(case.water_vapor_mixing_ratio)
        or case.water_vapor_mixing_ratio < 0.0
    ):
        raise ValueError("water-vapor mixing ratio must be finite and nonnegative")
    if len(case.gravity) != 3 or not all(
        math.isfinite(value) for value in case.gravity
    ):
        raise ValueError("gravity_m_s2 must contain three finite values")
    if case.pressure_backend != "fft":
        raise ValueError("the periodic low-Mach extension requires FFT pressure")
    if case.time_integration != "fast-rk3":
        raise ValueError("the low-Mach extension currently requires fast-rk3")
    if not case.source_workflow.is_file():
        raise FileNotFoundError(
            f"missing source workflow: {case.source_workflow}"
        )
    if case.source_checkpoint is not None and not case.source_checkpoint.is_file():
        raise FileNotFoundError(
            f"missing source checkpoint: {case.source_checkpoint}"
        )
    return case


def _add_velocity(left, right):
    from jaxwind import StaggeredVelocity

    return StaggeredVelocity(
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    )


def _cell_vector_to_faces(x, y, z, grid):
    from jaxwind import StaggeredVelocity
    from jaxwind.discretization import _cells_to_faces

    return StaggeredVelocity(
        _cells_to_faces(x, grid, 2, periodic=True, boundary="copy"),
        _cells_to_faces(y, grid, 1, periodic=True, boundary="copy"),
        _cells_to_faces(z, grid, 0, periodic=False, boundary="zero"),
    )


def build_simulation(case: ExtensionCase):
    import jax
    import jax.numpy as jnp

    from applications.fv_abl.workflow import (
        _initial_periodic,
        _models,
        load_workflow,
    )
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
    boundaries, momentum, _scalar, _buoyancy, _surface = _models(
        configured,
        periodic_x=True,
        evolve_scalar=False,
    )
    momentum_rhs = build_tendency(grid, boundaries, momentum)
    poisson = build_pressure_poisson(
        grid,
        backend=case.pressure_backend,
        dtype=physical.pressure.dtype,
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
        source_solution = _initial_periodic(workflow.case, jax, jnp)
        velocity = source_solution.velocity
        initial_time = source_solution.time
        initial_step = source_solution.step
        kinematic_pressure = source_solution.pressure
    else:
        assert case.source_checkpoint is not None
        source = np.load(case.source_checkpoint)
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


def _save_checkpoint(path: Path, state, grid, case: ExtensionCase) -> None:
    np.savez_compressed(
        path,
        lengths_m=np.asarray((grid.lx, grid.ly, grid.lz)),
        x_faces_m=np.asarray(grid.x_faces),
        y_faces_m=np.asarray(grid.y_faces),
        z_faces_m=np.asarray(grid.z_faces),
        velocity_x=np.asarray(state.velocity.x),
        velocity_y=np.asarray(state.velocity.y),
        velocity_z=np.asarray(state.velocity.z),
        pressure_dynamic_pa=np.asarray(state.pressure),
        density=np.asarray(state.density),
        momentum_tendency_x=np.asarray(state.momentum_tendency.x),
        momentum_tendency_y=np.asarray(state.momentum_tendency.y),
        momentum_tendency_z=np.asarray(state.momentum_tendency.z),
        temperature=np.asarray(state.temperature),
        water_vapor_mass_fraction=np.asarray(state.water_vapor),
        nitrogen_mass_fraction=np.asarray(state.nitrogen),
        continuity_error_kg_m3_s=np.asarray(state.continuity_error),
        thermodynamic_pressure_pa=np.asarray(case.pressure),
        time=np.asarray(state.time),
        step=np.asarray(state.step),
    )


def _write_mean_profile(
    directory: Path,
    workflow,
    grid,
    fields: tuple[np.ndarray, ...],
    frame_times: list[float],
    *,
    statistics_start_fraction: float = 0.0,
) -> dict[str, object]:
    """Write horizontally/time-averaged profiles and a neutral log-law check."""

    from applications.fv_abl.evaluate import resolved
    from applications.fv_abl.workflow import _models
    from applications.fv_abl.reporting import write_log_law_svg
    from jaxwind import logarithmic_profile

    configuration = resolved(workflow.case)
    _boundaries, momentum, _scalar, _buoyancy, _surface = _models(
        workflow.case,
        periodic_x=True,
        evolve_scalar=False,
    )
    wall = momentum.surface
    if wall is None:
        raise ValueError("mean-profile reporting requires a neutral wall model")
    z = np.asarray(grid.z_centers, dtype=np.float64)
    u_history, v_history, w_history = fields[6:9]
    u2_history, v2_history, w2_history = fields[9:12]
    wall_friction_history = fields[12]
    if not 0.0 <= statistics_start_fraction < 1.0:
        raise ValueError("statistics_start_fraction must lie in [0, 1)")
    statistics_start_index = int(
        statistics_start_fraction * len(frame_times)
    )
    statistics_slice = slice(statistics_start_index, None)
    statistics_times = frame_times[statistics_start_index:]
    u_samples = u_history[statistics_slice]
    v_samples = v_history[statistics_slice]
    w_samples = w_history[statistics_slice]
    u2_samples = u2_history[statistics_slice]
    v2_samples = v2_history[statistics_slice]
    w2_samples = w2_history[statistics_slice]
    wall_friction_samples = wall_friction_history[statistics_slice]
    mean_u = np.mean(u_samples, axis=0, dtype=np.float64)
    mean_v = np.mean(v_samples, axis=0, dtype=np.float64)
    mean_w = np.mean(w_samples, axis=0, dtype=np.float64)
    u_rms = np.sqrt(
        np.maximum(
            np.mean(u2_samples, axis=0, dtype=np.float64) - mean_u**2,
            0.0,
        )
    )
    v_rms = np.sqrt(
        np.maximum(
            np.mean(v2_samples, axis=0, dtype=np.float64) - mean_v**2,
            0.0,
        )
    )
    w_rms = np.sqrt(
        np.maximum(
            np.mean(w2_samples, axis=0, dtype=np.float64) - mean_w**2,
            0.0,
        )
    )
    pressure_acceleration = float(
        configuration["pressure_acceleration_m_s2"][0]
    )
    reference_friction = math.sqrt(max(pressure_acceleration * grid.lz, 0.0))
    if reference_friction <= 0.0:
        raise ValueError("log-law comparison requires positive pressure forcing")
    roughness = float(configuration["roughness_length_m"])
    von_karman = float(wall.von_karman)
    reference_u = np.asarray(
        logarithmic_profile(grid, reference_friction, wall),
        dtype=np.float64,
    )
    fit_upper = min(0.1 * grid.lz, 2.0)
    fit_mask = z <= fit_upper
    if np.count_nonzero(fit_mask) < 3:
        raise ValueError("log-law fit requires three cells below 0.1 H")
    # Use the exact finite-volume cell-average coordinate rather than the
    # centre-point logarithm used by a continuous wall-law fit.
    fit_coordinate = (
        reference_u[fit_mask] * von_karman / reference_friction
    )
    slope, intercept = np.polyfit(fit_coordinate, mean_u[fit_mask], 1)
    fitted = slope * fit_coordinate + intercept
    residual = mean_u[fit_mask] - fitted
    total = mean_u[fit_mask] - np.mean(mean_u[fit_mask])
    fitted_friction = von_karman * slope
    fitted_roughness = (
        roughness * math.exp(-intercept / slope)
        if slope > 0.0
        else math.nan
    )
    reference_residual = mean_u[fit_mask] - reference_u[fit_mask]

    profile_path = directory / "mean_velocity_profile.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "z_m",
                "mean_u_m_s",
                "mean_v_m_s",
                "mean_w_m_s",
                "u_rms_m_s",
                "v_rms_m_s",
                "w_rms_m_s",
                "log_law_u_m_s",
            ),
        )
        writer.writeheader()
        for index, height in enumerate(z):
            writer.writerow(
                {
                    "z_m": height,
                    "mean_u_m_s": mean_u[index],
                    "mean_v_m_s": mean_v[index],
                    "mean_w_m_s": mean_w[index],
                    "u_rms_m_s": u_rms[index],
                    "v_rms_m_s": v_rms[index],
                    "w_rms_m_s": w_rms[index],
                    "log_law_u_m_s": reference_u[index],
                }
            )
    archive_path = directory / "mean_velocity_profile.npz"
    np.savez_compressed(
        archive_path,
        z_m=z,
        time_seconds=np.asarray(frame_times),
        u_profile_history_m_s=u_history,
        v_profile_history_m_s=v_history,
        w_profile_history_m_s=w_history,
        surface_friction_velocity_history_m_s=wall_friction_history,
        mean_u_m_s=mean_u,
        mean_v_m_s=mean_v,
        mean_w_m_s=mean_w,
        u_rms_m_s=u_rms,
        v_rms_m_s=v_rms,
        w_rms_m_s=w_rms,
        log_law_u_m_s=reference_u,
    )
    figure_path = directory / "mean_velocity_log_law.svg"
    write_log_law_svg(
        profile_path,
        figure_path,
        friction_velocity_m_s=reference_friction,
        roughness_length_m=roughness,
        von_karman=von_karman,
        model_label="low-Mach FV",
        statistics_label=f"{len(statistics_times)} post-spinup profiles",
    )
    summary = {
        "sample_count": len(statistics_times),
        "discarded_sample_count": statistics_start_index,
        "statistics_start_fraction": statistics_start_fraction,
        "sample_start_time_seconds": statistics_times[0],
        "sample_end_time_seconds": statistics_times[-1],
        "fit_height_range_m": [float(z[fit_mask][0]), fit_upper],
        "pressure_balance_friction_velocity_m_s": reference_friction,
        "mean_surface_friction_velocity_m_s": float(
            np.mean(wall_friction_samples, dtype=np.float64)
        ),
        "surface_to_pressure_friction_velocity_ratio": float(
            np.mean(wall_friction_samples, dtype=np.float64)
            / reference_friction
        ),
        "fitted_friction_velocity_m_s": fitted_friction,
        "fitted_roughness_length_m": fitted_roughness,
        "fitted_r_squared": float(
            1.0 - np.sum(residual**2) / np.maximum(np.sum(total**2), 1.0e-30)
        ),
        "fitted_rmse_m_s": float(np.sqrt(np.mean(residual**2))),
        "configured_log_law_rmse_m_s": float(
            np.sqrt(np.mean(reference_residual**2))
        ),
        "configured_log_law_maximum_error_m_s": float(
            np.max(np.abs(reference_residual))
        ),
        "profile_csv": str(profile_path),
        "profile_archive": str(archive_path),
        "figure": str(figure_path),
    }
    summary_path = directory / "mean_velocity_profile_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    summary["summary"] = str(summary_path)
    return summary


def run(case: ExtensionCase, *, steps: int | None = None) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", False)
    workflow, grid, state, advance, courant = build_simulation(case)
    total_steps = case.steps if steps is None else steps
    if total_steps <= 0:
        raise ValueError("run steps must be positive")
    case.output.mkdir(parents=True, exist_ok=True)
    from applications.fv_abl.workflow import _main_frame_steps, _models
    from jaxwind import friction_velocity

    frame_steps = _main_frame_steps(total_steps, min(case.frame_count, total_steps))
    frame_step_set = set(frame_steps)
    hub_index = int(
        np.clip(
            np.searchsorted(grid.z_faces, 0.876) - 1,
            0,
            grid.nz - 1,
        )
    )
    y_index = grid.ny // 2
    _boundaries, profile_momentum, _scalar, _buoyancy, _surface = _models(
        workflow.case, periodic_x=True, evolve_scalar=False
    )
    profile_wall = profile_momentum.surface
    if profile_wall is None:
        raise ValueError("profile capture requires a neutral wall model")

    @jax.jit
    def capture(current):
        u = 0.5 * (
            current.velocity.x + jnp.roll(current.velocity.x, -1, axis=2)
        )
        v = 0.5 * (
            current.velocity.y + jnp.roll(current.velocity.y, -1, axis=1)
        )
        w = 0.5 * (current.velocity.z[:-1] + current.velocity.z[1:])
        return (
            u[hub_index],
            u[:, y_index],
            current.temperature[hub_index],
            current.temperature[:, y_index],
            current.density[hub_index],
            current.density[:, y_index],
            jnp.mean(u, axis=(1, 2)),
            jnp.mean(v, axis=(1, 2)),
            jnp.mean(w, axis=(1, 2)),
            jnp.mean(u * u, axis=(1, 2)),
            jnp.mean(v * v, axis=(1, 2)),
            jnp.mean(w * w, axis=(1, 2)),
            friction_velocity(current.velocity, grid, profile_wall),
        )

    frames: list[tuple[np.ndarray, ...]] = []
    frame_times: list[float] = []
    frame_steps_global: list[int] = []
    started = time.perf_counter()
    completed = 0
    maximum_cfl = 0.0
    while completed < total_steps:
        count = min(case.chunk_steps, total_steps - completed)
        future_frames = [value for value in frame_steps if value > completed]
        if future_frames:
            count = min(count, future_frames[0] - completed)
        state = advance(state, count)
        jax.block_until_ready(state.velocity.x)
        completed += count
        cfl = float(courant(state))
        maximum_cfl = max(maximum_cfl, cfl)
        if completed in frame_step_set:
            values = jax.device_get(capture(state))
            frames.append(tuple(np.asarray(value) for value in values))
            frame_times.append(float(state.time))
            frame_steps_global.append(int(state.step))
        print(
            f"low-mach {completed:8d}/{total_steps} "
            f"time={float(state.time):9.3f}s CFL={cfl:.3f} "
            f"mass_res={float(state.continuity_error):.3e}kg/m3/s "
            f"frames={len(frames)}/{len(frame_steps)}",
            flush=True,
        )
    elapsed = time.perf_counter() - started
    fields = tuple(np.stack([frame[index] for frame in frames]) for index in range(13))
    frame_path = case.output / "low_mach_extension_frames.npz"
    np.savez_compressed(
        frame_path,
        u_hub_yx=fields[0],
        u_center_zx=fields[1],
        temperature_hub_yx=fields[2],
        temperature_center_zx=fields[3],
        density_hub_yx=fields[4],
        density_center_zx=fields[5],
        u_profile_history_m_s=fields[6],
        v_profile_history_m_s=fields[7],
        w_profile_history_m_s=fields[8],
        u2_profile_history_m2_s2=fields[9],
        v2_profile_history_m2_s2=fields[10],
        w2_profile_history_m2_s2=fields[11],
        surface_friction_velocity_history_m_s=fields[12],
        time_seconds=np.asarray(frame_times),
        step=np.asarray(frame_steps_global),
        x_m=np.asarray(grid.x_centers),
        y_m=np.asarray(grid.y_centers),
        z_m=np.asarray(grid.z_centers),
        x_faces_m=np.asarray(grid.x_faces),
        y_faces_m=np.asarray(grid.y_faces),
        z_faces_m=np.asarray(grid.z_faces),
    )
    profile = _write_mean_profile(
        case.output,
        workflow,
        grid,
        fields,
        frame_times,
        statistics_start_fraction=(
            max(
                0.0,
                1.0
                - case.statistics_window_seconds / (total_steps * case.dt),
            )
            if case.statistics_window_seconds is not None
            else (0.8 if case.restart_formulation == "configured" else 0.0)
        ),
    )
    checkpoint = case.output / "low_mach_extension_final.npz"
    _save_checkpoint(checkpoint, state, grid, case)
    result = {
        "source_workflow": str(case.source_workflow),
        "source_checkpoint": (
            None if case.source_checkpoint is None else str(case.source_checkpoint)
        ),
        "restart_formulation": case.restart_formulation,
        "cells": [grid.nx, grid.ny, grid.nz],
        "steps": total_steps,
        "duration_seconds": total_steps * case.dt,
        "start_time_seconds": float(state.time) - total_steps * case.dt,
        "end_time_seconds": float(state.time),
        "elapsed_seconds": elapsed,
        "steps_per_second": total_steps / elapsed,
        "simulation_to_wall_ratio": total_steps * case.dt / elapsed,
        "pressure_backend": case.pressure_backend,
        "time_integration": case.time_integration,
        "flow_formulation": "variable-density-low-mach",
        "thermodynamic_pressure_pa": case.pressure,
        "temperature_k": case.temperature,
        "water_vapor_mixing_ratio": case.water_vapor_mixing_ratio,
        "initial_density_kg_m3": float(state.density[0, 0, 0]),
        "minimum_density_kg_m3": float(jnp.min(state.density)),
        "maximum_density_kg_m3": float(jnp.max(state.density)),
        "minimum_temperature_k": float(jnp.min(state.temperature)),
        "maximum_temperature_k": float(jnp.max(state.temperature)),
        "final_cfl": float(courant(state)),
        "maximum_sampled_cfl": maximum_cfl,
        "continuity_residual_kg_m3_s": float(state.continuity_error),
        "frame_count": len(frames),
        "statistics_window_seconds": case.statistics_window_seconds,
        "frames": str(frame_path),
        "checkpoint": str(checkpoint),
        "mean_profile": profile,
        "output": str(case.output),
    }
    (case.output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--steps", type=int)
    arguments = parser.parse_args(argv)
    run(load_case(arguments.config), steps=arguments.steps)
    return 0


__all__ = [
    "ExtensionCase",
    "LowMachABLState",
    "build_simulation",
    "load_case",
    "run",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
