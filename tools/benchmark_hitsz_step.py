#!/usr/bin/env python3
"""Benchmark and decompose the periodic HITSZ finite-volume warmup step.

End-to-end measurements use the production workflow builders. Component
measurements synchronize between phases; they locate backend regressions but
their sum is not expected to equal the fused end-to-end execution time.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import statistics
import sys
import time
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "cases" / "HITSZWindTunnel" / "fv_workflow.toml"
RK3_CURRENT = (8.0 / 15.0, 5.0 / 12.0, 3.0 / 4.0)
RK3_PREVIOUS = (0.0, -17.0 / 60.0, -5.0 / 12.0)

# Running a script by path adds tools/, not the checkout root, to sys.path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=DEFAULT_CONFIG,
        help=f"workflow TOML (default: {DEFAULT_CONFIG.relative_to(ROOT)})",
    )
    result.add_argument("--block-steps", type=_positive, default=100)
    result.add_argument("--setup-steps", type=_positive, default=2)
    result.add_argument("--samples", type=_positive, default=5)
    result.add_argument("--component-repeats", type=_positive, default=3)
    result.add_argument("--json", type=Path, dest="json_path")
    return result


@dataclass(frozen=True)
class Measurement:
    compile_seconds: float
    samples_ms: tuple[float, ...]
    median_ms: float
    minimum_ms: float
    maximum_ms: float


def _ready(jax, value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return value


def _measure(
    jax,
    function: Callable[[], Any],
    *,
    samples: int,
    repeats: int = 1,
    units_per_call: int = 1,
) -> tuple[Measurement, Any]:
    started = time.perf_counter()
    compiled_result = _ready(jax, function())
    compile_seconds = time.perf_counter() - started

    values: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        for _ in range(repeats):
            _ready(jax, function())
        elapsed = time.perf_counter() - started
        values.append(1.0e3 * elapsed / (repeats * units_per_call))
    return (
        Measurement(
            compile_seconds,
            tuple(values),
            statistics.median(values),
            min(values),
            max(values),
        ),
        compiled_result,
    )


def _as_report(value: Measurement, *, unit: str) -> dict[str, Any]:
    result = asdict(value)
    result["samples_ms"] = list(value.samples_ms)
    result["unit"] = unit
    return result


def _environment(jax, jaxlib) -> dict[str, Any]:
    names = (
        "JAX_PLATFORMS",
        "JAX_PLATFORM_NAME",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "SLURM_JOB_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_CPUS_PER_TASK",
    )
    devices = [
        {
            "id": getattr(device, "id", None),
            "platform": getattr(device, "platform", None),
            "device_kind": getattr(device, "device_kind", None),
            "process_index": getattr(device, "process_index", None),
        }
        for device in jax.devices()
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "default_backend": jax.default_backend(),
        "devices": devices,
        "variables": {name: os.environ.get(name) for name in names},
    }


def _print_table(
    title: str,
    rows: list[tuple[str, float, float | None, float]],
) -> None:
    print(f"\n{title}")
    print(f"{'phase':31s} {'ms':>11s} {'% full':>9s} {'compile s':>11s}")
    for name, milliseconds, share, compile_seconds in rows:
        percentage = "-" if share is None else f"{share:8.1f}"
        print(
            f"{name:31s} {milliseconds:11.4f} "
            f"{percentage:>9s} {compile_seconds:11.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    config_path = arguments.config.resolve()

    import jax
    import jax.numpy as jnp
    import jaxlib

    from applications.fv_abl.workflow import (
        _initial_periodic,
        _models,
        _periodic_advance,
        load_workflow,
    )
    from jaxwind import (
        StaggeredVelocity,
        advection,
        build_adaptive_atmospheric_run,
        build_atmospheric_run,
        build_pressure_poisson,
        build_tendency,
        divergence,
        eddy_viscosity,
        enforce_impermeability,
        pressure_gradient,
        project,
        scalar_tendency,
        stable_timestep,
        subfilter_tendency,
        wall_tendency,
    )

    workflow = load_workflow(config_path)
    configured = workflow.case
    case = configured.physical
    options = configured.options
    grid = case.physical_grid
    if options.time_integration != "fast-rk3":
        raise ValueError("the decomposition currently requires fast-rk3")
    if options.cfl_ceiling is None:
        raise ValueError("the decomposition requires adaptive CFL control")

    jax.config.update("jax_enable_x64", case.dtype == "float64")
    boundaries, momentum, scalar, buoyancy, coupled_surface = _models(
        configured, periodic_x=True
    )
    if scalar is None or momentum.subfilter is None:
        raise ValueError("the HITSZ decomposition requires scalar and SGS models")
    if buoyancy is not None or coupled_surface is not None:
        raise ValueError(
            "the benchmark currently targets the neutral HITSZ warmup closures"
        )

    step, _ = _periodic_advance(configured)
    fixed_run = build_atmospheric_run(step)
    adaptive_run = build_adaptive_atmospheric_run(
        step,
        grid,
        cfl_ceiling=options.cfl_ceiling,
        maximum_dt=case.dt_seconds,
    )
    solution = _ready(jax, _initial_periodic(configured, jax, jnp))

    setup_target = float(solution.time) + arguments.setup_steps * case.dt_seconds
    setup_started = time.perf_counter()
    solution = _ready(
        jax,
        adaptive_run(solution, setup_target, arguments.setup_steps),
    )
    setup_seconds = time.perf_counter() - setup_started

    cfl_kernel = jax.jit(
        lambda velocity: stable_timestep(
            velocity, grid, 0.0, courant=options.cfl_ceiling
        )
    )
    stable_dt = min(float(cfl_kernel(solution.velocity)), case.dt_seconds)
    adaptive_target = (
        float(solution.time) + arguments.block_steps * case.dt_seconds
    )

    end_to_end: dict[str, Measurement] = {}
    end_to_end["adaptive_workflow"], adaptive_result = _measure(
        jax,
        lambda: adaptive_run(
            solution, adaptive_target, arguments.block_steps
        ),
        samples=arguments.samples,
        units_per_call=arguments.block_steps,
    )
    end_to_end["fixed_dt"], _ = _measure(
        jax,
        lambda: fixed_run(solution, stable_dt, arguments.block_steps),
        samples=arguments.samples,
        units_per_call=arguments.block_steps,
    )
    actual_steps = int(adaptive_result.step) - int(solution.step)
    if actual_steps != arguments.block_steps:
        raise RuntimeError(
            f"adaptive benchmark advanced {actual_steps} steps, expected "
            f"{arguments.block_steps}"
        )

    poisson = build_pressure_poisson(grid, backend="fft", dtype=case.dtype)
    momentum_rhs = build_tendency(grid, boundaries, momentum)

    def scalar_rhs(velocity, scalar_field):
        viscosity = eddy_viscosity(
            velocity, grid, boundaries, momentum.subfilter
        )
        return scalar_tendency(
            scalar_field,
            velocity,
            grid,
            scalar,
            eddy_viscosity=viscosity,
        )

    def explicit_rhs(velocity, scalar_field, execution_time):
        return (
            momentum_rhs(velocity, execution_time),
            scalar_rhs(velocity, scalar_field),
        )

    lagged_kernel = jax.jit(lambda pressure: pressure_gradient(pressure, grid))
    explicit_kernel = jax.jit(explicit_rhs)
    momentum_kernel = jax.jit(momentum_rhs)
    scalar_kernel = jax.jit(scalar_rhs)
    advection_kernel = jax.jit(lambda velocity: advection(velocity, grid))
    sgs_kernel = jax.jit(
        lambda velocity: subfilter_tendency(
            velocity,
            grid,
            boundaries,
            momentum.subfilter,
            surface=momentum.surface,
        )
    )
    viscosity_kernel = jax.jit(
        lambda velocity: eddy_viscosity(
            velocity, grid, boundaries, momentum.subfilter
        )
    )
    wall_kernel = (
        None
        if momentum.surface is None
        else jax.jit(
            lambda velocity: wall_tendency(velocity, grid, momentum.surface)
        )
    )

    lagged = _ready(jax, lagged_kernel(solution.pressure))
    current_momentum, current_scalar = _ready(
        jax,
        explicit_kernel(solution.velocity, solution.scalar, solution.time),
    )

    def rk_update(
        velocity,
        scalar_field,
        previous_momentum,
        previous_scalar,
        current_momentum_value,
        current_scalar_value,
        lagged_gradient,
    ):
        current_scale = jnp.asarray(
            stable_dt * RK3_CURRENT[0], velocity.x.dtype
        )
        previous_scale = jnp.asarray(
            stable_dt * RK3_PREVIOUS[0], velocity.x.dtype
        )
        pressure_scale = jnp.asarray(
            stable_dt * (RK3_CURRENT[0] + RK3_PREVIOUS[0]),
            velocity.x.dtype,
        )
        candidate = enforce_impermeability(
            StaggeredVelocity(
                velocity.x
                + current_scale * current_momentum_value.x
                + previous_scale * previous_momentum.x
                - pressure_scale * lagged_gradient.x,
                velocity.y
                + current_scale * current_momentum_value.y
                + previous_scale * previous_momentum.y
                - pressure_scale * lagged_gradient.y,
                velocity.z
                + current_scale * current_momentum_value.z
                + previous_scale * previous_momentum.z
                - pressure_scale * lagged_gradient.z,
            )
        )
        next_scalar = (
            scalar_field
            + current_scale * current_scalar_value
            + previous_scale * previous_scalar
        )
        return candidate, next_scalar

    update_kernel = jax.jit(rk_update)
    update_arguments = (
        solution.velocity,
        solution.scalar,
        solution.momentum_tendency,
        solution.scalar_tendency,
        current_momentum,
        current_scalar,
        lagged,
    )
    candidate, _ = _ready(jax, update_kernel(*update_arguments))
    projection_dt = stable_dt * (RK3_CURRENT[-1] + RK3_PREVIOUS[-1])
    projection_rhs_kernel = jax.jit(
        lambda velocity: divergence(velocity, grid) / projection_dt
    )
    pressure_solve_kernel = jax.jit(lambda rhs: poisson.solve(rhs))
    right_hand_side = _ready(jax, projection_rhs_kernel(candidate))
    correction = _ready(jax, pressure_solve_kernel(right_hand_side))

    def apply_correction(velocity, pressure):
        gradient = pressure_gradient(pressure, grid)
        return StaggeredVelocity(
            velocity.x - projection_dt * gradient.x,
            velocity.y - projection_dt * gradient.y,
            velocity.z - projection_dt * gradient.z,
        )

    correction_kernel = jax.jit(apply_correction)
    projection_kernel = jax.jit(
        lambda velocity: project(velocity, poisson, projection_dt)
    )

    component_functions: dict[str, Callable[[], Any]] = {
        "cfl_control": lambda: cfl_kernel(solution.velocity),
        "lagged_pressure_gradient": lambda: lagged_kernel(solution.pressure),
        "explicit_tendencies": lambda: explicit_kernel(
            solution.velocity, solution.scalar, solution.time
        ),
        "rk_field_update": lambda: update_kernel(*update_arguments),
        "fft_projection": lambda: projection_kernel(candidate),
    }
    decomposition: dict[str, Measurement] = {}
    for name, function in component_functions.items():
        decomposition[name], _ = _measure(
            jax,
            function,
            samples=arguments.samples,
            repeats=arguments.component_repeats,
        )

    drilldown_functions: dict[str, Callable[[], Any]] = {
        "momentum_advection": lambda: advection_kernel(solution.velocity),
        "amd_subfilter": lambda: sgs_kernel(solution.velocity),
        "combined_momentum_rhs": lambda: momentum_kernel(
            solution.velocity, solution.time
        ),
        "eddy_viscosity_for_scalar": lambda: viscosity_kernel(
            solution.velocity
        ),
        "scalar_transport": lambda: scalar_kernel(
            solution.velocity, solution.scalar
        ),
        "projection_divergence_rhs": lambda: projection_rhs_kernel(candidate),
        "projection_fft_solve": lambda: pressure_solve_kernel(right_hand_side),
        "projection_gradient_update": lambda: correction_kernel(
            candidate, correction
        ),
    }
    if wall_kernel is not None:
        drilldown_functions["wall_stress"] = lambda: wall_kernel(
            solution.velocity
        )
    drilldown: dict[str, Measurement] = {}
    for name, function in drilldown_functions.items():
        drilldown[name], _ = _measure(
            jax,
            function,
            samples=arguments.samples,
            repeats=arguments.component_repeats,
        )

    calls_per_step = {
        "cfl_control": 1,
        "lagged_pressure_gradient": 1,
        "explicit_tendencies": 3,
        "rk_field_update": 3,
        "fft_projection": 1,
    }
    full_ms = end_to_end["adaptive_workflow"].median_ms
    estimated = {
        name: value.median_ms * calls_per_step[name]
        for name, value in decomposition.items()
    }
    estimated_total = sum(estimated.values())
    environment = _environment(jax, jaxlib)
    report = {
        "schema": "jaxwind.hitsz-step-benchmark.v1",
        "environment": environment,
        "case": {
            "config": str(config_path),
            "name": case.name,
            "cells": [grid.nx, grid.ny, grid.nz],
            "dtype": case.dtype,
            "time_integration": options.time_integration,
            "maximum_dt_seconds": case.dt_seconds,
            "representative_dt_seconds": stable_dt,
            "cfl_ceiling": options.cfl_ceiling,
        },
        "settings": {
            "block_steps": arguments.block_steps,
            "setup_steps": arguments.setup_steps,
            "samples": arguments.samples,
            "component_repeats": arguments.component_repeats,
            "setup_compile_and_run_seconds": setup_seconds,
        },
        "end_to_end": {
            name: _as_report(value, unit="ms/step")
            for name, value in end_to_end.items()
        },
        "decomposition": {
            name: {
                **_as_report(value, unit="ms/call"),
                "calls_per_step": calls_per_step[name],
                "estimated_ms_per_step": estimated[name],
                "percent_of_full_step": 100.0 * estimated[name] / full_ms,
            }
            for name, value in decomposition.items()
        },
        "decomposition_estimated_total_ms_per_step": estimated_total,
        "drilldown": {
            name: _as_report(value, unit="ms/call")
            for name, value in drilldown.items()
        },
        "notes": [
            "End-to-end timings synchronize once per production-style block.",
            "Component timings synchronize each call and prevent phase fusion.",
            "Decomposition totals need not equal end-to-end time.",
            "RHS and projection drilldowns overlap their parent measurements.",
        ],
    }

    print(
        f"HITSZ warmup step benchmark: {case.name}\n"
        f"host={environment['hostname']} backend={environment['default_backend']} "
        f"jax={environment['jax_version']} jaxlib={environment['jaxlib_version']}\n"
        f"device={environment['devices']}\n"
        f"grid={grid.nx}x{grid.ny}x{grid.nz} dtype={case.dtype} "
        f"dt_max={case.dt_seconds:g}s dt_sample={stable_dt:g}s"
    )
    _print_table(
        "End-to-end (synchronized once per block)",
        [
            (
                "adaptive workflow",
                end_to_end["adaptive_workflow"].median_ms,
                100.0,
                end_to_end["adaptive_workflow"].compile_seconds,
            ),
            (
                "fixed representative dt",
                end_to_end["fixed_dt"].median_ms,
                100.0 * end_to_end["fixed_dt"].median_ms / full_ms,
                end_to_end["fixed_dt"].compile_seconds,
            ),
        ],
    )
    _print_table(
        "Synchronized phase decomposition (multiplicity applied)",
        [
            (
                f"{name} x{calls_per_step[name]}",
                estimated[name],
                100.0 * estimated[name] / full_ms,
                value.compile_seconds,
            )
            for name, value in decomposition.items()
        ]
        + [
            (
                "diagnostic total",
                estimated_total,
                100.0 * estimated_total / full_ms,
                0.0,
            )
        ],
    )
    _print_table(
        "Overlapping drilldown (one isolated call)",
        [
            (name, value.median_ms, None, value.compile_seconds)
            for name, value in drilldown.items()
        ],
    )
    print(
        "\nNote: separately synchronized component timings locate regressions; "
        "they are not an additive model of fused execution."
    )

    if arguments.json_path is not None:
        arguments.json_path.parent.mkdir(parents=True, exist_ok=True)
        arguments.json_path.write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print(f"JSON report: {arguments.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
