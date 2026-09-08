#!/usr/bin/env python3
"""Compare steady open FV main-step variants on an existing precursor."""

from __future__ import annotations

import argparse
import dataclasses
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--steps", type=int, default=300)
    arguments = parser.parse_args()

    import jax

    from jaxwind.config.stages import load_workflow
    from jaxwind.simulation.open_atmospheric import build_open_components
    from jaxwind.io.state_fields import atmospheric_state
    from jaxwind.io.recording import InflowReader

    workflow = load_workflow(arguments.config)
    if arguments.steps <= 0:
        raise ValueError("steps must be positive")
    grid = workflow.case.physical.physical_grid
    inputs = workflow.options.input_directory or workflow.options.output_directory
    factor = workflow.options.main_substeps_per_inflow
    stage_dt = workflow.options.main_dt_seconds or workflow.options.precursor_dt_seconds or workflow.case.physical.dt_seconds
    dt = stage_dt / factor
    reader = InflowReader(inputs / "precursor/inflow", grid, samples=(arguments.steps + factor - 1) // factor, dt=stage_dt)
    warm = atmospheric_state(inputs / "warmup/checkpoint.npz", grid)
    first = reader.read(0, 1)
    first = type(first)(*(field[0] for field in first))
    variants = (
        ("full", workflow),
        ("without_turbine", dataclasses.replace(workflow, turbine=None, cooling=None)),
    )
    for name, variant in variants:
        started = time.perf_counter()
        state, advance = build_open_components(variant, warm, first)
        block_seconds, block_steps = [], []
        completed = 0
        while completed < arguments.steps:
            count = min(variant.options.chunk_steps * factor, arguments.steps - completed)
            planes = reader.read(completed // factor, (completed + count + factor - 1) // factor)
            planes = type(planes)(*(jax.numpy.repeat(field, factor, axis=0)[completed % factor:completed % factor + count] for field in planes))
            block_start = time.perf_counter()
            state = advance(state, dt, planes)
            jax.block_until_ready(state)
            block_seconds.append(time.perf_counter() - block_start)
            block_steps.append(count)
            completed += count
        elapsed = time.perf_counter() - started
        steady = sum(block_steps[1:]) / sum(block_seconds[1:]) if len(block_seconds) > 1 else None
        jax.clear_caches()
        print(
            f"{name}: advancement={sum(block_seconds):.6f}s "
            f"total={elapsed:.6f}s "
            f"steps_per_second={arguments.steps / sum(block_seconds):.3f} "
            f"steady={steady}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
