"""Periodic execution services."""
from __future__ import annotations
from datetime import datetime, timedelta
import time
import numpy as np

def _estimated_finish(
    started: float,
    completed: float,
    total: float,
) -> str:
    """Format a cumulative-throughput estimate of remaining wall time."""
    elapsed = max(time.perf_counter() - started, 0.0)
    progress = min(max(completed, 0.0), total)
    remaining = 0.0 if progress >= total else elapsed * (total - progress) / progress
    rounded = max(0, round(remaining))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    finish = datetime.now().astimezone() + timedelta(seconds=remaining)
    return (
        f"finish {finish.isoformat(timespec='seconds')} "
        f"remaining {hours:d}:{minutes:02d}:{seconds:02d}"
    )


@dataclass(frozen=True, slots=True)
def run_periodic_blocks(
    solution,
    advance,
    *,
    grid,
    dt: float,
    steps: int,
    chunk: int,
):
    import jax
    import jax.numpy as jnp
    from jaxwind import courant_number

    initial_time = float(solution.time)
    completed = 0
    started = time.perf_counter()
    maximum_cfl = 0.0
    final_cfl = 0.0
    while completed < steps:
        count = min(chunk, steps - completed)
        solution = advance(solution, dt, count)
        jax.block_until_ready(solution.velocity.x)
        completed += count
        # Avoid O(steps * eps) drift from repeatedly accumulating a float32 dt.
        solution = solution._replace(
            time=jnp.asarray(
                initial_time + completed * dt,
                solution.time.dtype,
            )
        )
        final_cfl = float(courant_number(solution.velocity, grid, dt))
        maximum_cfl = max(maximum_cfl, final_cfl)
        print(
            f"periodic {completed:8d}/{steps} CFL {final_cfl:.3f} "
            f"{_estimated_finish(started, completed, steps)}",
            flush=True,
        )
    return solution, time.perf_counter() - started, final_cfl, maximum_cfl


def run_adaptive_periodic_blocks(
    solution,
    advance,
    *,
    grid,
    maximum_dt: float,
    cfl_ceiling: float,
    duration_seconds: float,
    chunk: int,
):
    import jax
    import jax.numpy as jnp
    from jaxwind import courant_number, stable_timestep

    initial_time = float(solution.time)
    target_time = initial_time + duration_seconds
    started = time.perf_counter()
    initial_step = int(solution.step)
    maximum_sampled_cfl = 0.0
    minimum_block_dt = float("inf")
    maximum_block_dt = 0.0
    tolerance = 8.0 * np.finfo(np.float32).eps * max(1.0, target_time)
    next_dt = maximum_dt
    next_cfl = 0.0
    while float(solution.time) < target_time - tolerance:
        before_time = float(solution.time)
        before_step = int(solution.step)
        solution = advance(solution, target_time, chunk)
        jax.block_until_ready(solution.velocity.x)
        after_time = float(solution.time)
        after_step = int(solution.step)
        active_steps = after_step - before_step
        if active_steps <= 0 or after_time <= before_time:
            raise RuntimeError("adaptive RK block made no progress")
        block_dt = (after_time - before_time) / active_steps
        minimum_block_dt = min(minimum_block_dt, block_dt)
        maximum_block_dt = max(maximum_block_dt, block_dt)
        next_dt = float(
            jnp.minimum(
                maximum_dt,
                stable_timestep(
                    solution.velocity,
                    grid,
                    0.0,
                    courant=cfl_ceiling,
                ),
            )
        )
        next_cfl = float(courant_number(solution.velocity, grid, next_dt))
        maximum_sampled_cfl = max(maximum_sampled_cfl, next_cfl)
        finish = _estimated_finish(
            started,
            after_time - initial_time,
            duration_seconds,
        )
        print(
            f"periodic step {after_step:8d} "
            f"time {after_time:9.3f}/{target_time:.3f} s "
            f"dt {next_dt:.6f} CFL {next_cfl:.3f} {finish}",
            flush=True,
        )
    statistics = {
        "steps": int(solution.step) - initial_step,
        "duration_seconds": float(solution.time) - initial_time,
        "minimum_block_mean_dt_seconds": minimum_block_dt,
        "maximum_block_mean_dt_seconds": maximum_block_dt,
        "final_candidate_dt_seconds": next_dt,
        "final_candidate_cfl": next_cfl,
        "maximum_sampled_candidate_cfl": maximum_sampled_cfl,
    }
    return solution, time.perf_counter() - started, statistics
