"""Run Purdue D1 LN2 spray with experiment-matched time sampling."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time

import numpy as np

from applications.fv_ln2_jet.run import (
    _save_checkpoint,
    build_simulation,
    load_case,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class MeasurementProtocol:
    reference: Path
    source_physical_x: float
    spinup_steps: int
    sample_every_steps: int
    axial_half_width_cells: float
    radial_half_width: float
    minimum_observations: int
    convergence_blocks: int
    convergence_tolerance: float
    visualization_frames: int
    smd_uncertainty: float
    velocity_uncertainty: float


def load_protocol(path: str | Path) -> MeasurementProtocol:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    validation = document["purdue_validation"]
    measurement = document["measurement"]
    visualization = document["visualization"]
    protocol = MeasurementProtocol(
        reference=Path(validation["reference_profiles"]),
        source_physical_x=float(
            validation["physical_source_plane_from_nozzle_m"]
        ),
        spinup_steps=int(measurement["spinup_steps"]),
        sample_every_steps=int(measurement["sample_every_steps"]),
        axial_half_width_cells=float(
            measurement["axial_slab_half_width_cells"]
        ),
        radial_half_width=float(measurement["radial_bin_half_width_m"]),
        minimum_observations=int(
            measurement["minimum_observations_per_bin"]
        ),
        convergence_blocks=int(measurement["convergence_blocks"]),
        convergence_tolerance=float(
            measurement["convergence_tolerance_relative"]
        ),
        visualization_frames=int(visualization["frames"]),
        smd_uncertainty=float(
            validation["droplet_diameter_uncertainty"]
        ),
        velocity_uncertainty=float(
            validation["droplet_velocity_uncertainty"]
        ),
    )
    positive = (
        protocol.sample_every_steps,
        protocol.axial_half_width_cells,
        protocol.radial_half_width,
        protocol.minimum_observations,
        protocol.convergence_blocks,
        protocol.convergence_tolerance,
        protocol.visualization_frames,
    )
    if protocol.spinup_steps < 0 or not all(value > 0 for value in positive):
        raise ValueError("measurement sampling values must be positive")
    return protocol


def _read_reference(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            "physical_x_m": float(row["physical_x_m"]),
            "lateral_m": float(row["lateral_m"]),
            "smd_um": float(row["smd_um"]),
            "velocity_m_s": float(row["mean_axial_velocity_m_s"]),
            "role": row["role"],
        }
        for row in rows
    ]


def _profile_bins(
    reference: list[dict[str, object]],
    roles: tuple[str, ...] = ("validation",),
) -> list[tuple[float, float]]:
    return sorted(
        {
            (
                float(row["physical_x_m"]),
                abs(float(row["lateral_m"])),
            )
            for row in reference
            if row["role"] in roles
        }
    )


def _sample_sufficient_statistics(
    parcels,
    bins: list[tuple[float, float]],
    *,
    nozzle: tuple[float, float, float],
    source_physical_x: float,
    axial_half_width: float,
    radial_half_width: float,
) -> np.ndarray:
    active = np.asarray(parcels.active, dtype=bool)
    x = np.asarray(parcels.x)
    y = np.asarray(parcels.y)
    z = np.asarray(parcels.z)
    u = np.asarray(parcels.u)
    diameter = np.asarray(parcels.diameter)
    multiplicity = np.asarray(parcels.multiplicity)
    radius = np.sqrt((y - nozzle[1]) ** 2 + (z - nozzle[2]) ** 2)
    result = np.zeros((len(bins), 5), np.float64)

    for index, (physical_x, target_radius) in enumerate(bins):
        solver_x = nozzle[0] + physical_x - source_physical_x
        radial_mask = (
            radius < radial_half_width
            if target_radius == 0.0
            else np.abs(radius - target_radius) < radial_half_width
        )
        mask = (
            active
            & (np.abs(x - solver_x) < axial_half_width)
            & radial_mask
            & (u > 0.0)
        )
        count = int(np.count_nonzero(mask))
        if not count:
            continue
        # Flux weighting converts instantaneous parcel occupancy into the
        # crossing-event statistics observed by a fixed PDPA probe volume.
        weight = multiplicity[mask] * u[mask]
        sample_diameter = diameter[mask]
        result[index] = (
            np.sum(weight * sample_diameter**3),
            np.sum(weight * sample_diameter**2),
            np.sum(weight * u[mask]),
            np.sum(weight),
            count,
        )
    return result


def _profiles(statistics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    summed = np.sum(statistics, axis=0)
    smd = np.full(summed.shape[0], np.nan)
    velocity = np.full(summed.shape[0], np.nan)
    valid_diameter = summed[:, 1] > 0.0
    valid_velocity = summed[:, 3] > 0.0
    smd[valid_diameter] = (
        1.0e6
        * summed[valid_diameter, 0]
        / summed[valid_diameter, 1]
    )
    velocity[valid_velocity] = (
        summed[valid_velocity, 2] / summed[valid_velocity, 3]
    )
    return smd, velocity


def _convergence(
    history: np.ndarray,
    block_count: int,
) -> dict[str, object]:
    if len(history) < block_count:
        return {
            "blocks": 0,
            "maximum_last_block_relative_change": math.inf,
        }
    blocks = [
        _profiles(history[indexes])
        for indexes in np.array_split(np.arange(len(history)), block_count)
    ]
    previous_smd, previous_velocity = blocks[-2]
    final_smd, final_velocity = blocks[-1]
    changes = []
    for previous, final in (
        (previous_smd, final_smd),
        (previous_velocity, final_velocity),
    ):
        valid = np.isfinite(previous) & np.isfinite(final)
        changes.extend(
            np.abs(final[valid] - previous[valid])
            / np.maximum(np.abs(final[valid]), np.finfo(float).tiny)
        )
    return {
        "blocks": block_count,
        "maximum_last_block_relative_change": (
            float(np.max(changes)) if changes else math.inf
        ),
        "last_block_smd_um": final_smd.tolist(),
        "last_block_velocity_m_s": final_velocity.tolist(),
    }



def _comparisons_for_role(
    reference: list[dict[str, object]],
    role: str,
    bins: list[tuple[float, float]],
    total_counts: np.ndarray,
    smd: np.ndarray,
    velocity: np.ndarray,
) -> list[dict[str, object]]:
    """Compare one role without leaking calibration into validation."""
    bin_index = {key: index for index, key in enumerate(bins)}
    comparisons = []
    for row in reference:
        if row["role"] != role:
            continue
        key = (
            float(row["physical_x_m"]),
            abs(float(row["lateral_m"])),
        )
        index = bin_index[key]
        comparison = {
            **row,
            "radial_m": key[1],
            "observation_count": int(total_counts[index]),
        }
        if np.isfinite(smd[index]) and np.isfinite(velocity[index]):
            comparison.update(
                simulated_smd_um=float(smd[index]),
                simulated_mean_axial_velocity_m_s=float(velocity[index]),
                smd_relative_error=float(
                    smd[index] / float(row["smd_um"]) - 1.0
                ),
                velocity_relative_error=float(
                    velocity[index] / float(row["velocity_m_s"]) - 1.0
                ),
            )
        comparisons.append(comparison)
    return comparisons


def _error_summary(
    comparisons: list[dict[str, object]],
) -> dict[str, object]:
    populated = [
        row for row in comparisons if "smd_relative_error" in row
    ]
    return {
        "comparisons": comparisons,
        "populated_signed_points": len(populated),
        "total_signed_points": len(comparisons),
        "coverage_fraction": (
            len(populated) / len(comparisons) if comparisons else 0.0
        ),
        "mean_absolute_smd_relative_error": (
            float(
                np.mean(
                    [abs(row["smd_relative_error"]) for row in populated]
                )
            )
            if populated
            else math.inf
        ),
        "mean_absolute_velocity_relative_error": (
            float(
                np.mean(
                    [
                        abs(row["velocity_relative_error"])
                        for row in populated
                    ]
                )
            )
            if populated
            else math.inf
        ),
    }


def run(config_path: Path) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from jaxwind.fv import cell_velocity

    case = load_case(config_path)
    protocol = load_protocol(config_path)
    if protocol.spinup_steps >= case.steps:
        raise ValueError("measurement spin-up must end before the run")
    reference = _read_reference(protocol.reference)
    bins = _profile_bins(reference, roles=("source", "validation"))
    grid, _jet, _microphysics, state, advance, courant = build_simulation(case)
    if case.steps % protocol.visualization_frames != 0:
        raise ValueError("run steps must be divisible by visualization frames")
    axial_half_width = protocol.axial_half_width_cells * grid.dx
    statistics_history = []
    sample_times = []
    next_sample = protocol.spinup_steps + protocol.sample_every_steps
    frame_interval = case.steps // protocol.visualization_frames
    next_frame = frame_interval
    frame_times = []
    velocity_frames = []
    temperature_frames = []
    nitrogen_frames = []
    fog_frames = []
    plane_index = grid.nz // 2

    @jax.jit
    def capture_frame(current):
        u, v, w = cell_velocity(current.velocity)
        speed = jnp.sqrt(u**2 + v**2 + w**2)
        return (
            speed[plane_index],
            current.temperature[plane_index],
            current.nitrogen[plane_index],
            (
                current.liquid_water[plane_index]
                + current.ice_water[plane_index]
            ),
        )
    started = time.perf_counter()

    while int(state.step) < case.steps:
        completed = int(state.step)
        target = min(case.steps, next_sample, next_frame)
        count = min(case.chunk_steps, target - completed)
        if count <= 0:
            count = min(case.chunk_steps, case.steps - completed)
        state = advance(state, count)
        jax.block_until_ready(state.velocity.x)
        completed = int(state.step)

        if completed == next_sample:
            host_parcels = jax.device_get(state.parcels)
            statistics_history.append(
                _sample_sufficient_statistics(
                    host_parcels,
                    bins,
                    nozzle=case.nozzle,
                    source_physical_x=protocol.source_physical_x,
                    axial_half_width=axial_half_width,
                    radial_half_width=protocol.radial_half_width,
                )
            )
            sample_times.append(float(state.time))
            next_sample += protocol.sample_every_steps

        if completed == next_frame:
            captured = capture_frame(state)
            jax.block_until_ready(captured)
            velocity_frames.append(np.asarray(captured[0]))
            temperature_frames.append(np.asarray(captured[1]))
            nitrogen_frames.append(np.asarray(captured[2]))
            fog_frames.append(np.asarray(captured[3]))
            frame_times.append(float(state.time))
            next_frame += frame_interval

        if completed % max(200, case.chunk_steps) == 0:
            print(
                f"Purdue {completed:6d}/{case.steps} "
                f"time={float(state.time):.5f}s "
                f"CFL={float(courant(state.velocity, grid, case.dt)):.3f} "
                f"samples={len(statistics_history)} "
                f"frames={len(frame_times)}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    history = np.asarray(statistics_history)
    smd, velocity = _profiles(history)
    total_counts = np.sum(history[:, :, 4], axis=0).astype(int)
    validation_comparisons = _comparisons_for_role(
        reference, "validation", bins, total_counts, smd, velocity
    )
    calibration_comparisons = _comparisons_for_role(
        reference, "source", bins, total_counts, smd, velocity
    )
    validation = _error_summary(validation_comparisons)
    calibration_summary = _error_summary(calibration_comparisons)
    convergence = _convergence(history, protocol.convergence_blocks)
    validation_bins = set(_profile_bins(reference))
    validation_counts = [
        total_counts[index]
        for index, key in enumerate(bins)
        if key in validation_bins
    ]
    minimum_count = (
        int(np.min(validation_counts)) if validation_counts else 0
    )
    result = {
        "config": str(config_path),
        "reference": str(protocol.reference),
        "ambient": {
            "temperature_k": case.ambient_temperature,
            "relative_humidity": case.ambient_relative_humidity,
            "water_vapor_mixing_ratio": case.ambient_water_vapor,
            "pressure_pa": case.pressure,
            "moist_air_density_kg_m3": case.ambient_density,
            "dry_air_density_kg_m3": case.dry_air_density,
        },
        "steps": case.steps,
        "flow_formulation": case.flow_formulation,
        "momentum_closure": case.momentum_closure,
        "subgrid_boundary_jet": {
            "enabled": case.subgrid_jet_enabled,
            "support_radius_cells": case.subgrid_support_radius_cells,
            "transition_width_cells": case.subgrid_transition_width_cells,
            "momentum_length_cells": case.subgrid_momentum_length_cells,
            "turbulence_intensity": case.subgrid_turbulence_intensity,
            "integral_scale_cells": case.subgrid_integral_scale_cells,
            "correlation_time_seconds": case.subgrid_correlation_time,
            "transverse_ratio": case.subgrid_transverse_ratio,
        },
        "simulated_seconds": float(state.time),
        "elapsed_seconds": elapsed,
        "steps_per_second": case.steps / elapsed,
        "sample_count": len(history),
        "sample_times_seconds": sample_times,
        "visualization_frame_count": len(frame_times),
        "visualization_frames": str(
            case.output / "visualization_frames_100.npz"
        ),
        "axis_convention": "+x is vertically downward",
        "weighting": "positive axial parcel-crossing flux",
        "profile_bins": [
            {
                "physical_x_m": physical_x,
                "radial_m": radial,
                "observation_count": int(total_counts[index]),
            }
            for index, (physical_x, radial) in enumerate(bins)
        ],
        **validation,
        "calibration_first_section": calibration_summary,
        "minimum_observation_count": minimum_count,
        "minimum_observation_target": protocol.minimum_observations,
        "convergence": convergence,
        "acceptance": {
            "complete_coverage": bool(
                validation["populated_signed_points"]
                == validation["total_signed_points"]
            ),
            "enough_observations": bool(
                minimum_count >= protocol.minimum_observations
            ),
            "sampling_converged": bool(
                convergence["maximum_last_block_relative_change"]
                <= protocol.convergence_tolerance
            ),
            "smd_within_experimental_uncertainty": bool(
                validation["mean_absolute_smd_relative_error"]
                <= protocol.smd_uncertainty
            ),
            "velocity_within_experimental_uncertainty": bool(
                validation["mean_absolute_velocity_relative_error"]
                <= protocol.velocity_uncertainty
            ),
        },
    }

    case.output.mkdir(parents=True, exist_ok=True)
    if len(frame_times) != protocol.visualization_frames:
        raise RuntimeError("visualization frame capture count is incomplete")
    np.savez_compressed(
        case.output / "visualization_frames_100.npz",
        velocity_magnitude=np.stack(velocity_frames),
        temperature=np.stack(temperature_frames),
        nitrogen=np.stack(nitrogen_frames),
        fog_mass_fraction=np.stack(fog_frames),
        time_seconds=np.asarray(frame_times),
        lengths_m=np.asarray(case.lengths),
        source_m=np.asarray(case.nozzle),
        physical_x_offset_m=(
            protocol.source_physical_x - case.nozzle[0]
        ),
    )
    np.savez_compressed(
        case.output / "measurement_history.npz",
        statistics=history,
        sample_times_seconds=np.asarray(sample_times),
        physical_x_m=np.asarray([item[0] for item in bins]),
        radial_m=np.asarray([item[1] for item in bins]),
    )
    _save_checkpoint(
        case.output / f"checkpoint_{case.steps:08d}.npz",
        state,
        grid,
        _jet,
    )
    (case.output / "experiment_validation.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=2))


if __name__ == "__main__":
    main()
