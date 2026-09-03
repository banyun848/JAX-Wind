"""Run and evaluate the source-driven DLR IN-1 LN2 benchmark."""

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
class DLRProtocol:
    reference: Path
    nozzle_diameter: float
    source_physical_y: float
    spinup_steps: int
    sample_every_steps: int
    axial_half_width_cells: float
    radial_half_width: float
    minimum_observations: int
    visualization_frames: int


def load_protocol(path: str | Path) -> DLRProtocol:
    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    validation = document["dlr_validation"]
    measurement = document["measurement"]
    protocol = DLRProtocol(
        reference=Path(validation["reference_profiles"]),
        nozzle_diameter=float(validation["nozzle_diameter_m"]),
        source_physical_y=float(validation["source_plane_y_over_d"])
        * float(validation["nozzle_diameter_m"]),
        spinup_steps=int(measurement["spinup_steps"]),
        sample_every_steps=int(measurement["sample_every_steps"]),
        axial_half_width_cells=float(
            measurement["axial_slab_half_width_cells"]
        ),
        radial_half_width=float(measurement["radial_bin_half_width_m"]),
        minimum_observations=int(
            measurement["minimum_observations_per_bin"]
        ),
        visualization_frames=int(document["visualization"]["frames"]),
    )
    positive = (
        protocol.nozzle_diameter,
        protocol.sample_every_steps,
        protocol.axial_half_width_cells,
        protocol.radial_half_width,
        protocol.minimum_observations,
        protocol.visualization_frames,
    )
    if (
        protocol.spinup_steps < 0
        or protocol.source_physical_y < 0.0
        or not all(value > 0 for value in positive)
    ):
        raise ValueError(
            "DLR measurement values must be positive and source distance nonnegative"
        )
    return protocol


def _optional_float(value: str) -> float:
    return float(value) if value.strip() else math.nan


def read_reference(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [
        {
            "x_D": float(row["x_D"]),
            "y_D": float(row["y_D"]),
            "role": row["role"],
            "d10_um": _optional_float(row["d10_um"]),
            "u_mean_m_s": _optional_float(row["u_mean_m_s"]),
            "v_mean_m_s": _optional_float(row["v_mean_m_s"]),
            "d10_hi_um": _optional_float(row["d10_hi_um"]),
            "d10_lo_um": _optional_float(row["d10_lo_um"]),
            "u_mean_hi_m_s": _optional_float(row["u_mean_hi_m_s"]),
            "u_mean_lo_m_s": _optional_float(row["u_mean_lo_m_s"]),
        }
        for row in rows
    ]


def profile_bins(
    reference: list[dict[str, object]],
    roles: tuple[str, ...] = ("validation",),
) -> list[tuple[float, float]]:
    """Return unique (y/D, |x|/D) annular locations for selected roles."""

    return sorted(
        {
            (float(row["y_D"]), abs(float(row["x_D"])))
            for row in reference
            if row["role"] in roles
        }
    )


def sample_sufficient_statistics(
    parcels,
    bins: list[tuple[float, float]],
    *,
    source: tuple[float, float, float],
    nozzle_diameter: float,
    source_physical_y: float,
    axial_half_width: float,
    radial_half_width: float,
) -> np.ndarray:
    """Sample event-weighted D10 and two-component parcel velocity."""

    active = np.asarray(parcels.active, dtype=bool)
    x = np.asarray(parcels.x)
    y = np.asarray(parcels.y)
    z = np.asarray(parcels.z)
    u = np.asarray(parcels.u)
    v = np.asarray(parcels.v)
    w = np.asarray(parcels.w)
    diameter = np.asarray(parcels.diameter)
    multiplicity = np.asarray(parcels.multiplicity)
    dy = y - source[1]
    dz = z - source[2]
    radius = np.sqrt(dy**2 + dz**2)
    radial_velocity = np.divide(
        v * dy + w * dz,
        radius,
        out=np.zeros_like(radius),
        where=radius > 0.0,
    )
    speed = np.sqrt(u**2 + v**2 + w**2)
    result = np.zeros((len(bins), 5), np.float64)

    for index, (y_D, x_D_radius) in enumerate(bins):
        physical_y = y_D * nozzle_diameter
        solver_x = source[0] + physical_y - source_physical_y
        target_radius = x_D_radius * nozzle_diameter
        radial_mask = (
            radius < radial_half_width
            if target_radius == 0.0
            else np.abs(radius - target_radius) < radial_half_width
        )
        mask = (
            active
            & (np.abs(x - solver_x) < axial_half_width)
            & radial_mask
        )
        count = int(np.count_nonzero(mask))
        if not count:
            continue
        # PDA reports droplet crossing events. Multiplicity times total speed
        # converts an instantaneous axisymmetric parcel population into the
        # corresponding fixed-probe event weighting without deleting reverse
        # flow, which is a key feature of the DLR chamber data.
        weight = multiplicity[mask] * speed[mask]
        result[index] = (
            np.sum(weight * diameter[mask]),
            np.sum(weight * u[mask]),
            np.sum(weight * radial_velocity[mask]),
            np.sum(weight),
            count,
        )
    return result


def profiles(statistics: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    summed = np.sum(statistics, axis=0)
    d10 = np.full(summed.shape[0], np.nan)
    axial = np.full(summed.shape[0], np.nan)
    radial = np.full(summed.shape[0], np.nan)
    valid = summed[:, 3] > 0.0
    d10[valid] = 1.0e6 * summed[valid, 0] / summed[valid, 3]
    axial[valid] = summed[valid, 1] / summed[valid, 3]
    radial[valid] = summed[valid, 2] / summed[valid, 3]
    return d10, axial, radial


def _metric(rows: list[dict[str, object]], reference: str, prediction: str):
    pairs = np.asarray(
        [
            (float(row[reference]), float(row[prediction]))
            for row in rows
            if prediction in row and math.isfinite(float(row[reference]))
        ]
    )
    if not len(pairs):
        return {"count": 0, "mae": math.nan, "rmse": math.nan, "nrmse": math.nan}
    error = pairs[:, 1] - pairs[:, 0]
    scale = max(float(np.ptp(pairs[:, 0])), np.finfo(float).tiny)
    return {
        "count": len(pairs),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nrmse": float(np.sqrt(np.mean(error**2)) / scale),
    }


def build_report(
    reference: list[dict[str, object]],
    bins: list[tuple[float, float]],
    history: np.ndarray,
) -> dict[str, object]:
    d10, axial, radial = profiles(history)
    counts = np.sum(history[:, :, 4], axis=0).astype(int)
    bin_index = {key: index for index, key in enumerate(bins)}

    def comparisons_for(role: str) -> list[dict[str, object]]:
        comparisons = []
        for row in reference:
            if row["role"] != role:
                continue
            x_D = float(row["x_D"])
            index = bin_index[(float(row["y_D"]), abs(x_D))]
            comparison = {**row, "observation_count": int(counts[index])}
            if np.isfinite(d10[index]):
                sign = -1.0 if x_D < 0.0 else (1.0 if x_D > 0.0 else 0.0)
                comparison.update(
                    simulated_d10_um=float(d10[index]),
                    simulated_u_mean_m_s=float(axial[index]),
                    simulated_v_mean_m_s=float(sign * radial[index]),
                )
            comparisons.append(comparison)
        return comparisons

    def summary(comparisons: list[dict[str, object]]) -> dict[str, object]:
        populated = [
            row for row in comparisons if "simulated_d10_um" in row
        ]
        observation_counts = [
            int(row["observation_count"]) for row in comparisons
        ]
        return {
            "comparisons": comparisons,
            "populated_signed_points": len(populated),
            "total_signed_points": len(comparisons),
            "coverage_fraction": (
                len(populated) / len(comparisons) if comparisons else math.nan
            ),
            "minimum_observation_count": (
                min(observation_counts) if observation_counts else 0
            ),
            "metrics": {
                "d10_um": _metric(
                    comparisons, "d10_um", "simulated_d10_um"
                ),
                "u_mean_m_s": _metric(
                    comparisons, "u_mean_m_s", "simulated_u_mean_m_s"
                ),
                "v_mean_m_s": _metric(
                    comparisons, "v_mean_m_s", "simulated_v_mean_m_s"
                ),
            },
        }

    validation = summary(comparisons_for("validation"))
    calibration = summary(comparisons_for("source"))
    return {**validation, "calibration_y_over_d_5": calibration}


def run(config_path: Path) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from jaxwind.fv import cell_velocity

    case = load_case(config_path)
    protocol = load_protocol(config_path)
    if protocol.spinup_steps > case.steps:
        raise ValueError("measurement spin-up must not exceed the run")
    reference = read_reference(protocol.reference)
    bins = profile_bins(reference, roles=("source", "validation"))
    grid, jet, _microphysics, state, advance, courant = build_simulation(case)
    if case.steps % protocol.visualization_frames:
        raise ValueError("run steps must be divisible by visualization frames")
    axial_half_width = protocol.axial_half_width_cells * grid.dx
    next_sample = protocol.spinup_steps
    frame_interval = case.steps // protocol.visualization_frames
    next_frame = frame_interval
    histories: list[np.ndarray] = []
    sample_times: list[float] = []
    frames: list[tuple[np.ndarray, ...]] = []
    frame_times: list[float] = []
    plane = grid.nz // 2
    last_report = 0

    @jax.jit
    def capture(current):
        u, v, w = cell_velocity(current.velocity)
        return (
            jnp.sqrt(u**2 + v**2 + w**2)[plane],
            current.temperature[plane],
            current.nitrogen[plane],
            (current.liquid_water + current.ice_water)[plane],
            u[plane],
            v[plane],
            w[plane],
        )

    started = time.perf_counter()
    while int(state.step) < case.steps:
        completed = int(state.step)
        targets = [case.steps, next_frame]
        if next_sample <= case.steps:
            targets.append(next_sample)
        target = min(value for value in targets if value > completed)
        state = advance(state, min(case.chunk_steps, target - completed))
        jax.block_until_ready(state.velocity.x)
        completed = int(state.step)

        if completed == next_sample:
            histories.append(
                sample_sufficient_statistics(
                    jax.device_get(state.parcels),
                    bins,
                    source=case.nozzle,
                    nozzle_diameter=protocol.nozzle_diameter,
                    source_physical_y=protocol.source_physical_y,
                    axial_half_width=axial_half_width,
                    radial_half_width=protocol.radial_half_width,
                )
            )
            sample_times.append(float(state.time))
            next_sample += protocol.sample_every_steps

        if completed == next_frame:
            captured = jax.device_get(capture(state))
            frames.append(tuple(np.asarray(value) for value in captured))
            frame_times.append(float(state.time))
            next_frame += frame_interval

        if completed - last_report >= max(1000, case.chunk_steps):
            print(
                f"DLR IN-1 {completed:7d}/{case.steps} "
                f"t={float(state.time):.5f}s "
                f"CFL={float(courant(state.velocity, grid, case.dt)):.3f} "
                f"samples={len(histories)} frames={len(frames)}",
                flush=True,
            )
            last_report = completed

    elapsed = time.perf_counter() - started
    history = np.asarray(histories)
    evaluation = build_report(reference, bins, history)
    result = {
        "benchmark": "DLR IN-1 source-driven held-out wake validation",
        "config": str(config_path),
        "reference": str(protocol.reference),
        "source_plane": "measured y/D=5 profile",
        "validation_planes_y_over_d": sorted(
            {float(row["y_D"]) for row in reference if row["role"] == "validation"}
        ),
        "axis_convention": (
            "solver +x is physical downstream/downward y; signed "
            "experimental x maps to radial velocity"
        ),
        "weighting": (
            "parcel multiplicity times total crossing speed; reverse flow retained"
        ),
        "model_scope": (
            "injector geometry and primary breakup replaced by measured PDA "
            "source; downstream planes held out"
        ),
        "phase_limit": (
            "pressure is below the nitrogen triple point; effective "
            "triple-point-capped liquid closure has no solid-N2 phase"
        ),
        "steps": case.steps,
        "simulated_seconds": float(state.time),
        "elapsed_seconds": elapsed,
        "steps_per_second": case.steps / elapsed,
        "sample_times_seconds": sample_times,
        "minimum_observation_target": protocol.minimum_observations,
        **evaluation,
    }
    result["acceptance"] = {
        "complete_coverage": evaluation["populated_signed_points"]
        == evaluation["total_signed_points"],
        "enough_observations": evaluation["minimum_observation_count"]
        >= protocol.minimum_observations,
        "d10_mae_within_3_3_um": evaluation["metrics"]["d10_um"]["mae"] <= 3.3,
        "u_mae_within_6_1_m_s": evaluation["metrics"]["u_mean_m_s"]["mae"] <= 6.1,
        "v_mae_within_8_0_m_s": evaluation["metrics"]["v_mean_m_s"]["mae"] <= 8.0,
    }

    case.output.mkdir(parents=True, exist_ok=True)
    if len(frames) != protocol.visualization_frames:
        raise RuntimeError("visualization frame capture count is incomplete")
    frame_arrays = tuple(
        np.stack([frame[i] for frame in frames])
        for i in range(len(frames[0]))
    )
    np.savez_compressed(
        case.output / "visualization_frames_100.npz",
        velocity_magnitude=frame_arrays[0],
        temperature=frame_arrays[1],
        nitrogen=frame_arrays[2],
        fog_mass_fraction=frame_arrays[3],
        axial_velocity=frame_arrays[4],
        side_velocity=frame_arrays[5],
        vertical_velocity=frame_arrays[6],
        time_seconds=np.asarray(frame_times),
        lengths_m=np.asarray(case.lengths),
        source_m=np.asarray(case.nozzle),
        physical_x_offset_m=protocol.source_physical_y - case.nozzle[0],
    )
    np.savez_compressed(
        case.output / "measurement_history.npz",
        statistics=history,
        sample_times_seconds=np.asarray(sample_times),
        y_D=np.asarray([item[0] for item in bins]),
        absolute_x_D=np.asarray([item[1] for item in bins]),
    )
    _save_checkpoint(
        case.output / f"checkpoint_{case.steps:08d}.npz", state, grid, jet
    )
    (case.output / "benchmark_validation.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("cases/DLRIN1/fv_256_source.toml"),
    )
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config), indent=2))


if __name__ == "__main__":
    main()
