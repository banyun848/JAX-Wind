"""Calibrate the Purdue inlet through the full differentiable FV model.

Only the 38.1 mm PDPA section enters the objective. Batched forward-mode
JVPs use tangent-reset multiple shooting; downstream stations remain blind tests.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import re
from pathlib import Path
import time

import numpy as np

from applications.fv_ln2_jet.run import (
    DifferentiableInletControl,
    build_simulation,
    load_case,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


_CONTROL_NAMES = (
    "gas_inlet_radius_m",
    "initial_diameter_m",
    "speed_scale",
    "edge_speed_ratio",
    "edge_diameter_ratio",
)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    iterations: int
    damping: float
    sensitivity_batch_size: int
    sensitivity_window_steps: int
    window_sample_start_steps: int
    target_x: float
    axial_sigma: float
    radial_sigma: float
    bounds_low: np.ndarray
    bounds_high: np.ndarray
    prior_fraction: np.ndarray
    reference: Path
    spinup_steps: int
    sample_every_steps: int


def _load_calibration(path: Path) -> CalibrationConfig:
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    table = document["calibration"]
    validation = document["purdue_validation"]
    measurement = document["measurement"]
    bound_keys = (
        "gas_inlet_radius_bounds_m",
        "initial_diameter_bounds_m",
        "speed_scale_bounds",
        "edge_speed_ratio_bounds",
        "edge_diameter_ratio_bounds",
    )
    bounds = np.asarray([table[key] for key in bound_keys], np.float64)
    result = CalibrationConfig(
        iterations=int(table["iterations"]),
        damping=float(table["damping"]),
        sensitivity_batch_size=int(
            table.get("sensitivity_batch_size", 1)
        ),
        sensitivity_window_steps=int(
            table["sensitivity_window_steps"]
        ),
        window_sample_start_steps=int(
            table["window_sample_start_steps"]
        ),
        target_x=float(table["target_plane_from_nozzle_m"]),
        axial_sigma=float(table["probe_axial_sigma_m"]),
        radial_sigma=float(table["probe_radial_sigma_m"]),
        bounds_low=bounds[:, 0],
        bounds_high=bounds[:, 1],
        prior_fraction=np.asarray(
            table["prior_standard_deviation_fraction"], np.float64
        ),
        reference=Path(validation["reference_profiles"]),
        spinup_steps=int(measurement["spinup_steps"]),
        sample_every_steps=int(measurement["sample_every_steps"]),
    )
    if (
        result.iterations < 0
        or result.damping <= 0.0
        or result.sensitivity_batch_size <= 0
        or result.sensitivity_window_steps <= 0
        or result.window_sample_start_steps < 0
        or result.window_sample_start_steps
        >= result.sensitivity_window_steps
    ):
        raise ValueError("calibration iterations and damping are invalid")
    if not np.all(result.bounds_high > result.bounds_low):
        raise ValueError("each calibration upper bound must exceed its lower bound")
    if np.any(result.prior_fraction <= 0.0):
        raise ValueError("calibration prior scales must be positive")
    return result


def _source_targets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream) if row["role"] == "source"
        ]
    radii = sorted({abs(float(row["lateral_m"])) for row in rows})
    if len(radii) != 2 or radii[0] != 0.0:
        raise ValueError("calibration expects axis and one outer radius")
    diameter = []
    velocity = []
    for radius in radii:
        selected = [
            row
            for row in rows
            if abs(abs(float(row["lateral_m"])) - radius) < 1.0e-12
        ]
        diameter.append(
            np.mean([float(row["smd_um"]) for row in selected])
        )
        velocity.append(
            np.mean(
                [float(row["mean_axial_velocity_m_s"]) for row in selected]
            )
        )
    return np.asarray(radii), np.asarray(diameter + velocity)


def _logit(value):
    clipped = np.clip(value, 1.0e-5, 1.0 - 1.0e-5)
    return np.log(clipped / (1.0 - clipped))


def build_objective(config_path: Path):
    import jax
    import jax.numpy as jnp

    case = load_case(config_path)
    calibration = _load_calibration(config_path)
    if case.source_mode != "inflow" or case.nozzle[0] != 0.0:
        raise ValueError("calibration requires a nozzle on the inlet")
    if calibration.spinup_steps >= case.steps:
        raise ValueError("calibration spin-up must end before the run")
    sampling_steps = case.steps - calibration.spinup_steps
    if sampling_steps % calibration.sensitivity_window_steps:
        raise ValueError(
            "post-spin-up steps must form complete sensitivity windows"
        )
    sampling_part = (
        calibration.sensitivity_window_steps
        - calibration.window_sample_start_steps
    )
    if sampling_part % calibration.sample_every_steps:
        raise ValueError(
            "the sampled part of each window must form complete samples"
        )
    window_count = (
        sampling_steps // calibration.sensitivity_window_steps
    )
    samples_per_window = (
        sampling_part // calibration.sample_every_steps
    )
    radii, target = _source_targets(calibration.reference)
    target_scale = target * np.asarray([0.10, 0.10, 0.15, 0.15])
    (
        grid,
        _jet,
        _microphysics,
        initialize,
        advance,
        _courant,
        base_control,
    ) = build_simulation(case, differentiable_inlet=True)
    lower = jnp.asarray(calibration.bounds_low, jnp.float32)
    upper = jnp.asarray(calibration.bounds_high, jnp.float32)
    prior = jnp.asarray(
        [
            base_control.gas_radius,
            base_control.initial_diameter,
            base_control.speed_scale,
            base_control.edge_speed_ratio,
            base_control.edge_diameter_ratio,
        ],
        jnp.float32,
    )
    prior_sigma = (
        jnp.asarray(calibration.prior_fraction, jnp.float32) * prior
    )
    target_jax = jnp.asarray(target, jnp.float32)
    target_scale_jax = jnp.asarray(target_scale, jnp.float32)
    target_radii = jnp.asarray(radii, jnp.float32)

    def physical_parameters(theta):
        return lower + (upper - lower) * jax.nn.sigmoid(theta)

    def inlet_control(theta):
        physical = physical_parameters(theta)
        return DifferentiableInletControl(
            physical[0],
            base_control.gas_temperature,
            physical[1],
            physical[2],
            physical[3],
            physical[4],
        )

    def soft_statistics(state):
        parcels = state.parcels
        dtype = parcels.x.dtype
        radius = jnp.sqrt(
            (parcels.y - case.nozzle[1]) ** 2
            + (parcels.z - case.nozzle[2]) ** 2
            + jnp.asarray(1.0e-20, dtype)
        )
        axial = jnp.exp(
            -0.5
            * ((parcels.x - calibration.target_x) / calibration.axial_sigma)
            ** 2
        )
        radial = jnp.exp(
            -0.5
            * (
                (radius[None, :] - target_radii[:, None])
                / calibration.radial_sigma
            )
            ** 2
        )
        positive_u = 0.5 * (
            parcels.u + jnp.sqrt(parcels.u**2 + 1.0e-6)
        )
        weight = (
            parcels.active.astype(dtype)[None, :]
            * parcels.multiplicity[None, :]
            * positive_u[None, :]
            * axial[None, :]
            * radial
        )
        diameter = parcels.diameter[None, :]
        return jnp.stack(
            (
                jnp.sum(weight * diameter**3, axis=1),
                jnp.sum(weight * diameter**2, axis=1),
                jnp.sum(weight * parcels.u[None, :], axis=1),
                jnp.sum(weight, axis=1),
            ),
            axis=1,
        )

    def predictions(theta):
        control = inlet_control(theta)
        state = initialize(control)
        state = advance(state, control, calibration.spinup_steps)
        state = jax.tree.map(jax.lax.stop_gradient, state)

        def sample(current, _):
            current = advance(
                current, control, calibration.sample_every_steps
            )
            return current, soft_statistics(current)

        def sensitivity_window(current, _):
            current = jax.tree.map(jax.lax.stop_gradient, current)
            current = advance(
                current,
                control,
                calibration.window_sample_start_steps,
            )
            current, statistics = jax.lax.scan(
                sample,
                current,
                None,
                length=samples_per_window,
            )
            return current, jnp.sum(statistics, axis=0)

        _, windows = jax.lax.scan(
            sensitivity_window, state, None, length=window_count
        )
        summed = jnp.sum(windows, axis=0)
        tiny = jnp.finfo(summed.dtype).tiny
        smd = (
            1.0e6
            * summed[:, 0]
            / jnp.maximum(summed[:, 1], tiny)
        )
        velocity = summed[:, 2] / jnp.maximum(summed[:, 3], tiny)
        return jnp.concatenate((smd, velocity))

    def residuals(theta):
        physical = physical_parameters(theta)
        data_residual = (
            predictions(theta) - target_jax
        ) / target_scale_jax
        prior_residual = (physical - prior) / prior_sigma
        return jnp.concatenate((data_residual, prior_residual))

    initial_fraction = (prior - lower) / (upper - lower)
    initial_theta = jnp.asarray(
        _logit(np.asarray(initial_fraction)), jnp.float32
    )
    metadata = {
        "case": case,
        "calibration": calibration,
        "target_radii_m": radii,
        "target": target,
        "target_scale": target_scale,
        "physical_parameters": jax.jit(physical_parameters),
        "initialize": initialize,
        "advance": advance,
        "base_control": base_control,
    }
    return jax.jit(residuals), initial_theta, metadata


def _write_tuned_config(
    source: Path,
    destination: Path,
    controls: dict[str, float],
    base_speed: float,
    validation_output: str,
) -> None:
    text = source.read_text(encoding="utf-8")
    replacements = (
        (
            r"^name = .*$",
            "name = \"Purdue D1 differentiably tuned blind validation\"",
        ),
        (
            r"^gas_inlet_radius_m = .*$",
            f"gas_inlet_radius_m = {controls['gas_inlet_radius_m']:.12g}",
        ),
        (
            r"^speed_m_s = .*$",
            f"speed_m_s = {base_speed * controls['speed_scale']:.12g}",
        ),
        (
            r"^initial_diameter_m = .*$",
            f"initial_diameter_m = {controls['initial_diameter_m']:.12g}",
        ),
        (
            r"^edge_speed_ratio = .*$",
            f"edge_speed_ratio = {controls['edge_speed_ratio']:.12g}",
        ),
        (
            r"^edge_diameter_ratio = .*$",
            f"edge_diameter_ratio = {controls['edge_diameter_ratio']:.12g}",
        ),
        (
            r"^directory = .*$",
            f'directory = "{validation_output}"',
        ),
    )
    for pattern, replacement in replacements:
        text, count = re.subn(
            pattern, replacement, text, count=1, flags=re.MULTILINE
        )
        if count != 1:
            raise ValueError(f"missing tuned-config field: {pattern}")
    destination.write_text(text, encoding="utf-8")


def calibrate(
    config_path: Path, iterations: int | None = None
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    residual_function, theta, metadata = build_objective(config_path)
    calibration = metadata["calibration"]
    iteration_count = (
        calibration.iterations if iterations is None else iterations
    )
    damping = calibration.damping
    history = []
    started = time.perf_counter()

    def tangent_batch(current_theta, tangent_vectors):
        return jax.vmap(
            lambda tangent: jax.jvp(
                residual_function,
                (current_theta,),
                (tangent,),
            )[1]
        )(tangent_vectors)

    tangent_batch = jax.jit(tangent_batch)

    for iteration in range(iteration_count + 1):
        residual = np.asarray(residual_function(theta))
        physical = np.asarray(metadata["physical_parameters"](theta))
        prediction = (
            metadata["target"]
            + residual[:4] * metadata["target_scale"]
        )
        objective = 0.5 * float(residual @ residual)
        entry = {
            "iteration": iteration,
            "objective": objective,
            "data_objective": 0.5
            * float(residual[:4] @ residual[:4]),
            "parameters": dict(
                zip(_CONTROL_NAMES, physical.tolist())
            ),
            "predicted_smd_um": prediction[:2].tolist(),
            "predicted_velocity_m_s": prediction[2:].tolist(),
            "damping": damping,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if iteration == iteration_count:
            break

        basis = jnp.eye(theta.size, dtype=theta.dtype)
        tangent_chunks = []
        batch_size = calibration.sensitivity_batch_size
        for start in range(0, theta.size, batch_size):
            tangent_chunks.append(
                np.asarray(
                    tangent_batch(
                        theta, basis[start : start + batch_size]
                    )
                )
            )
        jacobian = np.concatenate(tangent_chunks, axis=0).T
        entry["jacobian_column_norms"] = np.linalg.norm(
            jacobian, axis=0
        ).tolist()
        print(
            json.dumps(
                {
                    "iteration": iteration,
                    "jacobian_column_norms": entry[
                        "jacobian_column_norms"
                    ],
                }
            ),
            flush=True,
        )
        normal = (
            jacobian.T @ jacobian
            + damping * np.eye(theta.size)
        )
        step = -np.linalg.solve(
            normal, jacobian.T @ residual
        )
        norm = np.linalg.norm(step)
        if norm > 1.0:
            step = step / norm
        candidate = theta + jnp.asarray(step, theta.dtype)
        candidate_residual = np.asarray(
            residual_function(candidate)
        )
        candidate_objective = 0.5 * float(
            candidate_residual @ candidate_residual
        )
        entry["candidate_objective"] = candidate_objective
        entry["candidate_parameters"] = dict(
            zip(
                _CONTROL_NAMES,
                np.asarray(
                    metadata["physical_parameters"](candidate)
                ).tolist(),
            )
        )
        if (
            np.isfinite(candidate_objective)
            and candidate_objective < objective
        ):
            theta = candidate
            damping = max(1.0e-6, 0.5 * damping)
        else:
            damping = min(1.0e6, 10.0 * damping)

    physical = np.asarray(metadata["physical_parameters"](theta))
    elapsed = time.perf_counter() - started
    control_values = dict(
        zip(_CONTROL_NAMES, physical.tolist())
    )
    output = metadata["case"].output
    output.mkdir(parents=True, exist_ok=True)
    resolution = metadata["case"].cells[0]
    tuned_config = output / f"fv_{resolution}_tuned_validation.toml"
    validation_output = (
        f"outputs/purdue_ln2_wake/fv_{resolution}_tuned_validation"
    )
    _write_tuned_config(
        config_path,
        tuned_config,
        control_values,
        metadata["case"].speed,
        validation_output,
    )
    result = {
        "config": str(config_path),
        "method": "exact JAX JVPs through coupled low-Mach FV and parcels",
        "differentiation_mode": (
            "batched forward-mode multiple shooting"
        ),
        "sensitivity_window_steps": (
            calibration.sensitivity_window_steps
        ),
        "window_sample_start_steps": (
            calibration.window_sample_start_steps
        ),
        "sensitivity_batch_size": calibration.sensitivity_batch_size,
        "calibration_station_m": calibration.target_x,
        "held_out_validation_stations_m": [0.0762, 0.1143, 0.1524],
        "target_radii_m": metadata["target_radii_m"].tolist(),
        "target_smd_um": metadata["target"][:2].tolist(),
        "target_velocity_m_s": metadata["target"][2:].tolist(),
        "controls": control_values,
        "tuned_validation_config": str(tuned_config),
        "elapsed_seconds": elapsed,
        "history": history,
    }
    result_path = output / "differentiable_inlet_calibration.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--iterations", type=int)
    arguments = parser.parse_args()
    print(
        json.dumps(
            calibrate(arguments.config, arguments.iterations), indent=2
        )
    )


if __name__ == "__main__":
    main()
