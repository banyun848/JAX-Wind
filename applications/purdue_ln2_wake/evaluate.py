"""Evaluate the Purdue D1 atmospheric LN2 downstream parcel wake."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_d32(diameter: np.ndarray, weights: np.ndarray) -> float:
    return float(
        np.sum(weights * diameter**3) / np.sum(weights * diameter**2)
    )


def evaluate(
    checkpoint: Path,
    reference_path: Path,
    output: Path,
) -> dict[str, object]:
    with reference_path.open(newline="", encoding="utf-8") as stream:
        reference = list(csv.DictReader(stream))
    data = np.load(checkpoint)
    lengths = data["lengths_m"]
    nozzle = data["nozzle_m"]
    nx = data["temperature"].shape[2]
    ny = data["temperature"].shape[1]
    dx = float(lengths[0] / nx)
    active = data["parcel_active"].astype(bool)
    x = data["parcel_x"]
    radius = np.sqrt(
        (data["parcel_y"] - nozzle[1]) ** 2
        + (data["parcel_z"] - nozzle[2]) ** 2
    )
    diameter = data["parcel_diameter"]
    axial_velocity = data["parcel_u"]
    multiplicity = data["parcel_multiplicity"]
    source_physical_x = 0.0381
    slab_half_width = 2.0 * dx
    radial_half_width = max(4.0 * float(lengths[1] / ny), 0.003)

    comparisons: list[dict[str, object]] = []
    for item in reference:
        if item["role"] != "validation":
            continue
        physical_x = float(item["physical_x_m"])
        target_radius = float(item["radius_m"])
        solver_x = float(nozzle[0]) + physical_x - source_physical_x
        radial_mask = (
            radius < radial_half_width
            if target_radius == 0.0
            else np.abs(radius - target_radius) < radial_half_width
        )
        mask = active & (np.abs(x - solver_x) < slab_half_width) & radial_mask
        count = int(np.count_nonzero(mask))
        row: dict[str, object] = {
            "physical_x_m": physical_x,
            "solver_x_m": solver_x,
            "radius_m": target_radius,
            "parcel_count": count,
            "reference_smd_um": float(item["smd_um"]),
            "reference_mean_axial_velocity_m_s": float(
                item["mean_axial_velocity_m_s"]
            ),
        }
        if count:
            weights = multiplicity[mask]
            smd_um = 1.0e6 * _weighted_d32(diameter[mask], weights)
            velocity = _weighted_mean(axial_velocity[mask], weights)
            row.update(
                simulated_smd_um=smd_um,
                simulated_mean_axial_velocity_m_s=velocity,
                smd_relative_error=(
                    smd_um / float(item["smd_um"]) - 1.0
                ),
                velocity_relative_error=(
                    velocity / float(item["mean_axial_velocity_m_s"]) - 1.0
                ),
            )
        comparisons.append(row)

    resolved = [
        row for row in comparisons if "smd_relative_error" in row
    ]
    result = {
        "checkpoint": str(checkpoint),
        "reference": str(reference_path),
        "source_plane_physical_x_m": source_physical_x,
        "slab_half_width_m": slab_half_width,
        "radial_half_width_m": radial_half_width,
        "comparisons": comparisons,
        "populated_comparisons": len(resolved),
        "total_comparisons": len(comparisons),
        "coverage_fraction": len(resolved) / len(comparisons),
        "mean_absolute_smd_relative_error": float(
            np.mean([abs(row["smd_relative_error"]) for row in resolved])
        ),
        "mean_absolute_velocity_relative_error": float(
            np.mean([abs(row["velocity_relative_error"]) for row in resolved])
        ),
        "experimental_uncertainty": {
            "smd_relative": 0.10,
            "velocity_relative": 0.15,
        },
        "note": (
            "Figure 9 profiles were visually digitized and mirrored into radial "
            "averages; errors therefore include digitization and experimental "
            "asymmetry."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("cases/PurdueLN2Wake/reference_profiles.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/purdue_ln2_wake/fv_256/validation.json"),
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            evaluate(arguments.checkpoint, arguments.reference, arguments.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
