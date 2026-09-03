"""Compare baseline and LN2-cooled HITSZ far-wake frame archives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _frame_path(path: Path) -> Path:
    return path / "main_flow_frames.npz" if path.is_dir() else path


def _mean_tail(values: np.ndarray, fraction: float) -> np.ndarray:
    start = int(np.floor((1.0 - fraction) * values.shape[0]))
    return np.mean(values[start:], axis=0)


def _vertical_moments(
    velocity: np.ndarray,
    reference_profile: np.ndarray,
    z_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    deficit = np.maximum(reference_profile[:, None] - velocity, 0.0)
    total = np.sum(deficit, axis=0)
    safe = np.maximum(total, np.finfo(velocity.dtype).tiny)
    center = np.sum(deficit * z_m[:, None], axis=0) / safe
    width = np.sqrt(
        np.sum(deficit * (z_m[:, None] - center[None, :]) ** 2, axis=0)
        / safe
    )
    center = np.where(total > 0.0, center, np.nan)
    width = np.where(total > 0.0, width, np.nan)
    return center, width


def _cold_moments(
    temperature_anomaly: np.ndarray,
    z_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weight = np.maximum(-temperature_anomaly, 0.0)
    total = np.sum(weight, axis=0)
    safe = np.maximum(total, np.finfo(temperature_anomaly.dtype).tiny)
    center = np.sum(weight * z_m[:, None], axis=0) / safe
    minimum = np.min(temperature_anomaly, axis=0)
    return np.where(total > 0.0, center, np.nan), minimum


def compare_far_wakes(
    baseline_path: str | Path,
    cooled_path: str | Path,
    output_directory: str | Path,
    *,
    rotor_x_m: float = 12.0,
    rotor_diameter_m: float = 1.26,
    hub_height_m: float = 0.876,
    averaging_fraction: float = 0.5,
    stations_d: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0),
) -> dict[str, Any]:
    """Create matched mean fields and vertical far-wake metrics."""

    if not 0.0 < averaging_fraction <= 1.0:
        raise ValueError("averaging_fraction must lie in (0, 1]")
    baseline_file = _frame_path(Path(baseline_path))
    cooled_file = _frame_path(Path(cooled_path))
    with np.load(baseline_file) as baseline_data, np.load(
        cooled_file
    ) as cooled_data:
        x_m = np.asarray(baseline_data["x_m"])
        z_m = np.asarray(baseline_data["z_m"])
        if not np.allclose(x_m, cooled_data["x_m"]):
            raise ValueError("baseline and cooled x coordinates differ")
        if not np.allclose(z_m, cooled_data["z_m"]):
            raise ValueError("baseline and cooled z coordinates differ")
        baseline_u = _mean_tail(
            np.asarray(baseline_data["u_center_zx"]),
            averaging_fraction,
        )
        cooled_u = _mean_tail(
            np.asarray(cooled_data["u_center_zx"]),
            averaging_fraction,
        )
        if "scalar_center_zx" not in cooled_data:
            raise ValueError("cooled frames do not contain scalar_center_zx")
        cooled_temperature = _mean_tail(
            np.asarray(cooled_data["scalar_center_zx"]),
            averaging_fraction,
        )
        baseline_frames = int(baseline_data["u_center_zx"].shape[0])
        cooled_frames = int(cooled_data["u_center_zx"].shape[0])

    upstream = (x_m >= rotor_x_m - 3.0 * rotor_diameter_m) & (
        x_m <= rotor_x_m - rotor_diameter_m
    )
    if not np.any(upstream):
        raise ValueError("the archive lacks the requested upstream reference window")
    reference_profile = np.mean(baseline_u[:, upstream], axis=1)
    baseline_center, baseline_width = _vertical_moments(
        baseline_u, reference_profile, z_m
    )
    cooled_center, cooled_width = _vertical_moments(
        cooled_u, reference_profile, z_m
    )
    cold_center, minimum_temperature = _cold_moments(
        cooled_temperature, z_m
    )

    rows: list[dict[str, float]] = []
    for station_d in stations_d:
        target = rotor_x_m + station_d * rotor_diameter_m
        index = int(np.argmin(np.abs(x_m - target)))
        rows.append(
            {
                "x_over_d": station_d,
                "x_m": float(x_m[index]),
                "baseline_wake_center_z_m": float(baseline_center[index]),
                "cooled_wake_center_z_m": float(cooled_center[index]),
                "wake_center_shift_m": float(
                    cooled_center[index] - baseline_center[index]
                ),
                "baseline_vertical_width_m": float(baseline_width[index]),
                "cooled_vertical_width_m": float(cooled_width[index]),
                "cold_plume_center_z_m": float(cold_center[index]),
                "minimum_temperature_anomaly_k": float(
                    minimum_temperature[index]
                ),
                "hub_velocity_change_m_s": float(
                    np.interp(hub_height_m, z_m, cooled_u[:, index])
                    - np.interp(hub_height_m, z_m, baseline_u[:, index])
                ),
            }
        )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "far_wake_stations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    npz_path = output / "far_wake_comparison.npz"
    np.savez_compressed(
        npz_path,
        x_m=x_m,
        z_m=z_m,
        baseline_mean_u_center_zx=baseline_u,
        cooled_mean_u_center_zx=cooled_u,
        velocity_change_center_zx=cooled_u - baseline_u,
        temperature_anomaly_center_zx=cooled_temperature,
        baseline_wake_center_z_m=baseline_center,
        cooled_wake_center_z_m=cooled_center,
        baseline_vertical_width_m=baseline_width,
        cooled_vertical_width_m=cooled_width,
        cold_plume_center_z_m=cold_center,
    )

    figure_path = output / "far_wake_comparison.png"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = (
        (x_m[0] - rotor_x_m) / rotor_diameter_m,
        (x_m[-1] - rotor_x_m) / rotor_diameter_m,
        (z_m[0] - hub_height_m) / rotor_diameter_m,
        (z_m[-1] - hub_height_m) / rotor_diameter_m,
    )
    fields = (
        (baseline_u, "Baseline mean velocity", "viridis"),
        (cooled_u, "Cooled mean velocity", "viridis"),
        (cooled_u - baseline_u, "Cooled - baseline velocity", "coolwarm"),
        (cooled_temperature, "Temperature anomaly", "magma"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 6.8), constrained_layout=True)
    for axis, (field, title, cmap) in zip(axes.flat, fields):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=cmap,
        )
        axis.axvline(0.0, color="white", linewidth=0.7, alpha=0.7)
        axis.set_title(title)
        axis.set_xlabel("(x - x_rotor) / D")
        axis.set_ylabel("(z - z_hub) / D")
        figure.colorbar(image, ax=axis, shrink=0.86)
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)

    summary = {
        "schema": "jaxwind.hitsz-ln2-far-wake.v1",
        "baseline_frames": baseline_frames,
        "cooled_frames": cooled_frames,
        "averaging_fraction": averaging_fraction,
        "rotor_x_m": rotor_x_m,
        "rotor_diameter_m": rotor_diameter_m,
        "hub_height_m": hub_height_m,
        "stations": rows,
        "csv": str(csv_path),
        "fields": str(npz_path),
        "figure": str(figure_path),
    }
    summary_path = output / "far_wake_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("cooled", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--averaging-fraction", type=float, default=0.5)
    arguments = parser.parse_args(argv)
    result = compare_far_wakes(
        arguments.baseline,
        arguments.cooled,
        arguments.output,
        averaging_fraction=arguments.averaging_fraction,
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
