#!/usr/bin/env python3
"""Average saved FV wake slices and reference them to precursor inflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jaxwind.io.recorded_field import RecordedField


def _stream_mean(path: Path, batch_size: int = 256) -> np.ndarray:
    values = RecordedField(path)
    total = np.zeros(values.shape[1:], dtype=np.float64)
    count = 0
    for start in range(0, values.shape[0], batch_size):
        batch = np.asarray(values[start : start + batch_size], dtype=np.float64)
        total += batch.sum(axis=0)
        count += batch.shape[0]
    if count == 0:
        raise ValueError(f"precursor archive is empty: {path}")
    return total / count


def _linear_index(coordinates: np.ndarray, value: float) -> tuple[int, int, float]:
    upper = int(np.searchsorted(coordinates, value, side="right"))
    upper = min(max(upper, 1), coordinates.size - 1)
    lower = upper - 1
    fraction = (value - coordinates[lower]) / (
        coordinates[upper] - coordinates[lower]
    )
    return lower, upper, float(np.clip(fraction, 0.0, 1.0))


def _plot_two_panel(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    *,
    grid,
    turbine,
    output: Path,
    label: str,
    colorbar_label: str,
    cmap: str,
    symmetric: bool = False,
) -> None:
    combined = np.concatenate((horizontal.ravel(), vertical.ravel()))
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        raise ValueError(f"averaged field contains no finite values: {label}")
    if symmetric:
        limit = max(float(np.percentile(np.abs(finite), 99.0)), 1.0e-12)
        lower, upper = -limit, limit
    else:
        lower, upper = np.percentile(finite, (1.0, 99.0))

    figure, axes = plt.subplots(
        2, 1, figsize=(12.8, 8.6), constrained_layout=True
    )
    panels = (
        (horizontal, grid.y_faces, "Hub-height x-y plane", "y [m]", turbine.y),
        (vertical, grid.z_faces, "Centerline x-z plane", "z [m]", turbine.z),
    )
    images = []
    for axis, (values, transverse, title, ylabel, center) in zip(
        axes, panels, strict=True
    ):
        image = axis.pcolormesh(
            grid.x_faces,
            transverse,
            values,
            shading="flat",
            cmap=cmap,
            vmin=float(lower),
            vmax=float(upper),
            rasterized=True,
        )
        images.append(image)
        axis.set(
            xlim=(0.0, grid.lx),
            ylim=(0.0, grid.ly if ylabel.startswith("y") else grid.lz),
            xlabel="x [m]",
            ylabel=ylabel,
            title=title,
            aspect="equal",
        )
        axis.plot(
            (turbine.x, turbine.x),
            (center - turbine.tip_radius, center + turbine.tip_radius),
            color="white",
            linewidth=2.2,
        )
        axis.plot(
            turbine.x,
            center,
            marker="+",
            color="black",
            markersize=7,
            markeredgewidth=1.5,
        )
    figure.colorbar(images[0], ax=axes, pad=0.02, shrink=0.92).set_label(
        colorbar_label
    )
    figure.suptitle(label)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--start-time-s", type=float)
    arguments = parser.parse_args()

    from jaxwind.simulation.turbines import build_turbine_definition
    from jaxwind.config.stages import load_workflow
    from jaxwind.domain import ScaleSystem

    workflow = load_workflow(arguments.config)
    if workflow.turbine is None:
        raise ValueError("averaged wake fields require a configured turbine")
    root = workflow.options.output_directory
    output = arguments.output_directory or root / "averaged_wake"
    output.mkdir(parents=True, exist_ok=True)

    with np.load(root / "main/flow_frames.npz") as archive:
        horizontal = np.asarray(archive["u_hub_yx"], dtype=np.float64)
        vertical = np.asarray(archive["u_center_zx"], dtype=np.float64)
        times = np.asarray(archive["time_seconds"], dtype=np.float64)
        x = np.asarray(archive["x_m"], dtype=np.float64)
        y = np.asarray(archive["y_m"], dtype=np.float64)
        z = np.asarray(archive["z_m"], dtype=np.float64)
    selected = np.ones(times.shape, dtype=bool)
    if arguments.start_time_s is not None:
        selected = times >= arguments.start_time_s
    if not np.any(selected):
        raise ValueError("the requested averaging window contains no frames")
    horizontal = horizontal[selected]
    vertical = vertical[selected]
    selected_times = times[selected]

    mean_horizontal = horizontal.mean(axis=0)
    mean_vertical = vertical.mean(axis=0)
    rms_horizontal = horizontal.std(axis=0)
    rms_vertical = vertical.std(axis=0)

    input_root = workflow.options.input_directory or root
    precursor = _stream_mean(input_root / "precursor/inflow" / "x_velocity.npy")
    turbine_definition = build_turbine_definition(workflow)
    turbine = turbine_definition.to_actuator_disk(scales=ScaleSystem(1.0, 1.0))
    z_lower, z_upper, z_fraction = _linear_index(z, turbine.z)
    y_lower, y_upper, y_fraction = _linear_index(y, turbine.y)
    reference_hub = (
        (1.0 - z_fraction) * precursor[z_lower]
        + z_fraction * precursor[z_upper]
    )
    reference_center = (
        (1.0 - y_fraction) * precursor[:, y_lower]
        + y_fraction * precursor[:, y_upper]
    )
    deficit_horizontal = reference_hub[:, None] - mean_horizontal
    deficit_vertical = reference_center[:, None] - mean_vertical
    deficit_fraction_horizontal = deficit_horizontal / np.maximum(
        np.abs(reference_hub[:, None]), 1.0e-12
    )
    deficit_fraction_vertical = deficit_vertical / np.maximum(
        np.abs(reference_center[:, None]), 1.0e-12
    )

    np.savez_compressed(
        output / "averaged_wake_fields.npz",
        x_m=x,
        y_m=y,
        z_m=z,
        time_seconds=selected_times,
        mean_u_hub_yx=mean_horizontal.astype(np.float32),
        rms_u_hub_yx=rms_horizontal.astype(np.float32),
        mean_u_center_zx=mean_vertical.astype(np.float32),
        rms_u_center_zx=rms_vertical.astype(np.float32),
        inflow_mean_u_hub_y=reference_hub.astype(np.float32),
        inflow_mean_u_center_z=reference_center.astype(np.float32),
        deficit_u_hub_yx=deficit_horizontal.astype(np.float32),
        deficit_u_center_zx=deficit_vertical.astype(np.float32),
        deficit_fraction_hub_yx=deficit_fraction_horizontal.astype(np.float32),
        deficit_fraction_center_zx=deficit_fraction_vertical.astype(np.float32),
    )
    metadata = {
        "schema": "jaxwind.averaged-wake.v1",
        "source": str(root / "main/flow_frames.npz"),
        "precursor_reference": str(
            input_root / "precursor/inflow" / "x_velocity.npy"
        ),
        "sample_count": int(selected_times.size),
        "time_start_seconds": float(selected_times[0]),
        "time_end_seconds": float(selected_times[-1]),
        "time_weighting": "uniform saved snapshots",
        "turbine_model": workflow.turbine.model,
        "turbine_x_m": workflow.turbine.x_m,
        "rotor_diameter_m": 2.0 * turbine.tip_radius,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    grid = workflow.case.physical.physical_grid
    label = (
        f"HITSZ R9 ALM — {selected_times[0]:.2f}–"
        f"{selected_times[-1]:.2f} s snapshot average"
    )
    _plot_two_panel(
        mean_horizontal,
        mean_vertical,
        grid=grid,
        turbine=turbine,
        output=output / "mean_streamwise_velocity.png",
        label=label,
        colorbar_label="mean u [m s$^{-1}$]",
        cmap="turbo",
    )
    _plot_two_panel(
        deficit_fraction_horizontal,
        deficit_fraction_vertical,
        grid=grid,
        turbine=turbine,
        output=output / "mean_velocity_deficit_fraction.png",
        label=label,
        colorbar_label="(U_inflow - mean u) / U_inflow",
        cmap="coolwarm",
        symmetric=True,
    )
    _plot_two_panel(
        rms_horizontal,
        rms_vertical,
        grid=grid,
        turbine=turbine,
        output=output / "resolved_streamwise_rms.png",
        label=label,
        colorbar_label="resolved u RMS [m s$^{-1}$]",
        cmap="viridis",
    )
    print(
        f"wrote {output / 'averaged_wake_fields.npz'} "
        f"from {selected_times.size} frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
