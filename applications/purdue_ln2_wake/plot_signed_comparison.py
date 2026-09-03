"""Plot signed Purdue PDPA profiles against an axisymmetric FV prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _comparison_rows(report: dict[str, object]) -> list[dict[str, object]]:
    calibration = report.get("calibration_first_section", {})
    calibration_rows = calibration.get("comparisons", [])
    return [*calibration_rows, *report["comparisons"]]


def plot(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _comparison_rows(report)
    calibration_stations = {
        float(row["physical_x_m"])
        for row in report["calibration_first_section"]["comparisons"]
    }
    stations = sorted({float(row["physical_x_m"]) for row in rows})
    figure, axes = plt.subplots(
        4,
        len(stations),
        figsize=(5.25 * len(stations), 14),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.05, 0.72]},
    )
    colors = {
        "experiment": "#222222",
        "simulation": "#7B2CBF",
        "smd": "#0072B2",
        "velocity": "#E69F00",
        "count": "#009E73",
        "missing": "#D55E00",
    }

    for column, station in enumerate(stations):
        group = sorted(
            [
                row
                for row in rows
                if float(row["physical_x_m"]) == station
            ],
            key=lambda row: float(row["lateral_m"]),
        )
        lateral = 1.0e3 * np.asarray(
            [float(row["lateral_m"]) for row in group]
        )
        reference_smd = np.asarray(
            [float(row["smd_um"]) for row in group]
        )
        reference_velocity = np.asarray(
            [float(row["velocity_m_s"]) for row in group]
        )
        simulated_smd = np.asarray(
            [float(row.get("simulated_smd_um", np.nan)) for row in group]
        )
        simulated_velocity = np.asarray(
            [
                float(
                    row.get(
                        "simulated_mean_axial_velocity_m_s",
                        np.nan,
                    )
                )
                for row in group
            ]
        )
        observations = np.asarray(
            [int(row["observation_count"]) for row in group]
        )
        missing = ~np.isfinite(simulated_smd)
        smd_axis, velocity_axis, error_axis, count_axis = axes[:, column]

        smd_axis.set_title(
            f"Axial station {1.0e3 * station:.1f} mm\n"
            f"({'calibrated' if station in calibration_stations else 'held out'})",
            fontsize=13,
        )
        smd_axis.errorbar(
            lateral,
            reference_smd,
            yerr=0.10 * reference_smd,
            fmt="o-",
            color=colors["experiment"],
            capsize=4,
            label="Signed Purdue PDPA ±10%",
        )
        smd_axis.plot(
            lateral,
            simulated_smd,
            "X--",
            color=colors["simulation"],
            markersize=9,
            linewidth=2.2,
            label="Axisymmetric FV",
        )
        velocity_axis.errorbar(
            lateral,
            reference_velocity,
            yerr=0.15 * reference_velocity,
            fmt="o-",
            color=colors["experiment"],
            capsize=4,
            label="Signed Purdue PDPA ±15%",
        )
        velocity_axis.plot(
            lateral,
            simulated_velocity,
            "X--",
            color=colors["simulation"],
            markersize=9,
            linewidth=2.2,
            label="Axisymmetric FV",
        )

        smd_error = 100.0 * (
            simulated_smd / reference_smd - 1.0
        )
        velocity_error = 100.0 * (
            simulated_velocity / reference_velocity - 1.0
        )
        error_axis.axhspan(
            -15.0, 15.0, color=colors["velocity"], alpha=0.10
        )
        error_axis.axhspan(
            -10.0, 10.0, color=colors["smd"], alpha=0.14
        )
        error_axis.axhline(0.0, color="black", linewidth=1.0)
        error_axis.plot(
            lateral,
            smd_error,
            "o-",
            color=colors["smd"],
            linewidth=2.0,
            label="SMD error",
        )
        error_axis.plot(
            lateral,
            velocity_error,
            "s-",
            color=colors["velocity"],
            linewidth=2.0,
            label="Velocity error",
        )
        count_axis.bar(
            lateral,
            observations,
            width=4.0,
            color=colors["count"],
            alpha=0.8,
            edgecolor="black",
            linewidth=0.6,
        )

        for x_value, observation in zip(lateral, observations):
            count_axis.text(
                x_value,
                observation
                + max(float(np.max(observations)) * 0.04, 0.6),
                str(observation),
                ha="center",
                fontsize=8,
            )

        if np.any(missing):
            for axis in (smd_axis, velocity_axis, error_axis):
                axis.scatter(
                    lateral[missing],
                    np.zeros(np.count_nonzero(missing)),
                    marker="x",
                    s=100,
                    linewidths=2.5,
                    color=colors["missing"],
                    zorder=5,
                )

        smd_axis.set_ylabel("SMD [µm]")
        velocity_axis.set_ylabel("Axial velocity [m/s]")
        error_axis.set_ylabel("Relative error [%]")
        count_axis.set_ylabel("Parcel observations")
        count_axis.set_xlabel("Signed lateral coordinate [mm]")
        error_axis.set_ylim(-85.0, 35.0)
        count_axis.set_ylim(
            0.0, max(10.0, float(np.max(observations)) * 1.25)
        )
        limit = max(25.0, float(np.max(np.abs(lateral))) + 4.0)
        for axis in (
            smd_axis,
            velocity_axis,
            error_axis,
            count_axis,
        ):
            axis.set_xlim(-limit, limit)
            axis.grid(True, alpha=0.25)

    axes[0, 0].legend(fontsize=8.5)
    axes[1, 0].legend(fontsize=8.5)
    axes[2, 0].legend(loc="lower left", fontsize=8.5)
    figure.suptitle(
        "Purdue D1 atmospheric LN₂ wake — calibration and held-out comparison\n"
        "Experimental asymmetry retained; FV annular means mirrored to ±r",
        fontsize=16,
        y=0.992,
    )
    calibration = report["calibration_first_section"]
    summary = (
        f"Held-out coverage {report['populated_signed_points']}/"
        f"{report['total_signed_points']} "
        f"({100.0 * report['coverage_fraction']:.0f}%)   |   "
        f"held-out mean |SMD error| "
        f"{100.0 * report['mean_absolute_smd_relative_error']:.1f}%   |   "
        f"held-out mean |velocity error| "
        f"{100.0 * report['mean_absolute_velocity_relative_error']:.1f}%\n"
        f"38.1 mm calibrated mean errors: SMD "
        f"{100.0 * calibration['mean_absolute_smd_relative_error']:.1f}%   |   "
        f"velocity "
        f"{100.0 * calibration['mean_absolute_velocity_relative_error']:.1f}%"
    )
    figure.text(
        0.5,
        0.946,
        summary,
        ha="center",
        fontsize=11,
        bbox={
            "facecolor": "#F3F4F6",
            "edgecolor": "0.75",
            "boxstyle": "round,pad=0.45",
        },
    )
    figure.text(
        0.5,
        0.012,
        "Calibrated plane: 38.1 mm. Held-out planes: 76.2, 114.3, "
        "and 152.4 mm. Simulation statistics use positive axial "
        "parcel-crossing flux after spin-up.",
        ha="center",
        fontsize=9.5,
        color="0.3",
    )
    figure.tight_layout(
        rect=(0.02, 0.04, 0.99, 0.91),
        h_pad=1.3,
        w_pad=1.4,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/purdue_ln2_wake/fv_256_experiment/"
            "signed_comprehensive_comparison.png"
        ),
    )
    arguments = parser.parse_args()
    plot(arguments.report, arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
