"""Plot all signed, held-out DLR IN-1 PDA profiles against the FV result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(report_path: Path, output: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["comparisons"]
    stations = sorted({float(row["y_D"]) for row in rows})
    variables = (
        (
            "d10_um",
            "simulated_d10_um",
            "D10 [µm]",
            3.3,
            "d10_hi_um",
            "d10_lo_um",
        ),
        (
            "u_mean_m_s",
            "simulated_u_mean_m_s",
            "Axial U [m/s]",
            6.1,
            "u_mean_hi_m_s",
            "u_mean_lo_m_s",
        ),
        (
            "v_mean_m_s",
            "simulated_v_mean_m_s",
            "Signed radial V [m/s]",
            8.0,
            None,
            None,
        ),
    )
    figure, axes = plt.subplots(
        3, len(stations), figsize=(3.1 * len(stations), 10.0), sharex=False
    )
    for column, station in enumerate(stations):
        group = sorted(
            [row for row in rows if float(row["y_D"]) == station],
            key=lambda row: float(row["x_D"]),
        )
        lateral = np.asarray([float(row["x_D"]) for row in group])
        for row_index, (
            reference_key,
            prediction_key,
            label,
            uncertainty,
            high_key,
            low_key,
        ) in enumerate(variables):
            axis = axes[row_index, column]
            reference = np.asarray(
                [float(row[reference_key]) for row in group]
            )
            prediction = np.asarray(
                [float(row.get(prediction_key, np.nan)) for row in group]
            )
            axis.errorbar(
                lateral,
                reference,
                yerr=uncertainty,
                fmt="o-",
                color="#222222",
                capsize=2.5,
                linewidth=1.3,
                markersize=4.0,
                label="DLR PDA",
            )
            axis.plot(
                lateral,
                prediction,
                "X--",
                color="#7B2CBF",
                linewidth=1.7,
                markersize=5.5,
                label="Axisymmetric FV",
            )
            if high_key is not None:
                high = np.asarray([float(row[high_key]) for row in group])
                low = np.asarray([float(row[low_key]) for row in group])
                axis.scatter(
                    lateral,
                    high,
                    marker="^",
                    s=18,
                    color="#D55E00",
                    alpha=0.65,
                    label="DLR Hi/Lo populations",
                )
                axis.scatter(
                    lateral,
                    low,
                    marker="v",
                    s=18,
                    color="#0072B2",
                    alpha=0.65,
                )
            axis.axhline(0.0, color="0.65", linewidth=0.7)
            axis.grid(True, alpha=0.22)
            axis.set_xlabel("Signed x/D")
            if column == 0:
                axis.set_ylabel(label)
            if row_index == 0:
                axis.set_title(f"y/D = {station:g}")

    axes[0, 0].legend(fontsize=7.5, loc="best")
    metrics = report["metrics"]
    summary = (
        f"Coverage {report['populated_signed_points']}/"
        f"{report['total_signed_points']}  |  "
        f"MAE: D10 {metrics['d10_um']['mae']:.2f} µm, "
        f"U {metrics['u_mean_m_s']['mae']:.2f} m/s, "
        f"V {metrics['v_mean_m_s']['mae']:.2f} m/s"
    )
    figure.suptitle(
        "DLR IN-1 source-driven LN₂ wake — all held-out signed PDA points\n"
        + summary,
        fontsize=15,
    )
    figure.text(
        0.5,
        0.01,
        "Measured y/D=5 defines the source. Error bars show published maximum "
        "uncertainties; Hi/Lo markers are separated droplet populations.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.015, 0.035, 0.995, 0.935))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or arguments.report.with_name(
        "comprehensive_signed_comparison.png"
    )
    plot(arguments.report, output)
    print(output)


if __name__ == "__main__":
    main()
