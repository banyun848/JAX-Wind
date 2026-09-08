"""Post-process a completed finite-volume LN2 jet checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from jaxwind.io.field_archive import open_fields


def diagnostics(checkpoint: str | Path) -> dict[str, object]:
    """Return physical plume and conservation diagnostics."""

    path = Path(checkpoint)
    with open_fields(path) as data:
        temperature = data["temperature"]
        nitrogen = data["nitrogen"]
        liquid = data["liquid_water"]
        ice = data["ice_water"]
        nz, ny, nx = temperature.shape
        lengths = tuple(
            float(value)
            for value in (
                data["lengths_m"]
                if "lengths_m" in data
                else np.asarray((6.0, 6.0, 3.6))
            )
        )
        dx, dy, dz = (lengths[0] / nx, lengths[1] / ny, lengths[2] / nz)
        volume = dx * dy * dz
        u = 0.5 * (data["velocity_x"][..., :-1] + data["velocity_x"][..., 1:])
        v = 0.5 * (data["velocity_y"][:, :-1] + data["velocity_y"][:, 1:])
        w = 0.5 * (data["velocity_z"][:-1] + data["velocity_z"][1:])
        speed = np.sqrt(u * u + v * v + w * w)

        def location(index):
            k, j, i = index
            return [(i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz]

        def extent(mask):
            indices = np.argwhere(mask)
            if not len(indices):
                return None
            return {
                "x_m": [(indices[:, 2].min() + 0.5) * dx, (indices[:, 2].max() + 0.5) * dx],
                "y_m": [(indices[:, 1].min() + 0.5) * dy, (indices[:, 1].max() + 0.5) * dy],
                "z_m": [(indices[:, 0].min() + 0.5) * dz, (indices[:, 0].max() + 0.5) * dz],
            }

        fields = (
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "pressure",
            "temperature",
            "water_vapor",
            "nitrogen",
            "liquid_water",
            "ice_water",
        )
        active = data["parcel_active"]
        return {
            "domain_lengths_m": list(lengths),
            "all_fields_finite": all(np.isfinite(data[name]).all() for name in fields),
            "minimum_temperature_k": float(temperature.min()),
            "minimum_temperature_location_m": location(
                np.unravel_index(np.argmin(temperature), temperature.shape)
            ),
            "maximum_nitrogen_mass_fraction": float(nitrogen.max()),
            "maximum_nitrogen_location_m": location(
                np.unravel_index(np.argmax(nitrogen), nitrogen.shape)
            ),
            "maximum_speed_m_s": float(speed.max()),
            "mean_speed_m_s": float(speed.mean()),
            "cold_plume_T_lt_290_extent": extent(temperature < 290.0),
            "nitrogen_plume_Y_gt_0p01_extent": extent(nitrogen > 0.01),
            "cold_volume_m3": float(np.sum(temperature < 290.0) * volume),
            "nitrogen_rich_volume_m3": float(np.sum(nitrogen > 0.01) * volume),
            "gas_nitrogen_inventory_kg_constant_density": float(
                1.225 * np.sum(nitrogen) * volume
            ),
            "liquid_parcel_inventory_kg": float(
                np.sum(data["parcel_mass"] * data["parcel_multiplicity"] * active)
            ),
            "liquid_fog_inventory_kg": float(1.225 * np.sum(liquid) * volume),
            "ice_fog_inventory_kg": float(1.225 * np.sum(ice) * volume),
            "active_parcels": int(np.sum(active)),
            "maximum_sidewall_normal_speed_m_s": float(
                max(
                    np.max(np.abs(data["velocity_y"][:, 0])),
                    np.max(np.abs(data["velocity_y"][:, -1])),
                )
            ),
        }


def render_slices(slices: str | Path, output: str | Path) -> None:
    """Render the saved centerplane temperature, N2, fog, and ice fields."""

    import matplotlib.pyplot as plt

    with open_fields(slices) as data:
        lengths = tuple(
            float(value)
            for value in (
                data["lengths_m"]
                if "lengths_m" in data
                else np.asarray((6.0, 6.0, 3.6))
            )
        )
        nozzle = tuple(
            float(value)
            for value in (
                data["nozzle_m"]
                if "nozzle_m" in data
                else np.asarray((3.0, 3.0, 0.876))
            )
        )
        fields = (
            (data["temperature"][:, data["temperature"].shape[1] // 2], "Temperature [K]", "turbo"),
            (data["nitrogen"][:, data["nitrogen"].shape[1] // 2], "Nitrogen mass fraction", "viridis"),
            (data["liquid_water"][:, data["liquid_water"].shape[1] // 2], "Liquid fog [kg/kg]", "Blues"),
            (data["ice_water"][:, data["ice_water"].shape[1] // 2], "Ice fog [kg/kg]", "Purples"),
        )
    figure, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    for axis, (field, title, colourmap) in zip(axes.flat, fields):
        image = axis.imshow(
            field,
            origin="lower",
            extent=(0.0, lengths[0], 0.0, lengths[2]),
            aspect="auto",
            cmap=colourmap,
        )
        axis.plot(nozzle[0], nozzle[2], "w>", markersize=5)
        axis.set(xlabel="x [m]", ylabel="z [m]", title=title)
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args(argv)
    directory = arguments.directory
    checkpoint = directory / "checkpoint.npz"
    result = diagnostics(checkpoint)
    (directory / "diagnostics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    render_slices(checkpoint, directory / "centerplane.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
