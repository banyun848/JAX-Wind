"""Render the DLR IN-1 100-frame four-panel archive as MP4 or GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.colors import LogNorm
import numpy as np


def _limits(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    result = np.percentile(finite, (low, high))
    if result[1] <= result[0]:
        result[1] = result[0] + max(abs(result[0]), 1.0) * 1.0e-6
    return float(result[0]), float(result[1])


def render(archive: Path, output: Path, fps: int = 20) -> None:
    with np.load(archive) as data:
        fields = (
            data["velocity_magnitude"],
            data["temperature"],
            data["nitrogen"],
            data["fog_mass_fraction"],
        )
        times = data["time_seconds"]
        lengths = data["lengths_m"]
        source = data["source_m"]
        offset = float(data["physical_x_offset_m"])
    if len(times) != 100:
        raise ValueError(f"expected 100 frames, found {len(times)}")

    positive_fog = fields[3][fields[3] > 0.0]
    fog_limits = (
        _limits(positive_fog, 1.0, 99.8)
        if positive_fog.size
        else (1.0e-12, 1.0e-11)
    )
    panels = (
        (
            "Velocity magnitude [m/s]",
            "viridis",
            (0.0, _limits(fields[0], 0, 99.8)[1]),
            None,
        ),
        ("Temperature [K]", "magma", _limits(fields[1], 0.2, 99.8), None),
        (
            "Injected N₂ mass fraction [-]",
            "plasma",
            (0.0, _limits(fields[2], 0, 99.8)[1]),
            None,
        ),
        ("Fog mass fraction [-]", "gray_r", fog_limits, LogNorm(*fog_limits)),
    )
    extent = 1.0e3 * np.asarray(
        (offset, offset + lengths[0], -source[1], lengths[1] - source[1])
    )
    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=True)
    images = []
    for axis, values, (title, cmap, limits, norm) in zip(
        axes.flat, fields, panels
    ):
        kwargs = {
            "origin": "lower",
            "extent": extent,
            "aspect": "equal",
            "cmap": cmap,
        }
        if norm is None:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        else:
            kwargs["norm"] = norm
        image = axis.imshow(values[0], **kwargs)
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("Downstream y [mm]")
        axis.set_ylabel("Signed x [mm]")
        for station in (5, 10, 15, 20, 30, 40, 50, 60, 70):
            axis.axvline(station, color="white", alpha=0.45, linewidth=0.65)
        figure.colorbar(image, ax=axis, shrink=0.86)
    timestamp = figure.suptitle(f"DLR IN-1 — t = {1e3 * times[0]:.1f} ms")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        writer = PillowWriter(
            fps=fps,
            metadata={"title": "DLR IN-1 source-driven LN2 wake"},
        )
        dpi = 90
    else:
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=6000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            metadata={"title": "DLR IN-1 source-driven LN2 wake"},
        )
        dpi = 120
    with writer.saving(figure, str(output), dpi=dpi):
        for index, instant in enumerate(times):
            for image, values in zip(images, fields):
                image.set_data(values[index])
            timestamp.set_text(f"DLR IN-1 — t = {1e3 * instant:.1f} ms")
            writer.grab_frame()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    arguments = parser.parse_args()
    render(arguments.archive, arguments.output, arguments.fps)


if __name__ == "__main__":
    main()
