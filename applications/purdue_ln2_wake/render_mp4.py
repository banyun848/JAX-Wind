"""Render the 100-frame Purdue LN2 experiment archive as a four-panel MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio_ffmpeg
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm
import numpy as np


def _limits(values: np.ndarray, lower: float, upper: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    low, high = np.percentile(finite, (lower, upper))
    if high <= low:
        high = low + max(abs(low), 1.0) * 1.0e-6
    return float(low), float(high)


def render(archive: Path, output: Path, *, fps: int = 20) -> None:
    with np.load(archive) as data:
        velocity = data["velocity_magnitude"]
        temperature = data["temperature"]
        nitrogen = data["nitrogen"]
        fog = data["fog_mass_fraction"]
        times = data["time_seconds"]
        lengths = data["lengths_m"]
        source = data["source_m"]
        physical_offset = float(data["physical_x_offset_m"])

    if velocity.shape[0] != 100:
        raise ValueError(f"expected 100 frames, found {velocity.shape[0]}")

    x_extent = (physical_offset, physical_offset + float(lengths[0]))
    y_extent = (-float(source[1]), float(lengths[1] - source[1]))
    extent_mm = tuple(1.0e3 * value for value in (*x_extent, *y_extent))
    velocity_limits = (0.0, _limits(velocity, 0.0, 99.8)[1])
    temperature_limits = _limits(temperature, 0.2, 99.8)
    nitrogen_limits = (0.0, _limits(nitrogen, 0.0, 99.8)[1])
    positive_fog = fog[fog > 0.0]
    if positive_fog.size:
        fog_limits = _limits(positive_fog, 1.0, 99.8)
    else:
        fog_limits = (1.0e-12, 1.0e-11)

    mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=True)
    panels = (
        (velocity, "Velocity magnitude [m/s]", "viridis", velocity_limits, None),
        (temperature, "Temperature [K]", "magma", temperature_limits, None),
        (nitrogen, "Nitrogen mass fraction [-]", "plasma", nitrogen_limits, None),
        (
            fog,
            "Fog condensate mass fraction [-]",
            "gray_r",
            fog_limits,
            LogNorm(*fog_limits),
        ),
    )
    images = []
    measurement_planes_mm = (38.1, 76.2, 114.3, 152.4)
    for axis, (values, title, cmap, limits, norm) in zip(axes.flat, panels):
        kwargs = {"cmap": cmap, "origin": "lower", "extent": extent_mm, "aspect": "equal"}
        if norm is None:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        else:
            kwargs["norm"] = norm
        image = axis.imshow(values[0], **kwargs)
        images.append(image)
        axis.set_title(title)
        axis.set_xlabel("Distance below nozzle [mm]")
        axis.set_ylabel("Signed lateral coordinate [mm]")
        for station in measurement_planes_mm:
            axis.axvline(station, color="white", linewidth=0.7, alpha=0.55, linestyle="--")
        figure.colorbar(image, ax=axis, shrink=0.88)

    timestamp = figure.suptitle(f"Purdue D1 LN2 spray — t = {times[0] * 1.0e3:.2f} ms")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=6000,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": "Purdue D1 LN2 wake reproduction"},
    )
    with writer.saving(figure, str(output), dpi=120):
        for index, instant in enumerate(times):
            for image, (values, *_rest) in zip(images, panels):
                image.set_data(values[index])
            timestamp.set_text(f"Purdue D1 LN2 spray — t = {instant * 1.0e3:.2f} ms")
            writer.grab_frame()
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    arguments = parser.parse_args()
    render(arguments.archive, arguments.output, fps=arguments.fps)


if __name__ == "__main__":
    main()
