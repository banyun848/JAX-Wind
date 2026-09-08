#!/usr/bin/env python3
"""Render FV hub-height main-flow frames as an animated wake GIF or MP4."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def _ffmpeg_executable() -> str:
    """Return the bundled encoder when available, otherwise system FFmpeg."""

    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError(
                "MP4 rendering requires imageio-ffmpeg or ffmpeg on PATH"
            ) from None
        return executable
    return imageio_ffmpeg.get_ffmpeg_exe()


def _render_two_panel(
    horizontal,
    vertical,
    times,
    grid,
    disk,
    output,
    fps,
    label,
):
    if horizontal.shape[0] != vertical.shape[0] or horizontal.shape[0] == 0:
        raise ValueError("horizontal and vertical archives must share frames")
    combined = np.concatenate((horizontal.ravel(), vertical.ravel()))
    lower, upper = np.nanpercentile(combined, (1.0, 99.0))
    figure, axes = plt.subplots(
        2, 1, figsize=(12.8, 8.6), constrained_layout=True
    )
    panels = (
        (
            horizontal,
            np.asarray(grid.y_faces),
            "Hub-height x-y plane",
            "y [m]",
            None if disk is None else disk.y,
        ),
        (
            vertical,
            np.asarray(grid.z_faces),
            "Centerline x-z plane",
            "z [m]",
            None if disk is None else disk.z,
        ),
    )
    x_faces = np.asarray(grid.x_faces)
    images = []
    for axis, (values, transverse_faces, title, ylabel, disk_center) in zip(
        axes, panels, strict=True
    ):
        image = axis.pcolormesh(
            x_faces,
            transverse_faces,
            values[0],
            shading="flat",
            cmap="turbo",
            vmin=float(lower),
            vmax=float(upper),
            rasterized=True,
        )
        images.append(image)
        axis.set(
            xlim=(float(x_faces[0]), float(x_faces[-1])),
            ylim=(float(transverse_faces[0]), float(transverse_faces[-1])),
            xlabel="x [m]",
            ylabel=ylabel,
            title=title,
            aspect="equal",
        )
        if disk is not None:
            axis.plot(
                [disk.x, disk.x],
                [disk_center - disk.tip_radius, disk_center + disk.tip_radius],
                color="white",
                linewidth=2.2,
            )
            axis.plot(
                disk.x,
                disk_center,
                marker="+",
                color="black",
                markersize=7,
                markeredgewidth=1.5,
            )
    colorbar = figure.colorbar(images[0], ax=axes, pad=0.02, shrink=0.92)
    colorbar.set_label("u [m s^-1]")
    timestamp = figure.suptitle(f"{label} — t={times[0]:.2f} s")

    def update(index):
        images[0].set_array(horizontal[index].ravel())
        images[1].set_array(vertical[index].ravel())
        timestamp.set_text(f"{label} — t={times[index]:.2f} s")
        return *images, timestamp

    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure,
        update,
        frames=horizontal.shape[0],
        interval=1000.0 / fps,
        blit=False,
    )
    if output.suffix.lower() == ".mp4":
        matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_executable()
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=8000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            metadata={"title": f"{label} x-y and x-z"},
        )
    else:
        writer = animation.PillowWriter(fps=fps)
    movie.save(output, writer=writer, dpi=120)
    plt.close(figure)
    print(f"wrote {output} ({horizontal.shape[0]} two-panel frames)")
    return 0


def _render_four_panel(
    horizontal_velocity,
    vertical_velocity,
    horizontal_temperature,
    vertical_temperature,
    times,
    grid,
    disk,
    output,
    fps,
    label,
):
    fields = (
        horizontal_velocity,
        vertical_velocity,
        horizontal_temperature,
        vertical_temperature,
    )
    if any(field.shape[0] != times.size for field in fields) or times.size == 0:
        raise ValueError("all four panel archives must share nonempty frames")
    velocity_values = np.concatenate(
        (horizontal_velocity.ravel(), vertical_velocity.ravel())
    )
    temperature_values = np.concatenate(
        (horizontal_temperature.ravel(), vertical_temperature.ravel())
    )
    velocity_limits = np.nanpercentile(velocity_values, (1.0, 99.0))
    temperature_limits = (
        float(np.nanpercentile(temperature_values, 1.0)),
        min(0.0, float(np.nanmax(temperature_values))),
    )
    if temperature_limits[0] == temperature_limits[1]:
        temperature_limits = (temperature_limits[0] - 1.0, temperature_limits[1])

    figure, axes = plt.subplots(
        2, 2, figsize=(16.0, 7.6), constrained_layout=True
    )
    x_faces = np.asarray(grid.x_faces)
    y_faces = np.asarray(grid.y_faces)
    z_faces = np.asarray(grid.z_faces)
    panels = (
        (
            horizontal_velocity,
            y_faces,
            "Velocity — hub-height x-y",
            "y [m]",
            "viridis",
            velocity_limits,
            None if disk is None else disk.y,
        ),
        (
            vertical_velocity,
            z_faces,
            "Velocity — centerline x-z",
            "z [m]",
            "viridis",
            velocity_limits,
            None if disk is None else disk.z,
        ),
        (
            horizontal_temperature,
            y_faces,
            "Temperature anomaly — hub-height x-y",
            "y [m]",
            "magma",
            temperature_limits,
            None if disk is None else disk.y,
        ),
        (
            vertical_temperature,
            z_faces,
            "Temperature anomaly — centerline x-z",
            "z [m]",
            "magma",
            temperature_limits,
            None if disk is None else disk.z,
        ),
    )
    images = []
    for axis, panel in zip(axes.flat, panels, strict=True):
        values, transverse_faces, title, ylabel, cmap, limits, disk_center = panel
        image = axis.pcolormesh(
            x_faces,
            transverse_faces,
            values[0],
            shading="flat",
            cmap=cmap,
            vmin=float(limits[0]),
            vmax=float(limits[1]),
            rasterized=True,
        )
        images.append(image)
        axis.set(
            xlim=(float(x_faces[0]), float(x_faces[-1])),
            ylim=(float(transverse_faces[0]), float(transverse_faces[-1])),
            xlabel="x [m]",
            ylabel=ylabel,
            title=title,
            aspect="equal",
        )
        if disk is not None:
            axis.plot(
                [disk.x, disk.x],
                [disk_center - disk.tip_radius, disk_center + disk.tip_radius],
                color="white",
                linewidth=1.8,
            )
            axis.plot(
                disk.x,
                disk_center,
                marker="+",
                color="black",
                markersize=6,
                markeredgewidth=1.3,
            )
    velocity_bar = figure.colorbar(
        images[0], ax=axes[0, :], pad=0.015, shrink=0.9
    )
    velocity_bar.set_label("u [m s$^{-1}$]")
    temperature_bar = figure.colorbar(
        images[2], ax=axes[1, :], pad=0.015, shrink=0.9
    )
    temperature_bar.set_label("temperature anomaly [K]")
    timestamp = figure.suptitle(f"{label} — t={times[0]:.2f} s")

    def update(index):
        for image, values in zip(images, fields, strict=True):
            image.set_array(values[index].ravel())
        timestamp.set_text(f"{label} — t={times[index]:.2f} s")
        return *images, timestamp

    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure,
        update,
        frames=times.size,
        interval=1000.0 / fps,
        blit=False,
    )
    if output.suffix.lower() == ".mp4":
        matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_executable()
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=10000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            metadata={"title": f"{label} velocity and temperature"},
        )
    else:
        writer = animation.PillowWriter(fps=fps)
    movie.save(output, writer=writer, dpi=120)
    plt.close(figure)
    print(f"wrote {output} ({times.size} four-panel frames)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    panel_group = parser.add_mutually_exclusive_group()
    panel_group.add_argument(
        "--two-panel",
        action="store_true",
        help="render hub-height x-y and centerline x-z velocity panels",
    )
    panel_group.add_argument(
        "--four-panel",
        action="store_true",
        help="render x-y/x-z velocity and temperature-anomaly panels",
    )
    arguments = parser.parse_args()
    if arguments.fps <= 0:
        raise ValueError("--fps must be positive")

    from jaxwind.simulation.turbines import build_turbine_definition
    from jaxwind.config.stages import load_workflow
    from jaxwind.domain import ScaleSystem

    workflow = load_workflow(arguments.config)
    source = workflow.options.output_directory / "main/flow_frames.npz"
    if not source.exists():
        raise FileNotFoundError(f"missing FV main frames: {source}")
    with np.load(source) as archive:
        fields = np.asarray(archive["u_hub_yx"])
        vertical = np.asarray(archive["u_center_zx"])
        times = np.asarray(archive["time_seconds"])
        x = np.asarray(archive["x_m"])
        y = np.asarray(archive["y_m"])
        horizontal_temperature = (
            np.asarray(archive["scalar_hub_yx"])
            if arguments.four_panel
            else None
        )
        vertical_temperature = (
            np.asarray(archive["scalar_center_zx"])
            if arguments.four_panel
            else None
        )
    if fields.ndim != 3 or fields.shape[0] == 0:
        raise ValueError("u_hub_yx must contain at least one (y, x) frame")

    turbine = build_turbine_definition(workflow)
    disk = (
        None
        if turbine is None
        else turbine.to_actuator_disk(scales=ScaleSystem(1.0, 1.0))
    )
    label = (
        "HITSZ R9 "
        + (
            "uniform-grid "
            if workflow.case.physical.physical_grid.is_uniform
            else "mapped-grid "
        )
        + (
            "flow"
            if workflow.turbine is None
            else workflow.turbine.model.upper().replace("HITSZ-R9-", "")
        )
    )
    if arguments.four_panel:
        output = (
            arguments.output
            if arguments.output is not None
            else workflow.options.output_directory / "main_flow_four_panel.mp4"
        )
        return _render_four_panel(
            fields,
            vertical,
            horizontal_temperature,
            vertical_temperature,
            times,
            workflow.case.physical.physical_grid,
            disk,
            output,
            arguments.fps,
            label,
        )
    if arguments.two_panel:
        output = (
            arguments.output
            if arguments.output is not None
            else workflow.options.output_directory / "main_flow_two_panel.mp4"
        )
        return _render_two_panel(
            fields,
            vertical,
            times,
            workflow.case.physical.physical_grid,
            disk,
            output,
            arguments.fps,
            label,
        )

    lower, upper = np.nanpercentile(fields, (1.0, 99.0))
    figure, axis = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    image = axis.imshow(
        fields[0],
        origin="lower",
        extent=(
            float(x[0] - 0.5 * (x[1] - x[0])),
            float(x[-1] + 0.5 * (x[-1] - x[-2])),
            float(y[0] - 0.5 * (y[1] - y[0])),
            float(y[-1] + 0.5 * (y[-1] - y[-2])),
        ),
        aspect="equal",
        cmap="turbo",
        vmin=float(lower),
        vmax=float(upper),
        interpolation="bilinear",
    )
    if disk is not None:
        axis.plot(
            [disk.x, disk.x],
            [disk.y - disk.tip_radius, disk.y + disk.tip_radius],
            color="white",
            linewidth=2.2,
        )
        axis.plot(
            disk.x,
            disk.y,
            marker="+",
            color="black",
            markersize=7,
            markeredgewidth=1.5,
        )
    title = axis.set_title("")
    axis.set(
        xlabel="x [m]",
        ylabel="y [m]",
        title="HITSZ R9 FV-GMG hub-height streamwise velocity",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("u [m s$^{-1}$]")

    def update(index: int):
        image.set_data(fields[index])
        title.set_text(
            "HITSZ R9 FV-GMG hub-height streamwise velocity "
            f"(t={times[index]:.2f} s)"
        )
        return image, title

    output = (
        arguments.output
        if arguments.output is not None
        else workflow.options.output_directory / "main_flow.gif"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        figure,
        update,
        frames=fields.shape[0],
        interval=1000.0 / arguments.fps,
        blit=False,
    )
    if output.suffix.lower() == ".mp4":
        matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_executable()
        writer = animation.FFMpegWriter(
            fps=arguments.fps,
            codec="libx264",
            bitrate=6000,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            metadata={"title": "HITSZ R9 FV-GMG main flow"},
        )
    else:
        writer = animation.PillowWriter(fps=arguments.fps)
    movie.save(output, writer=writer, dpi=120)
    plt.close(figure)
    print(f"wrote {output} ({fields.shape[0]} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
