"""Render the periodic low-Mach ABL extension as a two-panel MP4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from jaxwind.config.low_mach import load_case


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=12)
    arguments = parser.parse_args(argv)
    if arguments.fps <= 0:
        raise ValueError("fps must be positive")
    case = load_case(arguments.config)
    source = case.output / "flow_frames.npz"
    if not source.is_file():
        raise FileNotFoundError(f"missing low-Mach frames: {source}")
    with np.load(source) as archive:
        horizontal = np.asarray(archive["u_hub_yx"])
        vertical = np.asarray(archive["u_center_zx"])
        times = np.asarray(archive["time_seconds"])
    from jaxwind.config.stages import load_workflow
    from tools.render_fv_wake_gif import _render_two_panel

    grid = load_workflow(case.source_workflow).case.physical.physical_grid
    output = (
        arguments.output
        if arguments.output is not None
        else case.output / "low_mach_extension_velocity.mp4"
    )
    return _render_two_panel(
        horizontal,
        vertical,
        times,
        grid,
        None,
        output,
        arguments.fps,
        "HITSZ periodic low-Mach ABL extension",
    )


if __name__ == "__main__":
    raise SystemExit(main())
