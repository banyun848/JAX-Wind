#!/usr/bin/env python3
"""Prolong a periodic incompressible or low-Mach FV checkpoint.

Each MAC velocity component is linearly interpolated at its native face
coordinates. Cell fields are trilinearly interpolated, with periodic x/y and
wall-clamped z. The fine velocity is then projected with the target FFT
Poisson operator. The physical clock and fast-RK3 history are retained so a
warmup can continue from the prolonged state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.fv_abl.workflow import _models, load_workflow
from tools.prolong_pressure_driven_checkpoint import (
    _resample_axis,
    prolong_cell_field,
    prolong_vertical_faces,
)


def prolong_periodic_face_field(
    values: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    face_axis: int,
) -> np.ndarray:
    """Interpolate a periodic MAC field with faces along x or y."""

    if values.ndim != 3 or face_axis not in (1, 2):
        raise ValueError("periodic face field must be 3-D with face_axis 1 or 2")
    result = values
    axes = ((2, target_shape[2]), (1, target_shape[1]), (0, target_shape[0]))
    for axis, target_size in axes:
        if axis == face_axis:
            closed = np.concatenate(
                (result, np.take(result, [0], axis=axis)), axis=axis
            )
            result = _resample_axis(
                closed,
                target_size,
                axis=axis,
                cell_centred=False,
                periodic=False,
            )
            result = np.take(result, np.arange(target_size), axis=axis)
        else:
            result = _resample_axis(
                result,
                target_size,
                axis=axis,
                cell_centred=True,
                periodic=axis in (1, 2),
            )
    return result


def _prolong_vertical_field(
    values: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    if values.shape[0] < 2:
        raise ValueError("vertical face field must include both walls")
    upper, lower = prolong_vertical_faces(
        values[1:], values[0], target_shape
    )
    return np.concatenate((lower[None, ...], upper), axis=0)


def _same_domain(source_grid, target_grid) -> bool:
    return all(
        np.isclose(getattr(source_grid, name), getattr(target_grid, name))
        for name in ("lx", "ly", "lz")
    )


def prolong_fv_checkpoint(
    source_checkpoint: Path,
    source_config: Path,
    target_config: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    source_workflow = load_workflow(source_config)
    target_workflow = load_workflow(target_config)
    source_grid = source_workflow.case.physical.physical_grid
    target_grid = target_workflow.case.physical.physical_grid
    source_shape = (source_grid.nz, source_grid.ny, source_grid.nx)
    target_shape = (target_grid.nz, target_grid.ny, target_grid.nx)
    if not source_grid.is_uniform or not target_grid.is_uniform:
        raise ValueError("FV checkpoint prolongation currently requires uniform grids")
    if not _same_domain(source_grid, target_grid):
        raise ValueError("source and target physical domains differ")
    refinement = tuple(
        target // coarse for target, coarse in zip(target_shape, source_shape)
    )
    if any(
        target != factor * coarse or factor < 2
        for target, coarse, factor in zip(
            target_shape, source_shape, refinement, strict=True
        )
    ):
        raise ValueError("each target axis must be an integer refinement")

    with np.load(source_checkpoint, allow_pickle=False) as archive:
        u = np.asarray(archive["velocity_x"])
        v = np.asarray(archive["velocity_y"])
        w = np.asarray(archive["velocity_z"])
        if u.shape != source_shape or v.shape != source_shape:
            raise ValueError("source periodic horizontal velocity shape is invalid")
        if w.shape != (source_grid.nz + 1, source_grid.ny, source_grid.nx):
            raise ValueError("source vertical velocity shape is invalid")
        fine_u = prolong_periodic_face_field(u, target_shape, face_axis=2)
        fine_v = prolong_periodic_face_field(v, target_shape, face_axis=1)
        fine_w = _prolong_vertical_field(w, target_shape)
        source_time = np.asarray(archive["time"])
        source_step = np.asarray(archive["step"])

        import jax

        dtype_name = target_workflow.case.physical.pressure.dtype
        jax.config.update("jax_enable_x64", dtype_name == "float64")
        import jax.numpy as jnp
        from jaxwind.fv import (
            StaggeredVelocity,
            build_pressure_poisson,
            divergence,
            project,
        )

        dtype = jnp.dtype(dtype_name)
        candidate = StaggeredVelocity(
            jnp.asarray(fine_u, dtype),
            jnp.asarray(fine_v, dtype),
            jnp.asarray(fine_w, dtype),
        )
        before = float(jnp.max(jnp.abs(divergence(candidate, target_grid))))
        poisson = build_pressure_poisson(
            target_grid, backend="fft", dtype=dtype_name
        )
        projected, pressure_correction = project(candidate, poisson, 1.0)
        projected, pressure_correction = jax.device_get(
            (projected, pressure_correction)
        )
        after = float(
            np.max(
                np.abs(
                    np.asarray(
                        divergence(
                            StaggeredVelocity(
                                jnp.asarray(projected.x),
                                jnp.asarray(projected.y),
                                jnp.asarray(projected.z),
                            ),
                            target_grid,
                        )
                    )
                )
            )
        )

        pressure = prolong_cell_field(
            np.asarray(archive["pressure"]), target_shape
        ) + np.asarray(pressure_correction)
        momentum_x = prolong_periodic_face_field(
            np.asarray(archive["momentum_tendency_x"]),
            target_shape,
            face_axis=2,
        )
        momentum_y = prolong_periodic_face_field(
            np.asarray(archive["momentum_tendency_y"]),
            target_shape,
            face_axis=1,
        )
        momentum_z = _prolong_vertical_field(
            np.asarray(archive["momentum_tendency_z"]), target_shape
        )
        scalar = prolong_cell_field(
            np.asarray(archive["scalar"]), target_shape
        )
        scalar_tendency = prolong_cell_field(
            np.asarray(archive["scalar_tendency"]), target_shape
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        velocity_x=np.asarray(projected.x),
        velocity_y=np.asarray(projected.y),
        velocity_z=np.asarray(projected.z),
        pressure=np.asarray(pressure),
        momentum_tendency_x=np.asarray(momentum_x),
        momentum_tendency_y=np.asarray(momentum_y),
        momentum_tendency_z=np.asarray(momentum_z),
        scalar=np.asarray(scalar),
        scalar_tendency=np.asarray(scalar_tendency),
        time=source_time,
        step=source_step,
    )
    report: dict[str, object] = {
        "schema": "jaxwind.fv-checkpoint-prolongation.v1",
        "source_checkpoint": str(source_checkpoint),
        "source_config": str(source_config),
        "target_config": str(target_config),
        "output": str(output),
        "source_shape_zyx": list(source_shape),
        "target_shape_zyx": list(target_shape),
        "refinement_zyx": list(refinement),
        "time_seconds": float(source_time),
        "step": int(source_step),
        "velocity_transfer": (
            "native-MAC-face trilinear periodic-xy/clamped-z plus FFT projection"
        ),
        "cell_transfer": "trilinear periodic-xy/clamped-z",
        "maximum_divergence_before_projection_s": before,
        "maximum_divergence_after_projection_s": after,
        "integrator_history": "interpolated from source grid",
    }
    report_path = output.with_suffix(output.suffix + ".prolongation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report



def prolong_low_mach_checkpoint(
    source_checkpoint: Path,
    source_config: Path,
    target_config: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Prolong every field in a periodic low-Mach checkpoint."""

    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    source_workflow = load_workflow(source_config)
    target_workflow = load_workflow(target_config)
    source_grid = source_workflow.case.physical.physical_grid
    target_grid = target_workflow.case.physical.physical_grid
    source_shape = (source_grid.nz, source_grid.ny, source_grid.nx)
    target_shape = (target_grid.nz, target_grid.ny, target_grid.nx)
    if not source_grid.is_uniform or not target_grid.is_uniform:
        raise ValueError("low-Mach prolongation requires uniform grids")
    if not _same_domain(source_grid, target_grid):
        raise ValueError("source and target physical domains differ")
    refinement = tuple(
        target // coarse for target, coarse in zip(target_shape, source_shape)
    )
    dimensions = zip(target_shape, source_shape, refinement, strict=True)
    if any(
        target != factor * coarse or factor < 2
        for target, coarse, factor in dimensions
    ):
        raise ValueError("each target axis must be an integer refinement")

    required = {
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "pressure_dynamic_pa",
        "density",
        "momentum_tendency_x",
        "momentum_tendency_y",
        "momentum_tendency_z",
        "temperature",
        "water_vapor_mass_fraction",
        "nitrogen_mass_fraction",
        "thermodynamic_pressure_pa",
        "time",
        "step",
    }
    with np.load(source_checkpoint, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(
                "source is not a complete low-Mach checkpoint: "
                + ", ".join(sorted(missing))
            )
        u = np.asarray(archive["velocity_x"])
        v = np.asarray(archive["velocity_y"])
        w = np.asarray(archive["velocity_z"])
        if u.shape != source_shape or v.shape != source_shape:
            raise ValueError("source horizontal velocity shape is invalid")
        if w.shape != (source_grid.nz + 1, source_grid.ny, source_grid.nx):
            raise ValueError("source vertical velocity shape is invalid")

        fine_u = prolong_periodic_face_field(u, target_shape, face_axis=2)
        fine_v = prolong_periodic_face_field(v, target_shape, face_axis=1)
        fine_w = _prolong_vertical_field(w, target_shape)
        source_wall = _models(
            source_workflow.case, periodic_x=True, evolve_scalar=False
        )[1].surface
        target_wall = _models(
            target_workflow.case, periodic_x=True, evolve_scalar=False
        )[1].surface
        if source_wall is None or target_wall is None:
            raise ValueError("low-Mach prolongation requires wall models")
        wall_speed_scale = (
            source_wall.drag_coefficient(source_grid)
            / target_wall.drag_coefficient(target_grid)
        ) ** 0.5
        # The first fine centre lies below the coarse data range. Scale it by
        # the exact discrete wall law, depositing the removed momentum in its
        # sibling so the first coarse control volume remains conservative.
        for component in (fine_u, fine_v):
            transfer = component[0] * (1.0 - wall_speed_scale)
            component[0] *= wall_speed_scale
            component[1] += transfer
        density = prolong_cell_field(
            np.asarray(archive["density"]), target_shape
        )
        temperature = prolong_cell_field(
            np.asarray(archive["temperature"]), target_shape
        )
        water_vapor = prolong_cell_field(
            np.asarray(archive["water_vapor_mass_fraction"]), target_shape
        )
        nitrogen = prolong_cell_field(
            np.asarray(archive["nitrogen_mass_fraction"]), target_shape
        )
        pressure = prolong_cell_field(
            np.asarray(archive["pressure_dynamic_pa"]), target_shape
        )
        momentum_x = prolong_periodic_face_field(
            np.asarray(archive["momentum_tendency_x"]),
            target_shape,
            face_axis=2,
        )
        momentum_y = prolong_periodic_face_field(
            np.asarray(archive["momentum_tendency_y"]),
            target_shape,
            face_axis=1,
        )
        momentum_z = _prolong_vertical_field(
            np.asarray(archive["momentum_tendency_z"]), target_shape
        )
        source_time = np.asarray(archive["time"])
        source_step = np.asarray(archive["step"])
        thermodynamic_pressure = np.asarray(
            archive["thermodynamic_pressure_pa"]
        )

    import jax

    dtype_name = target_workflow.case.physical.pressure.dtype
    jax.config.update("jax_enable_x64", dtype_name == "float64")
    import jax.numpy as jnp
    from jaxwind.fv import (
        StaggeredVelocity,
        build_pressure_poisson,
        continuity_residual,
        project_low_mach,
    )

    dtype = jnp.dtype(dtype_name)
    candidate = StaggeredVelocity(
        jnp.asarray(fine_u, dtype),
        jnp.asarray(fine_v, dtype),
        jnp.asarray(fine_w, dtype),
    )
    fine_density = jnp.asarray(density, dtype)
    poisson = build_pressure_poisson(
        target_grid, backend="fft", dtype=dtype_name
    )
    before = float(
        jnp.max(
            jnp.abs(
                continuity_residual(
                    candidate,
                    fine_density,
                    fine_density,
                    target_grid,
                    1.0,
                )
            )
        )
    )
    projected, pressure_correction = project_low_mach(
        candidate,
        fine_density,
        fine_density,
        poisson,
        1.0,
    )
    after = float(
        jnp.max(
            jnp.abs(
                continuity_residual(
                    projected,
                    fine_density,
                    fine_density,
                    target_grid,
                    1.0,
                )
            )
        )
    )
    projected, pressure_correction = jax.device_get(
        (projected, pressure_correction)
    )
    dynamic_pressure = pressure + np.asarray(pressure_correction)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        lengths_m=np.asarray((target_grid.lx, target_grid.ly, target_grid.lz)),
        x_faces_m=np.asarray(target_grid.x_faces),
        y_faces_m=np.asarray(target_grid.y_faces),
        z_faces_m=np.asarray(target_grid.z_faces),
        velocity_x=np.asarray(projected.x),
        velocity_y=np.asarray(projected.y),
        velocity_z=np.asarray(projected.z),
        pressure_dynamic_pa=np.asarray(dynamic_pressure),
        density=np.asarray(density),
        momentum_tendency_x=np.asarray(momentum_x),
        momentum_tendency_y=np.asarray(momentum_y),
        momentum_tendency_z=np.asarray(momentum_z),
        temperature=np.asarray(temperature),
        water_vapor_mass_fraction=np.asarray(water_vapor),
        nitrogen_mass_fraction=np.asarray(nitrogen),
        continuity_error_kg_m3_s=np.asarray(after, dtype=np.float32),
        thermodynamic_pressure_pa=thermodynamic_pressure,
        time=source_time,
        step=source_step,
    )
    report: dict[str, object] = {
        "schema": "jaxwind.fv-low-mach-checkpoint-prolongation.v1",
        "source_checkpoint": str(source_checkpoint),
        "source_config": str(source_config),
        "target_config": str(target_config),
        "output": str(output),
        "source_shape_zyx": list(source_shape),
        "target_shape_zyx": list(target_shape),
        "refinement_zyx": list(refinement),
        "time_seconds": float(source_time),
        "step": int(source_step),
        "velocity_transfer": (
            "native-MAC-face trilinear interpolation plus "
            "density-weighted FFT projection"
        ),
        "wall_speed_scale": wall_speed_scale,
        "wall_transfer": (
            "first child follows discrete wall law; paired child conserves "
            "coarse-cell horizontal momentum"
        ),
        "thermodynamic_transfer": (
            "trilinear periodic-xy/clamped-z; continuation validates EOS"
        ),
        "maximum_mass_residual_before_projection_kg_m3_s": before,
        "maximum_mass_residual_after_projection_kg_m3_s": after,
        "integrator_history": "interpolated from source grid",
    }
    report_path = output.with_suffix(output.suffix + ".prolongation.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkpoint", type=Path)
    parser.add_argument("source_config", type=Path)
    parser.add_argument("target_config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--low-mach",
        action="store_true",
        help="prolong the complete variable-density low-Mach state",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    prolong = (
        prolong_low_mach_checkpoint
        if arguments.low_mach
        else prolong_fv_checkpoint
    )
    result = prolong(
        arguments.source_checkpoint,
        arguments.source_config,
        arguments.target_config,
        arguments.output,
        overwrite=arguments.overwrite,
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
