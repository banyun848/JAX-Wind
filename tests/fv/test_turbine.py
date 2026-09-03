from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxwind.domain import (
    AnalyticalGrid,
    ScaleSystem,
    TanhMapping,
    UniformGrid,
)
from jaxwind.fv import (
    StaggeredVelocity,
    build_adbem_forcing,
    build_actuator_line_forcing,
)
from jaxwind.windfarm import (
    HITSZR9BladeElementDisk,
    RigidBladeElementDisk,
    load_openfast_rigid_turbine,
)


ROOT = Path(__file__).resolve().parents[2]
OPENFAST = (
    ROOT
    / "tests"
    / "fixtures"
    / "openfast"
    / "nrel5mw"
    / "NREL5MW_Rigid_Smoke.fst"
)


def test_adbem_forcing_maps_shared_turbine_loads_to_open_fv_faces() -> None:
    grids = (
        UniformGrid(16, 8, 20, 320.0, 160.0, 240.0),
        AnalyticalGrid(
            16,
            8,
            20,
            320.0,
            160.0,
            240.0,
            x_mapping=TanhMapping(1.1, focus=0.375),
            y_mapping=TanhMapping(0.8, focus=0.5),
            z_mapping=TanhMapping(1.5, focus=0.0),
        ),
    )
    rotor = load_openfast_rigid_turbine(OPENFAST)
    turbine = RigidBladeElementDisk(
        rotor=rotor,
        x_m=120.0,
        y_m=80.0,
        smoothing_width_m=20.0,
        hub_height_m=90.0,
        rotor_speed_rpm=8.0,
        pitch_degrees=0.0,
        body_smoothing_width_m=20.0,
    )
    scales = ScaleSystem(1.0, 1.0)
    disk = turbine.to_actuator_disk(scales=scales)
    thrusts = []
    for grid in grids:
        forcing = build_adbem_forcing(grid, disk)
        velocity = StaggeredVelocity(
            jnp.full((grid.nz, grid.ny, grid.nx + 1), 10.0, jnp.float32),
            jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32),
            jnp.zeros((grid.nz + 1, grid.ny, grid.nx), jnp.float32),
        )

        result = jax.jit(forcing)(velocity, jnp.asarray(0.0, jnp.float32))

        assert result.x.shape == (grid.nz, grid.ny, grid.nx + 1)
        assert result.y.shape == (grid.nz, grid.ny, grid.nx)
        assert result.z.shape == (grid.nz + 1, grid.ny, grid.nx)
        assert bool(jnp.all(jnp.isfinite(result.x)))
        assert bool(jnp.all(jnp.isfinite(result.y)))
        assert bool(jnp.all(jnp.isfinite(result.z)))
        assert float(jnp.min(result.x)) < 0.0
        assert float(jnp.max(jnp.abs(result.y))) > 0.0
        assert float(jnp.max(jnp.abs(result.z))) > 0.0
        assert bool(jnp.all(result.z[0] == 0.0))
        assert bool(jnp.all(result.z[-1] == 0.0))

        x_widths = jnp.asarray(grid.x_widths)
        x_face_widths = jnp.concatenate(
            (
                0.5 * x_widths[:1],
                0.5 * (x_widths[:-1] + x_widths[1:]),
                0.5 * x_widths[-1:],
            )
        )
        face_volumes = (
            jnp.asarray(grid.z_widths)[:, None, None]
            * jnp.asarray(grid.y_widths)[None, :, None]
            * x_face_widths[None, None, :]
        )
        axial_load = -result.x * face_volumes
        thrusts.append(jnp.sum(axial_load))
        centroid = jnp.sum(
            axial_load * jnp.asarray(grid.z_centers)[:, None, None]
        ) / jnp.sum(axial_load)
        assert abs(float(centroid) - disk.z) < 5.0

    assert jnp.isclose(thrusts[0], thrusts[1], rtol=1.0e-3)



def test_actuator_line_forcing_rotates_and_conserves_load_on_mapped_grid() -> None:
    grids = (
        UniformGrid(24, 18, 20, 4.8, 3.6, 2.4),
        AnalyticalGrid(
            24,
            18,
            20,
            4.8,
            3.6,
            2.4,
            x_mapping=TanhMapping(0.8, focus=0.5),
            y_mapping=TanhMapping(0.7, focus=0.5),
            z_mapping=TanhMapping(0.9, focus=0.5),
        ),
    )
    turbine = HITSZR9BladeElementDisk(
        x_m=2.4,
        y_m=1.8,
        smoothing_width_m=0.2,
        hub_height_m=1.2,
    )
    line = turbine.to_actuator_line(
        scales=ScaleSystem(1.0, 1.0),
        initial_azimuth_degrees=11.0,
    )
    line = replace(
        line,
        element_gaussian_widths=tuple(
            0.5 * chord for chord in line.element_chords
        ),
    )
    assert len(line.point_smoothing_widths) == (
        line.blade_count * len(line.element_radii)
    )
    thrusts = []
    for grid in grids:
        forcing = jax.jit(build_actuator_line_forcing(grid, line))
        velocity = StaggeredVelocity(
            jnp.full((grid.nz, grid.ny, grid.nx + 1), 3.35, jnp.float32),
            jnp.zeros((grid.nz, grid.ny, grid.nx), jnp.float32),
            jnp.zeros((grid.nz + 1, grid.ny, grid.nx), jnp.float32),
        )
        at_start = forcing(velocity, jnp.asarray(0.0, jnp.float32))
        after_rotation = forcing(velocity, jnp.asarray(0.006, jnp.float32))

        assert at_start.x.shape == (grid.nz, grid.ny, grid.nx + 1)
        assert at_start.y.shape == (grid.nz, grid.ny, grid.nx)
        assert at_start.z.shape == (grid.nz + 1, grid.ny, grid.nx)
        assert bool(jnp.all(jnp.isfinite(at_start.x)))
        assert bool(jnp.all(jnp.isfinite(at_start.y)))
        assert bool(jnp.all(jnp.isfinite(at_start.z)))
        assert float(jnp.min(at_start.x)) < 0.0
        assert bool(jnp.all(at_start.z[0] == 0.0))
        assert bool(jnp.all(at_start.z[-1] == 0.0))
        assert not bool(jnp.allclose(at_start.y, after_rotation.y))

        x_widths = jnp.asarray(grid.x_widths)
        x_face_widths = jnp.concatenate(
            (
                0.5 * x_widths[:1],
                0.5 * (x_widths[:-1] + x_widths[1:]),
                0.5 * x_widths[-1:],
            )
        )
        face_volumes = (
            jnp.asarray(grid.z_widths)[:, None, None]
            * jnp.asarray(grid.y_widths)[None, :, None]
            * x_face_widths[None, None, :]
        )
        thrusts.append(jnp.sum(-at_start.x * face_volumes))

    assert jnp.isclose(thrusts[0], thrusts[1], rtol=1.0e-3)
