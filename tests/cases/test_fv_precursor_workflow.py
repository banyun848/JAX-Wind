from __future__ import annotations

from pathlib import Path

import pytest

from jaxwind.config.stages import load_workflow
from jaxwind.config.low_mach import load_case as load_low_mach_extension


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "cases" / "Andren1994" / "config.toml"
DTU_CONFIG = ROOT / "cases" / "DTU10MWPrecursor" / "fv_workflow.toml"
HITSZ_CONFIG = ROOT / "cases" / "HITSZWindTunnel" / "fv_workflow.toml"
HITSZ_COOLED_CONFIG = (
    ROOT / "cases" / "HITSZWindTunnel" / "fv_far_wake_cooled.toml"
)
HITSZ_ROUND_JET_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_uniform_512x128x64_l24_alm_x6_ln2_round_10s.toml"
)
HITSZ_FINE_WARMUP_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_uniform_1024x256x128_l24_warmup_plus10s.toml"
)
HITSZ_1024_FFT_PRECURSOR_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_uniform_1024x256x128_l24_warmup1000s_precursor100s_fft.toml"
)
HITSZ_LOW_MACH_EXTENSION_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_low_mach_extension_1024x256x128_100s.toml"
)
HITSZ_LOW_MACH_PRECURSOR_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_low_mach_precursor_1024x256x128_100s.toml"
)
HITSZ_LOW_MACH_SANITY_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_low_mach_warmup_256x64x32_900s.toml"
)
HITSZ_LOW_MACH_FINE_CONTINUATION_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_low_mach_continuation_512x128x64_300s.toml"
)
HITSZ_LOW_MACH_FINE_FRESH_WARMUP_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_low_mach_warmup_512x128x64_900s_fresh.toml"
)
HITSZ_LOW_MACH_REFINED_Z_FRESH_WARMUP_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_low_mach_warmup_512x128x256_900s_fresh.toml"
)
HITSZ_CLUSTERED_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_warmup_clustered_256x128x64.toml"
)
HITSZ_ALM_CONFIG = (
    ROOT
    / "cases"
    / "HITSZWindTunnel"
    / "fv_uniform_512x128x64_l24_alm_x4_10s.toml"
)


def test_one_toml_resolves_the_three_distinct_fv_stages() -> None:
    workflow = load_workflow(CONFIG)
    result = workflow.resolved()

    assert result["warmup"]["pressure_backend"] == "fft"
    assert result["warmup"]["periodic_x"] is True
    assert result["precursor"]["pressure_backend"] == "fft"
    assert result["precursor"]["stored_x_layers_per_sample"] == 1
    assert result["precursor"]["sample_every_steps"] == 1
    assert result["main"]["pressure_backend"] == "gmg"
    assert result["main"]["periodic_x"] is False
    assert result["main"]["x_velocity_faces"] == 41
    assert result["main"]["outflow"].startswith("second-order")


def test_dtu_configuration_adds_adbem_only_to_the_open_main_stage() -> None:
    workflow = load_workflow(DTU_CONFIG)
    result = workflow.resolved()

    assert result["warmup"]["pressure_backend"] == "fft"
    assert result["precursor"]["stored_x_layers_per_sample"] == 1
    assert result["main"]["pressure_backend"] == "gmg"
    assert result["main"]["pressure_force"] is False
    assert result["main"]["x_velocity_faces"] == 129
    assert result["turbine"]["model"] == "openfast-ad-bem"
    assert result["turbine"]["rotor_speed_rpm"] == 9.6


def test_main_cannot_outlive_the_recorded_precursor(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "main_steps = 4500",
            "main_steps = 4501",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="main_steps cannot exceed"):
        load_workflow(invalid)


def test_hitsz_main_is_fixed_fast_rk3_with_native_turbine_and_frames() -> None:
    result = load_workflow(HITSZ_CONFIG).resolved()

    assert result["main"]["dt_seconds"] == 0.01125
    assert result["main"]["duration_seconds"] == 90.0
    assert result["main"]["time_integration"] == "fast-rk3"
    assert result["main"]["frame_count"] == 100
    assert result["turbine"]["model"] == "hitsz-r9-ad-bem"
    assert result["turbine"]["rotor_speed_rpm"] == 480.0

    alm = load_workflow(HITSZ_ALM_CONFIG).resolved()
    assert alm["main"]["duration_seconds"] == 10.0
    assert alm["main"]["steps"] == 4000
    assert alm["input_directory"].endswith("512x128x64_l24")
    assert alm["turbine"]["model"] == "hitsz-r9-alm"
    assert alm["turbine"]["x_m"] == 4.0
    assert alm["turbine"]["smoothing_width_chord_factor"] == 0.5
    assert alm["turbine"]["smearing_azimuthal_elements"] is None

    cooled = load_workflow(HITSZ_COOLED_CONFIG).resolved()
    assert cooled["input_directory"].endswith(
        "outputs/hitsz_r9_fv_workflow_256x64x128"
    )
    assert cooled["main"]["evolve_scalar"] is True
    assert cooled["main"]["substeps_per_inflow"] == 2
    assert cooled["main"]["dt_seconds"] == 0.005625
    assert cooled["main"]["inflow_dt_seconds"] == 0.01125
    assert cooled["cooling"]["model"] == (
        "conservative-gaussian-temperature-sink"
    )
    assert cooled["cooling"]["cooling_power_w"] == pytest.approx(4_842.570359)

    round_jet = load_workflow(HITSZ_ROUND_JET_CONFIG).resolved()
    assert round_jet["main"]["duration_seconds"] == 10.0
    assert round_jet["main"]["evolve_scalar"] is True
    assert round_jet["turbine"]["x_m"] == 6.0
    assert round_jet["cooling"]["model"] == (
        "conservative-unresolved-round-jet"
    )
    assert round_jet["cooling"]["exit_vapor_quality"] == pytest.approx(
        0.02238146379800278
    )
    assert round_jet["cooling"]["exit_vapor_mass_flow_rate_kg_s"] == (
        pytest.approx(0.00027976829747503474)
    )
    assert round_jet["cooling"]["exit_liquid_mass_flow_rate_kg_s"] == (
        pytest.approx(0.012220231702524967)
    )
    assert round_jet["cooling"]["cooling_power_w"] == pytest.approx(
        5_328.605750508923
    )
    assert round_jet["cooling"]["axial_momentum_flux_n"] == pytest.approx(
        0.05
    )
    assert round_jet["cooling"]["nozzle_diameter_m"] == 0.005
    assert round_jet["cooling"]["cone_half_angle_degrees"] == 0.0

    fine_warmup = load_workflow(HITSZ_FINE_WARMUP_CONFIG).resolved()
    assert fine_warmup["case"]["cells"] == [1024, 256, 128]
    assert fine_warmup["warmup"]["duration_seconds"] == 10.0
    assert fine_warmup["warmup"]["pressure_backend"] == "fft"
    assert fine_warmup["warmup"]["restart_checkpoint"].endswith(
        "prolonged_warmup_initial.npz"
    )

    fft_precursor = load_workflow(HITSZ_1024_FFT_PRECURSOR_CONFIG).resolved()
    assert fft_precursor["case"]["cells"] == [1024, 256, 128]
    assert fft_precursor["case"]["grid_uniform"] is True
    assert fft_precursor["warmup"]["pressure_backend"] == "fft"
    assert fft_precursor["warmup"]["duration_seconds"] == 1000.0
    assert fft_precursor["warmup"]["cfl_ceiling"] == 1.0
    assert fft_precursor["precursor"]["pressure_backend"] == "fft"
    assert fft_precursor["precursor"]["steps"] == 40000
    assert fft_precursor["precursor"]["duration_seconds"] == 100.0
    assert fft_precursor["precursor"]["dt_seconds"] == 0.0025
    assert fft_precursor["precursor"]["record_plane"] == 20
    assert fft_precursor["turbine"] is None

    low_mach = load_low_mach_extension(HITSZ_LOW_MACH_EXTENSION_CONFIG)
    assert low_mach.steps * low_mach.dt == 100.0
    assert low_mach.frame_count == 100
    assert low_mach.pressure_backend == "fft"
    assert low_mach.time_integration == "fast-rk3"
    assert low_mach.source_checkpoint.name == "checkpoint.npz"
    assert low_mach.restart_formulation == "incompressible"

    precursor = load_low_mach_extension(HITSZ_LOW_MACH_PRECURSOR_CONFIG)
    assert precursor.steps * precursor.dt == 100.0
    assert precursor.frame_count == 100
    assert precursor.restart_formulation == "low-mach"
    assert precursor.source_checkpoint.name == "checkpoint.npz"

    sanity = load_low_mach_extension(HITSZ_LOW_MACH_SANITY_CONFIG)
    assert sanity.steps * sanity.dt == 900.0
    assert sanity.frame_count == 100
    assert sanity.restart_formulation == "configured"
    assert sanity.source_checkpoint is None
    sanity_workflow = load_workflow(sanity.source_workflow)
    assert sanity_workflow.resolved()["case"]["cells"] == [256, 64, 32]

    fine = load_low_mach_extension(
        HITSZ_LOW_MACH_FINE_CONTINUATION_CONFIG
    )
    assert fine.steps * fine.dt == 300.0
    assert fine.statistics_window_seconds == 60.0
    assert fine.restart_formulation == "low-mach"
    fine_workflow = load_workflow(fine.source_workflow)
    assert fine_workflow.resolved()["case"]["cells"] == [512, 128, 64]

    fresh_fine = load_low_mach_extension(
        HITSZ_LOW_MACH_FINE_FRESH_WARMUP_CONFIG
    )
    assert fresh_fine.steps * fresh_fine.dt == 900.0
    assert fresh_fine.frame_count == 300
    assert fresh_fine.statistics_window_seconds == 60.0
    assert fresh_fine.restart_formulation == "configured"
    assert fresh_fine.source_checkpoint is None
    fresh_fine_workflow = load_workflow(fresh_fine.source_workflow)
    assert fresh_fine_workflow.resolved()["case"]["cells"] == [512, 128, 64]

    refined_z = load_low_mach_extension(
        HITSZ_LOW_MACH_REFINED_Z_FRESH_WARMUP_CONFIG
    )
    assert refined_z.steps * refined_z.dt == 900.0
    assert refined_z.frame_count == 300
    assert refined_z.statistics_window_seconds == 60.0
    assert refined_z.restart_formulation == "configured"
    assert refined_z.source_checkpoint is None
    refined_z_workflow = load_workflow(refined_z.source_workflow)
    assert refined_z_workflow.resolved()["case"]["cells"] == [512, 128, 256]


def test_clustered_hitsz_toml_reproduces_the_cell_average_wall_law() -> None:
    import math

    import jax.numpy as jnp
    import numpy as np

    from jaxwind.io.initial_conditions import load_initial_profile
    from jaxwind.domain import AnalyticalGrid
    from jaxwind import (
        MoninObukhovWall,
        StaggeredVelocity,
        logarithmic_profile,
        surface_stress,
    )

    from jaxwind.simulation.abl import build_models

    workflow = load_workflow(HITSZ_CLUSTERED_CONFIG)
    case = workflow.case.physical
    grid = case.physical_grid
    result = workflow.resolved()

    assert isinstance(grid, AnalyticalGrid)
    assert (grid.nx, grid.ny, grid.nz) == (256, 128, 64)
    assert result["warmup"]["pressure_backend"] == "fft"
    assert result["precursor"]["duration_seconds"] == 90.0
    assert result["precursor"]["frame_count"] == 100
    assert result["case"]["gmg_tolerance"] == pytest.approx(1.0e-5)
    assert result["case"]["gmg_presweeps"] == 4
    assert result["case"]["gmg_postsweeps"] == 4
    assert result["case"]["gmg_anisotropy_aware"] is False
    assert result["case"]["grid_uniform"] is False
    assert result["case"]["minimum_cell_widths_m"][2] == pytest.approx(
        0.017206770872983057
    )
    assert result["case"]["maximum_cell_widths_m"][2] == pytest.approx(
        0.09319970903140984
    )
    _boundaries, momentum, _scalar, _buoyancy, _surface = build_models(
        workflow.case,
        periodic_x=True,
    )
    assert momentum.surface.sampling == "cell-average"

    table = load_initial_profile(case)
    np.testing.assert_allclose(
        table["z_m"], grid.z_centers, rtol=0.0, atol=1.0e-12
    )
    friction = math.sqrt(0.004198491620939015 * grid.lz)
    wall = MoninObukhovWall(1.6100320416141182e-5, von_karman=0.4)
    expected = np.asarray(logarithmic_profile(grid, friction, wall))
    np.testing.assert_allclose(table["u_m_s"], expected, rtol=5.0e-6)

    first_speed = float(table["u_m_s"][0])
    flow = StaggeredVelocity(
        jnp.full((1, 2, 2), first_speed),
        jnp.zeros((1, 2, 2)),
        jnp.zeros((2, 2, 2)),
    )
    stress_x, stress_y = surface_stress(flow, grid, wall)
    np.testing.assert_allclose(stress_x, friction**2, rtol=2.0e-6)
    np.testing.assert_allclose(stress_y, 0.0, atol=1.0e-12)


def test_fixed_warmup_blocks_anchor_float32_time_to_step_count() -> None:
    from typing import NamedTuple

    import jax.numpy as jnp

    from jaxwind.runtime.periodic import run_periodic_blocks
    from jaxwind.domain import UniformGrid
    from jaxwind import StaggeredVelocity

    class State(NamedTuple):
        velocity: StaggeredVelocity
        time: object

    grid = UniformGrid(2, 2, 2, 1.0, 1.0, 1.0)
    zeros = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    state = State(
        StaggeredVelocity(
            zeros,
            zeros,
            jnp.zeros((3, 2, 2), dtype=jnp.float32),
        ),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    def advance(current, dt, count):
        accumulated = current.time
        for _ in range(count):
            accumulated = accumulated + jnp.asarray(dt, accumulated.dtype)
        return current._replace(time=accumulated)

    result, _elapsed, _final_cfl, _maximum_cfl = run_periodic_blocks(
        state,
        advance,
        grid=grid,
        dt=0.01,
        steps=1001,
        chunk=1000,
    )

    assert float(result.time) == pytest.approx(10.01, abs=1.0e-6)


def test_far_wake_analyzer_reports_downward_cooling_shift(
    tmp_path: Path,
) -> None:
    import numpy as np

    from tools.compare_ln2_far_wakes import compare_far_wakes

    x_m = np.arange(6, dtype=np.float32) + 0.5
    y_m = np.arange(2, dtype=np.float32) + 0.5
    z_m = np.asarray((0.5, 1.0, 1.5), dtype=np.float32)
    baseline = np.full((4, 3, 6), 3.0, dtype=np.float32)
    cooled = baseline.copy()
    baseline[:, 1, 2:] -= 0.5
    cooled[:, 0, 2:] -= 0.7
    temperature = np.zeros_like(cooled)
    temperature[:, 0, 2:] = -2.0
    common = {
        "x_m": x_m,
        "y_m": y_m,
        "z_m": z_m,
        "u_hub_yx": np.zeros((4, 2, 6), dtype=np.float32),
    }
    baseline_path = tmp_path / "baseline.npz"
    cooled_path = tmp_path / "cooled.npz"
    np.savez(
        baseline_path,
        **common,
        u_center_zx=baseline,
    )
    np.savez(
        cooled_path,
        **common,
        u_center_zx=cooled,
        scalar_center_zx=temperature,
    )

    result = compare_far_wakes(
        baseline_path,
        cooled_path,
        tmp_path / "comparison",
        rotor_x_m=2.0,
        rotor_diameter_m=1.0,
        hub_height_m=1.0,
        stations_d=(1.0, 2.0),
    )

    assert all(row["wake_center_shift_m"] < 0.0 for row in result["stations"])
    assert all(
        row["minimum_temperature_anomaly_k"] == -2.0
        for row in result["stations"]
    )
    assert (tmp_path / "comparison" / "far_wake_comparison.png").exists()


def test_main_frame_capture_matches_full_field_host_interpolation() -> None:
    from types import SimpleNamespace

    import jax.numpy as jnp
    import numpy as np

    from jaxwind.runtime.frames import capture_frame
    from jaxwind.domain import UniformGrid

    grid = UniformGrid(8, 4, 6, 8.0, 4.0, 3.0)
    x_faces = jnp.arange(
        grid.nz * grid.ny * (grid.nx + 1), dtype=jnp.float32
    ).reshape(grid.nz, grid.ny, grid.nx + 1)
    scalar = jnp.arange(
        grid.nz * grid.ny * grid.nx, dtype=jnp.float32
    ).reshape(grid.nz, grid.ny, grid.nx)
    solution = SimpleNamespace(
        velocity=SimpleNamespace(x=x_faces),
        scalar=scalar,
        time=jnp.asarray(1.25),
        step=jnp.asarray(5),
    )
    y_m = 1.7
    z_m = 1.1
    result = capture_frame(solution, grid, y_m=y_m, z_m=z_m)

    host_faces = np.asarray(x_faces)
    u_cell = 0.5 * (host_faces[..., :-1] + host_faces[..., 1:])
    z_index = np.clip(z_m / grid.dz - 0.5, 0.0, grid.nz - 1.0)
    z_lower = int(np.floor(z_index))
    z_upper = min(z_lower + 1, grid.nz - 1)
    z_weight = z_index - z_lower
    expected_hub = (
        (1.0 - z_weight) * u_cell[z_lower] + z_weight * u_cell[z_upper]
    )
    y_index = y_m / grid.dy - 0.5
    y_floor = np.floor(y_index)
    y_lower = int(y_floor) % grid.ny
    y_upper = (y_lower + 1) % grid.ny
    y_weight = y_index - y_floor
    expected_centre = (
        (1.0 - y_weight) * u_cell[:, y_lower]
        + y_weight * u_cell[:, y_upper]
    )

    np.testing.assert_allclose(result["u_hub_yx"], expected_hub, rtol=1e-6)
    np.testing.assert_allclose(
        result["u_center_zx"], expected_centre, rtol=1e-6
    )
    host_scalar = np.asarray(scalar)
    expected_scalar_hub = (
        (1.0 - z_weight) * host_scalar[z_lower]
        + z_weight * host_scalar[z_upper]
    )
    expected_scalar_centre = (
        (1.0 - y_weight) * host_scalar[:, y_lower]
        + y_weight * host_scalar[:, y_upper]
    )
    np.testing.assert_allclose(result["scalar_hub_yx"], expected_scalar_hub)
    np.testing.assert_allclose(
        result["scalar_center_zx"], expected_scalar_centre
    )
    periodic_faces = jnp.arange(
        grid.nz * grid.ny * grid.nx, dtype=jnp.float32
    ).reshape(grid.nz, grid.ny, grid.nx)
    periodic_solution = SimpleNamespace(
        velocity=SimpleNamespace(x=periodic_faces),
        scalar=scalar,
        time=jnp.asarray(1.25),
        step=jnp.asarray(5),
    )
    periodic = capture_frame(
        periodic_solution, grid, y_m=y_m, z_m=z_m
    )
    host_periodic = np.asarray(periodic_faces)
    periodic_cells = 0.5 * (
        host_periodic + np.roll(host_periodic, -1, axis=-1)
    )
    expected_periodic_hub = (
        (1.0 - z_weight) * periodic_cells[z_lower]
        + z_weight * periodic_cells[z_upper]
    )
    expected_periodic_centre = (
        (1.0 - y_weight) * periodic_cells[:, y_lower]
        + y_weight * periodic_cells[:, y_upper]
    )
    assert periodic["u_hub_yx"].shape == (grid.ny, grid.nx)
    assert periodic["u_center_zx"].shape == (grid.nz, grid.nx)
    np.testing.assert_allclose(
        periodic["u_hub_yx"], expected_periodic_hub, rtol=1e-6
    )
    np.testing.assert_allclose(
        periodic["u_center_zx"], expected_periodic_centre, rtol=1e-6
    )

    assert result["time_seconds"] == 1.25
    assert result["step"] == 5
