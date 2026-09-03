from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from applications.dlr_in1.run_benchmark import (
    build_report,
    load_protocol as load_dlr_protocol,
    profile_bins as dlr_bins,
    read_reference as read_dlr_reference,
    sample_sufficient_statistics as sample_dlr,
)
from applications.fv_ln2_jet.run import load_case
from applications.purdue_ln2_wake.plot_signed_comparison import (
    _comparison_rows as purdue_comparison_rows,
)
from applications.purdue_ln2_wake.run_experiment import (
    _error_summary as purdue_error_summary,
    _profile_bins as purdue_bins,
    _read_reference as read_purdue_reference,
    _sample_sufficient_statistics as sample_purdue,
    load_protocol as load_purdue_protocol,
)


def test_ln2_case_files_encode_their_distinct_benchmark_contracts() -> None:
    hitsz = load_case("cases/HITSZLiquidNitrogenJet/fv_256.toml")
    hitsz_incompressible = load_case(
        "cases/HITSZLiquidNitrogenJet/fv_256_incompressible_rk3.toml"
    )
    hitsz_inlet = load_case(
        "cases/HITSZLiquidNitrogenJet/fv_256_incompressible_rk3_inlet.toml"
    )
    hitsz_mapped = load_case(
        "cases/HITSZLiquidNitrogenJet/fv_256_low_mach_rk3_inlet_mapped.toml"
    )
    hitsz_flashed = load_case(
        "cases/HITSZLiquidNitrogenJet/"
        "fv_512x128x64_l24_pure_nitrogen_low_mach_1s.toml"
    )
    purdue = load_case("cases/PurdueLN2Wake/fv_256_experiment.toml")
    purdue_inlet = load_case(
        "cases/PurdueLN2Wake/fv_128_inlet_calibration.toml"
    )
    purdue_incompressible = load_case(
        "cases/PurdueLN2Wake/fv_64_incompressible_validation.toml"
    )
    purdue_incompressible_amd = load_case(
        "cases/PurdueLN2Wake/fv_128_incompressible_amd_validation.toml"
    )
    purdue_subgrid = load_case(
        "cases/PurdueLN2Wake/fv_128_incompressible_amd_subgrid_validation.toml"
    )
    purdue_protocol = load_purdue_protocol(
        "cases/PurdueLN2Wake/fv_256_experiment.toml"
    )
    purdue_reference = read_purdue_reference(purdue_protocol.reference)
    dlr = load_case("cases/DLRIN1/fv_256_source.toml")
    dlr_protocol = load_dlr_protocol("cases/DLRIN1/fv_256_source.toml")
    dlr_reference = read_dlr_reference(dlr_protocol.reference)

    assert hitsz.cells == (256, 256, 256)
    assert hitsz.lengths == (3.0, 3.0, 1.8)
    assert hitsz.nozzle == (0.75, 1.5, 0.876)
    assert hitsz.ambient_relative_humidity == 0.8
    assert hitsz.flow_formulation == "low-mach"
    assert hitsz.mass_flow_rate == 0.0125
    assert hitsz.time_integration == "rk3"
    assert hitsz_incompressible.cells == hitsz.cells
    assert hitsz_incompressible.dt == hitsz.dt
    assert hitsz_incompressible.steps == hitsz.steps
    assert hitsz_incompressible.flow_formulation == "incompressible"
    assert hitsz_incompressible.time_integration == "rk3"
    assert hitsz_inlet.cells == hitsz.cells
    assert hitsz_inlet.dt == hitsz.dt
    assert hitsz_inlet.source_mode == "inflow"
    assert hitsz_inlet.nozzle == (0.0, 1.5, 0.876)
    assert hitsz_inlet.flow_formulation == "incompressible"
    assert hitsz_mapped.mapping_types == ("tanh", "tanh", "tanh")
    assert hitsz_mapped.mapping_focus == (0.0, 1.5, 0.876)
    assert hitsz_mapped.mapping_strength == (2.2, 2.2, 2.2)
    assert hitsz_mapped.source_mode == "inflow"
    assert hitsz_mapped.flow_formulation == "low-mach"
    assert not hitsz_mapped.subgrid_jet_enabled
    assert hitsz.momentum_closure == "amd"

    assert hitsz_flashed.cells == (512, 128, 64)
    assert hitsz_flashed.lengths == (24.0, 6.0, 3.6)
    assert hitsz_flashed.nozzle == (6.3, 3.0, 0.876)
    assert hitsz_flashed.radius == 0.001
    assert abs(hitsz_flashed.speed - 4.9358940805812895) < 1.0e-14
    assert hitsz_flashed.mass_flow_rate == 0.0125
    assert hitsz_flashed.vapor_quality == 0.0
    assert hitsz_flashed.fully_vaporized_within_source_cell
    assert hitsz_flashed.flow_formulation == "low-mach"
    assert hitsz_flashed.scalar_advection_scheme == "upwind"
    assert hitsz_flashed.steps * hitsz_flashed.dt == 1.0

    assert purdue.gravity == (9.81, 0.0, 0.0)
    assert purdue.nozzle == (0.02, 0.1, 0.1)
    assert purdue.steps == 8000
    assert purdue_protocol.spinup_steps == 4000
    assert purdue_protocol.visualization_frames == 100
    assert len(purdue_bins(purdue_reference)) == 7
    assert purdue_inlet.cells == (128, 128, 128)
    assert purdue_inlet.nozzle == (0.0, 0.1, 0.1)
    assert purdue_inlet.source_mode == "inflow"
    assert purdue_inlet.radius == 0.0003935
    assert abs(purdue_inlet.speed - 18.11951185) < 1.0e-10
    assert purdue_inlet.flow_formulation == "low-mach"
    assert purdue_incompressible.cells == (64, 64, 64)
    assert purdue_incompressible.flow_formulation == "incompressible"
    assert purdue_incompressible.momentum_closure == "classical-static-smagorinsky"
    assert purdue_incompressible_amd.cells == (128, 128, 128)
    assert purdue_incompressible_amd.momentum_closure == "amd"
    assert purdue_subgrid.subgrid_jet_enabled
    assert purdue_subgrid.subgrid_support_radius_cells == 2.5
    assert purdue_subgrid.subgrid_turbulence_intensity == 0.20
    assert purdue_subgrid.subgrid_integral_scale_cells == 4.0

    source = [row for row in dlr_reference if row["role"] == "source"]
    validation = [row for row in dlr_reference if row["role"] == "validation"]
    assert dlr.cells == (256, 256, 256)
    assert dlr.lengths == (0.09, 0.16, 0.16)
    assert dlr.nozzle == (0.005, 0.08, 0.08)
    assert dlr.steps == 16000
    assert dlr.profile_radius_m[-1] == dlr.radius
    assert dlr_protocol.source_physical_y == 0.005
    assert len(source) == 5
    assert len(validation) == 83
    assert {row["y_D"] for row in validation} == {
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
    }


def test_purdue_sampling_folds_both_sides_into_one_annulus() -> None:
    parcels = SimpleNamespace(
        active=np.array([True, True]),
        x=np.array([0.0581, 0.0581]),
        y=np.array([0.07968, 0.12032]),
        z=np.array([0.1, 0.1]),
        u=np.array([10.0, 10.0]),
        diameter=np.array([50.0e-6, 60.0e-6]),
        multiplicity=np.array([2.0, 2.0]),
    )
    statistics = sample_purdue(
        parcels,
        [(0.0762, 0.02032)],
        nozzle=(0.02, 0.1, 0.1),
        source_physical_x=0.0381,
        axial_half_width=0.002,
        radial_half_width=0.001,
    )

    assert statistics[0, 4] == 2
    assert statistics[0, 3] == 40.0



def test_purdue_error_summary_aggregates_its_own_comparisons() -> None:
    summary = purdue_error_summary(
        [
            {
                "smd_relative_error": -0.1,
                "velocity_relative_error": 0.2,
            },
            {
                "smd_relative_error": 0.3,
                "velocity_relative_error": -0.4,
            },
        ]
    )

    assert summary["mean_absolute_smd_relative_error"] == 0.2
    assert abs(summary["mean_absolute_velocity_relative_error"] - 0.3) < 1.0e-15
    rows = purdue_comparison_rows(
        {
            "comparisons": [{"physical_x_m": 0.0762}],
            "calibration_first_section": {
                "comparisons": [{"physical_x_m": 0.0381}]
            },
        }
    )
    assert [row["physical_x_m"] for row in rows] == [0.0381, 0.0762]


def test_dlr_sampling_mirrors_only_the_axisymmetric_prediction() -> None:
    parcels = SimpleNamespace(
        active=np.array([True, True]),
        x=np.array([0.020, 0.020]),
        y=np.array([0.085, 0.075]),
        z=np.array([0.080, 0.080]),
        u=np.array([2.0, 2.0]),
        v=np.array([10.0, -10.0]),
        w=np.array([0.0, 0.0]),
        diameter=np.array([10.0e-6, 10.0e-6]),
        multiplicity=np.array([1.0, 1.0]),
    )
    bins = [(20.0, 5.0)]
    statistics = sample_dlr(
        parcels,
        bins,
        source=(0.005, 0.080, 0.080),
        nozzle_diameter=0.001,
        source_physical_y=0.005,
        axial_half_width=0.001,
        radial_half_width=0.001,
    )
    reference = [
        {
            "x_D": sign * 5.0,
            "y_D": 20.0,
            "role": "validation",
            "d10_um": 10.0,
            "u_mean_m_s": 2.0,
            "v_mean_m_s": sign * 10.0,
            "d10_hi_um": np.nan,
            "d10_lo_um": np.nan,
            "u_mean_hi_m_s": np.nan,
            "u_mean_lo_m_s": np.nan,
        }
        for sign in (-1.0, 1.0)
    ]
    report = build_report(reference, bins, statistics[None, ...])

    assert dlr_bins(reference) == bins
    assert statistics[0, 4] == 2
    assert report["populated_signed_points"] == 2
    assert [
        row["simulated_v_mean_m_s"] for row in report["comparisons"]
    ] == [-10.0, 10.0]
