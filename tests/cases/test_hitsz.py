from __future__ import annotations

from jaxwind.config.jet import load_case


def test_tracked_hitsz_case_files_encode_their_benchmark_contracts() -> None:
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
