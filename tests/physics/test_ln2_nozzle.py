from __future__ import annotations

from applications.nasa_ln2_source import validate_case
from jaxwind.physics.ln2_nozzle import (
    NASAOrificeDatum,
    nasa_validation,
    saturation_pressure_nitrogen,
    source_state_from_orifice,
)


def test_nasa_orifice_closes_mass_and_matches_the_source_case() -> None:
    datum = NASAOrificeDatum()
    source = source_state_from_orifice(datum)
    validation = nasa_validation()
    case = validate_case("cases/NASALN2Orifice/fv_256.toml")

    assert abs(saturation_pressure_nitrogen(77.34) - 101_325.0) < 1_000.0
    assert validation["comparison"]["within_nasa_mass_flow_uncertainty"]
    assert abs(validation["comparison"]["mass_flux_relative_error"]) < 0.01
    assert abs(
        source.vapor_mass_flow_rate_kg_s
        + source.liquid_mass_flow_rate_kg_s
        - source.mass_flow_rate_kg_s
    ) < 1.0e-14
    assert abs(source.temperature_k - datum.equilibrium_back_temperature_k) < 0.5
    assert 0.10 < source.vapor_quality < 0.11
    assert case["fv_source"]["all_fields_match"]
    assert "no geometry" in case["fv_source"]["model"]
