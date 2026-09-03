"""Validate the unresolved LN2 source against NASA-TM-X-71760."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

from jaxwind.physics.ln2_nozzle import nasa_validation


def validate_case(path: str | Path) -> dict[str, object]:
    """Validate NASA discharge and ensure the FV source uses that state."""

    with Path(path).open("rb") as stream:
        document = tomllib.load(stream)
    result = nasa_validation()
    source = result["source"]
    jet = document["jet"]
    configured = {
        "mass_flow_rate_kg_s": float(jet["mass_flow_rate_kg_s"]),
        "speed_m_s": float(jet["speed_m_s"]),
        "temperature_k": float(jet["temperature_k"]),
        "post_flash_liquid_density_kg_m3": float(
            jet["liquid_density_kg_m3"]
        ),
        "post_flash_latent_heat_j_kg": float(
            jet["liquid_latent_heat_j_kg"]
        ),
        "vapor_quality": float(jet["vapor_quality"]),
    }
    tolerances = {
        "mass_flow_rate_kg_s": 1.0e-6,
        "speed_m_s": 1.0e-5,
        "temperature_k": 1.0e-5,
        "post_flash_liquid_density_kg_m3": 1.0e-5,
        "post_flash_latent_heat_j_kg": 1.0e-3,
        "vapor_quality": 1.0e-6,
    }
    matches = {
        key: abs(configured[key] - float(source[key])) <= tolerances[key]
        for key in configured
    }
    result["fv_source"] = {
        "model": "normalized compact mass/momentum/enthalpy source; no geometry",
        "config": str(path),
        "configured": configured,
        "matches_derived_state": matches,
        "all_fields_match": all(matches.values()),
    }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("cases/NASALN2Orifice/fv_256.toml"),
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = validate_case(arguments.config)
    encoded = json.dumps(result, indent=2) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["comparison"]["within_nasa_mass_flow_uncertainty"]:
        raise SystemExit("predicted mass flux is outside the NASA uncertainty band")
    if not result["fv_source"]["all_fields_match"]:
        raise SystemExit("FV source values do not match the validated source state")


if __name__ == "__main__":
    main()
