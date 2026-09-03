"""Reduced LN2 discharge and equilibrium-flash model for source terms.

The reference point is Table I of NASA-TM-X-71760 (Simoneau, 1975). The
orifice is not represented geometrically: its diameter enters only through
the area used to convert mass flux into an integral source mass flow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


NITROGEN_CRITICAL_TEMPERATURE = 126.192
NITROGEN_CRITICAL_PRESSURE = 3.3958e6
NITROGEN_ACENTRIC_FACTOR = 0.0372
NITROGEN_GAS_CONSTANT = 296.803
NITROGEN_RACKETT_FACTOR = 0.2895
NITROGEN_NORMAL_BOILING_TEMPERATURE = 77.34
NITROGEN_LATENT_HEAT_AT_NORMAL_BOILING = 199_180.0
NITROGEN_LIQUID_HEAT_CAPACITY = 2040.0


@dataclass(frozen=True, slots=True)
class NASAOrificeDatum:
    """One operating point transcribed from NASA-TM-X-71760 Table I."""

    report: str = "NASA-TM-X-71760"
    table: str = "Table I, 95.1 K isotherm"
    diameter_m: float = 3.58e-3
    length_to_diameter: float = 0.19
    upstream_temperature_k: float = 95.2
    upstream_pressure_pa: float = 1.98e6
    back_pressure_pa: float = 2.70e5
    equilibrium_back_temperature_k: float = 87.1
    measured_mass_flux_kg_m2_s: float = 30_000.0
    tabulated_flow_coefficient: float = 0.605
    mass_flow_relative_uncertainty: float = 0.02


@dataclass(frozen=True, slots=True)
class LN2SourceState:
    """Integral post-flash state supplied to an unresolved FV source."""

    mass_flux_kg_m2_s: float
    mass_flow_rate_kg_s: float
    speed_m_s: float
    temperature_k: float
    vapor_quality: float
    vapor_mass_flow_rate_kg_s: float
    liquid_mass_flow_rate_kg_s: float
    discharge_liquid_density_kg_m3: float
    post_flash_liquid_density_kg_m3: float
    post_flash_latent_heat_j_kg: float
    discharge_coefficient: float


def saturation_pressure_nitrogen(temperature_k: float) -> float:
    """Nitrogen saturation pressure [Pa], Lee--Kesler/Pitzer correlation."""

    reduced = temperature_k / NITROGEN_CRITICAL_TEMPERATURE
    if not 0.0 < reduced < 1.0:
        raise ValueError("nitrogen saturation temperature must lie below critical")
    f0 = (
        5.92714
        - 6.09648 / reduced
        - 1.28862 * math.log(reduced)
        + 0.169347 * reduced**6
    )
    f1 = (
        15.2518
        - 15.6875 / reduced
        - 13.4721 * math.log(reduced)
        + 0.43577 * reduced**6
    )
    return NITROGEN_CRITICAL_PRESSURE * math.exp(
        f0 + NITROGEN_ACENTRIC_FACTOR * f1
    )


def saturation_temperature_nitrogen(pressure_pa: float) -> float:
    """Invert the saturation-pressure correlation by bisection."""

    if not 0.0 < pressure_pa < NITROGEN_CRITICAL_PRESSURE:
        raise ValueError("nitrogen saturation pressure must lie below critical")
    lower = 50.0
    upper = NITROGEN_CRITICAL_TEMPERATURE * (1.0 - 1.0e-10)
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if saturation_pressure_nitrogen(midpoint) < pressure_pa:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def saturated_liquid_density_nitrogen(temperature_k: float) -> float:
    """Saturated-liquid density [kg/m3] from the Rackett correlation."""

    reduced = temperature_k / NITROGEN_CRITICAL_TEMPERATURE
    if not 0.0 < reduced < 1.0:
        raise ValueError("liquid nitrogen temperature must lie below critical")
    exponent = 1.0 + (1.0 - reduced) ** (2.0 / 7.0)
    specific_volume = (
        NITROGEN_GAS_CONSTANT
        * NITROGEN_CRITICAL_TEMPERATURE
        / NITROGEN_CRITICAL_PRESSURE
        * NITROGEN_RACKETT_FACTOR**exponent
    )
    return 1.0 / specific_volume


def latent_heat_nitrogen(temperature_k: float) -> float:
    """Latent heat [J/kg] from the Watson correlation."""

    reduced_gap = 1.0 - temperature_k / NITROGEN_CRITICAL_TEMPERATURE
    reference_gap = (
        1.0
        - NITROGEN_NORMAL_BOILING_TEMPERATURE
        / NITROGEN_CRITICAL_TEMPERATURE
    )
    if reduced_gap <= 0.0:
        raise ValueError("latent heat is defined below the critical temperature")
    return NITROGEN_LATENT_HEAT_AT_NORMAL_BOILING * (
        reduced_gap / reference_gap
    ) ** 0.38


def equilibrium_flash_quality(
    upstream_temperature_k: float,
    upstream_pressure_pa: float,
    back_pressure_pa: float,
) -> tuple[float, float]:
    """Return post-flash equilibrium quality and saturation temperature.

    The compressed-liquid enthalpy is approximated with constant liquid heat
    capacity plus an incompressible pressure correction. NASA reported
    metastable liquid at the orifice, so this quality belongs to the downstream
    relaxed source state, not to a geometrical orifice plane.
    """

    exit_temperature = saturation_temperature_nitrogen(back_pressure_pa)
    liquid_density = saturated_liquid_density_nitrogen(upstream_temperature_k)
    saturation_pressure = saturation_pressure_nitrogen(upstream_temperature_k)
    available_enthalpy = (
        NITROGEN_LIQUID_HEAT_CAPACITY
        * (upstream_temperature_k - exit_temperature)
        + (upstream_pressure_pa - saturation_pressure) / liquid_density
    )
    quality = available_enthalpy / latent_heat_nitrogen(exit_temperature)
    return min(max(quality, 0.0), 1.0), exit_temperature


def source_state_from_orifice(
    datum: NASAOrificeDatum = NASAOrificeDatum(),
    *,
    discharge_coefficient: float = 0.61,
) -> LN2SourceState:
    """Convert the NASA pressure state into an integral FV source state."""

    if not 0.0 < discharge_coefficient <= 1.0:
        raise ValueError("discharge coefficient must lie in (0, 1]")
    density = saturated_liquid_density_nitrogen(datum.upstream_temperature_k)
    pressure_drop = datum.upstream_pressure_pa - datum.back_pressure_pa
    if pressure_drop <= 0.0:
        raise ValueError("upstream pressure must exceed back pressure")
    mass_flux = discharge_coefficient * math.sqrt(2.0 * density * pressure_drop)
    area = math.pi * datum.diameter_m**2 / 4.0
    mass_flow = mass_flux * area
    quality, temperature = equilibrium_flash_quality(
        datum.upstream_temperature_k,
        datum.upstream_pressure_pa,
        datum.back_pressure_pa,
    )
    return LN2SourceState(
        mass_flux_kg_m2_s=mass_flux,
        mass_flow_rate_kg_s=mass_flow,
        speed_m_s=mass_flux / density,
        temperature_k=temperature,
        vapor_quality=quality,
        vapor_mass_flow_rate_kg_s=quality * mass_flow,
        liquid_mass_flow_rate_kg_s=(1.0 - quality) * mass_flow,
        discharge_liquid_density_kg_m3=density,
        post_flash_liquid_density_kg_m3=(
            saturated_liquid_density_nitrogen(temperature)
        ),
        post_flash_latent_heat_j_kg=latent_heat_nitrogen(temperature),
        discharge_coefficient=discharge_coefficient,
    )


def nasa_validation() -> dict[str, object]:
    """Return a serializable comparison with the selected NASA datum."""

    datum = NASAOrificeDatum()
    source = source_state_from_orifice(datum)
    flux_error = (
        source.mass_flux_kg_m2_s / datum.measured_mass_flux_kg_m2_s - 1.0
    )
    temperature_error = (
        source.temperature_k - datum.equilibrium_back_temperature_k
    )
    return {
        "reference": asdict(datum),
        "source": asdict(source),
        "comparison": {
            "mass_flux_relative_error": flux_error,
            "mass_flux_error_percent": 100.0 * flux_error,
            "within_nasa_mass_flow_uncertainty": (
                abs(flux_error) <= datum.mass_flow_relative_uncertainty
            ),
            "equilibrium_temperature_error_k": temperature_error,
            "vapor_quality_is_thermodynamically_derived": True,
        },
    }


__all__ = [
    "LN2SourceState",
    "NASAOrificeDatum",
    "equilibrium_flash_quality",
    "latent_heat_nitrogen",
    "nasa_validation",
    "saturated_liquid_density_nitrogen",
    "saturation_pressure_nitrogen",
    "saturation_temperature_nitrogen",
    "source_state_from_orifice",
]
