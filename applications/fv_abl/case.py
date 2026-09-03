"""Data products used to declare a Boussinesq case."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from jaxwind.domain.grid import Grid


@dataclass(frozen=True, slots=True)
class TabulatedBoussinesqState:
    """Tabulated means and perturbation amplitudes for every evolved field."""

    path: Path
    seed: int

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("initial-condition seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class SurfaceScalarEvolution:
    """Physical scalar value prescribed at a rough surface over time."""

    initial_value: float
    rate_per_second: float
    roughness_length_m: float
    positive_zeta_momentum_slope: float = 4.8
    positive_zeta_scalar_slope: float = 7.8
    negative_zeta_momentum_coefficient: float = 16.0
    negative_zeta_scalar_coefficient: float = 16.0
    iterations: int = 12
    relaxation: float = 0.5
    maximum_abs_zeta: float = 10.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.initial_value, self.rate_per_second)
        ):
            raise ValueError("surface-scalar values must be finite")
        positive = (
            self.roughness_length_m,
            self.positive_zeta_momentum_slope,
            self.positive_zeta_scalar_slope,
            self.negative_zeta_momentum_coefficient,
            self.negative_zeta_scalar_coefficient,
            self.maximum_abs_zeta,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError(
                "surface-transfer constants must be finite and positive"
            )
        if self.iterations <= 0:
            raise ValueError("surface-transfer iterations must be positive")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("surface-transfer relaxation must lie in (0, 1]")

@dataclass(frozen=True, slots=True)
class DiagnosticReference:
    """Case-data normalization for generic profiles, spectra, and bulk metrics."""

    length_m: float
    velocity_m_s: float
    scalar: float
    inversion_search_max_height_m: float
    spectrum_heights_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if min(self.length_m, self.velocity_m_s, self.scalar) <= 0.0:
            raise ValueError("diagnostic scales must be positive")
        if self.inversion_search_max_height_m <= 0.0:
            raise ValueError("inversion search height must be positive")
        if not self.spectrum_heights_m or any(
            value <= 0.0 for value in self.spectrum_heights_m
        ):
            raise ValueError("spectrum heights must be positive")


@dataclass(frozen=True, slots=True)
class BoussinesqCase:
    """Physical inputs and runtime controls for one finite-volume ABL case."""

    name: str
    citation: str
    physical_grid: Grid
    scalar_reference_value: float
    initial_condition: TabulatedBoussinesqState
    diagnostic_reference: DiagnosticReference
    reference_results: Path
    pressure_acceleration_m_s2: tuple[float, float]
    geostrophic_velocity_m_s: tuple[float, float]
    coriolis_s: tuple[float, float]
    roughness_length_m: float
    von_karman: float
    scalar_surface_flux: float
    buoyancy_acceleration_per_scalar: float
    surface_scalar: SurfaceScalarEvolution | None
    dt_seconds: float
    dtype: str
    sample_start_step: int
    sample_every_steps: int
    steps: int
    advection_frame_velocity_m_s: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if any(
            len(values) != 2
            for values in (
                self.pressure_acceleration_m_s2,
                self.geostrophic_velocity_m_s,
                self.coriolis_s,
                self.advection_frame_velocity_m_s,
            )
        ):
            raise ValueError("horizontal vector inputs must contain two values")
        if not self.name or not self.citation:
            raise ValueError("case name and citation must be non-empty")
        if not math.isfinite(self.scalar_reference_value):
            raise ValueError("scalar reference value must be finite")
        if self.steps <= 0 or self.dt_seconds <= 0.0:
            raise ValueError("case duration controls must be positive")
        if self.sample_start_step < 0 or self.sample_start_step >= self.steps:
            raise ValueError("sample start must precede the final step")
        if self.sample_every_steps <= 0:
            raise ValueError("sample interval must be positive")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("pressure dtype must be float32 or float64")
        positive = (self.roughness_length_m, self.von_karman)
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("wall constants must be finite and positive")
        physical_values = (
            *self.pressure_acceleration_m_s2,
            *self.geostrophic_velocity_m_s,
            *self.coriolis_s,
            self.scalar_surface_flux,
            self.buoyancy_acceleration_per_scalar,
            self.dt_seconds,
            *self.advection_frame_velocity_m_s,
        )
        if not all(math.isfinite(value) for value in physical_values):
            raise ValueError("physical case inputs must be finite")
        if self.coriolis_s[0] == 0.0 and self.coriolis_s[1] != 0.0:
            raise ValueError("horizontal Coriolis requires vertical Coriolis")
        if (
            any(self.advection_frame_velocity_m_s)
            and self.surface_scalar is None
        ):
            raise ValueError("a nonzero advection frame requires surface transfer")

    @property
    def duration_seconds(self) -> float:
        return self.steps * self.dt_seconds


__all__ = [
    "BoussinesqCase",
    "DiagnosticReference",
    "SurfaceScalarEvolution",
    "TabulatedBoussinesqState",
]
