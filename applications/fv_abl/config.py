"""Strict finite-volume options layered onto the shared ABL case schema."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import tomllib

from .case import BoussinesqCase
from .physical import load_abl


@dataclass(frozen=True, slots=True)
class FiniteVolumeOptions:
    """Numerical and effect-shell choices for the generic FV ABL core."""

    pressure_backend: str
    time_integration: str
    turbulent_prandtl: float
    chunk_steps: int
    spectrum_diagnostic: str
    output_directory: Path
    scalar_advection_scheme: str = "central"
    cfl_ceiling: float | None = None
    gmg_tolerance: float | None = None
    gmg_presweeps: int = 2
    gmg_postsweeps: int = 2
    gmg_anisotropy_aware: bool = True


@dataclass(frozen=True, slots=True)
class FiniteVolumeCase:
    """One physical ABL case paired with finite-volume solver options."""

    physical: BoussinesqCase
    options: FiniteVolumeOptions
    source: Path


def _string(table: dict[str, Any], key: str) -> str:
    value = table[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"finite_volume.{key} must be a non-empty string")
    return value


def _positive_integer(table: dict[str, Any], key: str) -> int:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"finite_volume.{key} must be a positive integer")
    return value


def _positive_number(table: dict[str, Any], key: str) -> float:
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"finite_volume.{key} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"finite_volume.{key} must be a positive number")
    return result


def _choice(value: str, choices: set[str], key: str) -> str:
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"finite_volume.{key} must be one of: {expected}")
    return value


def load_fv_abl(path: str | Path) -> FiniteVolumeCase:
    """Load shared ABL physics plus the strict ``[finite_volume]`` table."""

    source = Path(path)
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    table = document.get("finite_volume")
    if not isinstance(table, dict):
        raise ValueError("missing [finite_volume] table")
    expected = {
        "pressure_backend",
        "time_integration",
        "turbulent_prandtl",
        "chunk_steps",
        "spectrum_diagnostic",
        "output_directory",
    }
    missing = expected - table.keys()
    optional = {
        "cfl_ceiling",
        "gmg_tolerance",
        "gmg_presweeps",
        "gmg_postsweeps",
        "gmg_anisotropy_aware",
        "scalar_advection_scheme",
    }
    unknown = table.keys() - expected - optional
    if missing:
        raise ValueError(
            "[finite_volume] is missing: " + ", ".join(sorted(missing))
        )
    if unknown:
        raise ValueError(
            "[finite_volume] has unknown keys: " + ", ".join(sorted(unknown))
        )
    pressure_backend = _choice(
        _string(table, "pressure_backend"),
        {"fft", "gmg"},
        "pressure_backend",
    )
    output_directory = _string(table, "output_directory").replace(
        "{pressure_backend}", pressure_backend
    )
    options = FiniteVolumeOptions(
        pressure_backend=pressure_backend,
        time_integration=_choice(
            _string(table, "time_integration"),
            {"ab2", "fast-rk3", "rk3"},
            "time_integration",
        ),
        turbulent_prandtl=_positive_number(table, "turbulent_prandtl"),
        chunk_steps=_positive_integer(table, "chunk_steps"),
        spectrum_diagnostic=_choice(
            _string(table, "spectrum_diagnostic"),
            {"none", "radial", "streamwise"},
            "spectrum_diagnostic",
        ),
        output_directory=Path(output_directory),
        scalar_advection_scheme=_choice(
            table.get("scalar_advection_scheme", "central"),
            {"central", "upwind"},
            "scalar_advection_scheme",
        ),
        cfl_ceiling=(
            _positive_number(table, "cfl_ceiling")
            if "cfl_ceiling" in table
            else None
        ),
        gmg_tolerance=(
            _positive_number(table, "gmg_tolerance")
            if "gmg_tolerance" in table
            else None
        ),
        gmg_presweeps=(
            _positive_integer(table, "gmg_presweeps")
            if "gmg_presweeps" in table
            else 2
        ),
        gmg_postsweeps=(
            _positive_integer(table, "gmg_postsweeps")
            if "gmg_postsweeps" in table
            else 2
        ),
        gmg_anisotropy_aware=table.get("gmg_anisotropy_aware", True),
    )
    if not isinstance(options.gmg_anisotropy_aware, bool):
        raise ValueError("finite_volume.gmg_anisotropy_aware must be boolean")
    return FiniteVolumeCase(load_abl(source), options, source)


__all__ = ["FiniteVolumeCase", "FiniteVolumeOptions", "load_fv_abl"]
