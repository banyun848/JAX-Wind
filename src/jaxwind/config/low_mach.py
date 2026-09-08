"""Run or continue a periodic ABL with low-Mach projection."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import NamedTuple
from .document import ResolvedCase

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class ExtensionCase:
    source_workflow: Path | ResolvedCase
    source_checkpoint: Path | None
    restart_formulation: str
    dt: float
    steps: int
    chunk_steps: int
    frame_count: int
    statistics_window_seconds: float | None
    temperature: float
    pressure: float
    water_vapor_mixing_ratio: float
    air_gas_constant: float
    nitrogen_gas_constant: float
    water_vapor_gas_constant: float
    gravity: tuple[float, float, float]
    pressure_backend: str
    time_integration: str
    output: Path


def _path(value: object, *, base: Path, workspace_relative: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("configured paths must be non-empty strings")
    path = Path(value)
    if path.is_absolute() or workspace_relative:
        return path
    return base / path


def load_case(path: str | Path) -> ExtensionCase:
    source = Path(path)
    from .document import native_document
    document = native_document(path)
    restart = document["restart"]
    time_table = document["time"]
    thermodynamics = document["thermodynamics"]
    numerics = document["numerics"]
    output = document["output"]
    checkpoint_keys = tuple(
        key
        for key in ("incompressible_checkpoint", "low_mach_checkpoint")
        if key in restart
    )
    initial_condition = restart.get("initial_condition")
    if len(checkpoint_keys) + (initial_condition is not None) != 1:
        raise ValueError(
            "restart must define exactly one of incompressible_checkpoint, "
            "low_mach_checkpoint, or initial_condition"
        )
    if initial_condition is not None and initial_condition != "configured":
        raise ValueError("restart initial_condition must be configured")
    checkpoint_key = checkpoint_keys[0] if checkpoint_keys else None
    case = ExtensionCase(
        source_workflow=_path(
            restart["source_workflow"], base=source.parent
        ),
        source_checkpoint=(
            None
            if checkpoint_key is None
            else _path(
                restart[checkpoint_key],
                base=source.parent,
                workspace_relative=True,
            )
        ),
        restart_formulation=(
            "configured"
            if checkpoint_key is None
            else (
                "incompressible"
                if checkpoint_key == "incompressible_checkpoint"
                else "low-mach"
            )
        ),
        dt=float(time_table["dt_seconds"]),
        steps=int(time_table["steps"]),
        chunk_steps=int(time_table["chunk_steps"]),
        frame_count=int(time_table["frame_count"]),
        statistics_window_seconds=(
            None
            if "statistics_window_seconds" not in time_table
            else float(time_table["statistics_window_seconds"])
        ),
        temperature=float(thermodynamics["temperature_k"]),
        pressure=float(thermodynamics["pressure_pa"]),
        water_vapor_mixing_ratio=float(
            thermodynamics["water_vapor_mixing_ratio"]
        ),
        air_gas_constant=float(
            thermodynamics.get("air_gas_constant_j_kg_k", 287.05)
        ),
        nitrogen_gas_constant=float(
            thermodynamics.get("nitrogen_gas_constant_j_kg_k", 296.8)
        ),
        water_vapor_gas_constant=float(
            thermodynamics.get("water_vapor_gas_constant_j_kg_k", 461.5)
        ),
        gravity=tuple(
            float(value)
            for value in thermodynamics.get(
                "gravity_m_s2", (0.0, 0.0, -9.81)
            )
        ),
        pressure_backend=str(numerics["pressure_backend"]),
        time_integration=str(numerics["time_integration"]),
        output=_path(
            output["directory"], base=source.parent, workspace_relative=True
        ),
    )
    positive = (
        case.dt,
        case.temperature,
        case.pressure,
        case.air_gas_constant,
        case.nitrogen_gas_constant,
        case.water_vapor_gas_constant,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in positive):
        raise ValueError("timestep and thermodynamic constants must be positive")
    if case.steps <= 0 or case.chunk_steps <= 0:
        raise ValueError("steps and chunk_steps must be positive")
    if not 0 < case.frame_count <= case.steps:
        raise ValueError("frame_count must lie between one and steps")
    if case.statistics_window_seconds is not None and not (
        0.0 < case.statistics_window_seconds <= case.steps * case.dt
    ):
        raise ValueError(
            "statistics_window_seconds must be positive and no longer "
            "than the configured run"
        )
    if (
        not math.isfinite(case.water_vapor_mixing_ratio)
        or case.water_vapor_mixing_ratio < 0.0
    ):
        raise ValueError("water-vapor mixing ratio must be finite and nonnegative")
    if len(case.gravity) != 3 or not all(
        math.isfinite(value) for value in case.gravity
    ):
        raise ValueError("gravity_m_s2 must contain three finite values")
    if case.pressure_backend != "fft":
        raise ValueError("the periodic low-Mach extension requires FFT pressure")
    if case.time_integration != "fast-rk3":
        raise ValueError("the low-Mach extension currently requires fast-rk3")
    if not case.source_workflow.is_file():
        raise FileNotFoundError(
            f"missing source workflow: {case.source_workflow}"
        )
    if "domain" in document or "profile_resampling" in document["case"]:
        from .document import ResolvedCase, load_case as load_shared, merge, validate
        source_case = load_shared(case.source_workflow)
        overridden = merge(source_case.document, {"mesh": document.get("domain", {}),
                           "case": {"profile_resampling": document["case"].get("profile_resampling", "strict")}})
        validate(overridden)
        case = replace(case, source_workflow=ResolvedCase(source_case.source, overridden))
    return case
