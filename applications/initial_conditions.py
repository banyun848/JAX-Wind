"""Interpret one generic tabulated Boussinesq initial state."""

from __future__ import annotations

import numpy as np

from .fv_abl.case import BoussinesqCase, TabulatedBoussinesqState


REQUIRED_COLUMNS = (
    "z_m",
    "u_m_s",
    "v_m_s",
    "w_upper_m_s",
    "scalar",
    "u_rms_m_s",
    "v_rms_m_s",
    "w_upper_rms_m_s",
    "scalar_rms",
)


def load_initial_profile(case: BoussinesqCase) -> np.ndarray:
    """Read one mean-plus-RMS profile shared by every ABL case."""

    table = np.genfromtxt(
        case.initial_condition.path,
        delimiter=",",
        names=True,
    )
    if table.dtype.names != REQUIRED_COLUMNS:
        raise ValueError(
            "initial profile columns must be " + ", ".join(REQUIRED_COLUMNS)
        )
    if table.shape != (case.physical_grid.nz,):
        raise ValueError("initial profile must contain one row per vertical cell")
    z = np.asarray(table["z_m"], dtype=np.float64)
    expected_z = np.asarray(
        case.physical_grid.z_centers, dtype=np.float64
    )
    if not np.allclose(z, expected_z, rtol=0.0, atol=1.0e-12):
        raise ValueError("initial profile heights must match the case grid")
    if not all(np.all(np.isfinite(table[name])) for name in REQUIRED_COLUMNS):
        raise ValueError("initial profile values must be finite")
    for name in (
        "u_rms_m_s",
        "v_rms_m_s",
        "w_upper_rms_m_s",
        "scalar_rms",
    ):
        if np.any(table[name] < 0.0):
            raise ValueError(f"initial profile {name} must be nonnegative")
    return table


__all__ = ["TabulatedBoussinesqState", "load_initial_profile"]
