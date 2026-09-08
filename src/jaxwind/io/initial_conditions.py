"""Interpret one generic tabulated Boussinesq initial state."""

from __future__ import annotations

import numpy as np

from jaxwind.config.abl_types import BoussinesqCase, TabulatedBoussinesqState


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
    table = np.atleast_1d(table)
    if not all(np.all(np.isfinite(table[name])) for name in REQUIRED_COLUMNS):
        raise ValueError("initial profile values must be finite")
    z = np.asarray(table["z_m"], dtype=np.float64)
    expected_z = np.asarray(
        case.physical_grid.z_centers, dtype=np.float64
    )
    matches = z.shape == expected_z.shape and np.allclose(z, expected_z, rtol=0., atol=1.e-12)
    if not matches:
        if case.initial_condition.profile_resampling != "linear":
            raise ValueError("initial profile heights must match the case grid; explicitly select case.profile_resampling = 'linear' to resample")
        table = resample_profile(table, case.physical_grid.z_faces)
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


def resample_profile(table, target_faces):
    """Explicit linear tabulated-profile transformation, clamped at endpoints.

    Cell means/RMS use cell centers; vertical velocity uses inferred upper
    faces. This transforms initialization data, never a restart history.
    """
    z = np.asarray(table["z_m"], dtype=float)
    if len(z) < 2 or not (np.diff(z) > 0).all():
        raise ValueError("resampling requires at least two increasing profile heights")
    source_faces = np.zeros(len(z) + 1)
    for index, center in enumerate(z):
        source_faces[index + 1] = 2 * center - source_faces[index]
    target_faces = np.asarray(target_faces)
    if not (np.diff(source_faces) > 0).all() or not np.isclose(source_faces[-1], target_faces[-1]):
        raise ValueError("profile resampling requires matching vertical extents and valid cell centers")
    target_z = .5 * (target_faces[:-1] + target_faces[1:])
    result = np.empty(len(target_z), dtype=table.dtype)
    result["z_m"] = target_z
    for name in REQUIRED_COLUMNS[1:]:
        if name.startswith("w_upper"):
            result[name] = np.interp(target_faces[1:], source_faces, np.concatenate(([0.], table[name])))
            result[name][-1] = 0.
        else:
            result[name] = np.interp(target_z, z, table[name])
    return result


__all__ = ["TabulatedBoussinesqState", "load_initial_profile"]
