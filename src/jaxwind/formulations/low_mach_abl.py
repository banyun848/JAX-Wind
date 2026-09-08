"""Periodic low-Mach state and pure compiled advancement."""
from __future__ import annotations
from typing import NamedTuple
import numpy as np

class LowMachABLState(NamedTuple):
    velocity: object
    pressure: object
    density: object
    momentum_tendency: object
    temperature: object
    water_vapor: object
    nitrogen: object
    continuity_error: object
    time: object
    step: object


def _add_velocity(left, right):
    from jaxwind import StaggeredVelocity

    return StaggeredVelocity(
        left.x + right.x,
        left.y + right.y,
        left.z + right.z,
    )


def _cell_vector_to_faces(x, y, z, grid):
    from jaxwind import StaggeredVelocity
    from jaxwind.numerics.discretization import _cells_to_faces

    return StaggeredVelocity(
        _cells_to_faces(x, grid, 2, periodic=True, boundary="copy"),
        _cells_to_faces(y, grid, 1, periodic=True, boundary="copy"),
        _cells_to_faces(z, grid, 0, periodic=False, boundary="zero"),
    )
