"""Cryogenic state and compiled inlet/transport coupling."""
from __future__ import annotations
from typing import NamedTuple
import math
import numpy as np

class DifferentiableInletControl(NamedTuple):
    """Continuous physical controls carried as a JAX pytree."""

    gas_radius: object
    gas_temperature: object
    initial_diameter: object
    speed_scale: object
    edge_speed_ratio: object
    edge_diameter_ratio: object


class CryogenicState(NamedTuple):
    velocity: object
    pressure: object
    density: object
    momentum_tendency: object
    temperature: object
    temperature_tendency: object
    water_vapor: object
    water_vapor_tendency: object
    nitrogen: object
    nitrogen_density: object
    nitrogen_density_tendency: object
    liquid_water: object
    liquid_water_tendency: object
    ice_water: object
    ice_water_tendency: object
    parcels: object
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
        _cells_to_faces(x, grid, 2, periodic=False, boundary="copy"),
        _cells_to_faces(y, grid, 1, periodic=False, boundary="zero"),
        _cells_to_faces(z, grid, 0, periodic=False, boundary="zero"),
    )


def _enforce_scalar(field, ambient):
    """Zero-gradient outlet with a quiescent ambient inlet."""

    field = field.at[..., 0].set(ambient)
    outlet = (4.0 * field[..., -2] - field[..., -3]) / 3.0
    return field.at[..., -1].set(outlet)


def _conservative_nonnegative(field, cell_volumes):
    """Remove undershoots without changing the volume-weighted inventory."""

    import jax.numpy as jnp

    volumes = jnp.asarray(cell_volumes, field.dtype)
    target = jnp.maximum(jnp.sum(field * volumes), 0.0)
    positive = jnp.maximum(field, 0.0)
    positive_inventory = jnp.sum(positive * volumes)
    tiny = jnp.finfo(field.dtype).tiny
    scale = jnp.where(
        positive_inventory > tiny,
        target / jnp.maximum(positive_inventory, tiny),
        0.0,
    )
    return positive * scale
