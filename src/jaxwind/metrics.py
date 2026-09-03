"""Broadcast-ready geometry for separable rectilinear finite-volume grids."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxwind.domain.grid import Grid


def widths(grid: Grid, axis: int, dtype=None) -> jnp.ndarray:
    values = (grid.z_widths, grid.y_widths, grid.x_widths)[axis]
    return jnp.asarray(values, dtype=dtype)


def shaped_widths(grid: Grid, axis: int, dtype=None) -> jnp.ndarray:
    values = widths(grid, axis, dtype)
    shape = [1, 1, 1]
    shape[axis] = values.size
    return values.reshape(shape)


def center_distances(
    grid: Grid,
    axis: int,
    *,
    periodic: bool,
    dtype=None,
) -> jnp.ndarray:
    cell_widths = widths(grid, axis, dtype)
    if periodic:
        return 0.5 * (cell_widths + jnp.roll(cell_widths, 1))
    if cell_widths.size == 1:
        interior = cell_widths[:0]
    else:
        interior = 0.5 * (cell_widths[:-1] + cell_widths[1:])
    return jnp.concatenate(
        (0.5 * cell_widths[:1], interior, 0.5 * cell_widths[-1:])
    )


def shaped_center_distances(
    grid: Grid,
    axis: int,
    *,
    periodic: bool,
    dtype=None,
) -> jnp.ndarray:
    values = center_distances(grid, axis, periodic=periodic, dtype=dtype)
    shape = [1, 1, 1]
    shape[axis] = values.size
    return values.reshape(shape)


def cell_volumes(grid: Grid, dtype=None) -> jnp.ndarray:
    return (
        shaped_widths(grid, 0, dtype)
        * shaped_widths(grid, 1, dtype)
        * shaped_widths(grid, 2, dtype)
    )


def cell_face_weights(
    grid: Grid, axis: int, dtype=None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Linear weights for the lower/upper cells at every interior face."""
    cell_widths = widths(grid, axis, dtype)
    lower = cell_widths[:-1]
    upper = cell_widths[1:]
    total = lower + upper
    return upper / total, lower / total


def periodic_cell_face_weights(
    grid: Grid, axis: int, dtype=None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Weights for cells immediately below/above each periodic face."""
    upper_width = widths(grid, axis, dtype)
    lower_width = jnp.roll(upper_width, 1)
    total = lower_width + upper_width
    return upper_width / total, lower_width / total


def minimum_widths(grid: Grid) -> tuple[float, float, float]:
    return tuple(
        float(np.min(values))
        for values in (grid.x_widths, grid.y_widths, grid.z_widths)
    )


__all__ = [
    "cell_face_weights",
    "cell_volumes",
    "center_distances",
    "minimum_widths",
    "periodic_cell_face_weights",
    "shaped_center_distances",
    "shaped_widths",
    "widths",
]
