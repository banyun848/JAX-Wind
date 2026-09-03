"""Cartesian grid metadata and analytical coordinate mappings in SI units.

The finite-volume mesh is rectilinear: the three physical coordinates are
independent analytical maps of uniformly spaced computational coordinates,

``x = Lx * X(xi)``, ``y = Ly * Y(eta)``, ``z = Lz * Z(zeta)``.

Keeping the mapping separable preserves the compact seven-point pressure
stencil and tensor-product GPU storage while allowing cells to be clustered
around an inlet, wall, or any other coordinate plane.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias

import numpy as np

from .locations import Cell, Location, ZFace


class CoordinateMapping(Protocol):
    """Map a normalized coordinate in ``[0, 1]`` back into ``[0, 1]``."""

    def __call__(self, coordinate: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class IdentityMapping:
    """The uniform analytical mapping ``X(xi) = xi``."""

    def __call__(self, coordinate: np.ndarray) -> np.ndarray:
        return np.asarray(coordinate)


@dataclass(frozen=True, slots=True)
class SinhMapping:
    """Cluster cells smoothly around one normalized physical coordinate.

    ``focus`` is the clustering location in ``[0, 1]`` and ``strength`` sets
    the stretching. Zero strength is exactly the identity. At an endpoint the
    map reduces to one-sided hyperbolic-sine stretching; in the interior two
    branches meet with a continuous first derivative.

    The local-to-remote cell-size ratio is approximately ``cosh(strength)``.
    Values around 3--4 are useful starting points. Much stronger stretching
    can hurt both explicit stability and multigrid convergence.
    """

    focus: float
    strength: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.focus) or not 0.0 <= self.focus <= 1.0:
            raise ValueError("mapping focus must lie in [0, 1]")
        if not np.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("mapping strength must be finite and nonnegative")

    def __call__(self, coordinate: np.ndarray) -> np.ndarray:
        coordinate = np.asarray(coordinate, dtype=np.float64)
        if self.strength == 0.0:
            return coordinate
        scale = np.sinh(self.strength)
        if self.focus == 0.0:
            return np.sinh(self.strength * coordinate) / scale
        if self.focus == 1.0:
            return 1.0 - np.sinh(self.strength * (1.0 - coordinate)) / scale
        left_fraction = coordinate / self.focus
        right_fraction = (coordinate - self.focus) / (1.0 - self.focus)
        left = self.focus * (
            1.0
            - np.sinh(self.strength * (1.0 - left_fraction)) / scale
        )
        right = self.focus + (1.0 - self.focus) * (
            np.sinh(self.strength * right_fraction) / scale
        )
        return np.where(coordinate <= self.focus, left, right)


@dataclass(frozen=True, slots=True)
class TanhMapping:
    """Cluster cells around a normalized coordinate using tanh branches.

    ``focus`` defaults to the domain centre. The two branches have the same
    derivative at the focus, so the mapping is continuously differentiable.
    The outer-to-focus cell-size ratio is approximately
    ``cosh(strength)**2``.
    """

    strength: float
    focus: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.focus) or not 0.0 <= self.focus <= 1.0:
            raise ValueError("mapping focus must lie in [0, 1]")
        if not np.isfinite(self.strength) or self.strength < 0.0:
            raise ValueError("mapping strength must be finite and nonnegative")

    def __call__(self, coordinate: np.ndarray) -> np.ndarray:
        coordinate = np.asarray(coordinate, dtype=np.float64)
        if self.strength == 0.0:
            return coordinate
        scale = np.tanh(self.strength)
        if self.focus == 0.0:
            return 1.0 - np.tanh(
                self.strength * (1.0 - coordinate)
            ) / scale
        if self.focus == 1.0:
            return np.tanh(self.strength * coordinate) / scale
        left_fraction = coordinate / self.focus
        right_fraction = (coordinate - self.focus) / (1.0 - self.focus)
        left = self.focus * np.tanh(
            self.strength * left_fraction
        ) / scale
        right = self.focus + (1.0 - self.focus) * (
            1.0
            - np.tanh(self.strength * (1.0 - right_fraction)) / scale
        )
        return np.where(coordinate <= self.focus, left, right)


IDENTITY = IdentityMapping()


def _validate_dimensions(
    nx: int,
    ny: int,
    nz: int,
    lx: float,
    ly: float,
    lz: float,
) -> None:
    if min(nx, ny, nz) <= 0:
        raise ValueError("grid cell counts must be positive")
    if not all(np.isfinite(length) and length > 0.0 for length in (lx, ly, lz)):
        raise ValueError("grid lengths must be finite and positive")


def _sample_mapping(
    count: int,
    length: float,
    mapping: CoordinateMapping | Callable[[np.ndarray], np.ndarray],
    axis: str,
) -> tuple[float, ...]:
    computational = np.linspace(0.0, 1.0, count + 1, dtype=np.float64)
    normalized = np.asarray(mapping(computational), dtype=np.float64)
    if normalized.shape != computational.shape:
        raise ValueError(f"the {axis} mapping must preserve its input shape")
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"the {axis} mapping returned a non-finite coordinate")
    tolerance = 128.0 * np.finfo(np.float64).eps
    if not np.isclose(normalized[0], 0.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"the {axis} mapping must map 0 to 0")
    if not np.isclose(normalized[-1], 1.0, rtol=0.0, atol=tolerance):
        raise ValueError(f"the {axis} mapping must map 1 to 1")
    physical = length * normalized
    physical[0], physical[-1] = 0.0, length
    if np.any(np.diff(physical) <= 0.0):
        raise ValueError(f"the {axis} mapping must be strictly increasing")
    return tuple(float(value) for value in physical)


class CartesianGrid(Protocol):
    """Structural interface shared by uniform and analytically mapped grids."""

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float

    @property
    def x_faces(self) -> np.ndarray: ...

    @property
    def y_faces(self) -> np.ndarray: ...

    @property
    def z_faces(self) -> np.ndarray: ...

    @property
    def is_uniform(self) -> bool: ...


class _CartesianGridMixin:
    """Geometry derived from the three arrays of physical face positions."""

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float

    @property
    def x_widths(self) -> np.ndarray:
        return np.diff(self.x_faces)

    @property
    def y_widths(self) -> np.ndarray:
        return np.diff(self.y_faces)

    @property
    def z_widths(self) -> np.ndarray:
        return np.diff(self.z_faces)

    @property
    def x_centers(self) -> np.ndarray:
        faces = self.x_faces
        return 0.5 * (faces[:-1] + faces[1:])

    @property
    def y_centers(self) -> np.ndarray:
        faces = self.y_faces
        return 0.5 * (faces[:-1] + faces[1:])

    @property
    def z_centers(self) -> np.ndarray:
        faces = self.z_faces
        return 0.5 * (faces[:-1] + faces[1:])

    @property
    def dx(self) -> float:
        """Nominal x spacing retained for uniform-grid source compatibility."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """Nominal y spacing retained for uniform-grid source compatibility."""
        return self.ly / self.ny

    @property
    def dz(self) -> float:
        """Nominal z spacing retained for uniform-grid source compatibility."""
        return self.lz / self.nz

    @property
    def minimum_spacing(self) -> float:
        return float(
            min(
                np.min(self.x_widths),
                np.min(self.y_widths),
                np.min(self.z_widths),
            )
        )

    @property
    def maximum_spacing(self) -> float:
        return float(
            max(
                np.max(self.x_widths),
                np.max(self.y_widths),
                np.max(self.z_widths),
            )
        )

    @property
    def cell_count(self) -> int:
        return self.nx * self.ny * self.nz

    @property
    def cell_volumes(self) -> np.ndarray:
        return (
            self.z_widths[:, None, None]
            * self.y_widths[None, :, None]
            * self.x_widths[None, None, :]
        )

    def global_z_first_shape(self, location: type[Location]) -> tuple[int, int, int]:
        if location is Cell:
            return (self.nz, self.ny, self.nx)
        if location is ZFace:
            return (self.nz + 1, self.ny, self.nx)
        raise ValueError(f"unsupported location: {location!r}")


@dataclass(frozen=True, slots=True)
class UniformGrid(_CartesianGridMixin):
    """A uniform Cartesian cell grid with SI lengths."""

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float

    def __post_init__(self) -> None:
        _validate_dimensions(self.nx, self.ny, self.nz, self.lx, self.ly, self.lz)

    @property
    def x_faces(self) -> np.ndarray:
        return np.linspace(0.0, self.lx, self.nx + 1)

    @property
    def y_faces(self) -> np.ndarray:
        return np.linspace(0.0, self.ly, self.ny + 1)

    @property
    def z_faces(self) -> np.ndarray:
        return np.linspace(0.0, self.lz, self.nz + 1)

    @property
    def is_uniform(self) -> bool:
        return True


@dataclass(frozen=True, slots=True)
class AnalyticalGrid(_CartesianGridMixin):
    """A rectilinear grid sampled from three analytical normalized mappings.

    Mapping callables receive all face coordinates at once as a NumPy array.
    They are evaluated during construction, before JAX tracing, so arbitrary
    user functions incur no run-time cost. Sampled face coordinates become
    immutable tuples and remain safe static JAX metadata.
    """

    nx: int
    ny: int
    nz: int
    lx: float
    ly: float
    lz: float
    x_mapping: CoordinateMapping | Callable[[np.ndarray], np.ndarray] = IDENTITY
    y_mapping: CoordinateMapping | Callable[[np.ndarray], np.ndarray] = IDENTITY
    z_mapping: CoordinateMapping | Callable[[np.ndarray], np.ndarray] = IDENTITY
    _x_faces: tuple[float, ...] = field(init=False, repr=False)
    _y_faces: tuple[float, ...] = field(init=False, repr=False)
    _z_faces: tuple[float, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_dimensions(self.nx, self.ny, self.nz, self.lx, self.ly, self.lz)
        object.__setattr__(
            self,
            "_x_faces",
            _sample_mapping(self.nx, self.lx, self.x_mapping, "x"),
        )
        object.__setattr__(
            self,
            "_y_faces",
            _sample_mapping(self.ny, self.ly, self.y_mapping, "y"),
        )
        object.__setattr__(
            self,
            "_z_faces",
            _sample_mapping(self.nz, self.lz, self.z_mapping, "z"),
        )

    @property
    def x_faces(self) -> np.ndarray:
        return np.asarray(self._x_faces)

    @property
    def y_faces(self) -> np.ndarray:
        return np.asarray(self._y_faces)

    @property
    def z_faces(self) -> np.ndarray:
        return np.asarray(self._z_faces)

    @property
    def is_uniform(self) -> bool:
        return all(
            np.allclose(widths, widths[0], rtol=1.0e-13, atol=0.0)
            for widths in (self.x_widths, self.y_widths, self.z_widths)
        )

    def coarsen(self, factors: tuple[int, int, int]) -> AnalyticalGrid:
        """Resample the same analytical maps after factor-two agglomeration."""
        fx, fy, fz = factors
        return AnalyticalGrid(
            self.nx // fx,
            self.ny // fy,
            self.nz // fz,
            self.lx,
            self.ly,
            self.lz,
            self.x_mapping,
            self.y_mapping,
            self.z_mapping,
        )


Grid: TypeAlias = UniformGrid | AnalyticalGrid


__all__ = [
    "AnalyticalGrid",
    "CartesianGrid",
    "CoordinateMapping",
    "Grid",
    "IdentityMapping",
    "SinhMapping",
    "TanhMapping",
    "UniformGrid",
]
