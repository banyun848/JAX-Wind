"""Array-independent coherent scales for nondimensional JAX execution."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .grid import UniformGrid


@dataclass(frozen=True, slots=True)
class ScaleSystem:
    """Two-base incompressible mechanical scale system.

    Public case values remain SI.  Length and velocity determine time,
    acceleration, inverse-time, pressure-per-density, and viscosity scales.
    """

    length: float
    velocity: float
    version: str = "jaxwind.mechanical-scales.v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.length) or self.length <= 0.0:
            raise ValueError("length scale must be finite and positive")
        if not math.isfinite(self.velocity) or self.velocity <= 0.0:
            raise ValueError("velocity scale must be finite and positive")
        if not self.version:
            raise ValueError("scale-system version must be non-empty")

    @property
    def time(self) -> float:
        return self.length / self.velocity

    @property
    def acceleration(self) -> float:
        return self.velocity * self.velocity / self.length

    @property
    def inverse_time(self) -> float:
        return self.velocity / self.length

    @property
    def kinematic_pressure(self) -> float:
        return self.velocity * self.velocity

    @property
    def kinematic_viscosity(self) -> float:
        return self.length * self.velocity

    @property
    def fingerprint(self) -> str:
        return (
            f"{self.version}|length={float(self.length).hex()}"
            f"|velocity={float(self.velocity).hex()}"
        )

    def to_execution_length(self, value):
        return value / self.length

    def from_execution_length(self, value):
        return value * self.length

    def to_execution_velocity(self, value):
        return value / self.velocity

    def from_execution_velocity(self, value):
        return value * self.velocity

    def to_execution_time(self, value):
        return value / self.time

    def from_execution_time(self, value):
        return value * self.time

    def to_execution_acceleration(self, value):
        return value / self.acceleration

    def from_execution_acceleration(self, value):
        return value * self.acceleration

    def to_execution_kinematic_viscosity(self, value):
        return value / self.kinematic_viscosity

    def from_execution_kinematic_viscosity(self, value):
        return value * self.kinematic_viscosity

    def to_execution_inverse_time(self, value):
        return value / self.inverse_time

    def from_execution_inverse_time(self, value):
        return value * self.inverse_time

    def to_execution_inverse_time_squared(self, value):
        return value / (self.inverse_time * self.inverse_time)

    def from_execution_inverse_time_squared(self, value):
        return value * self.inverse_time * self.inverse_time

    def to_execution_grid(self, grid: UniformGrid) -> UniformGrid:
        return UniformGrid(
            grid.nx,
            grid.ny,
            grid.nz,
            self.to_execution_length(grid.lx),
            self.to_execution_length(grid.ly),
            self.to_execution_length(grid.lz),
        )
