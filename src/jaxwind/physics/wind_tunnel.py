"""Pure wind-tunnel forcing choices for actuator-disk and actuator-line LES.

The configurations in this module contain no array implementation.  The
solver algebra owns coordinates, reductions, and the distributed array layout.
"""

from __future__ import annotations

from dataclasses import dataclass
import math



@dataclass(frozen=True, slots=True)
class PureThrustActuatorDisk:
    """Uniform, non-rotating actuator disk in execution units.

    ``thrust_coefficient_prime`` is based on the disk-normal velocity.  The
    ``normal_smoothing_width`` and ``transverse_smoothing_width`` use the
    Gaussian convention ``exp(-(x / epsilon)**2)``.  The geometric disk is
    convolved with this anisotropic Gaussian and discretely renormalized so
    the integrated force is independent of grid alignment.
    """

    x: float
    y: float
    z: float
    diameter: float
    thrust_coefficient_prime: float
    normal_smoothing_width: float
    transverse_smoothing_width: float
    hub_diameter: float = 0.0
    yaw_degrees: float = 0.0
    filtered_velocity_correction: bool = True
    prescribed_inflow_velocity: float = 0.0
    prescribed_thrust_coefficient: float = 0.0

    def __post_init__(self) -> None:
        finite = (
            self.x,
            self.y,
            self.z,
            self.diameter,
            self.thrust_coefficient_prime,
            self.normal_smoothing_width,
            self.transverse_smoothing_width,
            self.hub_diameter,
            self.yaw_degrees,
            self.prescribed_inflow_velocity,
            self.prescribed_thrust_coefficient,
        )
        if not all(math.isfinite(value) for value in finite):
            raise ValueError("actuator-disk parameters must be finite")
        if self.diameter <= 0.0:
            raise ValueError("actuator-disk diameter must be positive")
        if self.thrust_coefficient_prime < 0.0:
            raise ValueError("local thrust coefficient must be nonnegative")
        prescribed = self.prescribed_inflow_velocity > 0.0
        if prescribed != (self.prescribed_thrust_coefficient > 0.0):
            raise ValueError(
                "prescribed inflow velocity and thrust coefficient must both "
                "be positive or both zero"
            )
        if min(
            self.normal_smoothing_width,
            self.transverse_smoothing_width,
        ) <= 0.0:
            raise ValueError("actuator-disk smoothing widths must be positive")
        if self.hub_diameter < 0.0 or self.hub_diameter >= self.diameter:
            raise ValueError("hub diameter must lie in [0, rotor diameter)")



@dataclass(frozen=True, slots=True)
class BladeElementActuatorLine:
    """Rigid or small-deflection modal actuator lines with tabulated airfoils.

    Geometry and angular velocity are expressed in the solver's execution
    units.  Each blade element carries one radial quadrature width, chord,
    twist, and airfoil-table index.  All polars share ``polar_alpha_degrees``;
    data importers may form that common grid by exactly resampling piecewise
    linear source tables onto their union.

    The azimuth convention is blade 1 pointing upward at zero degrees.
    Positive angular velocity advances it toward the rotor-plane horizontal
    basis.  Gaussian interpolation and projection use
    ``exp(-(distance / smoothing_width)**2)`` and are discretely normalized.
    ``element_gaussian_widths`` optionally overrides the scalar width with one
    physical Gaussian width per radial element; it is repeated for each blade.
    """

    x: float
    y: float
    z: float
    blade_count: int
    hub_radius: float
    tip_radius: float
    angular_velocity: float
    smoothing_width: float
    element_radii: tuple[float, ...]
    element_widths: tuple[float, ...]
    element_chords: tuple[float, ...]
    element_twist_degrees: tuple[float, ...]
    element_airfoil_ids: tuple[int, ...]
    polar_alpha_degrees: tuple[float, ...]
    polar_lift_coefficients: tuple[tuple[float, ...], ...]
    polar_drag_coefficients: tuple[tuple[float, ...], ...]
    pitch_degrees: float = 0.0
    yaw_degrees: float = 0.0
    tilt_degrees: float = 0.0
    precone_degrees: float = 0.0
    initial_azimuth_degrees: float = 0.0
    tip_loss: bool = True
    root_loss: bool = True
    element_gaussian_widths: tuple[float, ...] = ()
    element_flap_displacements: tuple[float, ...] = ()
    element_edge_displacements: tuple[float, ...] = ()
    element_flap_slopes: tuple[float, ...] = ()
    element_edge_slopes: tuple[float, ...] = ()
    element_flap_velocities: tuple[float, ...] = ()
    element_edge_velocities: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        scalar_values = (
            self.x,
            self.y,
            self.z,
            self.hub_radius,
            self.tip_radius,
            self.angular_velocity,
            self.smoothing_width,
            self.pitch_degrees,
            self.yaw_degrees,
            self.tilt_degrees,
            self.precone_degrees,
            self.initial_azimuth_degrees,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("actuator-line parameters must be finite")
        if isinstance(self.blade_count, bool) or self.blade_count <= 0:
            raise ValueError("actuator line requires a positive blade count")
        if self.hub_radius < 0.0 or self.tip_radius <= self.hub_radius:
            raise ValueError("actuator-line radii must satisfy 0 <= hub < tip")
        if self.smoothing_width <= 0.0:
            raise ValueError("actuator-line smoothing width must be positive")

        element_count = len(self.element_radii)
        element_arrays = (
            self.element_widths,
            self.element_chords,
            self.element_twist_degrees,
            self.element_airfoil_ids,
        )
        if element_count == 0 or any(
            len(values) != element_count for values in element_arrays
        ):
            raise ValueError("actuator-line element arrays must have equal length")
        if not all(
            math.isfinite(value)
            for values in (
                self.element_radii,
                self.element_widths,
                self.element_chords,
                self.element_twist_degrees,
            )
            for value in values
        ):
            raise ValueError("actuator-line element data must be finite")
        if any(
            right <= left
            for left, right in zip(self.element_radii, self.element_radii[1:])
        ):
            raise ValueError("actuator-line element radii must be strictly increasing")
        if (
            self.element_radii[0] < self.hub_radius
            or self.element_radii[-1] > self.tip_radius
        ):
            raise ValueError("actuator-line elements must lie between hub and tip")
        if min(self.element_widths) <= 0.0 or min(self.element_chords) <= 0.0:
            raise ValueError("actuator-line widths and chords must be positive")
        if self.element_gaussian_widths and (
            len(self.element_gaussian_widths) != element_count
            or not all(
                math.isfinite(value) and value > 0.0
                for value in self.element_gaussian_widths
            )
        ):
            raise ValueError(
                "actuator-line Gaussian widths must be positive and contain "
                "one value per radial element"
            )
        point_count = self.blade_count * element_count
        deformation_arrays = (
            self.element_flap_displacements,
            self.element_edge_displacements,
            self.element_flap_slopes,
            self.element_edge_slopes,
            self.element_flap_velocities,
            self.element_edge_velocities,
        )
        if any(
            values and len(values) != point_count
            for values in deformation_arrays
        ):
            raise ValueError(
                "actuator-line deformation arrays must be empty or contain "
                "one value per blade element"
            )
        if not all(
            math.isfinite(value)
            for values in deformation_arrays
            for value in values
        ):
            raise ValueError("actuator-line deformation values must be finite")

        alpha_count = len(self.polar_alpha_degrees)
        polar_count = len(self.polar_lift_coefficients)
        if alpha_count < 2 or polar_count == 0:
            raise ValueError("actuator line requires nonempty aerodynamic polars")
        if len(self.polar_drag_coefficients) != polar_count:
            raise ValueError("actuator-line lift and drag polar counts must match")
        if any(
            right <= left
            for left, right in zip(
                self.polar_alpha_degrees,
                self.polar_alpha_degrees[1:],
            )
        ):
            raise ValueError("polar angles of attack must be strictly increasing")
        if not all(
            len(row) == alpha_count
            for table in (
                self.polar_lift_coefficients,
                self.polar_drag_coefficients,
            )
            for row in table
        ):
            raise ValueError("all actuator-line polar rows must share the alpha grid")
        if not all(
            math.isfinite(value)
            for table in (
                self.polar_lift_coefficients,
                self.polar_drag_coefficients,
            )
            for row in table
            for value in row
        ):
            raise ValueError("actuator-line polar coefficients must be finite")
        if any(
            value < 0.0
            for row in self.polar_drag_coefficients
            for value in row
        ):
            raise ValueError("actuator-line drag coefficients must be nonnegative")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < polar_count
            for index in self.element_airfoil_ids
        ):
            raise ValueError("actuator-line airfoil index is outside the polar table")


    @property
    def point_smoothing_widths(self) -> tuple[float, ...]:
        """Return Gaussian widths in flattened blade-major point order."""
        radial = (
            self.element_gaussian_widths
            if self.element_gaussian_widths
            else (self.smoothing_width,) * len(self.element_radii)
        )
        return radial * self.blade_count


@dataclass(frozen=True, slots=True)
class BladeElementActuatorDisk(BladeElementActuatorLine):
    """Azimuthally averaged blade-element disk with thrust and swirl loading.

    The force-projection width follows the legacy WIRE-LES ADMR convention.
    For radial element ``i`` it is
    ``hypot(r_i * 2*pi/N_phi, delta_r_i)``, where ``N_phi`` is the number of
    virtual azimuthal elements used to form the disk average.  The inherited
    scalar ``smoothing_width`` remains the actuator-line width and is not used
    by the disk projection.
    """

    smearing_azimuthal_elements: int = 64

    def __post_init__(self) -> None:
        super(BladeElementActuatorDisk, self).__post_init__()
        if self.tilt_degrees != 0.0 or self.precone_degrees != 0.0:
            raise ValueError("AD-BEM currently supports an upright rotor axis")
        if self.yaw_degrees != 0.0:
            raise ValueError("AD-BEM currently supports zero yaw")
        if (
            isinstance(self.smearing_azimuthal_elements, bool)
            or self.smearing_azimuthal_elements <= 0
        ):
            raise ValueError("AD-BEM requires a positive azimuthal smearing count")
        if any(
            (
                self.element_flap_displacements,
                self.element_edge_displacements,
                self.element_flap_slopes,
                self.element_edge_slopes,
                self.element_flap_velocities,
                self.element_edge_velocities,
            )
        ):
            raise ValueError("AD-BEM does not accept blade deformation state")

    @property
    def element_smoothing_widths(self) -> tuple[float, ...]:
        """Legacy ADMR Gaussian widths, one for each radial element."""

        angle = 2.0 * math.pi / self.smearing_azimuthal_elements
        return tuple(
            math.hypot(radius * angle, radial_width)
            for radius, radial_width in zip(
                self.element_radii, self.element_widths
            )
        )



@dataclass(frozen=True, slots=True)
class NacelleTowerDrag:
    """Smoothed nacelle and tapered-tower drag in execution units."""

    x: float
    y: float
    hub_height: float
    nacelle_length: float
    nacelle_diameter: float
    nacelle_drag_coefficient: float
    tower_base_diameter: float
    tower_top_diameter: float
    tower_drag_coefficient: float
    smoothing_width: float

    def __post_init__(self) -> None:
        values = (
            self.x, self.y, self.hub_height, self.nacelle_length,
            self.nacelle_diameter, self.nacelle_drag_coefficient,
            self.tower_base_diameter, self.tower_top_diameter,
            self.tower_drag_coefficient, self.smoothing_width,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nacelle/tower parameters must be finite")
        if min(
            self.hub_height, self.nacelle_length, self.nacelle_diameter,
            self.tower_base_diameter, self.tower_top_diameter,
            self.smoothing_width,
        ) <= 0.0:
            raise ValueError("nacelle/tower dimensions must be positive")
        if min(self.nacelle_drag_coefficient, self.tower_drag_coefficient) < 0.0:
            raise ValueError("nacelle/tower drag coefficients must be nonnegative")



__all__ = [
    "BladeElementActuatorDisk",
    "BladeElementActuatorLine",
    "NacelleTowerDrag",
    "PureThrustActuatorDisk",
]
