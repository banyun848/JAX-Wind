"""Discrete-to-true gradient ratios for a Monin-Obukhov surface layer.

Why this module exists
----------------------
A large-eddy simulation cannot resolve the surface layer.  The mean wind varies
there as ``ln z`` modified by stability, so most of the shear lives inside the
first cell or two, and a difference formula -- which assumes the profile is
locally polynomial -- returns a gradient that is systematically too steep.  The
subfilter closure reads that gradient, so the error propagates into the modelled
stress and then into the mean profile.  Porte-Agel, Meneveau and Parlange
(J. Fluid Mech. 415, 261, 2000, Appendix) correct it by rescaling the plane-mean
wall-normal gradient onto what similarity theory predicts.

This module supplies the rescaling factors.  It is deliberately standalone: it
is pure similarity theory plus quadrature over the mesh, with no dependence on
the solver's state, boundary handling or array layout, so it can be read,
tested and reused on its own.  :mod:`jaxwind.wall` applies what is computed
here.

Finite volume versus finite difference
--------------------------------------
The original correction is derived for a code storing **point values** at cell
centres.  A finite-volume unknown is a **cell average**, and averaging a concave
function biases it low -- most strongly next to the wall, where ``ln z`` curves
hardest.  The two discretisations therefore need different factors.  Under
neutral conditions the first interior face gives

    finite volume    (ubar_2 - ubar_1)/h = (u*/kappa) ln 4 / h   ->  39% too steep
    finite difference  (u_2 - u_1)/h     = (u*/kappa) ln 3 / h   ->  10% too steep

so transplanting the finite-difference constant into a finite-volume code
under-corrects by about a factor of three.

The closed form
---------------
Write the surface-layer wind and its gradient as

    u(z)     = (u*/kappa) [ ln(z/z0) - psi_m(z/L) ]
    du/dz    = (u*/kappa z) phi_m(z/L)

and measure heights in cell counts, ``xi = z/h``, so the stability parameter of
the mesh is ``s = h/L``.  Averaging over cell ``m``, which spans
``[(m-1)h, m h]``, and differencing two neighbours makes every ``ln h`` and
``ln z0`` cancel, leaving a ratio that depends only on ``m`` and ``s``:

    ratio(m, s) = m [ Lambda(m) - D2Psi(m, s) / s ] / phi_m(m s)

    Lambda(m)   = (m+1) ln(m+1) - 2 m ln m + (m-1) ln(m-1)      (0 ln 0 = 0)
    D2Psi(m, s) = Psi((m+1)s) - 2 Psi(m s) + Psi((m-1)s)

where ``Psi`` is an antiderivative of ``psi_m``.  ``Lambda`` is the neutral
part: ``Lambda(1) = 2 ln 2 = ln 4``, the 39 percent above.  Everything else is
the stability correction, and it vanishes as ``s -> 0`` because ``Psi`` is
quadratic near the origin.

``Psi`` is elementary on both branches, so the whole ratio is closed form.  On
the stable branch ``psi_m = -beta zeta`` integrates immediately to
``-beta zeta^2 / 2`` and the ratio collapses to

    ratio(m, s) = m [ Lambda(m) + beta s ] / (1 + beta m s) .

On the unstable branch the Paulson function is integrated by substituting
``zeta = (1 - x^4)/gamma``, which turns ``Psi`` into a combination of
``x^3 ln(1+x)``, ``x^3 ln(1+x^2)`` and ``x^3 arctan x`` -- all elementary.
:func:`psi_m_integral` carries the result.

What the ratios look like
-------------------------
For a 12.5 m cell (GABLS1), with beta = 4.8 and gamma = 16:

    L (m)     h/L     regime      face 1   face 2   face 3
      -50   -0.250    unstable    1.532    1.063    1.027
     -200   -0.062    unstable    1.452    1.057    1.025
      inf    0.000    neutral     1.386    1.046    1.019
      400    0.031    stable      1.337    1.036    1.014
      100    0.125    stable      1.243    1.021    1.007
       25    0.500    very stable 1.115    1.008    1.002

Two consequences matter in practice.  Stable stratification straightens the
profile -- a difference formula is exact on a linear profile -- so the
correction switches itself off as ``s`` grows, while convective conditions
sharpen the near-wall curvature and make it larger.  The neutral value is
therefore **never conservative**: it over-corrects in stable air and
under-corrects in unstable air, by around ten percent at GABLS1-like stability.

Limitations
-----------
The derivation assumes a uniform vertical mesh, because the telescoping above
relies on equal cell heights, and it assumes the plane-mean profile follows
similarity theory, which is what makes a plane-mean rescaling meaningful.  On a
stretched grid the cell averages must be integrated with the actual face
heights.  ``maximum_abs_zeta`` mirrors the clipping in
:mod:`jaxwind.surface`, so very large ``|zeta|`` saturates rather than running
off into the free-convection or z-less limits.

References
----------
Businger et al., J. Atmos. Sci. 28, 181 (1971); Dyer, Bound.-Layer Meteorol. 7,
363 (1974) -- the stability functions.
Paulson, J. Appl. Meteorol. 9, 857 (1970) -- the integrated unstable form.
Porte-Agel, Meneveau and Parlange, J. Fluid Mech. 415, 261 (2000), Appendix --
the gradient correction, in its finite-difference form.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

__all__ = [
    "face_ratio_array",
    "phi_m_array",
    "psi_m_integral_array",
    "SurfaceLayerSimilarity",
    "face_ratio",
    "neutral_face_ratio",
    "phi_m",
    "psi_m",
    "psi_m_integral",
]


@dataclass(frozen=True, slots=True)
class SurfaceLayerSimilarity:
    """Businger-Dyer momentum coefficients, matching :mod:`jaxwind.surface`."""

    stable_slope: float = 4.8
    unstable_coefficient: float = 16.0
    maximum_abs_zeta: float = 10.0

    def __post_init__(self) -> None:
        if self.stable_slope <= 0.0:
            raise ValueError("the stable slope must be positive")
        if self.unstable_coefficient <= 0.0:
            raise ValueError("the unstable coefficient must be positive")
        if self.maximum_abs_zeta <= 0.0:
            raise ValueError("the zeta bound must be positive")

    def bounded(self, zeta: float) -> float:
        """Clip ``zeta`` the way the surface exchange does."""
        return max(-self.maximum_abs_zeta, min(self.maximum_abs_zeta, zeta))


def phi_m(zeta: float, model: SurfaceLayerSimilarity | None = None) -> float:
    """Dimensionless wind shear ``(kappa z / u*) du/dz``."""
    model = model or SurfaceLayerSimilarity()
    zeta = model.bounded(zeta)
    if zeta >= 0.0:
        return 1.0 + model.stable_slope * zeta
    return (1.0 - model.unstable_coefficient * zeta) ** -0.25


def psi_m(zeta: float, model: SurfaceLayerSimilarity | None = None) -> float:
    """Integrated stability correction appearing in the wind profile."""
    model = model or SurfaceLayerSimilarity()
    zeta = model.bounded(zeta)
    if zeta >= 0.0:
        return -model.stable_slope * zeta
    x = (1.0 - model.unstable_coefficient * zeta) ** 0.25
    return (
        2.0 * math.log(0.5 * (1.0 + x))
        + math.log(0.5 * (1.0 + x * x))
        - 2.0 * math.atan(x)
        + 0.5 * math.pi
    )


def _unstable_primitive(x: float, model: SurfaceLayerSimilarity) -> float:
    """``integral psi_m x^3 dx`` under ``zeta = (1 - x^4)/gamma``.

    Each term is elementary:

        integral x^3 ln(1+x)   = (x^4-1)/4 ln(1+x)   - (x^4/4 - x^3/3 + x^2/2 - x)/4
        integral x^3 ln(1+x^2) = (x^4-1)/4 ln(1+x^2) - (x^4/2 - x^2)/4
        integral x^3 arctan x  = (x^4-1)/4 arctan x  - (x^3/3 - x)/4

    The ``(x^4 - 1)`` numerators come from folding the logarithm that the
    integration by parts leaves behind back into the leading term.
    """
    del model
    quartic = x**4
    log_term = (quartic - 1.0) / 4.0 * math.log1p(x) - 0.25 * (
        quartic / 4.0 - x**3 / 3.0 + x * x / 2.0 - x
    )
    square_log_term = (quartic - 1.0) / 4.0 * math.log1p(x * x) - 0.25 * (
        quartic / 2.0 - x * x
    )
    arctan_term = (quartic - 1.0) / 4.0 * math.atan(x) - 0.25 * (x**3 / 3.0 - x)
    constant = 0.5 * math.pi - 3.0 * math.log(2.0)
    return (
        2.0 * log_term
        + square_log_term
        - 2.0 * arctan_term
        + constant * quartic / 4.0
    )


def psi_m_integral(
    zeta: float, model: SurfaceLayerSimilarity | None = None
) -> float:
    """``Psi(zeta) = integral_0^zeta psi_m(t) dt``, in closed form.

    ``Psi`` is what turns the point-wise profile into a cell average, so it is
    the one extra ingredient a finite-volume correction needs over the
    finite-difference one.  It is quadratic near the origin, which is why the
    stability correction vanishes smoothly in the neutral limit.
    """
    model = model or SurfaceLayerSimilarity()
    zeta = model.bounded(zeta)
    if zeta == 0.0:
        return 0.0
    if zeta > 0.0:
        return -0.5 * model.stable_slope * zeta * zeta
    gamma = model.unstable_coefficient
    upper = (1.0 - gamma * zeta) ** 0.25
    return -(4.0 / gamma) * (
        _unstable_primitive(upper, model) - _unstable_primitive(1.0, model)
    )


def neutral_face_ratio(face: int) -> float:
    """``Lambda(m) * m``: the neutral finite-volume ratio.

    ``Lambda(1) = 2 ln 2 = ln 4``, so the first interior face is 39 percent too
    steep -- against ``ln 3``, 10 percent, for a finite-difference code.
    """
    if face < 1:
        raise ValueError("face index starts at one")

    def term(count: int) -> float:
        """``n ln n`` extended by continuity so the m = 1 endpoint works."""
        return 0.0 if count <= 0 else count * math.log(count)

    return face * (term(face + 1) - 2.0 * term(face) + term(face - 1))


def face_ratio(
    face: int,
    mesh_stability: float = 0.0,
    model: SurfaceLayerSimilarity | None = None,
) -> float:
    """Discrete-to-true gradient ratio on interior face ``m``.

    ``mesh_stability`` is ``s = h / L``: the cell height in Obukhov units,
    positive for stable stratification.  Passing zero recovers
    :func:`neutral_face_ratio`.

    The second difference of ``Psi`` is divided by ``s``, and both vanish
    together in the neutral limit, so for small ``|s|`` the quotient is replaced
    by its series ``-psi_m'(0) s``, which is exact to the order that survives.
    """
    if face < 1:
        raise ValueError("face index starts at one")
    model = model or SurfaceLayerSimilarity()
    neutral = neutral_face_ratio(face)
    if mesh_stability == 0.0:
        return neutral

    # Psi ~ psi_m'(0) zeta^2 / 2, whose second difference is psi_m'(0) s^2;
    # evaluating the quotient directly below that scale is pure cancellation.
    if abs(mesh_stability) < 1.0e-4:
        slope = -model.stable_slope if mesh_stability > 0.0 else (
            0.25 * model.unstable_coefficient
        )
        stability = -slope * mesh_stability
    else:
        second_difference = (
            psi_m_integral((face + 1) * mesh_stability, model)
            - 2.0 * psi_m_integral(face * mesh_stability, model)
            + psi_m_integral((face - 1) * mesh_stability, model)
        )
        stability = -second_difference / mesh_stability
    return face * (neutral / face + stability) / phi_m(face * mesh_stability, model)


# ---------------------------------------------------------------------------
# Traceable form.
#
# Everything above is plain Python: readable, exactly testable, and the place
# the derivation lives.  The solver needs the same ratios for an Obukhov length
# that is a traced array, so the branch on the sign of zeta becomes a
# ``jnp.where`` and the small-|s| series is selected the same way.  The two
# implementations are checked against each other in the tests.
# ---------------------------------------------------------------------------


def _unstable_primitive_array(x, jnp):
    quartic = x**4
    log_term = (quartic - 1.0) / 4.0 * jnp.log1p(x) - 0.25 * (
        quartic / 4.0 - x**3 / 3.0 + x * x / 2.0 - x
    )
    square_log_term = (quartic - 1.0) / 4.0 * jnp.log1p(x * x) - 0.25 * (
        quartic / 2.0 - x * x
    )
    arctan_term = (quartic - 1.0) / 4.0 * jnp.arctan(x) - 0.25 * (x**3 / 3.0 - x)
    constant = 0.5 * math.pi - 3.0 * math.log(2.0)
    return (
        2.0 * log_term
        + square_log_term
        - 2.0 * arctan_term
        + constant * quartic / 4.0
    )


def psi_m_integral_array(zeta, model: SurfaceLayerSimilarity | None = None):
    """``Psi(zeta)`` for traced input; see :func:`psi_m_integral`."""
    import jax.numpy as jnp

    model = model or SurfaceLayerSimilarity()
    zeta = jnp.clip(zeta, -model.maximum_abs_zeta, model.maximum_abs_zeta)
    stable = -0.5 * model.stable_slope * zeta * zeta
    # Evaluate the unstable branch on a clipped argument so the stable side
    # never raises a negative base to a fractional power under vmap/grad.
    negative = jnp.minimum(zeta, 0.0)
    upper = (1.0 - model.unstable_coefficient * negative) ** 0.25
    unstable = -(4.0 / model.unstable_coefficient) * (
        _unstable_primitive_array(upper, jnp)
        - _unstable_primitive_array(jnp.asarray(1.0, upper.dtype), jnp)
    )
    return jnp.where(zeta >= 0.0, stable, unstable)


def phi_m_array(zeta, model: SurfaceLayerSimilarity | None = None):
    """Dimensionless shear for traced input; see :func:`phi_m`."""
    import jax.numpy as jnp

    model = model or SurfaceLayerSimilarity()
    zeta = jnp.clip(zeta, -model.maximum_abs_zeta, model.maximum_abs_zeta)
    negative = jnp.minimum(zeta, 0.0)
    return jnp.where(
        zeta >= 0.0,
        1.0 + model.stable_slope * zeta,
        (1.0 - model.unstable_coefficient * negative) ** -0.25,
    )


def face_ratio_array(
    face: int,
    mesh_stability,
    model: SurfaceLayerSimilarity | None = None,
):
    """Discrete-to-true gradient ratio for traced ``mesh_stability``.

    ``face`` stays a Python integer because it indexes the mesh, so the neutral
    part is a compile-time constant; only the stability part is traced.
    """
    import jax.numpy as jnp

    if face < 1:
        raise ValueError("face index starts at one")
    model = model or SurfaceLayerSimilarity()
    stability = jnp.asarray(mesh_stability)
    neutral = neutral_face_ratio(face)

    safe = jnp.where(jnp.abs(stability) < 1.0e-4, 1.0e-4, stability)
    second_difference = (
        psi_m_integral_array((face + 1) * safe, model)
        - 2.0 * psi_m_integral_array(face * safe, model)
        + psi_m_integral_array((face - 1) * safe, model)
    )
    quotient = -second_difference / safe
    slope = jnp.where(
        stability > 0.0, -model.stable_slope, 0.25 * model.unstable_coefficient
    )
    series = -slope * stability
    correction = jnp.where(jnp.abs(stability) < 1.0e-4, series, quotient)
    return (
        face
        * (neutral / face + correction)
        / phi_m_array(face * stability, model)
    )
