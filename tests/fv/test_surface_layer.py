"""The closed-form surface-layer gradient ratios against direct quadrature."""

import math
import unittest

from jaxwind.surface_layer import (
    SurfaceLayerSimilarity,
    face_ratio,
    neutral_face_ratio,
    phi_m,
    psi_m,
    psi_m_integral,
)

KAPPA = 0.4
ROUGHNESS = 2.0e-4


def _integrate(function, lower: float, upper: float, panels: int = 20_000) -> float:
    """Composite Simpson, adequate for the smooth integrands used here."""
    if panels % 2:
        panels += 1
    width = (upper - lower) / panels
    total = function(lower) + function(upper)
    for index in range(1, panels):
        total += (4.0 if index % 2 else 2.0) * function(lower + index * width)
    return total * width / 3.0


class SurfaceLayerTest(unittest.TestCase):
    model = SurfaceLayerSimilarity()

    def test_psi_integral_matches_quadrature(self) -> None:
        for zeta in (-8.0, -2.0, -0.5, -0.05, 0.05, 0.5, 2.0, 8.0):
            reference = _integrate(lambda t: psi_m(t, self.model), 0.0, zeta)
            self.assertAlmostEqual(
                psi_m_integral(zeta, self.model) / reference, 1.0, places=6
            )

    def test_neutral_first_face_is_log_four(self) -> None:
        # The finite-volume value; a finite-difference code would see ln 3.
        self.assertAlmostEqual(neutral_face_ratio(1), math.log(4.0), places=12)
        self.assertAlmostEqual(face_ratio(1, 0.0), math.log(4.0), places=12)

    def test_face_ratio_matches_quadrature_across_stability(self) -> None:
        # An independent route to the cell average that never touches Psi.
        # Integrating by parts turns the singular double integral into
        #     int_a^b u dz = b u(b) - a u(a) - (1/kappa) int_a^b phi_m dz ,
        # because z du/dz = phi_m(z/L)/kappa is smooth and bounded, so Simpson
        # converges quickly and the wind profile itself stays analytic.
        height = 12.5

        def gradient(z, obukhov):
            return phi_m(z / obukhov, self.model) / (KAPPA * z)

        def wind(z, obukhov):
            return (
                math.log(z / ROUGHNESS) - psi_m(z / obukhov, self.model)
            ) / KAPPA

        def average(cell, obukhov):
            lower = max((cell - 1) * height, 1.0e-12)
            upper = cell * height
            shear = _integrate(
                lambda t: phi_m(t / obukhov, self.model), lower, upper, 4_000
            )
            return (
                upper * wind(upper, obukhov)
                - (lower * wind(lower, obukhov) if cell > 1 else 0.0)
                - shear / KAPPA
            ) / (upper - lower)

        for obukhov in (-50.0, -200.0, 400.0, 100.0, 25.0):
            for face in (1, 2, 3):
                reference = (
                    (average(face + 1, obukhov) - average(face, obukhov)) / height
                ) / gradient(face * height, obukhov)
                self.assertAlmostEqual(
                    face_ratio(face, height / obukhov, self.model) / reference,
                    1.0,
                    places=4,
                    msg=f"L={obukhov} face={face}",
                )

    def test_stable_branch_matches_its_analytic_collapse(self) -> None:
        # psi_m = -beta zeta integrates to -beta zeta^2 / 2, so the ratio
        # collapses to m [Lambda(m) + beta s] / (1 + beta m s).
        beta = self.model.stable_slope
        for stability in (0.01, 0.1, 0.5, 1.0):
            for face in (1, 2, 3):
                expected = (
                    face
                    * (neutral_face_ratio(face) / face + beta * stability)
                    / (1.0 + beta * face * stability)
                )
                self.assertAlmostEqual(
                    face_ratio(face, stability, self.model), expected, places=10
                )

    def test_correction_vanishes_in_the_very_stable_limit(self) -> None:
        # A strongly stable profile is nearly linear, and a difference formula
        # is exact on a linear profile, so the correction switches itself off.
        self.assertLess(face_ratio(1, 4.0, self.model), 1.05)
        self.assertGreater(face_ratio(1, 4.0, self.model), 1.0)

    def test_convective_conditions_need_a_larger_correction(self) -> None:
        unstable = face_ratio(1, -0.25, self.model)
        neutral = face_ratio(1, 0.0, self.model)
        stable = face_ratio(1, 0.25, self.model)
        self.assertGreater(unstable, neutral)
        self.assertGreater(neutral, stable)

    def test_ratio_is_continuous_through_neutral(self) -> None:
        neutral = face_ratio(1, 0.0, self.model)
        for stability in (1.0e-6, 1.0e-5, 1.0e-4):
            self.assertAlmostEqual(face_ratio(1, stability, self.model), neutral, places=3)
            self.assertAlmostEqual(face_ratio(1, -stability, self.model), neutral, places=3)

    def test_face_index_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            face_ratio(0, 0.0)
        with self.assertRaises(ValueError):
            neutral_face_ratio(0)


if __name__ == "__main__":
    unittest.main()
