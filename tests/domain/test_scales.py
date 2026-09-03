from __future__ import annotations

import math
import unittest

from jaxwind.domain import ScaleSystem, UniformGrid


class ScaleSystemTests(unittest.TestCase):
    def test_mechanical_round_trips_and_derived_scales(self) -> None:
        scales = ScaleSystem(1000.0, 8.0)
        self.assertEqual(scales.time, 125.0)
        self.assertEqual(scales.acceleration, 0.064)
        self.assertEqual(scales.inverse_time, 0.008)
        pairs = (
            (scales.to_execution_length, scales.from_execution_length, 4000.0),
            (
                scales.to_execution_velocity,
                scales.from_execution_velocity,
                -3.25,
            ),
            (scales.to_execution_time, scales.from_execution_time, 3600.0),
            (
                scales.to_execution_acceleration,
                scales.from_execution_acceleration,
                0.002,
            ),
            (
                scales.to_execution_inverse_time,
                scales.from_execution_inverse_time,
                1.0e-4,
            ),
        )
        for lower, lift, value in pairs:
            with self.subTest(value=value):
                self.assertTrue(
                    math.isclose(lift(lower(value)), value, rel_tol=1.0e-15)
                )

    def test_grid_lowering_changes_only_lengths(self) -> None:
        grid = UniformGrid(16, 16, 32, 4000.0, 4000.0, 1000.0)
        execution = ScaleSystem(1000.0, 8.0).to_execution_grid(grid)
        self.assertEqual(
            (execution.nx, execution.ny, execution.nz), (16, 16, 32)
        )
        self.assertEqual((execution.lx, execution.ly, execution.lz), (4.0, 4.0, 1.0))


if __name__ == "__main__":
    unittest.main()
