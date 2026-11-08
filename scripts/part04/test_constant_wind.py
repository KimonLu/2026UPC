"""Check the wind convention, frame change and first-event export."""

import unittest
from dataclasses import replace

import numpy as np

from scripts.part04.run_part4 import evaluate_case
from scripts.common.still_air import Motion, Physics, Settings, aerodynamic_acceleration, simulate


class ConstantWindTests(unittest.TestCase):
    def test_crosswind_sets_drag_and_magnus_directions(self):
        physics = Physics(wind=(3.0, 4.0, 0.0))
        motion = Motion((30.0, 0.0, 0.0), 0.20, (0.0, 0.0, 40.0))
        scale = physics.density * physics.area / (2 * physics.mass)
        drag, magnus = aerodynamic_acceleration(0.0, motion.v0, motion, physics)
        np.testing.assert_allclose(drag, scale * 0.20 * np.hypot(27, 4) * np.array([-27., 4., 0.]))
        np.testing.assert_allclose(magnus, scale * physics.radius * 40 * np.array([4., 27., 0.]))

    def test_horizontal_boost_preserves_air_relative_motion(self):
        r0 = (0.0, 0.0, 0.11)
        motion = Motion(np.array([30.0, 9.0, 7.0]), 0.38, (12.0, -5.0, -60.0))
        physics = Physics(wind=(3.0, 4.0, 0.0))
        boost = np.array([-5.0, 2.0, 0.0])
        shifted_motion = replace(motion, v0=motion.v0 + boost)
        shifted_physics = replace(physics, wind=tuple(physics.wind + boost))
        original = simulate(r0, motion, physics)
        shifted = simulate(r0, shifted_motion, shifted_physics)
        times = np.array([0.0, 0.05, 0.10, 0.30])
        expected = original.sample(times)
        expected[:, :3] += times[:, None] * boost
        expected[:, 3:] += boost
        np.testing.assert_allclose(shifted.sample(times), expected, rtol=0, atol=1e-10)

    def test_ground_first_has_no_crossing_or_post_contact_residual(self):
        times = np.array([0.0, 0.05, 0.10])
        positions = np.array([[0., 0., 0.11], [1.50, 0.45, 0.44], [2.96, 0.86, 0.74]])
        record, _ = evaluate_case('4B', 'test', times, positions,
                                  Motion((20., 0., 0.4), 0.2),
                                  Physics(wind=(0., 3., 0.)), Settings(), np.eye(3))
        self.assertEqual(record['event'], 'ground_first')
        self.assertGreater(record['event_time_s'], times[1])
        self.assertLess(record['event_time_s'], times[2])
        self.assertIsNone(record['goal'])
        self.assertTrue(all(np.isfinite(record['residuals_m'][0])))
        self.assertEqual(record['residuals_m'][1], [None] * 3)
        self.assertIsNone(record['rms_m'])


if __name__ == '__main__':
    unittest.main()
