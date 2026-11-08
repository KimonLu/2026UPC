"""Analytical and synthetic checks for the Section 3 solver."""

import unittest
from dataclasses import replace

import numpy as np

from scripts.common.still_air import (
    Motion, Physics, Settings, aerodynamic_acceleration, fit_from_positions,
    simulate, spin_at, spin_basis,
)


class StillAirTests(unittest.TestCase):
    def setUp(self):
        self.physics = Physics()
        self.r0 = np.array([0.0, 0.0, self.physics.radius])
        self.v0 = np.array([30.0, 2.0, 6.0])

    def test_gravity_and_buoyancy_match_closed_form(self):
        for physics in (replace(self.physics, density=0.0), self.physics):
            with self.subTest(density=physics.density):
                flight = simulate(self.r0, Motion(self.v0, 0.0), physics)
                t = 11.0 / self.v0[0]
                g_eff = physics.gravity * (1 - physics.density * physics.volume / physics.mass)
                expected = self.r0 + self.v0 * t - np.array([0.0, 0.0, g_eff * t**2 / 2])
                self.assertEqual(flight.event, 'goal_inside')
                self.assertAlmostEqual(flight.time, t, places=11)
                np.testing.assert_allclose(flight.states[-1, :3], expected, atol=1e-10)

    def test_drag_and_magnus_magnitude_and_direction(self):
        physics = replace(self.physics, wind=(2.0, 0.0, 0.0))
        motion = Motion((30.0, 0.0, 0.0), 0.20, (0.0, 0.0, 40.0))
        drag, lift = aerodynamic_acceleration(0.0, motion.v0, motion, physics)
        scale = physics.density * physics.area / (2 * physics.mass)
        np.testing.assert_allclose(drag, [-scale * 0.20 * 28**2, 0.0, 0.0])
        lift_coefficient = physics.radius * 40 / 28
        np.testing.assert_allclose(lift, [0.0, scale * 28**2 * lift_coefficient, 0.0])
        np.testing.assert_allclose(aerodynamic_acceleration(0.0, physics.wind, motion, physics), 0)
        arbitrary_v = np.array([30.0, 1.0, 3.0])
        drag, lift = aerodynamic_acceleration(0.0, arbitrary_v, motion, physics)
        relative = arbitrary_v - physics.wind
        self.assertLess(drag @ relative, 0)
        self.assertAlmostEqual(lift @ relative, 0, places=12)

    def test_energy_and_horizontal_direction(self):
        motion = Motion(self.v0, 0.2)
        flight = simulate(self.r0, motion, self.physics)
        states = flight.states
        g_eff = self.physics.gravity * (1 - self.physics.density * self.physics.volume / self.physics.mass)
        energy = 0.5 * np.sum(states[:, 3:]**2, axis=1) + g_eff * states[:, 2]
        self.assertTrue(np.all(np.diff(energy) < 0))
        np.testing.assert_allclose(states[:, 4] / states[:, 3], self.v0[1] / self.v0[0], atol=1e-12)

    def test_event_classification(self):
        cases = [
            ((20.0, 0.0, 0.0), 2.0, 'ground_first'),
            ((20.0, 1.0, 1.0), 2.0, 'ground_first'),
            ((11.0, 3.66, 5.5), 2.0, 'goal_outside'),
            ((30.0, 0.0, 6.0), 0.01, 'timeout'),
            ((30.0, 0.0, 12.0), 2.0, 'goal_outside'),
            ((30.0, 15.0, 6.0), 2.0, 'goal_outside'),
        ]
        for v0, t_max, event in cases:
            with self.subTest(v0=v0, t_max=t_max):
                flight = simulate(self.r0, Motion(v0, 0.0), replace(self.physics, density=0.0), Settings(t_max=t_max))
                self.assertEqual(flight.event, event)
                if event == 'ground_first':
                    self.assertAlmostEqual(flight.time, 2 * v0[2] / self.physics.gravity, places=10)
                    self.assertAlmostEqual(flight.states[-1, 2], self.physics.radius, places=12)

    def test_spin_decay_and_near_vertical_basis(self):
        motion = Motion(self.v0, 0.2, (1.0, -2.0, 3.0), tau=5.0)
        np.testing.assert_allclose(spin_at(0.25, motion), np.array(motion.omega0) * np.exp(-0.05))
        basis = spin_basis((0.0, 0.0, 1.0))
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(basis), 1.0)

    def test_refit_noise_free_forward_positions(self):
        times = np.array([0.0, 0.05, 0.10])
        motion = Motion(self.v0, 0.27)
        flight = simulate(self.r0, motion, self.physics)
        observations = flight.sample(times)[:, :3]
        fit = fit_from_positions(times, observations, self.physics, fit_spin=False)
        self.assertIsNotNone(fit.best)
        np.testing.assert_allclose(fit.best.motion.v0, self.v0, atol=1e-7)
        self.assertAlmostEqual(fit.best.motion.cd, motion.cd, places=7)
        self.assertLess(np.max(np.abs(fit.best.residuals)), 1e-9)


if __name__ == '__main__':
    unittest.main()
