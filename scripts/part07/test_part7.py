import unittest
from dataclasses import replace
from math import exp

import numpy as np

from .part7_bounce import ground_impact, simulate_bounce
from scripts.common.still_air import Motion, Physics, Settings, simulate


class BounceTests(unittest.TestCase):
    vacuum = Physics(density=0)
    origin = (0, 0, vacuum.radius)
    low = Motion((20, 1, 1), 0)

    def test_ballistic_bounce(self):
        shot = simulate_bounce(self.origin, self.low, self.vacuum, e_n=.6)
        first = 2 / 9.81
        self.assertEqual(shot.event, 'second_ground')
        self.assertAlmostEqual(shot.impact.time, first, delta=1e-11)
        self.assertAlmostEqual(shot.time, first + 1.2 / 9.81, delta=1e-11)
        np.testing.assert_allclose(shot.impact.v_minus, [20, 1, -1], atol=1e-11)
        np.testing.assert_allclose(shot.impact.v_plus, [20, 1, .6], atol=1e-11)
        self.assertAlmostEqual(shot.impact.energy_loss, .1376, delta=1e-11)
        np.testing.assert_allclose(shot.state[:3], [20 * shot.time, shot.time, .11], atol=1e-11)

    def test_goal_before_any_bounce_matches_flight(self):
        motion = Motion((30.4154, 9.4173, 6.9073), .37898, (13.9692, -5.4529, -58.1893))
        baseline = simulate(self.origin, motion)
        bounce = simulate_bounce(self.origin, motion, e_n=.8, mu_f=.5)
        self.assertEqual(bounce.event, 'goal_inside')
        self.assertIsNone(bounce.impact)
        self.assertAlmostEqual(bounce.time, baseline.time, delta=1e-10)
        np.testing.assert_allclose(bounce.state, baseline.states[-1], atol=1e-10, rtol=0)

    def test_post_bounce_goal_and_timeout_use_release_clock(self):
        low = replace(self.low, v0=(20, 1, 1.8))
        goal = simulate_bounce(self.origin, low, self.vacuum, e_n=.6)
        self.assertEqual(goal.event, 'goal_inside')
        self.assertAlmostEqual(goal.time, .55, delta=1e-11)
        np.testing.assert_allclose(goal.state[:3], [11, .55, .143356766565], atol=1e-10)
        late = simulate_bounce(self.origin, low, self.vacuum, Settings(t_max=.5), e_n=.6)
        self.assertEqual(late.event, 'timeout')
        self.assertIsNotNone(late.impact)
        self.assertEqual(late.time, .5)
        wide = simulate_bounce(self.origin, replace(low, v0=(20, 8, 1.8)), self.vacuum, e_n=.6)
        self.assertEqual(wide.event, 'goal_outside')

    def test_contact_at_release_and_zero_restitution_stop(self):
        for vz in (-1, 0):
            with self.subTest(vz=vz):
                initial = replace(self.low, v0=(20, 1, vz))
                shot = simulate_bounce(self.origin, initial, self.vacuum)
                self.assertEqual((shot.event, shot.time, shot.impact), ('initial_ground', 0., None))
        stopped = simulate_bounce(self.origin, self.low, self.vacuum, e_n=0)
        self.assertEqual(stopped.event, 'no_liftoff')
        self.assertEqual(len(stopped.segments), 1)
        self.assertEqual(stopped.state[5], 0)

    def test_ties_are_resolved_by_time_on_both_legs(self):
        first = 2 / 9.81
        second = first + 1.2 / 9.81
        for ground_time in (first, second):
            for offset in (0., -5e-9, 5e-9, -2e-8, 2e-8):
                with self.subTest(ground_time=ground_time, offset=offset):
                    settings = Settings(goal_x=20 * (ground_time + offset))
                    shot = simulate_bounce(self.origin, self.low, self.vacuum, settings)
                    if abs(offset) < 1e-8:
                        self.assertEqual(shot.event, 'goal_ground_tie')
                        self.assertAlmostEqual(shot.tie_gap, offset, delta=1e-11)
                    elif offset < 0:
                        self.assertEqual(shot.event, 'goal_inside')
                    elif ground_time == second:
                        self.assertEqual(shot.event, 'second_ground')
                    else:
                        self.assertEqual(shot.event, 'goal_inside')
                        self.assertIsNotNone(shot.impact)

    def test_smooth_contact_preserves_tangent_velocity_and_spin(self):
        shot = simulate_bounce(
            self.origin, Motion((20, 1, 1), 0, (4, -5, 7), tau=.5), self.vacuum)
        impact = shot.impact
        np.testing.assert_allclose(impact.omega_plus, np.array([4, -5, 7]) * exp(-impact.time / .5))
        np.testing.assert_array_equal(impact.v_plus[:2], impact.v_minus[:2])
        np.testing.assert_allclose(shot.segments[1].omega_end,
                                   np.array([4, -5, 7]) * exp(-shot.time / .5))

    def test_sliding_impact_can_transfer_spin_energy_to_translation(self):
        # Contact point moves backwards at 1.3 m/s even though v_x is positive.
        impact = ground_impact(0., (0, 0, .11, 2, 0, -1), (0, 30, 0), self.vacuum, .6, .1)
        np.testing.assert_allclose(impact.j_t, [.0688, 0, 0], atol=1e-13)
        np.testing.assert_allclose(impact.v_plus, [2.16, 0, .6], atol=1e-13)
        np.testing.assert_allclose(impact.omega_plus, [0, 27.81818181818, 0], atol=1e-11)
        self.assertGreater(impact.energy_loss, 0.)
        self.assertAlmostEqual(impact.energy_loss, .21328, delta=1e-11)

    def test_sticking_cap_stops_slip_without_reversing_it(self):
        impact = ground_impact(0., (0, 0, .11, 1, .5, -2), (0, 0, 4), self.vacuum, .6, .5)
        np.testing.assert_allclose(impact.j_t, [-.172, -.086, 0], atol=1e-13)
        np.testing.assert_allclose(impact.slip_plus, [0, 0, 0], atol=1e-13)
        self.assertEqual(impact.omega_plus[2], 4.)
        self.assertGreater(impact.energy_loss, 0.)


if __name__ == '__main__':
    unittest.main()
