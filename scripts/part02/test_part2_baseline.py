"""Independent numerical and supplied-benchmark checks for the two baselines."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp

from .part2_baseline import compute_baselines, position, TIMES, OBSERVATIONS


class BaselineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = compute_baselines()

    def test_case_1a_supplied_benchmark_and_interpolation(self):
        model = self.models[0]
        assert_allclose(model['v0_m_s'], [30.4, 9.4, 6.9], atol=1e-12)
        assert_allclose(model['a_m_s2'], [-16, -16, -12], atol=1e-11)
        assert_allclose(position(model, TIMES), OBSERVATIONS, atol=1e-12)
        assert_allclose(model['goal_position_m'], [11, 2.4948244142, 1.9203670565], rtol=0, atol=6e-11)
        self.assertAlmostEqual(model['goal_time_s'], 0.4050083612, places=10)

    def test_case_1b_supplied_benchmark_and_least_squares_stationarity(self):
        model = self.models[1]
        assert_allclose(model['v0_m_s'], [29.68, 8.68, 6.80145], atol=1e-12)
        residuals = position(model, TIMES[1:]) - OBSERVATIONS[1:]
        assert_allclose(residuals, [[-.016, -.016, -.00219], [.008, .008, .001095]], atol=1e-12)
        assert_allclose(TIMES[1:] @ residuals, [0, 0, 0], atol=1e-13)
        assert_allclose(model['goal_position_m'], [11, 3.2169811321, 1.9570064289], rtol=0, atol=6e-11)
        self.assertAlmostEqual(model['goal_time_s'], .3706199461, places=10)

    def test_first_forward_crossing_precedes_downward_contact(self):
        for model in self.models:
            with self.subTest(model=model['model']):
                t = model['goal_time_s']
                self.assertGreater(model['ground_contact_time_s'], t)
                self.assertGreater(model['v0_m_s'][0] + model['a_m_s2'][0] * t, 0)
                earlier = position(model, np.linspace(0, t - 1e-8, 500))
                self.assertTrue(np.all(earlier[:, 0] < 11))
                self.assertTrue(np.all(earlier[1:, 2] > .11))
                self.assertEqual(model['event'], 'inside_goal')

    def test_ode_events_independently_reproduce_analytic_crossings(self):
        for model in self.models:
            with self.subTest(model=model['model']):
                acceleration = np.array(model['a_m_s2'])
                def rhs(t, state):
                    return np.r_[state[3:], acceleration]
                def goal(t, state):
                    return state[0] - 11
                goal.direction = 1
                goal.terminal = True
                sol = solve_ivp(rhs, (0, 2), np.r_[OBSERVATIONS[0], model['v0_m_s']],
                                events=goal, rtol=1e-10, atol=1e-12, max_step=.002)
                self.assertTrue(sol.success)
                self.assertEqual(len(sol.t_events[0]), 1)
                self.assertAlmostEqual(sol.t_events[0][0], model['goal_time_s'], places=11)
                assert_allclose(sol.y_events[0][0, :3], model['goal_position_m'], rtol=0, atol=1e-10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
