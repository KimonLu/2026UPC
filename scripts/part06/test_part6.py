import unittest

import numpy as np

from scripts.part06.part6_coverage import coverage_draws, region_result, wilson_interval
from scripts.part06.part6_drag_curves import drag_coefficient, variable_drag_flight
from scripts.part06.part6_experiments import build_grid, event_metrics, run_v1


class CoverageTests(unittest.TestCase):
    def test_trial_seeds_are_paired_and_nested(self):
        outer, inner = coverage_draws(20261107, 17, 200)
        same_outer, longer = coverage_draws(20261107, 17, 400)
        np.testing.assert_array_equal(outer, same_outer)
        np.testing.assert_array_equal(inner, longer[:200])
        self.assertFalse(np.array_equal(outer, inner[0]))
        self.assertFalse(np.array_equal(outer, coverage_draws(20261107, 18, 200)[0]))

    def test_joint_membership_and_region_failure(self):
        points = np.array([[0.4, np.cos(a), np.sin(a)] for a in np.arange(100)])
        prediction = dict(goal=points, event=np.full(len(points), 'goal_inside'))
        self.assertTrue(region_result(prediction, [0, 0])['covered'])
        self.assertFalse(region_result(prediction, [3, 3])['covered'])
        points[:, 2] = 0
        self.assertIsNone(region_result(prediction, [0, 0])['covered'])

    def test_wilson_interval(self):
        self.assertEqual(wilson_interval(0, 200)[0], 0)
        self.assertEqual(wilson_interval(200, 200)[1], 1)
        np.testing.assert_allclose(wilson_interval(190, 200), [0.91042, 0.97262],
                                   atol=1e-5)


class PredictionTests(unittest.TestCase):
    def test_full_factorial_grid(self):
        grid = build_grid()
        self.assertEqual(len(grid), 81)
        self.assertEqual(len(set(grid)), 81)
        self.assertIn((30, 10, 0, -40), grid)

    def test_conditional_error_keeps_unpaired_events_in_denominator(self):
        rows = [
            dict(truth_event='goal_inside', prediction_event='goal_outside',
                 error_m=0.2, time_error_s=-0.01, obs_eligible=True),
            dict(truth_event='goal_inside', prediction_event='ground_first',
                 error_m=None, time_error_s=None, obs_eligible=True),
            dict(truth_event='ground_first', prediction_event='not_observed',
                 error_m=None, time_error_s=None, obs_eligible=False),
        ]
        result = event_metrics(rows)
        self.assertEqual(result['both_cross'], 1)
        self.assertEqual(result['n_eligible'], 2)
        self.assertEqual(result['both_cross_fraction'], 0.5)
        self.assertEqual(result['confusion']['goal_inside']['ground_first'], 1)
        self.assertEqual(result['position_m']['rmse'], 0.2)

    def test_numerics_and_events(self):
        checks = run_v1()
        self.assertTrue(checks)
        self.assertTrue(all(check['passed'] for check in checks),
                        [check for check in checks if not check['passed']])

    def test_published_drag_curve_and_validity_domain(self):
        self.assertAlmostEqual(drag_coefficient('Jabulani', 17.3015), .163643 + .306765/2)
        with self.assertRaises(ValueError):
            drag_coefficient('Jabulani', 40.)
        _, speed = variable_drag_flight('Jabulani', (20, 10, 0, 0))
        self.assertGreaterEqual(min(speed), 7.)
        self.assertLessEqual(max(speed), 35.)


if __name__ == '__main__':
    unittest.main()
