"""Checks catch event-order, wind/Magnus scale and inverse-solver errors."""
import unittest
import numpy as np
try:
    from scripts.part05 import part5_solver as solver
except ImportError:
    solver = None
from scripts.common.still_air import Physics, Settings, Motion, simulate
from scripts.common.still_air import position_residuals, SCALES
from scipy.optimize import least_squares

OBS = np.array([[0., 0., .11], [1.5, .45, .44], [2.96, .86, .74]])
TIMES = np.array([0., .05, .1])


class Part5Checks(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(solver, 'Part 5 numerical solver is not implemented')

    def test_zero_noise_recovers_published_wind_prediction(self):
        fit = solver.fit_batch(OBS[None], TIMES[None], (2., 0., 0.))
        self.assertTrue(fit['success'][0])
        prediction = solver.predict_batch(fit['parameters'], fit['basis'], (2., 0., 0.))
        np.testing.assert_allclose(prediction['goal'][0], [.400137985524, 2.61033646864, 1.96477091361], atol=2e-7, rtol=0)
        self.assertEqual(prediction['event'][0], 'goal_inside')

    def test_unequal_times_recover_known_quadratic(self):
        t = np.array([0., .0493, .1008])
        positions = OBS[:1] + t[:, None]*[30.,9.,7.] + .5*t[:,None]**2*[-10.,-12.,-9.]
        result = solver.baseline(positions[None], t[None], 'B-CA')
        np.testing.assert_allclose(result['velocity'][0], [30.,9.,7.], atol=1e-11)
        np.testing.assert_allclose(result['acceleration'][0], [-10.,-12.,-9.], atol=1e-10)

    def test_ground_contact_precedes_hypothetical_goal(self):
        t = np.array([0., .05, .1])
        positions = OBS[:1]+t[:,None]*[20.,1.,1.]+.5*t[:,None]**2*[0.,0.,-9.81]
        result = solver.baseline(positions[None], t[None], 'B-G')
        self.assertEqual(result['event'][0], 'ground_first')
        self.assertTrue(np.isnan(result['goal'][0]).all())

    def test_event_interpolation_agrees_with_independent_dop853(self):
        params = np.array([[30.,9.,7.,.3,-12.,-65.]])
        basis = np.eye(3)[None]
        predicted = solver.predict_batch(params, basis, (2.,0.,0.))
        reference = simulate(OBS[0], Motion(params[0,:3], .3, [0.,-12.,-65.]), Physics(wind=(2.,0.,0.)), Settings())
        np.testing.assert_allclose(predicted['goal'][0], [reference.time,*reference.states[-1,1:3]], atol=2e-7, rtol=0)

    def test_rodrigues_rotation_preserves_magnitude(self):
        omega = np.array([13.,-5.,-60.])
        rotated = solver.rotate_spin(omega, [0.,0.,1.], .5, .4)
        self.assertAlmostEqual(np.linalg.norm(rotated), np.linalg.norm(omega), places=12)
        np.testing.assert_allclose(solver.rotate_spin([0.,0.,60.],[0.,0.,1.],.5,.4),[0.,0.,60.])

    def test_zero_gust_recovers_reference(self):
        p = np.array([30.,9.,7.,.3,-12.,-65.])
        flight = solver.general_flight(p, np.eye(3), Physics(wind=(2.,0.,0.)), gust=(.15,.1,[0.,0.,0.]))
        reference = simulate(OBS[0],Motion(p[:3],p[3],[0.,*p[4:]]),Physics(wind=(2.,0.,0.)))
        np.testing.assert_allclose(flight['goal'],[reference.time,*reference.states[-1,1:3]],atol=1e-8,rtol=0)

    def test_noisy_boundary_refit_matches_independent_dop853_optimum(self):
        rng=np.random.default_rng(20261107)
        observations=np.broadcast_to(OBS,(40,3,3)).copy()
        observations[:,1:]+=rng.normal(0,.015,(40,2,3))
        times=np.broadcast_to(TIMES,(40,3)).copy()
        batch=solver.fit_batch(observations,times,(2.,0.,0.))
        index=int(np.flatnonzero(batch['active'])[0])
        basis=batch['basis'][index];p=batch['parameters'][index]
        physics=Physics(wind=(2.,0.,0.))
        def residual(x):
            physical=x*SCALES
            omega=basis@np.r_[0.,physical[4:]]
            return position_residuals(TIMES,observations[index],Motion(physical[:3],physical[3],omega),physics,Settings()).ravel()
        refined=least_squares(residual,p/SCALES,bounds=(solver.LOW/SCALES,solver.HIGH/SCALES),ftol=1e-12,xtol=1e-12,gtol=1e-13)
        self.assertTrue(batch['success'][index])
        self.assertLess(np.sum(batch['residual'][index]**2)-np.sum(refined.fun**2),1e-10)
        np.testing.assert_allclose(p,refined.x*SCALES,atol=3e-4,rtol=0)

    def test_invalid_observation_order_is_not_silently_fitted(self):
        with self.assertRaises(ValueError):
            solver.fit_batch(OBS[None],np.array([[0.,.1,.05]]),(2.,0.,0.))

    def test_common_one_dimensional_error_has_zero_area_region(self):
        from scripts.part05.run_part5 import summarize
        u=np.linspace(-1.,1.,200)
        prediction={'event':np.full(200,'goal_inside'), 'goal':np.column_stack([.4+.01*u,2.6+.1*u,1.96+.2*u])}
        summary=summarize(prediction)
        self.assertEqual(summary['ellipse_area_m2'],0.)
        self.assertEqual(summary['region_dimension'],1)

    def test_pilot_ensemble_cannot_generate_published_4000_sample_claims(self):
        import scripts.part05.run_part5 as run_part5
        self.assertTrue(hasattr(run_part5,'require_published_ensemble'), 'Missing guard against publishing pilot statistics with fixed prose')
        with self.assertRaises(ValueError):
            run_part5.require_published_ensemble(200)
        run_part5.require_published_ensemble(4000)


if __name__ == '__main__':
    unittest.main()
