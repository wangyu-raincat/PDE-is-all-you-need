"""Manufactured equivalence/guard checks; no benchmark tasks or scoring."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import resample

from active_study import forecast as accelerated
from pde_discovery import generic_engine as baseline
from prefix_study import forecast as original_prefix
from resolved_study import forecast as resolved


def terms(**coefficients):
    return [{'name': name, 'coefficient': value} for name, value in coefficients.items()]


class ActiveForecastTests(unittest.TestCase):
    def setUp(self):
        self.x = .27 + np.arange(32)*2*np.pi/32
        self.t = np.linspace(.1, 1.1, 11)
        self.u0 = .1 + .3*np.sin(self.x) + .2*np.cos(4*self.x)

    def test_raw_matches_existing_linear_etdrk4(self):
        cases = [{}, {'u': -.3}, {'u_xx': .09}, {'u_xxxx': -.003},
                 {'u_x': -.5, 'u_xx': .08}, {'u_xxx': .03, 'u_xx': .05},
                 {'u': -.2, 'u_xx': .08}, {'1': .04},
                 {'1': .04, 'u': -.2, 'u_x': -.4, 'u_xx': .08},
                 {'u_xx': -.04, 'u_xxxx': -.004}, {'1': .03, 'u': 1e-310}]
        for coefficients in cases:
            with self.subTest(coefficients=coefficients):
                actual = accelerated.raw_forecast(terms(**coefficients), self.x, self.t, self.u0)
                expected = resolved.forecast(terms(**coefficients), self.x, self.t, self.u0)
                np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)

    def test_nyquist_odd_and_even_and_first_prediction_are_preserved(self):
        for n in (32, 33):
            x = .27+np.arange(n)*2*np.pi/n
            u0 = .1+.3*np.cos((n//2)*(x-x[0]))+.2*np.sin(x)
            model = terms(u_x=-.4, u_xx=.02)
            expected = resolved.forecast(model, x, self.t, u0)
            actual = accelerated.raw_forecast(model, x, self.t, u0)
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)
            internal_x = x[0]+np.arange(4*n)*2*np.pi/(4*n)
            _, retained, _ = baseline._geometry(internal_x)
            projected = np.fft.ifft(np.fft.fft(resample(u0, 4*n))*retained).real[::4]
            np.testing.assert_array_equal(actual[0], projected)
            np.testing.assert_allclose(actual[0], u0, atol=5e-16, rtol=0)

    def test_constant_forcing_is_exact_and_not_applied_to_initial_frame(self):
        for reaction in (0., -1e-14, -.2, .2):
            elapsed = self.t-self.t[0]
            expected = .3*np.exp(reaction*elapsed)
            expected += .04*elapsed if reaction == 0 else .04*np.expm1(reaction*elapsed)/reaction
            actual = accelerated.raw_forecast(terms(**{'u':reaction, '1':.04}), self.x, self.t, np.full(32,.3))
            np.testing.assert_allclose(actual, np.repeat(expected[:,None],32,axis=1), atol=2e-15, rtol=2e-15)
            np.testing.assert_array_equal(actual[0], np.full(32,.3))

    def test_zero_unforced_state_does_not_overflow_unused_exponentials(self):
        model = terms(u=1000.)
        expected = resolved.forecast(model, self.x, self.t, np.zeros(32))
        actual = accelerated.raw_forecast(model, self.x, self.t, np.zeros(32))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, np.zeros((len(self.t),32)))

    def test_prefix_prediction_diagnostics_and_full_horizon_match(self):
        pt = np.linspace(0,.1,4)
        rng = np.random.default_rng(881002)
        pu = np.array([np.exp(-.2*time)*self.u0 for time in pt]) + .002*rng.normal(size=(4,32))
        for model in (terms(u=-.2), terms(**{'u_x':-.4,'u_xx':.07,'1':.01})):
            actual, diagnostic = accelerated.forecast_with_diagnostics(model, self.x, self.t, pt, pu)
            expected, old_diagnostic = original_prefix.forecast_with_diagnostics(model, self.x, self.t, pt, pu)
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)
            self.assertEqual(set(diagnostic), set(old_diagnostic))
            for name in diagnostic:
                if name == 'last_observation_relative_change':
                    self.assertAlmostEqual(diagnostic[name], old_diagnostic[name], places=12)
                else:
                    self.assertEqual(diagnostic[name], old_diagnostic[name])
            self.assertEqual(diagnostic['forecast_duration'], 1.)
            self.assertFalse(diagnostic['future_observations_used'])

    def test_repeated_terms_accumulate_and_exact_nonlinear_cancellation_is_linear(self):
        model = terms(u=-.1) + terms(u=-.2) + terms(**{'u^2':2.}) + terms(**{'u^2':-2.})
        expected = resolved.forecast(model, self.x, self.t, self.u0)
        with patch.object(resolved, 'forecast', side_effect=AssertionError('linear path must not time step')):
            actual = accelerated.raw_forecast(model, self.x, self.t, self.u0)
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-13)

    def test_nonlinear_candidates_use_unmodified_paths(self):
        for nonlinear in ('u^2','u^3','u*u_x'):
            model = terms(**{nonlinear:.01, 'u_xx':.08})
            sentinel = object()
            with patch.object(resolved,'forecast',return_value=sentinel) as mocked:
                self.assertIs(accelerated.raw_forecast(model,self.x,self.t,self.u0),sentinel)
                mocked.assert_called_once()
            with patch.object(original_prefix,'forecast_with_diagnostics',return_value=(sentinel,{})) as mocked:
                value = accelerated.forecast(model,self.x,self.t,[0.,.1],np.stack([self.u0]*2))
                self.assertIs(value,sentinel)
                mocked.assert_called_once()

    def test_coefficient_dissipativity_and_growth_rejections_match(self):
        invalid_models = [terms(unknown=.1), terms(u=np.inf), terms(u=np.nan),
                          terms(u_xx=-.1), terms(u_xxxx=.001), terms(u=1000.),
                          terms(**{'1':1e6}), terms(u=1e308)+terms(u=1e308)]
        for model in invalid_models:
            with self.subTest(model=model), np.errstate(all='ignore'):
                with self.assertRaises(ValueError):
                    resolved.forecast(model,self.x,self.t,self.u0)
                with self.assertRaises(ValueError):
                    accelerated.raw_forecast(model,self.x,self.t,self.u0)
        with self.assertRaisesRegex(ValueError,'linear growth'):
            accelerated.raw_forecast(terms(u=20000.),self.x,[0.,1.],self.u0)

    def test_coordinates_state_and_computational_budgets_remain_guarded(self):
        cases = [(self.x.astype(complex), self.t, self.u0),
                 (self.x, self.t.astype(complex), self.u0),
                 (self.x, self.t, self.u0.astype(complex)),
                 (self.x, self.t, self.u0[:-1]),
                 (self.x, self.t, np.full(32,np.nan)),
                 (self.x, [0.,1.,1.5], self.u0),
                 (self.x, np.linspace(0,1,627), self.u0),
                 (np.arange(513)*2*np.pi/513, self.t, np.zeros(513))]
        for x,t,u0 in cases:
            with self.subTest(xn=len(x),tn=len(t)):
                with self.assertRaises(ValueError): accelerated.raw_forecast([],x,t,u0)
        with self.assertRaises(ValueError):
            accelerated.forecast([],self.x,self.t,[0.,.2],np.stack([self.u0]*2))
        with self.assertRaises(ValueError):
            accelerated.forecast([],self.x,self.t,np.linspace(0,.1,33),np.stack([self.u0]*33))


if __name__ == '__main__':
    unittest.main()
