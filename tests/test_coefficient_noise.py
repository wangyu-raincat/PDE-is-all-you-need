"""Manufactured noise-moment checks; no study fields or true fit parameters."""
from copy import deepcopy
import unittest

import numpy as np
from scipy.signal import resample

from coefficient_study import noise
from weak_study import engine as original


def field(n=64, nt=65, sigma=.03, seed=7401):
    x = np.arange(n)*2*np.pi/n
    t = np.linspace(0., 1., nt)
    u = .35+.4*np.exp(-t[:, None])*np.cos(2*x)[None, :]+.1*np.sin(3*x)[None, :]
    if sigma:
        u = u+np.random.default_rng(seed).normal(0., sigma, u.shape)
    return {'x': x, 't': t, 'u': u, 'boundary': 'periodic'}


class NoiseMomentTests(unittest.TestCase):
    def test_interpolated_variance_matches_independent_linear_operator(self):
        # Resampling independent basis vectors exposes every interpolation
        # weight, including the even-grid real Nyquist degree of freedom.
        for n in (16, 17, 32, 63, 128):
            weights = resample(np.eye(n), 2*n, axis=1)
            expected = np.sum(weights**2, axis=0)
            np.testing.assert_allclose(noise.interpolation_variance_factors(n), expected,
                                       atol=2e-15, rtol=2e-15)
            np.testing.assert_allclose(expected[::2], 1., atol=2e-15)
            if n % 2 == 0:
                self.assertLess(float(expected[1]), 1.)

    def test_zero_variance_spectra_exactly_match_original_for_nyquist_and_odd_grids(self):
        rng = np.random.default_rng(932)
        for n in (16, 17, 64):
            u = rng.normal(size=(19, n))
            for a, b in zip(noise._spectra(u, 0.), original._spectra(u)):
                np.testing.assert_array_equal(a, b)

    def test_monte_carlo_constant_and_manufactured_moments_remove_gaussian_bias(self):
        # True sigma enters only this isolated mathematical primitive test.
        # The production weak_block API has no oracle-variance argument.
        rng = np.random.default_rng(80017)
        n = 16
        x = np.arange(n)*2*np.pi/n
        sigma = .3
        for truth in (np.full(n, .4), .35+.2*np.cos(x)-.1*np.sin(2*x)):
            observed = truth+rng.normal(0., sigma, (40000, n))
            raw = original._spectra(observed)
            corrected = noise._spectra(observed, sigma**2)
            exact = original._spectra(truth[None, :])
            for order in (1, 2):
                raw_error = np.linalg.norm(raw[order].mean(axis=0)-exact[order][0])
                error = np.linalg.norm(corrected[order].mean(axis=0)-exact[order][0])
                self.assertLess(error, .1*raw_error)
                self.assertLess(error, .003)
            np.testing.assert_array_equal(raw[0], corrected[0])

    def test_pooled_scale_estimation_calibrated_on_white_noise_with_near_zero_mean(self):
        rng = np.random.default_rng(411)
        n = 128
        x = np.arange(n)*2*np.pi/n
        for mean, sigma in ((0., .07), (2., .03), (0., 1e-6)):
            u = mean+rng.normal(0., sigma, (512, n))
            variance, d = noise.estimate_noise_variance(x, u)
            self.assertLess(abs(np.sqrt(variance)/sigma-1.), .035)
            self.assertFalse(d['observations_filtered'])
            self.assertFalse(d['true_noise_used'])
            self.assertFalse(d['independent_physical_trials_claimed'])

    def test_scale_is_equivariant_to_field_units_offset_sign_and_circular_shift(self):
        d = field()
        variance, _ = noise.estimate_noise_variance(d['x'], d['u'])
        shifted, _ = noise.estimate_noise_variance(d['x'], np.roll(-3*d['u']+8., 9, axis=1))
        self.assertAlmostEqual(shifted/variance, 9., places=10)

    def test_disabled_correction_is_original_block_exactly(self):
        d = field()
        for variant in original.VARIANTS:
            a = noise.weak_block(d, variant, noise_correction=False)
            b = original.weak_block(d, variant)
            np.testing.assert_array_equal(a['X'], b['X'])
            np.testing.assert_array_equal(a['y'], b['y'])
            self.assertEqual(a['metadata'], b['metadata'])

    def test_zero_estimated_noise_constant_field_is_original_block_exactly(self):
        d = field(n=32, sigma=0.)
        d['u'] = np.tile((.5+.125*np.sin(d['t']))[:, None], (1, len(d['x'])))
        variance, _ = noise.estimate_noise_variance(d['x'], d['u'])
        self.assertEqual(variance, 0.)
        for variant in original.VARIANTS:
            a = noise.weak_block(d, variant)
            b = original.weak_block(d, variant)
            np.testing.assert_array_equal(a['X'], b['X'])
            np.testing.assert_array_equal(a['y'], b['y'])

    def test_correction_preserves_linear_columns_target_and_geometry(self):
        d = field()
        for variant in original.VARIANTS:
            a = noise.weak_block(d, variant)
            b = original.weak_block(d, variant)
            np.testing.assert_array_equal(a['X'][:, [0, 1, 4, 5, 6, 7]], b['X'][:, [0, 1, 4, 5, 6, 7]])
            np.testing.assert_array_equal(a['y'], b['y'])
            self.assertGreater(np.linalg.norm(a['X'][:, [2, 3]]-b['X'][:, [2, 3]]), 1e-4)
            for key in ('window_intervals', 'integration_windows', 'max_spatial_mode', 'time_test_degree'):
                self.assertEqual(a['metadata'][key], b['metadata'][key])
            self.assertFalse(a['metadata']['all_errors_in_variables_bias_removed'])
            self.assertFalse(a['metadata']['interpolated_noise_points_independent'])

    def test_no_input_mutation_and_no_public_oracle_argument(self):
        d = field()
        before = deepcopy(d)
        noise.weak_block(d)
        for name in ('x', 't', 'u'):
            np.testing.assert_array_equal(d[name], before[name])
        for key in ('true_sigma', 'noise_sigma', 'variance', 'family', 'reference_terms'):
            with self.assertRaises(TypeError):
                noise.weak_block(d, **{key: .01})

    def test_nonperiodic_nonfinite_complex_and_over_budget_inputs_rejected(self):
        cases = []
        for key, value in (('boundary', 'dirichlet'), ('u', np.full((65, 64), np.nan)),
                           ('u', np.ones((65, 64), complex)), ('u', np.full((65, 64), 1e7))):
            d = field()
            d[key] = value
            cases.append(d)
        cases.append(field(n=513))
        for d in cases:
            with self.assertRaises(ValueError):
                noise.weak_block(d)
        with self.assertRaises(ValueError):
            noise.weak_block(field(), noise_correction=1)

    def test_noiseless_upper_band_signal_exposes_estimator_limitation(self):
        d = field(sigma=0.)
        signal = sum(.05*np.cos(k*d['x']+.17*k) for k in range(16, 32))
        d['u'] = np.tile(signal, (len(d['t']), 1))
        variance, diagnostics = noise.estimate_noise_variance(d['x'], d['u'])
        self.assertGreater(variance, .01)
        self.assertTrue(diagnostics['upper_band_signal_can_bias_scale'])
        corrected = noise.weak_block(d)
        uncorrected = original.weak_block(d)
        self.assertGreater(np.linalg.norm(corrected['X']-uncorrected['X']), .01)
        self.assertFalse(noise.PROTOCOL['plug_in_estimated_variance_is_exactly_unbiased'])


if __name__ == '__main__':
    unittest.main()
