"""Reference solver checks using manufactured fields and calibration-only roles."""
import ast
import inspect
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import resample

from weak_study import generators as g


class WeakGeneratorTests(unittest.TestCase):
    def test_zero_symbol_contour_limits(self):
        h = .017
        e, e2, q, f1, f2, f3 = g._contour_coefficients(np.zeros(4), h)
        np.testing.assert_allclose(e, 1, atol=1e-14)
        np.testing.assert_allclose(e2, 1, atol=1e-14)
        for actual, target in [(q, h/2), (f1, h/6), (f2, h/6), (f3, h/6)]:
            np.testing.assert_allclose(actual, target, rtol=2e-14, atol=2e-16)

    def test_linear_solution_with_reaction_and_irregular_times(self):
        x = np.arange(32)*2*np.pi/32
        times = np.array([2., 2.011, 2.047, 2.083])
        elapsed = times-times[0]
        initial = .2+.7*np.sin(x)+.15*np.cos(3*x)
        expected = (.2*np.exp(.3*elapsed)[:, None]
                    +.7*np.exp((.3-.12-.003)*elapsed)[:, None]*np.sin(x)
                    +.15*np.exp((.3-.12*9-.003*81)*elapsed)[:, None]*np.cos(3*x))
        actual = g._integrate(initial, 2*np.pi, times, {0: .3, 2: .12, 4: -.003}, 0, 0, .006)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-13)

    def test_constant_allen_cahn_matches_closed_form(self):
        t = np.linspace(0, .4, 21)
        u0 = .2
        expected = u0/np.sqrt(u0*u0+(1-u0*u0)*np.exp(-10*t))
        actual = g._integrate(np.full(32, u0), 2., t, {0: 5., 2: .0001}, 0, -5, .001)
        np.testing.assert_allclose(actual, np.broadcast_to(expected[:, None], actual.shape), rtol=2e-10, atol=2e-12)

    def test_cubic_high_modes_match_independent_dealiased_dop853(self):
        n = 32
        x = np.arange(n)*2*np.pi/n
        # Cubing mode 9 aliases to mode 5 on an unpadded 32-point grid.
        initial = .1+.4*np.sin(9*x+.2)+.2*np.cos(2*x-.3)
        modes = np.fft.fftfreq(n)*n
        retained = np.abs(modes) <= n//3
        times = np.linspace(0, .08, 9)
        def rhs(_t, values):
            spectrum = np.fft.fft(values)*retained
            linear = np.fft.ifft((5-.0001*modes*modes)*spectrum).real
            field = np.fft.ifft(spectrum).real
            # Independent scipy 3N resampling rather than generator's 2N indexing.
            reaction = resample(-5*resample(field, 3*n)**3, n)
            reaction = np.fft.ifft(np.fft.fft(reaction)*retained).real
            return linear+reaction
        reference = solve_ivp(rhs, (0, times[-1]), initial, t_eval=times,
                              method='DOP853', rtol=2e-12, atol=2e-13, max_step=.0005)
        self.assertTrue(reference.success)
        actual = g._integrate(initial, 2*np.pi, times, {0: 5., 2: .0001}, 0, -5, .0005)
        np.testing.assert_allclose(actual, reference.y.T, rtol=2e-9, atol=2e-10)

    def test_transport_sign_matches_nonconservative_dop853(self):
        n = 32
        x = np.arange(n)*2*np.pi/n
        initial = .1+.7*np.sin(x+.2)+.2*np.cos(2*x-.3)+.1*np.sin(3*x)
        modes = np.fft.fftfreq(n)*n
        retained = np.abs(modes) <= n//3
        times = np.linspace(0, .16, 17)
        for linear in ({2: .03}, {2: -.06, 4: -.004}):
            def rhs(_t, values):
                spectrum = np.fft.fft(values)*retained
                derivative = np.fft.ifft(1j*modes*spectrum).real
                nonlinear = resample(-resample(values, 3*n)*resample(derivative, 3*n), n)
                result = np.fft.ifft(np.fft.fft(nonlinear)*retained).real
                for order, coefficient in linear.items():
                    result += coefficient*np.fft.ifft((1j*modes)**order*spectrum).real
                return result
            reference = solve_ivp(rhs, (0, times[-1]), initial, t_eval=times,
                                  method='DOP853', rtol=2e-12, atol=2e-13, max_step=.0005)
            self.assertTrue(reference.success)
            actual = g._integrate(initial, 2*np.pi, times, linear, -1, 0, .0005)
            np.testing.assert_allclose(actual, reference.y.T, rtol=2e-9, atol=2e-10)

    def test_role_and_family_random_streams_disjoint_without_generating_fields(self):
        keys = [(g.ROLE_SEEDS[role]+i, g.FAMILY_IDS[family]) for family in g.FAMILIES
                for role, count in g.ROLE_COUNTS.items() for i in range(count)]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(g.ROLE_COUNTS['final'], 3)
        self.assertTrue(all(seed >= 66000 for seed, _ in keys))
        for family in g.FAMILIES:
            first = g._initial_parameters(family, 'calibration', 0)
            second = g._initial_parameters(family, 'calibration', 0)
            np.testing.assert_array_equal(first[2], second[2])
            np.testing.assert_array_equal(first[3], second[3])

    def test_observed_grid_is_direct_subsampling_and_same_after_refinement(self):
        # No physical calibration or formal trajectory is generated here.
        def manufactured(initial, period, times, linear, transport, cubic, max_dt):
            return np.broadcast_to(initial, (len(times), len(initial))).copy()
        with patch.object(g, '_integrate', side_effect=manufactured):
            for family in g.FAMILIES:
                coarse = g.generate(family, 'calibration')
                fine = g.generate(family, 'calibration', refined=True)
                np.testing.assert_array_equal(coarse['x'], fine['x'])
                np.testing.assert_array_equal(coarse['t'], fine['t'])
                np.testing.assert_array_equal(coarse['u'], fine['u'])
                self.assertEqual(coarse['u'].shape, (101, 256 if family == 'allen_cahn' else 128))
                self.assertFalse(coarse['metadata']['is_measured_data'])

    def test_invalid_inputs_rejected(self):
        for args in [('bad', 'calibration', 0), (g.FAMILIES[0], 'bad', 0),
                     (g.FAMILIES[0], 'calibration', 1), (g.FAMILIES[0], 'calibration', True),
                     (g.FAMILIES[0], 'final', 3)]:
            with self.assertRaises(ValueError):
                g.generate(*args)
        with self.assertRaises(ValueError):
            g.generate('allen_cahn', 'calibration', refined=1)
        for u, t, linear, dt in [(np.ones(15), [0, 1], {2: .1}, .01),
                (np.ones(16), [0, 0], {2: .1}, .01), (np.ones(16), [0, 1], {3: 1}, .01),
                (np.ones(16), [0, 1], {2: .1}, 0), (np.ones(17), [0, 1], {2: .1}, .01),
                (np.ones(18), [0, 1], {2: .1}, .01)]:
            with self.assertRaises(ValueError):
                g._integrate(u, 2*np.pi, t, linear, -1, 0, dt)

    def test_generator_cannot_import_discovery_or_predictor(self):
        source = inspect.getsource(g)
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
        self.assertFalse(any(name.startswith(('pde_discovery', 'l1_study', 'weak_study.engine')) for name in imports))
        generate_tree = ast.parse(inspect.getsource(g.generate))
        calls = [n.func.id for n in ast.walk(generate_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        self.assertNotIn('reference_terms', calls)

    def test_calibration_uses_only_reserved_roles_and_keeps_failures(self):
        calls = []
        def fixture(family, role, index, refined):
            calls.append((family, role, index, refined))
            field = np.ones((3, 4))
            if family == 'low_viscosity_burgers' and refined:
                field[1:] += .01
            if family == 'ks_unstable':
                raise FloatingPointError('reference failure')
            return {'x': np.arange(4), 't': np.arange(3), 'u': field, 'metadata': {}}
        with patch.object(g, 'generate', side_effect=fixture):
            result = g.calibration()
        self.assertFalse(result['all_passed'])
        self.assertEqual([r['status'] for r in result['records']], ['passed', 'source_quality_failure', 'source_quality_failure'])
        self.assertTrue(all(role == 'calibration' and i == 0 for _, role, i, _ in calls))
        self.assertFalse(result['formal_training_development_final_generated'])


if __name__ == '__main__':
    unittest.main()
