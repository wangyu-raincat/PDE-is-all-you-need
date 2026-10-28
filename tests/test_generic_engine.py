"""Independent manufactured checks for general scalar PDE discovery.

These arrays are test fixtures, not public-data or experimental evidence.
Reference trajectories use analytic solutions or SciPy DOP853; they never call
the candidate engine's forecast function to generate observations.
"""
from copy import deepcopy
from functools import lru_cache
import inspect
import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import resample

from pde_discovery import generic_engine as generic


def terms(coefficients):
    return [{"name": name, "coefficient": float(value)}
            for name, value in coefficients.items()]


def independent_reference(coefficients, x, t, initial):
    """Independent method-of-lines reference, intentionally using DOP853."""
    x, t = np.asarray(x), np.asarray(t)
    wave = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    kept = np.abs(np.fft.fftfreq(len(x)) * len(x)) <= len(x) // 3

    def rhs(_time, values):
        spectrum = np.fft.fft(values) * kept
        u = np.fft.ifft(spectrum).real
        derivatives = {order: np.fft.ifft((1j * wave) ** order * spectrum).real
                       for order in (1, 2, 3, 4)}
        # Independent 3N interpolation and pointwise differentiation product;
        # the candidate solver uses a different padded-grid implementation.
        fine_u = resample(u, 3 * len(x))
        fine_ux = resample(derivatives[1], 3 * len(x))
        nonlinear = (coefficients.get("1", 0.)
                     + coefficients.get("u^2", 0.) * fine_u**2
                     + coefficients.get("u^3", 0.) * fine_u**3
                     + coefficients.get("u*u_x", 0.) * fine_u * fine_ux)
        nonlinear = resample(nonlinear, len(x))
        nonlinear = np.fft.ifft(np.fft.fft(nonlinear) * kept).real
        linear = coefficients.get("u", 0.) * u
        for name, order in (("u_x", 1), ("u_xx", 2), ("u_xxx", 3), ("u_xxxx", 4)):
            linear = linear + coefficients.get(name, 0.) * derivatives[order]
        return linear + nonlinear

    solution = solve_ivp(rhs, (float(t[0]), float(t[-1])), initial,
                         t_eval=t, method="DOP853", rtol=2e-11, atol=2e-12,
                         max_step=(t[-1] - t[0]) / 250)
    if not solution.success:
        raise AssertionError(solution.message)
    return solution.y.T


@lru_cache(maxsize=3)
def canonical_fixture(name):
    coefficients = {
        "burgers": {"u_xx": 0.12, "u*u_x": -0.7},
        "kdv": {"u_xxx": -0.07, "u*u_x": -0.8},
        "ks": {"u_xx": -0.06, "u_xxxx": -0.004, "u*u_x": -0.8},
    }[name]
    x = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    t = np.linspace(0, 0.4, 81)
    initial = (0.15 + 1.1 * np.sin(x + .12) + .4 * np.cos(2 * x)
               + .25 * np.sin(3 * x - .3) + .15 * np.cos(4 * x + .2))
    observed = independent_reference(coefficients, x, t, initial)
    return {"x": x.tolist(), "t": t.tolist(), "u": observed.tolist(),
            "boundary": "periodic"}, coefficients


class GenericEngineNumericalTests(unittest.TestCase):
    def test_linear_six_term_solution_matches_exact_fourier_evolution(self):
        x = np.linspace(-1, 2 * np.pi - 1, 48, endpoint=False)
        t = 4.2 + np.linspace(0, .7, 29)
        u0 = .3 + np.sin(x) + .2 * np.cos(2 * x - .2) + .1 * np.sin(4 * x)
        coefficients = {"1": .11, "u": -.17, "u_x": .23, "u_xx": .08,
                        "u_xxx": -.09, "u_xxxx": -.003}
        k = 2 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
        eigenvalues = (coefficients["u"] + coefficients["u_x"] * 1j*k
                       + coefficients["u_xx"] * (1j*k)**2
                       + coefficients["u_xxx"] * (1j*k)**3
                       + coefficients["u_xxxx"] * (1j*k)**4)
        elapsed = t - t[0]
        spectrum = np.exp(np.outer(elapsed, eigenvalues)) * np.fft.fft(u0)
        spectrum[:, 0] += len(x) * coefficients["1"] * np.expm1(coefficients["u"] * elapsed) / coefficients["u"]
        expected = np.fft.ifft(spectrum, axis=1).real
        actual = generic.forecast(terms(coefficients), x, t, u0)
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(generic.forecast(terms(coefficients), x, elapsed, u0), actual,
                                   rtol=2e-10, atol=2e-10)

    def test_constant_forcing_zero_linear_symbol_uses_finite_phi_limit(self):
        x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        t = np.linspace(0, 1.0, 21)
        initial = .2 + .1 * np.cos(x)
        actual = generic.forecast(terms({"1": .13}), x, t, initial)
        expected = initial[None, :] + .13 * t[:, None]
        np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)

    def test_quadratic_reaction_matches_closed_form_logistic_solution(self):
        x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        t = np.linspace(0, 2., 41)
        initial = np.full(len(x), .2)
        actual = generic.forecast(terms({"u": .8, "u^2": -.8}), x, t, initial)
        scalar = 1 / (1 + (1 / .2 - 1) * np.exp(-.8 * t))
        np.testing.assert_allclose(actual, np.repeat(scalar[:, None], len(x), axis=1),
                                   rtol=3e-8, atol=3e-9)

    def test_cubic_source_reaction_matches_independent_scalar_ode(self):
        x = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        t = np.linspace(0, 1.2, 31)
        initial = np.full(len(x), .17)
        coefficient = {"1": .1, "u": .3, "u^2": -.4, "u^3": -.2}
        reference = solve_ivp(lambda _t, y: .1 + .3*y - .4*y**2 - .2*y**3,
                              (t[0], t[-1]), [.17], t_eval=t, method="DOP853",
                              rtol=1e-12, atol=1e-13).y[0]
        actual = generic.forecast(terms(coefficient), x, t, initial)
        np.testing.assert_allclose(actual, np.repeat(reference[:, None], len(x), axis=1),
                                   rtol=3e-8, atol=3e-9)

    def test_three_nonlinear_equations_match_independent_ode_reference(self):
        for name in ("burgers", "kdv", "ks"):
            with self.subTest(equation=name):
                data, coefficients = canonical_fixture(name)
                expected = np.asarray(data["u"])
                actual = generic.forecast(terms(coefficients), data["x"], data["t"], expected[0])
                np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)

    def test_stabilized_negative_second_derivative_is_valid_but_backward_parabolic_is_not(self):
        x = np.linspace(0, 2*np.pi, 32, endpoint=False)
        t = np.linspace(0, .1, 17)
        initial = .2 * np.sin(x) + .1*np.cos(2*x)
        stabilized = generic.forecast(terms({"u_xx": -.06, "u_xxxx": -.004}), x, t, initial)
        self.assertTrue(np.isfinite(stabilized).all())
        for coefficients in ({"u_xx": -.06}, {"u_xxxx": .004}):
            with self.subTest(coefficients=coefficients), self.assertRaises(ValueError):
                generic.forecast(terms(coefficients), x, t, initial)

    def test_sparse_library_recovers_three_independently_generated_structures(self):
        for name in ("burgers", "kdv", "ks"):
            with self.subTest(equation=name):
                data, expected = canonical_fixture(name)
                candidates = generic.fit_candidates(deepcopy(data))
                matching = [candidate for candidate in candidates
                            if {term["name"] for term in candidate.get("terms", [])} == set(expected)]
                self.assertTrue(matching, (name, [(c.get("equation"), c.get("failure_reason")) for c in candidates]))
                recovered = {term["name"]: term["coefficient"] for term in matching[0]["terms"]}
                for term, coefficient in expected.items():
                    self.assertAlmostEqual(recovered[term], coefficient, delta=.02*abs(coefficient),
                                           msg=f"{name}: {term}")

    def test_ground_truth_metadata_cannot_change_discovered_candidates(self):
        data, _ = canonical_fixture("burgers")
        changed = deepcopy(data)
        changed.update({"scenario": "ks", "truth": {"u_xx": -999, "u_xxxx": 123},
                        "reference_equation": "u_t = 987 u", "mandatory_terms": ["u_xxxx"]})
        signature = lambda results: [(c.get("equation"), c.get("terms")) for c in results]
        self.assertEqual(signature(generic.fit_candidates(data)), signature(generic.fit_candidates(changed)))

    def test_static_observations_cannot_identify_a_time_evolution_equation(self):
        data, _ = canonical_fixture("burgers")
        static = deepcopy(data)
        static["u"] = [list(data["u"][0]) for _ in data["t"]]
        with self.assertRaises(ValueError):
            generic.fit_candidates(static)

    def test_single_mode_exposes_equivalent_derivative_columns(self):
        x = np.linspace(0, 2*np.pi, 48, endpoint=False)
        t = np.linspace(0, .5, 81)
        u = np.exp(-.2*t[:, None]) * np.sin(x)[None, :]
        candidates = generic.fit_candidates({"x": x, "t": t, "u": u, "boundary": "periodic"})
        self.assertTrue(candidates)
        diagnostics = candidates[0]["diagnostics"]
        self.assertLess(diagnostics["full_library_rank"], len(generic.TERM_NAMES))
        correlated = {frozenset(item["terms"]) for item in diagnostics["correlated_columns"]}
        self.assertIn(frozenset(("u", "u_xx")), correlated)
        self.assertIn(frozenset(("u", "u_xxxx")), correlated)

    def test_forecast_accepts_initial_state_not_future_observation_array(self):
        self.assertEqual(list(inspect.signature(generic.forecast).parameters), ["terms", "x", "t", "u0"])
        data, coefficients = canonical_fixture("burgers")
        before = deepcopy(data)
        initial = np.asarray(data["u"])[0]
        equation = terms(coefficients)
        equation_before = deepcopy(equation)
        generic.forecast(equation, data["x"], data["t"], initial)
        self.assertEqual(data, before)
        self.assertEqual(equation, equation_before)
        with self.assertRaises(ValueError):
            generic.forecast(equation, data["x"], data["t"], np.asarray(data["u"]))

    def test_malformed_inputs_and_unsupported_terms_fail_explicitly(self):
        data, coefficients = canonical_fixture("burgers")
        bad_inputs = []
        for modification in ("shape", "nonfinite", "space", "time", "boundary"):
            bad = deepcopy(data)
            if modification == "shape": bad["u"] = bad["u"][:-1]
            if modification == "nonfinite": bad["u"][3][4] = float("nan")
            if modification == "space": bad["x"][3] += .01
            if modification == "time": bad["t"][3] = bad["t"][2]
            if modification == "boundary": bad["boundary"] = "dirichlet"
            bad_inputs.append((modification, bad))
        for label, bad in bad_inputs:
            with self.subTest(label=label), self.assertRaises(ValueError):
                generic.fit_candidates(bad)
        for bad_terms in ([{"name": "u_yyyy", "coefficient": 1}],
                          [{"name": "u", "coefficient": float("inf")} ]):
            with self.subTest(terms=bad_terms), self.assertRaises(ValueError):
                generic.forecast(bad_terms, data["x"], data["t"], data["u"][0])


if __name__ == "__main__":
    unittest.main()
