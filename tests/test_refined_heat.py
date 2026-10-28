"""Independent numerical checks for the constrained heat-rod refinement.

All observations here are manufactured test fixtures.  These tests do not read
experimental records or establish that the chosen prior is physically correct.
"""

from copy import deepcopy
import inspect
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import solve_ivp

from pde_discovery import real_heat_refined as refined


def terms(diffusivity=7e-5, decay=0.06, source=0.0):
    result = [{"name": "theta_xx", "coefficient": diffusivity}]
    if decay:
        result.append({"name": "theta", "coefficient": -decay})
    if source:
        result.append({"name": "1", "coefficient": source})
    return result


def independent_driven_reference(t, boundary, u0, dx, diffusivity, decay, source):
    """High-accuracy ODE reference restarted at every boundary slope change."""
    laplacian = (
        np.diag([-2.0] * 4)
        + np.diag([1.0] * 3, 1)
        + np.diag([1.0] * 3, -1)
    ) / dx**2
    result = [np.asarray(u0, dtype=float).copy()]
    for i, duration in enumerate(np.diff(t)):
        initial_boundary = boundary[i].copy()
        slope = (boundary[i + 1] - initial_boundary) / duration

        def rhs(local_time, state):
            endpoints = initial_boundary + local_time * slope
            derivative = diffusivity * laplacian @ state - decay * state + source
            derivative[0] += diffusivity * endpoints[0] / dx**2
            derivative[-1] += diffusivity * endpoints[-1] / dx**2
            return derivative

        solution = solve_ivp(
            rhs, (0.0, duration), result[-1], method="DOP853",
            rtol=1e-11, atol=1e-12,
        )
        if not solution.success:
            raise AssertionError(solution.message)
        result.append(solution.y[:, -1])
    return np.array(result)


def identifiable_fixture():
    increments = 0.04 * (1.0 + 0.3 * np.sin(np.arange(250)))
    t = np.r_[0.0, np.cumsum(increments)]
    dx, diffusivity, decay = 0.01, 7e-5, 0.06
    modes = np.arange(1, 5)
    eigenvalues = -4 * np.sin(modes * np.pi / 10.0) ** 2 / dx**2
    vectors = np.sin(np.outer(np.arange(1, 5), modes) * np.pi / 5.0)
    interior = (
        np.exp(np.outer(t, diffusivity * eigenvalues - decay))
        * np.array([4.0, 1.0, 0.7, 0.4])
    ) @ vectors.T
    u = np.column_stack([np.zeros(len(t)), interior, np.zeros(len(t))])
    return t, u, dx, diffusivity, decay


class RefinedHeatNumericalTests(unittest.TestCase):
    def test_exact_propagation_matches_independent_ode_for_irregular_forcing(self):
        tau = np.array([0.0, 0.003, 0.03, 0.08, 0.21, 0.7, 1.1, 2.0, 3.0, 5.0])
        t = tau + 7.5
        boundary = np.column_stack([
            1.0 + np.sin(1.3 * tau), 0.7 + 0.3 * np.cos(0.8 * tau),
        ])
        u0 = np.array([0.2, 0.4, 0.6, 0.7])
        dx, diffusivity, decay, source = 0.01, 8e-5, 0.03, 0.4
        equation = terms(diffusivity, decay, source)
        expected = independent_driven_reference(
            t, boundary, u0, dx, diffusivity, decay, source
        )
        actual = refined.predict(equation, t, boundary, u0, dx)
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)
        np.testing.assert_allclose(
            refined.predict(equation, tau, boundary, u0, dx), actual,
            rtol=2e-11, atol=2e-11,
        )

    def test_tiny_and_large_steps_remain_finite_and_approach_steady_solution(self):
        t = np.array([0.0, 1e-8, 1000.0, 2000.0])
        boundary = np.tile([1.0, 2.0], (len(t), 1))
        actual = refined.predict(terms(1e-4, 0.0), t, boundary, np.zeros(4), 0.001)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreaterEqual(float(actual.min()), -1e-10)
        self.assertLessEqual(float(actual.max()), 2.0 + 1e-10)
        np.testing.assert_allclose(actual[-1], [1.2, 1.4, 1.6, 1.8], atol=1e-9)

    def test_multimode_training_recovers_identifiable_diffusion_and_decay(self):
        t, u, dx, diffusivity, decay = identifiable_fixture()
        candidate = refined.fit_candidate(t, u, dx, "D+h")
        self.assertTrue(candidate["fit_success"], candidate.get("diagnostics"))
        coefficients = {term["name"]: term["coefficient"] for term in candidate["terms"]}
        self.assertAlmostEqual(coefficients["theta_xx"], diffusivity, delta=diffusivity * 0.005)
        self.assertAlmostEqual(coefficients["theta"], -decay, delta=decay * 0.005)
        self.assertEqual(candidate["support_size"], 2)
        forecast = refined.predict(candidate["terms"], t, u[:, [0, -1]], u[0, 1:-1], dx)
        np.testing.assert_allclose(forecast, u[:, 1:-1], atol=1e-5, rtol=1e-5)

    def test_prediction_contract_excludes_future_interiors_and_preserves_inputs(self):
        self.assertEqual(
            list(inspect.signature(refined.predict).parameters),
            ["terms", "t", "boundaries", "u0", "dx"],
        )
        t, u, dx, diffusivity, decay = identifiable_fixture()
        equation = terms(diffusivity, decay)
        equation_before, u_before = deepcopy(equation), u.copy()
        original = refined.predict(equation, t, u[:, [0, -1]], u[0, 1:-1], dx)
        changed = u.copy()
        changed[1:, 1:-1] += 100.0
        other = refined.predict(equation, t, changed[:, [0, -1]], changed[0, 1:-1], dx)
        np.testing.assert_array_equal(original, other)
        np.testing.assert_array_equal(u, u_before)
        self.assertEqual(equation, equation_before)


class RefinedHeatConstraintTests(unittest.TestCase):
    def test_one_support_failure_preserves_other_candidate_evidence(self):
        t, u, dx, _, _ = identifiable_fixture()
        called = []

        def one_failed_support(time, observed, spacing, model_name):
            called.append(model_name)
            if model_name == "D+h":
                raise RuntimeError("manufactured support failure")
            return {"model_name": model_name, "fit_success": True,
                    "evidence": "manufactured surviving support"}

        with patch.object(refined, "fit_candidate", side_effect=one_failed_support):
            candidates = refined.fit_candidates(t, u, dx)
        self.assertEqual(called, list(refined.MODEL_NAMES))
        self.assertEqual(len(candidates), 4)
        failed = next(c for c in candidates if c["model_name"] == "D+h")
        self.assertFalse(failed["fit_success"])
        self.assertIn("manufactured support failure", failed["failure_reason"])
        self.assertTrue(all(c["fit_success"] for c in candidates if c is not failed))

    def test_one_optimizer_start_failure_does_not_discard_remaining_fixed_starts(self):
        t, u, dx, diffusivity, _ = identifiable_fixture()
        original_optimizer = refined.least_squares
        calls = []

        def first_start_fails(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("manufactured first-start failure")
            return original_optimizer(*args, **kwargs)

        with patch.object(refined, "least_squares", side_effect=first_start_fails):
            candidate = refined.fit_candidate(t, u, dx, "D+h")
        self.assertEqual(len(calls), len(refined.STARTS))
        self.assertTrue(candidate["fit_success"])
        runs = candidate["diagnostics"]["starts"]
        self.assertFalse(runs[0]["success"])
        self.assertIn("manufactured first-start failure", runs[0]["failure_reason"])
        self.assertTrue(any(run["success"] for run in runs[1:]))
        self.assertAlmostEqual(candidate["parameters"]["D"], diffusivity, delta=0.005 * diffusivity)

    def test_nonphysical_and_unsupported_equations_are_rejected(self):
        t = np.array([0.0, 0.1, 0.2])
        boundary, u0 = np.zeros((3, 2)), np.ones(4)
        invalid = [
            [],
            [{"name": "theta_xx", "coefficient": 0.0}],
            [{"name": "theta_xx", "coefficient": -1e-5}],
            terms() + [{"name": "theta_x", "coefficient": 0.1}],
            terms() + [{"name": "theta^2", "coefficient": -0.1}],
            [{"name": "theta_xx", "coefficient": 1e-5},
             {"name": "theta", "coefficient": 0.1}],
            [{"name": "theta_xx", "coefficient": float("nan")}],
        ]
        for equation in invalid:
            with self.subTest(equation=equation):
                with self.assertRaises(ValueError):
                    refined.predict(equation, t, boundary, u0, 0.01)

    def test_malformed_observations_and_boundaries_are_rejected(self):
        t, u, dx, _, _ = identifiable_fixture()
        bad_time, bad_u = t.copy(), u.copy()
        bad_time[4] = bad_time[3]
        bad_u[2, 3] = np.nan
        for args in [(bad_time, u, dx), (t, bad_u, dx), (t, u[:, :5], dx),
                     (t, u, 0.0), (t, u, float("nan"))]:
            with self.subTest(input_shapes=[np.shape(x) for x in args]):
                with self.assertRaises(ValueError):
                    refined.fit_candidates(*args)
        for boundary, initial in [(u, u[0, 1:-1]), (u[:, [0, -1]], np.ones(6)),
                                  (np.full((len(t), 2), np.nan), u[0, 1:-1])]:
            with self.subTest(boundary_shape=boundary.shape, initial_shape=initial.shape):
                with self.assertRaises(ValueError):
                    refined.predict(terms(), t, boundary, initial, dx)


if __name__ == "__main__":
    unittest.main()
