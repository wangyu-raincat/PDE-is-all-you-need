"""Independent acceptance checks for the six-sensor real-data adapter.

These manufactured observations validate the numerical adapter; they are not
the real experiment and do not count toward its held-out experimental result.
The reference solution below uses a matrix exponential of a nonperiodic,
six-node semi-discrete system, independently of the discovery and predictor.
"""

from copy import deepcopy
import unittest

import numpy as np
from scipy.linalg import expm

from pde_discovery.real_heat import build_candidates, evaluate, predict


def exact_six_sensor_data():
    """Multiple spatial modes separate diffusion from local decay."""
    increments = 0.004 * (1.0 + 0.25 * np.sin(np.arange(600)))
    t = np.r_[0.0, np.cumsum(increments)]
    dx, diffusivity, decay = 0.2, 0.08, 0.12
    mode = np.arange(1, 5)
    x_index = np.arange(1, 5)
    eigenvalues = -4.0 * np.sin(mode * np.pi / 10.0) ** 2 / dx**2
    eigenvectors = np.sin(np.outer(x_index, mode) * np.pi / 5.0)
    amplitudes = np.array([2.0, 0.6, 0.35, 0.15])
    interior = (
        np.exp(np.outer(t, diffusivity * eigenvalues - decay)) * amplitudes
    ) @ eigenvectors.T
    u = np.column_stack([np.zeros(len(t)), interior, np.zeros(len(t))])
    return t, u, dx, diffusivity, decay


def exact_driven_interiors(t, u0, dx, diffusivity, decay):
    """Exact response to left=1+0.5*tau, right=0.2-0.03*tau."""
    matrix = np.zeros((6, 6))
    matrix[:4, :4] = (
        diffusivity
        / dx**2
        * (np.diag([-2.0] * 4) + np.diag([1.0] * 3, 1)
           + np.diag([1.0] * 3, -1))
        - decay * np.eye(4)
    )
    matrix[[0, 3], 4] = diffusivity / dx**2 * np.array([1.0, 0.2])
    matrix[[0, 3], 5] = diffusivity / dx**2 * np.array([0.5, -0.03])
    matrix[5, 4] = 1.0
    initial = np.r_[u0, 1.0, 0.0]
    return np.array([(expm(matrix * tau) @ initial)[:4] for tau in t - t[0]])


def find_diffusion_decay(candidates):
    return next(
        (candidate for candidate in candidates
         if {term["name"] for term in candidate["terms"]}
         == {"theta", "theta_xx"}),
        None,
    )


class SixSensorNumericalAcceptanceTests(unittest.TestCase):
    def test_single_mode_reports_diffusion_decay_nonidentifiability(self):
        t = np.linspace(0.0, 2.0, 401)
        u = np.exp(-t[:, None]) * np.sin(np.arange(6)[None, :] * np.pi / 5.0)
        candidates = build_candidates(t, u, 0.2, 0.12)
        self.assertTrue(candidates)
        for candidate in candidates:
            diagnostics = candidate["diagnostics"]
            self.assertLess(diagnostics["full_library_rank"], 5)
            correlations = diagnostics["correlated_columns"]
            self.assertTrue(any(
                set(item["terms"]) == {"theta", "theta_xx"}
                and abs(item["correlation"]) > 0.9999
                for item in correlations
            ))

    def test_l1_can_recover_two_terms_from_independent_nonperiodic_data(self):
        t, u, dx, diffusivity, decay = exact_six_sensor_data()
        candidates = build_candidates(t, u, dx, window_seconds=0.12)
        candidate = find_diffusion_decay(candidates)
        self.assertIsNotNone(
            candidate,
            "The L1 path must offer the identifiable two-term mechanism; "
            "this test does not use the manufactured truth to select real-data models.",
        )
        coefficients = {
            term["name"]: term["coefficient"] for term in candidate["terms"]
        }
        self.assertAlmostEqual(
            coefficients["theta_xx"], diffusivity, delta=0.03 * diffusivity
        )
        self.assertAlmostEqual(coefficients["theta"], -decay, delta=0.03 * decay)

    def test_discovered_coefficients_transform_consistently_with_length_units(self):
        t, u, dx, _, _ = exact_six_sensor_data()
        metre = find_diffusion_decay(build_candidates(t, u, dx, 0.12))
        centimetre = find_diffusion_decay(build_candidates(t, u, dx * 100.0, 0.12))
        self.assertIsNotNone(metre)
        self.assertIsNotNone(centimetre)
        metre_terms = {term["name"]: term["coefficient"] for term in metre["terms"]}
        cm_terms = {term["name"]: term["coefficient"] for term in centimetre["terms"]}
        self.assertAlmostEqual(
            cm_terms["theta_xx"] / 10000.0, metre_terms["theta_xx"], delta=1e-8
        )
        self.assertAlmostEqual(cm_terms["theta"], metre_terms["theta"], delta=1e-8)
        boundary = u[:, [0, -1]]
        np.testing.assert_allclose(
            predict(metre["terms"], t, boundary, u[0, 1:-1], dx),
            predict(centimetre["terms"], t, boundary, u[0, 1:-1], dx * 100.0),
            atol=1e-7,
            rtol=1e-7,
        )

    def test_prediction_uses_nonperiodic_time_varying_measured_boundaries(self):
        tau = np.r_[0.0, np.cumsum(0.01 * (1.0 + 0.3 * np.sin(np.arange(120))))]
        t = tau + 7.5  # A shifted time origin must not change the dynamics.
        dx, diffusivity, decay = 0.2, 0.07, 0.1
        u0 = np.array([0.2, 0.4, 0.6, 0.7])
        boundary = np.column_stack([1.0 + 0.5 * tau, 0.2 - 0.03 * tau])
        terms = [
            {"name": "theta_xx", "coefficient": diffusivity},
            {"name": "theta", "coefficient": -decay},
        ]
        reference = exact_driven_interiors(t, u0, dx, diffusivity, decay)
        actual = predict(terms, t, boundary, u0, dx)
        np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            predict(terms, tau, boundary, u0, dx), actual, atol=1e-7, rtol=1e-7
        )
        changed_boundary = boundary.copy()
        changed_boundary[1:, 0] += 0.5 * tau[1:]
        changed = predict(terms, t, changed_boundary, u0, dx)
        self.assertGreater(np.max(np.abs(changed[1:] - actual[1:])), 0.01)

    def test_forecast_contract_excludes_future_interior_observations(self):
        t, measured, dx, diffusivity, decay = exact_six_sensor_data()
        terms = [
            {"name": "theta_xx", "coefficient": diffusivity},
            {"name": "theta", "coefficient": -decay},
        ]
        terms_before = deepcopy(terms)
        measured_before = measured.copy()
        boundaries = measured[:, [0, -1]]
        u0 = measured[0, 1:-1].copy()
        forecast = predict(terms, t, boundaries, u0, dx)
        original_score = evaluate(forecast, measured[:, 1:-1])
        contaminated = measured.copy()
        contaminated[1:, 1:-1] += 100.0
        other_forecast = predict(
            terms, t, contaminated[:, [0, -1]], contaminated[0, 1:-1], dx
        )
        other_score = evaluate(other_forecast, contaminated[:, 1:-1])
        np.testing.assert_array_equal(other_forecast, forecast)
        self.assertGreater(other_score["rmse"], original_score["rmse"] + 99.0)
        self.assertEqual(terms, terms_before)
        np.testing.assert_array_equal(measured, measured_before)
        np.testing.assert_array_equal(boundaries, measured_before[:, [0, -1]])
        np.testing.assert_array_equal(u0, measured_before[0, 1:-1])

    def test_empty_equation_has_zero_interior_dynamics(self):
        t = np.array([2.0, 2.1, 2.5, 3.0])
        u0 = np.array([1.0, 2.0, 3.0, 4.0])
        boundary = np.column_stack([t**2, -t])
        np.testing.assert_allclose(
            predict([], t, boundary, u0, 0.2), np.tile(u0, (len(t), 1)), atol=1e-12
        )


class SixSensorIntegrityAcceptanceTests(unittest.TestCase):
    def test_constant_measurements_cannot_establish_a_dynamic_equation(self):
        with self.assertRaises(ValueError):
            build_candidates(np.linspace(0.0, 2.0, 401), np.ones((401, 6)), 0.2, 0.12)

    def test_scores_exclude_initialization_and_use_only_four_interior_sensors(self):
        observed = np.arange(1.0, 13.0).reshape(3, 4)
        predicted = observed + np.array([10000.0, 1.0, -2.0])[:, None]
        observed_before, predicted_before = observed.copy(), predicted.copy()
        score = evaluate(predicted, observed)
        self.assertAlmostEqual(score["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(score["mae"], 1.5)
        self.assertAlmostEqual(score["max_absolute_error"], 2.0)
        self.assertAlmostEqual(
            score["relative_error"], np.sqrt(20.0) / np.linalg.norm(observed[1:])
        )
        np.testing.assert_allclose(score["per_sensor_rmse"], np.full(4, np.sqrt(2.5)))
        np.testing.assert_array_equal(observed, observed_before)
        np.testing.assert_array_equal(predicted, predicted_before)
        with self.assertRaises(ValueError):
            evaluate(np.zeros((3, 6)), np.ones((3, 6)))

    def test_invalid_discovery_measurements_are_rejected(self):
        t, u, dx, _, _ = exact_six_sensor_data()
        nonfinite = u.copy()
        nonfinite[10, 2] = np.nan
        duplicate = t.copy()
        duplicate[10] = duplicate[9]
        cases = [
            (t[::-1], u, dx),
            (duplicate, u, dx),
            (t, nonfinite, dx),
            (t, u[:, :5], dx),
            (t, u, 0.0),
            (t, u, float("nan")),
        ]
        for index, (bad_t, bad_u, bad_dx) in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    build_candidates(bad_t, bad_u, bad_dx, 0.12)

    def test_invalid_forecast_inputs_cannot_produce_experimental_scores(self):
        t = np.array([0.0, 0.1, 0.2])
        boundary = np.zeros((3, 2))
        u0 = np.ones(4)
        terms = [{"name": "theta_xx", "coefficient": 0.1}]
        boundary_nan = boundary.copy()
        boundary_nan[1, 0] = np.nan
        cases = [
            (terms, t[::-1], boundary, u0, 0.2),
            (terms, t, boundary_nan, u0, 0.2),
            (terms, t, np.zeros((3, 3)), u0, 0.2),
            (terms, t, boundary, np.ones(6), 0.2),
            (terms, t, boundary, u0, -0.2),
            ([{"name": "unknown_term", "coefficient": 1.0}], t, boundary, u0, 0.2),
            ([{"name": "theta_xx", "coefficient": float("nan")}],
             t, boundary, u0, 0.2),
        ]
        for index, arguments in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(ValueError):
                    predict(*arguments)


if __name__ == "__main__":
    unittest.main()
