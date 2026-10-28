"""Independent manufactured checks for the isolated L1 method assessment.

No public data, archived final arrays, or analytic-study final trajectories are
loaded here. These are implementation checks, not scientific evaluation cases.
"""
from copy import deepcopy
from unittest.mock import patch
import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from l1_study import engine
from pde_discovery import generic_engine as baseline


def manufactured_advection_diffusion():
    """Differentiate the Fourier solution directly, rather than a fitted RHS."""
    x = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    t = np.linspace(0, .8, 41)
    components = []
    time_components = []
    for k, amplitude, phase in ((1, .7, .13), (2, .3, -.21), (3, .16, .37), (4, .08, -.4)):
        eigenvalue = -.12 * k**2 - .65j * k
        value = amplitude * np.exp(eigenvalue * t[:, None] + 1j * (k * x[None, :] + phase))
        components.append((k, value))
        time_components.append(eigenvalue * value)
    u = .15 + sum(value.real for _, value in components)
    derivatives = [sum(((1j*k)**order * value).real for k, value in components)
                   for order in range(1, 5)]
    features = np.stack([np.ones_like(u), u, u**2, u**3, *derivatives,
                         u*derivatives[0]], axis=-1)
    return {"x": x, "t": t, "u": u, "boundary": "periodic",
            "exact_features": features, "exact_ut": sum(v.real for v in time_components)}


def matrix_fixture():
    rng = np.random.default_rng(8317)
    X = rng.normal(size=(67, len(engine.TERM_NAMES)))
    beta = np.array([.6, 0, 0, 0, -.4, .16, 0, 0, 0])
    y = X @ beta + .002 * rng.normal(size=len(X))
    return {"X": X, "y": y}


def signatures(candidates):
    """Ignore labels/path order while preserving every actual retained term."""
    return {tuple(t["name"] for t in c["terms"]):
            np.asarray([t["coefficient"] for t in c["terms"]]) for c in candidates}


class IsolatedL1EngineTests(unittest.TestCase):
    def test_plain_single_integral_matches_frozen_production_algorithm(self):
        data = manufactured_advection_diffusion()
        expected = signatures(baseline.fit_candidates(data))
        actual = signatures(engine.fit_blocks([engine.integrated_block(data)], "plain"))
        self.assertTrue(expected)
        self.assertEqual(set(expected), set(actual))
        for support in expected:
            np.testing.assert_allclose(actual[support], expected[support], rtol=0, atol=1e-12)

    def test_weighted_solve_matches_independent_orthogonal_soft_threshold(self):
        n = len(engine.TERM_NAMES)
        X = np.sqrt(n) * np.eye(n)
        beta = np.array([.7, -.6, .02, -.03, .5, 0, .01, 0, 0])
        weights = np.array([.5, 2, 1, 4, 1.5, .2, 10, 100, 1000])
        alpha = .1
        actual, diagnostic = engine._solve(X, X @ beta, alpha, weights, 1)
        expected = np.sign(beta) * np.maximum(np.abs(beta) - alpha*weights, 0)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        self.assertTrue(diagnostic["converged"])

    def test_adaptive_pilot_is_normalized_ridge_and_uses_one_solve_per_alpha(self):
        block = matrix_fixture()
        scales = np.sqrt(np.mean(block["X"]**2, axis=0))
        target_scale = np.sqrt(np.mean(block["y"]**2))
        X, y = block["X"]/scales, block["y"]/target_scale
        pilot = np.linalg.solve(X.T@X/len(y) + 1e-6*np.eye(9), X.T@y/len(y))
        expected_weights = np.clip(1/(np.abs(pilot)+1e-3), .01, 1000)
        solve = engine._solve
        observed = []

        def recording_solve(X_arg, y_arg, alpha, weights, iteration):
            observed.append((alpha, iteration, weights.copy()))
            return solve(X_arg, y_arg, alpha, weights, iteration)

        with patch.object(engine, "_solve", side_effect=recording_solve):
            candidates = engine.fit_blocks([block], "adaptive")
        self.assertEqual(len(observed), len(engine.ALPHA_PATH))
        for _, iteration, weights in observed:
            self.assertEqual(iteration, 1)
            np.testing.assert_allclose(weights, expected_weights, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(candidates[0]["diagnostics"]["ridge_pilot"], pilot,
                                   rtol=1e-12, atol=1e-12)

    def test_irl1_five_updates_and_each_alpha_restarts_from_same_pilot(self):
        solve, observations = engine._solve, []

        def recording_solve(X, y, alpha, weights, iteration):
            coefficients, diagnostic = solve(X, y, alpha, weights, iteration)
            observations.append((alpha, iteration, weights.copy(), coefficients.copy()))
            return coefficients, diagnostic

        with patch.object(engine, "_solve", side_effect=recording_solve):
            candidates = engine.fit_blocks([matrix_fixture()], "irl1")
        self.assertEqual(len(observations), 5*len(engine.ALPHA_PATH))
        initial = observations[0][2]
        for index, alpha in enumerate(engine.ALPHA_PATH):
            path = observations[5*index:5*(index+1)]
            self.assertEqual([r[1] for r in path], [1, 2, 3, 4, 5])
            self.assertTrue(all(r[0] == alpha for r in path))
            np.testing.assert_array_equal(path[0][2], initial)
            for previous, current in zip(path, path[1:]):
                expected = np.clip(1/(np.abs(previous[3])+1e-3), .01, 1000)
                np.testing.assert_array_equal(current[2], expected)
        self.assertTrue(candidates)
        self.assertTrue(all(r["reweighting_solves"] == 5
                            for r in candidates[0]["diagnostics"]["all_alpha_runs"]))

    def test_irl1_result_does_not_depend_on_alpha_traversal_order(self):
        block = matrix_fixture()
        forward = signatures(engine.fit_blocks([block], "irl1"))
        with patch.object(engine, "ALPHA_PATH", tuple(reversed(engine.ALPHA_PATH))):
            reverse = signatures(engine.fit_blocks([block], "irl1"))
        self.assertEqual(set(forward), set(reverse))
        for support in forward:
            np.testing.assert_allclose(forward[support], reverse[support], atol=1e-12, rtol=0)

    def test_pooling_is_equal_trajectory_mean_not_equal_raw_row_mean(self):
        first = {"X": np.ones((3, 9)), "y": np.ones(3)}
        second = {"X": 3*np.ones((9, 9)), "y": 5*np.ones(9)}
        X, y, diagnostics = engine.pool_blocks([first, second])
        np.testing.assert_allclose(np.mean(X**2, axis=0), (1+9)/2)
        self.assertAlmostEqual(float(np.mean(y**2)), (1+25)/2)
        self.assertEqual(diagnostics["rows_per_trajectory"], [3, 9])
        self.assertEqual(diagnostics["trajectory_count"], 2)

    def test_repeating_rows_inside_one_trajectory_does_not_add_trajectory_weight(self):
        first = matrix_fixture()
        second = {"X": first["X"][:29]*.8, "y": first["y"][:29]*.8 + .007}
        repeated = {"X": np.repeat(first["X"], 3, axis=0),
                    "y": np.repeat(first["y"], 3, axis=0)}
        for method in engine.METHODS:
            with self.subTest(method=method):
                original = signatures(engine.fit_blocks([first, second], method))
                duplicated = signatures(engine.fit_blocks([repeated, second], method))
                self.assertEqual(set(original), set(duplicated))
                for support in original:
                    np.testing.assert_allclose(original[support], duplicated[support],
                                               rtol=1e-9, atol=1e-10)

    def test_exact_and_numerical_pointwise_rows_have_matching_domains(self):
        data = manufactured_advection_diffusion()
        exact = engine.pointwise_block(data, exact=True)
        estimated = engine.pointwise_block(data, exact=False)
        expected_rows = (len(data["t"])-4)*len(data["x"])
        self.assertEqual(exact["X"].shape, (expected_rows, 9))
        self.assertEqual(estimated["X"].shape, exact["X"].shape)
        np.testing.assert_array_equal(exact["y"], data["exact_ut"][2:-2].ravel())
        np.testing.assert_allclose(exact["X"], estimated["X"], rtol=0, atol=1e-8)
        difference = np.linalg.norm(exact["y"]-estimated["y"])/np.linalg.norm(exact["y"])
        self.assertGreater(difference, 1e-8)
        self.assertLess(difference, .002)
        self.assertTrue(exact["metadata"]["exact_derivative_diagnostic"])
        self.assertFalse(estimated["metadata"]["exact_derivative_diagnostic"])

    def test_numerical_mode_does_not_use_supplied_exact_derivative_arrays(self):
        data = manufactured_advection_diffusion()
        expected = engine.pointwise_block(data)
        data["exact_ut"][:] = 10000
        data["exact_features"][:] = np.nan
        actual = engine.pointwise_block(data)
        np.testing.assert_array_equal(actual["X"], expected["X"])
        np.testing.assert_array_equal(actual["y"], expected["y"])

    def test_exact_mode_requires_real_matching_finite_derivatives(self):
        data = manufactured_advection_diffusion()
        for change in ({"exact_ut": np.zeros((3, 3))},
                       {"exact_features": np.full_like(data["exact_features"], np.nan)},
                       {"exact_ut": data["exact_ut"].astype(complex)+1j}):
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                engine.pointwise_block({**data, **change}, exact=True)
        del data["exact_ut"]
        with self.assertRaises(ValueError):
            engine.pointwise_block(data, exact=True)

    def test_truth_metadata_cannot_change_fitted_candidates(self):
        block = matrix_fixture()
        poisoned = {**block, "truth": {"u_xx": 999}, "reference_equation": "u_t=999u",
                    "metadata": {"truth": "prefer u^3", "mandatory_terms": ["u^3"]}}
        for method in engine.METHODS:
            expected = signatures(engine.fit_blocks([block], method))
            actual = signatures(engine.fit_blocks([poisoned], method))
            self.assertEqual(set(expected), set(actual))
            for support in expected:
                np.testing.assert_array_equal(actual[support], expected[support])

    def test_invalid_blocks_and_zero_excitation_are_rejected(self):
        good = matrix_fixture()
        invalid = [{}, {"X": np.zeros((3, 8)), "y": np.ones(3)},
                   {"X": np.zeros((3, 9)), "y": np.ones(2)},
                   {"X": np.full((3, 9), np.nan), "y": np.ones(3)},
                   {"X": np.ones((3, 9), complex)+1j, "y": np.ones(3)},
                   {"X": good["X"], "y": np.zeros(len(good["y"]))}]
        for block in invalid:
            with self.subTest(keys=list(block)), self.assertRaises(ValueError):
                engine.fit_blocks([block])
        with self.assertRaises(ValueError):
            engine.fit_blocks([])
        with self.assertRaises(ValueError):
            engine.fit_blocks([good], "frequency_vote")

    def test_convergence_warning_is_not_treated_as_success(self):
        class UnconvergedLasso:
            def __init__(self, **_kwargs):
                self.coef_ = np.ones(9)
                self.n_iter_ = 100000
                self.dual_gap_ = .1

            def fit(self, _X, _y):
                warnings.warn("manufactured convergence failure", ConvergenceWarning)
                return self

        with patch.object(engine, "Lasso", UnconvergedLasso):
            candidates = engine.fit_blocks([matrix_fixture()])
        self.assertTrue(candidates)
        self.assertTrue(all(not c["fit_success"] for c in candidates))
        result = engine.select_candidates(candidates, [manufactured_advection_diffusion()])
        self.assertIsNone(result["selected"])

    def test_rank_deficient_support_is_retained_as_failed_not_silently_repaired(self):
        x = np.linspace(-1, 1, 21)
        X = np.zeros((len(x), 9))
        X[:, 1] = x
        X[:, 2] = x
        beta = np.zeros(9)
        beta[1:3] = .5

        def force_alias(_X, _y, alpha, weights, iteration):
            return beta.copy(), {"alpha": alpha, "iteration": iteration, "converged": True,
                                 "iterations": 1, "dual_gap": 0, "weights": weights.tolist()}

        with patch.object(engine, "_solve", side_effect=force_alias):
            candidates = engine.fit_blocks([{"X": X, "y": x}])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["support_size"], 2)
        self.assertEqual(candidates[0]["diagnostics"]["support_rank"], 1)
        self.assertFalse(candidates[0]["fit_success"])

    def test_selection_uses_development_and_complexity_not_objective_or_frequency(self):
        data = manufactured_advection_diffusion()
        candidates = []
        for identifier, count, marker in (("a", 3, 1), ("b", 2, 2), ("c", 1, 3)):
            candidates.append({"candidate_id": identifier, "fit_success": True,
                               "support_size": count,
                               "terms": [{"name": engine.TERM_NAMES[i], "coefficient": marker}
                                         for i in range(count)],
                               "penalized_objective": -1e20 if identifier == "a" else 1e20,
                               "alpha_values": list(range(100)) if identifier == "a" else [1]})
        before = deepcopy(candidates)

        def fake_forecast(terms, _x, _t, _initial):
            error = {1: .1, 2: .104, 3: .2}[terms[0]["coefficient"]]
            return data["u"] + error

        with patch.object(engine.baseline, "forecast", side_effect=fake_forecast):
            result = engine.select_candidates(candidates, [data])
        self.assertEqual(result["selected"]["candidate_id"], "b")
        self.assertEqual(candidates, before)
        self.assertEqual(len(result["near_optimal_supports"]), 2)

    def test_development_forecast_failure_remains_in_candidate_audit(self):
        candidate = {"candidate_id": "only", "fit_success": True, "support_size": 1,
                     "terms": [{"name": "u_xx", "coefficient": -.1}]}
        with patch.object(engine.baseline, "forecast", side_effect=ValueError("unstable fixture")):
            result = engine.select_candidates([candidate], [manufactured_advection_diffusion()])
        self.assertIsNone(result["selected"])
        self.assertEqual(len(result["candidates"]), 1)
        self.assertIn("unstable fixture", result["candidates"][0]["rejection_reason"])


if __name__ == "__main__":
    unittest.main()
