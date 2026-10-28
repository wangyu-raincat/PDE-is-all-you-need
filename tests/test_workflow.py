"""Scientific acceptance checks for discovery and evidence-based completion.

Run from the project root with:
    python -m unittest discover -s tests -v

Synthetic ground-truth coefficients appear only in these acceptance checks;
the discovery controller receives observations, not these answers.
"""

from copy import deepcopy
import unittest

import numpy as np

from pde_discovery.controller import discover_uploaded, run_demo
from pde_discovery.engine import discover
from pde_discovery.simulator import make_dataset


def coefficient_map(result):
    return {term["name"]: term["coefficient"] for term in result["terms"]}


def observed_diffusion():
    return make_dataset(
        "diffusion", seed=7, noise=0.0, initial_condition="train"
    )


class DiscoveryAcceptanceTests(unittest.TestCase):
    """Recover the mechanism on clean data and avoid false certainty."""

    def test_clean_diffusion_recovers_structure_and_coefficient(self):
        result = run_demo("diffusion", noise=0.0, seed=7)
        terms = coefficient_map(result)
        self.assertEqual(result["status"], "identified")
        self.assertTrue(result["goal"]["achieved"])
        self.assertEqual(set(terms), {"u_xx"})
        self.assertAlmostEqual(terms["u_xx"], 0.2, delta=0.01)

    def test_clean_advection_diffusion_recovers_both_mechanisms(self):
        result = run_demo("advection_diffusion", noise=0.0, seed=11)
        terms = coefficient_map(result)
        self.assertEqual(result["status"], "identified")
        self.assertTrue(result["goal"]["achieved"])
        self.assertEqual(set(terms), {"u_x", "u_xx"})
        self.assertAlmostEqual(terms["u_x"], -0.7, delta=0.035)
        self.assertAlmostEqual(terms["u_xx"], 0.12, delta=0.006)

    def test_single_mode_does_not_establish_a_unique_mechanism(self):
        # For exp(-t) sin(x), diffusion and local decay are indistinguishable.
        result = run_demo(
            "ambiguity", noise=0.0, seed=7, auto_experiment=False
        )
        self.assertEqual(result["status"], "ambiguous")
        self.assertFalse(result["goal"]["achieved"])
        self.assertTrue(result["experiment"]["proposed"])
        self.assertFalse(result["experiment"]["executed"])

    def test_an_independent_informative_experiment_resolves_ambiguity(self):
        result = run_demo(
            "ambiguity", noise=0.0, seed=7, auto_experiment=True
        )
        terms = coefficient_map(result)
        self.assertTrue(result["experiment"]["executed"])
        self.assertEqual(result["status"], "identified")
        self.assertTrue(result["goal"]["achieved"])
        self.assertEqual(set(terms), {"u_xx"})
        self.assertAlmostEqual(terms["u_xx"], 1.0, delta=0.05)

    def test_constant_data_provides_insufficient_identification_evidence(self):
        data = observed_diffusion()
        data["u"] = np.ones_like(np.asarray(data["u"], dtype=float)).tolist()
        result = discover([data])
        self.assertEqual(result["status"], "insufficient")

    def test_proposed_experiment_is_more_informative_than_amplitude_change(self):
        result = run_demo("ambiguity", noise=0.0, seed=7, auto_experiment=False)
        experiment = result["experiment"]
        scores = {
            item["initial_condition"]: item["disagreement"]
            for item in experiment["probe_scores"]
        }
        selected = experiment["selected_initial_condition"]
        self.assertIn(selected, {"probe2", "probe3"})
        self.assertIsNotNone(scores[selected])
        self.assertIsNotNone(scores["amplitude"])
        # The two competing PDEs agree on amplitude changes of the first
        # mode, but genuinely disagree after changing the spatial mode.
        self.assertGreater(scores[selected], scores["amplitude"] + 0.01)

    def test_final_validation_does_not_double_as_structure_selection(self):
        training = make_dataset(
            "ambiguity", seed=7, noise=0.0, initial_condition="train"
        )
        validation = make_dataset(
            "ambiguity", seed=29, noise=0.0, initial_condition="validation"
        )
        training_only = discover([training])
        result = discover_uploaded({"training": [training], "validation": validation})
        # Validation may refute a candidate, but using it to choose the
        # structure consumes its independence as the final holdout. The
        # controller must request new training evidence and another holdout.
        self.assertEqual(result["status"], "ambiguous")
        self.assertFalse(result["goal"]["achieved"])
        self.assertEqual(result["terms"], training_only["terms"])
        self.assertEqual(
            [candidate["terms"] for candidate in result["candidates"]],
            [candidate["terms"] for candidate in training_only["candidates"]],
        )


class DataIntegrityTests(unittest.TestCase):
    """Invalid or reused evidence must not establish goal completion."""

    def test_invalid_observations_are_rejected(self):
        original = observed_diffusion()
        nonfinite = deepcopy(original)
        field = np.asarray(nonfinite["u"], dtype=float).copy()
        field[3, 3] = np.nan
        nonfinite["u"] = field.tolist()

        reversed_time = deepcopy(original)
        reversed_time["t"] = np.asarray(original["t"], dtype=float)[::-1].tolist()

        wrong_shape = deepcopy(original)
        wrong_shape["u"] = np.asarray(original["u"], dtype=float)[:-1].tolist()

        for name, invalid in [
            ("nonfinite field", nonfinite),
            ("decreasing timestamps", reversed_time),
            ("inconsistent shape", wrong_shape),
        ]:
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    discover([invalid])

    def test_a_copied_training_trajectory_is_not_independent_validation(self):
        data = observed_diffusion()
        payload = {"training": [data], "validation": deepcopy(data)}
        with self.assertRaises(ValueError):
            discover_uploaded(payload)

    def test_no_validation_data_means_goal_is_not_yet_complete(self):
        payload = {"training": [observed_diffusion()], "validation": None}
        result = discover_uploaded(payload)
        self.assertFalse(result["goal"]["achieved"])
        self.assertEqual(result["status"], "waiting_for_data")

    def test_shifting_time_origin_does_not_make_a_copy_independent(self):
        training = observed_diffusion()
        validation = deepcopy(training)
        validation["t"] = (np.asarray(validation["t"], dtype=float) + 10).tolist()
        validation["label"] = "claimed independent experiment"
        with self.assertRaises(ValueError):
            discover_uploaded({"training": [training], "validation": validation})

    def test_validation_from_a_different_mechanism_fails_the_goal(self):
        validation = make_dataset(
            "advection_diffusion", seed=29, noise=0.0,
            initial_condition="validation",
        )
        result = discover_uploaded({
            "training": [observed_diffusion()], "validation": validation,
        })
        self.assertFalse(result["goal"]["achieved"])
        self.assertGreater(
            result["diagnostics"]["validation_relative_error"],
            result["goal"]["validation_error_threshold"],
        )

    def test_fake_truth_metadata_cannot_change_the_discovered_equation(self):
        original = observed_diffusion()
        tampered = deepcopy(original)
        tampered.update({
            "truth": {"u_x": 1234.0, "u_xx": -999.0},
            "scenario": "advection_diffusion",
            "equation": "u_t = 1234 u_x - 999 u_xx",
            "goal": {"achieved": True},
        })
        baseline = discover([original])
        poisoned = discover([tampered])
        self.assertEqual(poisoned["terms"], baseline["terms"])
        self.assertEqual(poisoned["equation"], baseline["equation"])
        self.assertEqual(poisoned["status"], baseline["status"])


if __name__ == "__main__":
    unittest.main()
