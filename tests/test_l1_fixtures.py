"""Independent analytic-fixture checks; no final-role field is generated here."""
import unittest
from unittest.mock import patch

import numpy as np

from l1_study import fixtures


class AnalyticFixtureTests(unittest.TestCase):
    def test_shapes_fields_and_predeclared_grid(self):
        for family in fixtures.FAMILIES:
            with self.subTest(family=family):
                data = fixtures.generate(family, "training")
                self.assertEqual(data["u"].shape, (81, 64))
                self.assertEqual(data["exact_features"].shape, (81, 64, 9))
                self.assertEqual(data["exact_ut"].shape, (81, 64))
                self.assertTrue(np.isfinite(data["exact_features"]).all())
                self.assertEqual(data["boundary"], "periodic")
                self.assertEqual(data["x"][0], 0)
                self.assertLess(data["x"][-1], 2*np.pi)
                np.testing.assert_array_equal(data["exact_features"][..., 1], data["u"])
                np.testing.assert_array_equal(data["exact_features"][..., 8],
                                              data["u"]*data["exact_features"][..., 4])

    def test_analytic_fields_satisfy_the_separate_scoring_reference(self):
        for family in fixtures.FAMILIES:
            with self.subTest(family=family):
                data = fixtures.generate(family, "training", 1)
                rhs = np.zeros_like(data["u"])
                for term in fixtures.reference_terms(family):
                    rhs += term["coefficient"] * data["exact_features"][..., fixtures.TERM_NAMES.index(term["name"])]
                np.testing.assert_allclose(rhs, data["exact_ut"], rtol=2e-12, atol=2e-12)

    def test_linear_spatial_derivatives_match_independent_fft(self):
        for family in fixtures.FAMILIES[:2]:
            data = fixtures.generate(family, "training", 2)
            wave = np.fft.fftfreq(64, d=1/64)
            spectrum = np.fft.fft(data["u"], axis=1)
            for order in range(1, 5):
                expected = np.fft.ifft(spectrum*(1j*wave)**order, axis=1).real
                np.testing.assert_allclose(expected, data["exact_features"][..., 3+order],
                                           rtol=2e-9, atol=2e-9)

    def test_linear_time_derivative_matches_independent_complex_step(self):
        h = 1e-25
        for family in fixtures.FAMILIES[:2]:
            a, phase, _ = fixtures._initial_parameters(family, "training", 0)
            k = np.arange(1, 5)
            data = fixtures.generate(family, "training", 0)
            # Real-valued cos representation permits a complex time increment.
            if family == "advdiff":
                growth, frequency = -.12*k**2, -.65*k
            else:
                growth = .04*k**2 - .004*k**4
                frequency = -.4*k + .07*k**3
            tc = data["t"][:, None, None] + 1j*h
            uc = .15 + np.sum(a*np.exp(growth*tc)*np.cos(
                data["x"][None, :, None]*k + phase + frequency*tc), axis=-1)
            np.testing.assert_allclose(uc.imag/h, data["exact_ut"], rtol=2e-13, atol=2e-13)

    def test_cole_hopf_first_and_time_derivative_independent_complex_step(self):
        data = fixtures.generate("burgers_cole_hopf", "training", 0)
        a, phase, _ = fixtures._initial_parameters("burgers_cole_hopf", "training", 0)
        k = np.arange(1, 5)
        x, t, h = data["x"], data["t"], 1e-25

        def independent_field(xc, tc):
            argument = xc[None, :, None]*k + phase
            factor = a*np.exp(-.12*tc[:, None, None]*k**2)
            q = 1 + np.sum(factor*np.cos(argument), axis=-1)
            qx = np.sum(-factor*k*np.sin(argument), axis=-1)
            return -.24*qx/q

        np.testing.assert_allclose(independent_field(x+1j*h, t).imag/h,
                                   data["exact_features"][..., 4], atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(independent_field(x, t+1j*h).imag/h,
                                   data["exact_ut"], atol=2e-13, rtol=2e-13)

    def test_cole_hopf_high_spatial_derivatives_independent_fine_grid_fft(self):
        a, phase, _ = fixtures._initial_parameters("burgers_cole_hopf", "training", 1)
        x = np.linspace(0, 2*np.pi, 256, endpoint=False)
        t = np.array([0.0, .31, .8])
        k = np.arange(1, 5)
        argument = x[None, :, None]*k + phase
        factor = a*np.exp(-.12*t[:, None, None]*k**2)
        q = 1 + np.sum(factor*np.cos(argument), axis=-1)
        qx = -np.sum(factor*k*np.sin(argument), axis=-1)
        independently_evaluated_u = -.24*qx/q
        analytic, _ = fixtures._cole_hopf_fields(x, t, a, phase)
        spectrum = np.fft.fft(independently_evaluated_u, axis=1)
        wave = np.fft.fftfreq(256, d=1/256)
        for order in (2, 3, 4):
            expected = np.fft.ifft(spectrum*(1j*wave)**order, axis=1).real
            np.testing.assert_allclose(analytic[order], expected, atol=8e-8, rtol=2e-8)

    def test_generation_does_not_consult_scoring_reference(self):
        with patch.object(fixtures, "reference_terms", side_effect=AssertionError("Scoring reference read")):
            for family in fixtures.FAMILIES:
                fixtures.generate(family, "training", 0)

    def test_training_initial_conditions_are_distinct_reproducible_and_role_seeds_disjoint(self):
        declared_seeds = [base+i for role, base in fixtures.ROLE_SEEDS.items()
                          for i in range(fixtures.ROLE_COUNTS[role])]
        self.assertEqual(len(declared_seeds), len(set(declared_seeds)))
        for family in fixtures.FAMILIES:
            trajectories = [fixtures.generate(family, "training", i) for i in range(3)]
            self.assertEqual(len({d["metadata"]["initial_field_sha256"] for d in trajectories}), 3)
            repeated = fixtures.generate(family, "training", 0)
            np.testing.assert_array_equal(trajectories[0]["u"], repeated["u"])
            trajectories[0]["u"][0, 0] = 123
            self.assertNotEqual(trajectories[0]["u"][0, 0], repeated["u"][0, 0])

    def test_invalid_family_role_and_index_are_rejected(self):
        for family, role, index in [("missing", "training", 0), ("advdiff", "test", 0),
                                    ("advdiff", "training", -1), ("advdiff", "training", 3),
                                    ("advdiff", "training", True), ("advdiff", "training", 1.2),
                                    ("advdiff", "development", 2), ("advdiff", "final", 2)]:
            with self.subTest(family=family, role=role, index=index), self.assertRaises(ValueError):
                fixtures.generate(family, role, index)


if __name__ == "__main__":
    unittest.main()
